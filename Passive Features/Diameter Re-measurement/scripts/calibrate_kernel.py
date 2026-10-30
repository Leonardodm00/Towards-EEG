#!/usr/bin/env python3
"""Calibrate the defocus kernel from plane scans -- Blocks 10 and 15 in
specs/SPEC.md (procedure Eq. 4, s.3.4; mathematics Eq. 18, s.3.5; S2b).

  synthetic  Phase I: render N flat phantoms of one set (--set near: thin,
             d in (0.15, 0.25) um; --set far: d in (0.6, 0.8) um) with the
             configuration's kernel, measure each (the pilot) and scan its
             planes on the +-patch_half_um patch against the planar far-pixel
             background (10.P) -> scans JSON (+ the example node's patch
             images, patches .npz)
  real       Phase II (Colab: api.brain-map.org): the calibration candidates
             -- since 2026-10-29 (D-109, D-112, Block 10.C) every dendrite
             node of the cell in the near-field or the far-field set of
             D-067.4, judged from the trace before any fetch (flat, the
             stripe at the trace's plane and at the set's fit planes), the
             previous run_cell.py or run_node.py table giving the typical
             background and absorption; with calibration.near_candidates /
             far_candidates = "table", that table's rows in S and flat -- are
             measured (the pilot), judged again (10.I) and scanned: the near
             set on k* +- 3, the far set on k* +- 16 (Block 15.1) -> one
             scans JSON holding both sets, <set>_search_<specimen>.csv beside
             it, and the example nodes' patches .npz;
             as each candidate is evaluated, the figure of the patch its
             background is fitted on, with the traced neighbours whose
             predicted shadows decide not_isolated (10.F (5), D-098.4) ->
             <figures>/isolation/<set>_node_<id>.png and overview pages
  fit        the near field (10.A): Eq. (4) over the near scans; the kernel
             table sigma_r(delta_m) = sqrt(sigma_r(0)^2 + G(delta_m)) CUT at
             table_max_delta_um = 0.84 um (D-080.1); the anchor sigma_0_hat
             and its node-bootstrap SE -> calibration JSON and a report
  far-fit    the far field (Block 15): the planes selected by the predicted-
             dip SNR rule, the sequential far fit with the anchor (IRLS, both
             sides, axis free), its node bootstrap, the anchor term, the
             per-sign and quadratic comparisons; the production kernel (the
             cut table + the fitted slope) -> far-fit JSON and a report
  figures    the figures of 10.F (D-075, D-080.2, D-081.5) from the scans,
             the calibration, the far fit and the patches -> PNG files

The near field is fitted before the far field (D-081.2): `fit` first, then
`far-fit` with its calibration file.

Examples
    python scripts/calibrate_kernel.py synthetic --set near --n 12 --seed 5 --out cal/scans_near.json --workers 2
    python scripts/calibrate_kernel.py synthetic --set far --n 6 --seed 7 --out cal/scans_far.json --workers 2
    python scripts/calibrate_kernel.py fit --scans cal/scans_near.json --out cal/calibration.json
    python scripts/calibrate_kernel.py far-fit --scans cal/scans_far.json --calibration cal/calibration.json \\
        --out cal/far_fit.json
    python scripts/calibrate_kernel.py figures --scans cal/scans_near.json cal/scans_far.json \\
        --calibration cal/calibration.json --far-fit cal/far_fit.json --patches cal/patches_near.npz \\
        cal/patches_far.npz --out-dir cal/figures
    # Colab, after run_cell.py wrote nodes_529878215.csv:
    python scripts/calibrate_kernel.py real --specimen 529878215 --nodes-csv out/nodes_529878215.csv \\
        --shift-x 12.3 --shift-y -4.1 --cache-dir /content/drive/MyDrive/allen_cache --out cal/scans_529878215.json
    # the isolation figures go to cal/figures/isolation/ (--isolation-figures DIR elsewhere, --no-isolation-figures
    # for none)

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from allen_diameter.analysis import calibration, cell, far_field, phantoms  # noqa: E402
from allen_diameter.analysis import calibration_pool  # noqa: E402
from allen_diameter.loading import calibration_io, table_io  # noqa: E402
from allen_diameter.model import render  # noqa: E402
from build_table import load_config  # noqa: E402

SET_D_RANGES = {"near": (0.15, 0.25), "far": (0.6, 0.8)}      # um: a phantom's Allen diameter is its d (D-067.4)


def _pair(text):
    a, b = (float(x) for x in text.split(","))
    return a, b


def calibration_phantom(cfg, seed, index, d_range, alpha_range, phi_max_deg):
    """(Draw, generator) of a flat phantom: log d uniform on d_range, alpha = mu d / cos(phi) uniform on
    alpha_range, phi uniform on [0, phi_max), theta on [0, pi), the sub-pixel offset and the axis depth as in the
    table's design; replicate n draws from default_rng([seed, n])."""
    rng = np.random.default_rng([int(seed), int(index)])
    d = math.exp(rng.uniform(math.log(d_range[0]), math.log(d_range[1])))
    phi = math.radians(rng.uniform(0.0, phi_max_deg))
    theta = rng.uniform(0.0, math.pi)
    alpha = rng.uniform(*alpha_range)
    mu = alpha * math.cos(phi) / d
    p, dz = cfg.acquisition.res0_um, cfg.acquisition.dz_um
    cx, cy = rng.uniform(-0.5 * p, 0.5 * p, 2)
    cz = rng.uniform(-0.5 * dz, 0.5 * dz)
    # the branch carries the true radius d/2 (no D-062.1 noise): Allen's diameter of a calibration phantom is d
    return phantoms.Draw(int(index) + 1, 1, int(seed), d, phi, theta, mu, float(cx), float(cy), float(cz), 0.5 * d,
                         0), rng


def _one_synthetic(args):
    cfg, seed, index, d_range, alpha_range, phi_max, keep_images = args
    draw, rng = calibration_phantom(cfg, seed, index, d_range, alpha_range, phi_max)
    tube = phantoms.phantom_tube(draw, cfg)
    # a far phantom's branch must span the far line (+-far_profile_half_um) and the straightness window
    ph = cfg.phantom
    need = max(cfg.calibration.far_profile_half_um, cfg.calibration.node_straight_half_um) + 2.0
    n_each = max(int(ph.nodes_each_way), int(math.ceil(need / ph.phantom_node_step_um)))
    branch = phantoms.phantom_branch(draw, dataclasses.replace(cfg, phantom=dataclasses.replace(ph, nodes_each_way=n_each)), rng)
    pad = int(math.ceil(cfg.renderer.pad_um / cfg.acquisition.res0_um))

    def provider(left, top, width, height, k_lo, k_hi):
        return render.synthetic_block(tube, draw.mu_per_um, np.arange(k_lo, k_hi + 1), left, top, width, height,
                                      cfg, rng, pad)

    res, scan, reasons = calibration.scan_node(branch, n_each, provider, cfg, keep_images=keep_images)
    if scan is not None:                 # every phantom's measured node is the middle one: name it by index
        scan = dataclasses.replace(scan, node_id=int(index))
    return draw, res, scan, reasons


def run_synthetic(cfg, which, n, seed, out, d_range, alpha_range, phi_max, workers=1):
    jobs = [(cfg, seed, i, d_range, alpha_range, phi_max, False) for i in range(int(n))]
    if workers > 1:
        import multiprocessing
        with multiprocessing.Pool(workers) as pool:
            results = list(pool.imap(_one_synthetic, jobs, chunksize=1))
    else:
        results = [_one_synthetic(j) for j in jobs]
    scans = [s for _, _, s, _ in results if s is not None]
    meta = dict(source="synthetic", set=which, seed=int(seed), n=int(n), d_range_um=list(d_range),
                alpha_range=list(alpha_range), phi_max_deg=phi_max, config=json.loads(cfg.to_json()),
                calibration=dataclasses.asdict(cfg.calibration),
                truth=[dict(index=dr.draw_index - 1, d_um=dr.d_um, mu_per_um=dr.mu_per_um, phi_rad=dr.phi_rad, cz_um=dr.cz_um,
                            scanned=s is not None, reasons=";".join(rs)) for dr, _, s, rs in results])
    calibration_io.save_scans(scans, out, meta)
    for dr, res, s, rs in results:
        print("phantom %d: d %.3f um, alpha %.2f -> d_hat %.3f, %s" % (
            dr.draw_index - 1, dr.d_um, dr.mu_per_um * dr.d_um / math.cos(dr.phi_rad), res.d_hat_um,
            calibration.scan_verdict(s, rs)))
    print("%d of %d phantoms scanned -> %s" % (len(scans), len(results), out))
    ex = calibration.example_node(scans, which)
    if ex is not None:
        job = jobs[int(ex)][:6] + (True,)
        _, _, sc_ex, _ = _one_synthetic(job)       # the same draw and seed: the same patch, now kept
        patches_path = patches_name(out)
        calibration_io.save_patches(patches_path, {which: dataclasses.replace(sc_ex, node_id=int(ex))})
        print("example node %d (nearest the median width by plane): patches -> %s" % (ex, patches_path))
    return out


def patches_name(scans_path):
    root, _ext = os.path.splitext(scans_path)
    return root.replace("scans", "patches") + ".npz" if "scans" in os.path.basename(root) else root + "_patches.npz"


def candidates_from_rows(rows, cfg):
    """Node ids of a run_cell.py or run_node.py table that can be calibration nodes of either set: the rule of
    calibration.calibration_reasons on the CSV columns read as the pilot's (cell.pilot_row_view, Block 13) --
    in S by fit_status and the flags of calibration.selection_flags (SELECTION_FLAGS, with faint added when
    node_faint_flag_check; D-094, D-096), and calibration.criteria_reasons on phi_rad, alpha_hat and Allen's diameter
    2 x allen_radius_um (D-061.4, D-067.4). Straightness, isolation and the registration are checked when the
    node is re-measured before the scan, so this only saves fetches."""
    out = set()
    fail = calibration.selection_flags(cfg)
    for r in (cell.pilot_row_view(x) for x in rows):
        flags = [f for f in str(r.get("flags", "")).split(";") if f]
        try:
            ok = (r["fit_status"] == "converged" and not any(f in fail for f in flags)
                  and not calibration.criteria_reasons(float(r["phi_rad"]), float(r["alpha_hat"]),
                                                       2.0 * float(r["allen_radius_um"]), cfg))
        except (KeyError, ValueError, TypeError):
            ok = False
        if ok:
            out.add(int(r["node_id"]))
    return out


def isolation_dir(a):
    """Where `real` writes the isolation figures: --isolation-figures, else figures/isolation beside --out; "" with
    --no-isolation-figures."""
    if getattr(a, "no_isolation_figures", False):
        return ""
    return a.isolation_figures or os.path.join(os.path.dirname(os.path.abspath(a.out)), "figures", "isolation")


def isolation_hook(provider, cfg, fig_dir, views, log=print, dpi=110):
    """scan_cell's on_node for Cell 7 (10.F (5), D-098.4): for each measured candidate, its isolation_view (one
    patch plane through the provider, from the cache for a scanned node) drawn to <fig_dir>/<set>_node_<id>.png and
    kept, without its far mask, in views for the overview. A figure that fails is reported and skipped: it must not
    stop the calibration."""
    from allen_diameter.analysis import calibration_report
    from allen_diameter.plotting import calibration_figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)

    def hook(nid, branch, i, res, scan, reasons, scene):
        if res is None:
            return
        try:
            view = calibration_report.isolation_view(branch, i, res, scene, provider, cfg,
                                                     verdict=calibration.scan_verdict(scan, reasons))
            if view is None:
                log("node %d: no isolation figure (the pilot has no centre)" % nid)
                return
            fig = calibration_figures.isolation_figure(view)
            fig.savefig(os.path.join(fig_dir, "%s_node_%d.png" % (view["set"], int(nid))), dpi=dpi,
                        facecolor=calibration_figures.SURFACE)
            plt.close(fig)
            views.append({k: v for k, v in view.items() if k != "far"})
        except Exception as exc:          # noqa: BLE001 -- a figure must not stop the calibration (logged)
            log("node %d: isolation figure failed: %s: %s" % (nid, type(exc).__name__, exc))
    return hook


def candidate_sources(cfg):
    """{set: "cell" | "table"}: where each calibration set's candidates come from (Block 10.C, D-109, D-112)."""
    cal = cfg.calibration
    return {"near": cal.near_candidates, "far": cal.far_candidates}


def knobs_line(cfg):
    """The stripe's settings a run uses (D-101; Cell 7's knobs reach the scripts through --config-json) and where
    each set's candidates come from (D-109, D-112)."""
    cal = cfg.calibration
    src = candidate_sources(cfg)
    far = "".join("; %s set from the whole cell, at least %g um apart" % (w, getattr(cal, w + "_search_spacing_um"))
                  if src[w] == "cell" else "; %s set from the pilot table" % w for w in calibration.CALIBRATION_SETS)
    if cal.background_support != "stripe":
        return ("calibration knobs: background_support square (the far mask r + %g sigma, the any-shadow veto above "
                "%g gl); flatness phi <= %g deg near, %g deg far" % (cal.far_pixel_sigmas, cal.isolation_depth_gl,
                                                                     cal.node_phi_max_deg, cal.node_phi_max_deg_far)
                + far)
    return ("calibration knobs: background_support stripe, tau %g gl, W %s um, at least %d px and %g um per side, patch "
            "+-%g um; flatness phi <= %g deg near, %g deg far" % (
                cal.isolation_depth_gl, ", ".join("%g" % w for w in cal.stripe_half_widths_um), cal.stripe_min_pixels,
                cal.stripe_min_side_um, cal.patch_half_um, cal.node_phi_max_deg, cal.node_phi_max_deg_far) + far)


def far_planes_line(views, cfg):
    """How many of the far set's candidates have a usable stripe in at least far_min_planes_per_node of the far fit's
    planes (|k - k*| >= far_min_k_offset) -- the far fit drops the others (few_planes); from the views' stripe_planes
    (panel (c) of the isolation figures, 2026-10-28)."""
    from allen_diameter.plotting import calibration_figures
    cal = cfg.calibration
    far = [calibration_figures.fit_planes_usable(v) for v in views if v["set"] == "far"]
    far = [u for u in far if u is not None]
    enough = sum(1 for used, _total in far if used >= int(cal.far_min_planes_per_node))
    return ("far-set candidates with a usable stripe in at least %d of the far fit's planes (|k - k*| >= %d): %d of %d"
            % (cal.far_min_planes_per_node, cal.far_min_k_offset, enough, len(far)))


def search_name(out_dir, which, specimen):
    """Where Block 10.C's records of set `which` go: <out_dir>/<which>_search_<specimen>.csv (Cell 7: beside the
    scans file; S3's calibration step: the cell's calibration folder)."""
    return os.path.join(out_dir, "%s_search_%s.csv" % (which, specimen))


def search_record_name(out_dir, which, specimen):
    """The record beside a search's CSV (10.C.R, D-115.6): <out_dir>/<which>_search_<specimen>.json."""
    return os.path.join(out_dir, "%s_search_%s.json" % (which, specimen))


def search_key(swc, cfg):
    """What a search's result depends on and a reuse must find equal (10.C.R, D-115.6): the trace
    (candidate_search.swc_digest) and the settings (candidate_search.search_config). The cell's typical background
    and absorption and its transform are not in it (they go with the search: the record keeps them)."""
    from allen_diameter.analysis import candidate_search
    return dict(swc_sha256=candidate_search.swc_digest(swc), config=candidate_search.search_config(cfg))


def _file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_search_record(out_dir, which, specimen, key, B_bar, mu, transform, n_records, n_chosen, reused_from=None):
    """The record of the search CSV search_name(out_dir, which, specimen), which must exist (10.C.R, D-115.6),
    written as strict JSON to search_record_name(out_dir, which, specimen). Returns its path."""
    from allen_diameter.loading import survey_io
    path = search_record_name(out_dir, which, specimen)
    survey_io.write_json(path, dict(set=which, specimen=str(specimen), key=key,
                                    csv_sha256=_file_sha256(search_name(out_dir, which, specimen)),
                                    B_bar_gl=float(B_bar), mu_hat_per_um=float(mu), transform=transform,
                                    n_records=int(n_records), n_chosen=int(n_chosen), reused_from=reused_from))
    return path


def reuse_search(reuse_dir, which, specimen, key):
    """A search of set `which` made earlier for this specimen in reuse_dir (Cell 7's folder; 10.C.R, D-115.6):
    (chosen node ids, its record, "") when its CSV and record exist, the record's key equals `key` and the CSV is
    the one the record describes; else (None, the record or None, why)."""
    csv, rec_path = search_name(reuse_dir, which, specimen), search_record_name(reuse_dir, which, specimen)
    if not os.path.exists(csv):
        return None, None, "no search file %s" % csv
    if not os.path.exists(rec_path):
        return None, None, "no record %s" % rec_path
    with open(rec_path) as f:
        rec = json.load(f)
    if rec.get("csv_sha256") != _file_sha256(csv):
        return None, rec, "the CSV changed since its record"
    theirs = rec.get("key") or {}
    if theirs.get("swc_sha256") != key["swc_sha256"]:
        return None, rec, "another trace"
    conf = theirs.get("config") or {}
    differ = sorted(name for name in set(conf) | set(key["config"]) if conf.get(name) != key["config"].get(name))
    if differ:
        return None, rec, "other settings: %s" % ", ".join(differ)
    chosen = [int(r["node_id"]) for r in table_io.read_rows([csv]) if str(r["chosen"]) == "True"]
    return chosen, rec, ""


def _bm_text(B_bar, mu):
    f = (lambda x, fmt: fmt % x if x is not None and math.isfinite(float(x)) else "nan")
    return "B_bar %s gl, mu_hat %s /um" % (f(B_bar, "%.1f"), f(mu, "%.3f"))


def not_in_set(cand, swc, cfg, which):
    """The ids of cand whose Allen diameter (2 x the SWC radius) does not put them in calibration set `which`."""
    radius = {int(i): float(r) for i, r in zip(swc.ids, swc.radius)}
    return {int(n) for n in cand if calibration.calibration_set(2.0 * radius.get(int(n), float("nan")), cfg) != which}


def cell_candidates(swc, cfg, transform, table_rows, out_dir, specimen, log=None, reuse_dir=None):
    """Block 10.C (D-109, D-112): a cell's calibration candidates. For each set, the pilot table's
    (candidates_from_rows) when calibration.<set>_candidates is "table"; else the nodes candidate_search finds over
    the whole cell from its trace (transform: keyword arguments of cell.to_image_um), the table's medians in S as
    the cell's typical background and absorption -- they replace the table's candidates of that set (not_in_set).
    Every searched set's records go to search_name(out_dir, set, specimen) with a column `chosen`, and its record
    (10.C.R, D-115.6) to search_record_name(out_dir, set, specimen); log, when given, gets the search's lines.
    reuse_dir (S3's step on the pilot, Cell 15: Cell 7's folder): a set's search made there for this specimen from
    the same trace under the same settings (reuse_search) is taken instead of searching -- its chosen ids, its
    CSV and record copied to out_dir, its B_bar, mu_hat and transform printed beside the step's own. Used by
    `real` (Cell 7, which never reuses) and by S3's calibration step (survey_cells.py, Block 19.3). Returns the
    set of node ids."""
    from allen_diameter.analysis import candidate_search
    from allen_diameter.loading import survey_io
    say = log or (lambda m: None)
    cand = candidates_from_rows(table_rows, cfg)
    sets = [w for w in calibration.CALIBRATION_SETS if candidate_sources(cfg)[w] == "cell"]
    if not sets:
        return cand
    B_bar, mu = candidate_search.typical_background(table_rows, cfg)
    say("candidate search: the cell's typical background %.1f gl and absorption %.3f /um (the pilot table's medians "
        "in S)" % (B_bar, mu))
    key = search_key(swc, cfg)
    tr_plain = survey_io.clean(transform)                # tuples as lists, NaN as None: as the record holds it
    for which in sets:
        t0 = time.perf_counter()
        path = search_name(out_dir, which, specimen)
        kept = None
        if reuse_dir:
            ids, rec, why = reuse_search(reuse_dir, which, specimen, key)
            if ids is None:
                say("%s search not reused (%s): searching" % (which, why))
            else:
                kept = set(ids)
                src_csv = search_name(reuse_dir, which, specimen)
                src_rec = os.path.abspath(search_record_name(reuse_dir, which, specimen))
                if os.path.abspath(src_csv) != os.path.abspath(path):
                    os.makedirs(out_dir, exist_ok=True)
                    shutil.copyfile(src_csv, path)
                    survey_io.write_json(search_record_name(out_dir, which, specimen), dict(rec, reused_from=src_rec))
                say("%s search reused from %s: %d chosen of %s records, made with %s, transform %s; this step's own: "
                    "%s, transform %s" % (which, src_csv, len(kept), rec.get("n_records"),
                                          _bm_text(rec.get("B_bar_gl"), rec.get("mu_hat_per_um")),
                                          json.dumps(rec.get("transform")), _bm_text(B_bar, mu), json.dumps(tr_plain)))
        if kept is None:
            records = candidate_search.search(swc, cfg, which, transform, B_bar, mu, log=say)
            chosen = candidate_search.choose(records, cfg, which)
            kept = set(chosen)
            table_io.write_rows([dict(r, chosen=int(r["node_id"]) in kept) for r in records], path,
                                fieldnames=list(candidate_search.RECORD_KEYS) + ["chosen"])
            write_search_record(out_dir, which, specimen, key, B_bar, mu, tr_plain, len(records), len(chosen))
            say("%s (%.1f min) -> %s" % (candidate_search.summary_line(records, chosen, cfg, which),
                                         (time.perf_counter() - t0) / 60, path))
        cand = not_in_set(cand, swc, cfg, which) | kept
    return cand


def write_isolation_overview(views, fig_dir, dpi=110):
    """The overview pages of the isolation figures: <fig_dir>/isolation_overview_<page>.png. Returns the paths."""
    from allen_diameter.plotting import calibration_figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for page, fig in enumerate(calibration_figures.isolation_overview(views), start=1):
        path = os.path.join(fig_dir, "isolation_overview_%d.png" % page)
        fig.savefig(path, dpi=dpi, facecolor=calibration_figures.SURFACE)
        plt.close(fig)
        paths.append(path)
    return paths


def run_real(cfg, a):
    import allen_image_io as aio
    import run_cell
    from allen_diameter.loading import swc_io
    swc = swc_io.read_swc(a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir))
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    provider = run_cell.real_provider(aio.HttpFetcher(cache_dir=a.cache_dir), planes, cfg.acquisition.res0_um)
    table_rows = table_io.read_rows([a.nodes_csv])
    regs = {}
    if a.registration_json:
        with open(a.registration_json) as f:
            regs = {int(k): v for k, v in json.load(f).items()}
    transform = dict(shift_full_px=(a.shift_x, a.shift_y), flip_y_full_h=a.flip_h, z0_um=a.z0)
    cal = cfg.calibration          # Block 19.1 (D-088.3): a seeded order, at most calib_max_nodes_per_set per set
    print(knobs_line(cfg), flush=True)
    cand = cell_candidates(swc, cfg, transform, table_rows, os.path.dirname(os.path.abspath(a.out)), a.specimen,
                           log=lambda m: print(m, flush=True))        # Block 10.C (D-109, D-112)
    print("%d candidate nodes" % len(cand), flush=True)
    t0 = time.perf_counter()
    fig_dir, views = isolation_dir(a), []
    hook = isolation_hook(provider, cfg, fig_dir, views, log=lambda m: print(m, flush=True)) if fig_dir else None
    out = calibration.scan_cell(swc, provider, cfg, transform, regs, cand, log=lambda m: print(m, flush=True),
                                order_rng=np.random.default_rng([int(cal.calib_order_seed), int(a.specimen)]),
                                max_per_set=int(cal.calib_max_nodes_per_set), on_node=hook)
    scans = [s for _, _, s, _ in out if s is not None]
    nodes = calibration.node_records(out, swc, cfg)   # with each measured node's pilot values (D-115.5); strict JSON
    n_near = sum(1 for s in scans if s.set == "near")
    n_far = sum(1 for s in scans if s.set == "far")
    reasons_count = {}
    for _, _, _, rs in out:
        for r in rs:
            reasons_count[r] = reasons_count.get(r, 0) + 1
    calibration_io.save_scans(scans, a.out, dict(source="real", specimen=str(a.specimen), n_candidates=len(cand),
                                                 config=json.loads(cfg.to_json()),
                                                 calibration=dataclasses.asdict(cfg.calibration), nodes=nodes,
                                                 reasons_count=reasons_count))
    print("%d of %d candidates scanned: %d near-field, %d far-field (at most %d each, D-088.3); reasons %s; in %.1f "
          "min -> %s" % (len(scans), len(out), n_near, n_far, cal.calib_max_nodes_per_set, reasons_count,
                         (time.perf_counter() - t0) / 60, a.out))
    print(calibration.gof_line(scans, cfg), flush=True)              # Block 10.G (D-115.3)
    # the example nodes, re-scanned with their patch images (served from the cache)
    records = {}
    for which in calibration.CALIBRATION_SETS:
        ex = calibration.example_node(scans, which)
        if ex is None:
            continue
        again = calibration.scan_cell(swc, provider, cfg, transform, regs, {int(ex)}, keep_images=True)
        if again and again[0][2] is not None:
            records[which] = again[0][2]
            print("example %s-field node %d: patch images kept" % (which, ex))
    if records:
        calibration_io.save_patches(patches_name(a.out), records)
        print("patches -> %s" % patches_name(a.out))
    if fig_dir:
        pages = write_isolation_overview(views, fig_dir) if views else []
        if cal.background_support == "stripe":
            n_bad = sum(1 for v in views if v.get("stripe") is not None and not v["stripe"]["ok"])
            print("isolation figures: %d candidates (%d whose stripe at k* is not usable: fewer than %d px or a side "
                  "under %.1f um) -> %s; overview %s" % (len(views), n_bad, cal.stripe_min_pixels, cal.stripe_min_side_um,
                                                         fig_dir, ", ".join(os.path.basename(x) for x in pages) or "none"))
            print(far_planes_line(views, cfg))
        else:
            n_bad = sum(1 for v in views if v["evaluated"] and v["max_depth_gl"] > v["threshold_gl"])
            print("isolation figures: %d candidates (%d with a neighbour's predicted shadow above %.3g gl) -> %s; "
                  "overview %s" % (len(views), n_bad, cal.isolation_depth_gl, fig_dir,
                                   ", ".join(os.path.basename(x) for x in pages) or "none"))
    return a.out


def run_fit(cfg, scans_path, out, sigma_r0=None, seed=0):
    scans, meta = calibration_io.load_scans(scans_path)
    near = [s for s in scans if s.set == "near"]
    if len(near) < len(scans):
        print("fit: %d far-field scans in %s are left to far-fit" % (len(scans) - len(near), scans_path))
    try:
        g = calibration.fit_growth(near, cfg)
    except ValueError as exc:
        sys.exit("fit: %s (%d near-field scans in %s)" % (exc, len(near), scans_path))
    kernel, err = None, ""
    try:
        kernel = calibration.kernel_from_growth(g, cfg.renderer, sigma_r0, max_delta_um=cfg.calibration.table_max_delta_um)
    except ValueError as exc:
        err = str(exc)
    anchor = None
    if kernel is not None:
        rng = np.random.default_rng(int(seed))
        se, n_ok, _draws = calibration.bootstrap_sigma0(near, cfg, rng, sigma_r0)
        anchor = dict(sigma0_um=float(kernel.kernel_table_sigma_um[-1]), se_um=float(se), n_boot_ok=int(n_ok),
                      n_resamples=int(cfg.calibration.bootstrap_resamples), seed=int(seed),
                      delta0_um=float(cfg.calibration.table_max_delta_um))
    ratio = calibration.area_ratio_by_offset(near)
    suff = calibration_pool.sufficiency(g.node_ids.size, cfg)              # Block 19.2 (D-088.3)
    calibration_io.save_calibration(out, g, kernel, err, ratio,
                                    dict(scans=scans_path, scans_meta_source=meta.get("source", ""),
                                         config=json.loads(cfg.to_json()), sufficiency=suff), anchor)
    trust = (cfg.calibration.trust_planes + 0.5) * cfg.acquisition.dz_um
    print("growth fit (%s, %s, %s): %d nodes, %d dropped, cost %.3g, %s"
          % (g.statistic, g.convention, g.interp, g.node_ids.size, len(g.dropped_nodes), g.cost, g.message))
    print("  delta_um   G_um2      n_obs(interval to the next knot)  trusted")
    for m, (x, val) in enumerate(zip(g.knots_um, g.g_um2)):
        n = "%5d" % int(g.n_obs_per_interval[m]) if m < g.n_obs_per_interval.size else "    -"
        print("  %+7.3f  %9.5f  %s  %s" % (x, val, n, "yes" if abs(x) <= trust + 1e-9 else "after the residual check"))
    print("  residual RMS by |k - k*|: %s" % ", ".join("%d: %.4f" % kv for kv in g.rms_by_offset().items()))
    print("  A_W(k) / A_W(k*): %s" % ", ".join("%+d: %.3f" % kv for kv in ratio.items()))
    if kernel is not None:
        mono = bool(np.all(np.diff(kernel.kernel_table_sigma_um) > 0))
        print("  table (cut at %.2f um, D-080.1): delta %s" % (cfg.calibration.table_max_delta_um,
                                                              ", ".join("%.2f" % x for x in kernel.kernel_table_delta_um)))
        print("                     sigma %s  (sigma_r(0) = %.3f configured; %s)"
              % (", ".join("%.3f" % x for x in kernel.kernel_table_sigma_um), kernel.sigma_r0_um,
                 "increasing" if mono else "NOT increasing: do not render with it before step 2"))
        print("  anchor sigma_0_hat = %.4f um, SE %.4f um (node bootstrap, %d of %d resamples, seed %d)"
              % (anchor["sigma0_um"], anchor["se_um"], anchor["n_boot_ok"], anchor["n_resamples"], seed))
    else:
        print("  no table: %s" % err)
    if not suff["sufficient"]:
        print("  INSUFFICIENT: %d near-field nodes < %d (D-088.3): a diagnostic, not the production kernel's table"
              % (suff["n_nodes"], suff["min_nodes"]))
    print("-> %s" % out)
    return out


def run_far_fit(cfg, scans_path, calibration_path, out, seed=0):
    scans, meta = calibration_io.load_scans(scans_path)
    far = [s for s in scans if s.set == "far"]
    calib = calibration_io.load_calibration(calibration_path)
    if "anchor" not in calib or calib["anchor"].get("sigma0_um") is None:
        sys.exit("far-fit: %s holds no anchor (run `fit` on the near-field scans first)" % calibration_path)
    sigma0, se_sigma0 = float(calib["anchor"]["sigma0_um"]), float(calib["anchor"].get("se_um") or float("nan"))
    sel = far_field.select_far_planes(far, cfg)
    print("far planes: %d nodes kept (%s), %d dropped %s; %d planes; %d not converged, %d below the SNR rule"
          % (sel.node_ids.size, sel.n_planes_per_node.tolist(), len(sel.dropped), sel.dropped, sel.omega_um2.size,
             sel.n_not_converged, sel.n_snr_dropped))
    k_star = {int(s.node_id): int(s.k_star) for s in far}
    try:
        fit = far_field.fit_far_slope(sel, sigma0, cfg, k_star=k_star)
    except ValueError as exc:
        sys.exit("far-fit: %s" % exc)
    fit_sign = far_field.fit_far_slope(sel, sigma0, cfg, per_sign=True, k_star=k_star)
    fit_quad = far_field.fit_far_slope(sel, sigma0, cfg, quadratic=True, k_star=k_star)
    t0 = time.perf_counter()
    se_boot, n_ok, _ = far_field.bootstrap_gamma(sel, sigma0, cfg, np.random.default_rng(int(seed)))
    dg = far_field.anchor_sensitivity(sel, sigma0, cfg)
    summary = far_field.far_fit_summary(sel, fit, fit_sign, fit_quad, se_boot, n_ok, dg, sigma0, se_sigma0, cfg, seed)
    suff = dict(near=calibration_pool.sufficiency(len(calib.get("growth", {}).get("node_ids", [])), cfg),
                far=calibration_pool.sufficiency(sel.node_ids.size, cfg))         # Block 19.2 (D-088.3)
    kernel, err = None, ""
    if "kernel" in calib:
        kd = calib["kernel"]
        kernel = dataclasses.replace(cfg.renderer, kernel_family="gaussian_table", sigma_r0_um=float(kd["sigma_r0_um"]),
                                     kernel_table_delta_um=tuple(float(x) for x in kd["kernel_table_delta_um"]),
                                     kernel_table_sigma_um=tuple(float(x) for x in kd["kernel_table_sigma_um"]),
                                     kernel_continuation="linear", kernel_continuation_slope=float(fit.gamma))
        if not fit.success:
            kernel, err = None, "the far fit did not succeed (%s%s)" % (fit.message, "; gamma at a bound" if fit.at_bound else "")
    else:
        err = "the calibration file holds no table: %s" % calib.get("kernel_error", "")
    short = [s for s in ("near", "far") if not suff[s]["sufficient"]]
    if kernel is not None and short:
        kernel, err = None, "insufficient: %s (D-088.3: at least %d nodes per set; the fits are diagnostics)" % (
            ", ".join("%d %s-field nodes" % (suff[s]["n_nodes"], s) for s in short), suff["far"]["min_nodes"])
    calibration_io.save_far_fit(out, summary, kernel, err, dict(scans=scans_path, calibration=calibration_path,
                                                                config=json.loads(cfg.to_json()), sufficiency=suff))
    print("far fit: gamma_hat = %.4f +- %.4f (bootstrap %.4f over %d resamples, anchor term %.4f x %.4f), "
          "t_%d interval [%s, %s]; range %.2f um; %d IRLS rounds, %s" % (
              fit.gamma, summary["se"] or float("nan"), se_boot, n_ok, dg, se_sigma0, sel.node_ids.size - 1,
              summary["interval"][0], summary["interval"][1], fit.range_um, fit.rounds,
              "success" if fit.success else "FAILED: " + fit.message))
    print("  per node: c %s um^2, s %s um" % (np.round(fit.c_um2, 4).tolist(), np.round(fit.s_um, 3).tolist()))
    print("  residual RMS by |k - k*|: %s" % ", ".join("%d: %.4f" % kv for kv in fit.rms_by_offset().items()))
    print("  per-sign: gamma_+ %.4f, gamma_- %.4f; quadratic comparison: gamma %.4f, gamma_2 %.4f per um (labelled "
          "comparison, D-081.6)" % (fit_sign.gamma, fit_sign.gamma_minus, fit_quad.gamma, fit_quad.gamma2))
    if kernel is not None:
        print("  production kernel: table to %.2f um, continuation linear with slope %.4f (set "
              "renderer.kernel_continuation_slope in Cell 8)" % (kernel.kernel_table_delta_um[-1], kernel.kernel_continuation_slope))
    else:
        print("  no production kernel: %s" % err)
    print("  (%.0f s) -> %s" % (time.perf_counter() - t0, out))
    return out


def run_figures(cfg, a):
    from allen_diameter.plotting import calibration_figures as CF
    scans, id_map = [], {}
    for path in a.scans:
        sc, meta = calibration_io.load_scans(path)
        scans += sc
        id_map.update((meta or {}).get("id_map") or {})        # a pooled file's map (Block 19.5)
    calib = calibration_io.load_calibration(a.calibration) if a.calibration else None
    far_fit = calibration_io.load_calibration(a.far_fit) if a.far_fit else None
    patches = {}
    for path in a.patches or []:
        patches.update(calibration_io.load_patches(path))
    os.makedirs(a.out_dir, exist_ok=True)
    written = CF.write_all(scans, calib, far_fit, patches, cfg, a.out_dir, every_node=not a.no_node_figures,
                           id_map=id_map or None)
    for w in written:
        print("wrote %s" % w)
    return a.out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--set", dest="which", choices=list(calibration.CALIBRATION_SETS), default="near",
                   help="which calibration set the phantoms belong to (their Allen diameter is their d)")
    s.add_argument("--n", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--d-range", type=_pair, default=None,
                   help="true d, um, comma-separated; default per set: near (0.15, 0.25), far (0.6, 0.8)")
    s.add_argument("--alpha-range", type=_pair, default=(0.15, 0.4))
    s.add_argument("--phi-max", type=float, default=2.0, help="deg: the phantoms' largest tilt; the pilot reads the tilt to about 1 deg (4.5 deg on 0.2 um "
                        "tubes, the tester's F3 of 2026-10-28); the selection's bounds are node_phi_max_deg (near, 10) "
                        "and node_phi_max_deg_far (far, 15), D-101.1")
    s.add_argument("--workers", type=int, default=1)
    r = sub.add_parser("real")
    r.add_argument("--specimen", required=True)
    r.add_argument("--nodes-csv", required=True, help="nodes_<specimen>.csv of run_cell.py")
    r.add_argument("--swc", default="")
    r.add_argument("--cache-dir", default="allen_cache")
    r.add_argument("--shift-x", type=float, default=0.0)
    r.add_argument("--shift-y", type=float, default=0.0)
    r.add_argument("--flip-h", type=float, default=None)
    r.add_argument("--z0", type=float, default=0.0)
    r.add_argument("--registration-json", default="")
    r.add_argument("--isolation-figures", default="",
                   help="folder of the per-candidate isolation figures (10.F (5)); default: figures/isolation beside --out")
    r.add_argument("--no-isolation-figures", action="store_true", help="draw no isolation figure")
    f = sub.add_parser("fit")
    f.add_argument("--scans", required=True)
    f.add_argument("--sigma-r0", type=float, default=None, help="um; default: the configuration's sigma_r0_um")
    f.add_argument("--seed", type=int, default=0, help="the anchor bootstrap's seed")
    ff = sub.add_parser("far-fit")
    ff.add_argument("--scans", required=True, help="the scans file holding the far-field scans")
    ff.add_argument("--calibration", required=True, help="the calibration JSON of `fit` (the anchor and the table)")
    ff.add_argument("--seed", type=int, default=0, help="the node bootstrap's seed")
    g = sub.add_parser("figures")
    g.add_argument("--scans", nargs="+", required=True)
    g.add_argument("--calibration", default="")
    g.add_argument("--far-fit", default="")
    g.add_argument("--patches", nargs="*", default=None)
    g.add_argument("--out-dir", required=True)
    g.add_argument("--no-node-figures", action="store_true", help="skip the per-node fitted-Gaussian figures")
    for q in (s, r, f, ff):
        q.add_argument("--out", required=True)
    for q in (s, r, f, ff, g):
        q.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)
    if a.cmd == "synthetic":
        run_synthetic(cfg, a.which, a.n, a.seed, a.out, a.d_range or SET_D_RANGES[a.which], a.alpha_range, a.phi_max,
                      a.workers)
    elif a.cmd == "real":
        run_real(cfg, a)
    elif a.cmd == "fit":
        run_fit(cfg, a.scans, a.out, a.sigma_r0, a.seed)
    elif a.cmd == "far-fit":
        run_far_fit(cfg, a.scans, a.calibration, a.out, a.seed)
    else:
        run_figures(cfg, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
