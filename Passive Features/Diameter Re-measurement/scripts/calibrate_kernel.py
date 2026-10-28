#!/usr/bin/env python3
"""Calibrate the defocus kernel from plane scans -- Blocks 10 and 15 in
specs/SPEC.md (procedure Eq. 4, s.3.4; mathematics Eq. 18, s.3.5; S2b).

  synthetic  Phase I: render N flat phantoms of one set (--set near: thin,
             d in (0.15, 0.25) um; --set far: d in (0.6, 0.8) um) with the
             configuration's kernel, measure each (the pilot) and scan its
             planes on the +-patch_half_um patch against the planar far-pixel
             background (10.P) -> scans JSON (+ the example node's patch
             images, patches .npz)
  real       Phase II (Colab: api.brain-map.org): the candidates of a
             previous run_cell.py or run_node.py table -- by Allen's diameter
             in the near-field or the far-field set of D-067.4, and in S,
             flat and faint by the pilot's columns -- are re-measured, tested
             for straightness and isolation (10.I) and scanned: the near set
             on k* +- 3, the far set on k* +- 16 (Block 15.1) -> one scans
             JSON holding both sets, and the example nodes' patches .npz
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

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
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
    node_faint_flag_check; D-094, D-095), and calibration.criteria_reasons on phi_rad, alpha_hat and Allen's diameter
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


def run_real(cfg, a):
    import allen_image_io as aio
    import run_cell
    from allen_diameter.loading import swc_io
    swc = swc_io.read_swc(a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir))
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    provider = run_cell.real_provider(aio.HttpFetcher(cache_dir=a.cache_dir), planes, cfg.acquisition.res0_um)
    cand = candidates_from_rows(table_io.read_rows([a.nodes_csv]), cfg)
    regs = {}
    if a.registration_json:
        with open(a.registration_json) as f:
            regs = {int(k): v for k, v in json.load(f).items()}
    print("%d candidate nodes" % len(cand), flush=True)
    t0 = time.perf_counter()
    transform = dict(shift_full_px=(a.shift_x, a.shift_y), flip_y_full_h=a.flip_h, z0_um=a.z0)
    cal = cfg.calibration          # Block 19.1 (D-088.3): a seeded order, at most calib_max_nodes_per_set per set
    out = calibration.scan_cell(swc, provider, cfg, transform, regs, cand, log=lambda m: print(m, flush=True),
                                order_rng=np.random.default_rng([int(cal.calib_order_seed), int(a.specimen)]),
                                max_per_set=int(cal.calib_max_nodes_per_set))
    scans = [s for _, _, s, _ in out if s is not None]
    allen_d = {int(i): 2.0 * float(r) for i, r in zip(swc.ids, swc.radius)}
    nodes = [dict(node_id=int(nid), allen_diameter_um=allen_d[int(nid)] if math.isfinite(allen_d[int(nid)]) else None,
                  set=calibration.calibration_set(allen_d[int(nid)], cfg), reasons=";".join(rs), scanned=s is not None)
             for nid, _, s, rs in out]             # the scans file is strict JSON (allow_nan=False)
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
    s.add_argument("--phi-max", type=float, default=2.0, help="deg: the pilot reads the tilt to about 1 deg, node_phi_max_deg is 5")
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
