"""Cell 4l (D-046, the user's request of 2026-10-13, 14:50): fit variants at
the plane each node's Allen-guided iteration chose (Cell 4j's k_iter), with
other blur-kernel families and with centre weights. A diagnostic: no
measurement, Block 5 fit or estimator hash changes.

For each node of a Cell 4j run with a plane k_iter, the square is Cell 4j's
last one, rebuilt (iteration_fits.node_square), and Block 5's inputs at
k_iter are taken from plane_differences.refit_at_plane(with_profile=True):
the profile (v_n, I_n) through the pilot's fitted centre, the tilt phi and
B_bar; the plane is not re-chosen. Every variant of
analysis.fit_variants.variant_grid (by default 10 kernels x 2 width modes x
5 weightings) is fitted to it (fit_variants.fit_grid); the variant gauss,
fixed width, unweighted is Block 5's fit (d_base) and must equal Cell 4j's
d_iter, a difference being noted. Writes, into --out-dir, fitvar_<id>.png
per node (the profile with the fits and their edges, and d of every variant
as a table), fitvar_summary_<specimen>.png (d / d_base against d_base, one
panel per kernel and width mode), fitvar_<specimen>.csv (one row per node
and variant) and fitvar_<specimen>.json (per node the inputs, the rows and
notes).

    python scripts/fit_variants.py --specimen 529878215 \
        --iter-json OUT/pilot/planeiter/bydiameter_m3/planeiter_529878215.json \
        --out-dir OUT/pilot/planeiter/bydiameter_m3/fitvar --cache-dir CACHE

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for _p in (SRC, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import iteration_fits as IF  # noqa: E402
import plane_differences as PD  # noqa: E402
from allen_diameter.analysis import fit_variants as FV  # noqa: E402
from allen_diameter.model import blur_kernels as BK  # noqa: E402
from build_table import load_config  # noqa: E402

PREFIX = "fitvar"
NO_INPUT = ("no_plane", "profile_nan", "bbar_nonpositive")


# --------------------------------------------------------------- analysis ---

def node_inputs(swc, provider, cfg, rec, transform=None, half_um=None):
    """Block 5's inputs at Cell 4j's k_iter: (st, fit, notes), fit the dict of
    refit_at_plane(st, cfg, k_iter, with_profile=True) (v_um, profile,
    phi_rad, B_bar, and Block 5's d_hat_um there); notes as node_square's,
    plus a d_hat differing from the record's d_iter_um. ValueError without
    k_iter, or when the plane gives no profile or no positive B_bar."""
    nid = int(rec["node_id"])
    if rec.get("k_iter") is None:
        raise ValueError("node %d: no plane chosen (iteration %s)" % (nid, rec.get("iter_status")))
    st, notes = IF.node_square(swc, provider, cfg, rec, transform, half_um)
    k = int(rec["k_iter"])
    fit = PD.refit_at_plane(st, cfg, k, with_profile=True)
    if fit["fit_status"] in NO_INPUT:
        raise ValueError("node %d: no fit at k_iter %d (%s)" % (nid, k, fit["fit_status"]))
    d, d_rec = float(fit["d_hat_um"]), rec.get("d_iter_um")
    d_rec = float("nan") if d_rec is None else float(d_rec)
    if not (d == d_rec or (math.isnan(d) and math.isnan(d_rec))):
        notes.append("Block 5 at k_iter gives d %.4f um here, Cell 4j's d_iter %.4f um" % (d, d_rec))
    return st, fit, notes


def flag_of(f):
    """'x' failed, '*' at a bound, '~' IRLS not settled (after the first two)."""
    s = "x" if f.status == "failed" else ("*" if f.status == "at_bound" else "")
    if f.irls_ok is False and f.status != "failed":
        s += "~"
    return s


def least_rss(fits):
    """Per width mode, the unweighted fit of least residual sum of squares
    among those not failed: {width mode: VariantFit}. Kernels with more
    fitted parameters can only lower it; weighted fits minimise another sum
    and are left out."""
    out = {}
    for f in fits:
        if f.spec.lam_um is None and f.status != "failed" and np.isfinite(f.rss_gl2):
            m = f.spec.width_mode
            if m not in out or f.rss_gl2 < out[m].rss_gl2:
                out[m] = f
    return out


def kernel_rows(specs):
    """The kernel configurations present in specs, in KERNEL_CONFIGS order, as
    (kernel tuple, label, second), second True for a family's second fixed
    shape (gennorm beta = 4)."""
    present = {s.kernel for s in specs}
    out, seen_fixed = [], set()
    for kern in FV.KERNEL_CONFIGS:
        fam, shape, fitted = kern
        second = False
        if not fitted and shape is not None:
            second = fam in seen_fixed
            seen_fixed.add(fam)
        if kern in present:
            out.append((kern, BK.kernel_label(fam, shape, fitted), second))
    return out


def node_block(label, sublabel, fit5, fits, base, specs, cfg):
    """The per-node figure's block (figures.fit_variants_node_figure) from the
    inputs and the variants' fits."""
    v, I, B = np.asarray(fit5["v_um"], dtype=float), np.asarray(fit5["profile"], dtype=float), float(fit5["B_bar"])
    phi = float(fit5["phi_rad"])
    vf = np.linspace(v[0], v[-1], 241)
    by = {f.spec: f for f in fits}

    def curve(f):
        return FV.variant_curve(f, vf, phi, B) if np.isfinite(f.d_um) else None

    from allen_diameter.plotting import figures as fg
    krows = [(kern, lab) + fg.kernel_style(kern[0], kern[2], second) for kern, lab, second in kernel_rows(specs)]
    panels = {}
    for mode in ("fixed", "fitted"):
        items = []
        for kern, lab, colour, ls in krows:
            f = by.get(FV.VariantSpec(kern[0], kern[1], kern[2], mode == "fitted", None))
            if f is None:
                continue
            items.append(dict(label=lab, family=kern[0], ls=ls, curve=curve(f), d=f.d_um, v0=f.v0_um, flag=flag_of(f),
                              w=f.fwhm_um))
        panels[mode] = items
    lam_items = []
    for s in specs:
        if s.kernel == FV.BASE_KERNEL and not s.width_fitted:
            f = by[s]
            lam_items.append(dict(lam=s.lam_um, curve=curve(f), d=f.d_um, v0=f.v0_um, flag=flag_of(f),
                                  weights=None if s.lam_um is None or not np.isfinite(f.v0_um)
                                  else FV.centre_weights(v, f.v0_um, s.lam_um)))
    weights = []
    for s in specs:
        if s.weight_label not in weights:
            weights.append(s.weight_label)
    modes = [m for m in FV.WIDTH_MODES if any(s.width_mode == m for s in specs)]
    cols = [(m, w) for m in modes for w in weights]
    R, C = len(krows), len(cols)
    dmat, rmat, flags = [[float("nan")] * C for _ in range(R)], np.full((R, C), np.nan), [[""] * C for _ in range(R)]
    base_rc, best_rc = None, []
    best = {f.spec for f in least_rss(fits).values()}
    key = {(s.kernel, s.width_mode, s.weight_label): by[s] for s in specs}
    for r, (kern, _lab, _c, _ls) in enumerate(krows):
        for c, (m, w) in enumerate(cols):
            f = key.get((kern, m, w))
            if f is None:
                continue
            dmat[r][c] = f.d_um
            rmat[r, c] = FV.variant_row(f, base)["ratio"]
            flags[r][c] = flag_of(f)
            if kern == FV.BASE_KERNEL and m == "fixed" and w == "none":
                base_rc = (r, c)
            if f.spec in best:
                best_rc.append((r, c))
    table = dict(rows=[x[1] for x in krows], row_colours=[x[2] for x in krows], row_ls=[x[3] for x in krows],
                 cols=cols, d=dmat, ratio=rmat, flags=flags, base=base_rc, best=best_rc)
    return dict(label=label, sublabel=sublabel, v=v, I=I, B_bar=B, v_fine=vf, fixed=panels.get("fixed", []),
                fitted=panels.get("fitted", []), lam=lam_items, table=table,
                fixed_fwhm_um=FV.fixed_fwhm_um(cfg.measure))


def _range(fits):
    d = [f.d_um for f in fits if np.isfinite(f.d_um)]
    return (min(d), max(d)) if d else (float("nan"), float("nan"))


def node_log(nid, rec, st, fit5, fits, base, specs, seconds, notes):
    """The node's log line (module docstring of the notebook cell)."""
    by = {f.spec: f for f in fits}
    lam = [s for s in specs if s.kernel == FV.BASE_KERNEL and not s.width_fitted and s.lam_um is not None]
    wtxt = ("Gaussian centre-weighted d %s (lambda %s um)" % (
        "/".join("%.2f" % by[s].d_um for s in lam), "/".join("%g" % s.lam_um for s in lam))) if lam else "no weights"
    parts = []
    for mode in FV.WIDTH_MODES:
        sel = [by[s] for s in specs if s.lam_um is None and s.width_mode == mode]
        if sel:
            lo, hi = _range(sel)
            parts.append("%s width d %.2f-%.2f" % (mode, lo, hi))
    gw = by.get(FV.VariantSpec("gauss", None, False, True, None))
    if gw is not None and np.isfinite(gw.fwhm_um):
        parts.append("the Gaussian's fitted FWHM %.2f um" % gw.fwhm_um)
    ok = [f for f in fits if np.isfinite(f.d_um) and f.status != "failed"]
    small = min(ok, key=lambda f: f.d_um) if ok else None
    lr = least_rss(fits)
    parts.append("least residual unweighted: " + ", ".join(
        "%s width %s d %.2f" % (m, lr[m].spec.kernel_label, lr[m].d_um) for m in FV.WIDTH_MODES if m in lr))
    n_flag = sum(1 for f in fits if flag_of(f))
    r = st["result"]
    return ("[fitvar] node %d: k_iter %d (k* %d); Block 5 d %.2f um (%s); %s; unweighted kernels: %s; smallest d %s; "
            "%d variants, %d flagged; %.0f s%s"
            % (nid, int(rec["k_iter"]), int(r.k_star), base.d_um, base.status, wtxt, ", ".join(parts),
               "n/a" if small is None else "%.2f (%s)" % (small.d_um, small.spec.key), len(fits), n_flag, seconds,
               "; " + "; ".join(notes) if notes else ""))


# ----------------------------------------------------------------- wiring ---

def select_specs(families=None, lambdas_um=None, width_modes=None):
    """The variants to fit: the grid restricted to the families (the Gaussian
    always kept: it is the baseline and the weighted column), the lambdas
    and the width modes."""
    fams = None if not families else set(families) | {"gauss"}
    bad = sorted(set(families or ()) - set(BK.FAMILIES))
    if bad:
        raise ValueError("unknown families %s (from %s)" % (bad, ", ".join(BK.FAMILIES)))
    kernels = [k for k in FV.KERNEL_CONFIGS if fams is None or k[0] in fams]
    lams = FV.WEIGHT_LAMBDAS_UM if lambdas_um is None else tuple(float(x) for x in lambdas_um)
    modes = FV.WIDTH_MODES if not width_modes else tuple(width_modes)
    return FV.variant_grid(kernels, modes, lams)


def run(swc, provider, cfg, records, out_dir, specimen, transform=None, half_um=None, nodes=None, specs=None,
        dpi=110, log=print):
    """One figure per node with a plane chosen, the summary, the CSV and the
    JSON (the module docstring); nodes: ids to fit (default every node of
    the records); specs: the variants (default select_specs()). A record
    skipped by Cell 4j, without k_iter, or without inputs at k_iter is listed
    with its reason. Returns the per-node JSON records."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from allen_diameter.plotting import figures as fg
    os.makedirs(out_dir, exist_ok=True)
    specs = select_specs() if specs is None else list(specs)
    want = None if not nodes else {int(x) for x in nodes}
    out, rows, summ = [], [], []
    for rec in records:
        nid = int(rec["node_id"])
        if want is not None and nid not in want:
            continue
        if "skipped" in rec or rec.get("k_iter") is None:
            why = ("Cell 4j: " + str(rec["skipped"])) if "skipped" in rec else (
                "no plane chosen (iteration %s)" % rec.get("iter_status"))
            log("[fitvar] node %d skipped: %s" % (nid, why))
            out.append(dict(node_id=nid, skipped=why))
            continue
        try:
            st, fit5, notes = node_inputs(swc, provider, cfg, rec, transform, half_um)
        except ValueError as e:
            log("[fitvar] node %d skipped: %s" % (nid, e))
            out.append(dict(node_id=nid, skipped=str(e)))
            continue
        t0 = time.perf_counter()
        v, I = np.asarray(fit5["v_um"], dtype=float), np.asarray(fit5["profile"], dtype=float)
        fits, base = FV.fit_grid(v, I, float(fit5["phi_rad"]), float(fit5["B_bar"]), cfg.measure, specs)
        if not abs(base.d_um - float(fit5["d_hat_um"])) <= 1e-6:
            notes.append("the variants' baseline d %.6f um, Block 5's %.6f um" % (base.d_um, float(fit5["d_hat_um"])))
        sec = time.perf_counter() - t0
        r = st["result"]
        head = ("node %d: plane k_iter %d (k* %d); Block 5's fit there d %.2f um (%s); %d variants, d / d_base below"
                % (nid, int(rec["k_iter"]), int(r.k_star), base.d_um, base.status, len(fits)))
        sub = ("the profile is Block 5's at k_iter (the average of parallel lines through the pilot's fitted centre, "
               "+-%g um), phi %.1f deg, B_bar %.1f; the fixed width is the FWHM of Block 5's Gaussian (sigma_fit "
               "%g um); weights exp(-|v - v0| / lambda) re-centred on the fit's own v0 (IRLS)%s"
               % (float(cfg.measure.profile_half_um), math.degrees(float(fit5["phi_rad"])), float(fit5["B_bar"]),
                  float(cfg.measure.sigma_fit_um), "; " + "; ".join(notes) if notes else ""))
        block = node_block(head, sub, fit5, fits, base, specs, cfg)
        fig = fg.fit_variants_node_figure(block, "Fit variants at the plane the iteration chose, specimen %s" % specimen)
        png = os.path.join(out_dir, "%s_%d.png" % (PREFIX, nid))
        fig.savefig(png, dpi=dpi)
        plt.close(fig)
        vrows = [FV.variant_row(f, base) for f in fits]
        meta = dict(node_id=nid, k_iter=int(rec["k_iter"]), k_star=int(r.k_star), k_swc=int(st["k_swc"]),
                    d_iter_um=rec.get("d_iter_um"))
        for vr in vrows:
            rows.append(dict(meta, **vr))
        out.append(dict(meta, d_block5_um=float(fit5["d_hat_um"]), d_base_um=base.d_um, phi_rad=float(fit5["phi_rad"]),
                        B_bar=float(fit5["B_bar"]), v_um=[float(x) for x in v], profile=[float(x) for x in I],
                        variants=vrows, notes=notes, png=png, seconds=sec))
        summ.append(dict(node_id=nid, d_base=base.d_um,
                         ratio={(f.spec.kernel_label, f.spec.width_mode, f.spec.weight_label):
                                FV.variant_row(f, base)["ratio"] for f in fits},
                         flag={(f.spec.kernel_label, f.spec.width_mode, f.spec.weight_label): flag_of(f)
                               for f in fits}))
        log(node_log(nid, rec, st, fit5, fits, base, specs, sec, notes))
    if summ:
        kern_labels = [x[1] for x in kernel_rows(specs)]
        fig = fg.fit_variants_summary_figure(summ, kern_labels, "d / d_base per node for every kernel, width mode and "
                                                                "weighting, specimen %s" % specimen)
        fig.savefig(os.path.join(out_dir, "%s_summary_%s.png" % (PREFIX, specimen)), dpi=dpi)
        plt.close(fig)
    if rows:
        with open(os.path.join(out_dir, "%s_%s.csv" % (PREFIX, specimen)), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out_dir, "%s_%s.json" % (PREFIX, specimen)), "w") as f:
        json.dump(out, f, indent=1)
    done = [x for x in out if "skipped" not in x]
    same = sum(1 for x in done if not any(n.startswith("the variants' baseline") or n.startswith("Block 5 at k_iter")
                                          for n in x["notes"]))
    log("[fitvar] %d nodes fitted, %d skipped; %d variants per node; the baseline equals Cell 4j's d_iter on %d of %d"
        % (len(done), len(out) - len(done), len(specs), same, len(done)))
    return out


def _csv_list(text, conv, what, ap):
    try:
        return [conv(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as e:
        ap.error("%s: %s" % (what, e))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--iter-json", required=True, help="Cell 4j's planeiter_<specimen>.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nodes", default="", help="comma-separated node ids (default: every node of the record)")
    ap.add_argument("--families", default="", help="comma-separated kernel families (default all: %s); the Gaussian "
                                                   "is always fitted" % ",".join(BK.FAMILIES))
    ap.add_argument("--lambdas", default=",".join("%g" % x for x in FV.WEIGHT_LAMBDAS_UM),
                    help="comma-separated lambdas (um) of the centre weights; empty for unweighted fits only")
    ap.add_argument("--width-modes", default="fixed,fitted", help="fixed, fitted or both")
    ap.add_argument("--half-um", type=float, default=None,
                    help="Cell 4j's --half-um, if it was given one (Cell 4j passes none)")
    ap.add_argument("--swc", default="")
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--shift-x", type=float, default=0.0, help="full-res px (global alignment), as in Cell 4j")
    ap.add_argument("--shift-y", type=float, default=0.0)
    ap.add_argument("--flip-h", type=float, default=None)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    nodes = _csv_list(a.nodes, int, "--nodes takes comma-separated node ids", ap)
    lams = _csv_list(a.lambdas, float, "--lambdas takes comma-separated numbers", ap)
    modes = _csv_list(a.width_modes, str, "--width-modes", ap)
    if not modes or any(m not in FV.WIDTH_MODES for m in modes):
        ap.error("--width-modes takes fixed, fitted or both")
    if any(not (math.isfinite(x) and x > 0) for x in lams):
        ap.error("--lambdas must be finite and > 0")
    try:
        specs = select_specs(_csv_list(a.families, str, "--families", ap), lams, modes)
    except ValueError as e:
        ap.error(str(e))
    cfg = load_config(a.config_json)
    records = IF.read_records(a.iter_json)
    import allen_image_io as aio
    import run_cell
    from allen_diameter.loading import swc_io
    swc = swc_io.read_swc(a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir))
    fetcher = aio.HttpFetcher(cache_dir=a.cache_dir)
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
    t0 = time.perf_counter()
    print("[fitvar] %d variants per node (%s); 25-75 s per node on one CPU, longer on thick dendrites" % (len(specs), ", ".join(
        sorted({s.family for s in specs}, key=BK.FAMILIES.index))), flush=True)
    run(swc, provider, cfg, records, a.out_dir, a.specimen,
        dict(shift_full_px=(a.shift_x, a.shift_y), flip_y_full_h=a.flip_h, z0_um=a.z0), a.half_um, nodes, specs,
        log=lambda m: print(m, flush=True))
    print("[fitvar] files in %s; crops: %d from the cache, %d downloaded (%.1f MB); %.1f min"
          % (a.out_dir, fetcher.n_cache_hits, fetcher.n_requests, fetcher.bytes_downloaded / 1e6,
             (time.perf_counter() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
