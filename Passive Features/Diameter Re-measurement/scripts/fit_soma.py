"""Fit the soma of a cell's image stack: the entry point of SPEC Blocks 45-51 (notebook Cell 25; D-215, D-216, D-218,
D-219, D-220, D-222, D-252 to D-258).

The traced soma (the SWC's type-1 node in the image frame, ALIGN's transform; Block 44) gives the start: its centre and
radius, and its depth from --soma-k (the plane judged in focus by eye in Cell 24) or else the traced depth. The fit's
region is the disk of soma_fit.region_radius_um, or soma_fit.region_radius_scale x Allen's traced soma radius (D-241,
D-255), about the traced centre. The crop is that radius plus soma_fit.crop_margin_um, at least soma_fit.render_half_um
(the renderer draws the ellipsoid on the crop alone; D-243; the record's crop.ellipsoid_inside says whether the fitted
footprint stayed inside), on the planes within soma_fit.plane_half_um of the start depth (Block 44's window rule). Every
traced process in the crop is masked at its traced radius plus soma_fit.mask_widen_um (0 by default, D-241: only the
pixels the branches cover; Block 44's segments; those from the soma node to the stems' first nodes lie inside the soma
and are not masked). The ellipsoid is fitted by Block 46 with Block 45's renderer as the forward model (the kernel of
--config-json, whose continuation must be "linear", D-220.2), each plane's background tilted (b0 + b1 u + b2 v,
D-254.1; --background), the kernel's far slope held (D-240; --far-slope) and its in-focus width held (D-258; --sigma0).
The depth semi-axis c is held at each value of a grid (--c-grid, D-256.1) and the best grid fit found; with
--c-refine on, one more fit with c free in the best value's neighbourhood (Block 50; D-252, D-256). The fit delivered is
Block 50's (D-256.2); each usable fit's backgrounds are compared with the far background (Block 50's check, D-257), a
flag logged as a WARNING. A run with the far slope fitted, the background constant or sigma0 fitted is a labelled
COMPARISON: no corrected SWC (D-240, D-254.1, D-258.2). Written into --out (default ./soma_fit_<S>):

  soma_fit_<S>.json              the delivered fit (theta, SEs, covariance, the planes' backgrounds, chi2, status), the
                                 membrane area, volume and equivalent radius, the grid (its table, each fit's status,
                                 the best, the refinement, the source, the flags), the background check, the start, the
                                 crop, the settings, Allen's soma and the stems' moves; "analysis" deliverable or
                                 comparison
  soma_fit_<S>_planes.png        data, model and residual on the fitted planes nearest soma_fit.figure_planes_um from
                                 the fitted centre (Block 48)
  soma_fit_<S>_summary.png       the parameters, the derived quantities beside Allen's, the backgrounds and residuals by
                                 depth
  soma_fit_<S>_morphology.png    the soma and the stems' first nodes before and after, top and side views
  soma_fit_<S>_cgrid.png         the fits over the grid of held c (Block 48's c-grid figure), whenever a grid fit was made
  reconstruction_soma_<S>.swc    the corrected SWC (Block 47; the deliverable only): the soma as the sphere of equal area
                                 at the fitted centre, each stem's first node on the fitted surface (D-216.3, D-222.5)

Exit status 0 when a fit is delivered, 1 when no grid fit is usable (the record is still written, its fit NaN).

--synthetic: the smoke tests' synthetic cell's trace (geometry, alignment, mask) over a known ellipsoid soma rendered
on a tilted background, B(u, v) = 110 + 0.10 u - 0.05 v gl (u, v um from the traced centre), with noise, so the dry run
is also a recovery check (no network); without --c-grid its grid is (0.8, 1.0, 1.2) x the true c (Block 51 (10)).
Pure ASCII, LF.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import types as _types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from allen_diameter.analysis import cell, soma_fit, soma_morphology, soma_profile, soma_view   # noqa: E402
from allen_diameter.loading import swc_io                                         # noqa: E402
from allen_diameter.model import kernel                                           # noqa: E402
from allen_diameter.model.soma_render import SomaRenderer                         # noqa: E402

DPI = 130
# The dry run's soma (test data, not a parameter): offsets of the truth from the traced soma, and its shape relative to
# the traced radius r: centre (+0.6, -0.4, +0.9) um, a = 1.10 r, b = 0.85 r, c = 0.75 r, phi 0.6 rad, mu 0.15 /um;
# the background b0 + b1 u + b2 v = 110 + 0.10 u - 0.05 v gl (u, v um from the traced centre; Block 51 (10)), noise SD
# 3 gl, rounded (seed 529); the grid without --c-grid: these multiples of the true c.
SYN_OFFSET_UM = (0.6, -0.4, 0.9)
SYN_AXES_REL = (1.10, 0.85, 0.75)
SYN_PHI_RAD, SYN_MU_PER_UM, SYN_B_GL, SYN_NOISE_GL, SYN_SEED = 0.6, 0.15, 110.0, 3.0, 529
SYN_BETA = (SYN_B_GL, 0.10, -0.05)
SYN_GRID_REL = (0.8, 1.0, 1.2)
# the deliverable's settings (D-240, D-254.1, D-258.2): any other value makes the run a labelled comparison
DELIVERABLE = (("far_slope", "fixed"), ("background", "tilted"), ("sigma0", "fixed"))


def _circle(x, y, r, n=256):
    t = np.linspace(0.0, 2.0 * math.pi, n)
    return np.vstack([x + r * np.cos(t), y + r * np.sin(t)])


def _json_safe(v):
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, np.ndarray):
        return _json_safe(v.tolist())
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(float(v)) else float(v)
    return v


def _mask(xyz, swc, parent_index, types, box, shape, left, top, res0, widen):
    """Every traced process whose widened disc reaches the crop, at its traced radius + widen (um): Block 44's swept
    discs. The segments are taken from the box widened by `widen` (D-223.1: a process just outside the crop whose
    widened mask reaches into it is kept)."""
    x0, x1, y0, y1 = box
    seg = soma_view.trace_segments(xyz, swc.radius, parent_index, types,
                                   (x0 - widen, x1 + widen, y0 - widen, y1 + widen))
    if len(seg["r1"]) == 0:
        return np.zeros(shape, dtype=bool), 0
    m = soma_view.segment_mask(shape, left, top, res0, seg["p1"], seg["p2"], seg["r1"] + widen, seg["r2"] + widen)
    return m, int(len(seg["r1"]))


def _grid_arg(text):
    """--c-grid: comma-separated numbers (um); the order and the bounds are checked against the configuration later."""
    try:
        vals = tuple(float(t) for t in str(text).split(",") if t.strip())
    except ValueError:
        raise argparse.ArgumentTypeError("--c-grid must be comma-separated numbers (um), got %r" % text)
    if not vals:
        raise argparse.ArgumentTypeError("--c-grid holds no value")
    return vals


def _grid_problem(grid, axis_bounds_um, refine):
    """None, or why the grid cannot run: Block 50 (1)-(2)'s conditions (finite, strictly ascending, inside
    axis_bounds_um; at least two values to refine)."""
    c = np.asarray(grid, dtype=float)
    lo, hi = (float(v) for v in axis_bounds_um)
    if c.size < 1 or not np.all(np.isfinite(c)):
        return "the c grid must hold finite values, got %s" % list(grid)
    if c.size > 1 and not np.all(np.diff(c) > 0):
        return "the c grid must be strictly ascending, got %s" % list(grid)
    if c[0] < lo or c[-1] > hi:
        return "the c grid %s must lie inside soma_fit.axis_bounds_um [%g, %g]" % (list(grid), lo, hi)
    if refine and c.size < 2:
        return "--c-refine on needs at least two grid values (the neighbourhood of the best one)"
    return None


def _rms(f):
    try:
        return math.sqrt(float(f.chi2) / float(f.n_samples))
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def _usable(f):
    """Block 50 (1)'s rule: a fit is usable when its status is not "failed" and its chi2 is finite."""
    return f is not None and str(f.status) != "failed" and math.isfinite(float(f.chi2))


def _nan_fit(f):
    """The record's fit when nothing is delivered (Block 51 (9)): a grid fit's layout with NaN results."""
    nan = float("nan")
    return dataclasses.replace(f, theta=np.full(9, nan), se=np.full(9, nan), cov=np.full((9, 9), nan),
                               B_gl=np.full(np.asarray(f.ks).size, nan), beta=np.full((np.asarray(f.ks).size, 3), nan),
                               cov_fit=np.full(np.shape(f.cov_fit), nan), chi2=nan, area_um2=nan, se_area_um2=nan,
                               volume_um3=nan, r_eq_um=nan, at_bound=(), status="failed",
                               message="no usable grid fit: nothing delivered")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--shift-x", type=float, default=None)
    ap.add_argument("--shift-y", type=float, default=None)
    ap.add_argument("--flip-h", type=float, default=None)
    ap.add_argument("--z0", type=float, default=None)
    ap.add_argument("--config-json", default=None, help="the cleaning configuration (Cell 16): its kernel is used")
    ap.add_argument("--swc", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic", action="store_true", help="the smoke tests' synthetic cell with a known soma")
    ap.add_argument("--soma-k", type=int, default=None, help="the start plane (the soma judged in focus, Cell 24); "
                    "default: the traced depth")
    ap.add_argument("--far-slope", choices=("fit", "fixed"), default=None,
                    help="hold the kernel's far slope at --slope0 (D-240, the default: soma_fit.far_slope) or, as a "
                         "labelled comparison only, fit it (D-218.1, superseded)")
    ap.add_argument("--slope0", type=float, default=None,
                    help="the far slope's start, or its held value (um/um); default renderer.kernel_continuation_slope")
    ap.add_argument("--c-grid", type=_grid_arg, default=None,
                    help="the held values of the depth semi-axis c, comma-separated um, strictly ascending, inside "
                         "soma_fit.axis_bounds_um (default soma_fit.c_grid_um, D-256.1; --synthetic: 0.8, 1, 1.2 x "
                         "the true c)")
    ap.add_argument("--c-refine", choices=("on", "off"), default=None,
                    help="one more fit with c free in the best grid value's neighbourhood (default soma_fit.c_refine:"
                         " off on Colab, D-252.1)")
    ap.add_argument("--sigma0", choices=("fixed", "fit"), default=None,
                    help="the kernel's in-focus width held (the default, soma_fit.sigma0) or, as a labelled parallel "
                         "analysis, fitted (D-258)")
    ap.add_argument("--background", choices=("tilted", "constant"), default=None,
                    help="each plane's background b0 + b1 u + b2 v (the default, soma_fit.background, D-254.1) or, as a "
                         "labelled comparison, one scale per plane")
    import kernel_width                                           # Block 36 amendment 22 (D-221.1, D-222.3)
    kernel_width.add_arguments(ap)
    a = ap.parse_args(argv)
    import clean_cell
    from build_table import load_config
    from allen_diameter.config import default_config
    from allen_diameter.plotting import soma_fit_figures as FF
    import matplotlib.pyplot as plt                               # the figures are closed after saving (D-233, F5)
    from allen_diameter.plotting.figures import image_extent_um
    from allen_diameter.plotting.soma_figures import save
    cfg = load_config(a.config_json) if a.config_json else default_config()
    try:                                                          # the width switch (Block 36 amendment 22)
        cfg, kw = kernel_width.apply(cfg, a.kernel_width, a.sigma_json)
    except ValueError as exc:
        ap.error(str(exc))
    rcfg, q = cfg.renderer, cfg.soma_fit
    over = {}                                                     # Block 51 (2): the options replace the fields
    if a.far_slope is not None:
        over["far_slope"] = a.far_slope
    if a.c_grid is not None:
        over["c_grid_um"] = tuple(a.c_grid)
    if a.c_refine is not None:
        over["c_refine"] = a.c_refine == "on"
    if a.sigma0 is not None:
        over["sigma0"] = a.sigma0
    if a.background is not None:
        over["background"] = a.background
    q = dataclasses.replace(q, **over)
    if rcfg.kernel_continuation != "linear":                       # D-220.2: gamma is the linear continuation's slope
        ap.error("the soma fit needs renderer.kernel_continuation 'linear' (D-218, D-220.2), got %r"
                 % rcfg.kernel_continuation)
    slope0 = float(rcfg.kernel_continuation_slope if a.slope0 is None else a.slope0)
    if not (math.isfinite(slope0) and slope0 >= 0):
        ap.error("--slope0 must be finite and >= 0")
    syn_grid = a.synthetic and a.c_grid is None                  # Block 51 (10): the grid set from the truth below
    if not syn_grid:
        why = _grid_problem(q.c_grid_um, q.axis_bounds_um, q.c_refine)
        if why:
            ap.error(why)
    s0 = float(kernel.sigma_r(0.0, rcfg))                         # the configured kernel's in-focus width (um)
    res0, dz = float(cfg.acquisition.res0_um), float(cfg.acquisition.dz_um)
    light = int(cfg.acquisition.light_direction)
    out = a.out or os.path.join(os.getcwd(), "soma_fit_%s" % a.specimen)
    os.makedirs(out, exist_ok=True)
    log = lambda msg: print("[fit_soma] %s" % msg, flush=True)      # noqa: E731
    log(kernel_width.describe(kw))
    t0 = time.time()

    # the traced soma and the start (Block 44's geometry)
    swc, provider, label = clean_cell.load_cell(a, cfg, log, need_images=True)
    transform = clean_cell.transform_of(a)
    xyz = cell.to_image_um(swc.xyz, res0, **transform)
    types = np.asarray(swc.types)
    centre, radius = soma_view.soma_geometry(xyz, swc.radius, types)
    z_start = float(centre[2]) if a.soma_k is None else int(a.soma_k) * dz
    start_src = "traced depth" if a.soma_k is None else "plane %d (by eye)" % int(a.soma_k)
    start = (float(centre[0]), float(centre[1]), z_start)
    cxy = (float(centre[0]), float(centre[1]))

    # the crop: the fit's disk plus a margin, at least render_half_um (the renderer draws the ellipsoid on the crop
    # alone, D-243), the planes within plane_half_um of the start depth; the disk's radius is region_radius_um when a
    # number, else region_radius_scale x Allen's traced soma radius (D-241, D-255; Block 51 (1))
    r_region = (float(q.region_radius_scale) * float(radius) if q.region_radius_um is None
                else float(q.region_radius_um))
    half = max(float(q.render_half_um), r_region + float(q.crop_margin_um))
    left, top, width, height = soma_view.crop_box(centre[:2], half, res0)
    k_lo, k_hi = soma_view.plane_range(z_start, float(q.plane_half_um), dz)
    log("%s cell: traced soma at (%.2f, %.2f, %.2f) um, radius %.2f um; start depth %.2f um (%s); crop %d x %d px at "
        "(%d, %d); planes %d..%d" % (label, centre[0], centre[1], centre[2], radius, z_start, start_src, width, height,
                                     left, top, k_lo, k_hi))
    block, ks, valid, _frame = provider(left, top, width, height, k_lo, k_hi)
    block = np.asarray(block, dtype=float)
    ks = np.asarray(ks).astype(int)
    valid = np.asarray(valid, dtype=bool)
    make = lambda z: SomaRenderer((left, top, width, height), res0, z, rcfg, q, light).render   # noqa: E731
    truth = None
    if a.synthetic:                                                 # a known soma into the synthetic planes
        r = float(radius)
        truth = np.array([centre[0] + SYN_OFFSET_UM[0], centre[1] + SYN_OFFSET_UM[1], z_start + SYN_OFFSET_UM[2],
                          SYN_AXES_REL[0] * r, SYN_AXES_REL[1] * r, SYN_AXES_REL[2] * r, SYN_PHI_RAD, SYN_MU_PER_UM,
                          slope0])
        tau = make(ks * dz)(truth[:8], float(truth[8]))
        rng = np.random.default_rng(SYN_SEED)
        # the soma alone on the tilted background, every plane recorded: the synthetic cell's image covers only part of
        # the crop and records planes -7..7, and composing it leaves a step at its edge that the fit would explain
        # (2026-11-23, the first dry runs); the trace's mask is applied all the same
        bg = soma_fit.background_planes(np.tile(np.asarray(SYN_BETA, float), (ks.size, 1)), tau.shape[1:], left, top,
                                        res0, cxy)
        block = np.clip(np.rint(bg * tau + rng.normal(0.0, SYN_NOISE_GL, tau.shape)), 0, 255)
        valid = np.ones(ks.size, dtype=bool)
        log("synthetic soma rendered: centre (%.2f, %.2f, %.2f) um, a b c %.2f %.2f %.2f um, phi %.2f, mu %.2f /um; "
            "background %.1f + %.2f u + %.2f v gl" % (tuple(truth[:8]) + SYN_BETA))
        if syn_grid:
            q = dataclasses.replace(q, c_grid_um=tuple(float(m * truth[5]) for m in SYN_GRID_REL))
            why = _grid_problem(q.c_grid_um, q.axis_bounds_um, q.c_refine)
            if why:
                ap.error(why)

    # the mask from Allen's trace, widened
    index_of = {int(i): k for k, i in enumerate(swc.ids)}
    parent_index = np.array([index_of.get(int(pp), -1) for pp in np.asarray(swc.parent)])
    box = (left * res0, (left + width) * res0, top * res0, (top + height) * res0)
    mask, n_seg = _mask(xyz, swc, parent_index, types, box, block.shape[1:], left, top, res0, float(q.mask_widen_um))
    log("region: the %.2f um disk about the traced centre (%s); mask: %d traced segments at their radii + %.1f um, "
        "%.1f %% of the crop" % (r_region, "%.2f x Allen's soma radius" % q.region_radius_scale
                                  if q.region_radius_um is None else "configured", n_seg, q.mask_widen_um,
                                  100.0 * mask.mean()))
    diffs = ["%s %s" % (name, getattr(q, name)) for name, want in DELIVERABLE if getattr(q, name) != want]
    analysis = "comparison" if diffs else "deliverable"
    prefix = "COMPARISON (%s): " % ", ".join(diffs) if diffs else ""
    log("settings: far slope %s, background %s, sigma0 %s (start %.4f um), c grid %s um, refine %s -- %s"
        % (q.far_slope, q.background, q.sigma0, s0, ", ".join("%.3f" % c for c in q.c_grid_um),
           "on" if q.c_refine else "off", prefix[:-2] if diffs else "the deliverable"))

    # the fits (Block 46 with Block 45's renderer) over the grid of held c, and the refinement (Block 50)
    def fit_one(hold, start_theta, c_bounds_um):
        f = soma_fit.fit_soma(block, ks, valid, left, top, res0, dz, start, float(radius), mask, make, q, slope0,
                              hold=hold, start=start_theta, c_bounds_um=c_bounds_um, sigma0_um=s0)
        what = ("c held at %.3f um" % hold["c_um"]) if "c_um" in hold else (
            "refinement, c in [%.3f, %.3f] um" % tuple(c_bounds_um))
        log("fit (%s): %s, c %.3f um, RMS %.3f gl, a %.3f, b %.3f um, S %.2f um^2, %d evaluations, %.0f s%s"
            % (what, f.status, f.theta[5], _rms(f), f.theta[3], f.theta[4], f.area_um2, f.n_eval, f.seconds,
               ("; at a bound: " + ", ".join(f.at_bound)) if f.at_bound else ""))
        return f

    cg = soma_profile.fit_c_grid(fit_one, q.c_grid_um, q.c_refine, q.axis_bounds_um)
    table = soma_profile.grid_table(cg)
    if cg.best_index >= 0:
        log("best grid value: c %.3f um (index %d, RMS %.3f gl)" % (cg.c_grid_um[cg.best_index], cg.best_index,
                                                                    table["rms_gl"][cg.best_index]))
    else:
        log("no usable grid fit")
    if cg.neighbourhood_um is not None:
        log("neighbourhood [%.3f, %.3f] um; refinement %s" % (cg.neighbourhood_um + (
            "none" if cg.refinement is None else "%s, c %.3f um, RMS %.3f gl" % (
                cg.refinement.status, cg.refinement.theta[5], _rms(cg.refinement)),)))
    log("delivered: %s; flags: %s" % (cg.delivered_source, ", ".join(cg.flags) or "none"))

    # the far-background check (Block 50, D-257) for every usable fit: the far planes once (every fit has the same
    # planes, in the fit's ks order)
    i_of_k = {int(k): i for i, k in enumerate(ks)}
    ks_fit = np.asarray(cg.fits[0].ks).astype(int)
    z_fit = ks_fit * dz
    far = soma_profile.far_mask(block.shape[1:], left, top, res0, cxy, q.bgcheck_far_um, mask)
    far_b, far_ok, _n_used = soma_profile.far_background(np.stack([block[i_of_k[int(k)]] for k in ks_fit]), far,
                                                         left, top, res0, cxy,
                                                         cfg.calibration.background_trim_sigmas)

    def check(f):
        if not _usable(f):
            return None
        return soma_profile.background_check(f.beta, z_fit, f.theta[2], far_b, far_ok, q.bgcheck_near_um,
                                             q.bgcheck_tol_gl)

    checks = [check(f) for f in cg.fits]
    check_ref = check(cg.refinement) if cg.refinement is not None else None
    check_del = (check_ref if cg.delivered_source == "refinement" else
                 checks[cg.best_index] if cg.delivered_source == "grid" else None)
    check_mean = np.array([np.nan if c is None else c["mean_gl"] for c in checks], dtype=float)
    log("far background: %d pixels at least %.1f um from the traced centre, %d of %d planes fitted"
        % (int(far.sum()), q.bgcheck_far_um, int(far_ok.sum()), ks_fit.size))
    for name, c in [("c %.3f um" % cc, ch) for cc, ch in zip(cg.c_grid_um, checks)] + [("refinement", check_ref)]:
        if c is None:
            continue
        log("background check (%s): %s, %d near planes, mean %.3f gl, max |Delta| %.3f gl%s"
            % (name, c["status"], c["n_planes"], c["mean_gl"], c["max_abs_gl"],
               "" if not c["flag"] else " -- flag raised"))
        if c["flag"]:
            log("WARNING: the fitted background (%s) differs from the far background by %.3f gl on average over the "
                "near planes (tolerance %.3f gl, D-257): the fit is delivered all the same" % (name, c["mean_gl"],
                                                                                              q.bgcheck_tol_gl))

    fit = cg.delivered if cg.delivered is not None else _nan_fit(cg.fits[0])
    th = np.asarray(fit.theta, dtype=float)
    ok = cg.delivered is not None and bool(np.all(np.isfinite(th[:8])))
    under_fit = q.sigma0 == "fit"
    if cg.delivered_source == "refinement":
        c_src = "c refined in [%.3g, %.3g] um" % cg.neighbourhood_um
    elif cg.delivered_source == "grid":
        c_src = "c held at %.3g um (the grid's best)" % cg.c_grid_um[cg.best_index]
    else:
        c_src = "no usable fit"
    title = "%sspecimen %s: the soma fit (%s; far slope %s, background %s, sigma0 %s)" % (
        prefix, a.specimen, label, q.far_slope, q.background, q.sigma0)
    # the fitted ellipsoid's footprint inside the rendered crop (D-243; conservative: the larger in-plane semi-axis)
    inside = (bool(max(abs(th[0] - centre[0]), abs(th[1] - centre[1])) + max(th[3], th[4]) <= half) if ok else None)
    if inside is False:
        log("WARNING: the fitted ellipsoid reaches beyond the %.2f um crop half-width: the renderer drew only its part "
            "inside the crop (D-243); raise soma_fit.render_half_um" % half)
    if ok:
        log("delivered (%s): centre (%.2f, %.2f, %.2f) um; a %.2f, b %.2f, c %.2f um (+- %.3f, %.3f, %.3f); phi %.2f "
            "rad; mu %.3f /um; slope %.3f; sigma0 %.4f um (+- %.4f); S %.1f +- %.1f um^2 (Allen's 4 pi r^2 %.1f), "
            "r_eq %.3f um (Allen's r %.3f); chi2/dof %.3f"
            % (c_src, th[0], th[1], th[2], th[3], th[4], th[5], fit.se[3], fit.se[4], fit.se[5], th[6], th[7], th[8],
               fit.sigma0_um, fit.se_sigma0_um, fit.area_um2, fit.se_area_um2, 4 * math.pi * radius ** 2,
               fit.r_eq_um, radius, fit.chi2 / max(fit.dof, 1)))

    # the outputs: the c-grid figure, the corrected SWC (the deliverable only), the figures, the record
    if len(cg.fits):
        if cg.delivered is None:
            dl = dict(source="none", c_um=np.nan, rms_gl=np.nan, a_um=np.nan, b_um=np.nan, area_um2=np.nan,
                      check_mean_gl=np.nan)
        else:
            dl = dict(source=cg.delivered_source, c_um=th[5], rms_gl=_rms(fit), a_um=th[3], b_um=th[4],
                      area_um2=fit.area_um2, check_mean_gl=np.nan if check_del is None else check_del["mean_gl"])
        fig = FF.figure_soma_c_grid(table, cg.best_index, dl, check_mean, q.bgcheck_tol_gl,
                                    4 * math.pi * float(radius) ** 2, neighbourhood_um=cg.neighbourhood_um,
                                    title=title + "\nthe fits over the grid of held c; " + c_src)
        save(fig, os.path.join(out, "soma_fit_%s_cgrid.png" % a.specimen), DPI)
        plt.close(fig)
    stems = []
    swc_out = None
    rms = None
    if ok:
        edit = soma_morphology.correct_soma(swc, th[:8], float(fit.r_eq_um), res0, **transform)
        if analysis == "deliverable":
            swc_out = os.path.join(out, "reconstruction_soma_%s.swc" % a.specimen)
            swc_io.write_swc(swc, swc_out, xyz=edit.xyz, radius=edit.radius, parent=edit.parent, keep=edit.keep)
        stems = [dict(s) for s in edit.stems]
        log("%s: the soma node at the fitted centre with r_eq; %d stems: %s"
            % ("corrected SWC" if swc_out else "no SWC written (a comparison); the stems would move", len(stems),
               ", ".join("%s %s %.2f um" % (s["id"], s["flag"], s["moved_um"]) for s in stems)))
        frame_ns = _types.SimpleNamespace(left=int(left), top=int(top), res_um_px=res0, downsample=0)
        ext = image_extent_um(frame_ns, block.shape)
        region = soma_view.disk_mask(block.shape[1:], left, top, res0, centre[:2], r_region) & ~mask
        render = make(z_fit)
        tau_fit = (render(th[:8], float(th[8]), sigma0_um=float(fit.sigma0_um)) if under_fit
                   else render(th[:8], float(th[8])))
        model_all = soma_fit.background_planes(fit.beta, block.shape[1:], left, top, res0, cxy) * tau_fit
        data_all = np.stack([block[i_of_k[int(k)]] for k in ks_fit])
        rms = np.sqrt(np.mean(((data_all - model_all)[:, region]) ** 2, axis=1))
        pick = sorted(set(int(np.argmin(np.abs(z_fit - th[2] - o))) for o in q.figure_planes_um))
        sil = np.vstack(soma_morphology.ellipse_outline(th[:8]))
        sections = []
        for i in pick:
            sx, sy = soma_morphology.ellipse_outline(th[:8], z_um=float(z_fit[i]))
            sections.append(np.vstack([sx, sy]) if sx.size else None)
        fig = FF.figure_soma_fit_planes(data_all[pick], model_all[pick], region, mask, ks_fit[pick],
                                        z_fit[pick] - th[2], ext, sil, sections, (centre[0], centre[1], radius),
                                        (th[0], th[1], fit.r_eq_um), title=title)
        save(fig, os.path.join(out, "soma_fit_%s_planes.png" % a.specimen), DPI)
        plt.close(fig)
        derived = dict(area_um2=fit.area_um2, se_area_um2=fit.se_area_um2, volume_um3=fit.volume_um3,
                       r_eq_um=fit.r_eq_um, se_volume_um3=_se_volume(th, fit.cov),
                       se_r_eq_um=fit.se_area_um2 / (8.0 * math.pi * fit.r_eq_um))   # delta method: dr/dS = 1/(8 pi r)
        allen = dict(radius_um=float(radius), area_um2=4 * math.pi * float(radius) ** 2)
        names, vals, ses = soma_fit.PARAMS, th, fit.se
        if under_fit:                                  # the fitted in-focus width as a tenth row (D-258)
            names = soma_fit.PARAMS + ("sigma0_um",)
            vals, ses = np.r_[th, fit.sigma0_um], np.r_[fit.se, fit.se_sigma0_um]
        fig = FF.figure_soma_fit_summary(names, vals, ses, fit.fitted, derived, allen, z_fit,
                                         np.asarray(fit.beta)[:, 0], rms, title=title + "\n" + c_src,
                                         at_bound=fit.at_bound)
        save(fig, os.path.join(out, "soma_fit_%s_summary.png" % a.specimen), DPI)
        plt.close(fig)
        top_v, side_v = _morphology_views(swc, edit, xyz, th, centre, radius, fit.r_eq_um, res0, transform)
        fig = FF.figure_soma_morphology(top_v, side_v, title=title)
        save(fig, os.path.join(out, "soma_fit_%s_morphology.png" % a.specimen), DPI)
        plt.close(fig)

    def short(f, c):
        return dict(theta=f.theta, chi2=f.chi2, status=f.status, at_bound=list(f.at_bound), n_eval=f.n_eval,
                    seconds=f.seconds, check_mean_gl=None if c is None else c["mean_gl"])

    grid = dict(c_grid_um=cg.c_grid_um, table={k: np.asarray(v) for k, v in table.items()},
                fits=[dict(c_um=float(cc), status=f.status, at_bound=list(f.at_bound), n_eval=f.n_eval,
                           seconds=f.seconds, check_mean_gl=None if c is None else c["mean_gl"])
                      for cc, f, c in zip(cg.c_grid_um, cg.fits, checks)],
                best_index=cg.best_index, neighbourhood_um=cg.neighbourhood_um,
                refinement=None if cg.refinement is None else short(cg.refinement, check_ref),
                delivered_source=cg.delivered_source, flags=list(cg.flags))
    bgc = None
    if check_del is not None:
        bgc = dict(check_del, far_pixels=int(far.sum()), far_ok_planes=int(far_ok.sum()))
    rec = dict(specimen=str(a.specimen), source=label, transform=transform, analysis=analysis, comparison=diffs,
               start=dict(centre_um=list(start), radius_um=float(radius), source=start_src),
               traced=dict(centre_um=[float(x) for x in centre], radius_um=float(radius),
                           area_um2=4 * math.pi * float(radius) ** 2),
               crop=dict(left=int(left), top=int(top), width=int(width), height=int(height), res0_um=res0,
                         k_lo=int(k_lo), k_hi=int(k_hi), n_mask_segments=n_seg, mask_fraction=float(mask.mean()),
                         region_radius_um=r_region, region_radius_scale=float(q.region_radius_scale), half_um=half,
                         ellipsoid_inside=inside),
               fit=dict(params=list(soma_fit.PARAMS), theta=th, se=fit.se, cov=fit.cov, fitted=list(fit.fitted),
                        ks=fit.ks, B_gl=fit.B_gl, beta=fit.beta, background=fit.background, sigma0_um=fit.sigma0_um,
                        se_sigma0_um=fit.se_sigma0_um, cov_fit=fit.cov_fit, chi2=fit.chi2, dof=fit.dof,
                        n_samples=fit.n_samples, status=fit.status, at_bound=list(fit.at_bound), n_eval=fit.n_eval,
                        seconds=fit.seconds, message=fit.message, rms_gl_per_plane=rms),
               grid=grid, background_check=bgc,
               derived=dict(area_um2=fit.area_um2, se_area_um2=fit.se_area_um2, volume_um3=fit.volume_um3,
                            se_volume_um3=_se_volume(th, fit.cov) if ok else None, r_eq_um=fit.r_eq_um,
                            se_r_eq_um=(fit.se_area_um2 / (8.0 * math.pi * fit.r_eq_um)) if ok else None),
               stems=stems, swc_out=swc_out, slope0=slope0, settings=dataclasses.asdict(q),
               kernel=dict(sigma_r0_um=rcfg.sigma_r0_um, sigma0_start_um=s0,
                           table_delta_um=list(rcfg.kernel_table_delta_um),
                           table_sigma_um=list(rcfg.kernel_table_sigma_um), continuation=rcfg.kernel_continuation,
                           continuation_slope=rcfg.kernel_continuation_slope), kernel_width=kw,
               synthetic_truth=None if truth is None else truth,
               synthetic_background_beta=None if truth is None else list(SYN_BETA), seconds=time.time() - t0)
    with open(os.path.join(out, "soma_fit_%s.json" % a.specimen), "w") as f:
        json.dump(_json_safe(rec), f, indent=1, allow_nan=False)
    if truth is not None and ok:
        err = th - truth
        berr = np.asarray(fit.beta, float) - np.asarray(SYN_BETA, float)[None, :]
        log("synthetic truth, the delivered fit's errors: centre %+.3f %+.3f %+.3f um, axes %+.3f %+.3f %+.3f um, "
            "phi %+.3f rad, mu %+.4f /um; background b0 %+.3f gl, b1 %+.4f, b2 %+.4f gl/um (mean over the planes); "
            "best grid index %d (the true c is index %d)"
            % (tuple(err[:8]) + tuple(berr.mean(axis=0)) + (cg.best_index, SYN_GRID_REL.index(1.0) if syn_grid
                                                             else -1)))
    written = ["soma_fit_%s.json" % a.specimen] + (["_cgrid.png"] if len(cg.fits) else [])
    if ok:
        written += ["_planes.png", "_summary.png", "_morphology.png"]
        if swc_out:
            written.append("reconstruction_soma_%s.swc" % a.specimen)
    log("wrote %s/%s%s; %s" % (out, ", ".join(written), "" if ok else " (no usable fit: nothing delivered)", analysis))
    return 0 if ok else 1


def _se_volume(th, cov):
    """SE of V = 4/3 pi a b c by the delta method on the (a, b, c) block of Block 46's covariance (NaN if absent)."""
    g = (4.0 / 3.0) * math.pi * np.array([th[4] * th[5], th[3] * th[5], th[3] * th[4]])
    S = np.asarray(cov, dtype=float)[3:6, 3:6]
    v = float(g @ S @ g)
    return math.sqrt(v) if math.isfinite(v) and v >= 0 else float("nan")


def _morphology_views(swc, edit, xyz, th, centre, radius, r_eq, res0, transform):
    """The arrays of Block 48's morphology figure: the top view in the image frame, the side view in the fitted
    ellipsoid's (u, w) frame (Block 47's fit_frame)."""
    ids = np.asarray(swc.ids)
    index_of = {int(i): k for k, i in enumerate(ids)}
    children = {}                                    # the edited tree (D-223.2): kept rows under their new parents
    for k in np.flatnonzero(edit.keep):
        children.setdefault(int(edit.parent[k]), []).append(int(k))
    stems_top, stems_side = [], []
    for s in edit.stems:
        old = cell.to_image_um(np.asarray(s["old_um"], float)[None, :], res0, **transform)[0]
        new = cell.to_image_um(np.asarray(s["new_um"], float)[None, :], res0, **transform)[0]
        chain, k = [], index_of[int(s["id"])]
        for _ in range(6):                                          # the stem's next nodes (up to six)
            kids = children.get(int(ids[k]), [])
            if not kids:
                break
            k = kids[0]
            chain.append(xyz[k])
        nxt = np.array(chain).reshape(-1, 3)
        stems_top.append(dict(old=old[:2], new=new[:2], next=nxt[:, :2].T, flag=s["flag"]))
        uvw = soma_morphology.fit_frame(np.vstack([old, new, nxt]), th[:8])
        stems_side.append(dict(old=uvw[0, [0, 2]], new=uvw[1, [0, 2]], next=uvw[2:, [0, 2]].T, flag=s["flag"]))
    c_uvw = soma_morphology.fit_frame(np.asarray(centre, float)[None, :], th[:8])[0]
    top = dict(allen=_circle(centre[0], centre[1], radius), silhouette=np.vstack(soma_morphology.ellipse_outline(th[:8])),
               sphere=_circle(th[0], th[1], r_eq), stems=stems_top)
    side = dict(allen=_circle(c_uvw[0], c_uvw[2], radius), silhouette=np.vstack(soma_morphology.side_outline(th[:8])),
                sphere=_circle(0.0, 0.0, r_eq), stems=stems_side)
    return top, side


if __name__ == "__main__":
    sys.exit(main())
