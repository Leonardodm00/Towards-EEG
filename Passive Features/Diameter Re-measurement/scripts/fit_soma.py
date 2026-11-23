"""Fit the soma of a cell's image stack: the entry point of SPEC Blocks 45-48 (notebook Cell 25; D-215, D-216, D-218,
D-219, D-220, D-222).

The traced soma (the SWC's type-1 node in the image frame, ALIGN's transform; Block 44) gives the start: its centre and
radius, and its depth from --soma-k (the plane judged in focus by eye in Cell 24) or else the traced depth. The crop is
the fit's disk (soma_fit.region_radius_um) plus soma_fit.crop_margin_um about the traced centre, on the planes within
soma_fit.plane_half_um of the start depth (Block 44's window rule). Every traced process in the crop is masked at its
traced radius plus soma_fit.mask_widen_um (Block 44's segments; those from the soma node to the stems' first nodes lie
inside the soma and are not masked). The ellipsoid is fitted by Block 46 with Block 45's renderer as the forward model
(the kernel of --config-json, whose continuation must be "linear", D-220.2), the kernel's far slope fitted or held
(soma_fit.far_slope or --far-slope; D-218). Written into --out (default ./soma_fit_<S>):

  soma_fit_<S>.json              the fit (theta, SEs, covariance, the planes' scales, chi2, status), the membrane area,
                                 volume and equivalent radius, the start, the crop, the settings, Allen's soma and the
                                 stems' moves
  soma_fit_<S>_planes.png        data, model and residual on the fitted planes nearest soma_fit.figure_planes_um from
                                 the fitted centre (Block 48)
  soma_fit_<S>_summary.png       the parameters, the derived quantities beside Allen's, the scales and residuals by depth
  soma_fit_<S>_morphology.png    the soma and the stems' first nodes before and after, top and side views
  reconstruction_soma_<S>.swc    the corrected SWC (Block 47): the soma as the sphere of equal area at the fitted centre,
                                 each stem's first node on the fitted surface (D-216.3, D-222.5)

--synthetic: the smoke tests' synthetic cell's trace (geometry, alignment, mask) over a known ellipsoid soma rendered
on a uniform background with noise, so the dry run is also a recovery check (no network). Pure ASCII, LF.
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
from allen_diameter.analysis import cell, soma_fit, soma_morphology, soma_view   # noqa: E402
from allen_diameter.loading import swc_io                                         # noqa: E402
from allen_diameter.model.soma_render import SomaRenderer                         # noqa: E402

DPI = 130
# The dry run's soma (test data, not a parameter): offsets of the truth from the traced soma, and its shape relative to
# the traced radius r: centre (+0.6, -0.4, +0.9) um, a = 1.10 r, b = 0.85 r, c = 0.75 r, phi 0.6 rad, mu 0.15 /um;
# background 110 gl, noise SD 3 gl, rounded (seed 529).
SYN_OFFSET_UM = (0.6, -0.4, 0.9)
SYN_AXES_REL = (1.10, 0.85, 0.75)
SYN_PHI_RAD, SYN_MU_PER_UM, SYN_B_GL, SYN_NOISE_GL, SYN_SEED = 0.6, 0.15, 110.0, 3.0, 529


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
                    help="fit the kernel's far slope (D-218.1) or hold it at --slope0 (D-218.2); default soma_fit.far_slope")
    ap.add_argument("--slope0", type=float, default=None,
                    help="the far slope's start, or its held value (um/um); default renderer.kernel_continuation_slope")
    a = ap.parse_args(argv)
    import clean_cell
    from build_table import load_config
    from allen_diameter.config import default_config
    from allen_diameter.plotting import soma_fit_figures as FF
    from allen_diameter.plotting.figures import image_extent_um
    from allen_diameter.plotting.soma_figures import save
    cfg = load_config(a.config_json) if a.config_json else default_config()
    rcfg, q = cfg.renderer, cfg.soma_fit
    if a.far_slope is not None:
        q = dataclasses.replace(q, far_slope=a.far_slope)
    if rcfg.kernel_continuation != "linear":                       # D-220.2: gamma is the linear continuation's slope
        ap.error("the soma fit needs renderer.kernel_continuation 'linear' (D-218, D-220.2), got %r"
                 % rcfg.kernel_continuation)
    slope0 = float(rcfg.kernel_continuation_slope if a.slope0 is None else a.slope0)
    if not (math.isfinite(slope0) and slope0 >= 0):
        ap.error("--slope0 must be finite and >= 0")
    res0, dz = float(cfg.acquisition.res0_um), float(cfg.acquisition.dz_um)
    light = int(cfg.acquisition.light_direction)
    out = a.out or os.path.join(os.getcwd(), "soma_fit_%s" % a.specimen)
    os.makedirs(out, exist_ok=True)
    log = lambda msg: print("[fit_soma] %s" % msg, flush=True)      # noqa: E731
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

    # the crop: the fit's disk plus a margin, the planes within plane_half_um of the start depth
    half = float(q.region_radius_um) + float(q.crop_margin_um)
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
        # the soma alone on a uniform background, every plane recorded: the synthetic cell's image covers only part of
        # the crop and records planes -7..7, and composing it leaves a step at its edge that the fit would explain
        # (2026-11-23, the first dry runs); the trace's mask is applied all the same
        block = np.clip(np.rint(SYN_B_GL * tau + rng.normal(0.0, SYN_NOISE_GL, tau.shape)), 0, 255)
        valid = np.ones(ks.size, dtype=bool)
        log("synthetic soma rendered: centre (%.2f, %.2f, %.2f) um, a b c %.2f %.2f %.2f um, phi %.2f, mu %.2f /um"
            % tuple(truth[:8]))

    # the mask from Allen's trace, widened
    index_of = {int(i): k for k, i in enumerate(swc.ids)}
    parent_index = np.array([index_of.get(int(pp), -1) for pp in np.asarray(swc.parent)])
    box = (left * res0, (left + width) * res0, top * res0, (top + height) * res0)
    mask, n_seg = _mask(xyz, swc, parent_index, types, box, block.shape[1:], left, top, res0, float(q.mask_widen_um))
    log("mask: %d traced segments at their radii + %.1f um, %.1f %% of the crop" % (n_seg, q.mask_widen_um,
                                                                                   100.0 * mask.mean()))

    # the fit (Block 46 with Block 45's renderer)
    fit = soma_fit.fit_soma(block, ks, valid, left, top, res0, dz, start, float(radius), mask, make, q, slope0)
    log("fit %s in %d evaluations, %.0f s: %s" % (fit.status, fit.n_eval, fit.seconds, fit.message))
    th = np.asarray(fit.theta, dtype=float)
    ok = bool(np.all(np.isfinite(th[:8])))
    if ok:
        log("centre (%.2f, %.2f, %.2f) um; a %.2f, b %.2f, c %.2f um (+- %.3f, %.3f, %.3f); phi %.2f rad; mu %.3f /um; "
            "slope %.3f; S %.1f +- %.1f um^2 (Allen's 4 pi r^2 %.1f), r_eq %.3f um (Allen's r %.3f); chi2/dof %.3f"
            % (th[0], th[1], th[2], th[3], th[4], th[5], fit.se[3], fit.se[4], fit.se[5], th[6], th[7], th[8],
               fit.area_um2, fit.se_area_um2, 4 * math.pi * radius ** 2, fit.r_eq_um, radius,
               fit.chi2 / max(fit.dof, 1)))

    # the outputs: the corrected SWC, the figures, the record
    stems = []
    swc_out = None
    if ok:
        edit = soma_morphology.correct_soma(swc, th[:8], float(fit.r_eq_um), res0, **transform)
        swc_out = os.path.join(out, "reconstruction_soma_%s.swc" % a.specimen)
        swc_io.write_swc(swc, swc_out, xyz=edit.xyz, radius=edit.radius, parent=edit.parent, keep=edit.keep)
        stems = [dict(s) for s in edit.stems]
        log("corrected SWC: the soma node at the fitted centre with r_eq; %d stems: %s"
            % (len(stems), ", ".join("%s %s %.2f um" % (s["id"], s["flag"], s["moved_um"]) for s in stems)))
        frame_ns = _types.SimpleNamespace(left=int(left), top=int(top), res_um_px=res0, downsample=0)
        ext = image_extent_um(frame_ns, block.shape)
        region = soma_view.disk_mask(block.shape[1:], left, top, res0, centre[:2], float(q.region_radius_um)) & ~mask
        z_fit = fit.ks * dz
        model_all = fit.B_gl[:, None, None] * make(z_fit)(th[:8], float(th[8]))
        i_of_k = {int(k): i for i, k in enumerate(ks)}
        data_all = np.stack([block[i_of_k[int(k)]] for k in fit.ks])
        rms = np.sqrt(np.mean(((data_all - model_all)[:, region]) ** 2, axis=1))
        pick = sorted(set(int(np.argmin(np.abs(z_fit - th[2] - o))) for o in q.figure_planes_um))
        sil = np.vstack(soma_morphology.ellipse_outline(th[:8]))
        sections = []
        for i in pick:
            sx, sy = soma_morphology.ellipse_outline(th[:8], z_um=float(z_fit[i]))
            sections.append(np.vstack([sx, sy]) if sx.size else None)
        title = "specimen %s: the soma fit (%s; far slope %s)" % (a.specimen, label, q.far_slope)
        fig = FF.figure_soma_fit_planes(data_all[pick], model_all[pick], region, mask, fit.ks[pick],
                                        z_fit[pick] - th[2], ext, sil, sections, (centre[0], centre[1], radius),
                                        (th[0], th[1], fit.r_eq_um), title=title)
        save(fig, os.path.join(out, "soma_fit_%s_planes.png" % a.specimen), DPI)
        derived = dict(area_um2=fit.area_um2, se_area_um2=fit.se_area_um2, volume_um3=fit.volume_um3,
                       r_eq_um=fit.r_eq_um, se_volume_um3=_se_volume(th, fit.cov),
                       se_r_eq_um=fit.se_area_um2 / (8.0 * math.pi * fit.r_eq_um))   # delta method: dr/dS = 1/(8 pi r)
        allen = dict(radius_um=float(radius), area_um2=4 * math.pi * float(radius) ** 2)
        fig = FF.figure_soma_fit_summary(soma_fit.PARAMS, th, fit.se, fit.fitted, derived, allen, z_fit, fit.B_gl,
                                         rms, title=title)
        save(fig, os.path.join(out, "soma_fit_%s_summary.png" % a.specimen), DPI)
        top_v, side_v = _morphology_views(swc, edit, xyz, th, centre, radius, fit.r_eq_um, res0, transform)
        fig = FF.figure_soma_morphology(top_v, side_v, title=title)
        save(fig, os.path.join(out, "soma_fit_%s_morphology.png" % a.specimen), DPI)

    rec = dict(specimen=str(a.specimen), source=label, transform=transform,
               start=dict(centre_um=list(start), radius_um=float(radius), source=start_src),
               traced=dict(centre_um=[float(x) for x in centre], radius_um=float(radius),
                           area_um2=4 * math.pi * float(radius) ** 2),
               crop=dict(left=int(left), top=int(top), width=int(width), height=int(height), res0_um=res0,
                         k_lo=int(k_lo), k_hi=int(k_hi), n_mask_segments=n_seg, mask_fraction=float(mask.mean())),
               fit=dict(params=list(soma_fit.PARAMS), theta=th, se=fit.se, cov=fit.cov, fitted=list(fit.fitted),
                        ks=fit.ks, B_gl=fit.B_gl, chi2=fit.chi2, dof=fit.dof, n_samples=fit.n_samples,
                        status=fit.status, at_bound=list(fit.at_bound), n_eval=fit.n_eval, seconds=fit.seconds,
                        message=fit.message),
               derived=dict(area_um2=fit.area_um2, se_area_um2=fit.se_area_um2, volume_um3=fit.volume_um3,
                            se_volume_um3=_se_volume(th, fit.cov) if ok else None, r_eq_um=fit.r_eq_um,
                            se_r_eq_um=(fit.se_area_um2 / (8.0 * math.pi * fit.r_eq_um)) if ok else None),
               stems=stems, swc_out=swc_out, slope0=slope0, settings=dataclasses.asdict(q),
               kernel=dict(sigma_r0_um=rcfg.sigma_r0_um, table_delta_um=list(rcfg.kernel_table_delta_um),
                           table_sigma_um=list(rcfg.kernel_table_sigma_um), continuation=rcfg.kernel_continuation,
                           continuation_slope=rcfg.kernel_continuation_slope),
               synthetic_truth=None if truth is None else truth, seconds=time.time() - t0)
    with open(os.path.join(out, "soma_fit_%s.json" % a.specimen), "w") as f:
        json.dump(_json_safe(rec), f, indent=1, allow_nan=False)
    if truth is not None and ok:
        err = th - truth
        log("synthetic truth recovered: |error| centre %.3f %.3f %.3f um, axes %.3f %.3f %.3f um, phi %.3f rad, mu %.4f"
            % tuple(np.abs(err[:8])))
    log("wrote %s/soma_fit_%s.json%s" % (out, a.specimen, ", _planes.png, _summary.png, _morphology.png, "
                                         "reconstruction_soma_%s.swc" % a.specimen if ok else " (the fit failed)"))
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
