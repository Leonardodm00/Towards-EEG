#!/usr/bin/env python3
"""The soma's view in the image stack -- SPEC Block 44 (D-210, D-211).

    python3 scripts/show_soma.py --specimen 529878215 --cache-dir DIR [--shift-x X --shift-y Y --flip-h H --z0 Z]
                                 [--config-json FILE] [--swc PATH] [--out DIR] [--synthetic]
                                 [--soma-k K] [--soma-k-range LO HI] [--montage-k LO HI STEP]

Places the traced soma (the SWC's type-1 nodes) in the image frame (the transform of --shift-x/--shift-y/--flip-h/
--z0, as clean_cell.py) and fetches the square crop of soma_view.half_um about it from the planes within (its radius
+ soma_view.z_margin_um) of its depth: the window. The soma's plane is chosen by eye (D-211): --soma-k K is the plane
where the soma is judged in focus (without it, the plane nearest the traced depth is shown), --soma-k-range LO HI the
planes where the soma is seen; nothing is picked automatically. Draws the shown plane beside the fine-detail energy
of each plane over the traced disk (a diagnostic, analysis.soma_view.detail_energy), and a montage: every
soma_view.montage_step_planes-th recorded plane of the window aligned on the shown plane, or --montage-k LO HI STEP.
A plane outside the window or not recorded is refused. Writes into --out (default ./soma_<S>): soma_<S>_focus.png,
soma_<S>_planes.png, soma_<S>.json (geometry, window, energies, k_traced, k_by_eye, k_range_by_eye, montage_ks),
soma_<S>_block.npz (the crop: block, ks, valid, left, top, res0_um). Pure ASCII, LF.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types as _types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from allen_diameter.analysis import cell, soma_view            # noqa: E402

DPI = 130


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--shift-x", type=float, default=None)
    ap.add_argument("--shift-y", type=float, default=None)
    ap.add_argument("--flip-h", type=float, default=None)
    ap.add_argument("--z0", type=float, default=None)
    ap.add_argument("--config-json", default=None)
    ap.add_argument("--swc", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--synthetic", action="store_true", help="the smoke tests' synthetic cell (no network)")
    ap.add_argument("--soma-k", type=int, default=None, help="the plane where the soma is judged in focus (by eye)")
    ap.add_argument("--soma-k-range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="the first and last plane where the soma is seen (by eye)")
    ap.add_argument("--montage-k", type=int, nargs=3, default=None, metavar=("LO", "HI", "STEP"),
                    help="the montage's planes; default: the whole window, every soma_view.montage_step_planes-th")
    a = ap.parse_args(argv)
    import clean_cell
    from build_table import load_config
    from allen_diameter.config import default_config
    from allen_diameter.plotting import soma_figures as SF
    from allen_diameter.plotting.figures import image_extent_um
    cfg = load_config(a.config_json) if a.config_json else default_config()
    q = cfg.soma_view
    res0, dz = float(cfg.acquisition.res0_um), float(cfg.acquisition.dz_um)
    out = a.out or os.path.join(os.getcwd(), "soma_%s" % a.specimen)
    os.makedirs(out, exist_ok=True)
    log = lambda msg: print("[show_soma] %s" % msg, flush=True)      # noqa: E731
    swc, provider, label = clean_cell.load_cell(a, cfg, log, need_images=True)
    transform = clean_cell.transform_of(a)
    xyz = cell.to_image_um(swc.xyz, res0, **transform)
    types = np.asarray(swc.types)
    centre, radius = soma_view.soma_geometry(xyz, swc.radius, types)
    soma_ids = set(int(i) for i, t in zip(swc.ids, types) if int(t) == 1)
    parents = np.asarray(swc.parent)
    roots = np.array([xyz[k, :2] for k in range(len(swc.ids)) if int(parents[k]) in soma_ids and int(types[k]) != 1])
    left, top, width, height = soma_view.crop_box(centre[:2], q.half_um, res0)
    k_lo, k_hi = soma_view.plane_range(centre[2], radius + q.z_margin_um, dz)
    log("%s cell: traced soma at (%.2f, %.2f, %.2f) um (image frame), radius %.2f um; crop %d x %d px at (%d, %d); "
        "planes %d..%d (%d)" % (label, centre[0], centre[1], centre[2], radius, width, height, left, top, k_lo, k_hi,
                                k_hi - k_lo + 1))
    block, ks, valid, frame = provider(left, top, width, height, k_lo, k_hi)
    ks = np.asarray(ks).astype(int)
    valid = np.asarray(valid, dtype=bool)
    E = soma_view.detail_energy(block, valid, left, top, res0, centre[:2], radius + q.focus_margin_um, q.focus_sigma_px)
    try:
        j_traced = soma_view.reference_plane(ks, valid, centre[2], dz)[0]
        j_ref, ref_label = soma_view.reference_plane(ks, valid, centre[2], dz, a.soma_k)
        k_range = None
        if a.soma_k_range is not None:
            r_lo, r_hi = (int(v) for v in a.soma_k_range)
            if not (ks[0] <= r_lo <= r_hi <= ks[-1]):
                raise ValueError("--soma-k-range %d %d must lie in the window %d..%d with LO <= HI"
                                 % (r_lo, r_hi, ks[0], ks[-1]))
            k_range = (r_lo, r_hi)
        if a.montage_k is not None:
            m_lo, m_hi, m_step = (int(v) for v in a.montage_k)
            idx = soma_view.montage_planes(ks, valid, ks[j_ref], m_step, m_lo, m_hi)
        else:
            m_step = int(q.montage_step_planes)
            idx = soma_view.montage_planes(ks, valid, ks[j_ref], m_step)
        if not idx:
            raise ValueError("the montage holds no recorded plane (--montage-k %s)" % (a.montage_k,))
    except ValueError as exc:
        ap.error(str(exc))
    k_traced = int(ks[j_traced])
    log("%d of %d planes recorded; the traced depth is plane %d; shown: plane %d (%s); montage: %d planes, every %d"
        % (int(np.sum(valid)), len(ks), k_traced, ks[j_ref], ref_label, len(idx), m_step))
    frame_ns = _types.SimpleNamespace(left=int(left), top=int(top), res_um_px=res0, downsample=0)
    ext = image_extent_um(frame_ns, block.shape)
    title = "specimen %s: the soma (traced radius %.2f um) -- %s" % (a.specimen, radius, label)
    fig = SF.figure_soma_focus(block, ks, ext, centre, radius, dz, E, j_ref, ref_label, k_traced, k_range=k_range,
                               roots_xy=roots, title=title)
    SF.save(fig, os.path.join(out, "soma_%s_focus.png" % a.specimen), DPI)
    fig2 = SF.figure_soma_planes(block, ks, ext, centre, radius, dz, idx, j_ref, ref_label, k_traced, k_range=k_range,
                                 roots_xy=roots, title=title, ncol=int(q.montage_cols))
    SF.save(fig2, os.path.join(out, "soma_%s_planes.png" % a.specimen), DPI)
    rec = dict(specimen=str(a.specimen), source=label, transform=transform, centre_um=[float(x) for x in centre],
               radius_um=float(radius), crop=dict(left=left, top=top, width=width, height=height, res0_um=res0),
               dz_um=dz, ks=[int(k) for k in ks], valid=[bool(v) for v in valid],
               detail_energy=[None if not np.isfinite(e) else float(e) for e in E], k_traced=k_traced,
               k_by_eye=None if a.soma_k is None else int(a.soma_k),
               k_range_by_eye=None if k_range is None else [int(k_range[0]), int(k_range[1])],
               k_shown=int(ks[j_ref]), montage_ks=[int(ks[i]) for i in idx],
               primary_dendrite_roots_um=[[float(x), float(y)] for x, y in roots],
               soma_view=dict(vars(q)) if hasattr(q, "__dict__") else None)
    with open(os.path.join(out, "soma_%s.json" % a.specimen), "w") as f:
        json.dump(rec, f, indent=1)
    np.savez_compressed(os.path.join(out, "soma_%s_block.npz" % a.specimen), block=block, ks=ks, valid=valid,
                        left=left, top=top, res0_um=res0)
    log("wrote %s/soma_%s_focus.png, _planes.png, .json, _block.npz" % (out, a.specimen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
