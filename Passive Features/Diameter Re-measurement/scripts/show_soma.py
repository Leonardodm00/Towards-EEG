#!/usr/bin/env python3
"""The soma's view in the image stack -- SPEC Block 44 (D-208).

    python3 scripts/show_soma.py --specimen 529878215 --cache-dir DIR [--shift-x X --shift-y Y --flip-h H --z0 Z]
                                 [--config-json FILE] [--swc PATH] [--out DIR] [--synthetic]

Places the traced soma (the SWC's type-1 nodes) in the image frame (the transform of --shift-x/--shift-y/--flip-h/
--z0, as clean_cell.py), fetches the square crop of soma_view.half_um about it from the planes within (its radius +
soma_view.z_margin_um) of its depth, computes the focus measure of each plane over the disk of (radius +
soma_view.focus_margin_um) about it (analysis.soma_view), and draws the sharpest plane with the focus curve and a
montage of the planes about it (plotting.soma_figures). Writes into --out (default ./soma_<S>): soma_<S>_focus.png,
soma_<S>_planes.png, soma_<S>.json (geometry, planes, focus values, the sharpest plane), soma_<S>_block.npz (the
crop: block, ks, valid, left, top, res0_um). Pure ASCII, LF.
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
    F = soma_view.focus_measure(block, valid, left, top, res0, centre[:2], radius + q.focus_margin_um, q.focus_sigma_px)
    j = soma_view.sharpest(F)
    log("%d of %d planes recorded; sharpest plane k = %d, %+.2f um from the traced soma's depth"
        % (int(np.sum(valid)), len(ks), ks[j], ks[j] * dz - centre[2]))
    frame_ns = _types.SimpleNamespace(left=int(left), top=int(top), res_um_px=res0, downsample=0)
    ext = image_extent_um(frame_ns, block.shape)
    title = "specimen %s: the soma (traced radius %.2f um) -- %s" % (a.specimen, radius, label)
    fig = SF.figure_soma_focus(block, ks, ext, centre, radius, dz, F, j, roots_xy=roots, title=title)
    SF.save(fig, os.path.join(out, "soma_%s_focus.png" % a.specimen), DPI)
    step, nh = int(q.montage_step_planes), int(q.montage_half_n)
    idx = [i for i in range(j - nh * step, j + nh * step + 1, step) if 0 <= i < len(ks) and valid[i]]
    fig2 = SF.figure_soma_planes(block, ks, ext, centre, radius, dz, idx, j, roots_xy=roots, title=title)
    SF.save(fig2, os.path.join(out, "soma_%s_planes.png" % a.specimen), DPI)
    rec = dict(specimen=str(a.specimen), source=label, transform=transform, centre_um=[float(x) for x in centre],
               radius_um=float(radius), crop=dict(left=left, top=top, width=width, height=height, res0_um=res0),
               dz_um=dz, ks=[int(k) for k in ks], valid=[bool(v) for v in valid],
               focus=[None if not np.isfinite(f) else float(f) for f in F], k_sharpest=int(ks[j]),
               z_sharpest_minus_soma_um=float(ks[j] * dz - centre[2]), montage_ks=[int(ks[i]) for i in idx],
               primary_dendrite_roots_um=[[float(x), float(y)] for x, y in roots], soma_view=dict(vars(q)) if hasattr(q, "__dict__") else None)
    with open(os.path.join(out, "soma_%s.json" % a.specimen), "w") as f:
        json.dump(rec, f, indent=1)
    np.savez_compressed(os.path.join(out, "soma_%s_block.npz" % a.specimen), block=block, ks=ks, valid=valid,
                        left=left, top=top, res0_um=res0)
    log("wrote %s/soma_%s_focus.png, _planes.png, .json, _block.npz" % (out, a.specimen))
    return 0


if __name__ == "__main__":
    sys.exit(main())
