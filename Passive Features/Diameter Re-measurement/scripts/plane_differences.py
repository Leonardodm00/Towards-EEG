#!/usr/bin/env python3
"""Consecutive-plane differences around measured nodes (the user's proposal of
2026-10-08): planes k_swc - H .. k_swc + H of the node's block, the positive
part of I_{k+1} - I_k between each pair and its mean over the pixels -- Block
11 in specs/SPEC.md. A diagnostic; nothing here changes the measurement.

Each node is located as the pilot measures it (survey.node_planes: the same
stretch and the same block request, so with the pilot's image cache its planes
come from disk); the planes outside the pilot's range are fetched, one crop
each. Pixels counted: the square +-half_um about the SWC node (default the
measurement's block half-width, 5 um), or with --band-um only those within
that distance of the traced stretch. Writes, in --out-dir:

    planediff_<id>.png         the planes; the positive part of each difference;
                               S+ (the proposal's measure) and S- (the same
                               taken from the other end of the stack) at the
                               pair midpoints, with the dip of S+ between its
                               two largest maxima; frames and lines at k*
                               (gradient energy), the dip-depth plane and the
                               SWC plane (plotting.figures.plane_difference_figure)
    planediff_<specimen>.json  per node: planes, valid, S+, S-, the dip pair,
                               k*, the dip-depth plane, the SWC plane, pixels

Example (Colab, after Cells 0 and 1)
    python scripts/plane_differences.py --specimen 529878215 --nodes 2,3 \
        --cache-dir /content/drive/MyDrive/allen_cache --out-dir /content/drive/MyDrive/diameters/pilot/planediff

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
for p in (SRC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from allen_diameter.analysis import background  # noqa: E402
from allen_diameter.analysis import focus  # noqa: E402
from allen_diameter.analysis import survey  # noqa: E402
from build_table import load_config  # noqa: E402

# (colour, linestyle, short tag on the panel, legend label); the order is the order of precedence of the frame
_MARKS = (("k_star", "#ff1744", "-", "k*", "k* (gradient energy, D-030)"),
          ("k_star_depth", "#7a7a7a", "--", "dip", "plane of the dip depth"),
          ("k_swc", "#1a1a19", ":", "SWC", "the SWC node's plane"))


def node_stack(swc, provider, cfg, node_id, transform=None, planes_half=6, half_um=None, band_um=0.0):
    """The planes k_swc - planes_half .. k_swc + planes_half of node node_id's
    block, cropped to the square +-half_um about the node; the pixel mask
    (all True, or within band_um of the traced stretch when band_um > 0).
    ValueError for a node survey.node_planes refuses."""
    from allen_image_io import CropFrame
    from allen_diameter.plotting.figures import image_extent_um
    pl = survey.node_planes(swc, provider, cfg, node_id, transform)
    frame, (H, W) = pl["frame"], pl["block"].shape[1:]
    k_swc = int(pl["k_swc"])
    block, ks, valid, _ = provider(frame.left, frame.top, W, H, k_swc - int(planes_half), k_swc + int(planes_half))
    p = float(frame.res_um_px)
    half = float(cfg.measure.block_half_um if half_um is None else half_um)
    o = pl["branch"].xyz_um[pl["index"], :2]
    c0 = max(0, int(math.ceil((o[0] - half) / p - frame.left)))
    c1 = min(W, int(math.floor((o[0] + half) / p - frame.left)) + 1)
    r0 = max(0, int(math.ceil((o[1] - half) / p - frame.top)))
    r1 = min(H, int(math.floor((o[1] + half) / p - frame.top)) + 1)
    if c1 - c0 < 2 or r1 - r0 < 2:
        raise ValueError("node %d: the square +-%g um about the node misses its block" % (node_id, half))
    stack = np.asarray(block[:, r0:r1, c0:c1], dtype=float)
    if band_um and band_um > 0:
        xyz = pl["branch"].xyz_um
        mask = background.mask_near_branch(stack.shape[1:], frame.left + c0, frame.top + r0, p, xyz,
                                           np.zeros(xyz.shape[0]), float(band_um))
    else:
        mask = np.ones(stack.shape[1:], dtype=bool)
    sub_frame = CropFrame(int(frame.left + c0), int(frame.top + r0), 0, p)
    return dict(stack=stack, ks=np.asarray(ks), valid=np.asarray(valid, dtype=bool), mask=mask,
                extent=image_extent_um(sub_frame, stack.shape[1:]), k_swc=k_swc, result=pl["result"])


def _marks(res, k_swc):
    """Frames (one per plane; the first mark sets the colour, the tags are joined) and curve lines; k* only
    when the measurement found a sharpest plane (finite z_sub)."""
    at = dict(k_star=int(res.k_star) if math.isfinite(res.z_sub_um) else None, k_star_depth=int(res.k_star_depth),
              k_swc=int(k_swc))
    frames, lines = {}, []
    for key, colour, ls, tag, label in _MARKS:
        k = at[key]
        if k is None:
            continue
        lines.append((k, colour, ls, label))
        frames[k] = (frames[k][0], frames[k][1], frames[k][2] + " " + tag) if k in frames else (colour, ls, tag)
    return frames, lines


def run(swc, provider, cfg, node_ids, out_dir, specimen, transform=None, planes_half=6, half_um=None, band_um=0.0,
        dpi=110, log=print):
    """One figure per node id; returns the per-node records (also written to planediff_<specimen>.json)."""
    from allen_diameter.plotting import figures as fg
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for nid in node_ids:
        try:
            st = node_stack(swc, provider, cfg, nid, transform, planes_half, half_um, band_um)
        except ValueError as e:
            log("[planediff] node %d skipped: %s" % (nid, e))
            records.append(dict(node_id=int(nid), skipped=str(e)))
            continue
        res = focus.plane_differences(st["stack"], st["valid"], st["mask"])
        dip = focus.difference_dip(res["pos"])
        r = st["result"]
        frames, lines = _marks(r, st["k_swc"])
        ks = st["ks"]
        where = "the whole square" if not band_um else "pixels within %.2g um of the stretch" % band_um
        label = ("node %d: planes %d..%d, %s (%d pixels); k* %d, dip-depth plane %d, SWC plane %d"
                 % (nid, ks[0], ks[-1], where, int(st["mask"].sum()), r.k_star, r.k_star_depth, st["k_swc"]))
        fig = fg.plane_difference_figure([dict(label=label, stack=st["stack"], ks=ks, valid=st["valid"], res=res,
                                               dip=dip, frames=frames, lines=lines, extent=st["extent"])],
                                         "Consecutive-plane differences, specimen %s" % specimen)
        path = os.path.join(out_dir, "planediff_%d.png" % nid)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        rec = dict(node_id=int(nid), png=path, ks=[int(k) for k in ks], valid=[bool(x) for x in st["valid"]],
                   pos=[float(x) for x in res["pos"]], neg=[float(x) for x in res["neg"]],
                   dip_pair=None if dip is None else [int(ks[dip]), int(ks[dip + 1])],
                   k_star=int(r.k_star), k_star_depth=int(r.k_star_depth), k_swc=int(st["k_swc"]),
                   z_sub_um=float(r.z_sub_um), n_pixels=int(st["mask"].sum()), n_missing=int((~st["valid"]).sum()),
                   band_um=float(band_um), half_um=float(cfg.measure.block_half_um if half_um is None else half_um))
        log("[planediff] node %d: planes %d..%d (%d missing), %d pixels; dip of S+ at %s; k* %d, dip-depth plane %d, "
            "SWC plane %d" % (nid, ks[0], ks[-1], rec["n_missing"], rec["n_pixels"],
                              "none" if dip is None else "%d->%d" % tuple(rec["dip_pair"]), rec["k_star"],
                              rec["k_star_depth"], rec["k_swc"]))
        records.append(rec)
    with open(os.path.join(out_dir, "planediff_%s.json" % specimen), "w") as f:
        json.dump(records, f, indent=1, sort_keys=True, allow_nan=True)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--nodes", required=True, help="comma-separated node ids")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--planes-half", type=int, default=6, help="planes on each side of the SWC plane")
    ap.add_argument("--half-um", type=float, default=None, help="half-width of the square of pixels, um "
                                                                "(default: the measurement's block_half_um)")
    ap.add_argument("--band-um", type=float, default=0.0, help="> 0: count only pixels within this distance "
                                                               "of the traced stretch")
    ap.add_argument("--swc", default="")
    ap.add_argument("--cache-dir", default="allen_cache")
    ap.add_argument("--shift-x", type=float, default=0.0, help="full-res px (global alignment)")
    ap.add_argument("--shift-y", type=float, default=0.0)
    ap.add_argument("--flip-h", type=float, default=None)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--config-json", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config_json)
    node_ids = [int(x) for x in a.nodes.split(",") if x.strip()]
    if not node_ids:
        print("[planediff] no node given: nothing to do", flush=True)
        return 0
    import allen_image_io as aio
    import run_cell
    from allen_diameter.loading import swc_io
    swc = swc_io.read_swc(a.swc or aio.fetch_swc(int(a.specimen), a.cache_dir))
    fetcher = aio.HttpFetcher(cache_dir=a.cache_dir)
    planes = aio.plane_table(aio.list_images(int(a.specimen)))
    provider = run_cell.real_provider(fetcher, planes, cfg.acquisition.res0_um)
    t0 = time.perf_counter()
    recs = run(swc, provider, cfg, node_ids, a.out_dir, a.specimen,
               dict(shift_full_px=(a.shift_x, a.shift_y), flip_y_full_h=a.flip_h, z0_um=a.z0),
               a.planes_half, a.half_um, a.band_um, log=lambda m: print(m, flush=True))
    n_ok = sum(1 for r in recs if "png" in r)
    print("[planediff] wrote %d figures (%d skipped) in %s; crops: %d from the cache, %d downloaded (%.1f MB); %.1f min"
          % (n_ok, len(recs) - n_ok, a.out_dir, fetcher.n_cache_hits, fetcher.n_requests,
             fetcher.bytes_downloaded / 1e6, (time.perf_counter() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
