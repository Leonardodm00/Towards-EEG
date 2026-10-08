#!/usr/bin/env python3
"""Plane-to-plane evaluations around measured nodes, the user's proposals of
2026-10-08 -- Block 11 in specs/SPEC.md. Diagnostics; nothing here changes the
measurement.

--evaluation profile (default; the proposal of 16:26): in every plane
k_swc - H .. k_swc + H, the profile I_k(v) along the node's measuring line (the
line the focus scores use: through the SWC node, across the node's fitted
heading, |v| <= --profile-half-um, bilinear), its area A_k = int I_k dv, and
dA = A_{k+1} - A_k. Beside it, the same areas with each plane divided by its
own background B_k and multiplied by the planes' mean background, so that a
plane-wide change of brightness does not enter: B_k is the interquartile mean
(25th-75th percentile) of the block's pixels farther than Allen's radius +
--bg-margin-um from every traced dendrite segment in the block and from the
soma; the margin shrinks (3, 2, 1, 0.5 um) until 5 % of the block's pixels
are left.
The plane of the smallest area is framed (blue: as measured; violet:
normalised).

--evaluation image (the proposal of 15:45): the positive part of
I_{k+1} - I_k over the pixels of the square, its mean S+ and the mean S- of
the negative part (focus.plane_differences), with the dip of S+.

Each node is located as the pilot measures it (survey.node_planes: the same
stretch and block request, so with the pilot's image cache its planes come
from disk); the planes outside the pilot's range are fetched, one crop each.
The square: +-half_um about the SWC node (default block_half_um, 5 um).
Writes, in --out-dir: planediff_<id>.png and planediff_<specimen>.json.

Example (Colab, after Cells 0 and 1)
    python scripts/plane_differences.py --specimen 529878215 --nodes 2,3 \
        --cache-dir /content/drive/MyDrive/allen_cache --out-dir /content/drive/MyDrive/diameters/pilot/planediff

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
for p in (SRC, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from allen_diameter.analysis import background  # noqa: E402
from allen_diameter.analysis import cell  # noqa: E402
from allen_diameter.analysis import focus  # noqa: E402
from allen_diameter.analysis import profiles  # noqa: E402
from allen_diameter.analysis import survey  # noqa: E402
from build_table import load_config  # noqa: E402

# (key, colour, linestyle, short tag on the panel, legend label); the order is the order of precedence of a frame
_MARKS_PIPELINE = (("k_star", "#ff1744", "-", "k*", "k* (gradient energy, D-030)"),
                   ("k_star_depth", "#7a7a7a", "--", "dip", "plane of the dip depth"),
                   ("k_swc", "#1a1a19", ":", "SWC", "the SWC node's plane"))
_MARKS_AREA = (("k_min_area", "#2a78d6", "-", "minA", "smallest area under the profile, as measured"),
               ("k_min_area_norm", "#4a3aa7", "--", "minA/B", "smallest area, background-normalised"))


def node_stack(swc, provider, cfg, node_id, transform=None, planes_half=6, half_um=None, band_um=0.0):
    """The planes k_swc - planes_half .. k_swc + planes_half of node node_id's
    block, cropped to the square +-half_um about the node, with what the
    evaluations need: the pixel mask (all True, or within band_um of the traced
    stretch when band_um > 0), the crop's frame and extent, the node's xy and
    heading, the stretch's traced path and radii. ValueError for a node
    survey.node_planes refuses."""
    from allen_image_io import CropFrame
    from allen_diameter.plotting.figures import image_extent_um
    pl = survey.node_planes(swc, provider, cfg, node_id, transform)
    frame, (H, W) = pl["frame"], pl["block"].shape[1:]
    k_swc = int(pl["k_swc"])
    block, ks, valid, _ = provider(frame.left, frame.top, W, H, k_swc - int(planes_half), k_swc + int(planes_half))
    p = float(frame.res_um_px)
    half = float(cfg.measure.block_half_um if half_um is None else half_um)
    o = np.asarray(pl["branch"].xyz_um[pl["index"], :2], dtype=float)
    c0 = max(0, int(math.ceil((o[0] - half) / p - frame.left)))
    c1 = min(W, int(math.floor((o[0] + half) / p - frame.left)) + 1)
    r0 = max(0, int(math.ceil((o[1] - half) / p - frame.top)))
    r1 = min(H, int(math.floor((o[1] + half) / p - frame.top)) + 1)
    if c1 - c0 < 2 or r1 - r0 < 2:
        raise ValueError("node %d: the square +-%g um about the node misses its block" % (node_id, half))
    stack = np.asarray(block[:, r0:r1, c0:c1], dtype=float)
    sub = CropFrame(int(frame.left + c0), int(frame.top + r0), 0, p)
    xyz, radius = pl["branch"].xyz_um, np.asarray(pl["branch"].radius_um, dtype=float)
    som = np.asarray(swc.types) == 1
    soma = np.column_stack([cell.to_image_um(swc.xyz[som], cfg.acquisition.res0_um, **(transform or {}))[:, :2],
                            np.asarray(swc.radius, dtype=float)[som]]) if som.any() else np.empty((0, 3))
    if band_um and band_um > 0:
        mask = background.mask_near_branch(stack.shape[1:], sub.left, sub.top, p, xyz, np.zeros(xyz.shape[0]),
                                           float(band_um))
    else:
        mask = np.ones(stack.shape[1:], dtype=bool)
    return dict(stack=stack, ks=np.asarray(ks), valid=np.asarray(valid, dtype=bool), mask=mask, frame=sub,
                extent=image_extent_um(sub, stack.shape[1:]), k_swc=k_swc, result=pl["result"], o=o,
                theta=float(pl["result"].theta_rad), xyz=xyz, radius=radius, block=np.asarray(block, dtype=float),
                block_frame=frame, segments=np.asarray(pl["segments"], dtype=float), soma=soma)


def far_from_dendrites(shape, frame, segments, margin_um, discs=()):
    """(H, W) bool: True where a pixel centre is farther than max(r0, r1) +
    margin_um from every traced dendrite segment (the rows of
    survey.node_planes' segments: x0, y0, z0, x1, y1, z1, r0, r1, in_stretch)
    and farther than r + margin_um from every disc (x, y, r) (the soma)."""
    H, W = int(shape[0]), int(shape[1])
    p = float(frame.res_um_px)
    rows, cols = np.mgrid[0:H, 0:W]
    px, py = (frame.left + cols) * p, (frame.top + rows) * p
    near = np.zeros((H, W), dtype=bool)
    for s in np.atleast_2d(segments):
        if s.size < 8:
            continue
        near |= background.segment_distance(px, py, s[0:2], s[3:5]) <= max(s[6], s[7]) + margin_um
    for x, y, r in np.atleast_2d(np.asarray(discs, dtype=float)).reshape(-1, 3):
        near |= np.hypot(px - x, py - y) <= r + margin_um
    return ~near


def plane_background(block, valid, far, min_frac=0.05):
    """B_k (n,): the interquartile mean (25th-75th percentile) of each valid
    plane's pixels in `far`; NaN when fewer than min_frac of the pixels are far."""
    B = np.full(block.shape[0], np.nan)
    if far.mean() < min_frac:
        return B
    for k in range(block.shape[0]):
        if valid[k]:
            x = np.sort(block[k][far])
            lo, hi = int(np.floor(0.25 * x.size)), int(np.ceil(0.75 * x.size))
            B[k] = float(x[lo:max(hi, lo + 1)].mean())
    return B


def profile_evaluation(st, cfg, profile_half_um=None, bg_margin_um=4.0):
    """Profiles along the node's measuring line in every plane, their areas as
    measured and background-normalised, and each plane's background B_k."""
    m = cfg.measure
    h = float(m.profile_half_um if profile_half_um is None else profile_half_um)
    v = profiles.profile_offsets(dataclasses.replace(m, profile_half_um=h))
    th = st["theta"]
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    stack, valid, fr = st["stack"], st["valid"], st["frame"]
    prof = np.full((stack.shape[0], v.size), np.nan)
    for k in range(stack.shape[0]):
        if valid[k]:
            prof[k] = profiles.sample_profile(stack[k], fr, st["o"], y_hat, e_u, v)
    # the margin shrinks (4, 3, 2, 1, 0.5 um, never above the one asked) until 5 % of the block's pixels are far:
    # next to the soma a wide margin leaves no pixel
    margins = [float(bg_margin_um)] + [x for x in (3.0, 2.0, 1.0, 0.5) if x < float(bg_margin_um)]
    for margin in margins:
        far = far_from_dendrites(st["block"].shape[1:], st["block_frame"], st["segments"], margin, st["soma"])
        if far.mean() >= 0.05:
            break
    B = plane_background(st["block"], valid, far)
    area = focus.profile_areas(prof, v, valid)["area"]
    ok = valid & np.isfinite(B) & (B > 0)
    area_norm = np.full_like(area, np.nan)
    if ok.any():
        scale = np.where(ok, np.nanmean(B[ok]) / np.where(ok, B, 1.0), np.nan)
        area_norm = focus.profile_areas(prof * scale[:, None], v, ok)["area"]
    line = (st["o"] - h * y_hat, st["o"] + h * y_hat)
    return dict(v=v, prof=prof, area=area, area_norm=area_norm, background=B, line=line, half_um=h,
                bg_frac=float(far.mean()), bg_margin_used=margin if np.isfinite(B).any() else None)


def _argmin_plane(ks, A):
    A = np.asarray(A, dtype=float)
    return None if not np.isfinite(A).any() else int(ks[int(np.nanargmin(A))])


def _marks(at, specs):
    """Frames (one per plane; the first mark sets the colour, the tags are joined) and curve lines."""
    frames, lines = {}, []
    for key, colour, ls, tag, label in specs:
        k = at.get(key)
        if k is None:
            continue
        lines.append((k, colour, ls, label))
        frames[k] = (frames[k][0], frames[k][1], frames[k][2] + " " + tag) if k in frames else (colour, ls, tag)
    return frames, lines


def _pipeline_at(res, k_swc):
    return dict(k_star=int(res.k_star) if math.isfinite(res.z_sub_um) else None, k_star_depth=int(res.k_star_depth),
                k_swc=int(k_swc))


def run(swc, provider, cfg, node_ids, out_dir, specimen, transform=None, planes_half=6, half_um=None, band_um=0.0,
        evaluation="profile", profile_half_um=None, bg_margin_um=4.0, dpi=110, log=print):
    """One figure per node id; returns the per-node records (also written to planediff_<specimen>.json)."""
    from allen_diameter.plotting import figures as fg
    import matplotlib.pyplot as plt
    if evaluation not in ("profile", "image"):
        raise ValueError("evaluation must be 'profile' or 'image', got %r" % (evaluation,))
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for nid in node_ids:
        try:
            st = node_stack(swc, provider, cfg, nid, transform, planes_half, half_um, band_um)
        except ValueError as e:
            log("[planediff] node %d skipped: %s" % (nid, e))
            records.append(dict(node_id=int(nid), skipped=str(e)))
            continue
        r, ks = st["result"], st["ks"]
        at = _pipeline_at(r, st["k_swc"])
        rec = dict(node_id=int(nid), evaluation=evaluation, ks=[int(k) for k in ks],
                   valid=[bool(x) for x in st["valid"]], k_star=int(r.k_star), k_star_depth=int(r.k_star_depth),
                   k_swc=int(st["k_swc"]), z_sub_um=float(r.z_sub_um), n_missing=int((~st["valid"]).sum()),
                   half_um=float(cfg.measure.block_half_um if half_um is None else half_um))
        if evaluation == "profile":
            ev = profile_evaluation(st, cfg, profile_half_um, bg_margin_um)
            at.update(k_min_area=_argmin_plane(ks, ev["area"]), k_min_area_norm=_argmin_plane(ks, ev["area_norm"]))
            frames, lines = _marks(at, _MARKS_AREA + _MARKS_PIPELINE)
            na = lambda x: "n/a" if x is None else "%d" % x  # noqa: E731
            label = ("node %d: planes %d..%d; profile +-%.1f um across heading %.0f deg; smallest area at k %s "
                     "(normalised: k %s); k* %d, dip-depth plane %d, SWC plane %d"
                     % (nid, ks[0], ks[-1], ev["half_um"], math.degrees(st["theta"]), na(at["k_min_area"]),
                        na(at["k_min_area_norm"]), r.k_star, r.k_star_depth, st["k_swc"]))
            fig = fg.profile_area_figure([dict(label=label, stack=st["stack"], ks=ks, valid=st["valid"],
                                               extent=st["extent"], line=ev["line"], frames=frames, lines=lines,
                                               v=ev["v"], prof=ev["prof"], area=ev["area"], area_norm=ev["area_norm"],
                                               background=ev["background"], k_ref=st["k_swc"])],
                                         "Area under the profile along the measuring line, specimen %s" % specimen)
            B = ev["background"]
            rec.update(profile_half_um=ev["half_um"], theta_rad=st["theta"], area=[float(x) for x in ev["area"]],
                       area_norm=[float(x) for x in ev["area_norm"]], background=[float(x) for x in B],
                       bg_frac=ev["bg_frac"], bg_margin_um=float(bg_margin_um), bg_margin_used=ev["bg_margin_used"],
                       k_min_area=at["k_min_area"],
                       k_min_area_norm=at["k_min_area_norm"])
            fin = B[np.isfinite(B)]
            msg = ("smallest area at k %s, normalised k %s; background %s"
                   % (na(at["k_min_area"]), na(at["k_min_area_norm"]),
                      "%.1f..%.1f gl (margin %g um)" % (fin.min(), fin.max(), ev["bg_margin_used"]) if fin.size
                      else "n/a (no pixel far from the traced dendrites)"))
        else:
            res = focus.plane_differences(st["stack"], st["valid"], st["mask"])
            dip = focus.difference_dip(res["pos"])
            frames, lines = _marks(at, _MARKS_PIPELINE)
            where = "the whole square" if not band_um else "pixels within %.2g um of the stretch" % band_um
            label = ("node %d: planes %d..%d, %s (%d pixels); k* %d, dip-depth plane %d, SWC plane %d"
                     % (nid, ks[0], ks[-1], where, int(st["mask"].sum()), r.k_star, r.k_star_depth, st["k_swc"]))
            fig = fg.plane_difference_figure([dict(label=label, stack=st["stack"], ks=ks, valid=st["valid"], res=res,
                                                   dip=dip, frames=frames, lines=lines, extent=st["extent"])],
                                             "Consecutive-plane differences, specimen %s" % specimen)
            rec.update(pos=[float(x) for x in res["pos"]], neg=[float(x) for x in res["neg"]],
                       dip_pair=None if dip is None else [int(ks[dip]), int(ks[dip + 1])],
                       n_pixels=int(st["mask"].sum()), band_um=float(band_um))
            msg = "%d pixels; dip of S+ at %s" % (rec["n_pixels"], "none" if dip is None else
                                                 "%d->%d" % tuple(rec["dip_pair"]))
        path = os.path.join(out_dir, "planediff_%d.png" % nid)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        rec["png"] = path
        log("[planediff] node %d: planes %d..%d (%d missing); %s; k* %d, dip-depth plane %d, SWC plane %d"
            % (nid, ks[0], ks[-1], rec["n_missing"], msg, rec["k_star"], rec["k_star_depth"], rec["k_swc"]))
        records.append(rec)
    with open(os.path.join(out_dir, "planediff_%s.json" % specimen), "w") as f:
        json.dump(records, f, indent=1, sort_keys=True, allow_nan=True)
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--specimen", required=True)
    ap.add_argument("--nodes", required=True, help="comma-separated node ids")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--evaluation", choices=("profile", "image"), default="profile")
    ap.add_argument("--planes-half", type=int, default=6, help="planes on each side of the SWC plane")
    ap.add_argument("--profile-half-um", type=float, default=None,
                    help="profile evaluation: half-length of the measuring line, um (default: profile_half_um)")
    ap.add_argument("--bg-margin-um", type=float, default=4.0,
                    help="profile evaluation: B_k uses the block's pixels farther than Allen's radius + this from "
                         "every traced dendrite")
    ap.add_argument("--half-um", type=float, default=None, help="half-width of the square, um "
                                                                "(default: the measurement's block_half_um)")
    ap.add_argument("--band-um", type=float, default=0.0, help="image evaluation, > 0: count only the pixels "
                                                               "within this distance of the traced stretch")
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
               a.planes_half, a.half_um, a.band_um, a.evaluation, a.profile_half_um, a.bg_margin_um,
               log=lambda m: print(m, flush=True))
    n_ok = sum(1 for r in recs if "png" in r)
    print("[planediff] wrote %d figures (%d skipped) in %s; crops: %d from the cache, %d downloaded (%.1f MB); %.1f min"
          % (n_ok, len(recs) - n_ok, a.out_dir, fetcher.n_cache_hits, fetcher.n_requests,
             fetcher.bytes_downloaded / 1e6, (time.perf_counter() - t0) / 60), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
