"""What blocks the stripe on the pilot's real reconstruction, and the stripe if broad shadows are left to the plane.
For each D-101 candidate (pilot table of 2026-10-07; scene = the pilot's SWC, no transform) and plane (near: k*;
far: k* and |k - k*| = 5..16 on both sides):
  centre: the neighbours whose predicted shadow (10.I's D_qk with the pilot's B_bar, mu) at the pilot's centre
          exceeds tau, by kind: soma (at the stage's largest blur), junction (a segment with one end on the stretch),
          broad (sigma_r(delta) >= 2 um), sharp (the rest);
  caps:   the stripe (CS.shadow_masks + CS.stripe_support, the soma at the stage's blur) when neighbours with
          sigma_r(delta_qk) >= sigma_cap are left out of the masks, and the residual their summed predicted shadow
          leaves on the measuring line after the least-squares plane over the stripe (subsampled 1 in 6 x 6).
Usage: python cap_check.py --swc SWC --pilot-csv CSV --out OUT.jsonl IDS..."""
import json, math, sys, time, types
import numpy as np
from numpy.polynomial.legendre import leggauss
import argparse
import os
WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
for p in (WS + "/src", WS + "/scripts"):
    sys.path.insert(0, p)
from allen_diameter.config import default_config
from allen_diameter.loading import swc_io, table_io
from allen_diameter.analysis import cell, calibration as CAL, calibration_scene as CS, background as BG
from allen_diameter.model.kernel import sigma_r

ap = argparse.ArgumentParser(description="see the module docstring")
ap.add_argument("--swc", required=True, help="the pilot's SWC (Drive: MyDrive/allen_cache/H16.06.010.01.03.05.03_668702901_m.swc)")
ap.add_argument("--pilot-csv", required=True, help="Cell 4b's pilot table (Drive: MyDrive/diameters/pilot/pilot_529878215.csv)")
ap.add_argument("--out", required=True, help="the JSONL written, one line per node")
ap.add_argument("ids", nargs="+", type=int, help="node ids (the D-101 candidates: checks/d105_2026-10-28/ids.txt)")
args = ap.parse_args()
SWC, CSV = args.swc, args.pilot_csv
CAPS = (2.0, 3.0, 5.0)
out_path, want = args.out, list(args.ids)
cfg = default_config()
cal, dz, res0, rc = cfg.calibration, cfg.acquisition.dz_um, cfg.acquisition.res0_um, cfg.renderer
tau = cal.isolation_depth_gl
swc = swc_io.read_swc(SWC)
tps = cfg.acquisition.dendrite_swc_types
xyz_img = cell.to_image_um(swc.xyz, res0)
scene = cell.cell_scene(swc, tps, res0, None)
where = {}
for run in cell.stretches(swc, tps):
    br, off = cell.stretch_branch(swc, run, xyz_img)
    for t, r in enumerate(run):
        where[int(swc.ids[r])] = (br, off + t)
rows_by_id = {int(cell.pilot_row_view(r)["node_id"]): cell.pilot_row_view(r) for r in table_io.read_rows(CSV)}
A, Bq = scene.seg_xyz[:, :3], scene.seg_xyz[:, 3:]
R = scene.seg_radius.max(axis=1)
ids = scene.seg_ids.astype(int)
XG, WG = leggauss(24)


def depth_many(rho, r, sig, mu, B):
    """(m, n) predicted shadow of n tubes at distances rho (m, n); 24-node Gauss-Legendre (sigma >= 2 r / 4 here)."""
    out = np.zeros(rho.shape)
    for s in range(0, r.size, 64):
        rr, ss = r[s:s + 64], sig[s:s + 64]
        vp = rr[None, :, None] * XG[None, None, :]
        deficit = 1.0 - np.exp(-2.0 * mu * np.sqrt(np.maximum(rr[None, :, None] ** 2 - vp ** 2, 0.0)))
        g = np.exp(-0.5 * ((rho[:, s:s + 64, None] - vp) / ss[None, :, None]) ** 2) / (math.sqrt(2 * math.pi) * ss[None, :, None])
        out[:, s:s + 64] = B * rr[None, :] * np.sum(WG * deficit * g, axis=-1)
    return out


def seg_dist(px, py, a, b):
    """(m, n) 2-D distances from m points to n segments."""
    ax, ay, bx, by = a[:, 0][None], a[:, 1][None], b[:, 0][None], b[:, 1][None]
    vx, vy = bx - ax, by - ay
    L2 = np.maximum(vx * vx + vy * vy, 1e-12)
    t = np.clip(((px[:, None] - ax) * vx + (py[:, None] - ay) * vy) / L2, 0.0, 1.0)
    return np.hypot(px[:, None] - (ax + t * vx), py[:, None] - (ay + t * vy))


with open(out_path, "w") as fo:
    for nid in want:
        rv = rows_by_id[nid]
        Da = 2.0 * float(rv["allen_radius_um"])
        which = CAL.calibration_set(Da, cfg)
        br, i = where[nid]
        node = types.SimpleNamespace(node_id=nid, cx_um=float(rv["cx_um"]), cy_um=float(rv["cy_um"]),
                                     theta_rad=float(rv["theta_rad"]), z_sub_um=float(rv["z_sub_um"]),
                                     k_star=int(rv["k_star"]), B_bar=float(rv["B_bar"]), mu=float(rv["mu_hat_per_um"]),
                                     d_hat_um=float(rv["d_hat_um"]))
        own_ids = {int(x) for x in np.asarray(br.ids).ravel()}
        own = np.array([a in own_ids and b in own_ids for a, b in ids])
        junction = np.array([(a in own_ids) != (b in own_ids) for a, b in ids])
        P = cal.offsets_planes if which == "near" else cal.far_offsets_planes
        sig_stage = float(sigma_r(P * dz, rc))
        offs = [0] if which == "near" else [0] + [o for o in range(-P, P + 1) if abs(o) >= cal.far_min_k_offset]
        shape, frame = CS.patch_frame(node.cx_um, node.cy_um, cal.patch_half_um, res0)
        px, py = CS.pixel_centres(shape, frame)
        th = node.theta_rad
        nv, tv = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
        h = cfg.measure.profile_half_um if which == "near" else cal.far_profile_half_um
        vline = np.arange(-h, h + 1e-9, 0.25)
        lx, ly = node.cx_um + vline * nv[0], node.cy_um + vline * nv[1]
        rho_c = seg_dist(np.array([node.cx_um]), np.array([node.cy_um]), A, Bq)[0]
        t0 = time.perf_counter()
        planes = []
        for o in offs:
            zk = (node.k_star + o) * dz
            delta = np.maximum(np.abs(A[:, 2] - zk), np.abs(Bq[:, 2] - zk))
            sig = sigma_r(delta, rc)
            dc = depth_many(rho_c[None, :], R, sig, node.mu, node.B_bar)[0]
            over = (dc > tau) & ~own
            soma_c = 0.0
            for (x, y, rs) in np.atleast_2d(scene.soma):
                soma_c = max(soma_c, CS.shadow_depth(math.hypot(node.cx_um - x, node.cy_um - y), rs, node.mu, sig_stage, node.B_bar))
            rec = dict(k_off=o, own_sigma=float(sigma_r(abs(zk - node.z_sub_um), rc)),
                       centre=dict(soma=bool(soma_c > tau), soma_gl=soma_c, junction=int((over & junction).sum()),
                                   broad=int((over & ~junction & (sig >= 2.0)).sum()),
                                   sharp=int((over & ~junction & (sig < 2.0)).sum()),
                                   sum_gl=float(dc[~own].sum()), sum_broad_gl=float(dc[~own & (sig >= 2.0)].sum())),
                       caps={})
            for cap in CAPS:
                keep = own | (sig < cap)
                # rows that can reach the patch: bbox within r + 6 sigma (the shadow_margin bracket's end)
                reach = R + 6.0 * sig
                lo_x, hi_x = (frame.left - 0.5) * res0, (frame.left + shape[1] - 0.5) * res0
                lo_y, hi_y = (frame.top - 0.5) * res0, (frame.top + shape[0] - 0.5) * res0
                near_box = ((np.maximum(A[:, 0], Bq[:, 0]) >= lo_x - reach) & (np.minimum(A[:, 0], Bq[:, 0]) <= hi_x + reach)
                            & (np.maximum(A[:, 1], Bq[:, 1]) >= lo_y - reach) & (np.minimum(A[:, 1], Bq[:, 1]) <= hi_y + reach))
                sel = keep & near_box
                rows = np.column_stack([A, Bq, scene.seg_radius[:, 0], scene.seg_radius[:, 1], own.astype(float)])[sel]
                nb, ow = CS.shadow_masks(shape, frame, rows, scene.soma, np.array([zk]), node.B_bar, node.mu, cfg,
                                         node.d_hat_um, sigma_soma_um=sig_stage)
                mask, info = CS.stripe_support(nb[0], ow[0], frame, node.cx_um, node.cy_um, th, cfg)
                res_max = res_c = float("nan")
                broad = ~own & ~(sig < cap)
                if info["n_pixels"] >= 6 and broad.any():
                    sub = np.zeros_like(mask)
                    sub[::6, ::6] = True
                    m2 = mask & sub
                    if m2.sum() < 6:
                        m2 = mask
                    qx, qy = px[m2], py[m2]
                    bi = np.flatnonzero(broad)
                    s_st = depth_many(seg_dist(qx, qy, A[bi], Bq[bi]), R[bi], sig[bi], node.mu, node.B_bar).sum(axis=1)
                    s_ln = depth_many(seg_dist(lx, ly, A[bi], Bq[bi]), R[bi], sig[bi], node.mu, node.B_bar).sum(axis=1)
                    Am = np.column_stack([np.ones(qx.size), qx - node.cx_um, qy - node.cy_um])
                    coef = np.linalg.lstsq(Am, s_st, rcond=None)[0]
                    pl = coef[0] + coef[1] * (lx - node.cx_um) + coef[2] * (ly - node.cy_um)
                    res = s_ln - pl
                    res_max, res_c = float(np.max(np.abs(res))), float(res[vline.size // 2])
                rec["caps"][str(cap)] = dict(ok=bool(info["ok"]), reason=info["reason"], n=int(info["n_pixels"]),
                                             w=info["w_um"], sm=info["side_minus_um"], sp=info["side_plus_um"],
                                             n_rows=int(sel.sum()), res_max=res_max, res_c=res_c)
            planes.append(rec)
        out = dict(node_id=nid, set=which, k_star=node.k_star, seconds=time.perf_counter() - t0, planes=planes)
        fo.write(json.dumps(out) + "\n")
        fo.flush()
        p0 = planes[0]
        msg = "node %5d %-4s centre@k*: soma %s junction %d broad %d sharp %d (sum %.2f gl, broad %.2f)" % (
            nid, which, p0["centre"]["soma"], p0["centre"]["junction"], p0["centre"]["broad"], p0["centre"]["sharp"],
            p0["centre"]["sum_gl"], p0["centre"]["sum_broad_gl"])
        for cap in CAPS:
            c = p0["caps"][str(cap)]
            msg += " | cap %.0f: %s %d px res %.3f" % (cap, c["reason"] or "ok", c["n"], c["res_max"])
        if which == "far":
            for cap in CAPS:
                okp = [p for p in planes[1:] if p["caps"][str(cap)]["ok"]]
                okr = [p for p in okp if not (p["caps"][str(cap)]["res_max"] > tau)]
                msg += " || far cap %.0f: %d ok, %d ok & res<=tau of %d" % (cap, len(okp), len(okr), len(planes) - 1)
        print(msg + "  [%.0f s]" % out["seconds"], flush=True)
