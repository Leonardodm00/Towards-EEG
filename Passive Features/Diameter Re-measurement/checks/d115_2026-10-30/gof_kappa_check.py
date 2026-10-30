"""D-115.3 check: a scanned plane's core-fit residual RMS against its background's noise SD, on rendered phantoms
with and without a neighbouring dip -- the data to set kappa_g (PROVISIONAL until this check).

Each phantom is drawn as calibrate_kernel.py synthetic draws the near set (d log-uniform on (0.15, 0.25) um, alpha
uniform on (0.15, 0.4), phi < 2 deg, the default configuration) and scanned by the pipeline's own
calibration.scan_node (the pilot, the verdict, the patch, plane_scan) with keep_images:
  clean      the phantom alone (render.synthetic_block's chain);
  wide       a clean phantom as wide and deep as the pilot's real near nodes measure (d log-uniform on
             WIDE_D_RANGE um, alpha uniform on WIDE_ALPHA_RANGE), traced with Allen's radius WIDE_ALLEN_R_UM (a near
             node by Allen's diameter, as the pilot's are: their d_hat is 0.77-2.32 um, 10th-90th percentile, for
             Allen's 2r <= 0.6 um);
  composite  the phantom and a parallel neighbour of the same diameter and absorption, at a lateral offset s
             (perpendicular to the heading, in the image plane; s uniform on S_RANGE um) and a depth offset dz
             (uniform on DZ_RANGE um); its transmittance multiplied into the phantom's before the camera chain (the
             neighbour is not traced: the scene is the phantom's own branch, as for an untraced dendrite).
Per scanned plane with status ok: g = rms_gl / sd_bg, with sd_bg = 1.4826 x MAD of the residuals of the support
pixels (the scan's own far mask, which planar_background fitted) about the plane's fitted background b_coef.

Usage (from the workstream directory):
    python3 -I checks/d115_2026-10-30/gof_kappa_check.py OUT_JSON N_CLEAN N_COMPOSITE [WORKERS] [SET] [N_WIDE]
SET: near (default) or far. Pure ASCII.
"""
import dataclasses
import json
import math
import os
import sys
import time

sys.path[:0] = [os.path.abspath("src"), os.path.abspath("scripts")]

import numpy as np  # noqa: E402

SEED = 20261030
S_RANGE = (0.5, 1.5)      # um: the neighbour's lateral offset from the phantom's axis
DZ_RANGE = (-0.5, 0.5)    # um: its depth offset
D_RANGES = {"near": (0.15, 0.25), "far": (0.6, 0.8)}
WIDE_D_RANGE = (0.6, 2.5)       # um
WIDE_ALPHA_RANGE = (0.15, 0.6)
WIDE_ALLEN_R_UM = 0.25          # um: Allen's 2r = 0.5 um, inside the near set


def _sd_bg(scan):
    """{k: 1.4826 x MAD of the support pixels' residuals about b_coef} over the planes kept in scan.images."""
    from allen_image_io import CropFrame
    from allen_diameter.analysis import calibration_scene
    im = scan.images
    frame = CropFrame(int(im["left"]), int(im["top"]), 0, float(im["res_um_px"]))
    ks = list(np.asarray(scan.ks, dtype=int))
    out = {}
    for q, k in enumerate(np.asarray(im["ks"], dtype=int)):
        img = np.asarray(im["stack"][q], dtype=float)
        far = np.asarray(im["far"][q], dtype=bool)
        b = np.asarray(scan.b_coef[ks.index(int(k))], dtype=float)
        if far.sum() < 6 or not np.all(np.isfinite(b)):
            continue
        px, py = calibration_scene.pixel_centres(img.shape, frame)
        r = img[far] - (b[0] + b[1] * (px[far] - im["cx_um"]) + b[2] * (py[far] - im["cy_um"]))
        out[int(k)] = float(1.4826 * np.median(np.abs(r - np.median(r))))
    return out


def _one(args):
    kind, index, which = args
    import calibrate_kernel as CK
    from allen_image_io import CropFrame
    from allen_diameter.analysis import calibration, phantoms
    from allen_diameter.config import default_config
    from allen_diameter.model import camera, render
    cfg = default_config()
    if kind == "wide":
        draw, rng = CK.calibration_phantom(cfg, SEED + 2, index, WIDE_D_RANGE, WIDE_ALPHA_RANGE, 2.0)
    else:
        draw, rng = CK.calibration_phantom(cfg, SEED, index, D_RANGES[which], (0.15, 0.4), 2.0)
    tube = phantoms.phantom_tube(draw, cfg)
    if kind == "wide":
        draw = dataclasses.replace(draw, radius_um=WIDE_ALLEN_R_UM)     # the trace's radius only; the tube is rendered
    ph = cfg.phantom
    need = max(cfg.calibration.far_profile_half_um, cfg.calibration.node_straight_half_um) + 2.0
    n_each = max(int(ph.nodes_each_way), int(math.ceil(need / ph.phantom_node_step_um)))
    branch = phantoms.phantom_branch(draw, dataclasses.replace(cfg, phantom=dataclasses.replace(ph, nodes_each_way=n_each)), rng)
    pad = int(math.ceil(cfg.renderer.pad_um / cfg.acquisition.res0_um))
    acq, rcfg = cfg.acquisition, cfg.renderer
    nb = None
    if kind == "composite":
        g = np.random.default_rng([SEED, 1, int(index)])
        s, dz = float(g.uniform(*S_RANGE)), float(g.uniform(*DZ_RANGE))
        side = 1.0 if g.uniform() < 0.5 else -1.0
        th = draw.theta_rad
        draw2 = dataclasses.replace(draw, cx_um=draw.cx_um - side * s * math.sin(th),
                                    cy_um=draw.cy_um + side * s * math.cos(th), cz_um=draw.cz_um + dz)
        nb = (phantoms.phantom_tube(draw2, cfg), draw2.mu_per_um, s, dz)

    def provider(left, top, width, height, k_lo, k_hi):
        ks = np.arange(k_lo, k_hi + 1)
        if nb is None:
            return render.synthetic_block(tube, draw.mu_per_um, ks, left, top, width, height, cfg, rng, pad)
        f = max(render.fine_factor(tube.d, rcfg, acq.res0_um), render.fine_factor(nb[0].d, rcfg, acq.res0_um))
        grid, inner = render.block_fine_grid(left, top, width, height, acq.res0_um, f, pad)

        def red(plane):
            return render.pixel_integrate(plane[inner[0], inner[1]], f)
        t1 = render.render_transmittance(tube, draw.mu_per_um, ks * acq.dz_um, grid, rcfg, acq.light_direction,
                                         None, reduce=red).tau
        t2 = render.render_transmittance(nb[0], nb[1], ks * acq.dz_um, grid, rcfg, acq.light_direction, None,
                                         reduce=red).tau
        block = camera.camera_chain(t1 * t2, rcfg, rng, None)
        return block, ks, np.ones(ks.size, dtype=bool), CropFrame(int(left), int(top), 0, float(acq.res0_um))

    res, scan, reasons = calibration.scan_node(branch, n_each, provider, cfg, keep_images=True)
    rec = dict(kind=kind, index=int(index), set=which, d_um=draw.d_um,
               alpha=draw.mu_per_um * draw.d_um / math.cos(draw.phi_rad),
               s_um=None if nb is None else nb[2], dz_um=None if nb is None else nb[3],
               reasons=list(reasons), planes=[])
    if scan is None:
        return rec
    sd = _sd_bg(scan)
    for q, k in enumerate(np.asarray(scan.ks, dtype=int)):
        rms = float(scan.rms_gl[q]) if scan.rms_gl is not None else float("nan")
        rec["planes"].append(dict(k=int(k), offset=int(k - scan.k_star), status=str(scan.status[q]), rms_gl=rms,
                                  sd_bg_gl=sd.get(int(k), float("nan")), amp_gl=float(scan.amp_gl[q]),
                                  depth_gl=float(scan.depth_gl[q]), omega_um2=float(scan.omega_um2[q])))
    return rec


def main(out_json, n_clean, n_comp, workers=2, which="near", n_wide=0):
    jobs = [("clean", i, which) for i in range(int(n_clean))] + [("composite", i, which) for i in range(int(n_comp))]
    jobs += [("wide", i, which) for i in range(int(n_wide))]
    t0 = time.perf_counter()
    if int(workers) > 1:
        import multiprocessing
        with multiprocessing.Pool(int(workers)) as pool:
            recs = list(pool.imap(_one, jobs, chunksize=1))
    else:
        recs = [_one(j) for j in jobs]
    with open(out_json, "w") as f:
        json.dump(dict(seed=SEED, set=which, s_range_um=S_RANGE, dz_range_um=DZ_RANGE, records=recs,
                       minutes=(time.perf_counter() - t0) / 60), f, indent=1, allow_nan=True)
    print("%d phantoms (%d clean, %d composite) in %.1f min -> %s" % (len(recs), int(n_clean), int(n_comp),
                                                                     (time.perf_counter() - t0) / 60, out_json))


if __name__ == "__main__":
    main(*sys.argv[1:7])
