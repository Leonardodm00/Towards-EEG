"""End-to-end check of the wave-1/2 chain on a minimal synthetic cell (tester 2026-11-13).

No script entry point exists yet for Blocks 23-31 (the round loop is Block
32, wave 3); SPEC.md section 6 lists only their smoke suites. This runs the
chain the way the entries' data-flow lines wire it: SWC (Block 8) ->
cell_scene (Block 13) -> initial_features / featured_scene (23) -> the
cell's image block (24, through the camera chain, seeded) -> contamination
(27) -> tiers (28) -> tier-1 shadow (25) -> cleaning (28) -> the line's
profiles (Block 5's sample_profile) -> per-node fit with line_profiles as
forward (26 -> 30) -> features row via offset_xyz (29 -> 23) -> CSV.
Checks: it finishes; every output finite with the stated shape and dtype;
two runs with one seed identical.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.testing import assert_array_equal

import nc_helpers as H
from allen_diameter.analysis import cleaning as CL
from allen_diameter.analysis import coarse_fit as CF
from allen_diameter.analysis import contamination as CT
from allen_diameter.analysis import node_features as NF
from allen_diameter.analysis import node_inference as NI
from allen_diameter.analysis.cell import cell_scene, stretch_branch, to_image_um
from allen_diameter.analysis.profiles import sample_profile
from allen_diameter.loading import features_io as FIO
from allen_diameter.loading.swc_io import read_swc
from allen_diameter.model import geometry
from allen_diameter.model import line_render as LR
from allen_diameter.model import scene_render as SR
from allen_diameter.model import shadow as SH

SWC = """# minimal cell
1 1 -6.0 0.0 1.0 1.0 -1
2 3 2.0 0.0 1.0 0.3 1
3 3 3.5 0.0 1.0 0.3 2
4 3 5.0 0.0 1.0 0.3 3
5 3 6.5 0.0 1.0 0.3 4
6 3 8.0 0.0 1.0 0.3 5
7 3 3.0 1.6 1.0 0.25 3
8 3 7.0 1.6 1.0 0.25 7
9 4 3.0 3.0 1.3 0.3 1
10 4 7.0 3.0 1.3 0.3 9
"""

OV = {"renderer.absorption": "partition_vertical", "cleaning.shadow_absorption": "partition_vertical",
      "inference.absorption": "partition_vertical", "inference.multistart_factors": [1.0],
      "inference.max_nfev": 40, "renderer.fft_split_sigma_um": 50.0}


def seg_tube(fs, q):
    p0, p1 = fs.seg_xyz[q, :3], fs.seg_xyz[q, 3:]
    t = p1 - p0
    if t[2] < 0:
        t = -t
    L = float(np.linalg.norm(t))
    c = 0.5 * (p0 + p1)
    return geometry.Tube(c=tuple(c), r=0.25 * float(fs.seg_d_um[q].sum()),
                         phi=math.atan2(t[2], math.hypot(t[0], t[1])), theta=math.atan2(t[1], t[0]),
                         half_length=0.5 * L)


def run_chain(tmp_path, seed):
    cfg = H.cfg_with(OV)
    p = cfg.acquisition.res0_um
    path = tmp_path / ("cell_%d.swc" % seed)
    path.write_text(SWC)
    swc = read_swc(str(path))
    sc = cell_scene(swc, cfg.cleaning.neighbour_swc_types, p)
    f0 = NF.initial_features(swc, cfg.cleaning.neighbour_swc_types, 1.0)
    fs = NF.featured_scene(sc, f0)
    # the cell's block around node 4 (x = 5 um): +-3 um, planes 3..5 (z_k = k dz)
    node_xyz = to_image_um(swc.xyz, p)[3]
    left, top, W, Hh = int(round(node_xyz[0] / p)) - 26, int(round(node_xyz[1] / p)) - 26, 53, 53
    ks = np.array([3, 4, 5])
    tubes = [seg_tube(fs, q) for q in range(fs.seg_xyz.shape[0])]
    mus = [float(fs.seg_mu_per_um[q].mean()) for q in range(fs.seg_xyz.shape[0])]
    block, ks_out, valid, frame = SR.synthetic_scene_block(tubes, mus, ks, left, top, W, Hh, cfg,
                                                           np.random.default_rng(seed), 30)
    B = cfg.renderer.background_B_gl
    # contamination survey of node 4 on its stretch (rows of nodes 2..6)
    rows = np.array([1, 2, 3, 4, 5])
    br, off = stretch_branch(swc, rows, to_image_um(swc.xyz, p))
    surv = CT.node_contamination(br, off + 2, sc, B, 1.0, cfg)
    own = [2, 3, 4, 5, 6]
    tiers = CL.neighbour_tiers(fs, own, node_xyz, cfg)
    segs, somas = SH.neighbour_indices(fs, own)
    sh = SH.shadow_image(fs, own, left, top, W, Hh, ks * cfg.acquisition.dz_um, cfg, B_gl=B, include=tiers == 1)
    clean = CL.clean_block(block, sh.S, B, cfg)
    yh, eu = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    v = np.arange(-17, 18) * cfg.measure.profile_step_um
    prof = np.stack([sample_profile(clean.block[i], frame, node_xyz[:2], yh, eu, v, 0, 0.0) for i in range(3)])

    def fwd(d, mu, v0, z):
        c = node_xyz[:2] + v0 * yh
        tube = geometry.Tube(c=(float(c[0]), float(c[1]), float(z)), r=0.5 * d, phi=0.0, theta=0.0, half_length=1.5)
        return LR.line_profiles(tube, mu, ks, node_xyz[:2], yh, eu, v, 0, 0.0, cfg)

    fit = NI.fit_node(prof, B, cfg.renderer.noise_sd_gl, fwd, (0.6, 1.0, 0.0, 1.0), cfg)
    offv = CF.offset_xyz(fit.theta[2], fit.theta[3] - node_xyz[2], yh)
    f1 = NF.with_rows(f0, [{"node_id": 4, "d_um": fit.theta[0], "mu_per_um": fit.theta[1], "offset_um": offv,
                            "source": "fine"}], 1)
    FIO.write_features(str(tmp_path / ("f_%d.csv" % seed)), f1)
    back = FIO.read_features(str(tmp_path / ("f_%d.csv" % seed)))
    return dict(block=block, valid=valid, surv=surv, tiers=tiers, n_nb=segs.size + somas.size, S=sh.S,
                clean=clean.block, prof=prof, fit=fit, f1=f1, back=back)


def test_chain_runs_and_is_reproducible(tmp_path):
    a = run_chain(tmp_path, 11)
    b = run_chain(tmp_path, 11)
    assert a["block"].dtype == np.uint8 and a["block"].shape == (3, 53, 53) and a["valid"].all()
    assert a["S"].shape == (3, 53, 53) and np.all(np.isfinite(a["S"])) and np.all(a["S"] >= 0)
    assert a["clean"].dtype == np.float64 and np.all(np.isfinite(a["clean"]))
    assert a["tiers"].shape == (a["n_nb"],) and set(a["tiers"].tolist()) <= {1, 2}
    assert a["prof"].shape == (3, 35) and np.all(np.isfinite(a["prof"]))
    fit = a["fit"]
    print("survey:", a["surv"]["class"], "%.3f" % a["surv"]["index"], "tiers", a["tiers"].tolist(),
          "fit:", fit.status, np.round(fit.theta, 4), "se", np.round(fit.se, 4), "chi2/N", np.round(fit.chi2_per_plane, 2))
    assert np.all(np.isfinite(fit.theta)) and fit.theta.shape == (4,)
    for k in ("block", "S", "clean", "prof", "tiers"):
        assert_array_equal(a[k], b[k])
    assert_array_equal(a["fit"].theta, b["fit"].theta)
    assert_array_equal(a["back"].d_um, a["f1"].d_um) and assert_array_equal(a["back"].offset_um, a["f1"].offset_um) is None


def test_swappability_through_configuration(tmp_path, monkeypatch):
    """Directive audit: one component changed through the configuration only (the kernel's continuation,
    the line renderer's fine factor, the shadow model's absorption to the hybrid with a small quadrature);
    the chain still runs and its outputs stay finite."""
    swapped = dict(OV, **{"renderer.kernel_continuation": "proportional", "inference.line_fine_factor": 4,
                          "cleaning.shadow_absorption": "hybrid", "cleaning.shadow_ray_n_rho": 3,
                          "cleaning.shadow_ray_n_psi": 6})
    monkeypatch.setitem(globals(), "OV", swapped)
    a = run_chain(tmp_path, 3)
    assert np.all(np.isfinite(a["clean"])) and np.all(np.isfinite(a["fit"].theta))
    print("swapped config fit:", a["fit"].status, np.round(a["fit"].theta, 4))
