"""Spec-derived tests of Block 55 (analysis/step2_measure.py; tester, 2026-11-27). Oracles from SPEC.md Block 55 and
the common definitions: camera_planes per plane with its own background and SD against camera.camera_chain (Block 4,
a neighbour) with rcfg_q built here from the spec's mapping; invalid planes; the generators' seeds
[seed, cell, node, replicate, q]; twin_scene's node and straight branch; measure_twin's refusal of planes that are
not stage_planes, the near and far scans; the measured background (M3); common random numbers (M4); the replicate
statistics and their NaN rule through a stand-in measure_twin (M5); a non-finite B_gl (M7).

Stacks are rendered here by render.render_transmittance (partition absorption, for time) on small patches.

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import step2_helpers as H  # noqa: F401  (sys.path)
from allen_diameter import config as C
from allen_diameter.analysis import calibration as CAL
from allen_diameter.analysis import calibration_scene as CS
from allen_diameter.analysis import step2_measure as M
from allen_diameter.analysis.step2_render import TwinStack
from allen_diameter.analysis.step2_twins import Twin
from allen_diameter.model import camera, geometry, render

NEAR_HALF = 8.0
FAR_HALF = 10.5
_CACHE = {}


def _cfg(half, **rkw):
    c = C.default_config()
    c = dataclasses.replace(c, calibration=dataclasses.replace(c.calibration, patch_half_um=half))
    if rkw:
        c = dataclasses.replace(c, renderer=dataclasses.replace(c.renderer, **rkw))
    return c


def _stack(tube, mu, ks, cx, cy, half, cfg):
    acq = cfg.acquisition
    left, top, w, h = CS.patch_bounds(cx, cy, half, acq.res0_um)
    pad = int(math.ceil(cfg.renderer.pad_um / acq.res0_um))
    f = render.fine_factor(tube.d, cfg.renderer, acq.res0_um)
    grid, inner = render.block_fine_grid(left, top, w, h, acq.res0_um, f, pad)
    res = render.render_transmittance(tube, mu, np.asarray(ks) * acq.dz_um, grid, cfg.renderer, acq.light_direction,
                                      None, reduce=lambda p: render.pixel_integrate(p[inner[0], inner[1]], f))
    return TwinStack(tau=np.asarray(res.tau, float), ks=np.asarray(ks, np.int64), left=int(left), top=int(top),
                     res0_um=float(acq.res0_um))


def _twin(ks, B, sd, set_="near", d=0.4, mu=1.0, cx=0.03, cy=-0.02, z=0.0, theta=0.5, k_star=0, node_id=11):
    n = len(ks)
    z0 = np.zeros(n)
    return Twin(cell_id=7, node_id=node_id, set=set_, role="train", d_um=d, mu_per_um=mu, k_star=k_star, z_sub_um=z,
                cx_um=cx, cy_um=cy, theta_rad=theta, phi_rad=0.0, B_pilot_gl=float(np.nanmean(B)), d_pilot_um=d,
                alpha_pilot=mu * d, ks=np.asarray(ks, np.int64), B_gl=np.asarray(B, float),
                noise_sd_gl=np.asarray(sd, float), omega_um2=z0, se_omega_um2=z0, area_gl_um=z0, status=("ok",) * n)


def _near(d=0.4, mu=1.0, key="near"):
    if key not in _CACHE:
        cfg = _cfg(NEAR_HALF)
        tube = geometry.Tube((0.03, -0.02, 0.0), d / 2, 0.0, 0.5, 1.0, 40.0, cfg.renderer.end_cut)
        _CACHE[key] = (cfg, _stack(tube, mu, np.arange(-3, 4), 0.03, -0.02, NEAR_HALF, cfg))
    return _CACHE[key]


def test_noise_rngs_seeds():
    s2 = C.default_config().step2
    t = _twin(np.arange(3), np.full(3, 200.0), np.full(3, 2.0))
    for rep in (0, 3):
        gens = M.noise_rngs(t, 3, s2, rep)
        assert len(gens) == 3
        for q, g in enumerate(gens):
            ref = np.random.default_rng([s2.seed, 7, 11, rep, q]).standard_normal(5)
            assert_array_equal(g.standard_normal(5), ref)


def test_camera_planes_per_plane_background_and_sd():
    cfg, st = _near()
    rc = cfg.renderer
    n = st.tau.shape[0]
    B = np.linspace(150.0, 220.0, n)
    B[2] = np.nan                                     # invalid: not finite
    B[4] = rc.black_level_gl                          # invalid: background_B_gl would be 0
    sd = np.linspace(1.0, 4.0, n)
    gens = [np.random.default_rng([1, q]) for q in range(n)]
    blk, valid = M.camera_planes(st, B, sd, rc, gens)
    assert blk.shape == st.tau.shape and blk.dtype == np.uint8 and valid.dtype == bool
    assert_array_equal(valid, [True, True, False, True, False, True, True])
    for q in range(n):
        if not valid[q]:
            assert not blk[q].any()
            continue
        rq = dataclasses.replace(rc, background_B_gl=(B[q] - rc.black_level_gl) / rc.gain, noise_sd_gl=sd[q])
        assert_array_equal(blk[q], camera.camera_chain(st.tau[q], rq, np.random.default_rng([1, q])))


def test_twin_scene_node_and_branch():
    cfg = C.default_config()
    s2 = cfg.step2
    t = dataclasses.replace(_twin(np.arange(-3, 4), np.full(7, 200.0), np.full(7, 2.0)), phi_rad=0.3, z_sub_um=1.1,
                            d_pilot_um=0.61, alpha_pilot=0.44, B_pilot_gl=203.0)
    node, branch, scene = M.twin_scene(t, cfg, s2)
    for a, b in (("node_id", "node_id"), ("k_star", "k_star"), ("z_sub_um", "z_sub_um"), ("cx_um", "cx_um"),
                 ("cy_um", "cy_um"), ("theta_rad", "theta_rad"), ("phi_rad", "phi_rad"), ("d_hat_um", "d_pilot_um"),
                 ("alpha_hat", "alpha_pilot"), ("B_bar", "B_pilot_gl"), ("mu_hat_per_um", "mu_per_um")):
        assert getattr(node, a) == getattr(t, b), a
    P = np.asarray(branch.xyz_um)
    step = cfg.phantom.phantom_node_step_um
    tdir = np.array([math.cos(0.3) * math.cos(0.5), math.cos(0.3) * math.sin(0.5), math.sin(0.3)])
    c = np.array([t.cx_um, t.cy_um, t.z_sub_um])
    rel = P - c
    s = rel @ tdir
    assert_allclose(rel - s[:, None] * tdir[None, :], 0.0, atol=1e-12)          # on the axis
    assert_allclose(np.diff(s), step, rtol=1e-12)                              # every node step
    assert s.min() >= -s2.tube_half_length_um - 1e-9 and s.max() <= s2.tube_half_length_um + 1e-9
    assert s.max() > s2.tube_half_length_um - step                             # covers +-L
    assert np.any(np.abs(s) < 1e-12)                                           # passes through the centre
    assert_allclose(branch.radius_um, t.d_um / 2, rtol=0, atol=0)


def test_measure_twin_refuses_wrong_planes():
    cfg, st = _near()
    t = _twin(st.ks, np.full(7, 200.0), np.full(7, 2.0), k_star=1)            # k* 1: stage_planes -2..4
    with pytest.raises(ValueError):
        M.measure_twin(t, st, cfg, cfg.step2)


def test_M3_background_and_sd_jpeg_off():
    cfg0, st = _near()
    cfg = dataclasses.replace(cfg0, renderer=dataclasses.replace(cfg0.renderer, jpeg=False))
    B = np.array([190.0, 195, 200, 205, 210, 200, 190])
    sd = np.full(7, 3.0)
    t = _twin(st.ks, B, sd)
    sc = M.measure_twin(t, st, cfg, cfg.step2)
    assert sc.set == "near" and sc.ks.tolist() == list(range(-3, 4))
    ok = np.isfinite(sc.B_bar)
    assert ok.sum() >= 5
    # 0.5 gl: the planar fit of thousands of pixels of SD 3 gl has an SE of a few hundredths of a grey level, plus
    # the 8-bit rounding's bias, below 0.5
    assert np.max(np.abs(sc.B_bar[ok] - B[ok])) <= 0.5
    # 30 %: the SD of a quantised Gaussian of SD 3 is sqrt(9 + 1/12) = 3.01; the support's thousands of pixels
    # make the sample SD's own spread about 1 %
    bs = np.asarray(sc.bg_sd_gl)
    assert np.all(np.abs(bs[ok] / 3.0 - 1.0) <= 0.30)


def test_M4_common_random_numbers():
    cfg0, st = _near()
    cfg = dataclasses.replace(cfg0, renderer=dataclasses.replace(cfg0.renderer, jpeg=False, bit_depth=16))
    t = _twin(st.ks, np.full(7, 2000.0), np.full(7, 3.0))
    gens = lambda rep: M.noise_rngs(t, 7, cfg.step2, rep)  # noqa: E731
    a, _ = M.camera_planes(st, t.B_gl, t.noise_sd_gl, cfg.renderer, gens(0))
    b, _ = M.camera_planes(st, t.B_gl, t.noise_sd_gl, cfg.renderer, gens(0))
    c, _ = M.camera_planes(st, t.B_gl, t.noise_sd_gl, cfg.renderer, gens(1))
    assert_array_equal(a, b) and not np.array_equal(a, c)
    st2 = dataclasses.replace(st, tau=np.clip(st.tau * 0.98 + 0.01, 0, 1))
    d, _ = M.camera_planes(st2, t.B_gl, t.noise_sd_gl, cfg.renderer, gens(0))
    rc = cfg.renderer
    bg = (2000.0 - rc.black_level_gl) / rc.gain
    expect = rc.gain * bg * (st2.tau - st.tau)
    assert np.max(np.abs((d.astype(float) - a.astype(float)) - expect)) <= 1.0      # rounding of both, 1 gl


class _FakeScan:
    def __init__(self, om, area, bsd):
        self.omega_um2, self.area_gl_um, self.bg_sd_gl = np.asarray(om, float), np.asarray(area, float), bsd


def test_M5_statistics_rule(monkeypatch):
    cfg, st = _near()
    t = _twin(st.ks[:3], np.full(3, 200.0), np.full(3, 2.0))
    nan = np.nan
    # four replicates x three planes: plane 0 all finite; plane 1 two finite (= ceil(4/2)); plane 2 one finite
    om = np.array([[1.0, 2.0, nan], [2.0, nan, 5.0], [4.0, 6.0, nan], [5.0, nan, nan]])
    area = np.array([[1.0, nan, nan], [3.0, 2.0, nan], [5.0, 4.0, nan], [7.0, 6.0, nan]])
    bsd = [np.array([1.0, 2.0, 3.0]), None, np.array([3.0, nan, 5.0]), np.array([5.0, 4.0, nan])]
    calls = []

    def fake(twin, stack, scan_cfg, s2cfg, replicate=0, noise=True):
        calls.append((replicate, noise))
        return _FakeScan(om[replicate], area[replicate], bsd[replicate])
    monkeypatch.setattr(M, "measure_twin", fake)
    s2 = dataclasses.replace(cfg.step2, noise_replicates=4)
    out = M.twin_statistics(t, st, cfg, s2)
    assert sorted(calls) == [(0, True), (1, True), (2, True), (3, True)]
    assert_allclose(out["omega_um2"], [3.0, 4.0, nan], equal_nan=True)
    assert_allclose(out["omega_sd_um2"], [np.std([1, 2, 4, 5], ddof=1), np.std([2, 6], ddof=1), nan], equal_nan=True)
    assert_array_equal(out["n_ok"], [4, 2, 1])
    assert_allclose(out["area_gl_um"], [4.0, 4.0, nan], equal_nan=True)
    assert_allclose(out["bg_sd_gl"], [3.0, 3.0, 4.0], equal_nan=True)
    calls.clear()
    out1 = M.twin_statistics(t, st, cfg, s2, noise=False)
    assert calls == [(0, False)]                                                  # R = 1 without noise
    assert_allclose(out1["omega_um2"], [1.0, 2.0, nan], equal_nan=True)
    assert np.all(np.isnan(out1["omega_sd_um2"]))
    # R = 3: ceil(3/2) = 2 finite needed
    s3 = dataclasses.replace(cfg.step2, noise_replicates=3)
    out3 = M.twin_statistics(t, st, cfg, s3)
    assert_allclose(out3["omega_um2"], [7 / 3, 4.0, nan], equal_nan=True)


def test_M6_far_twin_goes_through_the_far_scan():
    cfg = _cfg(FAR_HALF)
    P = cfg.calibration.far_offsets_planes
    ks = np.arange(-P, P + 1)
    tube = geometry.Tube((0.03, -0.02, 0.0), 0.35, 0.0, 0.5, 1.0, 40.0, cfg.renderer.end_cut)
    st = _stack(tube, 0.6, ks, 0.03, -0.02, FAR_HALF, cfg)
    t = _twin(ks, np.full(ks.size, 200.0), np.full(ks.size, 2.0), set_="far", d=0.7, mu=0.6)
    sc = M.measure_twin(t, st, cfg, cfg.step2)
    assert sc.set == "far" and sc.ks.size == 2 * P + 1 == 33
    assert_array_equal(sc.ks, ks)


def test_M7_non_finite_background_gives_an_invalid_plane():
    cfg, st = _near()
    B = np.full(7, 200.0)
    B[1] = np.nan
    sc = M.measure_twin(_twin(st.ks, B, np.full(7, 2.0)), st, cfg, cfg.step2)
    assert sc.status[1] != "ok" and not np.isfinite(sc.omega_um2[1])
    assert sum(s == "ok" for s in sc.status) >= 3


# ---------------------------------------------------------------------------------------------------------------
# Amendment 3 (tester, 2026-11-27 22:11; SPEC Block 55, M8): the twin's scan runs with calibration.gof_check False
# whatever scan_cfg says. Oracles: no "poor_fit" in the twin's scan with gof_check True in scan_cfg, noise on and
# off, near and far; the scan equals the one made with gof_check False, field for field (NaN equal NaN); the
# caller's configuration is unchanged. Non-vacuity: the same patch scanned directly by calibration.plane_scan with
# gof_check True (the real scans' screen) does flag planes, so the screen would have acted.
# ---------------------------------------------------------------------------------------------------------------

def _gof(cfg, on):
    return dataclasses.replace(cfg, calibration=dataclasses.replace(cfg.calibration, gof_check=on))


def _scan_equal(a, b):
    assert a.set == b.set and a.status == b.status and a.k_star == b.k_star
    for f in ("ks", "omega_um2", "se_omega_um2", "area_gl_um", "B_bar", "vw_um2", "depth_gl"):
        assert_array_equal(np.asarray(getattr(a, f)), np.asarray(getattr(b, f)), err_msg=f)
    if a.bg_sd_gl is not None or b.bg_sd_gl is not None:
        assert_array_equal(np.asarray(a.bg_sd_gl), np.asarray(b.bg_sd_gl))


def _direct_scan(t, st, cfg, s2, noise, replicate=0):
    """The patch measure_twin builds (SPEC Block 55), scanned by calibration.plane_scan under cfg as given."""
    from allen_image_io import CropFrame
    n = st.ks.size
    sd = s2.noise_scale * np.asarray(t.noise_sd_gl) if noise else np.zeros(n)
    blk, valid = M.camera_planes(st, t.B_gl, sd, cfg.renderer, M.noise_rngs(t, n, s2, replicate))
    node, branch, scene = M.twin_scene(t, cfg, s2)
    patch = (blk, st.ks, valid, CropFrame(int(st.left), int(st.top), 0, float(st.res0_um)))
    return CAL.plane_scan(patch, node, branch, scene, cfg)


@pytest.mark.parametrize("noise", [True, False])
def test_M8_near_twin_scan_ignores_the_gof_screen(noise):
    cfg0, st = _near(d=0.5, mu=0.8, key="near05")
    on, off = _gof(cfg0, True), _gof(cfg0, False)
    t = _twin(st.ks, np.full(7, 200.0), np.full(7, 1.5), d=0.5, mu=0.8)
    s2 = cfg0.step2
    a = M.measure_twin(t, st, on, s2, noise=noise)
    b = M.measure_twin(t, st, off, s2, noise=noise)
    assert "poor_fit" not in a.status
    _scan_equal(a, b)
    assert on.calibration.gof_check is True                    # the caller's configuration is not changed
    direct = _direct_scan(t, st, on, s2, noise)
    print("\nnoise %s: twin statuses %s; direct scan with gof on %s" % (noise, a.status, direct.status))
    if not noise:
        # SPEC amendment 3's reason: with the noise off the screen rejects every plane the core fit reaches
        assert sum(s == "poor_fit" for s in direct.status) >= 1
        assert all(s == "ok" for s in a.status)


def test_M8_far_twin_scan_ignores_the_gof_screen():
    cfg = _cfg(FAR_HALF)
    P = cfg.calibration.far_offsets_planes
    ks = np.arange(-P, P + 1)
    if "far07" not in _CACHE:
        tube = geometry.Tube((0.03, -0.02, 0.0), 0.35, 0.0, 0.5, 1.0, 40.0, cfg.renderer.end_cut)
        _CACHE["far07"] = _stack(tube, 0.6, ks, 0.03, -0.02, FAR_HALF, cfg)
    st = _CACHE["far07"]
    t = _twin(ks, np.full(ks.size, 200.0), np.full(ks.size, 2.0), set_="far", d=0.7, mu=0.6)
    for noise in (True, False):
        a = M.measure_twin(t, st, _gof(cfg, True), cfg.step2, noise=noise)
        b = M.measure_twin(t, st, _gof(cfg, False), cfg.step2, noise=noise)
        assert "poor_fit" not in a.status
        _scan_equal(a, b)
