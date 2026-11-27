"""Spec-derived tests of Block 54 (analysis/step2_render.py; tester, 2026-11-27). Oracles R1 to R5 of SPEC.md's
Block 54, and the boundary with Block 4's amendment 4 (trial_renderer -> kernel.sigma_r at the knots, the
continuation per stage) and with Block 53 (a Twin's fields -> twin_tube). The reference for R1 is
render.synthetic_block (Block 4, a neighbour), called here with the tube, the planes and the patch built from the
spec's words; the patch bounds' reference is spec_helpers.ref_patch_bounds (an earlier tester's).

Small patches (calibration.patch_half_um 2.5 um) and thin tubes, for time.

Pure ASCII.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import step2_helpers as H  # noqa: F401  (sys.path)
from spec_helpers import ref_patch_bounds
from allen_diameter import config as C
from allen_diameter.analysis import step2_render as SR
from allen_diameter.analysis.step2_twins import Twin
from allen_diameter.model import camera, geometry, kernel, render

HALF = 2.5
THETA = (0.12, 0.33, 0.55)          # a trial table (um) above sigma_r(0) = the default's 0.081 (approx.)


def _cfg(half=HALF, **s2):
    c = C.default_config()
    c = dataclasses.replace(c, calibration=dataclasses.replace(c.calibration, patch_half_um=half))
    if s2:
        c = dataclasses.replace(c, step2=dataclasses.replace(c.step2, **s2))
    return c


def _twin(ks, d=0.3, mu=1.0, cx=1.003, cy=-0.517, phi=0.1, theta=0.6, k_star=0):
    n = len(ks)
    z = np.zeros(n)
    return Twin(cell_id=1, node_id=2, set="near", role="train", d_um=d, mu_per_um=mu, k_star=k_star, z_sub_um=0.02,
                cx_um=cx, cy_um=cy, theta_rad=theta, phi_rad=phi, B_pilot_gl=200.0, d_pilot_um=d,
                alpha_pilot=mu * d, ks=np.asarray(ks, np.int64), B_gl=z + 200.0, noise_sd_gl=z + 2.0, omega_um2=z,
                se_omega_um2=z, area_gl_um=z, status=("ok",) * n)


def test_R2_trial_renderer_fields_and_knots():
    c = _cfg()
    s2 = c.step2
    s0 = float(c.renderer.sigma_r0_um)
    for stage, g in (("near", None), ("near", 0.7), ("far", 0.85)):
        r = SR.trial_renderer(c.renderer, s0, THETA, g, s2, stage)
        assert r.kernel_family == s2.kernel_family == "gaussian_table_spline"
        assert tuple(r.kernel_table_delta_um) == (0.0,) + tuple(s2.knots_um)
        assert tuple(r.kernel_table_sigma_um) == (s0,) + THETA and r.sigma_r0_um == s0
        assert r.absorption == s2.absorption == "hybrid"
        assert_allclose(kernel.sigma_r(np.array((0.0,) + tuple(s2.knots_um)), r), (s0,) + THETA, rtol=0, atol=1e-13)
        if stage == "near":
            assert r.kernel_continuation == "tangent"
        else:
            assert r.kernel_continuation == "linear" and r.kernel_continuation_slope == 0.85
            assert_allclose(kernel.sigma_r(0.84 + 2.0, r), THETA[-1] + 0.85 * 2.0, rtol=0, atol=1e-13)
        # every other renderer field unchanged
        keep = {f.name for f in dataclasses.fields(r)} - {"kernel_family", "kernel_table_delta_um",
                                                          "kernel_table_sigma_um", "sigma_r0_um", "absorption",
                                                          "kernel_continuation", "kernel_continuation_slope"}
        for name in keep:
            assert getattr(r, name) == getattr(c.renderer, name), name
        dataclasses.replace(c, renderer=r).validate()


@pytest.mark.parametrize("args", [
    ("mid", THETA, None), ("near", THETA[:2], None), ("near", (0.1, -0.2, 0.3), None),
    ("near", (0.1, float("nan"), 0.3), None), ("far", THETA, None), ("far", THETA, float("nan")),
    ("far", THETA, -0.1), ("far", THETA, float("inf"))])
def test_R2_refusals(args):
    stage, theta, gamma = args
    c = _cfg()
    with pytest.raises(ValueError):
        SR.trial_renderer(c.renderer, 0.08, theta, gamma, c.step2, stage)
    for s0 in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError):
            SR.trial_renderer(c.renderer, s0, THETA, 0.8, c.step2, "far")


def test_twin_tube_from_the_twin():
    c = _cfg()
    t = _twin(np.arange(-3, 4))
    tube = SR.twin_tube(t, 0.137, c, c.step2)
    assert isinstance(tube, geometry.Tube)
    assert_allclose(tube.c, (t.cx_um, t.cy_um, 0.137), rtol=0, atol=0)
    assert tube.r == t.d_um / 2 and tube.phi == t.phi_rad and tube.theta == t.theta_rad
    assert tube.half_length == c.step2.tube_half_length_um and tube.end_cut == c.renderer.end_cut
    assert tube.aspect == c.renderer.cross_section_aspect


def _render_pair(absorption, ks=np.arange(-3, 4), z_ax=0.05, half=HALF, **tw):
    c = _cfg(half=half, absorption=absorption)
    t = _twin(ks, **tw)
    r = SR.trial_renderer(c.renderer, c.renderer.sigma_r0_um, THETA, None, c.step2, "near")
    st = SR.render_twin(t, z_ax, r, c, c.step2)
    return c, t, r, st


@pytest.mark.parametrize("absorption", ["partition_vertical", "hybrid"])
def test_R1_render_twin_then_camera_equals_synthetic_block(absorption):
    c, t, r, st = _render_pair(absorption)
    p = c.acquisition.res0_um
    left, top, w, h = ref_patch_bounds(t.cx_um, t.cy_um, HALF, p)
    assert (st.left, st.top, st.tau.shape[1:]) == (left, top, (h, w))
    tube = geometry.Tube((t.cx_um, t.cy_um, 0.05), t.d_um / 2, t.phi_rad, t.theta_rad,
                         c.renderer.cross_section_aspect, c.step2.tube_half_length_um, c.renderer.end_cut)
    pad = int(math.ceil(c.renderer.pad_um / p))
    cfg_r = dataclasses.replace(c, renderer=r)
    ref, ks, valid, frame = render.synthetic_block(tube, t.mu_per_um, t.ks, left, top, w, h, cfg_r,
                                                   np.random.default_rng(11), pad)
    got = camera.camera_chain(st.tau, r, np.random.default_rng(11))
    assert_array_equal(got, ref)                                        # bit for bit
    assert_array_equal(st.ks, t.ks) and st.res0_um == p


def test_R3_R5_invariants_shapes_determinism():
    c, t, r, st = _render_pair("hybrid", half=5.5, d=0.3, mu=1.0, phi=0.0)
    assert st.tau.dtype == np.float64 and st.tau.shape[0] == t.ks.size and st.ks.dtype == np.int64
    assert np.all(st.tau > -0.1) and np.all(st.tau <= 1 + 1e-12)
    # far from the tube: pixels whose perpendicular distance to the axis' projection exceeds 5 um: at the outermost
    # plane sigma_r is about 0.6 um (tangent continuation past 0.84 um), so 5 um is > 8 sigma_r and the Gaussian
    # tail there is below 1e-14 of the dip: tau = 1 to the spec's 1e-12
    p = c.acquisition.res0_um
    H_, W_ = st.tau.shape[1:]
    yy, xx = np.mgrid[0:H_, 0:W_]
    X, Y = (st.left + xx) * p - t.cx_um, (st.top + yy) * p - t.cy_um
    dist = np.abs(-X * math.sin(t.theta_rad) + Y * math.cos(t.theta_rad))
    far = dist > 5.0
    assert far.sum() > 50
    assert np.max(np.abs(st.tau[:, far] - 1.0)) <= 1e-12
    # the dip on the twin's line deeper at focus than three planes away (z_ax 0.05 um: focus at k = 0)
    q0 = int(np.flatnonzero(t.ks == 0)[0])
    assert st.tau[q0].min() < st.tau[q0 + 3].min() and st.tau[q0].min() < st.tau[q0 - 3].min()
    st2 = SR.render_twin(t, 0.05, r, c, c.step2)
    assert st2.left == st.left and st2.top == st.top
    assert_array_equal(st2.tau, st.tau)                                 # determinism
    st3 = SR.render_twin(t, 0.05, r, c, c.step2, ks=np.array([0, 2]))   # ks given
    assert_array_equal(st3.ks, [0, 2])
    assert_allclose(st3.tau, st.tau[[q0, q0 + 2]], rtol=0, atol=1e-12)


@pytest.mark.parametrize("step", [0.02])
def test_R4_wider_theta3_widens_far_plane_only(step):
    # R4 with theta_3 larger by 0.02 um (4 x diff_step_knot_um). With a larger change (0.2 um) the focus plane moves
    # by 3e-3, beyond the spec's 1e-3: the not-a-knot spline is global (finding S-R4 of the report, and
    # test_R4_spline_is_not_local below)
    c = _cfg(absorption="partition_vertical")
    t = _twin(np.array([0, 3]), d=0.3, mu=1.0, phi=0.0)
    a = SR.render_twin(t, 0.0, SR.trial_renderer(c.renderer, c.renderer.sigma_r0_um, THETA, None, c.step2, "near"),
                       c, c.step2)
    b = SR.render_twin(t, 0.0, SR.trial_renderer(c.renderer, c.renderer.sigma_r0_um, THETA[:2] + (THETA[2] + step,), None,
                                                 c.step2, "near"), c, c.step2)
    # the focus plane's within 1e-3 (spec); three planes away (delta 0.84 um) the dip is wider: its deficit spreads
    # out, so the minimum of tau rises and the deficit's second moment about the axis grows
    assert np.max(np.abs(a.tau[0] - b.tau[0])) <= 1e-3
    assert b.tau[1].min() > a.tau[1].min()
    p = c.acquisition.res0_um
    H_, W_ = a.tau.shape[1:]
    yy, xx = np.mgrid[0:H_, 0:W_]
    X, Y = (a.left + xx) * p - t.cx_um, (a.top + yy) * p - t.cy_um
    v = -X * math.sin(t.theta_rad) + Y * math.cos(t.theta_rad)
    m2 = [float(np.sum((1 - s.tau[1]) * v ** 2) / np.sum(1 - s.tau[1])) for s in (a, b)]
    assert m2[1] > m2[0]


def test_R4_spline_is_not_local():
    # documents the observation behind the report's note: theta_3 (the knot at 0.84 um) moves sigma_r inside the
    # first interval [0, 0.28] um under the spline reading, not under the linear one; and the floor of D-275 is
    # active over [0, 0.15] um on this trial table (sigma_r = sigma_r(0) exactly there)
    r = C.default_config().renderer
    s0 = 0.08
    mk = lambda th, fam: dataclasses.replace(r, kernel_family=fam, kernel_table_delta_um=(0.0, 0.28, 0.56, 0.84),  # noqa: E731
                                             kernel_table_sigma_um=(s0,) + th, sigma_r0_um=s0,
                                             kernel_continuation="linear")
    d = np.array([0.05, 0.10, 0.15])
    a = kernel.sigma_r(d, mk((0.12, 0.33, 0.55), "gaussian_table_spline"))
    b = kernel.sigma_r(d, mk((0.12, 0.33, 0.75), "gaussian_table_spline"))
    assert_allclose(a, s0, rtol=0, atol=0)                               # floored
    assert np.all(b - a > 5e-4)                                          # moved by the 0.84 um knot
    la = kernel.sigma_r(d, mk((0.12, 0.33, 0.55), "gaussian_table"))
    lb = kernel.sigma_r(d, mk((0.12, 0.33, 0.75), "gaussian_table"))
    assert_array_equal(la, lb)
