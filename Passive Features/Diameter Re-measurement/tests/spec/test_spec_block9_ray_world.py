"""Block 9 (ray world): oracles from SPEC.md Block 9 (S6), (S10)."""
import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from allen_diameter.config import default_config
from allen_diameter.model import geometry as G, ray_world as RW, render

R = default_config().renderer


def test_directions_and_mean_inverse_cos():
    S, W = RW.directions(R.ray_n_rho, R.ray_n_psi, R.ray_na, R.ray_n_oil)
    sm = R.ray_na / R.ray_n_oil
    assert S.shape == (R.ray_n_rho * R.ray_n_psi, 3) and W.shape == (S.shape[0],)
    assert_allclose(W.sum(), 1.0, rtol=1e-14)
    assert_allclose(np.linalg.norm(S, axis=1), 1.0, rtol=1e-14)
    assert np.all(S[:, 2] > 0) and np.all(np.hypot(S[:, 0], S[:, 1]) <= sm + 1e-14)
    exact = (2 / sm ** 2) * (1 - math.sqrt(1 - sm ** 2))
    assert_allclose(RW.mean_inverse_cos(S, W), exact, rtol=1e-10)
    assert_allclose(exact, 1.4470, atol=5e-5)
    # the quadrature is exact for <1/cos> (a polynomial-free but smooth function): check against W @ 1/S_z
    assert_allclose(RW.mean_inverse_cos(S, W), float(W @ (1 / S[:, 2])), rtol=1e-14)


def test_ray_transmittance_bounds_and_faint_chord():
    S, W = RW.directions(R.ray_n_rho, R.ray_n_psi, R.ray_na, R.ray_n_oil)
    tube = G.Tube((0, 0, 0), 0.5, 0.0, 0.0, 1.0, 5.0)
    pts = np.array([[0, 0, 0], [0, 0.3, 0.2], [0, 2.0, 0.0]], float)
    t = RW.ray_transmittance(pts, tube, 0.8, S, W)
    assert np.all((t >= 0) & (t <= 1 + 1e-15)) and abs(t[2] - 1.0) < 1e-15   # weights sum to 1 up to rounding
    # faint limit: 1 - T ~ mu <L>; through the axis, L = 2 r / sqrt(1 - s_y^2) for a tube along x
    mu = 1e-6
    t0 = RW.ray_transmittance(pts[:1], tube, mu, S, W)[0]
    L = 2 * 0.5 / np.sqrt(1 - S[:, 1] ** 2)
    assert_allclose((1 - t0) / mu, float(W @ L), rtol=1e-5)


def test_flat_tube_planes_symmetric_and_no_end_cut_refused():
    rc = dataclasses.replace(R, absorption="ray_world", ray_n_rho=6, ray_n_psi=8)
    tube = G.Tube((0, 0, 0), 0.4, 0.0, 0.3, 1.0, 1.0)
    grid = render.FineGrid(-1.2, -1.2, 0.1144 / 2, 42, 42)
    out = render.render_transmittance(tube, 1.0, [-0.84, 0.84], grid, rc)
    assert out.backend == "ray_world" and out.tau.shape == (2, 42, 42) and np.all(out.tau <= 1)
    assert_allclose(out.tau[0], out.tau[1], atol=1e-12)
    with pytest.raises(ValueError):
        render.render_transmittance(G.Tube((0, 0, 0), 0.4, 0.2, 0.3, 1.0, None), 1.0, [0.0], grid, rc)
    with pytest.raises(ValueError):
        render.absorbed_fractions(tube, 1.0, grid, np.array([0.0]), 0.02, +1, "ray_world")
