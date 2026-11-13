"""Spec-derived tests of Block 9 (the ray-world generator) -- integration
tester, run D1, 2026-11-13. Expected values: SPEC Block 9's quadrature
written out with numpy.polynomial.legendre.leggauss, the (S10) closed form,
and the chord of a round capped tube from its quadratic (spec_helpers_D1);
the code under test only produces the outputs that are checked.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block9_ray_world.py -q
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import spec_helpers_D1 as H

from allen_diameter.config import default_config
from allen_diameter.model import ray_world as RW
from allen_diameter.model.geometry import Tube
from allen_diameter.model.render import FineGrid

RC = default_config().renderer


def test_directions_equal_the_spec_quadrature():
    S, W = RW.directions(RC.ray_n_rho, RC.ray_n_psi, RC.ray_na, RC.ray_n_oil)
    Sr, Wr = H.ref_directions(RC.ray_n_rho, RC.ray_n_psi, RC.ray_na, RC.ray_n_oil)
    assert S.shape == (RC.ray_n_rho * RC.ray_n_psi, 3) and W.shape == (S.shape[0],)
    # as sets of (direction, weight): the order of the nodes is not specified
    A = H.sort_rows(np.column_stack([S, W]))
    B = H.sort_rows(np.column_stack([Sr, Wr]))
    assert_allclose(A, B, rtol=0, atol=1e-14)   # roundoff of sqrt/cos/sin
    assert_allclose(np.linalg.norm(S, axis=1), 1.0, rtol=0, atol=1e-15)
    assert np.all(S[:, 2] > 0)
    assert abs(W.sum() - 1.0) < 1e-14
    # the transverse parts lie inside the condenser's disc of radius s_m
    sm = RC.ray_na / RC.ray_n_oil
    assert np.all(np.hypot(S[:, 0], S[:, 1]) <= sm + 1e-15)


def test_mean_inverse_cos_matches_S10():
    S, W = RW.directions(RC.ray_n_rho, RC.ray_n_psi, RC.ray_na, RC.ray_n_oil)
    exact = H.ref_mean_inverse_cos(RC.ray_na, RC.ray_n_oil)
    assert abs(exact - 1.4470) < 5e-5            # the spec's 4-digit value
    assert abs(RW.mean_inverse_cos(S, W) - exact) < 1e-10
    # and it is <1/s_z>_W of the returned set
    assert abs(RW.mean_inverse_cos(S, W) - float(np.sum(W / S[:, 2]))) < 1e-14


@pytest.mark.parametrize("phi_deg,theta", [(0.0, 0.0), (20.0, 0.7), (40.0, -2.0)])
def test_ray_transmittance_equals_chord_average(phi_deg, theta):
    rng = np.random.default_rng(int(phi_deg) + 3)
    c = np.array([0.3, -0.2, 0.1])
    r, U, mu = 0.45, 2.0, 1.7
    tube = Tube(tuple(c), r, math.radians(phi_deg), theta, 1.0, U, "axial")
    S, W = RW.directions(8, 16, RC.ray_na, RC.ray_n_oil)
    P = c + rng.uniform(-1.5, 1.5, size=(400, 3)) * np.array([1.0, 1.0, 0.8])
    got = RW.ray_transmittance(P, tube, mu, S, W)
    L = np.zeros((P.shape[0], S.shape[0]))
    for m in range(S.shape[0]):
        t1, t2, hit = H.ref_line_tube(P, S[m], c, r, math.radians(phi_deg), theta, U)
        L[:, m] = np.where(hit, t2 - t1, 0.0)
    exp = np.exp(-mu * L) @ W
    assert_allclose(got, exp, rtol=0, atol=1e-12)   # closed forms, roundoff only
    assert np.all(got <= 1.0 + 1e-15)


def _grid(n=36, h=0.06):
    return FineGrid(-n * h / 2, -n * h / 2, h, n, n)


def test_render_planes_refuses_an_uncapped_tube():
    tube = Tube((0.0, 0.0, 0.0), 0.3, 0.2, 0.0, 1.0, None, "axial")
    rc = dataclasses.replace(RC, ray_n_rho=4, ray_n_psi=8)
    with pytest.raises(ValueError):
        RW.render_planes(tube, 1.0, [0.0], _grid(), rc)


def test_flat_tube_planes_mirror_with_even_n_psi():
    """Block 9: a tube mirror-symmetric about z = c_z (phi = 0) gives equal
    planes at c_z -+ delta when ray_n_psi is even; tau <= 1; light does not
    enter (no argument)."""
    rc = dataclasses.replace(RC, ray_n_rho=6, ray_n_psi=12)
    tube = Tube((0.0, 0.0, 0.4), 0.35, 0.0, 0.3, 1.0, 1.5, "axial")
    z = [0.4 - 0.84, 0.4, 0.4 + 0.84]
    tau = np.asarray(RW.render_planes(tube, 2.0, z, _grid(), rc))
    assert tau.shape == (3, 36, 36)
    assert_allclose(tau[0], tau[2], rtol=0, atol=1e-12)
    assert np.all(tau <= 1.0 + 1e-15) and np.all(np.isfinite(tau))
    assert tau[1].min() < tau[0].min()   # the node plane is the darkest here
