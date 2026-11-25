"""Spec-derived tests of Block 24 (fd_jacobian.py; SPEC 2.10 (b) and the Block 24
entry). Integration tester, 2026-11-25.

Oracles: 2.10 (b) written out by hand (forward with a flipped step on the upper
face, central with the per-axis forward fallback, one batch call), scipy's own
finite differences (`scipy.optimize._numdiff.approx_derivative` with an absolute
step) as the reference implementation, analytic Jacobians and the chain rule.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize._numdiff import approx_derivative

import fd_jacobian as FJ


def _lin(A, b):
    def batch(U):
        U = np.asarray(U, dtype=float)
        return U @ A.T + b
    return batch


def _counting(fun):
    calls = []

    def batch(U):
        calls.append(np.array(U, copy=True))
        return np.array([fun(u) for u in U])
    return batch, calls


def _smooth(u):
    # a smooth nonlinear map R^3 -> R^4
    return np.array([np.sin(u[0]) * u[1], np.exp(u[2] - u[0]), u[0] ** 3 + u[1] * u[2],
                     np.log1p(u[1] + 2.0 * u[2])])


def _smooth_jac(u):
    return np.array([[np.cos(u[0]) * u[1], np.sin(u[0]), 0.0],
                     [-np.exp(u[2] - u[0]), 0.0, np.exp(u[2] - u[0])],
                     [3 * u[0] ** 2, u[2], u[1]],
                     [0.0, 1.0 / (1 + u[1] + 2 * u[2]), 2.0 / (1 + u[1] + 2 * u[2])]])


@pytest.mark.parametrize("scheme", ["forward", "central"])
@pytest.mark.parametrize("u", [[0.3, 0.6, 0.45], [0.0, 1.0, 0.5], [1.0, 0.0, 0.9995], [0.0005, 0.5, 1.0]])
def test_linear_map_exact(scheme, u):
    rng = np.random.default_rng(3)
    A = rng.normal(size=(5, 3))
    b = rng.normal(size=5)
    r = FJ.fd_jacobian(_lin(A, b), u, h=1e-3, scheme=scheme)
    # a linear map is differentiated exactly by any difference rule; 1e-9 relative
    # leaves room only for the roundoff of (f(u+h) - f(u)) / h at h = 1e-3 (~1e-13 / 1e-3).
    assert_allclose(r.J, A, rtol=1e-9, atol=1e-9)
    assert r.J.shape == (5, 3) and r.J.dtype == np.float64


def test_points_and_plan_written_by_hand():
    u = np.array([0.2, 0.9995, 0.0])
    U, plan = FJ.fd_points(u, h=1e-3, scheme="forward")
    exp = np.array([[0.201, 0.9995, 0.0], [0.2, 0.9985, 0.0], [0.2, 0.9995, 0.001]])
    assert_allclose(U, exp, rtol=0, atol=1e-15)
    assert [p[0] for p in plan] == ["forward"] * 3
    assert [p[2] for p in plan] == [1e-3, -1e-3, 1e-3]          # flipped on the upper face only
    U, plan = FJ.fd_points(u, h=1e-3, scheme="central")
    # axis 0 two-sided (2 rows: plus then minus), axis 1 forward with -h (u + h > 1), axis 2 forward +h (u - h < 0)
    exp = np.array([[0.201, 0.9995, 0.0], [0.199, 0.9995, 0.0], [0.2, 0.9985, 0.0], [0.2, 0.9995, 0.001]])
    assert_allclose(U, exp, rtol=0, atol=1e-15)
    assert plan[0] == ("central", 0, 1, 1e-3)
    assert plan[1][0] == "forward" and plan[1][1] == 2 and plan[1][2] == -1e-3
    assert plan[2][0] == "forward" and plan[2][1] == 3 and plan[2][2] == 1e-3


def test_forward_against_scipy_absolute_step():
    u = np.array([0.31, 0.47, 0.62])
    batch, calls = _counting(_smooth)
    r = FJ.fd_jacobian(batch, u, h=1e-3, scheme="forward")
    ref = approx_derivative(_smooth, u, method="2-point", abs_step=1e-3)
    # same rule, same absolute step; scipy divides by the representable step
    # (u + h) - u, the code by h: they differ by ~1e-13 relative -> 1e-9.
    assert_allclose(r.J, ref, rtol=1e-9, atol=1e-12)
    # and both are within O(h) of the analytic Jacobian (second derivatives O(1): 2e-3 abs)
    assert_allclose(r.J, _smooth_jac(u), atol=2e-3)
    assert len(calls) == 1 and calls[0].shape == (4, 3)
    assert_array_equal(calls[0][0], u)                       # u first when f0 is not given
    assert_allclose(r.f0, _smooth(u), rtol=0, atol=0)
    assert r.n_points == 4


def test_central_against_scipy_and_analytic():
    u = np.array([0.31, 0.47, 0.62])
    batch, calls = _counting(_smooth)
    r = FJ.fd_jacobian(batch, u, h=1e-3, scheme="central")
    ref = approx_derivative(_smooth, u, method="3-point", abs_step=1e-3)
    assert_allclose(r.J, ref, rtol=1e-8, atol=1e-11)
    # central error O(h^2) = 1e-6 with O(1) third derivatives
    assert_allclose(r.J, _smooth_jac(u), atol=5e-6)
    # every axis two-sided: f0 not needed, u not asked
    assert len(calls) == 1 and calls[0].shape == (6, 3)
    assert not any(np.array_equal(row, u) for row in calls[0])
    assert r.f0 is None and r.n_points == 6


def test_upper_face_forward_sign():
    u = np.array([0.4, 1.0, 0.5])
    r = FJ.fd_jacobian(_counting(_smooth)[0], u, h=1e-3, scheme="forward")
    # backward difference at u_1 = 1 is the left derivative: O(h) from the analytic one
    assert_allclose(r.J[:, 1], _smooth_jac(u)[:, 1], atol=2e-3)
    ref = approx_derivative(_smooth, u, method="2-point", abs_step=1e-3, bounds=(0.0, 1.0))
    assert_allclose(r.J, ref, rtol=1e-9, atol=1e-12)


def test_f0_given_not_asked_and_returned():
    u = np.array([0.31, 0.47, 0.62])
    batch, calls = _counting(_smooth)
    f0 = _smooth(u)
    r = FJ.fd_jacobian(batch, u, h=1e-3, scheme="forward", f0=f0)
    assert len(calls) == 1 and calls[0].shape == (3, 3)
    assert not any(np.array_equal(row, u) for row in calls[0])
    assert_array_equal(r.f0, f0)
    # the derivative uses the given f0: shifting f0 shifts J by -shift / h
    r2 = FJ.fd_jacobian(_counting(_smooth)[0], u, h=1e-3, scheme="forward", f0=f0 + 1.0)
    assert_allclose(r2.J, r.J - 1.0 / 1e-3, rtol=1e-9)


def test_central_with_a_one_sided_axis_asks_u_once():
    u = np.array([0.5, 0.0, 0.5])
    batch, calls = _counting(_smooth)
    r = FJ.fd_jacobian(batch, u, h=1e-3, scheme="central")
    assert len(calls) == 1 and calls[0].shape == (1 + 2 + 1 + 2, 3)
    assert_array_equal(calls[0][0], u)
    assert_allclose(r.J[:, 0], _smooth_jac(u)[:, 0], atol=5e-6)
    assert_allclose(r.J[:, 1], _smooth_jac(u)[:, 1], atol=2e-3)


def test_coordinates_and_chain_rule():
    bounds = [(np.log(0.3), np.log(3.0)), (-10.0, 10.0)]
    rng = np.random.default_rng(1)
    U = rng.random((7, 2))
    Q = FJ.to_q(U, bounds)
    assert_allclose(FJ.to_unit(Q, bounds), U, rtol=0, atol=1e-15)
    assert_allclose(FJ.to_q(U[0], bounds), Q[0])
    # clipping into the box
    assert_allclose(FJ.to_q([-0.5, 1.5], bounds), [bounds[0][0], bounds[1][1]])
    # chain rule: g(q) = (exp(q0) * q1,) ; dg/du_i = dg/dq_i * (hi - lo)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    u = np.array([0.4, 0.7])
    q = lo + u * (hi - lo)
    J_u = np.array([[np.exp(q[0]) * q[1] * (hi[0] - lo[0]), np.exp(q[0]) * (hi[1] - lo[1])]])
    assert_allclose(FJ.jac_unit_to_q(J_u, bounds), [[np.exp(q[0]) * q[1], np.exp(q[0])]], rtol=1e-14)


@pytest.mark.parametrize("bad", [
    dict(u=[0.5, 1.2]), dict(u=[-1e-9, 0.5]), dict(u=[np.nan, 0.5]), dict(u=[[0.5, 0.5]]),
    dict(h=0.0), dict(h=0.6), dict(h=np.nan), dict(scheme="backward"),
])
def test_refusals(bad):
    kw = dict(u=[0.5, 0.5], h=1e-3, scheme="forward")
    kw.update(bad)
    u = kw.pop("u")
    with pytest.raises(ValueError):
        FJ.fd_jacobian(_lin(np.eye(2), np.zeros(2)), u, **kw)


def test_refusals_of_batch_and_bounds():
    with pytest.raises(ValueError):          # wrong number of rows
        FJ.fd_jacobian(lambda U: np.zeros((1, 3)), [0.5, 0.5])
    with pytest.raises(ValueError):          # 1-D return
        FJ.fd_jacobian(lambda U: np.zeros(U.shape[0]), [0.5, 0.5])
    with pytest.raises(ValueError):          # non-finite
        FJ.fd_jacobian(lambda U: np.full((U.shape[0], 2), np.inf), [0.5, 0.5])
    with pytest.raises(ValueError):          # f0 of the wrong length
        FJ.fd_jacobian(lambda U: np.zeros((U.shape[0], 2)), [0.5, 0.5], f0=np.zeros(3))
    with pytest.raises(ValueError):
        FJ.to_unit([0.1], [(1.0, 1.0)])
    with pytest.raises(ValueError):
        FJ.to_q([0.1], [(2.0, 1.0)])
