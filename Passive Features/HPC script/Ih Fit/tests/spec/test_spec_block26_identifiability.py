"""Spec-derived tests of Block 26 (identifiability.py, trace_sensitivity.py; SPEC
2.10 (d) and the Block 26 entry). Integration tester, 2026-11-25.

Oracles: 2.10 (d)'s formulas recomputed with numpy on hand-built layouts; the
normal equations (J^T J)^{-1} J^T for the linearised displacement; the sandwich
covariance A^{-1} J^T Sigma J A^{-1} for the predicted scatter under correlated,
heteroscedastic noise (Monte Carlo, fixed seed); analytic derivative traces of a
smooth toy model through the chain rule of the unit coordinates.
"""
import json
import types

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import identifiability as IDF
import trace_sensitivity as TS


def _layout(lengths, n_pre, win, seed=0):
    """A hand-built layout of the duck type: per bundle n_t samples, the first n_pre
    the pre-window, `win` = (start, stop) indices of the window; random weights."""
    rng = np.random.default_rng(seed)
    masks, pre, sqa, slices = [], [], [], []
    start = 0
    for n_t in lengths:
        m = np.zeros(n_t, bool); m[win[0]:win[1]] = True
        pm = np.zeros(n_t, bool); pm[:n_pre] = True
        k = int(m.sum())
        masks.append(m); pre.append(pm); sqa.append(rng.uniform(0.05, 0.4, k))
        slices.append(slice(start, start + k)); start += k
    D = rng.uniform(2.0, 9.0, len(lengths))
    return types.SimpleNamespace(masks=masks, pre_masks=pre, sqrt_a=sqa, D=D, slices=slices, n_residuals=start)


def test_jacobian_from_sensitivities_formula():
    lay = _layout([40, 55, 30], 8, (10, 28))
    rng = np.random.default_rng(1)
    S = [rng.normal(size=(n, 4)) for n in (40, 55, 30)]
    J = IDF.jacobian_from_sensitivities(S, lay)
    exp = np.vstack([lay.sqrt_a[b][:, None] * S[b][lay.masks[b]] / lay.D[b] for b in range(3)])
    assert_allclose(J, exp, rtol=0, atol=0)
    assert J.shape == (lay.n_residuals, 4)


def test_svd_summary_and_row():
    rng = np.random.default_rng(2)
    J = rng.normal(size=(30, 4)) @ np.diag([5.0, 1.0, 0.1, 0.01])
    s = IDF.svd_summary(J)
    U, sig, Vt = np.linalg.svd(J, full_matrices=False)
    assert_allclose(s["sigma"], sig, rtol=1e-12)
    assert np.all(np.diff(s["sigma"]) <= 0)
    assert_allclose(np.abs(s["V"]), np.abs(Vt), rtol=1e-10, atol=1e-12)
    for j in range(4):               # sign: the largest-magnitude component positive
        assert s["V"][j, np.argmax(np.abs(s["V"][j]))] > 0
        assert_allclose(np.linalg.norm(J @ s["V"][j]), sig[j], rtol=1e-10)
    assert_allclose(s["ratio"], sig[-1] / sig[0], rtol=1e-12)
    names = ["a", "b", "c", "d"]
    row = IDF.identifiability_row(J, names)
    assert [row["sigma_%d" % k] for k in range(1, 5)] == [float(x) for x in s["sigma"]]
    assert [row["v_p_" + n] for n in names] == [float(x) for x in s["V"][3]]
    assert [row["v_pm1_" + n] for n in names] == [float(x) for x in s["V"][2]]
    assert row["sigma_ratio"] == s["ratio"]
    assert not any(k.startswith("sd_lin_") or k.startswith("corr_") for k in row)


def test_collinear_pair_is_detected():
    """2.10 (d) / I1: a model depending on g e^{dv/k} only -> the smallest singular value
    vanishes and v_p is proportional to (1, -k) on (ln g, dv)."""
    k = 8.05
    t = np.linspace(0, 1, 200)
    # residual Jacobian columns: d/d ln g and d/d dv of A(t) * g e^{dv/k}, A arbitrary
    A = np.exp(-t / 0.3)
    base = A * 1.7
    J = np.column_stack([base, base / k, np.sin(5 * t)])
    row = IDF.identifiability_row(J, ["lng", "dv", "x"])
    assert row["sigma_ratio"] < 1e-8
    v = np.array([row["v_p_lng"], row["v_p_dv"], row["v_p_x"]])
    ref = np.array([1.0, -k, 0.0]) / np.hypot(1.0, k)
    assert abs(abs(v @ ref) - 1.0) < 1e-8


def test_noise_residuals_formula_sign_and_offset():
    lay = _layout([40, 50], 8, (10, 30), seed=3)
    rng = np.random.default_rng(4)
    eta = [rng.normal(size=40), rng.normal(size=50)]
    r = IDF.noise_residuals(lay, eta)
    exp = np.concatenate([-lay.sqrt_a[b] * (eta[b] - eta[b][:8].mean())[lay.masks[b]] / lay.D[b]
                          for b in range(2)])
    assert_allclose(r, exp, rtol=1e-14, atol=1e-16)
    r2 = IDF.noise_residuals(lay, [eta[0] + 3.7, eta[1] - 1.1])
    assert_allclose(r2, r, rtol=0, atol=1e-14)            # a constant does not enter
    with pytest.raises(ValueError):
        IDF.noise_residuals(lay, [eta[0]])


def test_predicted_scatter_is_the_least_squares_displacement():
    rng = np.random.default_rng(5)
    J = rng.normal(size=(60, 3))
    R = rng.normal(size=(50, 60))
    sc = IDF.predicted_scatter(J, R)
    dq = -np.linalg.solve(J.T @ J, J.T @ R.T).T
    assert_allclose(sc["dq"], dq, rtol=1e-10, atol=1e-12)
    assert_allclose(sc["sd"], dq.std(axis=0, ddof=1), rtol=1e-10)
    assert_allclose(sc["corr"], np.corrcoef(dq, rowvar=False), rtol=1e-10, atol=1e-12)


def test_predicted_scatter_matches_the_sandwich_covariance():
    """Correlated, heteroscedastic residual noise (an AR(1) series with a rising SD):
    the predicted SD over K realisations estimates sqrt(diag(A^-1 J^T Sigma J A^-1)).
    With K = 400 the relative sampling error of an SD is about 1/sqrt(2K) = 3.5 %;
    0.15 is about 4 such errors."""
    rng = np.random.default_rng(6)
    N, p, K = 120, 3, 400
    t = np.linspace(0, 1, N)
    J = np.column_stack([np.exp(-t / 0.2), t, np.cos(3 * t)])
    rho = 0.8
    sd = 0.5 + 2.0 * t
    C = rho ** np.abs(np.subtract.outer(np.arange(N), np.arange(N)))
    Sigma = np.outer(sd, sd) * C
    L = np.linalg.cholesky(Sigma)
    R = (L @ rng.normal(size=(N, K))).T
    sc = IDF.predicted_scatter(J, R)
    Ainv = np.linalg.inv(J.T @ J)
    cov = Ainv @ J.T @ Sigma @ J @ Ainv
    assert_allclose(sc["sd"], np.sqrt(np.diag(cov)), rtol=0.15)
    naive = np.sqrt(np.diag(Ainv) * np.mean(np.diag(Sigma)))
    assert np.max(np.abs(naive / np.sqrt(np.diag(cov)) - 1)) > 0.3   # the check can tell them apart


def test_predicted_scatter_refusals():
    J = np.ones((10, 2))                   # rank 1
    with pytest.raises(ValueError):
        IDF.predicted_scatter(J, np.random.default_rng(0).normal(size=(5, 10)))
    Jg = np.random.default_rng(1).normal(size=(10, 2))
    with pytest.raises(ValueError):
        IDF.predicted_scatter(Jg, np.zeros((1, 10)))       # K < 2
    with pytest.raises(ValueError):
        IDF.predicted_scatter(Jg, np.zeros((3, 9)))        # shapes
    with pytest.raises(ValueError):
        IDF.predicted_scatter(Jg, np.full((3, 10), np.nan))


def test_identifiability_row_with_noise_columns():
    rng = np.random.default_rng(7)
    J = rng.normal(size=(40, 3))
    R = rng.normal(size=(30, 40))
    row = IDF.identifiability_row(J, ["Cm", "Rm", "gbar"], R)
    sc = IDF.predicted_scatter(J, R)
    for i, n in enumerate(["Cm", "Rm", "gbar"]):
        assert row["sd_lin_" + n] == float(sc["sd"][i])
    assert set(k for k in row if k.startswith("corr_")) == {"corr_Cm_Rm", "corr_Cm_gbar", "corr_Rm_gbar"}
    assert row["corr_Cm_gbar"] == float(sc["corr"][0, 2])


def test_relative_difference():
    a = np.array([[3.0, 0.0], [0.0, 4.0]])
    b = np.array([[3.0, 0.0], [0.0, 0.0]])
    assert IDF.relative_difference(a, b) == pytest.approx(4.0 / 3.0, rel=1e-15)
    with pytest.raises(ValueError):
        IDF.relative_difference(a, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        IDF.relative_difference(a, np.zeros((3, 2)))


# ------------------------------------------------------------- trace_sensitivity
BOUNDS = [(np.log(0.3), np.log(3.0)), (-10.0, 10.0), (np.log(50.0), np.log(1e3))]
T_LIST = [np.linspace(-0.01, 0.1, 60), np.linspace(0.0, 0.9, 80)]


def _traces(q):
    a, b, c = q
    return [np.exp(a) * np.exp(-T_LIST[0] / np.exp(c) * 50.0) + 0.1 * b * T_LIST[0],
            np.sin(a + 0.1 * b) * T_LIST[1] ** 2 + np.exp(c - 6.0) * T_LIST[1]]


def _dtraces(q):
    a, b, c = q
    t0, t1 = T_LIST
    e = np.exp(-t0 / np.exp(c) * 50.0)
    S0 = np.column_stack([np.exp(a) * e, 0.1 * t0, np.exp(a) * e * (t0 * 50.0 / np.exp(c))])
    S1 = np.column_stack([np.cos(a + 0.1 * b) * t1 ** 2, 0.1 * np.cos(a + 0.1 * b) * t1 ** 2,
                          np.exp(c - 6.0) * t1])
    return [S0, S1]


class Ev:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def __call__(self, Q):
        Q = np.asarray(Q, float)
        self.calls.append(Q.copy())
        out = [_traces(q) for q in Q]
        if self.fail_at is not None:
            out[self.fail_at] = None
        return out


def test_compute_sensitivities_against_the_analytic_derivatives():
    q = np.array([np.log(1.1), 2.5, np.log(200.0)])
    ev = Ev()
    s = TS.compute_sensitivities(ev, q, BOUNDS, h=1e-3, scheme="central")
    assert len(ev.calls) == 1                         # one evaluator call, base point included
    assert_array_equal(ev.calls[0][0], q)
    assert ev.calls[0].shape == (1 + 2 * 3, 3) and s["n_points"] == 7
    for b in range(2):
        assert_array_equal(s["traces0"][b], _traces(q)[b])
        exact = _dtraces(q)[b]
        # central difference in u with h = 1e-3: relative error O(h^2 (hi-lo)^2) ~ 1e-4
        assert_allclose(s["S"][b], exact, rtol=2e-4, atol=1e-6 * np.max(np.abs(exact)))
        assert s["S"][b].shape == (T_LIST[b].size, 3)
    assert s["scheme"] == "central" and s["h"] == 1e-3
    assert_array_equal(s["q"], q)


def test_compute_sensitivities_on_a_face_and_failures():
    q = np.array([BOUNDS[0][0], 0.0, np.log(200.0)])        # axis 0 on its lower face
    ev = Ev()
    s = TS.compute_sensitivities(ev, q, BOUNDS)
    assert len(ev.calls) == 1 and ev.calls[0].shape[0] == 1 + 1 + 2 + 2
    for b in range(2):
        exact = _dtraces(q)[b]
        # axis 0 forward: O(h (hi-lo)) ~ 2e-3 relative; the others central
        assert_allclose(s["S"][b][:, 0], exact[:, 0], rtol=5e-3, atol=1e-6)
        assert_allclose(s["S"][b][:, 1:], exact[:, 1:], rtol=2e-4, atol=1e-6)
    with pytest.raises(RuntimeError):
        TS.compute_sensitivities(Ev(fail_at=3), np.array([0.1, 0.0, 5.0]), BOUNDS)
    with pytest.raises(RuntimeError):
        TS.compute_sensitivities(Ev(fail_at=0), np.array([0.1, 0.0, 5.0]), BOUNDS)
    with pytest.raises(ValueError):
        TS.compute_sensitivities(Ev(), np.array([5.0, 0.0, 5.0]), BOUNDS)   # outside the box


def test_save_load_round_trip(tmp_path):
    q = np.array([np.log(1.1), 2.5, np.log(200.0)])
    s = TS.compute_sensitivities(Ev(), q, BOUNDS)
    meta = {"specimen_id": 7, "names": ["Cm", "x", "Ra"], "nested": {"a": [1, 2]}}
    p = tmp_path / "sens.npz"
    TS.save_sensitivities(p, s, t_list=T_LIST, names=["Cm", "x", "Ra"], meta=meta)
    with np.load(p, allow_pickle=False) as z:            # no object arrays
        assert set(z.files) >= {"q", "h", "scheme", "names", "meta", "t_0", "V0_0", "S_0", "t_1", "V0_1", "S_1"}
    d = TS.load_sensitivities(p)
    assert_array_equal(d["q"], q)
    for b in range(2):
        assert_array_equal(d["S"][b], s["S"][b])
        assert_array_equal(d["traces0"][b], s["traces0"][b])
        assert_array_equal(d["t_list"][b], T_LIST[b])
    assert d["names"] == ["Cm", "x", "Ra"] and d["meta"] == meta and d["scheme"] == "central"
