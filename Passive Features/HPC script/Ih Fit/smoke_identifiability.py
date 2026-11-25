"""Smoke test for the sensitivities and the identifiability table -- Block 26 in
specs/SPEC.md (I1-I8), SPEC 2.10 (d).

Pure: no NEURON. The model is the toy of the review's smoke_sensitivity_reference.py
(five parameters; six with dv entering only through g exp(dv / K)), reproduced here,
on a hand-built layout: every bundle has N_PRE pre-window samples (t < 0, where the
toy is 0) and the 400 samples of the reference's grid T (t >= 0) as its window W_b;
the weights a_{b,s} and the deflections D_b are the reference's data_weights.
The Block 23 evaluator is a plain callable of its form: ev(Q) -> list per point of
per-bundle traces (minus their own pre-window mean), or None.

Block 24 (fd_jacobian.py) is written in parallel. When its functions still raise
NotImplementedError, the module's functions are replaced, before the checks, by a
private reference implementation of its contract (_ref_* below); trace_sensitivity.py
always calls the module, so the same checks run unchanged on the real Block 24.

Checks
    test_I1_null_direction    six-parameter toy, exact (complex-step) sensitivities:
                              sigma_ratio < 1e-8, |cos(v_p, (1, -K) on (ln g, dv))| > 1 - 1e-8;
                              five-parameter toy: sigma_ratio > 1e-3; svd_summary's sign rule
                              and identifiability_row's keys
    test_I2_predicted_scatter predicted_scatter (K = 200) against 200 noisy fits by
                              scipy.optimize.least_squares (white noise over the whole sweep,
                              data minus its pre-window mean, weights fixed from the clean
                              trace): empirical / predicted sd in (0.8, 1.25) for every
                              parameter; the naive s^2 (J^T J)^{-1} outside that band for at
                              least one parameter (planted defect)
    test_I3_sensitivities     compute_sensitivities on the toy evaluator: S within 1e-4 relative
                              of the exact derivative traces; one evaluator call; n_points;
                              a None point raises RuntimeError
    test_I4_jacobian          jacobian_from_sensitivities == fd_jacobian of the toy's residual
                              vector (1e-6 relative), interior and with one axis on a face
    test_I5_noise_residuals   a constant added to a bundle's noise changes nothing; the sign:
                              r_eta equals the residual at the truth of data = clean + noise
    test_I6_save_load         round trip, np.array_equal, allow_pickle=False
    test_I7_reldiff_refusals  relative_difference hand case; ValueError / RuntimeError refusals
    test_I8_ascii_lf          the three files of the block are ASCII with LF line endings

Run (from this folder)
    python smoke_identifiability.py [--only I1,I3]
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit code 1 if any check failed, errored or is still TODO.

Debug
    A failed check_close() saves both arrays to smoke_artifacts/<check name>.npz.

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import re
import sys
import tempfile
import time
import traceback
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                         # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402
from scipy.optimize import least_squares   # noqa: E402

import fd_jacobian as FJ                   # noqa: E402
import identifiability as ID               # noqa: E402
import trace_sensitivity as TS             # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = HERE / "smoke_artifacts"
_SAVED_ARTIFACTS = []
FD_SOURCE = "fd_jacobian.py (Block 24)"


def check_close(name, actual, expected, rtol=0.0, atol=0.0):
    """assert_allclose that keeps the evidence (smoke_artifacts/<name>.npz) on failure."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / ("%s.npz" % re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


# ------------------------------------------ reference of Block 24's contract ---
def _ref_to_unit(q, bounds):
    b = np.asarray(bounds, dtype=np.float64)
    return (np.asarray(q, dtype=np.float64) - b[:, 0]) / (b[:, 1] - b[:, 0])


def _ref_to_q(u, bounds):
    b = np.asarray(bounds, dtype=np.float64)
    return np.clip(b[:, 0] + np.asarray(u, dtype=np.float64) * (b[:, 1] - b[:, 0]),
                   b[:, 0], b[:, 1])


def _ref_fd_points(u, *, h=1e-3, scheme="forward"):
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 1 or not np.all(np.isfinite(u)) or np.any(u < 0) or np.any(u > 1):
        raise ValueError("u")
    if not (0.0 < h <= 0.5):
        raise ValueError("h")
    if scheme not in ("forward", "central"):
        raise ValueError("scheme")
    rows, plan = [], []
    for i in range(u.shape[0]):
        e = np.zeros_like(u)
        e[i] = 1.0
        if scheme == "central" and not (u[i] + h > 1.0 or u[i] - h < 0.0):
            plan.append(("central", len(rows), len(rows) + 1, h))
            rows.append(u + h * e)
            rows.append(u - h * e)
        else:
            delta = -h if u[i] + h > 1.0 else h
            plan.append(("forward", len(rows), delta))
            rows.append(u + delta * e)
    return np.array(rows, dtype=np.float64), plan


def _ref_fd_jacobian(batch_fun, u, *, h=1e-3, scheme="forward", f0=None):
    u = np.asarray(u, dtype=np.float64)
    U, plan = _ref_fd_points(u, h=h, scheme=scheme)
    need_f0 = any(pl[0] == "forward" for pl in plan)
    if f0 is None and need_f0:
        F = np.asarray(batch_fun(np.vstack([u[None, :], U])), dtype=np.float64)
        if F.ndim != 2 or F.shape[0] != U.shape[0] + 1 or not np.all(np.isfinite(F)):
            raise ValueError("batch_fun")
        f0, FU, n_points = F[0], F[1:], U.shape[0] + 1
    else:
        FU = np.asarray(batch_fun(U), dtype=np.float64)
        if FU.ndim != 2 or FU.shape[0] != U.shape[0] or not np.all(np.isfinite(FU)):
            raise ValueError("batch_fun")
        n_points = U.shape[0]
        if f0 is not None:
            f0 = np.asarray(f0, dtype=np.float64)
            if f0.shape != (FU.shape[1],):
                raise ValueError("f0")
    J = np.empty((FU.shape[1], u.shape[0]), dtype=np.float64)
    for i, pl in enumerate(plan):
        if pl[0] == "central":
            J[:, i] = (FU[pl[1]] - FU[pl[2]]) / (2.0 * pl[3])
        else:
            J[:, i] = (FU[pl[1]] - f0) / pl[2]
    return FJ.FdResult(J=J, f0=f0, n_points=n_points, plan=plan)


def _ref_jac_unit_to_q(J_u, bounds):
    b = np.asarray(bounds, dtype=np.float64)
    return np.asarray(J_u, dtype=np.float64) / (b[:, 1] - b[:, 0])[None, :]


def _install_fd_reference_if_stub():
    """Replace fd_jacobian's functions by the reference when Block 24 is still a stub."""
    global FD_SOURCE
    try:
        FJ.to_unit(np.array([0.5]), [(0.0, 1.0)])
    except NotImplementedError:
        FJ.to_unit = _ref_to_unit
        FJ.to_q = _ref_to_q
        FJ.fd_points = _ref_fd_points
        FJ.fd_jacobian = _ref_fd_jacobian
        FJ.jac_unit_to_q = _ref_jac_unit_to_q
        FD_SOURCE = "the smoke's private reference (fd_jacobian.py is still a stub)"


# ----------------------------------------------------------- the toy model ---
K = 8.05                                             # mV, slope of the toy degeneracy
T = np.linspace(0.0, 1.0, 400)                       # s, the window grid of every bundle
DT = T[1] - T[0]
N_PRE = 50                                           # pre-window samples (t < 0)
T_FULL = np.concatenate([-DT * np.arange(N_PRE, 0, -1), T])
AMPS = np.array([-50.0, -100.0, -150.0, -200.0])     # pA; bundle 0 plays the short square
IS_LS = np.array([False, True, True, True])
Q_LO = np.array([np.log(0.02), np.log(0.005), np.log(0.05), np.log(0.1), np.log(0.5), -10.0])
Q_HI = np.array([np.log(0.20), np.log(0.050), np.log(1.00), np.log(10.), np.log(2.0), 10.0])
NAMES6 = ["ln_R", "ln_tau", "ln_tau_h", "ln_g", "ln_a", "dv"]
U_TRUE = np.array([0.55, 0.40, 0.60, 0.45, 0.50])


def bounds_of(p):
    return [(float(Q_LO[i]), float(Q_HI[i])) for i in range(p)]


def q_of(u):
    p = len(u)
    return Q_LO[:p] + np.asarray(u) * (Q_HI[:p] - Q_LO[:p])


def simulate(q, six=False):
    """The reference's toy on the window grid T: (n_bundles, 400); complex q allowed."""
    R, tau, tau_h, g, a = np.exp(q[0]), np.exp(q[1]), np.exp(q[2]), np.exp(q[3]), np.exp(q[4])
    dv = q[5] if six else 0.0
    out = []
    for amp in AMPS:
        G = g * np.exp(dv / K) * (abs(amp) / 100.0) ** a
        sag = G / (1.0 + G) * (1.0 - np.exp(-T / tau_h))
        out.append(amp * R * (1.0 - np.exp(-T / tau)) * (1.0 - sag))
    return np.array(out)


def full_traces(q, six=False):
    """Per bundle the whole sweep on T_FULL: 0 in the pre-window, the toy after."""
    V = simulate(q, six)
    return [np.concatenate([np.zeros(N_PRE, dtype=V.dtype), V[b]]) for b in range(V.shape[0])]


def data_weights(V):
    """The reference's data_weights: a (n_b, 400) with c_b and 1/sum(w) included, D, c, w."""
    D = np.max(np.abs(V), axis=1)
    w = np.empty_like(V)
    for b in range(V.shape[0]):
        if IS_LS[b]:
            k = max(int(np.argmax(np.abs(V[b]))), 1)
            w[b, :k + 1] = T[:k + 1] / T[k]
            w[b, k + 1:] = np.exp(-(T[k + 1:] - T[k]) / 0.5)
        else:
            w[b] = np.exp(-T / 0.005)
    n_ss, n_ls = int((~IS_LS).sum()), int(IS_LS.sum())
    omega = np.abs(AMPS) * IS_LS / np.sum(np.abs(AMPS[IS_LS]))
    c = np.where(IS_LS, n_ls * omega, 1.0) / (n_ss + n_ls)
    a = c[:, None] * w / w.sum(axis=1, keepdims=True)
    return a, D, c, w


def make_layout(V_full_rec):
    """A hand-built object with Block 23's ResidualLayout attributes, from a recording
    (list of whole-sweep traces on T_FULL). a and D from the clean window part."""
    n_b = len(V_full_rec)
    mask = T_FULL >= 0.0
    pre = T_FULL < 0.0
    Vw = np.array([np.asarray(v)[mask] - np.mean(np.asarray(v)[pre]) for v in V_full_rec])
    a, D, c, _w = data_weights(Vw)
    n_w = int(mask.sum())
    slices = [slice(b * n_w, (b + 1) * n_w) for b in range(n_b)]
    return types.SimpleNamespace(
        masks=[mask.copy() for _ in range(n_b)], pre_masks=[pre.copy() for _ in range(n_b)],
        sqrt_a=[np.sqrt(a[b]) for b in range(n_b)], v_exp=[Vw[b] for b in range(n_b)],
        D=D, c=c, slices=slices, n_residuals=n_b * n_w)


def layout_with_vexp(layout, V_full_rec):
    """The same weights and D (fixed design), v_exp from another recording."""
    new = types.SimpleNamespace(**vars(layout))
    new.v_exp = [np.asarray(v)[m] - np.mean(np.asarray(v)[pm])
                 for v, m, pm in zip(V_full_rec, layout.masks, layout.pre_masks)]
    return new


def assemble(layout, traces):
    """2.10 (a): sqrt_a[b] * (traces[b][masks[b]] - v_exp[b]) / D[b], concatenated."""
    return np.concatenate([layout.sqrt_a[b] * (np.asarray(traces[b])[layout.masks[b]]
                                               - layout.v_exp[b]) / layout.D[b]
                           for b in range(len(traces))])


class ToyEvaluator:
    """Block 23's ev(Q) form: per point the list of bundle traces minus their own
    pre-window mean (a q-dependent offset is added and removed, as the real one
    removes its baseline), or None at a point listed in fail_rows."""

    def __init__(self, six=False, fail_rows=()):
        self.six = six
        self.fail_rows = set(fail_rows)
        self.calls = []

    def __call__(self, Q):
        Q = np.asarray(Q, dtype=np.float64)
        self.calls.append(Q.copy())
        out = []
        for k in range(Q.shape[0]):
            if k in self.fail_rows:
                out.append(None)
                continue
            raw = [x + 0.3 * Q[k, 0] + 1.0 for x in full_traces(Q[k], self.six)]
            out.append([x - np.mean(x[T_FULL < 0.0]) for x in raw])
        return out


def exact_S(q, six=False):
    """Exact sensitivities (complex step, h = 1e-20): list of (n_t, p), mV per unit of q_i."""
    p = len(q)
    cols = []
    for i in range(p):
        qc = np.asarray(q, dtype=complex)
        qc[i] += 1e-20j
        cols.append([x.imag / 1e-20 for x in full_traces(qc, six)])
    return [np.stack([cols[i][b] for i in range(p)], axis=1) for b in range(len(AMPS))]


# ---------------------------------------------------------------- checks ---
def test_I1_null_direction():
    q6 = np.append(q_of(U_TRUE), 0.0)
    lay = make_layout(full_traces(q6, True))
    J6 = ID.jacobian_from_sensitivities(exact_S(q6, True), lay)
    row6 = ID.identifiability_row(J6, NAMES6)
    svd6 = ID.svd_summary(J6)
    v_exp = np.zeros(6)
    v_exp[3], v_exp[5] = 1.0, -K
    v_exp /= np.linalg.norm(v_exp)
    v_p = np.array([row6["v_p_%s" % n] for n in NAMES6])
    cos = abs(float(v_p @ v_exp)) / np.linalg.norm(v_p)
    q5 = q_of(U_TRUE)
    lay5 = make_layout(full_traces(q5))
    J5 = ID.jacobian_from_sensitivities(exact_S(q5), lay5)
    row5 = ID.identifiability_row(J5, NAMES6[:5])
    print("  [I1] six: sigma_ratio %.3g, |cos| %.12f; five: sigma_ratio %.4g"
          % (row6["sigma_ratio"], cos, row5["sigma_ratio"]))
    bad = []
    if not row6["sigma_ratio"] < 1e-8:
        bad.append("six-parameter sigma_ratio %.3g >= 1e-8" % row6["sigma_ratio"])
    if not cos > 1.0 - 1e-8:
        bad.append("|cos| %.12f <= 1 - 1e-8" % cos)
    if not row5["sigma_ratio"] > 1e-3:
        bad.append("five-parameter sigma_ratio %.3g <= 1e-3" % row5["sigma_ratio"])
    # svd_summary: descending sigma, orthonormal rows, sign rule, ratio
    s, V = svd6["sigma"], svd6["V"]
    if not (s.shape == (6,) and V.shape == (6, 6) and np.all(np.diff(s) <= 0)):
        bad.append("sigma not (6,) descending or V not (6, 6)")
    if not np.allclose(V @ V.T, np.eye(6), atol=1e-12):
        bad.append("rows of V not orthonormal")
    for j in range(6):
        if V[j, int(np.argmax(np.abs(V[j])))] <= 0:
            bad.append("row %d: largest component not positive" % j)
    sv_np = np.linalg.svd(J6, compute_uv=False)
    if not np.allclose(s, sv_np, rtol=1e-12, atol=0):
        bad.append("sigma differs from numpy.linalg.svd")
    # J v_j = sigma_j u_j: |J v_j| == sigma_j (relative to sigma_1, float64 roundoff)
    if not np.allclose(np.linalg.norm(J6 @ V.T, axis=0), s, rtol=0, atol=1e-12 * s[0]):
        bad.append("|J v_j| != sigma_j")
    keys = (["sigma_%d" % j for j in range(1, 6)] + ["sigma_ratio"]
            + ["v_p_%s" % n for n in NAMES6[:5]] + ["v_pm1_%s" % n for n in NAMES6[:5]])
    if sorted(row5) != sorted(keys):
        bad.append("identifiability_row keys %s" % sorted(row5))
    if any(not isinstance(v, float) for v in row5.values()):
        bad.append("identifiability_row values not floats")
    assert not bad, "; ".join(bad)


def test_I2_predicted_scatter():
    p = 5
    q_true = q_of(U_TRUE)
    B = bounds_of(p)
    clean = full_traces(q_true)
    lay = make_layout(clean)
    sigma = 0.02 * float(lay.D.min())                 # mV, as the reference's S5
    sens = TS.compute_sensitivities(ToyEvaluator(), q_true, B)
    J = ID.jacobian_from_sensitivities(sens["S"], lay)
    n_k = 200
    rng_pred = np.random.default_rng([SEED, 1])
    R_eta = np.array([ID.noise_residuals(lay, [sigma * rng_pred.standard_normal(T_FULL.shape)
                                               for _ in AMPS]) for _ in range(n_k)])
    pred = ID.predicted_scatter(J, R_eta)
    if not (pred["dq"].shape == (n_k, p) and pred["sd"].shape == (p,)
            and pred["corr"].shape == (p, p)):
        raise AssertionError("predicted_scatter shapes %s %s %s"
                             % (pred["dq"].shape, pred["sd"].shape, pred["corr"].shape))
    # 200 noisy fits, independent noise; exact (complex-step) Jacobian for the solver
    span = Q_HI[:p] - Q_LO[:p]
    rng_fit = np.random.default_rng([SEED, 2])
    est, rss, hit_bound = [], [], 0
    for _k in range(n_k):
        noisy = [x + sigma * rng_fit.standard_normal(T_FULL.shape) for x in clean]
        lay_k = layout_with_vexp(lay, noisy)
        fun = lambda u, L=lay_k: assemble(L, full_traces(q_of(u)))
        jac = lambda u, L=lay: ID.jacobian_from_sensitivities(exact_S(q_of(u)), L) * span
        r = least_squares(fun, U_TRUE.copy(), jac=jac, bounds=(0.0, 1.0), method="trf",
                          x_scale=1.0, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=200)
        hit_bound += int(np.any(r.active_mask != 0))
        est.append(q_of(r.x))
        rss.append(float(np.sum(r.fun ** 2)))
    est = np.array(est)
    emp_sd = np.std(est, axis=0, ddof=1)
    ratio = emp_sd / pred["sd"]
    # planted defect: the naive covariance s^2 (J^T J)^{-1}, s^2 = RSS / (N - p)
    s2 = float(np.mean(rss)) / (J.shape[0] - p)
    naive_sd = np.sqrt(np.diag(s2 * np.linalg.inv(J.T @ J)))
    ratio_naive = emp_sd / naive_sd
    print("  [I2] empirical / predicted sd: %s; empirical / naive sd: %s; fits on a bound: %d"
          % (np.array2string(ratio, precision=3), np.array2string(ratio_naive, precision=3),
             hit_bound))
    bad = []
    if not np.all((ratio > 0.8) & (ratio < 1.25)):
        bad.append("empirical / predicted sd %s outside (0.8, 1.25)" % ratio)
    if np.all((ratio_naive > 0.8) & (ratio_naive < 1.25)):
        bad.append("the naive covariance is inside the band for every parameter %s" % ratio_naive)
    if hit_bound:
        bad.append("%d fits ended on a bound (the linearisation ignores the box)" % hit_bound)
    assert not bad, "; ".join(bad)


def test_I3_sensitivities():
    p = 5
    B = bounds_of(p)
    q = q_of(U_TRUE)
    ev = ToyEvaluator()
    sens = TS.compute_sensitivities(ev, q, B)
    S_ex = exact_S(q)
    errs = [np.linalg.norm(sens["S"][b] - S_ex[b]) / np.linalg.norm(S_ex[b])
            for b in range(len(AMPS))]
    print("  [I3] central h=1e-3: per-bundle relative error of S %s; evaluator calls %d; "
          "n_points %d" % (["%.2e" % e for e in errs], len(ev.calls), sens["n_points"]))
    bad = []
    if not max(errs) < 1e-4:          # central differences: O(h^2) truncation, ~1e-6 here
        bad.append("relative error of S %.3g >= 1e-4" % max(errs))
    if len(ev.calls) != 1:
        bad.append("evaluator called %d times" % len(ev.calls))
    if sens["n_points"] != 2 * p + 1 or ev.calls[0].shape != (2 * p + 1, p):
        bad.append("n_points %d (expected %d)" % (sens["n_points"], 2 * p + 1))
    if not np.array_equal(ev.calls[0][0], q):
        bad.append("the base point is not the first row asked")
    if not (len(sens["traces0"]) == 4 and all(x.shape == T_FULL.shape for x in sens["traces0"])
            and all(S.shape == (T_FULL.size, p) for S in sens["S"])):
        bad.append("shapes of traces0 / S")
    check_close("I3_traces0", np.concatenate(sens["traces0"]),
                np.concatenate(full_traces(q)), atol=1e-12)
    if not (sens["h"] == 1e-3 and sens["scheme"] == "central" and np.array_equal(sens["q"], q)):
        bad.append("h / scheme / q echoed wrongly")
    # one axis on its upper face: forward there, the base point asked once (fd_jacobian's)
    u_f = U_TRUE.copy()
    u_f[2] = 1.0
    ev2 = ToyEvaluator()
    sens_f = TS.compute_sensitivities(ev2, q_of(u_f), B)
    if len(ev2.calls) != 1 or sens_f["n_points"] != 2 * (p - 1) + 1 + 1:
        bad.append("face: calls %d, n_points %d (expected 1, %d)"
                   % (len(ev2.calls), sens_f["n_points"], 2 * p))
    # forward scheme: p + 1 points
    ev3 = ToyEvaluator()
    sens_fw = TS.compute_sensitivities(ev3, q, B, scheme="forward")
    if len(ev3.calls) != 1 or sens_fw["n_points"] != p + 1:
        bad.append("forward: n_points %d (expected %d)" % (sens_fw["n_points"], p + 1))
    # a None point raises, at the base point and at a perturbed one
    for rows in ((0,), (3,)):
        try:
            TS.compute_sensitivities(ToyEvaluator(fail_rows=rows), q, B)
            bad.append("no RuntimeError when row %s returns None" % (rows,))
        except RuntimeError:
            pass
    assert not bad, "; ".join(bad)


def test_I4_jacobian():
    p = 5
    B = bounds_of(p)
    bad = []
    for label, u in (("interior", U_TRUE), ("face", np.where(np.arange(p) == 1, 0.0, U_TRUE))):
        q = q_of(u)
        lay = make_layout(full_traces(q_of(U_TRUE)))
        sens = TS.compute_sensitivities(ToyEvaluator(), q, B)
        J = ID.jacobian_from_sensitivities(sens["S"], lay)
        batch = lambda U, L=lay: np.array([assemble(L, full_traces(FJ.to_q(x, B))) for x in U])
        J_direct = FJ.jac_unit_to_q(FJ.fd_jacobian(batch, u, h=1e-3, scheme="central").J, B)
        rd = ID.relative_difference(J, J_direct)
        print("  [I4] %s: relative difference %.3g; J shape %s" % (label, rd, J.shape))
        if J.shape != (lay.n_residuals, p):
            bad.append("%s: J shape %s" % (label, J.shape))
        if not rd < 1e-6:           # same steps, same points: differences are roundoff
            bad.append("%s: relative difference %.3g >= 1e-6" % (label, rd))
    assert not bad, "; ".join(bad)


def test_I5_noise_residuals():
    q = q_of(U_TRUE)
    clean = full_traces(q)
    lay = make_layout(clean)
    rng = np.random.default_rng([SEED, 5])
    eta = [0.4 * rng.standard_normal(T_FULL.shape) for _ in AMPS]
    r = ID.noise_residuals(lay, eta)
    bad = []
    if r.shape != (lay.n_residuals,) or r.dtype != np.float64:
        bad.append("shape / dtype %s %s" % (r.shape, r.dtype))
    eta_c = [x + (7.5 if b == 2 else 0.0) for b, x in enumerate(eta)]
    check_close("I5_constant", ID.noise_residuals(lay, eta_c), r, atol=1e-12)
    # sign: data = clean + noise; at the truth r(q_true) = r_eta
    lay_n = layout_with_vexp(lay, [x + e for x, e in zip(clean, eta)])
    check_close("I5_sign_truth", assemble(lay_n, clean), r, rtol=0, atol=1e-12)
    # hand case: 0 in the pre-window, +1 in the window -> -sqrt_a / D
    one = [np.where(T_FULL >= 0.0, 1.0, 0.0) for _ in AMPS]
    exp = np.concatenate([-lay.sqrt_a[b] / lay.D[b] for b in range(len(AMPS))])
    check_close("I5_hand", ID.noise_residuals(lay, one), exp, rtol=1e-15)
    print("  [I5] |r_eta| = %.4g; constant shift and sign checks done" % np.linalg.norm(r))
    assert not bad, "; ".join(bad)


def test_I6_save_load():
    q = q_of(U_TRUE)
    sens = TS.compute_sensitivities(ToyEvaluator(), q, bounds_of(5))
    t_list = [T_FULL.copy() for _ in AMPS]
    meta = {"specimen_id": 508282493, "protocol": "d207", "note": "toy"}
    bad = []
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "sens.npz")
        TS.save_sensitivities(path, sens, t_list=t_list, names=NAMES6[:5], meta=meta)
        with np.load(path, allow_pickle=False) as z:
            files = sorted(z.files)
        back = TS.load_sensitivities(path)
        path2 = os.path.join(d, "sens_nometa.npz")
        TS.save_sensitivities(path2, sens, t_list=t_list, names=NAMES6[:5])
        back2 = TS.load_sensitivities(path2)
    expected = sorted(["q", "h", "scheme", "names", "meta"]
                      + ["%s_%d" % (k, b) for k in ("t", "V0", "S") for b in range(4)])
    if files != expected:
        bad.append("npz keys %s" % files)
    if not np.array_equal(back["q"], sens["q"]):
        bad.append("q")
    if back["h"] != sens["h"] or back["scheme"] != sens["scheme"]:
        bad.append("h / scheme")
    for b in range(4):
        if not (np.array_equal(back["traces0"][b], sens["traces0"][b])
                and np.array_equal(back["S"][b], sens["S"][b])
                and np.array_equal(back["t_list"][b], t_list[b])):
            bad.append("bundle %d arrays" % b)
    if back["names"] != NAMES6[:5] or back["meta"] != meta or back2["meta"] is not None:
        bad.append("names / meta")
    print("  [I6] %d keys; round trip exact" % len(files))
    assert not bad, "; ".join(bad)


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def test_I7_reldiff_refusals():
    Jb = np.array([[3.0, 0.0], [0.0, 4.0]])
    Ja = Jb + np.array([[0.0, 0.0], [0.0, 1.0]])
    rd = ID.relative_difference(Ja, Jb)
    bad = []
    if abs(rd - 0.2) > 1e-15:
        bad.append("relative_difference %.17g != 0.2" % rd)
    lay = make_layout(full_traces(q_of(U_TRUE)))
    S_ok = exact_S(q_of(U_TRUE))
    J = ID.jacobian_from_sensitivities(S_ok, lay)
    R = np.zeros((3, J.shape[0]))
    cases = [
        ("reldiff shapes", ValueError, ID.relative_difference, (Ja, Jb[:1]), {}),
        ("reldiff zero J_b", ValueError, ID.relative_difference, (Ja, 0 * Jb), {}),
        ("scatter K = 1", ValueError, ID.predicted_scatter, (J, R[:1]), {}),
        ("scatter N mismatch", ValueError, ID.predicted_scatter, (J, R[:, :-1]), {}),
        ("scatter non-finite R", ValueError, ID.predicted_scatter,
         (J, np.where(np.arange(J.shape[0]) == 0, np.nan, 0.0)[None, :].repeat(3, 0)), {}),
        ("scatter non-finite J", ValueError, ID.predicted_scatter,
         (np.where(np.ones_like(J, bool) & (np.arange(J.shape[1]) == 0), np.inf, J), R), {}),
        ("scatter rank < p", ValueError, ID.predicted_scatter,
         (np.column_stack([J[:, :4], J[:, 0]]), R), {}),
        ("scatter R 1-D", ValueError, ID.predicted_scatter, (J, R[0]), {}),
        ("jac S count", ValueError, ID.jacobian_from_sensitivities, (S_ok[:3], lay), {}),
        ("jac S rows", ValueError, ID.jacobian_from_sensitivities,
         ([S_ok[0][1:]] + S_ok[1:], lay), {}),
        ("jac S cols", ValueError, ID.jacobian_from_sensitivities,
         ([S_ok[0][:, :4]] + S_ok[1:], lay), {}),
        ("noise count", ValueError, ID.noise_residuals, (lay, [np.zeros(T_FULL.size)] * 3), {}),
        ("noise length", ValueError, ID.noise_residuals,
         (lay, [np.zeros(T_FULL.size - 1)] * 4), {}),
        ("svd N < p", ValueError, ID.svd_summary, (np.ones((2, 3)),), {}),
        ("svd non-finite", ValueError, ID.svd_summary, (np.full((3, 2), np.nan),), {}),
        ("row names length", ValueError, ID.identifiability_row, (J, NAMES6[:4]), {}),
        ("row names repeated", ValueError, ID.identifiability_row, (J, ["a"] * 5), {}),
        ("sens q outside box", ValueError, TS.compute_sensitivities,
         (ToyEvaluator(), Q_HI[:5] + 1.0, bounds_of(5)), {}),
        ("sens bounds length", ValueError, TS.compute_sensitivities,
         (ToyEvaluator(), q_of(U_TRUE), bounds_of(4)), {}),
        ("sens h", ValueError, TS.compute_sensitivities,
         (ToyEvaluator(), q_of(U_TRUE), bounds_of(5)), {"h": 0.0}),
        ("sens scheme", ValueError, TS.compute_sensitivities,
         (ToyEvaluator(), q_of(U_TRUE), bounds_of(5)), {"scheme": "backward"}),
        ("sens None", RuntimeError, TS.compute_sensitivities,
         (ToyEvaluator(fail_rows=(1,)), q_of(U_TRUE), bounds_of(5)), {}),
        ("save names length", ValueError, TS.save_sensitivities,
         (os.path.join(tempfile.gettempdir(), "never_written_b26.npz"),
          {"q": np.zeros(2), "h": 1e-3, "scheme": "central", "traces0": [np.zeros(3)],
           "S": [np.zeros((3, 2))]}), {"t_list": [np.zeros(3)], "names": ["a"]}),
    ]
    for label, exc, fn, a, k in cases:
        if not _raises(exc, fn, *a, **k):
            bad.append(label)
    # p = 1 and p = 2 rows: keys present, corr of a pair
    rng = np.random.default_rng([SEED, 7])
    J2 = rng.standard_normal((50, 2))
    row2 = ID.identifiability_row(J2, ["x", "y"], R_eta=rng.standard_normal((20, 50)))
    if "corr_x_y" not in row2 or "sd_lin_y" not in row2 or "v_pm1_x" not in row2:
        bad.append("p = 2 row keys %s" % sorted(row2))
    row1 = ID.identifiability_row(J2[:, :1], ["x"], R_eta=rng.standard_normal((20, 50)))
    if row1["sigma_ratio"] != 1.0 or "v_pm1_x" in row1:
        bad.append("p = 1 row %s" % row1)
    # predicted_scatter against the closed form on a small case
    Rs = rng.standard_normal((20, 50))
    sc = ID.predicted_scatter(J2, Rs)
    dq_ref = -(np.linalg.solve(J2.T @ J2, J2.T @ Rs.T)).T
    check_close("I7_dq_closed_form", sc["dq"], dq_ref, rtol=1e-10, atol=1e-14)
    check_close("I7_corr", sc["corr"], np.corrcoef(dq_ref.T), rtol=1e-10, atol=1e-12)
    print("  [I7] relative_difference %.17g; %d refusals checked" % (rd, len(cases)))
    assert not bad, "refusal or case missing: " + "; ".join(bad)


def test_I8_ascii_lf():
    bad = []
    for name in ("identifiability.py", "trace_sensitivity.py", "smoke_identifiability.py"):
        data = (HERE / name).read_bytes()
        if any(b > 127 for b in data):
            bad.append("%s: non-ASCII byte" % name)
        if b"\r" in data:
            bad.append("%s: CR byte" % name)
    assert not bad, "; ".join(bad)


# ------------------------------------------------------------------ runner ---
def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    parts.append("fd: %s" % FD_SOURCE)
    return " | ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated check ids to run, e.g. I1,I3")
    args = ap.parse_args(argv)
    only = [t.strip() for t in args.only.split(",") if t.strip()]
    _install_fd_reference_if_stub()
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)
              and (not only or any(name.startswith("test_%s_" % o) for o in only))]
    print("== %s" % Path(__file__).name)
    print(_environment())
    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, headline[:200])).rstrip())
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print("-- arrays from a failed comparison: %s" % path)
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
