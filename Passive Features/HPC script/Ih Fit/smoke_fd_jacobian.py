"""Smoke test for the finite-difference Jacobian -- Block 24 in specs/SPEC.md (J1-J7), SPEC 2.10 (b).

Pure: numpy and scipy only, no NEURON, no file I/O by the code under test.

Checks
    test_J1_linear            f(u) = A u + b (p = 4, N = 7): J == A (atol 1e-9 x max|A|) for forward
                              and central, at an interior point, on each lower and each upper face,
                              and at the two corners
    test_J2_toy_model         the toy model of the review's smoke_sensitivity_reference.py (five
                              parameters, data weights, residual assembly) reproduced here; its exact
                              Jacobian by complex step. At h = 1e-3: forward within 2e-2 relative
                              (Frobenius), central within 1e-4, forward at a point on an upper bound
                              within 2e-2 with the step of that axis flipped to -h
    test_J3_approx_fprime     per residual j, scipy.optimize.approx_fprime on the scalar field f_j at
                              an interior point (absolute step h): equal to row j of J within 1e-6 rel.
    test_J4_one_call          batch_fun is called once; it receives [u; U] when f0 is needed and not
                              given, U alone when f0 is given or not needed (central, all two-sided);
                              n_points; the plan is fd_points'; the given f0 is the one used; u not
                              mutated; fd_points' row order and plan
    test_J5_coordinates       to_unit / to_q inverse on the box ((p,) and (m, p)), to_q clips;
                              jac_unit_to_q against a chain-rule case (linear exact, nonlinear 1e-6)
    test_J6_refusals          the ValueErrors of the Block 24 entry; h = 0.5 and u on the faces accepted
    test_J7_ascii_lf          fd_jacobian.py and this file are pure ASCII with LF line endings

Run (from this folder)
    python smoke_fd_jacobian.py [--only J1,J4]
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit code 1 if any check failed, errored or is still TODO.

Debug
    A failed check_close() saves both arrays to smoke_artifacts/<check name>.npz.

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                         # noqa: E402
from numpy.testing import assert_allclose  # noqa: E402
from scipy.optimize import approx_fprime   # noqa: E402

import fd_jacobian as FD                   # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = HERE / "smoke_artifacts"
_SAVED_ARTIFACTS = []


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


def _rows(fun):
    """batch_fun from a one-point function u (p,) -> (N,)."""
    return lambda U: np.array([fun(x) for x in U])


# ------------------------------------------------- the review's toy model ---
# Reproduced from smoke_sensitivity_reference.py (review of 2026-11-25): simulate,
# data_weights, assemble_residuals, to_q; the exact Jacobian by complex step.
TOY_K = 8.05                                            # mV
TOY_T = np.linspace(0.0, 1.0, 400)                      # s
TOY_AMPS = np.array([-50.0, -100.0, -150.0, -200.0])    # pA
TOY_IS_LS = np.array([False, True, True, True])
TOY_Q_LO = np.array([np.log(0.02), np.log(0.005), np.log(0.05), np.log(0.1), np.log(0.5), -10.0])
TOY_Q_HI = np.array([np.log(0.20), np.log(0.050), np.log(1.00), np.log(10.), np.log(2.0), 10.0])
TOY_U_TRUE = np.array([0.55, 0.40, 0.60, 0.45, 0.50])


def toy_simulate(q, six=False):
    R, tau, tau_h, g, a = np.exp(q[0]), np.exp(q[1]), np.exp(q[2]), np.exp(q[3]), np.exp(q[4])
    dv = q[5] if six else 0.0
    out = []
    for amp in TOY_AMPS:
        G = g * np.exp(dv / TOY_K) * (abs(amp) / 100.0) ** a
        sag = G / (1.0 + G) * (1.0 - np.exp(-TOY_T / tau_h))
        out.append(amp * R * (1.0 - np.exp(-TOY_T / tau)) * (1.0 - sag))
    return np.array(out)


def toy_data_weights(V):
    D = np.max(np.abs(V), axis=1)
    w = np.empty_like(V)
    for b in range(V.shape[0]):
        if TOY_IS_LS[b]:
            k = max(int(np.argmax(np.abs(V[b]))), 1)
            w[b, :k + 1] = TOY_T[:k + 1] / TOY_T[k]
            w[b, k + 1:] = np.exp(-(TOY_T[k + 1:] - TOY_T[k]) / 0.5)
        else:
            w[b] = np.exp(-TOY_T / 0.005)
    n_ss, n_ls = int((~TOY_IS_LS).sum()), int(TOY_IS_LS.sum())
    omega = np.abs(TOY_AMPS) * TOY_IS_LS / np.sum(np.abs(TOY_AMPS[TOY_IS_LS]))
    c = np.where(TOY_IS_LS, n_ls * omega, 1.0) / (n_ss + n_ls)
    a = c[:, None] * w / w.sum(axis=1, keepdims=True)
    return a, D, c, w


def toy_assemble(Vhat, V, a, D):
    return (np.sqrt(a) * (Vhat - V) / D[:, None]).ravel()


def toy_to_q(u, p):
    return TOY_Q_LO[:p] + u * (TOY_Q_HI[:p] - TOY_Q_LO[:p])


def _toy_problem():
    V = toy_simulate(toy_to_q(TOY_U_TRUE, 5))
    a, D, _c, _w = toy_data_weights(V)

    def res(u):
        return toy_assemble(toy_simulate(toy_to_q(u, 5)), V, a, D)

    def J_exact(u):
        # complex step, h = 1e-20: exact to rounding (no subtraction)
        cols = []
        for i in range(5):
            uc = np.asarray(u, dtype=complex).copy()
            uc[i] += 1e-20j
            cols.append((np.sqrt(a) * toy_simulate(toy_to_q(uc, 5)).imag / 1e-20 / D[:, None]).ravel())
        return np.array(cols).T
    return res, J_exact


# ----------------------------------------------------------------- checks ---
def test_J1_linear():
    rng = np.random.default_rng(SEED)
    p, N = 4, 7
    A = rng.normal(size=(N, p)) * 10.0
    b = rng.normal(size=N)
    batch = lambda U: U @ A.T + b
    # Tolerance: J is exact for an affine map up to rounding, ~ eps * |f| / h ~ 1e-12 x |A|.
    atol = 1e-9 * np.max(np.abs(A))
    points = [("interior", rng.uniform(0.2, 0.8, p)), ("corner0", np.zeros(p)), ("corner1", np.ones(p))]
    for i in range(p):
        for face in (0.0, 1.0):
            x = rng.uniform(0.2, 0.8, p)
            x[i] = face
            points.append(("axis%d=%g" % (i, face), x))
    for label, x in points:
        for scheme in ("forward", "central"):
            r = FD.fd_jacobian(batch, x, h=1e-3, scheme=scheme)
            assert r.J.dtype == np.float64 and r.J.shape == (N, p), "J dtype/shape %s %s" % (r.J.dtype, r.J.shape)
            check_close("J1_%s_%s" % (label, scheme), r.J, A, atol=atol)


def test_J2_toy_model():
    res, J_exact = _toy_problem()
    batch = _rows(res)
    u = TOY_U_TRUE
    Jx = J_exact(u)
    nrm = np.linalg.norm(Jx)
    e_f = np.linalg.norm(FD.fd_jacobian(batch, u, h=1e-3, scheme="forward").J - Jx) / nrm
    e_c = np.linalg.norm(FD.fd_jacobian(batch, u, h=1e-3, scheme="central").J - Jx) / nrm
    u_edge = u.copy()
    u_edge[3] = 1.0                                             # ON the upper bound of axis 3
    r_b = FD.fd_jacobian(batch, u_edge, h=1e-3, scheme="forward")
    Jx_b = J_exact(u_edge)
    e_b = np.linalg.norm(r_b.J - Jx_b) / np.linalg.norm(Jx_b)  # relative to the exact J at that point
    e_b_ref = np.linalg.norm(r_b.J - Jx_b) / nrm                # the reference's normalisation
    msg = ("forward %.2e, central %.2e, forward on a bound %.2e (ref. norm %.2e); limits 2e-2, 1e-4, 2e-2"
           % (e_f, e_c, e_b, e_b_ref))
    print("    J2: " + msg)
    # Tolerances: SPEC Block 24 J2 (forward O(h) error; central O(h^2)).
    assert e_f < 2e-2 and e_c < 1e-4 and e_b < 2e-2 and e_b_ref < 2e-2, msg
    assert r_b.plan[3] == ("forward", 3, -1e-3), "axis on the upper bound: plan %r" % (r_b.plan[3],)
    for i in (0, 1, 2, 4):
        assert r_b.plan[i] == ("forward", i, 1e-3), "interior axis %d: plan %r" % (i, r_b.plan[i])


def test_J3_approx_fprime():
    res, _ = _toy_problem()
    h = 1e-3
    u = np.array([0.31, 0.62, 0.47, 0.28, 0.73])
    r = FD.fd_jacobian(_rows(res), u, h=h, scheme="forward")
    J_sc = np.array([approx_fprime(u, (lambda x, j=j: float(res(x)[j])), h) for j in range(r.J.shape[0])])
    # Tolerance 1e-6 relative to max|J| (SPEC): both are the forward rule with the same absolute
    # step; scipy divides by the float step actually taken, so they differ at rounding level only.
    scale = np.max(np.abs(J_sc))
    err = np.max(np.abs(r.J - J_sc)) / scale
    print("    J3: max |J - approx_fprime| / max|J| = %.2e (limit 1e-6)" % err)
    check_close("J3_vs_approx_fprime", r.J, J_sc, atol=1e-6 * scale)


def test_J4_one_call():
    rng = np.random.default_rng(SEED + 4)
    p, N = 3, 5
    A = rng.normal(size=(N, p))
    calls = []

    def batch(U):
        calls.append(np.array(U, copy=True))
        return np.array([A @ x for x in U])            # row by row: the same float result per point

    def run(u, **kw):
        del calls[:]
        u_in = np.array(u, dtype=float)
        u_copy = u_in.copy()
        r = FD.fd_jacobian(batch, u_in, **kw)
        assert np.array_equal(u_in, u_copy), "u mutated"
        assert len(calls) == 1, "batch_fun called %d times" % len(calls)
        U, plan = FD.fd_points(u_copy, h=kw.get("h", 1e-3), scheme=kw.get("scheme", "forward"))
        assert r.plan == plan, "plan differs from fd_points'"
        assert r.n_points == calls[0].shape[0], "n_points %d, rows asked %d" % (r.n_points, calls[0].shape[0])
        return r, calls[0], U

    u_int = np.array([0.3, 0.5, 0.7])
    u_face = np.array([0.3, 1.0, 0.0])
    # forward, f0 not given: [u; U], f0 computed
    r, X, U = run(u_int, scheme="forward")
    assert U.shape == (p, p) and np.array_equal(X, np.vstack([u_int, U])), "forward rows"
    f0 = batch(u_int[None, :])[0]                    # f(u) as batch_fun computes it
    assert r.f0 is not None and np.array_equal(r.f0, f0) and r.n_points == p + 1, "forward f0 / n_points"
    # forward, f0 given: U only, the given f0 used
    r, X, U = run(u_int, scheme="forward", f0=f0)
    assert np.array_equal(X, U) and r.n_points == p, "forward with f0: u asked"
    assert np.array_equal(r.f0, f0), "forward with f0: f0 returned"
    r_off, _, _ = run(u_int, scheme="forward", f0=f0 + 1.0)    # a different f0 changes J by -1/h
    check_close("J4_f0_used", r_off.J - r.J, np.full((N, p), -1.0 / 1e-3), rtol=1e-9)
    # central, every axis two-sided: U only, f0 None, plus then minus
    r, X, U = run(u_int, scheme="central")
    assert np.array_equal(X, U) and U.shape == (2 * p, p) and r.f0 is None and r.n_points == 2 * p, "central, all two-sided"
    for i in range(p):
        kind, rp, rm, hh = r.plan[i]
        assert kind == "central" and (rp, rm) == (2 * i, 2 * i + 1) and hh == 1e-3, r.plan[i]
        e = np.zeros(p)
        e[i] = 1e-3
        assert np.array_equal(U[rp], u_int + e) and np.array_equal(U[rm], u_int - e), "central rows %d" % i
    # central with one-sided axes (on the faces), f0 not given: [u; U]
    r, X, U = run(u_face, scheme="central")
    assert U.shape == (2 + 1 + 1, p) and np.array_equal(X, np.vstack([u_face, U])), "central, faces"
    assert r.plan == [("central", 0, 1, 1e-3), ("forward", 2, -1e-3), ("forward", 3, 1e-3)], r.plan
    assert np.array_equal(r.f0, batch(u_face[None, :])[0]), "central, faces: f0"
    # central with one-sided axes, f0 given: U only
    r, X, U = run(u_face, scheme="central", f0=batch(u_face[None, :])[0])
    assert np.array_equal(X, U) and r.n_points == 4, "central, faces, f0 given"
    # forward plan: k = p, row i perturbs axis i only, step flipped on the upper face
    U, plan = FD.fd_points(u_face, h=1e-3, scheme="forward")
    assert plan == [("forward", 0, 1e-3), ("forward", 1, -1e-3), ("forward", 2, 1e-3)], plan
    D = U - u_face
    assert np.array_equal(D != 0, np.eye(p, dtype=bool)) and U.dtype == np.float64, "forward rows"
    assert np.all((U >= 0.0) & (U <= 1.0)), "a perturbed point left the box"


def test_J5_coordinates():
    rng = np.random.default_rng(SEED + 5)
    bounds = [(np.log(1e-5), np.log(1e-3)), (-10.0, 10.0), (0.3, 3.0)]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    Q = lo + rng.random((50, 3)) * (hi - lo)
    # Tolerance 1e-12 absolute: two affine maps on numbers of order 10.
    check_close("J5_roundtrip_m", FD.to_q(FD.to_unit(Q, bounds), bounds), Q, atol=1e-12)
    check_close("J5_roundtrip_1", FD.to_q(FD.to_unit(Q[0], bounds), bounds), Q[0], atol=1e-12)
    check_close("J5_unit_faces", FD.to_unit(np.vstack([lo, hi]), bounds), [[0, 0, 0], [1, 1, 1]], atol=1e-15)
    assert FD.to_unit(Q, bounds).shape == (50, 3) and FD.to_unit(Q[0], bounds).shape == (3,)
    clipped = FD.to_q(np.array([[-0.2, 0.5, 1.3]]), bounds)
    check_close("J5_clip", clipped, [[lo[0], 0.0, hi[2]]], atol=1e-15)
    # chain rule, linear: f(u) = A q(u) -> J_u = A diag(hi - lo), J_q = A (exact)
    A = rng.normal(size=(6, 3))
    batch_lin = lambda U: FD.to_q(U, bounds) @ A.T
    u0 = np.array([0.4, 0.5, 0.6])
    J_u = FD.fd_jacobian(batch_lin, u0, scheme="central").J
    check_close("J5_chain_linear", FD.jac_unit_to_q(J_u, bounds), A, atol=1e-8 * np.max(np.abs(A)))
    # chain rule, nonlinear: g(q) = [exp(q0) q1, sin(q2) + q0^2]; central h = 1e-3 in u.
    # Tolerance 1e-6 relative: central truncation ~ h^2 |g'''| (hi - lo)^2 / 6 is far below it here.
    def g(q):
        return np.array([np.exp(q[0]) * q[1], np.sin(q[2]) + q[0] ** 2])
    q0 = FD.to_q(u0, bounds)
    J_exact = np.array([[np.exp(q0[0]) * q0[1], np.exp(q0[0]), 0.0],
                        [2 * q0[0], 0.0, np.cos(q0[2])]])
    J_u = FD.fd_jacobian(lambda U: np.array([g(q) for q in FD.to_q(U, bounds)]), u0, scheme="central").J
    J_q = FD.jac_unit_to_q(J_u, bounds)
    for row in range(2):
        check_close("J5_chain_nonlinear_row%d" % row, J_q[row], J_exact[row],
                    atol=1e-6 * np.max(np.abs(J_exact[row])))


def test_J6_refusals():
    bad = []
    B = [(0.0, 1.0), (0.0, 1.0)]
    lin = lambda U: U * 2.0

    def refused(label, fn):
        try:
            fn()
            bad.append("accepted: %s" % label)
        except ValueError:
            pass

    for label, u in (("u 2-D", [[0.5, 0.5]]), ("u nan", [np.nan, 0.5]), ("u inf", [np.inf, 0.5]),
                     ("u < 0", [-1e-12, 0.5]), ("u > 1", [0.5, 1.0 + 1e-12]), ("u empty", []),
                     ("u scalar", 0.5)):
        refused("fd_jacobian " + label, lambda u=u: FD.fd_jacobian(lin, u))
        refused("fd_points " + label, lambda u=u: FD.fd_points(u))
    for h in (0.0, -1e-3, 0.5000001, 1.0, float("nan"), float("inf"), "x"):
        refused("h=%r" % (h,), lambda h=h: FD.fd_jacobian(lin, [0.5, 0.5], h=h))
        refused("fd_points h=%r" % (h,), lambda h=h: FD.fd_points([0.5, 0.5], h=h))
    for s in ("backward", "Central", None):
        refused("scheme %r" % (s,), lambda s=s: FD.fd_jacobian(lin, [0.5, 0.5], scheme=s))
    for label, fn in (("1-D output", lambda U: U.sum(axis=1)),
                      ("too few rows", lambda U: U[1:] * 2.0),
                      ("too many rows", lambda U: np.vstack([U, U])),
                      ("3-D output", lambda U: U[:, :, None]),
                      ("nan output", lambda U: np.where(U > 0.5004, np.nan, U)),
                      ("inf output", lambda U: np.full(U.shape, np.inf))):
        refused("batch_fun " + label, lambda fn=fn: FD.fd_jacobian(fn, [0.5, 0.5]))
    refused("f0 wrong length", lambda: FD.fd_jacobian(lin, [0.5, 0.5], f0=np.zeros(3)))
    refused("f0 2-D", lambda: FD.fd_jacobian(lin, [0.5, 0.5], f0=np.zeros((1, 2))))
    refused("f0 nan", lambda: FD.fd_jacobian(lin, [0.5, 0.5], f0=np.array([np.nan, 0.0])))
    for label, b in (("hi == lo", [(0.0, 1.0), (1.0, 1.0)]), ("hi < lo", [(0.0, 1.0), (2.0, 1.0)]),
                     ("not pairs", [0.0, 1.0]), ("non-finite", [(0.0, np.inf), (0.0, 1.0)])):
        refused("to_unit " + label, lambda b=b: FD.to_unit([0.5, 0.5], b))
        refused("to_q " + label, lambda b=b: FD.to_q([0.5, 0.5], b))
        refused("jac_unit_to_q " + label, lambda b=b: FD.jac_unit_to_q(np.ones((3, 2)), b))
    refused("to_unit wrong width", lambda: FD.to_unit([0.5, 0.5, 0.5], B))
    refused("jac_unit_to_q wrong width", lambda: FD.jac_unit_to_q(np.ones((3, 3)), B))
    assert not bad, "; ".join(bad)
    # closed ends accepted
    FD.fd_jacobian(lin, [0.5, 0.5], h=0.5, scheme="central")
    FD.fd_jacobian(lin, [0.0, 1.0], h=0.5)


def test_J7_ascii_lf():
    for name in ("fd_jacobian.py", Path(__file__).name):
        data = (HERE / name).read_bytes()
        assert all(b < 128 for b in data), "%s has non-ASCII bytes" % name
        assert b"\r" not in data, "%s has CR bytes" % name


# ---------------------------------------------------------------- runner ---
def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated check ids to run, e.g. J1,J4")
    args = ap.parse_args(argv)
    only = [t.strip() for t in args.only.split(",") if t.strip()]
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
