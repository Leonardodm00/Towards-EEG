"""Smoke test for the blurred-tube model and the per-node fit -- Block 3 in
specs/SPEC.md (handoff Eqs. 10-11; D-018.2; D-019.1).

Checks
    test_known_answer   exact dip area B (pi d / 2)(I_1(alpha) - L_1(alpha)) and
                        its faint (B mu pi d^2 / (4 cos phi)) and opaque
                        (B d (1 - 1/alpha^2 - 3/alpha^4 - 45/alpha^6)) limits;
                        faint-limit centroid v0 and variance sigma^2 + d^2/16
                        (handoff Eq. 12); noise-free recovery of (d, mu, v0)
                        (handoff "Numbers checked"); the logged D-019 (a) values
                        on the checks/mu_tie.py profiles
    test_reference      (B3.1) vs scipy.integrate.quad of the unsubstituted
                        integral; vs the grid model of checks/mu_tie.py; the
                        quadrature rule integrates 1 and cos^2 exactly
    test_convergence    quadrature error falls with N and is <= 1e-12 B at the
                        fit's N in the worst case (d = d_hi, smallest sigma)
    test_invariants     symmetry about v0; darker with alpha; 0 <= I <= B and
                        I -> B far away; mu form = alpha form (bit-identical);
                        the mu and alpha fits agree wherever no bound is active
                        (D-019 bijection guard)
    test_contract       FitResult fields and types, shapes, start points in bounds
    test_determinism    a repeated fit is bit-identical
    test_edge_cases     flat profile -> mu_lo; 8 um dip -> d_hi; half-depth width
                        of a triangular dip; the "fixed" start rule; phi near
                        90 deg; invalid inputs raise

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_fit.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import ast
import dataclasses
import functools
import importlib.metadata
import math
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from scipy import integrate, special

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import fit as F  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import tube_model as M  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy")
B = 200.0                      # grey levels; any positive value (the model is linear in B)


def cfg_measure(**changes):
    return dataclasses.replace(default_config().measure, **changes)


def fit_nodes(cfg):
    return M.n_quadrature_nodes(cfg.fit_d_bounds_um[1], cfg.sigma_fit_um, cfg.fit_quad_min_nodes,
                                cfg.fit_quad_nodes_per_sigma)


def nodes_for(d, sigma, cfg=None):
    cfg = cfg or cfg_measure()
    return M.n_quadrature_nodes(d, sigma, cfg.fit_quad_min_nodes, cfg.fit_quad_nodes_per_sigma)


def profile_axis(cfg):
    """v_n of handoff Eq. 9: [-profile_half_um, profile_half_um] in steps of profile_step_um."""
    n = int(round(cfg.profile_half_um / cfg.profile_step_um))
    return np.arange(-n, n + 1) * cfg.profile_step_um


def reference_quad(v, d, alpha, v0, sigma):
    """(T * g_sigma)(v) from the unsubstituted integral 1 - Int_{-d/2}^{d/2}
    (1 - exp(-alpha s_d(u))) g_sigma(v - v0 - u) du, by adaptive quadrature."""
    x = v - v0

    def f(u):
        s = math.sqrt(max(0.0, 1.0 - (2.0 * u / d) ** 2))
        return (1.0 - math.exp(-alpha * s)) * math.exp(-0.5 * ((x - u) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

    pts = sorted(p for p in (x - 8 * sigma, x, x + 8 * sigma) if -0.5 * d < p < 0.5 * d)
    val, _err = integrate.quad(f, -0.5 * d, 0.5 * d, points=pts or None, epsabs=1e-15, epsrel=1e-13, limit=500)
    return 1.0 - val


def load_mu_tie():
    """The grid model of checks/mu_tie.py (D-019's [run]) taken from the file
    itself -- its imports, the grid assignments dx, vf, vn and `model` --
    without running the script's own fits."""
    path = WS / "checks" / "mu_tie.py"
    tree = ast.parse(path.read_text())
    keep = [node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            or (isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in ("dx", "vf", "vn")
                                                     for t in node.targets))
            or (isinstance(node, ast.FunctionDef) and node.name == "model")]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return ns["model"], ns["vn"]


MU_TIE_PHI = math.radians(15.0)                 # mu_tie.py PHI
MU_TIE_SIGMA_TRUE, MU_TIE_SIGMA_FIT = 0.10, 0.125
D019A_LOGGED = {0.6: {0.5: 0.88, 1.0: 0.98, 2.0: 1.00}, 3.0: {0.5: 0.89, 1.0: 0.98, 2.0: 0.98}}


@functools.lru_cache(maxsize=1)
def mu_tie_fits():
    """Fit the mu_tie.py profiles (sigma_true 0.10, B = 1) with sigma_fit
    0.125 in both forms. Returns {(mu, d): (FitResult mu form, FitResult alpha form)}."""
    model, vn = load_mu_tie()
    cfg = cfg_measure(sigma_fit_um=MU_TIE_SIGMA_FIT)
    out = {}
    for mu in (0.6, 3.0):
        for d in (0.3, 0.5, 1.0, 2.0):
            y = model(d, mu * d / math.cos(MU_TIE_PHI), 0.0, MU_TIE_SIGMA_TRUE)
            out[(mu, d)] = (F.fit_profile(vn, y, MU_TIE_PHI, 1.0, cfg), F.fit_profile_alpha(vn, y, MU_TIE_PHI, 1.0, cfg))
    return out


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    sigma, d, v0 = 0.1, 1.3, 0.07
    vv = np.linspace(-6.0, 6.0, 6001)          # h = 0.002 um = sigma / 50; trapezoid is spectral here
    N = nodes_for(d, sigma)
    # exact dip area for every alpha >= 0
    for alpha in (1e-3, 0.3, 3.0, 10.0):
        area = np.trapezoid(B - M.model_profile(vv, d, alpha, v0, sigma, B, N), vv)
        exact = B * 0.5 * math.pi * d * (special.iv(1, alpha) - special.modstruve(1, alpha))
        assert abs(area / exact - 1) <= 1e-9, "dip area alpha %g: %.3e rel" % (alpha, area / exact - 1)
    # faint limit: B mu pi d^2 / (4 cos phi); the first correction is -4 alpha / (3 pi) relative
    phi, mu = math.radians(40.0), 1e-6
    alpha = float(M.alpha_from_mu(mu, d, phi))
    area = np.trapezoid(B - M.model_profile(vv, d, alpha, v0, sigma, B, N), vv)
    faint = B * mu * math.pi * d * d / (4 * math.cos(phi))
    assert abs(area / faint - 1) <= 1e-6, "faint limit %.3e" % (area / faint - 1)
    # opaque limit: B d (1 - 1/a^2 - 3/a^4 - 45/a^6), next term O(a^-8) ~ 1e-11 at a = 50
    a = 50.0
    area = np.trapezoid(B - M.model_profile(vv, d, a, v0, sigma, B, N), vv)
    opaque = B * d * (1 - a ** -2 - 3 * a ** -4 - 45 * a ** -6)
    assert abs(area / opaque - 1) <= 1e-8, "opaque limit %.3e" % (area / opaque - 1)
    # faint-limit moments (handoff Eq. 12): centroid v0, variance sigma^2 + d^2/16; O(alpha) corrections
    d2, v02 = 0.7, 0.05
    vv2 = np.linspace(-6.0, 6.0, 12001)
    dip = 1.0 - M.model_profile(vv2, d2, 1e-6, v02, sigma, 1.0, nodes_for(d2, sigma))
    m0 = np.trapezoid(dip, vv2)
    m1 = np.trapezoid(vv2 * dip, vv2) / m0
    var = np.trapezoid((vv2 - m1) ** 2 * dip, vv2) / m0
    assert abs(m1 - v02) <= 1e-9, "centroid %.3e" % (m1 - v02)
    assert abs(var / (sigma ** 2 + d2 ** 2 / 16) - 1) <= 1e-6, "variance %.3e" % (var / (sigma ** 2 + d2 ** 2 / 16) - 1)
    # noise-free recovery (truth with twice the nodes: not bit-identical to the fit's model)
    cfg = cfg_measure()
    v = profile_axis(cfg)
    N2 = 2 * fit_nodes(cfg)
    worst = 0.0
    for d in (0.5, 1.0, 2.0):
        for alpha in (0.3, 3.0):
            for phi_deg in (0.0, 30.0, 60.0):
                phi = math.radians(phi_deg)
                mu = alpha * math.cos(phi) / d
                y = M.model_profile(v, d, alpha, 0.03, cfg.sigma_fit_um, B, N2)
                r = F.fit_profile(v, y, phi, B, cfg)
                assert r.status == "converged", (d, alpha, phi_deg, r.status, r.message)
                err = max(abs(r.d_hat_um / d - 1), abs(r.mu_hat_per_um / mu - 1), abs(r.v0_hat_um - 0.03))
                worst = max(worst, err)
    # observed ~1e-14; 1e-6 is 1000x tighter than the handoff's 1e-3 and leaves room for platforms
    assert worst <= 1e-6, "noise-free recovery worst %.3e" % worst
    # the logged D-019 (a) values (2 decimals; mu_tie's own fit): +-0.006 = rounding + 0.001
    fits = mu_tie_fits()
    for mu, row in D019A_LOGGED.items():
        for d, logged in row.items():
            got = fits[(mu, d)][0].d_hat_um / d
            assert abs(got - logged) <= 0.006, "D-019 (a) mu %g d %g: %.4f vs logged %.2f" % (mu, d, got, logged)


def test_reference():
    # (B3.1) by Gauss-Legendre vs the unsubstituted integral by scipy.integrate.quad
    worst = 0.0
    for sigma in (0.08, 0.125):
        N = fit_nodes(cfg_measure(sigma_fit_um=sigma))
        for d in (0.05, 0.5, 2.0, 6.0):
            vs = np.linspace(-0.5 * d - 4 * sigma, 0.5 * d + 4 * sigma, 9) + 0.0137
            for alpha in (0.3, 3.0, 30.0):
                got = M.model_profile(vs, d, alpha, 0.0137, sigma, 1.0, N)
                ref = np.array([reference_quad(x, d, alpha, 0.0137, sigma) for x in vs])
                worst = max(worst, float(np.max(np.abs(got - ref))))
    # quad's own accuracy (epsrel 1e-13) and float64 summation over N <= 450 nodes
    assert worst <= 1e-12, "model vs quad: %.3e (in units of B)" % worst
    # the grid model of checks/mu_tie.py (dx = 0.002 um grid, 4-sigma truncated filter)
    model, vn = load_mu_tie()
    worst = 0.0
    for d in (0.3, 0.5, 1.0, 2.0):
        for alpha in (0.3, 3.0):
            ours = M.model_profile(vn, d, alpha, 0.0, 0.10, 1.0, nodes_for(6.0, 0.10))
            worst = max(worst, float(np.max(np.abs(model(d, alpha, 0.0, 0.10) - ours))))
    assert worst <= 1e-3, "vs mu_tie.py grid model: %.3e (observed 7e-4 is that grid's error)" % worst
    # the rule integrates 1 exactly for every N, and cos^2 (analytic, not a
    # polynomial in the Legendre variable) to roundoff once N >= 64
    for N in (8, 64, 364):
        s, c, w = M.quadrature_rule(N)
        assert abs(np.sum(w) - math.pi) <= 1e-12, ("weights", N)
        assert N < 64 or abs(np.sum(w * c * c) - 0.5 * math.pi) <= 1e-12, ("cos^2", N)


def test_convergence():
    cfg = cfg_measure(sigma_fit_um=min(cfg_measure().sigma_fit_study_um))
    N = fit_nodes(cfg)
    d, sigma = cfg.fit_d_bounds_um[1], cfg.sigma_fit_um
    v = np.linspace(-0.5 * d - 0.5, 0.5 * d + 0.5, 101) + 0.0123
    for alpha in (0.3, 3.0):
        ref = M.model_profile(v, d, alpha, 0.0, sigma, 1.0, 4 * N)
        errs = [float(np.max(np.abs(M.model_profile(v, d, alpha, 0.0, sigma, 1.0, n) - ref)))
                for n in (N // 4, N // 2, N)]
        # N/4 is far from converged in this worst case; beyond N/2 only roundoff (~1e-15) is left
        assert errs[0] > errs[1] and errs[1] >= errs[2] - 1e-14, ("not decreasing", alpha, errs)
        assert errs[2] <= 1e-12, ("error at the fit's N = %d" % N, alpha, errs)


def test_invariants():
    sigma, d, v0 = 0.099, 0.9, 0.21
    N = nodes_for(d, sigma)
    t = np.linspace(0.0, 2.0, 41)
    left = M.model_profile(v0 - t, d, 2.0, v0, sigma, B, N)
    right = M.model_profile(v0 + t, d, 2.0, v0, sigma, B, N)
    assert np.max(np.abs(left - right)) <= 1e-12 * B, "symmetry about v0"
    v = np.linspace(-3, 3, 121)
    prof = [M.model_profile(v, d, a, v0, sigma, B, N) for a in (0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 40.0)]
    assert np.array_equal(prof[0], np.full(v.shape, B)), "alpha = 0 must give B everywhere"
    for lighter, darker in zip(prof[:-1], prof[1:]):
        assert np.all(darker <= lighter + 1e-12 * B), "darker with alpha"
    assert all(np.all(p >= -1e-12 * B) and np.all(p <= B) for p in prof), "0 <= I <= B"
    far = np.array([v0 - 0.5 * d - 12 * sigma, v0 + 0.5 * d + 12 * sigma])
    assert np.max(np.abs(M.model_profile(far, d, 40.0, v0, sigma, B, N) - B)) <= 1e-12 * B, "I -> B far away"
    # mu form = alpha form at alpha = mu d / cos(phi): the same function, bit-identical
    phi, mu = math.radians(37.0), 2.3
    a1 = M.alpha_from_mu(mu, d, phi)
    assert abs(a1 - mu * d / math.cos(phi)) <= 4e-16 * a1   # np.cos vs math.cos may differ by 1 ulp
    assert np.array_equal(M.model_profile(v, d, a1, v0, sigma, B, N), M.model_profile(v, d, float(a1), v0, sigma, B, N))
    # D-019 bijection guard: the two fits agree wherever no bound is active
    for (mu, d), (rm, ra) in mu_tie_fits().items():
        if rm.status == "converged" and ra.status == "converged":
            assert abs(rm.d_hat_um - ra.d_hat_um) <= 2e-3 * d, ("d_hat differs", mu, d, rm.d_hat_um, ra.d_hat_um)
            assert abs(rm.rss_gl2 / ra.rss_gl2 - 1) <= 1e-6, ("rss differs", mu, d, rm.rss_gl2, ra.rss_gl2)
        else:
            assert rm.status == "at_bound" and ra.status == "at_bound", ("one form at a bound only", mu, d,
                                                                         rm.status, rm.at_bound, ra.status, ra.at_bound)


def test_contract():
    cfg = cfg_measure()
    v = profile_axis(cfg)
    y = M.model_profile(v, 0.8, 1.0, -0.1, cfg.sigma_fit_um, B, fit_nodes(cfg))
    st = F.start_points(v, y, 0.3, B, cfg)
    lb = np.array([cfg.fit_d_bounds_um[0], cfg.fit_mu_bounds_per_um[0], cfg.fit_v0_bounds_um[0]])
    ub = np.array([cfg.fit_d_bounds_um[1], cfg.fit_mu_bounds_per_um[1], cfg.fit_v0_bounds_um[1]])
    assert st.shape == (len(cfg.fit_multistart_factors), 3) and st.dtype == np.float64
    assert np.all(st >= lb) and np.all(st <= ub), st
    r = F.fit_profile(v, y, 0.3, B, cfg)
    assert isinstance(r, F.FitResult) and r.form == "mu" and r.status in F.FIT_STATUSES
    for name in ("d_hat_um", "mu_hat_per_um", "v0_hat_um", "alpha_hat", "phi_rad", "rss_gl2"):
        val = getattr(r, name)
        assert isinstance(val, float) and math.isfinite(val), name
    for name in ("n_samples", "n_starts", "best_start", "n_converged", "nfev", "n_nodes"):
        assert isinstance(getattr(r, name), int), name
    assert r.n_samples == v.size and r.n_starts == len(cfg.fit_multistart_factors) and 0 <= r.best_start < r.n_starts
    assert r.n_nodes == fit_nodes(cfg) and isinstance(r.at_bound, tuple) and r.rss_gl2 >= 0
    assert abs(r.alpha_hat - r.mu_hat_per_um * r.d_hat_um / math.cos(0.3)) <= 1e-12 * r.alpha_hat
    ra = F.fit_profile_alpha(v, y, 0.3, B, cfg)
    assert ra.form == "alpha" and abs(ra.mu_hat_per_um - ra.alpha_hat * math.cos(0.3) / ra.d_hat_um) <= 1e-12 * ra.mu_hat_per_um
    grid = np.linspace(-1, 1, 12).reshape(3, 4)
    out = M.model_profile(grid, 0.8, 1.0, 0.0, 0.1, B, 64)
    assert out.shape == (3, 4) and out.dtype == np.float64
    assert np.ndim(M.model_profile(0.2, 0.8, 1.0, 0.0, 0.1, B, 64)) == 0
    T = M.transmittance(grid, 0.8, 1.0, 0.0)
    assert T.shape == (3, 4) and np.all((T > 0) & (T <= 1))


def test_determinism():
    cfg = cfg_measure()
    v = profile_axis(cfg)
    rng = np.random.default_rng(SEED)
    y = M.model_profile(v, 1.1, 0.8, 0.05, cfg.sigma_fit_um, B, fit_nodes(cfg)) + rng.normal(0.0, 3.0, v.size)
    assert F.fit_profile(v, y, 0.2, B, cfg) == F.fit_profile(v, y, 0.2, B, cfg)


def test_edge_cases():
    cfg = cfg_measure()
    v = profile_axis(cfg)
    flat = F.fit_profile(v, np.full(v.shape, B), 0.0, B, cfg)
    assert flat.status == "at_bound" and "mu_lo" in flat.at_bound, (flat.status, flat.at_bound)
    wide = F.fit_profile(v, M.model_profile(v, 8.0, 2.0, 0.0, cfg.sigma_fit_um, B, 2 * fit_nodes(cfg)), 0.0, B, cfg)
    assert wide.status == "at_bound" and "d_hi" in wide.at_bound, (wide.status, wide.at_bound)
    # half-depth width of a sampled triangular dip: B - D max(0, 1 - |v - c| / a) -> exactly a
    vt = np.arange(-30, 31) * 0.1
    for c, a in ((0.0, 1.0), (0.3, 0.73), (-0.2, 2.17)):
        It = B - 50.0 * np.clip(1 - np.abs(vt - c) / a, 0, None)
        n_star = int(np.argmin(np.abs(vt - c)))
        It[n_star] = B - 50.0            # make the apex a sample
        vt2 = vt.copy()
        vt2[n_star] = c
        w = F.half_depth_width(vt2, It, B, n_star)
        assert w is not None and abs(w - a) <= 1e-12, (c, a, w)
    assert F.half_depth_width(vt, np.full(vt.shape, B), B, 30) is None
    one_sided = B - 50.0 * np.clip(1 - np.abs(vt + 3.0) / 1.0, 0, None)
    assert F.half_depth_width(vt, one_sided, B, 0) is None
    fixed = F.start_points(v, M.model_profile(v, 0.5, 1.0, 0.0, cfg.sigma_fit_um, B, 64), 0.0, B,
                           cfg_measure(fit_start_rule="fixed"))
    assert np.allclose(fixed[:, 0], [cfg.fit_d0_um * f for f in cfg.fit_multistart_factors])
    # phi close to 90 deg: alpha = mu d / cos(phi) is large for any mu; the fit still returns
    phi = math.radians(89.9)
    y = M.model_profile(v, 0.7, 1.5, 0.0, cfg.sigma_fit_um, B, fit_nodes(cfg))
    r = F.fit_profile(v, y, phi, B, cfg)
    assert r.status == "converged" and abs(r.d_hat_um / 0.7 - 1) <= 1e-6, (r.status, r.d_hat_um)
    # invalid inputs
    bad = [lambda: F.fit_profile(v, np.where(v > 0, np.nan, B), 0.0, B, cfg),
           lambda: F.fit_profile(v[::-1], np.full(v.shape, B), 0.0, B, cfg),
           lambda: F.fit_profile(v, np.full(v.shape, B), 0.5 * math.pi, B, cfg),
           lambda: F.fit_profile(v, np.full(v.shape, B), -0.1, B, cfg),
           lambda: F.fit_profile(v, np.full(v.shape, B), 0.0, 0.0, cfg),
           lambda: F.fit_profile(v[:3], np.full(3, B), 0.0, B, cfg),
           lambda: F.fit_profile(v, np.full(v.size - 1, B), 0.0, B, cfg),
           lambda: F.fit_profile(v[v > 1.5], np.full(int(np.sum(v > 1.5)), B), 0.0, B, cfg),
           lambda: F.start_points(v, np.full(v.shape, B), 0.0, B, cfg_measure(fit_start_rule="other")),
           lambda: M.model_profile(v, 0.0, 1.0, 0.0, 0.1, B, 64),
           lambda: M.model_profile(v, 1.0, -1.0, 0.0, 0.1, B, 64),
           lambda: M.model_profile(v, 1.0, 1.0, 0.0, 0.0, B, 64),
           lambda: M.model_profile(np.array([0.0, np.inf]), 1.0, 1.0, 0.0, 0.1, B, 64),
           lambda: M.quadrature_rule(0),
           lambda: M.alpha_from_mu(1.0, 1.0, 0.5 * math.pi)]
    for k, call in enumerate(bad):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("invalid input %d accepted" % k)
    try:
        F.fit_profile(v, np.full(v.shape, B), 0.0, B, cfg_measure(mu_mode="shared_branch"))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("mu_mode shared_branch must raise NotImplementedError in Block 3")


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


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
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
        except Exception:
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))
    width = max([len(n) for n, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, headline))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
