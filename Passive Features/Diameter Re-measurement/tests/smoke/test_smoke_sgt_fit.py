"""Smoke test for the skewed generalized-t kernel and the seven-parameter fit
at a plane -- Block 12 in specs/SPEC.md (D-052.1-2, D-053 (ii), D-054 (iv)).

The kernel (model/blur_kernels.py, (H1)-(H3)): h_{beta,xi} the generalized-t
core, f( . | kappa) its two-piece skewed form of unit integral and mode 0,
W(kappa) its FWHM, g_{w,kappa}(z) = (W / w) f(W z / w | kappa) the kernel of
FWHM w. The fit (analysis/fit_sgt.py): theta = (d, mu, v0, w, beta, xi,
zeta) by least squares on (H5), unweighted from the Gaussian starts, then
re-weighted by omega_n = exp(-|v_n - v0| / lambda) re-centred on the fit's
own v0 (IRLS), with the status rule of D-053 (ii).

Checks
    test_known_answer   the kernel's mass on [0, X] and [-X, 0] by
                        scipy.integrate.quad against the closed forms through
                        the regularized incomplete beta (xi > 0) and gamma
                        (xi = 0) functions, on a grid of (beta, xi, zeta) with
                        the box corners (1e-11); the unit integral over R by
                        quad with the tail mapped to a regular integrand
                        (1e-10); FWHM = w by brentq on the kernel itself
                        (1e-12 um) and the half-width ratio zeta^2 (1e-10
                        relative); the mode at 0 and the decrease on each
                        side; continuity at xi -> 0 and the expm1 branch;
                        noise-free recovery of theta on a 1.5 um tube
    test_reference      the members against scipy.stats (norm, laplace, t,
                        cauchy, gennorm) rescaled to FWHM w and against
                        blur_kernels.kernel_pdf (1e-12); two-piece members
                        against scipy.stats written as two pieces; the model
                        against a direct convolution by quad, by member class
                        (analytic 1e-7, skewed 1e-4, Laplace 2e-3 grey
                        levels at the fit's N); nesting: the unweighted SGT
                        costs at most the Gaussian of fitted width on the same
                        profile; five draws of the D-052 Monte Carlo (seed
                        20261017) reproduced, the budget stop, a mu = 20
                        collapse and xi = 0 fits among them; the reported cost
                        is Q of (H5)
    test_convergence    the quadrature's observed order at the box corner
                        (d 6 um, w 0.1 um) per member class: exponential for
                        the analytic members, about k + 1 for a jump in the
                        k-th derivative of g at its peak (skew 3, beta 1.5
                        2.5, Laplace 2); the bias the decided N puts on d_hat
                        in the fitted regime (<= 2e-3 um); the IRLS stops by
                        its rule (last step within tolerance, earlier not)
    test_invariants     a gain on I and B_bar; a shift of v by 0.15 um; the
                        mirror v -> -v (v0 -> -v0, zeta -> 1/zeta); the
                        kernel's mirror identity; I_model <= B_bar exactly at
                        N and > -0.02 gl at 16 N over the box; the weights
    test_contract       refusals (shapes, FWHM, profiles); the SGTFit and
                        PlaneSGT contract; the status rule of D-053 (ii); the
                        failure paths (IRLS cap 1, budget 5, no finite start)
    test_determinism    two fit_plane runs: identical numbers
    test_edge_cases     the defaults (D-052.2, D-054 (iv)); config.validate's
                        refusals; overrides and the dictionary round trip of
                        the nested start shapes; the estimator hash moved off
                        the two recorded ones; FAMILIES unchanged; a flat
                        profile

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_sgt_fit.py
        About 3 minutes on one core (the five stored draws take about one).
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure; exit code 1 if any check
        failed, errored or is TODO.

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter import config as C  # noqa: E402
from allen_diameter.analysis import fit as block5  # noqa: E402
from allen_diameter.analysis import fit_sgt as FS  # noqa: E402
from allen_diameter.analysis import fit_variants as FV  # noqa: E402
from allen_diameter.analysis import profiles  # noqa: E402
from allen_diameter.model import blur_kernels as BK  # noqa: E402
from allen_diameter.model import tube_model as TM  # noqa: E402

SEED = 20261018
REPORT_PACKAGES = ("numpy", "scipy")
W_TEST = 0.5                                     # um, the kernel width of the kernel checks (the D-052 truths')
# the shape grid of the kernel checks: beta, xi, zeta including the fit box's corners (D-052.2)
GRID = [(b, x, z) for b in (1.0, 2.0, 4.0, 10.0) for x in (0.0, 1e-3, 0.5, 2.0)
        for z in (math.exp(-0.7), 1.0, 1.5, math.exp(0.7))]
# the D-052 Monte Carlo (checks/estimator_2026-10-15/identifiability_sgt.py, part B, seed 20261017): truths,
# and five draws' stored fits (ident_sgt/identifiability_sgt.json, raw "sgt" and "sgt_l050"), embedded so that
# the test does not read checks/: (d_true, kernel, draw, weighting, (d, mu, v0, w, beta, xi, zeta), rule met,
# the check's bound names). The check names the shapes (p, r, gamma) and the coordinates log10p, r, lngamma.
MC = dict(seed=20261017, phi=0.2, B=180.0, mu=1.0, v0=0.1, w=0.5, n_nodes=848, noise=2.0,
          kernels={"G": (2.0, 0.0, 1.0), "S4": (2.0, 0.5, 1.0), "L": (1.0, 0.0, 1.0), "B4": (4.0, 0.0, 1.0),
                   "K": (2.0, 0.0, 1.5)})
STORED_DRAWS = (
    (0.6, 'G', 0, 'none', (0.6505790086713267, 0.8552067714918783, 0.08326531343298557, 0.3097598881278323, 1.192321280352694, 1.2390697685327749e-24, 1.0639999880006876), True, ('r_lo',)),
    (0.6, 'G', 0, 'irls', (0.6544427239027227, 0.8474041196639747, 0.09519657262127797, 0.27901377979436914, 1.103894808854585, 3.8416248523913315e-13, 1.0106898257723902), True, ('r_lo',)),
    (0.6, 'S4', 1, 'none', (0.23025170423560634, 19.999999999999996, 0.06443327672043828, 0.6855513156713294, 2.814195014854455, 0.7061547010699174, 1.0756189317297087), True, ('mu_hi',)),
    (0.6, 'S4', 1, 'irls', (0.23021289869997036, 19.999999999999996, 0.05860730382546735, 0.6852284578301461, 2.799985798237416, 0.6961424153853071, 1.0951665045637815), True, ('mu_hi',)),
    (1.5, 'G', 1, 'none', (1.4963220085957312, 1.0290591021112179, 0.13110587765112006, 0.5401650234195104, 2.1785304451363543, 1.5066002081443835e-21, 0.9114778597712114), True, ('r_lo',)),
    (1.5, 'G', 1, 'irls', (1.4979584923100908, 1.02492691747146, 0.1392219530143215, 0.5525614322482845, 2.2971755636976128, 4.3665318010357026e-16, 0.8861295614490494), True, ('r_lo',)),
    (1.5, 'S4', 39, 'none', (1.465280019520962, 1.0960839723815654, 0.09581920914038294, 0.36002804408753075, 1.1281489447223414, 0.06472340190416254, 1.0001533019399254), True, ()),
    (1.5, 'S4', 39, 'irls', (1.4640840250609795, 1.094375489177722, 0.08426606168579592, 0.3749123526215089, 1.1008306952592148, 0.00029853622219953315, 1.0309892467364556), False, ()),
    (1.5, 'B4', 0, 'none', (1.495157622867926, 0.9965215794158678, 0.11586607428886736, 0.4445446372086881, 2.9841430432058895, 0.20773032408544406, 0.9386009529833228), True, ()),
    (1.5, 'B4', 0, 'irls', (1.4911867643499332, 1.0085695480707524, 0.09617398220516732, 0.4434477388732854, 4.851565222416668, 1.2030724346231925, 1.0131948049014983), True, ()),
)
CHECK_NAMES = {"d": "d", "mu": "mu", "v0": "v0", "w": "w", "log10p": "beta", "r": "xi", "lngamma": "zeta"}


# --------------------------------------------------------------- helpers ---

def _measure(**over):
    m = C.default_config().measure
    return dataclasses.replace(m, **over) if over else m


def _offsets(cfg):
    return profiles.profile_offsets(cfg)


def _profile(cfg, theta, phi=0.2, B=180.0, noise=0.0, rng=None, n_nodes=None):
    """Block 5's offsets and a profile of the SGT model at theta = (d, mu, v0, w, beta, xi, zeta)."""
    v = _offsets(cfg)
    n = FS.n_nodes_for(cfg) if n_nodes is None else n_nodes
    I = FS.sgt_model(v, FS.to_q(*theta), phi, B, n)
    if noise > 0:
        I = I + rng.normal(0.0, noise, v.size)
    return v, I, phi, B


def _W_closed(beta, xi, zeta):
    """(H3)'s W written out here independently of blur_kernels (expm1 form)."""
    s = zeta + 1.0 / zeta
    if xi <= BK.SGT_XI_ZERO:
        return s * math.log(2.0) ** (1.0 / beta)
    e = xi * beta / (beta + xi)
    return s * (math.expm1(e * math.log(2.0)) / xi) ** (1.0 / beta)


def _H(y, beta, xi):
    """int_0^y h_{beta,xi}: (1/2) I_t(1/beta, 1/xi), t = xi y^beta / (1 + xi y^beta), or (1/2) P(1/beta, y^beta)."""
    from scipy import special
    if xi <= BK.SGT_XI_ZERO:
        return 0.5 * float(special.gammainc(1.0 / beta, y ** beta))
    t = xi * y ** beta / (1.0 + xi * y ** beta)
    return 0.5 * float(special.betainc(1.0 / beta, 1.0 / xi, t))


def _half_mass(g, beta, xi, side, L):
    """int over one side of R of the kernel g by quad: [0, L] directly, [L, inf) mapped x = L t^-m with
    m = max(1, 2 xi / beta), so that a tail |x|^-(1 + beta/xi) becomes an integrand ~ t^(m beta / xi - 1),
    regular (exponent >= 1); m = 1 also for the exponential tails (xi = 0), and m >= 1 keeps the map from
    collapsing [L, 2L] below the smallest float when xi is tiny."""
    import warnings
    from scipy import integrate
    sgn = 1.0 if side > 0 else -1.0
    f = lambda x: float(g(sgn * x))  # noqa: E731
    m = max(1.0, 2.0 * xi / beta) if xi > BK.SGT_XI_ZERO else 1.0
    with warnings.catch_warnings():
        # quad warns that roundoff keeps it from its 1e-15 target; the check asks 1e-10 of the total
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        near = integrate.quad(f, 0.0, L, limit=500, epsabs=1e-15, epsrel=1e-14)[0]
        tail = integrate.quad(lambda t: f(L * t ** (-m)) * L * m * t ** (-m - 1.0) if t > 0 else 0.0, 0.0, 1.0,
                              limit=500, epsabs=1e-15, epsrel=1e-14)[0]
    return near + tail


def _quad_model(v, d, alpha, v0, g, B):
    """B (1 - int (1 - T(u)) g(v - v0 - u) du) by scipy.integrate.quad, the kernel's peak as a break point."""
    from scipy import integrate
    out = []
    for vi in np.atleast_1d(v):
        f = lambda u: (1.0 - float(TM.transmittance(u, d, alpha, 0.0))) * float(g(vi - v0 - u))  # noqa: E731
        pts = [x for x in (vi - v0,) if -0.5 * d < x < 0.5 * d]
        val = integrate.quad(f, -0.5 * d, 0.5 * d, points=pts or None, limit=500, epsabs=1e-14, epsrel=1e-13)[0]
        out.append(B * (1.0 - val))
    return np.array(out)


def _mc_draw(cfg, d_t, kname, n):
    """One profile of the D-052 Monte Carlo, made as identifiability_sgt.run_mc made it."""
    v = _offsets(cfg)
    k = MC["kernels"][kname]
    I0 = TM.model_profile_kernel(v, d_t, MC["mu"] * d_t / math.cos(MC["phi"]), MC["v0"],
                                 BK.sgt_kernel_pdf(MC["w"], *k), MC["B"], MC["n_nodes"])
    rng = np.random.default_rng([MC["seed"], int(round(d_t * 100)), sorted(MC["kernels"]).index(kname), n])
    return v, I0 + rng.normal(0.0, MC["noise"], v.size)


def _theta(f):
    return np.array([f.d_um, f.mu_per_um, f.v0_um, f.w_um, f.beta, f.xi, f.zeta])


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    from scipy import integrate, optimize
    worst_cdf, worst_area, worst_fwhm, worst_ratio = 0.0, 0.0, 0.0, 0.0
    for beta, xi, zeta in GRID:
        g = BK.sgt_kernel_pdf(W_TEST, beta, xi, zeta)
        W = _W_closed(beta, xi, zeta)
        a = W_TEST / W
        s = zeta + 1.0 / zeta
        # mass on [0, X] and [-X, 0] against the closed forms (tolerance: quad on a finite range of a smooth
        # integrand, epsabs 1e-14; betainc / gammainc accurate to ~1e-15 relative)
        for X in (0.1, 0.5, 2.0):
            right = integrate.quad(lambda z: float(g(z)), 0.0, X, limit=500, epsabs=1e-14, epsrel=1e-13)[0]
            left = integrate.quad(lambda z: float(g(z)), -X, 0.0, limit=500, epsabs=1e-14, epsrel=1e-13)[0]
            r_ex = (2.0 * zeta / s) * _H(X / (a * zeta), beta, xi)
            l_ex = (2.0 / (s * zeta)) * _H(zeta * X / a, beta, xi)
            worst_cdf = max(worst_cdf, abs(right - r_ex), abs(left - l_ex))
        # unit integral over R (the tail mapped to a regular integrand; tolerance 1e-10, the handoff's oracle)
        total = _half_mass(g, beta, xi, +1, 5.0 * W_TEST) + _half_mass(g, beta, xi, -1, 5.0 * W_TEST)
        worst_area = max(worst_area, abs(total - 1.0))
        # FWHM = w and the half-width ratio zeta^2 by brentq on the kernel itself
        g0 = float(g(0.0))
        span = W_TEST
        while float(g(span)) > 0.5 * g0 or float(g(-span)) > 0.5 * g0:
            span *= 2.0
        hr = optimize.brentq(lambda z: float(g(z)) - 0.5 * g0, 0.0, span, xtol=1e-15, rtol=4.0 * np.finfo(float).eps)
        hl = optimize.brentq(lambda z: float(g(-z)) - 0.5 * g0, 0.0, span, xtol=1e-15, rtol=4.0 * np.finfo(float).eps)
        worst_fwhm = max(worst_fwhm, abs(hr + hl - W_TEST))
        worst_ratio = max(worst_ratio, abs(hr / hl / zeta ** 2 - 1.0))
        # mode at 0, non-increasing away from it on each side (strictly lower at the end); at beta = 10 the top
        # is flat to float64 precision near 0, so a strict decrease at every step cannot hold there
        z = np.linspace(0.0, 3.0 * W_TEST, 601)
        gr, gl = g(z), g(-z)
        assert np.all(np.diff(gr) <= 0) and np.all(np.diff(gl) <= 0), ("increasing away from 0", beta, xi, zeta)
        assert gr[-1] < 0.5 * g0 and gl[-1] < 0.5 * g0 and g0 >= gr.max() and g0 >= gl.max()
    assert worst_cdf <= 1e-11, "mass on [0, X] / [-X, 0] off the closed form by %.3e" % worst_cdf
    assert worst_area <= 1e-10, "unit integral off by %.3e" % worst_area
    assert worst_fwhm <= 1e-12, "FWHM off w by %.3e um" % worst_fwhm
    assert worst_ratio <= 1e-10, "half-width ratio off zeta^2 by %.3e (relative)" % worst_ratio
    # the right mass is zeta^2 / (1 + zeta^2) (closed form, from H(inf) = 1/2)
    for zeta in (0.5, 1.0, 2.0):
        g = BK.sgt_kernel_pdf(W_TEST, 2.0, 0.5, zeta)
        right = _half_mass(g, 2.0, 0.5, +1, 5.0 * W_TEST)
        assert abs(right - zeta ** 2 / (1.0 + zeta ** 2)) <= 1e-10, (zeta, right)
    print("    kernel: mass vs closed form %.1e, unit integral %.1e, FWHM %.1e um, ratio %.1e over %d shapes"
          % (worst_cdf, worst_area, worst_fwhm, worst_ratio, len(GRID)))
    # continuity at xi -> 0 and the expm1 branch: at xi = 1e-10 the true W differs from the xi = 0 value by
    # O(xi) relative; the code's W stays within 1e-9, while the naive 2^e - 1 loses ~ulp / (e ln 2) ~ 3e-6
    for beta in (1.0, 2.0, 10.0):
        w0 = BK.sgt_unit_fwhm(beta, 0.0, 1.0)
        xi = 1e-10
        w_code = BK.sgt_unit_fwhm(beta, xi, 1.0)
        e = xi * beta / (beta + xi)
        w_naive = 2.0 * ((2.0 ** e - 1.0) / xi) ** (1.0 / beta)
        assert abs(w_code / w0 - 1.0) <= 1e-9, (beta, w_code, w0)
        assert abs(w_naive / w0 - 1.0) > 1e-8, "the naive form should lose precision here (else the branch is untested)"
        z = np.linspace(-2.0, 2.0, 801)
        dg = float(np.max(np.abs(BK.sgt_kernel_pdf(W_TEST, beta, 1e-9, 1.0)(z) - BK.sgt_kernel_pdf(W_TEST, beta, 0.0, 1.0)(z))))
        assert dg <= 1e-6, ("kernel continuity at xi -> 0", beta, dg)   # O(xi) x the kernel's scale (~2/um)
    # noise-free recovery on a 1.5 um tube (the truth is an exact zero of Q, inside the box, at the fit's own N)
    cfg = _measure()
    truth = (1.5, 1.0, 0.1, 0.5, 2.0, 0.5, 1.2)
    v, I, phi, B = _profile(cfg, truth)
    res = FS.fit_plane(v, I, phi, B, cfg)
    for f in (res.unweighted, res.final):
        err = np.abs(_theta(f) - np.array(truth))
        print("    noise-free 1.5 um, %s: |theta - truth| = %s, status %s, rss %.2e gl^2"
              % (f.weighting, np.array2string(err, precision=2), f.status, f.rss_gl2))
        assert f.status == "converged", (f.weighting, f.status, f.at_bound)
        # tolerances: the solver stops on ftol/xtol 1e-10 near a zero residual; the identified directions
        # (d, v0, w) to 1e-6, mu to 1e-5, the shapes (flatter directions of Q) to 1e-3
        assert err[0] <= 1e-6 and err[2] <= 1e-6 and err[3] <= 1e-6, err
        assert err[1] <= 1e-5 and np.all(err[4:] <= 1e-3), err
        assert f.rss_gl2 <= 1e-12, f.rss_gl2


def test_reference():
    from scipy import stats
    z = np.linspace(-3.0, 3.0, 6001)
    sig = W_TEST / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    worst = 0.0
    refs = [((2.0, 0.0, 1.0), stats.norm.pdf(z, scale=sig)),
            ((1.0, 0.0, 1.0), stats.laplace.pdf(z, scale=W_TEST / (2.0 * math.log(2.0)))),
            ((2.0, 2.0, 1.0), stats.cauchy.pdf(z, scale=W_TEST / 2.0))]
    nu = 4.0
    refs.append(((2.0, 2.0 / nu, 1.0), stats.t.pdf(z, nu, scale=W_TEST / (2.0 * math.sqrt(nu * (2.0 ** (2.0 / (nu + 1.0)) - 1.0))))))
    for beta in (1.5, 4.0, 10.0):
        refs.append(((beta, 0.0, 1.0), stats.gennorm.pdf(z, beta, scale=W_TEST / (2.0 * math.log(2.0) ** (1.0 / beta)))))
    for shape, ref in refs:
        worst = max(worst, float(np.max(np.abs(BK.sgt_kernel_pdf(W_TEST, *shape)(z) - ref))))
    # tolerance: kernel values are at most ~2/um, so 1e-12 is a few hundred ulps of float64 roundoff
    assert worst <= 1e-12, "members vs scipy.stats: %.3e" % worst
    worst_bk = 0.0
    pairs = [((2.0, 0.0, 1.0), ("gauss", None)), ((2.0, 2.0, 1.0), ("lorentz", None))]
    pairs += [((2.0, 2.0 / nu, 1.0), ("student", nu)) for nu in (1.0, 1.5, 4.0, 100.0)]
    pairs += [((b, 0.0, 1.0), ("gennorm", b)) for b in (1.0, 4.0)]
    for shape, (fam, s) in pairs:
        worst_bk = max(worst_bk, float(np.max(np.abs(BK.sgt_kernel_pdf(W_TEST, *shape)(z) - BK.kernel_pdf(fam, W_TEST, s)(z)))))
    assert worst_bk <= 1e-12, "members vs blur_kernels.kernel_pdf: %.3e" % worst_bk
    # two-piece members against scipy.stats written as two pieces: h_{beta,0} = gennorm(beta); h_{2,xi} = sqrt(2) t(sqrt(2) x, 2/xi)
    worst_2p = 0.0
    for (beta, xi, zeta) in ((4.0, 0.0, 1.5), (1.0, 0.0, 0.6), (2.0, 0.5, 1.5), (2.0, 2.0, 0.7)):
        s = zeta + 1.0 / zeta
        a = W_TEST / _W_closed(beta, xi, zeta)
        h = (lambda x: stats.gennorm.pdf(x, beta)) if xi == 0.0 else (lambda x: math.sqrt(2.0) * stats.t.pdf(math.sqrt(2.0) * x, 2.0 / xi))
        ref = np.where(z >= 0, (2.0 / s) * h(z / (a * zeta)) / a, (2.0 / s) * h(zeta * z / a) / a)
        worst_2p = max(worst_2p, float(np.max(np.abs(BK.sgt_kernel_pdf(W_TEST, beta, xi, zeta)(z) - ref))))
        # sgt_unit_pdf is f of (H2): g = f(z / a) / a
        worst_2p = max(worst_2p, float(np.max(np.abs(BK.sgt_unit_pdf(z / a, beta, xi, zeta) / a - ref))))
    assert worst_2p <= 1e-12, "two-piece members vs scipy.stats: %.3e" % worst_2p
    print("    members: scipy.stats %.1e, kernel_pdf %.1e, two-piece %.1e" % (worst, worst_bk, worst_2p))
    # the model against a direct convolution by quad, at the fit's N (848). Gauss-Legendre converges
    # exponentially only on the analytic members (zeta = 1 with beta = 2 or 4); the two-piece skew (a jump
    # of g'' at the peak), a non-even beta and the Laplace (a kink) converge algebraically (test_convergence),
    # so each class gets the tolerance its measured error at w = 0.5 um, d <= 1.5 um supports (x10 margin)
    cfg = _measure()
    n = FS.n_nodes_for(cfg)
    vq = np.array([-1.2, -0.6, -0.2, 0.0, 0.1, 0.35, 0.9])
    classes = (("analytic", 1e-7, ((0.6, 1.0, (2.0, 0.5, 1.0)), (1.5, 1.0, (2.0, 0.5, 1.0)), (1.5, 2.0, (4.0, 0.0, 1.0)),
                                   (0.6, 0.5, (2.0, 2.0, 1.0)))),
               ("skewed", 1e-4, ((0.6, 1.0, (2.0, 0.5, 1.3)), (1.5, 1.0, (2.0, 0.5, 1.3)), (1.5, 2.0, (4.0, 0.0, 0.7)))),
               ("Laplace", 2e-3, ((1.5, 1.0, (1.0, 0.0, 1.0)),)))     # recorded 8.3e-4 gl on 2026-10-15
    msg = []
    for label, tol, cases in classes:
        worst_m = 0.0
        for (d, mu, kap) in cases:
            q = FS.to_q(d, mu, 0.1, 0.5, *kap)
            mine = FS.sgt_model(vq, q, 0.2, 180.0, n)
            ref = _quad_model(vq, d, TM.alpha_from_mu(mu, d, 0.2), 0.1, BK.sgt_kernel_pdf(0.5, *kap), 180.0)
            worst_m = max(worst_m, float(np.max(np.abs(mine - ref))))
        assert worst_m <= tol, "model vs direct quadrature, %s members: %.3e gl > %.0e" % (label, worst_m, tol)
        msg.append("%s %.1e" % (label, worst_m))
    print("    model vs quad at N = %d (gl): %s" % (n, ", ".join(msg)))
    # nesting: the unweighted SGT costs at most the Gaussian of fitted width, which it contains (2, 0, 1)
    rng = np.random.default_rng(SEED)
    for d in (0.6, 1.5):
        v, I, phi, B = _profile(cfg, (d, 1.0, 0.1, 0.5, 2.0, 0.5, 1.0), noise=2.0, rng=rng)
        res = FS.fit_plane(v, I, phi, B, cfg)
        g = res.gauss_fitted
        # slack 1e-6: the starts are clipped START_MARGIN inside the box, so the Gaussian start sits at
        # xi = 1e-9, whose cost exceeds the Gaussian's by about 1e-7 relative; trf only descends from there
        assert res.unweighted.rss_gl2 <= g.rss_gl2 * (1.0 + 1e-6), (d, res.unweighted.rss_gl2, g.rss_gl2)
        print("    nesting d %.1f um: rss SGT %.4f <= Gaussian of fitted width %.4f gl^2"
              % (d, res.unweighted.rss_gl2, g.rss_gl2))
        q_g = FS.to_q(g.d_um, g.mu_per_um, g.v0_um, g.fwhm_um, 2.0, 0.0, 1.0)
        diff = float(np.max(np.abs(FS.sgt_model(v, q_g, phi, B, n) - g.model_gl)))
        assert diff <= 1e-10, "SGT (2, 0, 1) vs the Gaussian variant's model: %.3e gl" % diff
        # the reported cost is Q of (H5): unweighted with omega = 1; IRLS with the weights it was solved with
        u, f = res.unweighted, res.final
        assert abs(FS.objective(v, I, phi, B, np.array(u.q), None, n) - u.cost) <= 1e-12 * max(u.cost, 1e-300)
        v0_prev = f.irls_path[-2][1] if f.n_irls >= 2 else u.v0_um
        om = FV.centre_weights(v, v0_prev, cfg.sgt_lambda_um)
        assert abs(FS.objective(v, I, phi, B, np.array(f.q), om, n) - f.cost) <= 1e-12 * max(f.cost, 1e-300)
    # the five stored draws of the D-052 Monte Carlo, reproduced through fit_plane
    by_draw = {}
    for d_t, kname, n_draw, wt, ref, rule_ok, names in STORED_DRAWS:
        key = (d_t, kname, n_draw)
        if key not in by_draw:
            v, I = _mc_draw(cfg, d_t, kname, n_draw)
            by_draw[key] = FS.fit_plane(v, I, MC["phi"], MC["B"], cfg)
        res = by_draw[key]
        f = res.unweighted if wt == "none" else res.final
        err = np.abs(_theta(f) - np.array(ref))
        mine_names = sorted(f.at_bound)
        want = sorted(CHECK_NAMES[x.rsplit("_", 1)[0]] + "_" + x.rsplit("_", 1)[1] for x in names)
        assert mine_names == want, (key, wt, mine_names, want)
        if wt == "none":
            assert f.solve_ok == rule_ok, (key, f.solve_ok)
        else:
            assert f.irls_ok == rule_ok, (key, f.irls_ok)
        # tolerance: the same arithmetic as the check; 1e-6 um on d leaves room for platform roundoff only.
        # Draw 39's weighted solve stops at the 3000 budget, a path-dependent point: 1e-4 um there.
        tol_d = 1e-4 if not rule_ok else 1e-6
        assert err[0] <= tol_d and err[2] <= max(tol_d, 1e-6), (key, wt, err)
        print("    D-052 MC %g %s draw %d %-4s: max |diff| %.1e, status %s %s"
              % (d_t, kname, n_draw, wt, float(err.max()), f.status, ",".join(f.at_bound)))
    # D-053 (ii) on the stored fits: xi = 0 keeps converged, mu = 20 is at_bound, the budget stop failed
    assert by_draw[(0.6, "G", 0)].final.status == "converged" and "xi_lo" in by_draw[(0.6, "G", 0)].final.at_bound
    assert by_draw[(0.6, "S4", 1)].final.status == "at_bound" and "mu_hi" in by_draw[(0.6, "S4", 1)].final.at_bound
    f39 = by_draw[(1.5, "S4", 39)].final
    assert f39.status == "failed" and f39.irls_ok is False and f39.n_irls == 1 and f39.nfev == 3000, f39


def test_convergence():
    # the model's quadrature in N at the hardest case of the box (d = d_hi = 6 um, w = w_lo = 0.1 um), the
    # error against 16 N. Expected [reasoning; Gauss-Legendre on a piecewise-smooth integrand, textbook from
    # memory]: exponential for the analytic members, order k + 1 for a jump in the k-th derivative of g at its
    # peak: 3 for the two-piece skew (g''), 2 for the Laplace (g'), 2.5 for beta = 1.5 (|x|^1.5)
    cfg = _measure()
    n = FS.n_nodes_for(cfg)
    v = np.linspace(-3.0, 3.0, 241)
    orders = {}
    for label, kap, lo, hi in (("analytic (2, 0.5, 1)", (2.0, 0.5, 1.0), 8.0, None),
                               ("skewed (2, 0.5, 1.3)", (2.0, 0.5, 1.3), 2.6, 3.8),
                               ("beta 1.5", (1.5, 0.0, 1.0), 2.1, 3.0),
                               ("Laplace", (1.0, 0.0, 1.0), 1.6, 2.5)):
        q = FS.to_q(6.0, 0.5, 0.05, 0.1, *kap)
        ref = FS.sgt_model(v, q, 0.2, 180.0, 16 * n)
        e2, e1 = (float(np.max(np.abs(FS.sgt_model(v, q, 0.2, 180.0, k) - ref))) for k in (n // 2, n))
        p = math.log(e2 / max(e1, 1e-300), 2.0)
        orders[label] = (e1, p)
        assert p >= lo and (hi is None or p <= hi), (label, e2, e1, p)
    assert orders["analytic (2, 0.5, 1)"][0] <= 1e-6, orders
    print("    quadrature at the box corner, error at N (gl) and order N/2 -> N: %s"
          % "; ".join("%s %.1e, %.1f" % (k, e, p) for k, (e, p) in orders.items()))
    # what the rule costs the fit in the fitted regime: noise-free profiles of a 16 N truth fitted at N
    # (D-052.2's rule); measured 2026-10-18 at most 7.5e-4 um over d 0.6-5.5 um, w 0.3-0.5 um, four shapes
    # (check_quadrature_bias.py); tolerance 2e-3 um, against a Monte Carlo SD of d_hat of 0.011-0.015 um
    for d, w, kap in ((5.5, 0.5, (1.0, 0.0, 1.0)), (3.0, 0.3, (2.0, 0.5, 1.5))):
        vv = _offsets(cfg)
        qt = FS.to_q(d, 1.0, 0.1, w, *kap)
        I = FS.sgt_model(vv, qt, 0.2, 180.0, 16 * n)
        f = FS.fit_plane(vv, I, 0.2, 180.0, cfg).final
        assert f.status == "converged" and abs(f.d_um - d) <= 2e-3, (d, w, kap, f.d_um, f.status)
        print("    quadrature bias of d_hat, d %.1f um, w %.1f um, shape %s: %+.1e um" % (d, w, kap, f.d_um - d))
    # the IRLS stops by its rule: its last step within the tolerance, every earlier one not
    rng = np.random.default_rng(SEED + 1)
    for d in (0.6, 1.5):
        v, I, phi, B = _profile(cfg, (d, 1.0, 0.1, 0.5, 2.0, 0.5, 1.1), noise=2.0, rng=rng)
        res = FS.fit_plane(v, I, phi, B, cfg)
        f, u = res.final, res.unweighted
        assert f.irls_ok, (d, f.status, f.message)
        prev = [(u.d_um, u.v0_um)] + list(f.irls_path[:-1])
        steps = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(f.irls_path, prev)]
        assert steps[-1] <= cfg.sgt_irls_tol_um, steps
        assert all(s > cfg.sgt_irls_tol_um for s in steps[:-1]), steps
        print("    IRLS d %.1f um: %d solves, steps %s um" % (d, f.n_irls, ", ".join("%.1e" % s for s in steps)))


def test_invariants():
    cfg = _measure()
    rng = np.random.default_rng(SEED + 2)
    truth = (1.5, 1.0, 0.1, 0.5, 2.0, 0.5, 1.25)
    v, I, phi, B = _profile(cfg, truth, noise=2.0, rng=rng)
    base = FS.fit_plane(v, I, phi, B, cfg)
    th0 = _theta(base.final)
    # a gain on I and B_bar (the residuals are divided by B_bar); tolerance 1e-6: roundoff of the scaled model
    # perturbs the solver's path in Q's flat directions by far less
    g = FS.fit_plane(v, I * 1.37, phi, B * 1.37, cfg)
    e_g = np.abs(_theta(g.final) - th0)
    assert np.all(e_g[:4] <= 1e-6) and np.all(e_g[4:] <= 1e-4), ("gain", e_g)
    # a shift of v by 0.15 um shifts v0 and keeps the rest
    s = FS.fit_plane(v + 0.15, I, phi, B, cfg)
    e_s = np.abs(_theta(s.final) - th0 - np.array([0, 0, 0.15, 0, 0, 0, 0]))
    assert np.all(e_s[:4] <= 1e-6) and np.all(e_s[4:] <= 1e-4), ("shift", e_s)
    # the mirror v -> -v (the offsets are symmetric): v0 -> -v0, zeta -> 1/zeta, the rest kept
    mres = FS.fit_plane(v, I[::-1].copy(), phi, B, cfg)
    th_m = _theta(mres.final)
    exp_m = th0 * np.array([1, 1, -1, 1, 1, 1, 1])
    exp_m[6] = 1.0 / th0[6]
    e_m = np.abs(th_m - exp_m)
    assert np.all(e_m[:4] <= 1e-6) and np.all(e_m[4:] <= 1e-4), ("mirror", e_m)
    print("    gain %.1e, shift %.1e, mirror %.1e (max |delta theta|)" % (e_g.max(), e_s.max(), e_m.max()))
    # the kernel's mirror identity g(z | beta, xi, zeta) = g(-z | beta, xi, 1/zeta): 1/(1/zeta) differs from
    # zeta by an ulp, an error in ln g of about beta |u|^beta 1e-16, so the relative check (1e-12) is made where
    # g >= 1e-8 of its peak (|u|^beta <= ~20 there)
    z = np.linspace(-2.0, 2.0, 801)
    for beta, xi, zeta in GRID:
        a = BK.sgt_kernel_pdf(W_TEST, beta, xi, zeta)(z)
        b = BK.sgt_kernel_pdf(W_TEST, beta, xi, 1.0 / zeta)(-z)
        sel = a >= 1e-8 * a.max()
        assert np.all(np.abs(a[sel] - b[sel]) <= 1e-12 * a[sel]), (beta, xi, zeta)
    # I_model <= B_bar holds exactly at any N (every quadrature term of the dip is >= 0); I_model > 0 holds for
    # the continuum model only: at N the box corner (d 6 um, w 0.1 um, beta 10, mu 20) reaches -7.5 gl (the
    # algebraic convergence of test_convergence), at 16 N -5.2e-3 gl (scan of 2026-10-18): checked at 16 N with
    # tolerance 0.02 gl. Random parameters over the whole box, plus those corners.
    lb, ub = FS.box(cfg)
    n = FS.n_nodes_for(cfg)
    vv = _offsets(cfg)
    qs = [lb + (ub - lb) * rng.random(lb.size) for _ in range(30)]
    qs += [np.array([6.0, mu, 0.0, 0.1, b, 0.0, z]) for b in (0.0, 1.0) for z in (-0.7, 0.7) for mu in (0.5, 20.0)]
    for q in qs:
        assert np.all(FS.sgt_model(vv, q, 0.3, 200.0, n) <= 200.0 * (1.0 + 1e-15)), q
        assert FS.sgt_model(vv, q, 0.3, 200.0, 16 * n).min() >= -0.02, q
    # the weights: 1 at v0, symmetric about it
    om = FV.centre_weights(np.array([-0.4, 0.1, 0.6]), 0.1, cfg.sgt_lambda_um)
    assert om[1] == 1.0 and abs(om[0] - om[2]) <= 1e-15 and abs(om[0] - math.exp(-1.0)) <= 1e-15, om


def test_contract():
    cfg = _measure()
    for bad in ((0.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (float("nan"), 0.0, 1.0), (2.0, -0.1, 1.0), (2.0, float("inf"), 1.0),
                (2.0, 0.0, 0.0), (2.0, 0.0, -1.0)):
        for call in (lambda: BK.check_sgt_shape(*bad), lambda: BK.sgt_kernel_pdf(0.5, *bad)):
            try:
                call()
            except ValueError:
                pass
            else:
                raise AssertionError("shape %r accepted" % (bad,))
    for w in (0.0, -0.5, float("nan"), float("inf")):
        try:
            BK.sgt_kernel_pdf(w, 2.0, 0.0, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("FWHM %r accepted" % (w,))
    rng = np.random.default_rng(SEED + 3)
    v, I, phi, B = _profile(cfg, (1.2, 1.0, 0.1, 0.5, 2.0, 0.5, 1.0), noise=2.0, rng=rng)
    for args in ((v[::-1], I, phi, B), (v, np.where(np.arange(v.size) == 3, np.nan, I), phi, B), (v, I, 0.5 * math.pi, B),
                 (v, I, phi, 0.0), (v, I[:-1], phi, B)):
        try:
            FS.fit_plane(*args, cfg)
        except ValueError:
            pass
        else:
            raise AssertionError("fit_plane accepted a bad profile")
    res = FS.fit_plane(v, I, phi, B, cfg)
    assert isinstance(res, FS.PlaneSGT) and isinstance(res.gauss_fitted, FV.VariantFit)
    allowed = {n + s for n in FS.BOUND_NAMES for s in ("_lo", "_hi")}
    for f, wt in ((res.final, "irls"), (res.unweighted, "none")):
        assert f.weighting == wt and f.weighting in FS.WEIGHTINGS
        assert f.status in block5.FIT_STATUSES and set(f.at_bound) <= allowed, (f.status, f.at_bound)
        assert len(f.q) == len(FS.PARAM_NAMES) and np.all(np.isfinite(f.q))
        assert f.model_gl.shape == v.shape and f.model_gl.dtype == np.float64 and np.all(np.isfinite(f.model_gl))
        assert len(f.irls_path) == f.n_irls and f.n_nodes == FS.n_nodes_for(cfg) == 848
        assert abs(f.alpha_hat - f.mu_per_um * f.d_um / math.cos(phi)) <= 1e-12 * max(1.0, f.alpha_hat)
        assert np.allclose(FS.fit_curve(f, v, B), f.model_gl, rtol=0, atol=1e-12)
        assert np.allclose(FS.from_q(FS.to_q(*_theta(f))), _theta(f), rtol=1e-14, atol=1e-300)
    assert res.unweighted.irls_ok is None and res.final.irls_ok is True and res.final.status == "converged"
    assert res.final.n_irls >= 2, "the cap-1 check below needs a profile whose IRLS takes 2 or more solves"
    # the status rule (D-053 (ii)): only d, mu, v0, w at a bound, or the rule not met
    assert FS._status(True, ("xi_lo",)) == "converged" and FS._status(True, ("beta_hi", "zeta_lo")) == "converged"
    assert FS._status(True, ("w_lo",)) == "at_bound" and FS._status(True, ("mu_hi", "xi_lo")) == "at_bound"
    assert FS._status(True, ("d_lo",)) == "at_bound" and FS._status(True, ("v0_hi",)) == "at_bound"
    assert FS._status(False, ()) == "failed" and FS._status(False, ("xi_lo",)) == "failed"
    # failure paths: the IRLS cap reached (1 solve), the budget (5 evaluations), no finite start
    c1 = _measure(sgt_irls_max_iter=1)
    r1 = FS.fit_plane(v, I, phi, B, c1)
    assert r1.final.n_irls == 1 and r1.final.irls_ok is False and r1.final.status == "failed", r1.final
    assert r1.final.solve_ok, "the cap is not a failed solve"
    c5 = _measure(sgt_max_nfev=5)
    r5 = FS.fit_plane(v, I, phi, B, c5)
    assert r5.unweighted.solve_ok is False and r5.unweighted.status == "failed", r5.unweighted
    assert r5.final.irls_ok is False and r5.final.status == "failed" and r5.final.solve_ok is False, r5.final
    nan_start = dataclasses.replace(res.unweighted, q=(float("nan"),) * 7)
    rn = FS.fit_irls(v, I, phi, B, cfg, nan_start)
    assert rn.status == "failed" and rn.irls_ok is False and np.isnan(rn.d_um) and rn.n_irls == 0
    ru = FS.fit_unweighted(v, I, phi, B, cfg, [])
    assert ru.status == "failed" and np.isnan(ru.d_um) and ru.n_starts == 0
    nan_fit = dataclasses.replace(res.gauss_fitted, d_um=float("nan"))
    sp = FS.start_points(nan_fit, res.gauss_fixed, cfg)
    assert [s[0] for s in sp] == ["gauss_fixed"], [s[0] for s in sp]
    sp = FS.start_points(res.gauss_fitted, res.gauss_fixed, cfg)
    assert [s[0] for s in sp] == ["gauss_fitted:%d" % k for k in range(4)] + ["gauss_fixed"]
    assert abs(sp[-1][1][3] - cfg.sgt_start_w_um) <= 0 and abs(10.0 ** sp[0][1][4] - 2.0) <= 1e-15


def test_determinism():
    cfg = _measure()
    rng = np.random.default_rng(SEED + 4)
    v, I, phi, B = _profile(cfg, (0.9, 1.5, -0.05, 0.45, 3.0, 0.3, 0.8), noise=2.0, rng=rng)
    a = FS.fit_plane(v, I, phi, B, cfg)
    b = FS.fit_plane(v, I, phi, B, cfg)
    for fa, fb in ((a.final, b.final), (a.unweighted, b.unweighted)):
        assert fa.q == fb.q and fa.irls_path == fb.irls_path and fa.nfev == fb.nfev and fa.status == fb.status
        assert np.array_equal(fa.model_gl, fb.model_gl)


def test_edge_cases():
    m = C.MeasureConfig()
    assert m.sgt_w_bounds_um == (0.1, 2.0) and m.sgt_beta_bounds == (1.0, 10.0) and m.sgt_xi_bounds == (0.0, 2.0)
    assert m.sgt_ln_zeta_bounds == (-0.7, 0.7) and m.sgt_lambda_um == 0.5                        # D-052.2, D-052
    assert m.sgt_irls_tol_um == 1e-4 and m.sgt_irls_max_iter == 30 and m.sgt_max_nfev == 3000     # D-052.2, D-054 (iv)
    assert m.sgt_start_shapes == ((2.0, 0.0, 1.0), (1.0, 0.0, 1.0), (5.0, 0.0, 1.0), (2.0, 1.0, 1.0))
    assert m.sgt_start_w_um == 0.7 and m.sgt_base_start_shape == (2.0, 0.0, 1.0)
    assert FS.n_nodes_for(m) == 848 and FS.WEIGHTINGS[0] == "irls"          # the deliverable is the IRLS fit
    lb, ub = FS.box(m)
    assert np.array_equal(lb, [0.05, 0.0, -1.0, 0.1, 0.0, 0.0, -0.7]) and np.array_equal(ub, [6.0, 20.0, 1.0, 2.0, 1.0, 2.0, 0.7])
    assert BK.FAMILIES == ("gauss", "lorentz", "voigt", "student", "gennorm", "skewnorm") and BK.SGT not in BK.FAMILIES
    assert m.sgt_w_bounds_um == FV.FITTED_FWHM_BOUNDS_UM, "the Gaussian start fit's box is fit_variants' (SPEC Block 12)"
    cfg = C.default_config()
    bad = [dict(sgt_w_bounds_um=(0.0, 2.0)), dict(sgt_w_bounds_um=(2.0, 2.0)), dict(sgt_beta_bounds=(0.0, 10.0)),
           dict(sgt_xi_bounds=(-0.1, 2.0)), dict(sgt_ln_zeta_bounds=(0.7, 0.7)), dict(sgt_lambda_um=0.0),
           dict(sgt_irls_tol_um=0.0), dict(sgt_irls_max_iter=0), dict(sgt_max_nfev=0), dict(sgt_start_shapes=()),
           dict(sgt_start_shapes=((0.5, 0.0, 1.0),)), dict(sgt_start_shapes=((2.0, 0.0, 0.0),)),
           dict(sgt_start_shapes=((2.0, 3.0, 1.0),)), dict(sgt_start_shapes=((2.0, 0.0, 3.0),)),
           dict(sgt_base_start_shape=(20.0, 0.0, 1.0)), dict(sgt_start_w_um=3.0),
           dict(sgt_w_bounds_um=(0.1, float("nan")))]
    for over in bad:
        new = dataclasses.replace(cfg, measure=dataclasses.replace(cfg.measure, **over))
        try:
            new.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("validate accepted %r" % (over,))
    # overrides of the nested start shapes are coerced to floats (one configuration, one hash); the dictionary
    # round trip keeps the nested tuples
    same = C.with_overrides(cfg, {"measure.sgt_start_shapes": [[2, 0, 1], [1, 0, 1], [5, 0, 1], [2, 1, 1]],
                                  "measure.sgt_base_start_shape": [2, 0, 1]})
    assert same == cfg and same.signature_hash() == cfg.signature_hash()
    back = C.config_from_dict(__import__("json").loads(cfg.to_json()))
    assert back == cfg and back.measure.sgt_start_shapes == cfg.measure.sgt_start_shapes
    # the SGT settings are estimator settings: the hash moved off the gate-2 hash and the default at ea49181
    h = cfg.signature_hash("estimator")
    assert h not in ("535847535e83cf7d", "6f0397236447fe6f"), h
    h2 = dataclasses.replace(cfg, measure=dataclasses.replace(cfg.measure, sgt_lambda_um=0.25)).signature_hash("estimator")
    assert h2 != h, "sgt_lambda_um must enter the estimator signature"
    # a flat profile (no dip): no exception, statuses from the vocabulary
    v = _offsets(m)
    rng = np.random.default_rng(SEED + 5)
    res = FS.fit_plane(v, 180.0 + rng.normal(0.0, 2.0, v.size), 0.1, 180.0, m)
    for f in (res.final, res.unweighted):
        assert f.status in block5.FIT_STATUSES, f.status
    print("    estimator hash at the defaults: %s; flat profile: %s / %s" % (h, res.unweighted.status, res.final.status))


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
