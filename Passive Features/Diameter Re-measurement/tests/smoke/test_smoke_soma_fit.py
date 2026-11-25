"""Smoke test for the soma fit (analysis/soma_fit.py) -- Block 46 in specs/SPEC.md (D-215, D-216, D-218, D-219).

Checks (T1 to T8 are the entry's oracles; T9 to T14 its wave-8 amendment's, D-254, D-255, D-256, D-258). Every
fit below runs with the default background "tilted" (D-254.1): no toy scene of T4 to T8 needed "constant" (their data
have no tilt, b1 = b2 = 0, and every tolerance holds); "constant" is asked for explicitly only in T10, as the
comparison the amendment names. The region of T4 to T13 is the 13 um disc the toy scenes were built for
(fit_cfg); T5 and T14 check the default, 1.2 x the traced radius (D-255).
    test_known_answer         T1: a sphere 4 pi r^2; oblate and prolate spheroids against their closed forms written
                              out here [1e-12 relative]; axes equal to 1e-9 relative continuous with the sphere [1e-8
                              relative], and the general branch within 1e-12 of a spheroid continuous with it [1e-10
                              relative];
                              T2: select_planes on a window with invalid planes, steps 1 to 3 (the window's closed
                              ends, k_0 the nearest valid plane, a tie to the lower k); T3: vp_scale on D = B tau
                              exactly gives B [1e-12 relative] and a zero residual [1e-9 gl]; start_theta on an
                              elliptical shadow at 45 deg centred on the traced centre: phi_0 within 0.01 rad, a_0 > b_0,
                              a_0 b_0 = r_s^2 [1e-12 relative]
    test_reference            T1: a general ellipsoid (and the spheroids) against scipy.integrate.dblquad of the
                              parametric surface's area element (epsrel 1e-12) [1e-8 relative]; T3: vp_scale's B
                              equals numpy.linalg.lstsq of the one-column problem per plane [1e-12]; start_theta
                              against its rule written out with explicit sums, a hand-made percentile and the 2 x 2
                              eigenvalues in closed form [1e-10 relative]
    test_convergence          not a discretisation: skipped (S's central differences are checked by Euler's relation)
    test_noise_free_recovery  T4(a): the injected forward model below, a non-spherical truth (a/b 1.35, phi 46.1 deg),
                              the traced centre off by 1 um in each coordinate and the radius by 15 %,
                              ftol = xtol = 1e-10: every fitted parameter within 1e-4 of the truth in its unit;
                              T5: far_slope "fixed" -- theta[8] = slope0 exactly, se[8] NaN, eight names fitted,
                              gamma's row and column of cov NaN, the others as T4(a); a held slope0 outside
                              slope_bounds is not clipped
    test_noisy_recovery       T4(b): one seeded noisy data set (Gaussian, SD 1 % of B): every fitted parameter
                              within 4 SE; T4(c): 40 seeded noisy data sets, the fraction with
                              |c_hat - c| <= SE(c_hat) in [0.45, 0.88]
    test_invariants           T6: masked pixels, pixels outside the disk and planes not selected do not enter --
                              changing the data there leaves theta unchanged [1e-10]; the SomaFit's relations on
                              T4(b)'s fit (se = sqrt(diag(cov)), dof, chi2 and B_gl recomputed with vp_scale at
                              theta_hat, the derived area, volume, r_eq and SE(S) by the delta method); vp_scale
                              linear in D and blind to the scale of tau; T1: the six permutations [1e-12
                              relative]; S's gradient: Euler's relation a S_a + b S_b + c S_c = 2 S [1e-8
                              relative] and 8 pi r / 3 per axis on a sphere
    test_bounds               T7: a start outside a bound is clipped into it (the first render inside the bounds,
                              the names in the message; start_theta itself does not clip); a truth beyond a bound
                              puts that parameter in at_bound
    test_contract             T8: shapes, dtypes and types of SomaFit and of every function; a uint8 block gives
                              the fit of its float64 copy; the inputs are not modified
    test_determinism          T8: two fits of one input give the same SomaFit [exact] (seconds aside)
    test_edge_cases           T8: a forward returning NaN gives status "failed" with NaN results and no raise; a
                              forward NaN only beyond a's upper bound (the truth within the step of it) is never
                              met (the backward column) and the fit converges to the truth; at a NaN inside the
                              bounds (Block 42's rule) the Jacobian takes the backward column (checked column by
                              column), the fit returns without raising on the finite side, and a column NaN both
                              ways gives status "failed"; the ValueErrors of every function (T2's among them)
    test_wave8_vp_background  T9: vp_background on D = (b0 + b1 u + b2 v) tau, coefficients per plane, gives them
                              [1e-9 relative] and a zero residual [1e-8 gl]; equals numpy.linalg.lstsq written out
                              per plane [1e-10]; with b1 = b2 = 0 its b0 is vp_scale's B [1e-10 relative]; a plane of
                              rank < 3 or not finite gives NaN, the others unchanged; background_planes against its
                              formula at three pixels [1e-12]; the ValueErrors of both
    test_wave8_tilt_recovery  T10: T4's truth with slopes up to 0.3 gl/um differing between planes, noise-free,
                              ftol = xtol = 1e-10: "tilted" recovers theta [1e-4] and beta [1e-5]; "constant" on the
                              same data has chi2 >= 1e3 x the tilted fit's; the constant fit's relations (beta =
                              (B, 0, 0), B and chi2 from vp_scale, dof with one scale per plane)
    test_wave8_hold           T11: c held at the truth: theta[5] the value exactly, se[5] and its cov row/column NaN,
                              "c_um" not fitted, every forward call at that c, the others within 1e-4, dof without c,
                              cov_fit (8, 8), SE(S) finite and the delta method over a, b only; c held 20 % off gives
                              a larger chi2 than at the truth (one noisy set)
    test_wave8_start_and_c_bounds  T12: a start at the truth returns it [1e-8] with chi2 < 1e-12 sum d^2; the start's
                              slope entry ignored (slope0 rules; NaN accepted), its entries clipped, a held entry
                              set; c_bounds_um (3.4, 6.0) put c at 3.4 and "c_um" in at_bound; the ValueErrors of
                              hold, start, c_bounds_um, background, sigma0, sigma0_um, everything held
    test_wave8_sigma0         T13: sigma0 "fit" with a stand-in forward taking sigma0_um (ToyForwardS0), the truth
                              30 % from the start, noise-free: sigma0 within 1e-4 um, theta within 1e-4, fitted ends
                              with "sigma0_um", cov_fit (10, 10), dof; noisy: se_sigma0_um = cov_fit's last diagonal's
                              root and cov = cov_fit's first nine's block; a start beyond sigma0_bounds_um clipped;
                              "fixed": a forward without the keyword is called without error, sigma0_um recorded
    test_wave8_region         T14: region_radius_um None -> n_samples = K x the pixels of the disc of 1.2 x the traced
                              radius minus a mask; another scale is read; a number overrides the scale

The injected forward model (the entry's T4: "fast, smooth, and with every fitted parameter identifiable"): the
soma's shadow on plane k is an elliptical Gaussian about (x0, y0), its axes along (cos phi, sin phi) and across, of
SDs sqrt(a^2 + s_k^2) and sqrt(b^2 + s_k^2), and of depth alpha a b / (a_k b_k), alpha = 1 - exp(-2 mu c) the light
the centre's column absorbs (the blur conserves the shadow's light); the blur
s_k = s0 + gamma w softplus((sqrt((z_k - z0)^2 + w^2) - c) / w) grows at the slope gamma with the distance from the
plane to the soma's nearest surface (smoothed over w = 0.5 um; s0 = 0.3 um), so that mu and c separate (mu c from
the depth, c from where the blur starts to grow) and z0, c and gamma are fixed by the blur's two branches. The test's
pixel is 0.25 um (the pilot's is 0.1144 um) to keep 40 fits fast; the planes and the bounds are the
configuration's defaults, the region the 13 um disc (fit_cfg). Block 45's renderer is not imported. T13's stand-in
ToyForwardS0 takes Block 45's wave-8 keyword sigma0_um (a two-Gaussian shadow: see its docstring).

Run
    cd "Passive Features/Diameter Re-measurement"
    python3 tests/smoke/test_smoke_soma_fit.py
        Prints the environment, the observed numbers, and a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit code 1 if any check failed, errored or is still TODO. About 2 minutes
        (OMP_NUM_THREADS=1).
    python -m pytest tests/smoke/test_smoke_soma_fit.py -q
        The same checks through pytest.

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
if str(WS / "src") not in sys.path:
    sys.path.insert(0, str(WS / "src"))

from allen_diameter.analysis import soma_fit as SF  # noqa: E402
from allen_diameter.config import SomaFitConfig  # noqa: E402

SEED = 20261122
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run
_CACHE = {}

P_UM, DZ_UM = 0.25, 0.28            # the test's pixel (um; coarser than the pilot's 0.1144 um, for speed) and dz
CROP_HALF_UM = 13.5                 # the crop about the traced centre: holds the default 13 um region
PLANES_HALF_UM = 12.0               # the recorded window about the traced depth (the fit selects +-10 um)
TRUTH = np.array([0.4, -0.3, 30.1, 4.6, 3.4, 3.0, math.radians(45.0) + 0.02, 0.12, 0.95])  # theta (PARAMS' units)
OFFSET = np.array([-1.0, 1.0, 1.0])  # traced centre - truth (um): off by 1 um in each coordinate (T4(a))
RADIUS_FACTOR = 1.15                # traced radius = 1.15 (a b c)^(1/3): off by 15 % (T4(a))
SLOPE0 = 0.79                       # the configured far slope: gamma's start (truth 0.95) or its held value
B_MEAN, B_WIGGLE = 150.0, 3.0       # the planes' scales B_k = 150 + 3 sin(0.37 k) (gl)
NOISE_GL = 0.01 * B_MEAN            # T4(b), T4(c): Gaussian noise of SD 1 % of B
N_COVERAGE = 40                     # T4(c)
TOY_S0_UM, TOY_W_UM = 0.3, 0.5      # the injected model's in-focus blur and its smoothing width (um)


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure.

    On failure both arrays are saved to ARTIFACT_DIR/<name>.npz before the
    AssertionError propagates. Justify rtol and atol at the call site
    (float64 roundoff, discretisation-error estimate, statistical spread).
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def observed_order(h, err):
    """Least-squares slope p of log e(h) = log C + p * log h.

    h: step sizes (any order); err: positive errors e(h) at those steps.
    """
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    if np.any(err <= 0.0):
        raise ValueError("observed_order needs strictly positive errors")
    slope, _intercept = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


def report(line):
    """An observed number, printed for the run's record."""
    print("   " + line, flush=True)


def raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


class ToyForward(object):
    """The injected forward model (module docstring): tau (K, H, W) on the crop's pixel centres
    ((left + col) p, (top + row) p), for theta_g (8,) and the slope gamma. Counts its renders."""

    def __init__(self, shape, left, top, p, s0=TOY_S0_UM, w=TOY_W_UM):
        H, W = shape
        x = (left + np.arange(W)) * p
        y = (top + np.arange(H)) * p
        self.X, self.Y = np.meshgrid(x, y)
        self.s0, self.w = float(s0), float(w)
        self.calls = []

    def render(self, th, slope, z):
        x0, y0, z0, a, b, c, phi, mu = (float(t) for t in th)
        cp, sp = math.cos(phi), math.sin(phi)
        dx, dy = self.X - x0, self.Y - y0
        u = dx * cp + dy * sp
        v = -dx * sp + dy * cp
        e = np.sqrt((np.asarray(z, dtype=float) - z0) ** 2 + self.w ** 2) - c
        s = self.s0 + float(slope) * self.w * np.logaddexp(0.0, e / self.w)
        A2, B2 = a * a + s * s, b * b + s * s
        amp = -math.expm1(-2.0 * mu * c) * a * b / np.sqrt(A2 * B2)
        Q = (u * u)[None] / A2[:, None, None] + (v * v)[None] / B2[:, None, None]
        return 1.0 - amp[:, None, None] * np.exp(-0.5 * Q)

    def factory(self, z_planes_um):
        z = np.array(z_planes_um, dtype=float)

        def forward(theta_g, slope):
            self.calls.append((np.array(theta_g, dtype=float), float(slope)))
            return self.render(theta_g, slope, z)
        return forward


def scene(truth=TRUTH):
    """The synthetic soma stack of T4: every recorded plane B_k tau_k(truth) (noise-free), with the traced centre
    (truth + OFFSET) and radius (RADIUS_FACTOR (a b c)^(1/3)); dict with block, ks, valid, left, top, H, W, centre,
    radius, toy (the injected model) and truth."""
    key = ("scene", tuple(np.asarray(truth, dtype=float)))
    if key in _CACHE:
        return _CACHE[key]
    truth = np.asarray(truth, dtype=float)
    centre = truth[:3] + OFFSET
    radius = RADIUS_FACTOR * float(np.prod(truth[3:6])) ** (1.0 / 3.0)
    left = int(math.floor((centre[0] - CROP_HALF_UM) / P_UM))
    top = int(math.floor((centre[1] - CROP_HALF_UM) / P_UM))
    H = W = int(math.ceil(2.0 * CROP_HALF_UM / P_UM)) + 1
    ks = np.arange(int(math.floor((centre[2] - PLANES_HALF_UM) / DZ_UM)),
                   int(math.ceil((centre[2] + PLANES_HALF_UM) / DZ_UM)) + 1)
    valid = np.ones(ks.size, dtype=bool)
    toy = ToyForward((H, W), left, top, P_UM)
    Bk = B_MEAN + B_WIGGLE * np.sin(0.37 * ks)
    block = Bk[:, None, None] * toy.render(truth[:8], truth[8], ks * DZ_UM)
    sc = dict(block=block, ks=ks, valid=valid, left=left, top=top, H=H, W=W, centre=centre, radius=radius,
              toy=toy, truth=truth, Bk=Bk)
    _CACHE[key] = sc
    return sc


def noisy(sc, seed):
    return sc["block"] + make_rng(seed).normal(0.0, NOISE_GL, sc["block"].shape)


def fit_cfg(**kw):
    """The settings these checks exercise: Block 46's defaults with the far slope fitted -- the nine-parameter path,
    kept as an opt-in comparison since D-240 made "fixed" the default -- so that path stays tested; and the 13 um
    region the toy scenes were built for (CROP_HALF_UM), the default being Allen's traced radius since D-241."""
    base = dict(far_slope="fit", region_radius_um=13.0)
    base.update(kw)
    return dataclasses.replace(SomaFitConfig(), **base)


def fit(sc, data=None, scfg=None, slope0=SLOPE0, mask=None, factory=None, radius=None):
    scfg = fit_cfg() if scfg is None else scfg
    mask = np.zeros((sc["H"], sc["W"]), dtype=bool) if mask is None else mask
    return SF.fit_soma(sc["block"] if data is None else data, sc["ks"], sc["valid"], sc["left"], sc["top"], P_UM,
                       DZ_UM, sc["centre"], sc["radius"] if radius is None else radius, mask,
                       sc["toy"].factory if factory is None else factory, scfg, slope0)


def tight():
    return fit_cfg(ftol=1e-10, xtol=1e-10)


def fit_noise_free():
    if "T4a" not in _CACHE:
        _CACHE["T4a"] = fit(scene(), scfg=tight())
    return _CACHE["T4a"]


def fit_noisy_one():
    if "T4b" not in _CACHE:
        _CACHE["T4b"] = fit(scene(), data=noisy(scene(), SEED))
    return _CACHE["T4b"]


def disk_um(sc, radius_um):
    x = (sc["left"] + np.arange(sc["W"])) * P_UM
    y = (sc["top"] + np.arange(sc["H"])) * P_UM
    return np.hypot(x[None, :] - sc["centre"][0], y[:, None] - sc["centre"][1]) <= radius_um


def uv_um(sc):
    """(U, V) (H, W) um: the pixel centres from the traced centre, u = x - x_s, v = y - y_s (the tilt's origin)."""
    x = (sc["left"] + np.arange(sc["W"])) * P_UM - sc["centre"][0]
    y = (sc["top"] + np.arange(sc["H"])) * P_UM - sc["centre"][1]
    return np.meshgrid(x, y)


class ToyForwardS0(ToyForward):
    """T13's stand-in for Block 45's sigma0_um keyword (its wave-8 amendment is not relied on): the in-focus blur
    sigma0 added in quadrature to the defocus growth, s_k^2 = sigma0^2 + (gamma w softplus((sqrt((z_k - z0)^2 + w^2)
    - c) / w))^2; sigma0 None gives TOY_S0_UM. The shadow is a mixture of two elliptical Gaussians of SDs (a, b) and
    (a/2, b/2), each blurred: tau = 1 - alpha sum_i 0.5 (a_i b_i) / sqrt(A_i B_i) exp(-Q_i / 2), A_i = a_i^2 + s^2.
    Not ToyForward's single Gaussian, which is exactly degenerate in sigma0 (it shows only a^2 + sigma0^2 and
    b^2 + sigma0^2, mu absorbing the depth: a fit from 30 % off stopped at chi2 6e-22 with sigma0 0.315 for 0.40):
    the two widths a^2 + s^2 and a^2/4 + s^2 separate a from sigma0."""

    def render_s0(self, th, slope, z, sigma0):
        x0, y0, z0, a, b, c, phi, mu = (float(t) for t in th)
        s0 = TOY_S0_UM if sigma0 is None else float(sigma0)
        cp, sp = math.cos(phi), math.sin(phi)
        dx, dy = self.X - x0, self.Y - y0
        u = dx * cp + dy * sp
        v = -dx * sp + dy * cp
        e = np.sqrt((np.asarray(z, dtype=float) - z0) ** 2 + self.w ** 2) - c
        grow = float(slope) * self.w * np.logaddexp(0.0, e / self.w)
        s2 = s0 * s0 + grow * grow
        alpha = -math.expm1(-2.0 * mu * c)
        shadow = 0.0
        for ai, bi in ((a, b), (0.5 * a, 0.5 * b)):
            A2, B2 = ai * ai + s2, bi * bi + s2
            amp = 0.5 * alpha * ai * bi / np.sqrt(A2 * B2)
            Q = (u * u)[None] / A2[:, None, None] + (v * v)[None] / B2[:, None, None]
            shadow = shadow + amp[:, None, None] * np.exp(-0.5 * Q)
        return 1.0 - shadow

    def factory(self, z_planes_um):
        z = np.array(z_planes_um, dtype=float)

        def forward(theta_g, slope, sigma0_um=None):
            self.calls.append((np.array(theta_g, dtype=float), float(slope), sigma0_um))
            return self.render_s0(theta_g, slope, z, sigma0_um)
        return forward


def tilt_coefficients(ks):
    """T10's backgrounds per recorded plane: b0 = B_k (scene), b1, b2 up to 0.3 gl/um, differing between planes."""
    ks = np.asarray(ks, dtype=float)
    return (B_MEAN + B_WIGGLE * np.sin(0.37 * ks), 0.3 * np.sin(0.53 * ks + 0.4), -0.3 * np.cos(0.29 * ks))


def tilted_data(sc):
    """T10's noise-free data: (b0_k + b1_k u + b2_k v) tau_k(truth), u, v from the traced centre."""
    b0, b1, b2 = tilt_coefficients(sc["ks"])
    U, V = uv_um(sc)
    tau = sc["toy"].render(sc["truth"][:8], sc["truth"][8], sc["ks"] * DZ_UM)
    return (b0[:, None, None] + b1[:, None, None] * U[None] + b2[:, None, None] * V[None]) * tau


def prolate_closed(a, b):
    """a > b = c: 2 pi b^2 (1 + a / (b e) arcsin e), e^2 = 1 - b^2 / a^2."""
    e = math.sqrt(1.0 - (b / a) ** 2)
    return 2.0 * math.pi * b * b * (1.0 + a / (b * e) * math.asin(e))


def oblate_closed(a, c):
    """a = b > c: 2 pi a^2 (1 + (1 - e^2) / e artanh e), e^2 = 1 - c^2 / a^2."""
    e = math.sqrt(1.0 - (c / a) ** 2)
    return 2.0 * math.pi * a * a * (1.0 + (1.0 - e * e) / e * 0.5 * math.log((1.0 + e) / (1.0 - e)))


def area_dblquad(a, b, c):
    """The ellipsoid's area by scipy.integrate.dblquad of the parametric surface's area element
    |r_t x r_p| = sin t sqrt((b c sin t cos p)^2 + (a c sin t sin p)^2 + (a b cos t)^2), one octant x 8."""
    from scipy.integrate import dblquad

    def f(t, p):
        st, ct, sp, cp = math.sin(t), math.cos(t), math.sin(p), math.cos(p)
        return st * math.sqrt((b * c * st * cp) ** 2 + (a * c * st * sp) ** 2 + (a * b * ct) ** 2)
    return 8.0 * dblquad(f, 0.0, 0.5 * math.pi, 0.0, 0.5 * math.pi, epsabs=0.0, epsrel=1e-12)[0]


def shadow_plane(sc, a, b, phi, depth=0.5, B=150.0):
    """One plane: an elliptical Gaussian shadow (SDs a, b um at angle phi) centred on the traced centre."""
    x = (sc["left"] + np.arange(sc["W"])) * P_UM - sc["centre"][0]
    y = (sc["top"] + np.arange(sc["H"])) * P_UM - sc["centre"][1]
    X, Y = np.meshgrid(x, y)
    u = X * math.cos(phi) + Y * math.sin(phi)
    v = -X * math.sin(phi) + Y * math.cos(phi)
    return B * (1.0 - depth * np.exp(-0.5 * (u * u / (a * a) + v * v / (b * b))))


def start_by_hand(plane, sc, region, radius, slope0):
    """start_theta's rule written out: explicit sums, a hand-made linear percentile, closed-form 2 x 2 eigenvalues."""
    xs, ys, zs = (float(t) for t in sc["centre"])
    vals, X, Y = [], [], []
    for i in range(plane.shape[0]):
        for j in range(plane.shape[1]):
            if region[i, j]:
                vals.append(float(plane[i, j]))
                X.append((sc["left"] + j) * P_UM)
                Y.append((sc["top"] + i) * P_UM)
    vals, X, Y = np.array(vals), np.array(X), np.array(Y)
    rho = np.sqrt((X - xs) ** 2 + (Y - ys) ** 2)
    far = vals[rho > 1.5 * radius]
    srt = np.sort(far if far.size else vals)
    n = srt.size
    B_far = float(srt[n // 2]) if n % 2 == 1 else 0.5 * float(srt[n // 2 - 1] + srt[n // 2])
    core = np.sort(vals[rho <= 0.5 * radius])
    if core.size:
        pos = 0.05 * (core.size - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, core.size - 1)
        q = core[lo] + (pos - lo) * (core[hi] - core[lo])
        varrho = min(max(q / B_far, 0.05), 0.95)
    else:
        varrho = 0.5
    mu0 = -math.log(varrho) / (2.0 * radius)
    sel = rho <= 1.5 * radius
    w = np.maximum(0.0, 1.0 - vals[sel] / B_far)
    xw, yw = X[sel], Y[sel]
    sw = float(sum(w))
    mx, my = float(sum(w * xw)) / sw, float(sum(w * yw)) / sw
    cxx = float(sum(w * (xw - mx) ** 2)) / sw
    cyy = float(sum(w * (yw - my) ** 2)) / sw
    cxy = float(sum(w * (xw - mx) * (yw - my))) / sw
    mid, dev = 0.5 * (cxx + cyy), math.sqrt((0.5 * (cxx - cyy)) ** 2 + cxy ** 2)
    l1, l2 = mid + dev, mid - dev
    phi0 = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    return np.array([xs, ys, zs, radius * (l1 / l2) ** 0.25, radius * (l2 / l1) ** 0.25, radius, phi0, mu0, slope0])


def nan_equal(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    """T1 known forms, T2 the plane selection, T3 the exact scale, start_theta's orientation."""
    # T1: the sphere; the spheroids against their closed forms (roundoff only: 1e-12 relative)
    for r in (1.0, 3.7, 14.0):
        check_close("sphere", SF.ellipsoid_area(r, r, r), 4.0 * math.pi * r * r, rtol=1e-15)
    for big, small in ((5.0, 2.0), (6.6, 5.8), (14.0, 1.0)):
        check_close("prolate", SF.ellipsoid_area(big, small, small), prolate_closed(big, small), rtol=1e-12)
        check_close("oblate", SF.ellipsoid_area(big, big, small), oblate_closed(big, small), rtol=1e-12)
    # T1: axes equal to 1e-9 relative are continuous with the sphere (S moves by about 2/3 of the axes' relative
    # change, 1e-9: 1e-8 relative is the bound); the general branch within 1e-12 of a spheroid is continuous with it
    r = 3.0
    for t in ((r * (1 + 1e-9), r, r), (r, r, r * (1 - 1e-9)), (r * (1 + 2e-9), r * (1 + 1e-9), r),
              (r * (1 + 1e-9), r * (1 - 1e-9), r)):
        check_close("near sphere %s" % (t,), SF.ellipsoid_area(*t), 4.0 * math.pi * r * r, rtol=1e-8)
    check_close("near prolate", SF.ellipsoid_area(5.0, 2.0 * (1 + 1e-12), 2.0), prolate_closed(5.0, 2.0), rtol=1e-10)
    check_close("near oblate", SF.ellipsoid_area(5.0, 5.0 * (1 - 1e-12), 2.0), oblate_closed(5.0, 2.0), rtol=1e-10)

    # T2: a window k = 100..140 with invalid planes 119, 121, 125; dz = 0.25 um (exact in binary, so the window's
    # closed ends are tested without roundoff)
    ks = np.arange(100, 141)
    valid = ~np.isin(ks, (119, 121, 125))
    dz = 0.25
    got = {s: [int(ks[i]) for i in SF.select_planes(ks, valid, 120.0 * dz, dz, 2.0, s)] for s in (1, 2, 3)}
    assert got[1] == [k for k in range(112, 129) if k not in (119, 121, 125)], got[1]   # |k dz - z| <= 2 um: ends in
    assert got[2] == list(range(112, 129, 2)), got[2]
    assert got[3] == [114, 117, 120, 123, 126], got[3]
    # k_0 the valid plane nearest z: z at k = 119.1, 119 invalid -> k_0 = 120 (0.9 planes) not 118 (1.1)
    got = [int(ks[i]) for i in SF.select_planes(ks, valid, 119.1 * dz, dz, 2.0, 2)]
    assert got == list(range(112, 127, 2)), got                        # 112 <= k <= 127.1, k - 120 even
    # a tie (z at k = 122.5, both valid) goes to the lower k: k_0 = 122 -> even offsets from 122
    got = [int(ks[i]) for i in SF.select_planes(ks, valid, 122.5 * dz, dz, 2.0, 3)]
    assert got == [116, 122, 128] and 122 in got, got                  # 114.5 <= k <= 130.5, k - 122 in 3Z, 119 out
    idx = SF.select_planes(ks, valid, 120.0 * dz, dz, 2.0, 1)
    assert idx.dtype == np.int64 and np.all(np.diff(idx) > 0) and np.all(valid[idx])

    # T3: D = B tau exactly (B per plane) gives B [1e-12 relative] and a zero residual [1e-9 gl]
    rng = make_rng(SEED)
    tau = rng.uniform(0.2, 1.0, (4, 30, 40))
    B = np.array([150.0, 97.5, 210.25, 3.0])
    region = rng.uniform(size=(30, 40)) < 0.6
    Bh, resid = SF.vp_scale(B[:, None, None] * tau, tau, region)
    check_close("T3 B", Bh, B, rtol=1e-12)
    check_close("T3 resid", resid, np.zeros_like(resid), rtol=0, atol=1e-9)
    assert resid.shape == (4, int(region.sum()))

    # start_theta on an elliptical shadow at 45 deg (SDs 4.0, 2.5 um) centred on the traced centre: the moments'
    # orientation is the shadow's (the disk is centred on it; the pixel grid's anisotropy leaves well under 0.01 rad)
    sc = scene()
    plane = shadow_plane(sc, 4.0, 2.5, math.radians(45.0))
    region = disk_um(sc, 13.0)
    th = SF.start_theta(plane, sc["left"], sc["top"], P_UM, sc["centre"], 3.2, region, 0.79)
    assert abs(th[6] - math.radians(45.0)) < 0.01, th
    assert th[3] > th[4] and th[5] == 3.2 and th[8] == 0.79, th
    check_close("a0 b0 = r_s^2", th[3] * th[4], 3.2 ** 2, rtol=1e-12)
    assert_array_equal(th[:3], sc["centre"])
    report("start_theta on a 45 deg shadow: phi_0 - phi = %.2e rad, a_0/b_0 = %.3f (shadow 1.6, the disk of 1.5 r_s "
           "truncates it)" % (th[6] - math.radians(45.0), th[3] / th[4]))

    # [2026-11-22, D-220.4 (final check F4)] the start's settings are SomaFitConfig fields: the defaults give the same
    # start as scfg=None [exact]; a higher core percentile gives a lighter core, so a smaller mu_0; with no core pixel
    # (core radius below one pixel off the grid) varrho = start_ratio_none, mu_0 = -ln(start_ratio_none) / (2 r_s)
    from allen_diameter.config import SomaFitConfig
    d = SomaFitConfig()
    assert_array_equal(SF.start_theta(plane, sc["left"], sc["top"], P_UM, sc["centre"], 3.2, region, 0.79, d), th)
    th50 = SF.start_theta(plane, sc["left"], sc["top"], P_UM, sc["centre"], 3.2, region, 0.79,
                          dataclasses.replace(d, start_core_percentile=50.0))
    assert th50[7] < th[7], (th50[7], th[7])
    hole = region & ~disk_um(sc, 0.5 * 3.2)
    thn = SF.start_theta(plane, sc["left"], sc["top"], P_UM, sc["centre"], 3.2, hole, 0.79,
                         dataclasses.replace(d, start_ratio_none=0.25))
    check_close("mu_0 without a core", thn[7], -math.log(0.25) / (2 * 3.2), rtol=1e-15)

    # [2026-11-22, D-220.5 (final check F7)] Block 44's window: with z and half exact multiples of dz = 0.28 um the
    # edge planes are kept, as soma_view.plane_range keeps them (36 x 0.28 = 10.080000000000002 > 10.08)
    from allen_diameter.analysis import soma_view as SV
    kk = np.arange(-40, 41)
    sel = kk[SF.select_planes(kk, np.ones(kk.size, bool), 0.0, 0.28, 10.08, 1)]
    assert (int(sel[0]), int(sel[-1])) == SV.plane_range(0.0, 10.08, 0.28) == (-36, 36), (sel[[0, -1]],)


def test_reference():
    """T1 against dblquad, T3 against lstsq, start_theta against its rule written out."""
    worst = 0.0
    for abc in ((6.6, 5.8, 3.5), (4.6, 3.4, 3.0), (12.0, 2.0, 1.1), (3.0, 7.0, 5.0), (5.0, 5.0, 2.0),
                (5.0, 2.0, 2.0)):
        ref = area_dblquad(*abc)
        S = SF.ellipsoid_area(*abc)
        worst = max(worst, abs(S - ref) / ref)
        check_close("area vs dblquad %s" % (abc,), S, ref, rtol=1e-8)   # the quadrature's epsrel 1e-12, the bound 1e-8
    report("T1: ellipsoid_area against dblquad (6 ellipsoids): max relative difference %.1e" % worst)

    rng = make_rng(SEED + 1)
    D = rng.uniform(50.0, 200.0, (3, 20, 25))
    tau = rng.uniform(0.1, 1.0, (3, 20, 25))
    region = rng.uniform(size=(20, 25)) < 0.5
    B, resid = SF.vp_scale(D, tau, region)
    for k in range(3):
        sol = np.linalg.lstsq(tau[k][region][:, None], D[k][region], rcond=None)[0][0]
        check_close("T3 lstsq plane %d" % k, B[k], sol, rtol=1e-12)    # one column: the same solution, roundoff only
        check_close("T3 resid plane %d" % k, resid[k], D[k][region] - sol * tau[k][region], rtol=0, atol=1e-10)

    sc = scene()
    region = disk_um(sc, 13.0) & ~((np.abs((sc["left"] + np.arange(sc["W"]))[None, :] * P_UM - sc["centre"][0]) < 1.0)
                                   & ((sc["top"] + np.arange(sc["H"]))[:, None] * P_UM < sc["centre"][1] - 4.0))
    k0 = int(np.argmin(np.abs(sc["ks"] * DZ_UM - sc["centre"][2])))
    plane = noisy(sc, SEED + 2)[k0]
    th = SF.start_theta(plane, sc["left"], sc["top"], P_UM, sc["centre"], sc["radius"], region, SLOPE0)
    ref = start_by_hand(plane, sc, region, sc["radius"], SLOPE0)
    check_close("start_theta vs by hand", th, ref, rtol=1e-10)          # different summation orders: roundoff only
    report("start_theta against the rule by hand: max relative difference %.1e"
           % float(np.max(np.abs(th - ref) / np.abs(ref))))


def test_convergence():
    raise unittest.SkipTest("not a discretisation: the fit's Jacobian steps are absolute by the entry, and the "
                            "gradient of S by central differences is checked by Euler's relation (test_invariants)")


def test_noise_free_recovery():
    """T4(a) and T5."""
    sc = scene()
    f = fit_noise_free()
    err = np.abs(f.theta - sc["truth"])
    assert f.status == "converged", f.message
    assert f.fitted == SF.PARAMS, f.fitted
    assert np.all(err <= 1e-4), dict(zip(SF.PARAMS, err))
    report("T4(a) noise-free, start off by 1 um and 15 %%: max |error| %.1e (%s), %d forward evaluations, %.1f s"
           % (float(err.max()), SF.PARAMS[int(np.argmax(err))], f.n_eval, f.seconds))
    report("      start: %s" % np.array2string(SF.start_theta(
        sc["block"][int(np.argmin(np.abs(sc["ks"] * DZ_UM - sc["centre"][2])))], sc["left"], sc["top"], P_UM,
        sc["centre"], sc["radius"], disk_um(sc, 13.0), SLOPE0), precision=3))
    report("      truth: %s" % np.array2string(sc["truth"], precision=3))

    # T5: far_slope "fixed", the truth's gamma = slope0
    truth5 = TRUTH.copy()
    truth5[8] = SLOPE0
    sc5 = scene(truth5)
    f5 = fit(sc5, scfg=dataclasses.replace(tight(), far_slope="fixed"))
    assert f5.status == "converged", f5.message
    assert f5.theta[8] == SLOPE0 and np.isnan(f5.se[8]), (f5.theta[8], f5.se[8])
    assert f5.fitted == SF.PARAMS[:8] and len(f5.fitted) == 8
    assert np.all(np.isnan(f5.cov[8, :])) and np.all(np.isnan(f5.cov[:, 8])) and np.all(np.isfinite(f5.cov[:8, :8]))
    assert all(abs(call[1] - SLOPE0) == 0 for call in sc5["toy"].calls[-f5.n_eval:]), "a held slope moved"
    err5 = np.abs(f5.theta[:8] - truth5[:8])
    assert np.all(err5 <= 1e-4), dict(zip(SF.PARAMS, err5))
    report("T5 fixed slope: max |error| %.1e over the eight, theta[8] = %.17g (slope0 %.17g), %d evaluations"
           % (float(err5.max()), f5.theta[8], SLOPE0, f5.n_eval))
    # a held slope0 outside slope_bounds is held as it is (not clipped): only gamma's start is clipped
    assert SomaFitConfig().far_slope == "fixed", SomaFitConfig().far_slope     # the default (D-240, user)
    # D-241, D-255 (user): with region_radius_um None the region is the disc of region_radius_scale (1.2) x the
    # traced radius passed in [exact count] (T14 checks it with a mask and an override)
    assert SomaFitConfig().region_radius_um is None and SomaFitConfig().mask_widen_um == 0.0
    assert SomaFitConfig().region_radius_scale == 1.2 and SomaFitConfig().background == "tilted"
    assert SomaFitConfig().render_half_um == 14.0                     # the crop's floor (D-243, orchestrator)
    f_none = fit(sc5, scfg=dataclasses.replace(SomaFitConfig(), max_nfev=1), slope0=5.0)
    n_disc = int(np.count_nonzero(disk_um(sc5, 1.2 * sc5["radius"])))
    assert f_none.n_samples == len(f_none.ks) * n_disc, (f_none.n_samples, len(f_none.ks), n_disc)
    f5b = fit(sc5, scfg=dataclasses.replace(SomaFitConfig(), far_slope="fixed", max_nfev=1), slope0=5.0)
    assert f5b.theta[8] == 5.0 and "slope_um_per_um" not in f5b.message, (f5b.theta[8], f5b.message)


def test_noisy_recovery():
    """T4(b) and T4(c)."""
    sc = scene()
    f = fit_noisy_one()
    assert f.status == "converged", f.message
    z = np.abs(f.theta - sc["truth"]) / f.se
    assert np.all(np.isfinite(z)) and np.all(z <= 4.0), dict(zip(SF.PARAMS, z))
    report("T4(b) one noisy set (SD %.1f gl): max |error| / SE = %.2f (%s); SE(c) %.4f um, SE(S) %.3f um^2; %d "
           "evaluations, %.1f s" % (NOISE_GL, float(z.max()), SF.PARAMS[int(np.argmax(z))], f.se[5],
                                    f.se_area_um2, f.n_eval, f.seconds))
    inside, c_hat, se_c, n_eval = 0, [], [], []
    t0 = time.perf_counter()
    for i in range(N_COVERAGE):
        fi = fit(sc, data=noisy(sc, SEED + 100 + i))
        assert fi.status == "converged", (i, fi.message)
        c_hat.append(fi.theta[5])
        se_c.append(fi.se[5])
        n_eval.append(fi.n_eval)
        inside += int(abs(fi.theta[5] - sc["truth"][5]) <= fi.se[5])
    frac = inside / float(N_COVERAGE)
    report("T4(c) %d noisy sets: |c_hat - c| <= SE(c_hat) in %d (%.3f; band [0.45, 0.88]); SD of c_hat %.4f um, "
           "median SE(c) %.4f um; mean c_hat - c %+.4f um; %d to %d evaluations, %.0f s"
           % (N_COVERAGE, inside, frac, float(np.std(c_hat, ddof=1)), float(np.median(se_c)),
              float(np.mean(c_hat) - sc["truth"][5]), min(n_eval), max(n_eval), time.perf_counter() - t0))
    assert 0.45 <= frac <= 0.88, frac          # 0.683 +- about 3 binomial SDs (sqrt(0.683 0.317 / 40) = 0.074)


def test_invariants():
    """T6, the SomaFit's relations, vp_scale's linearity, T1's permutations, S's gradient."""
    sc = scene()
    # T6: a stem strip masked (|x - x_s| <= 1.5 um above the soma); the data changed on it, outside the 13 um disk
    # and on planes not selected: theta unchanged [1e-10] (the same arithmetic: bit for bit in fact)
    x = (sc["left"] + np.arange(sc["W"])) * P_UM - sc["centre"][0]
    y = (sc["top"] + np.arange(sc["H"])) * P_UM - sc["centre"][1]
    mask = (np.abs(x)[None, :] <= 1.5) & (y[:, None] <= -5.0)
    data = noisy(sc, SEED + 3)
    f1 = fit(sc, data=data, mask=mask)
    data2 = data.copy()
    data2[:, mask] = 40.0                                   # a dark stem in the masked strip
    data2[:, ~disk_um(sc, 13.5)] = 255.0                    # outside the region's disk
    sel = SF.select_planes(sc["ks"], sc["valid"], sc["centre"][2], DZ_UM, 10.0, 2)
    others = np.setdiff1d(np.arange(sc["ks"].size), sel)
    data2[others] = 0.0                                     # the planes not selected
    f2 = fit(sc, data=data2, mask=mask)
    assert f1.status == "converged", f1.message
    check_close("T6 theta", f2.theta, f1.theta, rtol=0, atol=1e-10)
    assert nan_equal(f2.cov, f1.cov) and f2.chi2 == f1.chi2 and np.array_equal(f2.B_gl, f1.B_gl)
    # the negative control: without the mask the same dark stem moves theta (the check above can fail)
    f3 = fit(sc, data=np.where(mask[None], 40.0, data))
    shift = float(np.max(np.abs(f3.theta - f1.theta)))
    assert shift > 1e-3, shift
    report("T6: masked, outside-disk and unselected data changed: max |d theta| = %.1e; the stem unmasked moves "
           "theta by up to %.3f" % (float(np.max(np.abs(f2.theta - f1.theta))), shift))

    # the SomaFit's relations on T4(b)'s fit (the default background "tilted": beta and chi2 recomputed with
    # vp_background at theta_hat, u and v from the traced centre; dof with 3 coefficients per plane)
    f = fit_noisy_one()
    assert f.background == "tilted"
    data = noisy(sc, SEED)
    region = disk_um(sc, 13.0)
    tau = sc["toy"].render(f.theta[:8], f.theta[8], f.ks * DZ_UM)
    kidx = np.searchsorted(sc["ks"], f.ks)
    U, V = uv_um(sc)
    beta, resid = SF.vp_background(data[kidx], tau, region, U, V)
    check_close("beta at theta_hat", f.beta, beta, rtol=1e-12, atol=1e-12)
    check_close("B_gl = beta[:, 0]", f.B_gl, beta[:, 0], rtol=1e-12)
    chi2 = float(np.sum(resid ** 2))
    check_close("chi2 at theta_hat", f.chi2, chi2, rtol=1e-10)
    assert f.n_samples == f.ks.size * int(region.sum()) and f.dof == f.n_samples - 9 - 3 * f.ks.size
    check_close("cov_fit = cov (all nine fitted)", f.cov_fit, f.cov, rtol=0, atol=0)
    check_close("se = sqrt(diag(cov))", f.se, np.sqrt(np.diag(f.cov)), rtol=1e-15)
    assert np.allclose(f.cov, f.cov.T, rtol=1e-10, atol=0)
    check_close("area", f.area_um2, SF.ellipsoid_area(*f.theta[3:6]), rtol=1e-15)
    check_close("volume", f.volume_um3, 4.0 / 3.0 * math.pi * float(np.prod(f.theta[3:6])), rtol=1e-14)
    check_close("r_eq", f.r_eq_um, math.sqrt(f.area_um2 / (4.0 * math.pi)), rtol=1e-15)
    h = 1e-4
    g = np.array([(SF.ellipsoid_area(*(f.theta[3:6] + h * e)) - SF.ellipsoid_area(*(f.theta[3:6] - h * e))) / (2 * h)
                  for e in np.eye(3)])
    check_close("SE(S) delta method", f.se_area_um2, math.sqrt(g @ f.cov[3:6, 3:6] @ g), rtol=1e-6)  # step 1e-4 here
    lo = np.r_[sc["centre"] - [4.0, 4.0, 6.0], [1.0] * 3, -math.pi, 0.001, 0.2]
    hi = np.r_[sc["centre"] + [4.0, 4.0, 6.0], [15.0] * 3, math.pi, 3.0, 3.0]
    assert np.all(f.theta >= lo) and np.all(f.theta <= hi) and f.at_bound == ()

    # vp_scale: linear in D, blind to the scale of tau (B scales inversely, the residual does not change)
    rng = make_rng(SEED + 4)
    D = rng.uniform(50.0, 200.0, (2, 9, 11))
    T = rng.uniform(0.1, 1.0, (2, 9, 11))
    reg = rng.uniform(size=(9, 11)) < 0.7
    B1, r1 = SF.vp_scale(D, T, reg)
    B2, r2 = SF.vp_scale(2.0 * D, T, reg)
    B3, r3 = SF.vp_scale(D, 4.0 * T, reg)
    check_close("vp linear B", B2, 2.0 * B1, rtol=1e-14)
    check_close("vp linear resid", r2, 2.0 * r1, rtol=1e-12, atol=1e-12)
    check_close("vp tau scale B", B3, B1 / 4.0, rtol=1e-14)
    check_close("vp tau scale resid", r3, r1, rtol=0, atol=1e-11)

    # T1: the six permutations of the arguments [1e-12 relative]; Euler's relation (S homogeneous of degree 2) for
    # the gradient by central differences [1e-8 relative], and 8 pi r / 3 per axis on a sphere
    import itertools
    for abc in ((6.6, 5.8, 3.5), (2.0, 2.0, 7.0), (1.0, 9.0, 4.0)):
        S0 = SF.ellipsoid_area(*abc)
        for perm in itertools.permutations(abc):
            check_close("permutation %s" % (perm,), SF.ellipsoid_area(*perm), S0, rtol=1e-12)
        gS = SF._area_gradient(*abc)
        check_close("Euler %s" % (abc,), float(np.dot(abc, gS)), 2.0 * S0, rtol=1e-8)
    check_close("sphere gradient", SF._area_gradient(2.5, 2.5, 2.5), [8.0 * math.pi * 2.5 / 3.0] * 3, rtol=1e-8)


def test_bounds():
    """T7: the start clipped into the bounds; a truth beyond a bound puts the parameter in at_bound."""
    sc = scene()
    # a traced radius beyond the axes' upper bound (15 um) and slope0 beyond gamma's (3.0): the start clipped
    lo = np.r_[sc["centre"] - [4.0, 4.0, 6.0], [1.0] * 3, -math.pi, 0.001, 0.2]
    hi = np.r_[sc["centre"] + [4.0, 4.0, 6.0], [15.0] * 3, math.pi, 3.0, 3.0]
    k0 = int(np.argmin(np.abs(sc["ks"] * DZ_UM - sc["centre"][2])))
    st = SF.start_theta(sc["block"][k0], sc["left"], sc["top"], P_UM, sc["centre"], 20.0, disk_um(sc, 13.0), 5.0)
    assert st[5] == 20.0 and st[8] == 5.0 and st[3] >= 20.0, st        # start_theta itself does not clip
    expect = [SF.PARAMS[j] for j in range(9) if not lo[j] <= st[j] <= hi[j]]
    assert {"a_um", "c_um", "slope_um_per_um"} <= set(expect), expect
    toy = ToyForward((sc["H"], sc["W"]), sc["left"], sc["top"], P_UM)
    f = fit(sc, scfg=fit_cfg(max_nfev=1), slope0=5.0, radius=20.0, factory=toy.factory)
    th0, g0 = toy.calls[0]
    assert np.array_equal(np.r_[th0, g0], np.clip(st, lo, hi)), (th0, g0)   # the first render: the clipped start
    assert f.message.split("start clipped into the bounds: ")[-1] == ", ".join(expect), f.message
    assert all(np.all(c[0] >= lo[:8]) and np.all(c[0] <= hi[:8]) and lo[8] <= c[1] <= hi[8] for c in toy.calls)

    # the truth 1 um from the traced centre in x and y, the bound 0.5 um: x0 and y0 at their bounds
    f2 = fit(sc, scfg=fit_cfg(shift_xy_um=0.5))
    assert "x0_um" in f2.at_bound and "y0_um" in f2.at_bound, (f2.at_bound, f2.theta)
    check_close("x0 at bound", f2.theta[0], sc["centre"][0] + 0.5, rtol=0, atol=1e-6)
    check_close("y0 at bound", f2.theta[1], sc["centre"][1] - 0.5, rtol=0, atol=1e-6)
    report("T7: at_bound %s with shift_xy_um 0.5 (status %s); start clipped: %s"
           % (f2.at_bound, f2.status, f.message.split("start clipped into the bounds: ")[-1]))


def test_contract():
    """T8: shapes, dtypes and types; uint8 data; inputs unchanged."""
    assert SF.PARAMS == ("x0_um", "y0_um", "z0_um", "a_um", "b_um", "c_um", "phi_rad", "mu_per_um", "slope_um_per_um")
    assert SF.STATUSES == ("converged", "max_nfev", "failed")
    f = fit_noisy_one()
    K = f.ks.size
    for name, shape in (("theta", (9,)), ("se", (9,)), ("cov", (9, 9)), ("B_gl", (K,)), ("beta", (K, 3)),
                        ("cov_fit", (9, 9))):
        arr = getattr(f, name)
        assert isinstance(arr, np.ndarray) and arr.shape == shape and arr.dtype == np.float64, (name, arr.shape)
    assert f.ks.dtype == np.int64 and np.all(np.diff(f.ks) > 0)
    for name in ("chi2", "seconds", "area_um2", "se_area_um2", "volume_um3", "r_eq_um", "sigma0_um", "se_sigma0_um"):
        assert isinstance(getattr(f, name), float), name
    for name in ("dof", "n_samples", "n_eval"):
        assert isinstance(getattr(f, name), int) and not isinstance(getattr(f, name), bool), name
    assert isinstance(f.fitted, tuple) and isinstance(f.at_bound, tuple) and isinstance(f.message, str)
    assert f.background == "tilted" and math.isnan(f.sigma0_um) and math.isnan(f.se_sigma0_um)   # sigma0_um None
    assert np.array_equal(f.B_gl, f.beta[:, 0])
    assert f.status in SF.STATUSES and f.n_eval > 0 and f.seconds > 0
    assert np.all(np.isfinite(f.theta)) and np.all(np.isfinite(f.se)) and np.all(np.isfinite(f.cov))
    assert isinstance(SF.ellipsoid_area(1.0, 2.0, 3.0), float)
    try:
        f.theta = None
        raise AssertionError("SomaFit is not frozen")
    except dataclasses.FrozenInstanceError:
        pass

    # a uint8 block (the real data's dtype) gives the fit of its float64 copy; the inputs are not modified
    sc = scene()
    data8 = np.clip(np.round(noisy(sc, SEED + 5)), 0, 255).astype(np.uint8)
    keep = data8.copy()
    mask = np.zeros((sc["H"], sc["W"]), dtype=bool)
    mask[:5, :5] = True
    fa = fit(sc, data=data8, mask=mask)
    fb = fit(sc, data=data8.astype(np.float64), mask=mask)
    assert np.array_equal(data8, keep) and mask.sum() == 25 and mask[:5, :5].all()
    assert np.array_equal(fa.theta, fb.theta) and fa.chi2 == fb.chi2, (fa.theta - fb.theta)


def test_determinism():
    """T8: the fit is deterministic (no randomness): two fits of one input are identical, seconds aside."""
    sc = scene()
    data = noisy(sc, SEED + 6)
    a = fit(sc, data=data)
    b = fit(sc, data=data)
    for field in dataclasses.fields(SF.SomaFit):
        if field.name == "seconds":
            continue
        va, vb = getattr(a, field.name), getattr(b, field.name)
        if isinstance(va, np.ndarray):
            assert nan_equal(va, vb), field.name
        elif isinstance(va, float) and math.isnan(va):
            assert math.isnan(vb), field.name
        else:
            assert va == vb, (field.name, va, vb)


def test_edge_cases():
    """T8: a NaN forward, the backward Jacobian at an upper bound and at a NaN, and the ValueErrors (T2's too)."""
    sc = scene()

    def nan_factory(z):
        return lambda th, g: np.full((len(z), sc["H"], sc["W"]), np.nan)
    f = fit(sc, factory=nan_factory)
    assert f.status == "failed" and np.all(np.isnan(f.theta)) and np.all(np.isnan(f.se)), f
    assert np.all(np.isnan(f.cov)) and np.all(np.isnan(f.B_gl)) and math.isnan(f.chi2) and f.at_bound == ()
    for name in ("area_um2", "se_area_um2", "volume_um3", "r_eq_um"):
        assert math.isnan(getattr(f, name)), name
    assert f.n_eval == 1 and f.ks.size > 0 and f.fitted == SF.PARAMS and "not finite" in f.message, f.message
    assert f.beta.shape == (f.ks.size, 3) and np.all(np.isnan(f.beta)) and f.cov_fit.shape == (9, 9)
    assert np.all(np.isnan(f.cov_fit)) and math.isnan(f.sigma0_um) and math.isnan(f.se_sigma0_um)

    # the truth's a (4.6 um) within the step (0.02 um) of the axes' upper bound 4.61 um (b 3.4, c 3.0 and the start's
    # axes below it): a forward NaN beyond the bound is never met (the backward column), and the fit converges to
    # the truth
    a_true = float(TRUTH[3])

    def guarded(limit, record):
        def factory(z):
            fwd = sc["toy"].factory(z)

            def forward(th, g):
                record.append(float(th[3]))
                if th[3] > limit:
                    return np.full((len(z), sc["H"], sc["W"]), np.nan)
                return fwd(th, g)
            return forward
        return factory
    seen = []
    f2 = fit(sc, scfg=dataclasses.replace(tight(), axis_bounds_um=(1.0, a_true + 0.01)),
             factory=guarded(a_true + 0.01, seen))
    assert f2.status == "converged" and max(seen) <= a_true + 0.01, (f2.status, max(seen), f2.message)
    assert np.all(np.abs(f2.theta - TRUTH) <= 1e-4), f2.theta - TRUTH
    # a NaN beyond a = 4.605 um inside the bounds (Block 42's rule): (i) the Jacobian takes the backward column where
    # the forward step's residual is not finite, the forward column elsewhere [exact: the same arithmetic];
    # (ii) a fit through it returns without raising, its iterate on the finite side (the NaN wall stops trf's
    # descent there: the entry does not promise convergence through it, and Block 45 is finite inside the bounds);
    # (iii) a column NaN both ways ends the fit with status "failed", without raising
    scfg = fit_cfg()
    sel = SF.select_planes(sc["ks"], sc["valid"], sc["centre"][2], DZ_UM, scfg.plane_half_um, scfg.plane_step)
    region = disk_um(sc, scfg.region_radius_um)
    lo, hi = (t[:9] for t in SF._bounds(scfg, *sc["centre"]))
    fwd = guarded(a_true + 0.005, [])(sc["ks"][sel] * DZ_UM)
    U, V = uv_um(sc)
    prob = SF._Problem(sc["block"][sel][:, region], region, U[region], V[region], (sel.size, sc["H"], sc["W"]), fwd,
                       np.r_[TRUTH, 0.0], np.arange(9), "tilted", False, hi, np.asarray(scfg.diff_steps), SF.PARAMS)
    J = prob.jac(TRUTH)
    r0 = prob.compute(TRUTH)[0]
    for j, h in enumerate(scfg.diff_steps):
        xs = TRUTH.copy()
        xs[j] += -h if j == 3 else h                    # a: a + h = 4.62 > 4.605 -> NaN -> backward
        col = (prob.compute(xs)[0] - r0) / (xs[j] - TRUTH[j])
        assert np.array_equal(J[:, j], col), SF.PARAMS[j]
    seen = []
    f3 = fit(sc, scfg=dataclasses.replace(tight(), axis_bounds_um=(1.0, 4.8)), factory=guarded(a_true + 0.005, seen))
    assert f3.status in ("converged", "max_nfev") and np.all(np.isfinite(f3.theta)), (f3.status, f3.message)
    assert f3.theta[3] <= a_true + 0.005 and max(seen) > a_true + 0.005, (f3.theta[3], max(seen))
    k0 = int(np.argmin(np.abs(sc["ks"] * DZ_UM - sc["centre"][2])))
    a_start = float(np.clip(SF.start_theta(sc["block"][k0], sc["left"], sc["top"], P_UM, sc["centre"], sc["radius"],
                                           region, SLOPE0), lo, hi)[3])

    def only_start(z):
        fwd0 = sc["toy"].factory(z)
        return lambda th, g: (fwd0(th, g) if th[3] == a_start else np.full((len(z), sc["H"], sc["W"]), np.nan))
    f4 = fit(sc, factory=only_start)
    assert f4.status == "failed" and "not finite either way" in f4.message and np.all(np.isnan(f4.theta)), f4.message
    report("T8: NaN forward -> %s after %d evaluation; a within its bound's step: converged, max |error| %.1e; NaN "
           "beyond a = 4.605 um: backward column, the fit %s at a = %.4f um; a column NaN both ways -> %s"
           % (f.status, f.n_eval, float(np.max(np.abs(f2.theta - TRUTH))), f3.status, f3.theta[3], f4.status))

    # the ValueErrors
    ks = np.arange(100, 141)
    v = np.ones(ks.size, dtype=bool)
    bad = {
        "area 0": lambda: SF.ellipsoid_area(0.0, 1.0, 1.0),
        "area < 0": lambda: SF.ellipsoid_area(1.0, -1.0, 1.0),
        "area nan": lambda: SF.ellipsoid_area(1.0, 1.0, float("nan")),
        "area inf": lambda: SF.ellipsoid_area(float("inf"), 1.0, 1.0),
        "select step 0": lambda: SF.select_planes(ks, v, 30.0, 0.25, 2.0, 0),
        "select step 1.5": lambda: SF.select_planes(ks, v, 30.0, 0.25, 2.0, 1.5),
        "select half < 0": lambda: SF.select_planes(ks, v, 30.0, 0.25, -0.1, 1),
        "select < 3": lambda: SF.select_planes(ks, v, 30.0, 0.25, 0.2, 1),            # k = 120 only
        "select < 3 by step": lambda: SF.select_planes(ks, v, 30.0, 0.25, 0.6, 3),    # 118..122, 3 | k - 120: 120
        "select none valid": lambda: SF.select_planes(ks, np.zeros(ks.size, bool), 30.0, 0.25, 2.0, 1),
        "select not ascending": lambda: SF.select_planes(ks[::-1], v, 30.0, 0.25, 2.0, 1),
        "select valid shape": lambda: SF.select_planes(ks, v[:-1], 30.0, 0.25, 2.0, 1),
        "select dz 0": lambda: SF.select_planes(ks, v, 30.0, 0.0, 2.0, 1),
        "select z nan": lambda: SF.select_planes(ks, v, float("nan"), 0.25, 2.0, 1),
        "vp shapes": lambda: SF.vp_scale(np.ones((2, 3, 4)), np.ones((2, 3, 5)), np.ones((3, 4), bool)),
        "vp 2-D": lambda: SF.vp_scale(np.ones((3, 4)), np.ones((3, 4)), np.ones((3, 4), bool)),
        "vp region dtype": lambda: SF.vp_scale(np.ones((2, 3, 4)), np.ones((2, 3, 4)), np.ones((3, 4))),
        "vp region empty": lambda: SF.vp_scale(np.ones((2, 3, 4)), np.ones((2, 3, 4)), np.zeros((3, 4), bool)),
        "start region shape": lambda: SF.start_theta(sc["block"][0], sc["left"], sc["top"], P_UM, sc["centre"], 4.0,
                                                     np.ones((3, 3), bool), 0.79),
        "start region empty": lambda: SF.start_theta(sc["block"][0], sc["left"], sc["top"], P_UM, sc["centre"], 4.0,
                                                     np.zeros((sc["H"], sc["W"]), bool), 0.79),
        "start radius 0": lambda: SF.start_theta(sc["block"][0], sc["left"], sc["top"], P_UM, sc["centre"], 0.0,
                                                 disk_um(sc, 13.0), 0.79),
        "start centre nan": lambda: SF.start_theta(sc["block"][0], sc["left"], sc["top"], P_UM, [0.0, np.nan, 1.0],
                                                   4.0, disk_um(sc, 13.0), 0.79),
        "start nan data": lambda: SF.start_theta(np.full((sc["H"], sc["W"]), np.nan), sc["left"], sc["top"], P_UM,
                                                 sc["centre"], 4.0, disk_um(sc, 13.0), 0.79),
        "start B_far 0": lambda: SF.start_theta(np.zeros((sc["H"], sc["W"])), sc["left"], sc["top"], P_UM,
                                                sc["centre"], 4.0, disk_um(sc, 13.0), 0.79),
        "fit far_slope": lambda: fit(sc, scfg=dataclasses.replace(SomaFitConfig(), far_slope="free")),
        "fit block 2-D": lambda: SF.fit_soma(sc["block"][0], sc["ks"], sc["valid"], sc["left"], sc["top"], P_UM,
                                             DZ_UM, sc["centre"], sc["radius"], np.zeros((sc["H"], sc["W"]), bool),
                                             sc["toy"].factory, fit_cfg(), SLOPE0),
        "fit ks length": lambda: SF.fit_soma(sc["block"], sc["ks"][1:], sc["valid"], sc["left"], sc["top"], P_UM,
                                             DZ_UM, sc["centre"], sc["radius"], np.zeros((sc["H"], sc["W"]), bool),
                                             sc["toy"].factory, fit_cfg(), SLOPE0),
        "fit valid length": lambda: SF.fit_soma(sc["block"], sc["ks"], sc["valid"][1:], sc["left"], sc["top"], P_UM,
                                                DZ_UM, sc["centre"], sc["radius"], np.zeros((sc["H"], sc["W"]), bool),
                                                sc["toy"].factory, fit_cfg(), SLOPE0),
        "fit mask shape": lambda: fit(sc, mask=np.zeros((sc["H"], sc["W"] + 1), bool)),
        "fit mask dtype": lambda: fit(sc, mask=np.zeros((sc["H"], sc["W"]))),
        "fit region empty": lambda: fit(sc, mask=np.ones((sc["H"], sc["W"]), bool)),
        "fit < 3 planes": lambda: fit(sc, scfg=fit_cfg(plane_half_um=0.2)),
        "fit radius 0": lambda: fit(sc, radius=0.0),
        "fit slope0 < 0": lambda: fit(sc, slope0=-0.1),
        "fit slope0 nan": lambda: fit(sc, slope0=float("nan")),
        "fit res0 0": lambda: SF.fit_soma(sc["block"], sc["ks"], sc["valid"], sc["left"], sc["top"], 0.0, DZ_UM,
                                          sc["centre"], sc["radius"], np.zeros((sc["H"], sc["W"]), bool),
                                          sc["toy"].factory, fit_cfg(), SLOPE0),
        "fit data nan": lambda: fit(sc, data=np.where(disk_um(sc, 1.0)[None], np.nan, sc["block"])),
        "fit forward shape": lambda: fit(sc, factory=lambda z: (lambda th, g: np.ones((len(z), 3, 3)))),
    }
    missing = [name for name, fn in bad.items() if not raises_value_error(fn)]
    assert not missing, "no ValueError for: %s" % missing


def test_wave8_vp_background():
    """T9: vp_background exact on a tilted product, against lstsq written out, against vp_scale without a tilt;
    background_planes against its formula."""
    rng = make_rng(SEED + 20)
    K, H, W = 4, 21, 27
    tau = rng.uniform(0.2, 1.0, (K, H, W))
    region = rng.uniform(size=(H, W)) < 0.6
    U = rng.uniform(-12.0, 12.0, (H, W))
    V = rng.uniform(-12.0, 12.0, (H, W))
    bt = np.array([[150.0, 0.3, -0.2], [97.5, -0.25, 0.1], [210.25, 0.05, 0.3], [3.0, 0.01, -0.02]])
    D = (bt[:, 0, None, None] + bt[:, 1, None, None] * U + bt[:, 2, None, None] * V) * tau
    beta, resid = SF.vp_background(D, tau, region, U, V)
    assert beta.shape == (K, 3) and resid.shape == (K, int(region.sum())) and beta.dtype == np.float64
    # roundoff of a 3-column least squares (condition number ~ 30): 1e-9 relative, 1e-8 gl
    check_close("T9 beta exact", beta, bt, rtol=1e-9, atol=1e-12)
    check_close("T9 resid exact", resid, np.zeros_like(resid), rtol=0, atol=1e-8)
    # numpy.linalg.lstsq written out for one plane, on noisy data
    Dn = D + rng.normal(0.0, 1.5, D.shape)
    beta_n, resid_n = SF.vp_background(Dn, tau, region, U, V)
    for k in range(K):
        t = tau[k][region]
        M = np.column_stack([t, U[region] * t, V[region] * t])
        sol = np.linalg.lstsq(M, Dn[k][region], rcond=None)[0]
        check_close("T9 lstsq plane %d" % k, beta_n[k], sol, rtol=1e-10, atol=1e-10)   # the same call: roundoff
        check_close("T9 lstsq resid %d" % k, resid_n[k], Dn[k][region] - M @ sol, rtol=0, atol=1e-10)
    # b1 = b2 = 0: b0 equals vp_scale's B [1e-10 relative]
    Dc = bt[:, 0, None, None] * tau
    beta_c = SF.vp_background(Dc, tau, region, U, V)[0]
    check_close("T9 b0 = vp_scale B", beta_c[:, 0], SF.vp_scale(Dc, tau, region)[0], rtol=1e-10)
    check_close("T9 b1 = b2 = 0", beta_c[:, 1:], 0.0, rtol=0, atol=1e-10)
    # a plane of rank < 3 (tau zero on the region) and a plane not finite: NaN coefficients and residual, the others
    # unchanged
    tz = tau.copy()
    tz[1][region] = 0.0
    tz[2][region] = np.where(np.arange(int(region.sum())) == 3, np.nan, tz[2][region])
    bz, rz = SF.vp_background(D, tz, region, U, V)
    assert np.all(np.isnan(bz[1:3])) and np.all(np.isnan(rz[1:3])), bz
    assert np.array_equal(bz[[0, 3]], beta[[0, 3]]) and np.array_equal(rz[[0, 3]], resid[[0, 3]])
    report("T9: exact tilt recovered to %.1e (relative), residual %.1e gl; lstsq per plane %.1e"
           % (float(np.max(np.abs(beta - bt) / np.abs(bt))), float(np.max(np.abs(resid))),
              float(np.max(np.abs(beta_n[-1] - sol)))))

    # background_planes against its formula at three pixels [1e-12]
    left, top, p, cxy = 37, -5, 0.1144, (5.1, 1.3)
    bp = SF.background_planes(bt, (H, W), left, top, p, cxy)
    assert bp.shape == (K, H, W) and bp.dtype == np.float64
    for (row, col) in ((0, 0), (7, 19), (H - 1, W - 1)):
        u, v = (left + col) * p - cxy[0], (top + row) * p - cxy[1]
        check_close("T9 background_planes %d,%d" % (row, col), bp[:, row, col], bt[:, 0] + bt[:, 1] * u + bt[:, 2] * v,
                    rtol=1e-12, atol=1e-12)
    bad = {
        "vpb shapes": lambda: SF.vp_background(D, tau[:, :, :-1], region, U, V),
        "vpb region dtype": lambda: SF.vp_background(D, tau, region.astype(float), U, V),
        "vpb region empty": lambda: SF.vp_background(D, tau, np.zeros_like(region), U, V),
        "vpb uv shape": lambda: SF.vp_background(D, tau, region, U[:, :-1], V),
        "bp beta shape": lambda: SF.background_planes(bt[:, :2], (H, W), left, top, p, cxy),
        "bp shape": lambda: SF.background_planes(bt, (H, W, 1), left, top, p, cxy),
        "bp res0": lambda: SF.background_planes(bt, (H, W), left, top, 0.0, cxy),
        "bp centre": lambda: SF.background_planes(bt, (H, W), left, top, p, (1.0, np.nan)),
    }
    missing = [name for name, fn in bad.items() if not raises_value_error(fn)]
    assert not missing, "no ValueError for: %s" % missing


def test_wave8_tilt_recovery():
    """T10: recovery under a tilt (noise-free): "tilted" recovers theta [1e-4] and beta [1e-5]; "constant" on the same
    data leaves chi2 at least 1e3 times larger. The constant fit's own relations (beta = (B, 0, 0), vp_scale, dof)."""
    sc = scene()
    data = tilted_data(sc)
    f = fit(sc, data=data, scfg=tight())
    assert f.status == "converged" and f.background == "tilted", f.message
    err = np.abs(f.theta - sc["truth"])
    assert np.all(err <= 1e-4), dict(zip(SF.PARAMS, err))
    b0, b1, b2 = tilt_coefficients(f.ks)
    berr = np.abs(f.beta - np.column_stack([b0, b1, b2]))
    assert np.all(berr <= 1e-5), float(berr.max())
    fc = fit(sc, data=data, scfg=fit_cfg(background="constant"))
    assert fc.background == "constant" and fc.status in ("converged", "max_nfev"), fc.message
    ratio = fc.chi2 / f.chi2 if f.chi2 > 0 else float("inf")
    assert fc.chi2 >= 1e3 * f.chi2, (fc.chi2, f.chi2)
    # the constant model's relations at its theta_hat (the model before wave 8): beta = (B, 0, 0), B from vp_scale,
    # dof with one scale per plane
    tau = sc["toy"].render(fc.theta[:8], fc.theta[8], fc.ks * DZ_UM)
    kidx = np.searchsorted(sc["ks"], fc.ks)
    B, resid = SF.vp_scale(data[kidx], tau, disk_um(sc, 13.0))
    check_close("T10 constant B", fc.B_gl, B, rtol=1e-12)
    assert np.all(fc.beta[:, 1:] == 0.0) and np.array_equal(fc.beta[:, 0], fc.B_gl)
    check_close("T10 constant chi2", fc.chi2, float(np.sum(resid ** 2)), rtol=1e-10)
    assert fc.dof == fc.n_samples - 9 - fc.ks.size and f.dof == f.n_samples - 9 - 3 * f.ks.size
    report("T10 tilt (slopes up to 0.3 gl/um): tilted max |theta error| %.1e (%s), max |beta error| %.1e, chi2 %.2e "
           "gl^2, %d evaluations; constant chi2 %.3e gl^2 (ratio %.2e), its max |theta error| %.3f"
           % (float(err.max()), SF.PARAMS[int(np.argmax(err))], float(berr.max()), f.chi2, f.n_eval, fc.chi2, ratio,
              float(np.max(np.abs(fc.theta - sc["truth"])))))


def test_wave8_hold():
    """T11: c held at the truth (exact value, NaN SE, not fitted, the others recovered, dof, SE(S) finite); c held
    20 % off gives a larger chi2 (on a noisy set, so both are > 0)."""
    sc = scene()
    c_true = float(sc["truth"][5])
    toy = ToyForward((sc["H"], sc["W"]), sc["left"], sc["top"], P_UM)
    f = SF.fit_soma(sc["block"], sc["ks"], sc["valid"], sc["left"], sc["top"], P_UM, DZ_UM, sc["centre"],
                    sc["radius"], np.zeros((sc["H"], sc["W"]), bool), toy.factory, tight(), SLOPE0,
                    hold={"c_um": c_true})
    assert f.status == "converged", f.message
    assert f.theta[5] == c_true and np.isnan(f.se[5]) and "c_um" not in f.fitted and "c_um" not in f.at_bound
    assert f.fitted == tuple(n for n in SF.PARAMS if n != "c_um")
    assert np.all(np.isnan(f.cov[5, :])) and np.all(np.isnan(f.cov[:, 5]))
    assert all(call[0][5] == c_true for call in toy.calls), "a held c moved"
    others = [j for j in range(9) if j != 5]
    err = np.abs(f.theta[others] - sc["truth"][others])
    assert np.all(err <= 1e-4), err
    assert f.dof == f.n_samples - 8 - 3 * f.ks.size and f.cov_fit.shape == (8, 8)
    assert math.isfinite(f.se_area_um2), f.se_area_um2
    # SE(S) is the delta method over the fitted axes a, b only (c contributes no variance)
    g = SF._area_gradient(*f.theta[3:6])[:2]
    check_close("T11 SE(S) over a, b", f.se_area_um2, math.sqrt(g @ f.cov[3:5, 3:5] @ g), rtol=1e-12)
    data = noisy(sc, SEED + 30)
    f_t = fit(sc, data=data, scfg=fit_cfg())
    f_at = SF.fit_soma(data, sc["ks"], sc["valid"], sc["left"], sc["top"], P_UM, DZ_UM, sc["centre"], sc["radius"],
                       np.zeros((sc["H"], sc["W"]), bool), sc["toy"].factory, fit_cfg(), SLOPE0,
                       hold={"c_um": c_true})
    f_off = SF.fit_soma(data, sc["ks"], sc["valid"], sc["left"], sc["top"], P_UM, DZ_UM, sc["centre"], sc["radius"],
                        np.zeros((sc["H"], sc["W"]), bool), sc["toy"].factory, fit_cfg(), SLOPE0,
                        hold={"c_um": 1.2 * c_true})
    assert f_off.chi2 > f_at.chi2, (f_off.chi2, f_at.chi2)
    report("T11 c held at the truth: max |error| of the others %.1e, SE(S) %.2e um^2 (noise-free), dof %d; noisy: "
           "chi2 c free %.1f, c held at the truth %.1f, held 20 %% off %.1f gl^2"
           % (float(err.max()), f.se_area_um2, f.dof, f_t.chi2, f_at.chi2, f_off.chi2))


def test_wave8_start_and_c_bounds():
    """T12: a start at the truth returns the truth; c_bounds_um excluding the true c puts c at the nearer bound; the
    start's slope entry ignored and held entries set; the ValueErrors of the amendment's (b) to (d)."""
    sc = scene()
    toy = ToyForward((sc["H"], sc["W"]), sc["left"], sc["top"], P_UM)
    st = sc["truth"].copy()
    f = SF.fit_soma(sc["block"], sc["ks"], sc["valid"], sc["left"], sc["top"], P_UM, DZ_UM, sc["centre"],
                    sc["radius"], np.zeros((sc["H"], sc["W"]), bool), toy.factory, fit_cfg(), float(st[8]),
                    start=st)
    sel = np.isin(sc["ks"], f.ks)
    d2 = float(np.sum(sc["block"][sel][:, disk_um(sc, 13.0)] ** 2))
    err = np.abs(f.theta - sc["truth"])
    assert np.all(err <= 1e-8) and f.chi2 < 1e-12 * d2, (err, f.chi2, d2)
    # the start's slope entry is ignored (slope0 rules); held entries take their held values; then clipped
    toy2 = ToyForward((sc["H"], sc["W"]), sc["left"], sc["top"], P_UM)
    st2 = sc["truth"].copy()
    st2[8] = 99.0
    st2[3] = 40.0                                           # beyond the axes' bound: clipped to 15
    SF.fit_soma(sc["block"], sc["ks"], sc["valid"], sc["left"], sc["top"], P_UM, DZ_UM, sc["centre"], sc["radius"],
                np.zeros((sc["H"], sc["W"]), bool), toy2.factory, fit_cfg(max_nfev=1), SLOPE0, start=st2,
                hold={"mu_per_um": 0.2})
    th0, g0 = toy2.calls[0]
    assert g0 == SLOPE0 and th0[3] == 15.0 and th0[7] == 0.2 and np.array_equal(th0[[0, 1, 2, 4, 5, 6]],
                                                                                 st2[[0, 1, 2, 4, 5, 6]]), (th0, g0)
    # c_bounds_um that exclude the true c (3.0 um): c at the nearer bound, in at_bound
    fb = SF.fit_soma(
        sc["block"], sc["ks"], sc["valid"], sc["left"], sc["top"], P_UM, DZ_UM, sc["centre"], sc["radius"],
        np.zeros((sc["H"], sc["W"]), bool), sc["toy"].factory, fit_cfg(), SLOPE0, c_bounds_um=(3.4, 6.0))
    assert "c_um" in fb.at_bound and abs(fb.theta[5] - 3.4) <= 1e-3 * (6.0 - 3.4), (fb.at_bound, fb.theta[5])
    report("T12 start at the truth: max |error| %.1e, chi2 / sum d^2 = %.1e, %d evaluations; c_bounds (3.4, 6.0): "
           "c = %.6f um, at_bound %s" % (float(err.max()), f.chi2 / d2, f.n_eval, fb.theta[5], fb.at_bound))

    z = np.zeros((sc["H"], sc["W"]), bool)

    def call(scfg=None, **kw):
        return lambda: SF.fit_soma(sc["block"], sc["ks"], sc["valid"], sc["left"], sc["top"], P_UM, DZ_UM,
                                   sc["centre"], sc["radius"], z, sc["toy"].factory,
                                   fit_cfg(max_nfev=1) if scfg is None else scfg, SLOPE0, **kw)
    lo_c = float(sc["centre"][0]) - 4.0
    bad = {
        "hold unknown": call(hold={"d_um": 3.0}),
        "hold slope": call(hold={"slope_um_per_um": 0.8}),
        "hold sigma0": call(hold={"sigma0_um": 0.3}),
        "hold nan": call(hold={"c_um": float("nan")}),
        "hold outside": call(hold={"c_um": 16.0}),
        "hold x0 outside": call(hold={"x0_um": lo_c - 0.01}),
        "hold outside c_bounds": call(hold={"c_um": 3.0}, c_bounds_um=(3.4, 6.0)),
        "hold not a mapping": call(hold=[("c_um", 3.0)]),
        "start shape": call(start=np.zeros(8)),
        "start nan": call(start=np.r_[sc["truth"][:5], np.nan, sc["truth"][6:]]),
        "c_bounds order": call(c_bounds_um=(5.0, 3.0)),
        "c_bounds zero": call(c_bounds_um=(0.0, 3.0)),
        "c_bounds nan": call(c_bounds_um=(1.0, float("nan"))),
        "c_bounds length": call(c_bounds_um=(1.0, 2.0, 3.0)),
        "background name": call(scfg=fit_cfg(background="planar")),
        "sigma0 name": call(scfg=fit_cfg(sigma0="free")),
        "sigma0 fit without sigma0_um": call(scfg=fit_cfg(sigma0="fit")),
        "sigma0 fit sigma0_um 0": call(scfg=fit_cfg(sigma0="fit"), sigma0_um=0.0),
        "sigma0_um nan": call(sigma0_um=float("nan")),
        "all held": call(scfg=fit_cfg(far_slope="fixed"), hold=dict(zip(SF.PARAMS[:8], sc["truth"][:8]))),
    }
    missing = [name for name, fn in bad.items() if not raises_value_error(fn)]
    assert not missing, "no ValueError for: %s" % missing
    # a start whose slope entry is NaN is accepted (the entry is ignored)
    ok = call(start=np.r_[sc["truth"][:8], np.nan])()
    assert ok.status in ("converged", "max_nfev"), ok.message


def test_wave8_sigma0():
    """T13: sigma0 "fit" with a stand-in forward taking sigma0_um (blur in quadrature), truth 30 % from the start:
    recovered [1e-4 um]; fitted ends with sigma0_um; cov_fit and se_sigma0_um. Under "fixed" a forward without the
    keyword is called without error and sigma0_um is recorded as passed."""
    sc0 = scene()
    toy = ToyForwardS0((sc0["H"], sc0["W"]), sc0["left"], sc0["top"], P_UM)
    s0_true = 0.40
    s0_start = s0_true / 1.3                               # the truth 30 % above the start
    tau = toy.render_s0(TRUTH[:8], TRUTH[8], sc0["ks"] * DZ_UM, s0_true)
    data = sc0["Bk"][:, None, None] * tau
    scfg = fit_cfg(ftol=1e-10, xtol=1e-10, sigma0="fit")
    f = SF.fit_soma(data, sc0["ks"], sc0["valid"], sc0["left"], sc0["top"], P_UM, DZ_UM, sc0["centre"],
                    sc0["radius"], np.zeros((sc0["H"], sc0["W"]), bool), toy.factory, scfg, SLOPE0,
                    sigma0_um=s0_start)
    assert f.status == "converged", f.message
    first = toy.calls[0]
    assert first[2] == s0_start, first[2]                  # the start: the sigma0_um argument
    assert f.fitted == SF.PARAMS + ("sigma0_um",) and f.cov_fit.shape == (10, 10), (f.fitted, f.cov_fit.shape)
    assert abs(f.sigma0_um - s0_true) <= 1e-4, f.sigma0_um
    err = np.abs(f.theta - TRUTH)
    assert np.all(err <= 1e-4), dict(zip(SF.PARAMS, err))
    assert f.dof == f.n_samples - 10 - 3 * f.ks.size
    # on a noisy set: se_sigma0_um is cov_fit's last diagonal's root, cov the first nine's block
    fn = SF.fit_soma(data + make_rng(SEED + 40).normal(0.0, NOISE_GL, data.shape), sc0["ks"], sc0["valid"],
                     sc0["left"], sc0["top"], P_UM, DZ_UM, sc0["centre"], sc0["radius"],
                     np.zeros((sc0["H"], sc0["W"]), bool), toy.factory, fit_cfg(sigma0="fit"), SLOPE0,
                     sigma0_um=s0_start)
    assert fn.status in ("converged", "max_nfev"), fn.message
    check_close("T13 se_sigma0", fn.se_sigma0_um, math.sqrt(fn.cov_fit[-1, -1]), rtol=1e-15)
    check_close("T13 cov block", fn.cov, fn.cov_fit[:9, :9], rtol=0, atol=0)
    assert math.isfinite(fn.se_sigma0_um) and fn.se_sigma0_um > 0
    # a start outside sigma0_bounds_um (0.02, 0.6) is clipped into it
    toy_c = ToyForwardS0((sc0["H"], sc0["W"]), sc0["left"], sc0["top"], P_UM)
    fc = SF.fit_soma(data, sc0["ks"], sc0["valid"], sc0["left"], sc0["top"], P_UM, DZ_UM, sc0["centre"],
                     sc0["radius"], np.zeros((sc0["H"], sc0["W"]), bool), toy_c.factory, fit_cfg(sigma0="fit",
                     max_nfev=1), SLOPE0, sigma0_um=0.9)
    assert toy_c.calls[0][2] == 0.6 and "sigma0_um" in fc.message, (toy_c.calls[0][2], fc.message)
    # "fixed": a forward that accepts no keyword (ToyForward's), sigma0_um recorded as passed, its SE NaN
    sc = scene()
    ff = SF.fit_soma(sc["block"], sc["ks"], sc["valid"], sc["left"], sc["top"], P_UM, DZ_UM, sc["centre"],
                     sc["radius"], np.zeros((sc["H"], sc["W"]), bool), sc["toy"].factory, fit_cfg(max_nfev=2),
                     SLOPE0, sigma0_um=0.31)
    assert ff.sigma0_um == 0.31 and math.isnan(ff.se_sigma0_um) and "sigma0_um" not in ff.fitted
    report("T13 sigma0 fit (truth %.3f um, start %.4f um): sigma0 %.8f um (error %.1e), max |theta error| %.1e, %d "
           "evaluations, %.1f s; noisy: SE(sigma0) %.2e um, sigma0 %.4f um"
           % (s0_true, s0_start, f.sigma0_um, abs(f.sigma0_um - s0_true), float(err.max()), f.n_eval, f.seconds,
              fn.se_sigma0_um, fn.sigma0_um))


def test_wave8_region():
    """T14: region_radius_um None -> the disc of 1.2 x the traced radius minus the mask (n_samples, exact count); a
    number overrides the scale; the scale is read."""
    sc = scene()
    x = (sc["left"] + np.arange(sc["W"])) * P_UM - sc["centre"][0]
    y = (sc["top"] + np.arange(sc["H"])) * P_UM - sc["centre"][1]
    mask = (np.abs(x)[None, :] <= 1.0) & (y[:, None] <= -2.0)
    base = dataclasses.replace(SomaFitConfig(), max_nfev=1)
    out = []
    for scfg, r_expect in ((base, 1.2 * sc["radius"]),
                           (dataclasses.replace(base, region_radius_scale=1.5), 1.5 * sc["radius"]),
                           (dataclasses.replace(base, region_radius_um=5.0), 5.0),
                           (dataclasses.replace(base, region_radius_um=5.0, region_radius_scale=2.0), 5.0)):
        f = fit(sc, scfg=scfg, mask=mask)
        n_reg = int(np.count_nonzero(disk_um(sc, r_expect) & ~mask))
        assert f.n_samples == f.ks.size * n_reg, (f.n_samples, f.ks.size, n_reg)
        out.append(n_reg)
    report("T14: region pixels %d (1.2 r_s = %.3f um, masked), %d (scale 1.5), %d (5 um override, twice)"
           % (out[0], 1.2 * sc["radius"], out[1], out[2]))


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main():
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
    print(_environment())

    results = []
    for name, func in checks:
        start = time.perf_counter()
        print(f"-- {name}", flush=True)
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

    print()
    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
