"""Independent references for the spec-derived tests of wave 7 (Block 49, Block 30 amendment 3, Block 36
amendments 22-23, Block 1's new inference fields; integration tester, 2026-11-23 13:55). Written from
specs/SPEC.md only (section 2.2's (3''), Block 12's (H1)-(H3) and its test oracle's closed forms, Block 49's
mathematics, Block 4 amendment 2, Block 42's summary): nothing here imports or calls the code under test
(allen_diameter.model.line_sgt, line_fast, analysis.node_inference, node_inputs, sigma_calibration, scripts/*).

  FWHM_PER_SD                 2 sqrt(2 ln 2)
  unit_fwhm(beta, xi, zeta)   W(kappa) of (H3): (zeta + 1/zeta) [expm1(e ln 2)/xi]^(1/beta), e = xi beta/(beta + xi);
                              (zeta + 1/zeta) (ln 2)^(1/beta) at xi = 0
  core_pdf(x, beta, xi)       h_{beta,xi}(x) of (H1), unit scale, symmetric
  core_half_mass(y, b, xi)    H(y) = int_0^y h: I_t(1/beta, 1/xi)/2, t = xi y^beta/(1 + xi y^beta); P(1/beta, y^beta)/2
  sgt_pdf(z, w, b, xi, zeta)  g_{w,kappa}(z) = (1/a) f(z/a | kappa), a = w/W(kappa), (H2) right side z > 0
  sgt_cdf(z, w, b, xi, zeta)  its CDF from H (Block 12's mass on [0, X] and [-X, 0])
  tail_mass_2d(beta, xi, r)   the isotropic 2-D kernel's mass beyond r (core units) by the Abel identity
                              M(r) = -2 int_r^inf h'(x) sqrt(x^2 - r^2) dx (scipy.integrate.quad)
  flat_line_ref(...)          Block 49's quantity for a tube along x read at pixel centres, depth-independent law,
                              "linear" absorption: the 1-D model of the projection (see the function)
  with_kernel_table(...)      Block 4 amendment 2's table, by hand
  summary_ref(x, se, ok)      Block 42's summarize statistics, by hand (MAD unscaled, accepted at Block 42's gate)
  sgt_tube_profile(...)       a stand-in forward for Block 30: (3'') with phi = 0 and an SGT kernel, own
                              Gauss-Legendre quadrature in psi

Pure ASCII.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, special

FWHM_PER_SD = 2.0 * math.sqrt(2.0 * math.log(2.0))
BASE_COMMIT = "f1bc7cb57fa801d105bd24028d32da9f1b66d9fc"     # the incremental run's base (before wave 7)
_PKG = "Passive Features/Diameter Re-measurement"


def run_at_commit(code, out_dir, commit=BASE_COMMIT, timeout=1800):
    """Run the Python source `code` with the package `src` of `commit` (git archive of the repository's own history
    into out_dir/base_<commit>) first on sys.path; the code writes its results where it likes (out_dir is passed as
    the environment variable OUT). -> CompletedProcess (check=True). Used for the regression oracles "unchanged bit
    for bit" (Block 35 amendment 3, Block 30 amendment 3 (vii))."""
    here = Path(__file__).resolve().parent
    pkg = here.parent.parent
    repo = pkg.parent.parent
    out_dir = Path(out_dir)
    base = out_dir / ("base_" + commit[:7])
    if not (base / _PKG / "src").is_dir():
        base.mkdir(parents=True, exist_ok=True)
        arch = subprocess.run(["git", "-C", str(repo), "archive", commit, _PKG + "/src"], check=True,
                              capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(base)], input=arch, check=True)
    script = out_dir / ("run_%s.py" % commit[:7])
    script.write_text(code)
    env = dict(os.environ, PYTHONPATH=str(base / _PKG / "src"), PYTHONDONTWRITEBYTECODE="1", OUT=str(out_dir))
    return subprocess.run([sys.executable, str(script)], cwd=str(out_dir), env=env, check=True, capture_output=True,
                          text=True, timeout=timeout)
XI_ZERO = 1e-12          # Block 12's SGT_XI_ZERO: at or below it the xi = 0 member


# ------------------------------------------------------------------ (H1)-(H3) ---

def unit_fwhm(beta, xi, zeta=1.0):
    s = zeta + 1.0 / zeta
    if xi <= XI_ZERO:
        return s * math.log(2.0) ** (1.0 / beta)
    e = xi * beta / (beta + xi)
    return s * (math.expm1(e * math.log(2.0)) / xi) ** (1.0 / beta)


def _log_c(beta, xi):
    if xi <= XI_ZERO:
        return math.log(beta) - math.log(2.0) - special.gammaln(1.0 / beta)
    return math.log(beta) + math.log(xi) / beta - math.log(2.0) - special.betaln(1.0 / beta, 1.0 / xi)


def core_pdf(x, beta, xi):
    ax = np.abs(np.asarray(x, dtype=float))
    lc = _log_c(beta, xi)
    if xi <= XI_ZERO:
        return np.exp(lc - ax ** beta)
    return np.exp(lc - (1.0 / xi + 1.0 / beta) * np.log1p(xi * ax ** beta))


def core_dpdf(x, beta, xi):
    """h'(x) for x > 0."""
    x = np.asarray(x, dtype=float)
    lc = _log_c(beta, xi)
    if xi <= XI_ZERO:
        return -np.exp(lc + math.log(beta) + (beta - 1.0) * np.log(x) - x ** beta)
    return -np.exp(lc + math.log(beta + xi) + (beta - 1.0) * np.log(x)
                   - (1.0 / xi + 1.0 / beta + 1.0) * np.log1p(xi * x ** beta))


def core_half_mass(y, beta, xi):
    y = np.abs(np.asarray(y, dtype=float))
    if xi <= XI_ZERO:
        return 0.5 * special.gammainc(1.0 / beta, y ** beta)
    q = xi * y ** beta
    return 0.5 * special.betainc(1.0 / beta, 1.0 / xi, q / (1.0 + q))


def sgt_pdf(z, w, beta, xi, zeta=1.0):
    z = np.asarray(z, dtype=float)
    a = w / unit_fwhm(beta, xi, zeta)
    x = z / a
    s = zeta + 1.0 / zeta
    return np.where(x >= 0, (2.0 / s) * core_pdf(x / zeta, beta, xi), (2.0 / s) * core_pdf(zeta * x, beta, xi)) / a


def sgt_cdf(z, w, beta, xi, zeta=1.0):
    z = np.asarray(z, dtype=float)
    a = w / unit_fwhm(beta, xi, zeta)
    s = zeta + 1.0 / zeta
    base = 1.0 / (1.0 + zeta * zeta)
    right = base + (2.0 * zeta / s) * core_half_mass(z / (a * zeta), beta, xi)
    left = base - (2.0 / (s * zeta)) * core_half_mass(zeta * np.abs(z) / a, beta, xi)
    return np.where(z >= 0, right, left)


def tail_mass_2d(beta, xi, r):
    """1 - (mass of the isotropic 2-D PSF within radius r), core units, the PSF whose line projection is h.
    M(r) = -2 int_r^inf h'(x) sqrt(x^2 - r^2) dx, x = r cosh(s)."""
    r = float(r)
    if r == 0.0:
        return 1.0

    def f(s):
        x = r * math.cosh(s)
        sh = math.sinh(s)
        return -2.0 * float(core_dpdf(x, beta, xi)) * r * r * sh * sh

    # the integrand is smooth; split the range so that quad sees the peak
    total = 0.0
    edges = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 40.0, 120.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        total += integrate.quad(f, lo, hi, epsabs=1e-17, epsrel=1e-13, limit=400)[0]
    return total


# ----------------------------------------------------- Block 49's 1-D reduction ---

def fine_ys(y_lo, y_hi, p, f):
    """The fine lattice's sample centres in y between y_lo and y_hi: pixels centred on multiples of p, each cut
    into f samples, centres at (i + 1/2) h from the pixel's lower edge (Block 4's block_fine_grid)."""
    h = p / f
    n0 = int(math.floor((y_lo + 0.5 * p) / h - 0.5)) - 2
    n1 = int(math.ceil((y_hi + 0.5 * p) / h - 0.5)) + 2
    return (np.arange(n0, n1 + 1) + 0.5) * h - 0.5 * p


def psf_radial(rho, beta, xi):
    """P_1(rho): the isotropic 2-D PSF (core units) whose line projection is h, for the members where it is known
    in closed form: beta = 2 (the bivariate t, (1/pi)(1 + xi rho^2)^-(1/xi + 1); the Gaussian (1/pi) e^{-rho^2} at
    xi = 0) and the Laplace (1, 0), K_0(rho)/(2 pi) (its transfer 1/(1 + 4 pi^2 nu^2))."""
    rho = np.asarray(rho, dtype=float)
    if beta == 2.0 and xi <= XI_ZERO:
        return np.exp(-rho * rho) / math.pi
    if beta == 2.0:
        return (1.0 + xi * rho * rho) ** (-(1.0 / xi + 1.0)) / math.pi
    if beta == 1.0 and xi <= XI_ZERO:
        return special.k0(rho) / (2.0 * math.pi)
    raise ValueError("no closed form for (%g, %g)" % (beta, xi))


def trunc_core_cdf(y, beta, xi, rho_c):
    """G(y) = int_{-inf}^y A_tr(x) dx, A_tr the line projection of P_1 truncated at rho_c (its mass beyond dropped):
    the 2-D mass of the disc rho <= rho_c in the half-plane x <= y. With Q(t) = mass in 0 < x <= t,
    Q(t) = (1/2)(M(0) - M(min(t, rho_c))) + int_t^{rho_c} 2 rho P_1(rho) arcsin(t/rho) drho (t < rho_c),
    M the untruncated tail mass (closed form here), G(y) = (1 - M(rho_c))/2 + sign(y) Q(|y|)."""
    def M(r):
        if beta == 2.0 and xi <= XI_ZERO:
            return math.exp(-r * r)
        if beta == 2.0:
            return (1.0 + xi * r * r) ** (-1.0 / xi)
        return r * special.k1(r) if r > 0 else 1.0

    def Q(t):
        T = min(t, rho_c)
        q = 0.5 * (1.0 - M(T))
        if t < rho_c:
            f = lambda rr: 2.0 * rr * float(psf_radial(rr, beta, xi)) * math.asin(min(1.0, t / rr))  # noqa: E731
            # split [t, rho_c] so that quad sees the integrand's peak near t (a long interval hides it)
            edges = [t]
            step = max(0.25, 0.5 * t)
            while edges[-1] + step < rho_c:
                edges.append(edges[-1] + step)
                step *= 1.6
            edges.append(rho_c)
            for lo, hi in zip(edges[:-1], edges[1:]):
                q += integrate.quad(f, lo, hi, epsabs=1e-16, epsrel=1e-12, limit=400)[0]
        return q

    y = float(y)
    half = 0.5 * (1.0 - M(rho_c))
    return half + math.copysign(Q(abs(y)), y) if y != 0.0 else half


def trunc_sgt_cdf(z, w, beta, xi, zeta, reach_sigmas):
    """The CDF of the line projection of Block 49's truncated, stretched kernel (D-225.3-4): core scale a =
    w/W(kappa), truncation radius rho_c = R/a = reach x W(kappa)/FWHM_PER_SD in the radial variable before the
    stretch; for u >= 0 the projection is (2/s) A_tr(u/(a zeta))/a, for u < 0 (2/s) A_tr(zeta u/a)/a, s = zeta +
    1/zeta. Evaluated on each distinct z."""
    W = unit_fwhm(beta, xi, zeta)
    a = w / W
    rho_c = reach_sigmas * W / FWHM_PER_SD
    s = zeta + 1.0 / zeta
    z = np.asarray(z, dtype=float)
    uniq, inv = np.unique(z, return_inverse=True)
    G0 = trunc_core_cdf(0.0, beta, xi, rho_c)
    out = np.empty(uniq.size)
    for i, zz in enumerate(uniq):
        if zz < 0:
            out[i] = (2.0 / (s * zeta)) * trunc_core_cdf(zeta * zz / a, beta, xi, rho_c)
        else:
            out[i] = (2.0 / (s * zeta)) * G0 + (2.0 * zeta / s) * (trunc_core_cdf(zz / (a * zeta), beta, xi, rho_c)
                                                                    - G0)
    return out[inv].reshape(z.shape)


def flat_line_ref(r, mu, y_c, phi, v, p, f, sigma, beta, xi, zeta, reach_sigmas=None):
    """Block 49's deficit 1 - tau at v (pixel centres on the measuring line along +y through x = 0) for a tube
    along x (heading 0) of radius r, tilt phi, axis through (., y_c), under "linear" absorption with a law of
    constant SD sigma (all planes equal), the kernel untruncated:
        deficit(v_n) = sum_m (h/p) mu l(y_m) [F(v_n + p/2 - y_m) - F(v_n - p/2 - y_m)],
    l(y) = 2 sqrt(r^2 - (y - y_c)^2) / cos(phi) the column's length in the tube (S3), F the CDF of the line
    projection g_{w,kappa} (w = FWHM_PER_SD sigma), y_m the fine samples. The pixel mean over x of the blur of an
    x-invariant lattice is the projection across y-hat exactly (each x is covered by f windows of the f x-samples
    per pixel); the pixel mean over y is the CDF difference; the bilinear read at pixel centres is the pixel.
    reach_sigmas: None, the untruncated kernel; a number, the kernel truncated as Block 49 states (trunc_sgt_cdf,
    members with a closed-form P_1 only)."""
    h = p / f
    y = fine_ys(y_c - r, y_c + r, p, f)
    ell = 2.0 * np.sqrt(np.clip(r * r - (y - y_c) ** 2, 0.0, None)) / math.cos(phi)
    src = (h / p) * mu * ell
    m = src > 0
    y, src = y[m], src[m]
    w = FWHM_PER_SD * sigma
    v = np.asarray(v, dtype=float)
    zhi = v[:, None] + 0.5 * p - y[None, :]
    zlo = v[:, None] - 0.5 * p - y[None, :]
    if reach_sigmas is None:
        hi = sgt_cdf(zhi, w, beta, xi, zeta)
        lo = sgt_cdf(zlo, w, beta, xi, zeta)
    else:
        # the arguments lie on a lattice of h/2 (v on multiples of p, y_m on odd multiples of h/2 about them):
        # rounding to it makes equal arguments exact repeats, which trunc_sgt_cdf evaluates once
        zhi = np.round(zhi / (0.5 * h)) * (0.5 * h)
        zlo = np.round(zlo / (0.5 * h)) * (0.5 * h)
        both = trunc_sgt_cdf(np.concatenate([zhi.ravel(), zlo.ravel()]), w, beta, xi, zeta, reach_sigmas)
        hi, lo = both[:zhi.size].reshape(zhi.shape), both[zhi.size:].reshape(zlo.shape)
    return (src[None, :] * (hi - lo)).sum(axis=1)


def flat_line_mass(r, mu, y_c, phi, p, f):
    """sum_m h mu l(y_m): the absorbed light per unit length booked on the fine lattice (um)."""
    h = p / f
    y = fine_ys(y_c - r, y_c + r, p, f)
    ell = 2.0 * np.sqrt(np.clip(r * r - (y - y_c) ** 2, 0.0, None)) / math.cos(phi)
    return float(h * mu * ell.sum())


# ------------------------------------------------ Block 4 amendment 2, by hand ---

def with_kernel_table(deltas, sigmas, slope, sigma0, c):
    """(new sigmas, new sigma_r0, new linear slope): sigma'(d)^2 = c (sigma(d)^2 - sigma(0)^2) + sigma0^2,
    sigma'(0) = sigma0, the linear continuation's slope times sqrt(c)."""
    s = np.asarray(sigmas, dtype=float)
    new = np.sqrt(c * (s * s - s[0] * s[0]) + sigma0 * sigma0)
    new[0] = sigma0
    return new, float(sigma0), float(slope) * math.sqrt(c)


# ------------------------------------------------------- Block 42's summary ---

def summary_ref(x, se, ok):
    """n, median, MAD (unscaled), inverse-variance weighted mean and its SE, chi2_cons, dof over ok."""
    x = np.asarray(x, float)[np.asarray(ok, bool)]
    se = np.asarray(se, float)[np.asarray(ok, bool)]
    n = x.size
    if n == 0:
        return dict(n=0, median=math.nan, mad=math.nan, ivw_mean=math.nan, ivw_se=math.nan, chi2_cons=math.nan,
                    dof=0)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    wts = 1.0 / se ** 2
    mean = float(np.sum(wts * x) / np.sum(wts))
    ivw_se = float(1.0 / math.sqrt(np.sum(wts)))
    chi2 = float(np.sum((x - mean) ** 2 * wts)) if n > 1 else math.nan
    return dict(n=n, median=med, mad=mad, ivw_mean=mean, ivw_se=ivw_se, chi2_cons=chi2, dof=n - 1)


# ------------------------------------------- a stand-in forward for Block 30 ---

_GL = {}


def _gl(n):
    if n not in _GL:
        x, wq = np.polynomial.legendre.leggauss(n)
        psi = 0.5 * math.pi * x
        _GL[n] = (np.sin(psi), np.cos(psi), 0.5 * math.pi * wq)
    return _GL[n]


def sgt_tube_profile(v, d, mu, v0, w, beta, xi, zeta, n_nodes=400):
    """tau(v) = 1 - (d/2) int (1 - exp(-alpha cos psi)) cos psi g_{w,kappa}(v - v0 - (d/2) sin psi) dpsi with
    alpha = mu d (section 2.2, (3''), phi = 0): the chord profile of a round tube convolved with the SGT kernel."""
    sp, cp, wq = _gl(n_nodes)
    alpha = mu * d
    v = np.asarray(v, dtype=float)
    g = sgt_pdf(v[:, None] - v0 - 0.5 * d * sp[None, :], w, beta, xi, zeta)
    integ = (-np.expm1(-alpha * cp)) * cp * wq
    return 1.0 - 0.5 * d * (g * integ[None, :]).sum(axis=1)


def make_sgt_forward(v, z_planes, w0=0.25, growth=0.8, n_nodes=400, calls=None):
    """forward(d, mu, v0, z_ax, beta, xi, zeta) -> (n_planes, n_v): each plane's profile is the tube's chord
    profile blurred by the SGT of FWHM w0 + growth |z_k - z_ax| (the shape entering as in the renderer: the
    law's width and the shape apart). calls: a list the arguments are appended to (a spy), or None."""
    v = np.asarray(v, dtype=float)
    z_planes = np.asarray(z_planes, dtype=float)

    def forward(d, mu, v0, z_ax, beta=2.0, xi=0.0, zeta=1.0):
        if calls is not None:
            calls.append((d, mu, v0, z_ax, beta, xi, zeta))
        out = np.empty((z_planes.size, v.size))
        for k, zk in enumerate(z_planes):
            out[k] = sgt_tube_profile(v, d, mu, v0, w0 + growth * abs(zk - z_ax), beta, xi, zeta, n_nodes)
        return out
    return forward


def make_gauss_forward(v, z_planes, w0=0.25, growth=0.8, n_nodes=400, calls=None):
    """The Gaussian member of make_sgt_forward with forward(d, mu, v0, z_ax) (Block 30's Gaussian arity)."""
    f7 = make_sgt_forward(v, z_planes, w0, growth, n_nodes, calls)

    def forward(d, mu, v0, z_ax):
        return f7(d, mu, v0, z_ax, 2.0, 0.0, 1.0)
    return forward


# ------------------------------------- a brute-force 2-D reference (Block 49 (iii)) ---

def kernel_2d(X, Y, a, xi, zeta, rho_c, yh):
    """Block 49's kernel at image-frame offsets (X, Y) (um), beta = 2 members: (2/(zeta + 1/zeta)) (1/a^2) P_1(rho),
    rho = sqrt((u/zeta)^2 + s^2)/a (u >= 0) or sqrt((zeta u)^2 + s^2)/a (u < 0), u = x . y_hat, s along the normal
    (yh_y, -yh_x), P_1 the bivariate t (the Gaussian at xi = 0), zero beyond rho_c."""
    u = X * yh[0] + Y * yh[1]
    s = X * yh[1] - Y * yh[0]
    uu = np.where(u >= 0, u / zeta, u * zeta)
    rho = np.sqrt(uu * uu + s * s) / a
    if xi <= XI_ZERO:
        P = np.exp(-rho * rho) / math.pi
    else:
        P = (1.0 + xi * rho * rho) ** (-(1.0 / xi + 1.0)) / math.pi
    return np.where(rho <= rho_c, P, 0.0) * (2.0 / (zeta + 1.0 / zeta)) / (a * a)


def brute_force_line(r, mu, phi, theta, c_xy, v, n_avg, step, p, f, sigma, xi, zeta, reach_sigmas, bilinear_read,
                     sub=16):
    """Block 49's tau (the law's SD constant, so one kernel for every slab and plane) for a tilted tube of radius r,
    tilt phi, heading theta, axis through c_xy, "linear" absorption, the measuring line through the origin along
    y-hat = (-sin theta, cos theta), e_u = (cos theta, sin theta), offsets v, n_avg lines step apart, by brute force:
    each fine sample x_s (pixels centred on multiples of p, f x f samples each) a point mass h^2 mu l(x_s), l(x_s) =
    2 sqrt(r^2 - v_t^2)/cos(phi) the vertical chord of the infinite tube (v_t across the projected axis); the pixel
    mean of the blurred field at a pixel centre c is sum_s h^2 mu l(x_s) KB(c - x_s), KB the pixel-box integral of the
    truncated kernel (sub x sub midpoint rule) tabulated on the offsets c - x_s, which lie on the half-integer
    multiples of h; the sum is a convolution on the lattice (scipy.signal.fftconvolve); then Block 5's read
    (bilinear_read). beta = 2 members only; the tube is taken infinite (the caller's caps lie beyond the support)."""
    from scipy.signal import fftconvolve
    h = p / f
    yh = np.array([-math.sin(theta), math.cos(theta)])
    eu = np.array([math.cos(theta), math.sin(theta)])
    W = unit_fwhm(2.0, xi, zeta)
    a = FWHM_PER_SD * sigma / W
    rho_c = reach_sigmas * W / FWHM_PER_SD
    R_out = max(zeta, 1.0 / zeta) * rho_c * a + p          # every offset that can carry weight (um)
    K = int(math.ceil(R_out / h)) + 1
    k = np.arange(-K, K)                                     # offsets d_k = (k + 1/2) h
    d = (k + 0.5) * h
    DX, DY = np.meshgrid(d, d, indexing="xy")
    t = (np.arange(sub) + 0.5) / sub * p - 0.5 * p
    KB = np.zeros(DX.shape)
    for ox in t:
        for oy in t:
            KB += kernel_2d(DX + ox, DY + oy, a, xi, zeta, rho_c, yh)
    KB /= sub * sub
    # the pixels the read touches (with a margin) and the sources that reach them
    xs, ys = [], []
    for kk in range(-n_avg, n_avg + 1):
        xs.append(np.asarray(v) * yh[0] + kk * step * eu[0])
        ys.append(np.asarray(v) * yh[1] + kk * step * eu[1])
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    c0, c1 = int(math.floor(xs.min() / p)) - 1, int(math.ceil(xs.max() / p)) + 1
    r0, r1 = int(math.floor(ys.min() / p)) - 1, int(math.ceil(ys.max() / p)) + 1
    # sample n sits at (n + 1/2) h - p/2; pixel q's centre at q p = q f h; c - x_s = (q f - n + f/2 - 1 + 1/2) h
    n0x, n1x = c0 * f - K - f, c1 * f + K + f
    n0y, n1y = r0 * f - K - f, r1 * f + K + f
    gx = (np.arange(n0x, n1x + 1) + 0.5) * h - 0.5 * p
    gy = (np.arange(n0y, n1y + 1) + 0.5) * h - 0.5 * p
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    vt = -(X - c_xy[0]) * math.sin(theta) + (Y - c_xy[1]) * math.cos(theta)
    S = h * h * mu * 2.0 * np.sqrt(np.clip(r * r - vt * vt, 0.0, None)) / math.cos(phi)
    full = fftconvolve(S, KB, mode="full")
    # full[l] = sum_n S[n] KB[l - (n - n0) ], KB index j <-> k = j - K; need k = q f - n + f/2 - 1:
    # l = q f + f/2 - 1 + K - n0
    qx = np.arange(c0, c1 + 1)
    qy = np.arange(r0, r1 + 1)
    lx = qx * f + f // 2 - 1 + K - n0x
    ly = qy * f + f // 2 - 1 + K - n0y
    plane = full[np.ix_(ly, lx)]
    deficit = bilinear_read(plane, c0, r0, p, np.array([0.0, 0.0]), yh, eu, v, n_avg, step)
    return 1.0 - deficit
