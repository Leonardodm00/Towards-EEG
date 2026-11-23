"""The SGT point-spread function in the fast line renderer -- Block 49 in specs/SPEC.md (wave 7; D-221, D-222, D-224,
D-225).

The quantity is Block 35's (model/line_fast.py): for a capped straight tube, mu, plane indices ks and Block 5's
measuring line, the value tau_k(v_n) that profiles.sample_profile reads from the pixel means of plane k,

    tau_k = 1 - read( pixel means of sum_j Delta A_j * K_jk ),

with another kernel K_jk in place of Block 4's Gaussian. Slab j (centre zeta_j) and plane k (depth z_k = k dz):
delta_jk = zeta_j - z_k; the law's SD sigma_jk = sigma_r(delta_jk) (Block 4, the configured table); the SGT's FWHM
w_jk = 2 sqrt(2 ln 2) sigma_jk (D-222.2); the shape kappa = (beta, xi, zeta) (D-052.1); the core scale
a_jk = w_jk / W(kappa), W of (H3) (blur_kernels.sgt_unit_fwhm). h is the symmetric core density of (H1) at unit
scale (blur_kernels.sgt_unit_pdf(x, beta, xi, 1)), hhat(t) = int h(x) exp(-2 pi i t x) dx its characteristic
function (real, even).

- zeta = 1 (D-222.1, D-066.1): K_jk is the isotropic PSF whose line projection is g_{w_jk,(beta,xi,1)}; its 2-D
  Fourier transform is hhat(a_jk |nu|) (projection-slice theorem). P_jk(r) denotes its radial profile.
- zeta != 1 (inference.sgt_skew "fit"; D-225.3): with u = x . y_hat (u > 0 towards increasing v) and s = x . e_u,
      K_jk(x) = (2 / (zeta + 1/zeta)) P_jk(sqrt((u / zeta)^2 + s^2))     (u >= 0)
      K_jk(x) = (2 / (zeta + 1/zeta)) P_jk(sqrt((zeta u)^2 + s^2))       (u < 0),
  whose line projection across y_hat, int K_jk(u y_hat + s e_u) ds, is the two-piece SGT g_{w_jk,kappa}(u) of
  (H2)-(H3); zeta = 1 gives the isotropic kernel. s is taken along the unit normal of y_hat, (y_hat_y, -y_hat_x):
  that is +-e_u whenever e_u is perpendicular to y_hat, as on every line of Block 5 (y_hat = (-sin theta, cos theta),
  e_u = (cos theta, sin theta)), and K depends on s^2 only; for an oblique e_u it keeps zeta = 1 isotropic.
- Truncation (D-225.4): the kernel is applied within R_jk = inference.sgt_reach_sigmas x sigma_jk of its centre
  in the radial variable of P_jk (before the stretch), and its mass beyond is dropped, not renormalised. R_jk / a_jk
  = sgt_reach_sigmas x W(kappa) / (2 sqrt(2 ln 2)) depends on kappa only, so the truncated kernel scales with the
  width and tau is smooth in kappa. The dropped fraction is the isotropic kernel's 2-D mass beyond that radius (the
  stretch keeps the fraction): truncated_mass.
- Samples are drawn as Block 35 draws them (line_fast._render_setup: those within max_k R_jk of the pixel box; a pair
  skipped when the slab's nearest drawn sample is farther than R_jk), with Z R_jk in place of line_reach_sigmas x
  sigma_jk: Z = 1 for zeta = 1, Z = max(STRETCH_ALLOWANCE, zeta, 1/zeta) otherwise. The stretched support reaches
  max(zeta, 1/zeta) R_jk from the centre and is drawn whole, so the dropped fraction is truncated_mass as stated; a
  superset drawn changes nothing, the kernel being 0 beyond its support. Within a pair's level, samples farther
  than Z R_jk from the read's strip (the rectangle holding every pixel the bilinear read touches) are left out: they
  reach no pixel that is read (the other pixels of the box are not read).
- Slabs, absorption, the hybrid's cross-section, the pixel means (the mean over the pixel square of the blurred
  field, each fine-lattice sample a point mass h^2 Delta A_j) and the read are Block 35's (line_fast._render_setup,
  line_fast._slab_maps, line_fast._read: one code with the Gaussian path).

Numerical scheme (the implementer's, within the entry's oracles).

(1) The unit kernel, per shape (cached in the call's SectionCache when one is passed). In core units (a = 1) the
    truncated isotropic kernel P_1(rho) 1[rho <= rho_c], rho_c = R / a, has the line projection
        A_tr(x) = h(x) - A_tail(x)  (|x| < rho_c),  0 beyond,
        A_tail(x) = -(1/pi) int_{rho_c}^inf h'(y) arccos((2 rho_c^2 - x^2 - y^2) / (y^2 - x^2)) dy,
    the projection of the dropped tail written with h' alone (the Abel transform of the tail, its inner integral in
    closed form). A_tail is evaluated by a fixed Gauss-Legendre rule in s (y = rho_c cosh s) at TABLE_THETA nodes
    x = rho_c cos(theta) (smooth in theta) and interpolated (cubic spline in theta); A_tr is sampled on the fixed
    lattice x = n TABLE_DX; its transfer H(t) (= the truncated kernel's 2-D transfer at |nu| = t, projection-slice)
    and H'(t) follow by one real FFT each on the fixed grid t = k TABLE_DT, t <= TABLE_T_MAX, and are read by cubic
    Hermite interpolation (C1, fourth order). Every grid is independent of the shape, so tau is smooth in it.
    The 2-D tail mass is -2 int_r^inf h'(x) sqrt(x^2 - r^2) dx (the same Abel identity; scipy.integrate.quad).
(2) Each (slab j, plane k) pair is convolved in the Fourier domain on a lattice of its level. Level 0 and 1: the
    fine lattice itself (h = p / line_fine_factor); level l >= 2: spacing H = 2^(l-1) h; a pair's level is fixed by
    its FWHM w_jk alone (w_jk in [w_A 2^(l-1), w_A 2^l), w_A = OVERSAMPLE x sgt_samples_per_fwhm x h; level 0 below
    w_A), so its lattice holds at least OVERSAMPLE x sgt_samples_per_fwhm samples per FWHM. Level 0 is the fine
    lattice, the finest there is (the samples live on it): a pair narrower than sgt_samples_per_fwhm x h has fewer
    samples per FWHM there (the in-focus pairs at line_fine_factor 8: about 13). The pair's transfer is applied on
    |nu_x|, |nu_y| <= sgt_samples_per_fwhm / (2 w_jk) (its band at sgt_samples_per_fwhm samples per FWHM), clipped
    to the lattice's Nyquist frequency on the fine lattice (to a quarter of the coarse lattice's sampling rate on a
    spread level); beyond, 0.
(3) A slab's samples enter a level >= 2 by Gaussian spreading (SD SPREAD_TAU coarse cells, cut at SPREAD_HALF
    cells; separable) divided out in the Fourier domain: with the band at most a quarter of the coarse lattice's,
    the aliases are below exp(-pi^2 SPREAD_TAU^2) (NUFFT type 1, Greengard and Lee 2004). Levels 0 and 1 take the
    samples as they are (exact).
(4) A level's period is at least (the farthest source-to-box distance along each axis) + the level's largest
    support + a margin, so no periodic image reaches the box; the transform of each slab is computed on its
    footprint only (a real FFT along x of its rows, then an FFT along y of the kept columns) and kept on the level's
    band; the deficit at the pixel centres is the inverse transform evaluated there (separable products).
(5) Transfers, per distinct sigma (cached within the call; the slabs taken in an order that reuses them,
    _slab_order): zeta = 1, H(a |nu|); zeta != 1, the even part
    (c/2) [zeta H(a |(zeta nu_u, nu_s)|) + H(a |(nu_u/zeta, nu_s)|)/zeta] exactly, and the odd part
    (c/2) FT[sign(u) (E_zeta - E_1/zeta)], E the stretched isotropic kernels, computed on each level at
    CHEB_NODES Chebyshev nodes in sigma (the band-limited difference brought to real space, multiplied by sign(u)
    on a lattice whose period holds its support, transformed back) and interpolated in sigma (barycentric). All
    times sinc(p nu_x) sinc(p nu_y) (the pixel box, exact) and, on spread levels, divided by the window's transfer.

Custom: the whole scheme above. No library convolves a depth-dependent, non-separable, non-Gaussian kernel with
pixel means on chosen pixels (checked: scipy.signal.fftconvolve and oaconvolve need one kernel per call on one
lattice; scipy.ndimage filters are separable or sampled; FINUFFT/pynufft are not in the cluster's environment).
Library calls: numpy, scipy.fft (rfft, fft, irfft2, rfft2, next_fast_len), scipy.special (via blur_kernels),
scipy.integrate.quad, scipy.interpolate.CubicSpline.

Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy import fft as sfft
from scipy import integrate, interpolate

from . import blur_kernels

FWHM_PER_SD = 2.0 * math.sqrt(2.0 * math.log(2.0))     # a Gaussian's FWHM over its SD (D-222.2)

# -- the unit kernel's tables (core units; fixed grids, so the model is smooth in the shape) --
TABLE_DX = 1.0 / 2048.0      # projection lattice (core units): the cusp of beta < 2 aliases below 1e-8 (Laplace)
TABLE_DT = 0.004             # transfer grid (core frequency units); cubic Hermite error about 1e-9
TABLE_T_MAX = 400.0          # the table ends here; beyond, the transfer is taken as 0
TABLE_THETA = 256            # nodes x = rho_c cos(theta) of the tail's projection (cubic spline in theta)
TABLE_S_NODES = 400          # Gauss-Legendre nodes in s, y = rho_c cosh(s), s in [0, TABLE_S_MAX]
TABLE_S_MAX = 60.0           # exp(-60 (beta/xi + 1)) of the tail integrand left out
# -- the lattices --
OVERSAMPLE = 2               # a spread level's lattice holds OVERSAMPLE x sgt_samples_per_fwhm samples per FWHM
SPREAD_TAU = 1.5             # the spreading Gaussian's SD (coarse cells): aliases below exp(-pi^2 tau^2) = 2e-10
SPREAD_HALF = 10             # the spreading window's half-width (coarse cells): mass beyond 2.6e-11
MARGIN_FWHM = 2.0            # each level's period exceeds the reach by this many of its largest FWHM
STRETCH_ALLOWANCE = 2.1      # zeta != 1: supports and lattices sized for max(zeta, 1/zeta) up to this (> e^0.7)
CHEB_NODES = 12              # Chebyshev nodes in sigma per level for the odd (skew) part of the transfer
TRANSFER_CACHE_MB = 256.0    # transfers kept within one call (beyond, recomputed)


@dataclasses.dataclass(frozen=True)
class SGTKernel:
    """The SGT shape kappa = (beta, xi, zeta) of Block 49's kernel (D-052.1); ValueError outside the family's domain
    (blur_kernels.check_sgt_shape: beta > 0, xi >= 0, zeta > 0, all finite)."""

    beta: float
    xi: float
    zeta: float = 1.0

    def __post_init__(self):
        b, x, z = blur_kernels.check_sgt_shape(self.beta, self.xi, self.zeta)
        object.__setattr__(self, "beta", b)
        object.__setattr__(self, "xi", x)
        object.__setattr__(self, "zeta", z)


# ------------------------------------------------------------- the core ---

def _log_neg_dh(logy, beta, xi):
    """log(-h'(y)) at y = exp(logy) > 0, h the unit core of (H1) (array). -h' = c b y^(b-1) exp(-y^b) (xi = 0) or
    c (b + xi) y^(b-1) (1 + xi y^b)^-(1/xi + 1/b + 1); evaluated in logs (no overflow at large y)."""
    logy = np.asarray(logy, dtype=float)
    lc = blur_kernels._sgt_log_c(beta, xi)
    if xi <= blur_kernels.SGT_XI_ZERO:
        with np.errstate(over="ignore"):
            yb = np.exp(np.minimum(beta * logy, 700.0))
        return lc + math.log(beta) + (beta - 1.0) * logy - yb
    lq = np.logaddexp(0.0, math.log(xi) + beta * logy)          # log(1 + xi y^b)
    return lc + math.log(beta + xi) + (beta - 1.0) * logy - (1.0 / xi + 1.0 / beta + 1.0) * lq


def _log_cosh_sinh(s):
    """(log cosh s, log sinh s) for s > 0 (arrays), without overflow."""
    e = np.exp(-2.0 * s)
    return s + np.log1p(e) - math.log(2.0), s + np.log1p(-e) - math.log(2.0)


def sgt_tail_mass(beta, xi, r_over_a):
    """The isotropic unit kernel's 2-D mass beyond radius r (core-scale units, r_over_a >= 0):
    1 - int_0^r P_1(rho) 2 pi rho drho, P_1 the radial profile of the isotropic PSF whose line projection is h of
    (H1) (zeta = 1). -> float in [0, 1]. Closed forms for the oracles: the Gaussian member (2, 0) gives exp(-r^2);
    the Laplace member (1, 0) gives r K_1(r).

    Computed as -2 int_r^inf h'(x) sqrt(x^2 - r^2) dx (the Abel identity: the 2-D mass outside the disc of radius r
    from the line projection h alone), x = r cosh s, by scipy.integrate.quad. ValueError for a shape outside the
    family's domain or r not finite and >= 0."""
    beta, xi, _ = blur_kernels.check_sgt_shape(beta, xi, 1.0)
    r = float(r_over_a)
    if not (math.isfinite(r) and r >= 0.0):
        raise ValueError("r_over_a must be finite and >= 0, got %r" % (r_over_a,))
    if r == 0.0:
        return 1.0
    lr = math.log(r)

    def f(s):
        if s <= 0.0:
            return 0.0
        lc, ls = _log_cosh_sinh(np.array([s]))
        return 2.0 * math.exp(2.0 * lr + float(_log_neg_dh(lr + lc, beta, xi)[0]) + 2.0 * float(ls[0]))

    val = integrate.quad(f, 0.0, np.inf, epsabs=1e-16, epsrel=1e-13, limit=500)[0]
    return float(min(1.0, max(0.0, val)))


def truncated_mass(kernel, cfg):
    """The fraction of every pair's kernel dropped by the truncation (D-225.4): sgt_tail_mass(beta, xi, R / a) with
    R / a = cfg.inference.sgt_reach_sigmas x W(kappa) / FWHM_PER_SD. kernel an SGTKernel; cfg a DiameterConfig."""
    if not isinstance(kernel, SGTKernel):
        raise ValueError("kernel must be a line_sgt.SGTKernel, got %r" % (kernel,))
    return sgt_tail_mass(kernel.beta, kernel.xi, _rho_c(kernel, _check_reach(cfg)))


def _check_reach(cfg):
    reach = float(cfg.inference.sgt_reach_sigmas)
    if not (math.isfinite(reach) and reach > 0.0):
        raise ValueError("inference.sgt_reach_sigmas must be finite and > 0")
    return reach


def _check_spf(cfg):
    spf = cfg.inference.sgt_samples_per_fwhm
    if int(spf) != spf or int(spf) < 1:
        raise ValueError("inference.sgt_samples_per_fwhm must be an integer >= 1")
    return int(spf)


def _rho_c(kernel, reach):
    """R / a in core units: reach x W(kappa) / FWHM_PER_SD."""
    return reach * blur_kernels.sgt_unit_fwhm(kernel.beta, kernel.xi, kernel.zeta) / FWHM_PER_SD


# ------------------------------------------------------- the unit table ---

class _UnitTable(object):
    """H(t), H'(t) of the truncated isotropic unit kernel on t = k TABLE_DT, k = 0 .. n - 1; H(t) at t >= 0 by cubic
    Hermite interpolation (0 from the table's last node on). The cubic of interval k, written in s = t / TABLE_DT - k
    in [0, 1), is c0 + c1 s + c2 s^2 + c3 s^3 with c0 = H_k, c1 = D_k, c2 = 3 (H_k+1 - H_k) - 2 D_k - D_k+1,
    c3 = 2 (H_k - H_k+1) + D_k + D_k+1, D = TABLE_DT H' (its coefficients kept per interval, a zero row last)."""

    def __init__(self, H, dH):
        self.n = H.size
        D = TABLE_DT * dH
        zero = np.zeros(1)
        self.c0 = np.concatenate((H[:-1], zero))
        self.c1 = np.concatenate((D[:-1], zero))
        self.c2 = np.concatenate((-3.0 * H[:-1] - 2.0 * D[:-1] + 3.0 * H[1:] - D[1:], zero))
        self.c3 = np.concatenate((2.0 * H[:-1] + D[:-1] - 2.0 * H[1:] + D[1:], zero))

    def __call__(self, t):
        u = np.asarray(t, dtype=float) * (1.0 / TABLE_DT)
        k = np.minimum(u, self.n - 1).astype(np.intp)    # t >= 0: the cast is the floor; the last row is zero
        s = u - k
        out = self.c3.take(k) * s
        out += self.c2.take(k)
        out *= s
        out += self.c1.take(k)
        out *= s
        out += self.c0.take(k)
        return out


def _a_tail(x, beta, xi, rho_c):
    """The tail's line projection A_tail(x) at 0 <= x < rho_c (array), module docstring (1)."""
    sn, wn = np.polynomial.legendre.leggauss(TABLE_S_NODES)
    s = 0.5 * TABLE_S_MAX * (sn + 1.0)
    w = 0.5 * TABLE_S_MAX * wn
    lc, ls = _log_cosh_sinh(s)
    y = rho_c * np.exp(lc)
    ndh_dy = np.exp(_log_neg_dh(math.log(rho_c) + lc, beta, xi) + math.log(rho_c) + ls) * w    # -h'(y) dy
    x = np.asarray(x, dtype=float)[:, None]
    yy = y[None, :]
    arg = (2.0 * rho_c * rho_c - x * x - yy * yy) / (yy * yy - x * x)
    return (1.0 / math.pi) * np.sum(ndh_dy[None, :] * np.arccos(np.clip(arg, -1.0, 1.0)), axis=1)


def _build_table(beta, xi, rho_c):
    """_UnitTable of the truncated isotropic unit kernel (module docstring (1))."""
    n_x = int(math.floor(rho_c / TABLE_DX))
    x = np.arange(n_x + 1) * TABLE_DX
    theta = np.linspace(0.0, 0.5 * math.pi, TABLE_THETA + 1)
    tail = interpolate.CubicSpline(theta, _a_tail(rho_c * np.cos(theta), beta, xi, rho_c))
    a_tr = blur_kernels.sgt_unit_pdf(x, beta, xi, 1.0) - tail(np.arccos(np.clip(x / rho_c, 0.0, 1.0)))
    a_tr[x >= rho_c] = 0.0
    P0 = int(round(1.0 / (TABLE_DX * TABLE_DT)))
    m = max(1, int(math.ceil((2 * n_x + 2) / P0)))      # a long projection (large reach): a period m times longer,
    P = m * P0                                           # every m-th transfer kept (the t grid stays TABLE_DT)
    n_t = int(round(TABLE_T_MAX / TABLE_DT)) + 1
    even = np.zeros(P)
    even[:n_x + 1] = a_tr
    even[P - n_x:] = a_tr[1:][::-1]
    H = TABLE_DX * sfft.rfft(even, workers=1).real[:m * (n_t - 1) + 1:m]
    odd = np.zeros(P)
    odd[:n_x + 1] = x * a_tr
    odd[P - n_x:] = -(x * a_tr)[1:][::-1]
    dH = 2.0 * math.pi * TABLE_DX * sfft.rfft(odd, workers=1).imag[:m * (n_t - 1) + 1:m]
    return _UnitTable(np.ascontiguousarray(H), np.ascontiguousarray(dH))


def _table_for(kernel, rho_c, cache):
    if cache is None:
        return _build_table(kernel.beta, kernel.xi, rho_c)
    key = ("sgt_unit_table", kernel.beta, kernel.xi, float(rho_c), TABLE_DX, TABLE_DT, TABLE_T_MAX, TABLE_THETA,
           TABLE_S_NODES, TABLE_S_MAX)
    return cache.get(key, lambda: _build_table(kernel.beta, kernel.xi, rho_c))


# ----------------------------------------------------------- the levels ---

class _Level(object):
    """One lattice of the scheme (module docstring (2)-(4)): spacing H = M h, period (Nx, Ny), the reference
    lattice point (local index 0) at (x_ref, y_ref) um, the band |kx| <= nx, |ky| <= ny kept, the pixel box's and
    the spreading window's transfer C on it, the output matrices."""


def _cheb_nodes(lo, hi, n):
    """Chebyshev points of the second kind on [lo, hi] and their barycentric weights."""
    k = np.arange(n)
    x = 0.5 * (lo + hi) + 0.5 * (hi - lo) * np.cos(np.pi * k / (n - 1))
    w = (-1.0) ** k
    w[0] *= 0.5
    w[-1] *= 0.5
    return x, w


def _bary(xq, xn, wn):
    """Barycentric Lagrange weights (n,) at the scalar xq."""
    d = xq - xn
    hit = np.flatnonzero(d == 0.0)
    if hit.size:
        out = np.zeros(xn.size)
        out[hit[0]] = 1.0
        return out
    c = wn / d
    return c / c.sum()


def _spread_matrix(i0, n, M, H, f, h, p, q_lo, q_hi):
    """(q_hi - q_lo + 1, n): the Gaussian window psi((x_c - Q H) / H) for the fine samples x_c = (i0 + c + 1/2) h - p/2,
    c = 0 .. n - 1, and the coarse points Q = q_lo .. q_hi (H = M h), cut at SPREAD_HALF. Entry (Q, c) depends on
    m = c - M Q alone (x_c / H - Q = (i0 + 1/2 - f/2 + m) / M): the window is evaluated once on the range of m and
    gathered (a banded Toeplitz pattern)."""
    m_lo = -M * q_hi
    m_hi = (n - 1) - M * q_lo
    m = np.arange(m_lo, m_hi + 1)
    u = (i0 + 0.5 - 0.5 * f + m) / M
    vals = np.where(np.abs(u) <= SPREAD_HALF, np.exp(-0.5 * (u / SPREAD_TAU) ** 2), 0.0)
    idx = np.arange(n)[None, :] - M * np.arange(q_lo, q_hi + 1)[:, None] - m_lo
    return vals[idx]


def _psi_hat(fH):
    """The (untruncated) window's transfer at frequency f in coarse units: sqrt(2 pi) tau exp(-2 pi^2 tau^2 f^2)."""
    return math.sqrt(2.0 * math.pi) * SPREAD_TAU * np.exp(-2.0 * (math.pi * SPREAD_TAU * fH) ** 2)


def _forward(g, r0, c0, lev):
    """The DFT of g (rows r0 + r, cols c0 + c, local lattice indices) on the level's band: (2 ny + 1, nx + 1)
    complex, rows ky = -ny .. ny, cols kx = 0 .. nx."""
    Nx, Ny, nx, ny = lev.Nx, lev.Ny, lev.nx, lev.ny
    nr, nc = g.shape
    if nc > Nx:                                         # fold onto the period (does not occur: the period holds it)
        pad = (-nc) % Nx
        g = np.pad(g, ((0, 0), (0, pad))).reshape(nr, -1, Nx).sum(axis=1)
        nc = Nx
    A = sfft.rfft(g, n=Nx, axis=1)[:, :nx + 1]
    A *= np.exp(-2j * math.pi * lev.kx * (c0 / Nx))[None, :]
    if nr > Ny:
        pad = (-nr) % Ny
        A = np.pad(A, ((0, pad), (0, 0))).reshape(-1, Ny, nx + 1).sum(axis=0)
        nr = Ny
    if (2 * ny + 1) * nr <= 3.0 * Ny * math.log2(max(Ny, 2)):
        B = _stage2_matrix(lev, nr) @ A
    else:
        F = sfft.fft(A, n=Ny, axis=0)
        B = np.concatenate((F[Ny - ny:], F[:ny + 1]), axis=0)
    B *= np.exp(-2j * math.pi * lev.ky * (r0 / Ny))[:, None]
    return B


def _stage2_matrix(lev, nr):
    """exp(-2 pi i ky r / Ny) for ky = -ny .. ny, r = 0 .. nr - 1 (kept per level, grown as needed)."""
    E = lev.E2
    if E is None or E.shape[1] < nr:
        n = max(nr, 2 * (0 if E is None else E.shape[1]))
        E = lev.E2 = np.exp(-2j * math.pi * np.outer(lev.ky, np.arange(n)) / lev.Ny)
    return E[:, :nr]


def _make_level(l, M, st, pairs_sig, reach, allow, spf, skew_sig_hi=None):
    """_Level l from the sigmas of its pairs (pairs_sig, um) (module docstring (2)-(4)); skew_sig_hi (um, or None):
    the period also holds whole the stretched support of sigma up to this (2 x allow x reach x sigma + the margin),
    the top Chebyshev node of the odd part (5)."""
    lev = _Level()
    h, p = st.h, st.p
    H = M * h
    w_min, w_max = FWHM_PER_SD * float(pairs_sig.min()), FWHM_PER_SD * float(pairs_sig.max())
    Zr = allow * reach * float(pairs_sig.max())          # the largest support's reach (um)
    margin = MARGIN_FWHM * w_max + 4.0 * H
    sizes = []
    for b_lo, b_hi, P in ((st.bx0, st.bx1, st.X), (st.by0, st.by1, st.Y)):
        s_lo = max(float(P.min()), b_lo - Zr)
        s_hi = min(float(P.max()), b_hi + Zr)
        L = max(s_hi - b_lo, b_hi - s_lo, b_hi - b_lo) + Zr + margin
        if skew_sig_hi is not None:
            L = max(L, 2.0 * allow * reach * skew_sig_hi + MARGIN_FWHM * FWHM_PER_SD * skew_sig_hi + 4.0 * H)
        sizes.append(int(math.ceil(L / H)))
    lev.l, lev.M, lev.H = l, M, H
    lev.Nx = sfft.next_fast_len(sizes[0], real=True)
    lev.Ny = sfft.next_fast_len(sizes[1], real=False)
    nu_c = min(spf / (2.0 * w_min), 0.5 / H if M == 1 else 0.25 / H)
    lev.nx = min(int(math.floor(nu_c * lev.Nx * H)), (lev.Nx - 1) // 2)
    lev.ny = min(int(math.floor(nu_c * lev.Ny * H)), (lev.Ny - 1) // 2)
    lev.kx = np.arange(lev.nx + 1)
    lev.ky = np.arange(-lev.ny, lev.ny + 1)
    lev.nu_x = lev.kx / (lev.Nx * H)
    lev.nu_y = lev.ky / (lev.Ny * H)
    # the reference lattice point (local index 0), near the box's lower corner
    if M == 1:
        lev.qx_ref = int(math.floor((st.bx0 + 0.5 * p) / h - 0.5))
        lev.qy_ref = int(math.floor((st.by0 + 0.5 * p) / h - 0.5))
        lev.x_ref = (lev.qx_ref + 0.5) * h - 0.5 * p
        lev.y_ref = (lev.qy_ref + 0.5) * h - 0.5 * p
    else:
        lev.qx_ref = int(math.floor(st.bx0 / H))
        lev.qy_ref = int(math.floor(st.by0 / H))
        lev.x_ref, lev.y_ref = lev.qx_ref * H, lev.qy_ref * H
    # the pixel box (exact) and, on spread levels, the window's transfer divided out; the 1 / M^2 of the masses
    C = np.sinc(p * lev.nu_y)[:, None] * np.sinc(p * lev.nu_x)[None, :]
    if M > 1:
        C = C / (M * M * _psi_hat(H * lev.nu_y)[:, None] * _psi_hat(H * lev.nu_x)[None, :])
    lev.C = C
    lev.nu = np.hypot(lev.nu_y[:, None], lev.nu_x[None, :])
    # the output at the pixel centres x = c p, y = r p (separable inverse transform, Hermitian half in x)
    xc = np.arange(st.c_lo, st.c_hi + 1) * p - lev.x_ref
    yr = np.arange(st.r_lo, st.r_hi + 1) * p - lev.y_ref
    lev.Ex = np.exp(2j * math.pi * np.outer(xc, lev.nu_x)) * np.where(lev.kx == 0, 1.0, 2.0)[None, :]
    lev.Ey = np.exp(2j * math.pi * np.outer(yr, lev.nu_y))
    lev.norm = 1.0 / (lev.Nx * lev.Ny)
    lev.acc = {}
    lev.odd = None
    lev.E2 = None
    return lev


def _source(st, lev, r_start, c_start, dA):
    """(g, r0, c0): a slab's samples dA (first sample at row r_start, col c_start of the kept lattice) on the level's
    lattice in local indices (module docstring (3))."""
    iy = st.iy0 + r_start
    ix = st.ix0 + c_start
    if lev.M == 1:
        return dA, iy - lev.qy_ref, ix - lev.qx_ref
    h, p, H, M, f = st.h, st.p, lev.H, lev.M, st.f
    nr, nc = dA.shape
    # coarse range: every Q within SPREAD_HALF cells of a sample (x / H = (i + 1/2 - f/2) / M)
    qx_lo = int(math.floor((ix + 0.5 - 0.5 * f) / M)) - SPREAD_HALF
    qx_hi = int(math.ceil((ix + nc - 0.5 - 0.5 * f) / M)) + SPREAD_HALF
    qy_lo = int(math.floor((iy + 0.5 - 0.5 * f) / M)) - SPREAD_HALF
    qy_hi = int(math.ceil((iy + nr - 0.5 - 0.5 * f) / M)) + SPREAD_HALF
    Wx = _spread_matrix(ix, nc, M, H, f, h, p, qx_lo, qx_hi)
    Wy = _spread_matrix(iy, nr, M, H, f, h, p, qy_lo, qy_hi)
    g = Wy @ (dA @ Wx.T)
    return g, qy_lo - lev.qy_ref, qx_lo - lev.qx_ref


# ------------------------------------------------------- the transfers ---

class _Frame(object):
    """The kernel's frame on the image: u = x . yh (measuring axis), s = x . sh, sh the unit normal of yh."""

    def __init__(self, yh):
        self.yh = np.asarray(yh, dtype=float)
        self.sh = np.array([self.yh[1], -self.yh[0]])


def _even(table, a, nu_u, nu_s, zeta):
    """The skewed kernel's even part (c/2) [zeta H(a |(zeta nu_u, nu_s)|) + H(a |(nu_u/zeta, nu_s)|) / zeta]."""
    c = 2.0 / (zeta + 1.0 / zeta)
    return 0.5 * c * (zeta * table(a * np.hypot(zeta * nu_u, nu_s)) + table(a * np.hypot(nu_u / zeta, nu_s)) / zeta)


def _odd_lattice(Nx, Ny, H, frame, zeta):
    """The lattice of the odd part's construction (spacing H, period (Nx H, Ny H)): the stretched radii
    |(zeta nu_u, nu_s)| and |(nu_u / zeta, nu_s)| on its half spectrum (Ny, Nx // 2 + 1) and sign(u) at its points
    (the kernel's centre at index 0, minimum-image positions) -> (r_plus, r_minus, sgn)."""
    fx = np.arange(Nx // 2 + 1) / (Nx * H)
    fy = sfft.fftfreq(Ny, H)
    nu_u = fx[None, :] * frame.yh[0] + fy[:, None] * frame.yh[1]
    nu_s = fx[None, :] * frame.sh[0] + fy[:, None] * frame.sh[1]
    r_plus = np.hypot(zeta * nu_u, nu_s)
    r_minus = np.hypot(nu_u / zeta, nu_s)
    del nu_u, nu_s
    qx = np.arange(Nx)
    qy = np.arange(Ny)
    xq = np.where(qx < (Nx + 1) // 2, qx, qx - Nx) * H
    yq = np.where(qy < (Ny + 1) // 2, qy, qy - Ny) * H
    sgn = np.sign(xq[None, :] * frame.yh[0] + yq[:, None] * frame.yh[1])
    return r_plus, r_minus, sgn


def _odd_on(table, zeta, a, Nx, Ny, lattice):
    """The odd part of the skewed kernel's transfer at core scale a (um), (c/2) FT[sign(u) (E_zeta - E_1/zeta)], on
    the half spectrum (Ny, Nx // 2 + 1) of an _odd_lattice whose period holds the support (module docstring (5)): the
    difference's transfer zeta H(a |(zeta nu_u, nu_s)|) - H(a |(nu_u/zeta, nu_s)|) / zeta on the lattice's whole
    band, brought to real space, multiplied by sign(u), transformed back (the real-space scale 1/H^2 and the H^2 of
    the transform cancel). Purely imaginary up to rounding."""
    r_plus, r_minus, sgn = lattice
    c = 2.0 / (zeta + 1.0 / zeta)
    d = sfft.irfft2(zeta * table(a * r_plus) - table(a * r_minus) / zeta, s=(Ny, Nx), workers=1)
    d *= sgn
    return 0.5 * c * sfft.rfft2(d, workers=1)


def _odd_transfer(table, zeta, a, Nx, Ny, H, frame):
    """_odd_on on its own lattice (spacing H, period (Nx H, Ny H)): (Ny, Nx // 2 + 1) complex. The renderer's
    construction (_odd_nodes), exposed for the smoke test's check of the transfer along y_hat."""
    return _odd_on(table, zeta, a, Nx, Ny, _odd_lattice(Nx, Ny, H, frame, zeta))


def _odd_nodes(lev, st, table, kern, W, reach, allow, frame, lo, hi):
    """The odd part on the level's band at CHEB_NODES sigmas in [lo, hi] (module docstring (5)), each by _odd_on on
    the level's own lattice (its period holds the support when skewed: _make_level):
    -> (nodes, barycentric weights, (2 ny + 1, nx + 1, n_nodes) complex: the nodes last, so that a pair's
    interpolation is one matrix-vector product per frequency)."""
    zeta = kern.zeta
    nodes, wts = _cheb_nodes(lo, hi, CHEB_NODES)
    Nx, Ny, H = lev.Nx, lev.Ny, lev.H
    lattice = _odd_lattice(Nx, Ny, H, frame, zeta)
    rows = lev.ky % Ny
    out = np.empty((lev.ky.size, lev.kx.size, CHEB_NODES), dtype=complex)
    for m, sg in enumerate(nodes):
        need = 2.0 * allow * reach * sg + MARGIN_FWHM * FWHM_PER_SD * sg
        if need > min(Nx, Ny) * H * (1.0 + 1e-12):
            raise RuntimeError("the level's period does not hold the odd part's support")   # internal consistency
        out[:, :, m] = _odd_on(table, zeta, FWHM_PER_SD * sg / W, Nx, Ny, lattice)[np.ix_(rows, lev.kx)]
    return nodes, wts, out


class _Transfers(object):
    """Per-sigma transfers on each pair's band, kept within the call up to TRANSFER_CACHE_MB."""

    def __init__(self):
        from collections import OrderedDict
        self.store = OrderedDict()
        self.bytes = 0

    def get(self, key, build):
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        val = build()
        self.store[key] = val
        self.bytes += val.nbytes
        while self.bytes > TRANSFER_CACHE_MB * 1e6 and len(self.store) > 1:
            _k, old = self.store.popitem(last=False)
            self.bytes -= old.nbytes
        return val


def _transfer(lev, sg, nyq, nxq, table, kern, W, frame):
    """The pair's transfer times C on its band rows ny - nyq .. ny + nyq, cols 0 .. nxq."""
    rows = slice(lev.ny - nyq, lev.ny + nyq + 1)
    cols = slice(0, nxq + 1)
    a = FWHM_PER_SD * sg / W
    if kern.zeta == 1.0:
        half = table(a * lev.nu[lev.ny:lev.ny + nyq + 1, cols])          # ky = 0 .. nyq; even in ky
        K = np.concatenate((half[:0:-1], half), axis=0)
    else:
        zeta = kern.zeta
        c = 2.0 / (zeta + 1.0 / zeta)
        nodes, wts, O = lev.odd
        L = _bary(sg, nodes, wts)
        K = O[rows, cols, :] @ L
        K += 0.5 * c * (zeta * table(a * lev.r_plus[rows, cols]) + table(a * lev.r_minus[rows, cols]) / zeta)
    return K * lev.C[rows, cols]


def _stretched_radii(lev, frame, zeta):
    """|(zeta nu_u, nu_s)| and |(nu_u / zeta, nu_s)| on the level's band (the skewed kernel's even part)."""
    nu_u = lev.nu_x[None, :] * frame.yh[0] + lev.nu_y[:, None] * frame.yh[1]
    nu_s = lev.nu_x[None, :] * frame.sh[0] + lev.nu_y[:, None] * frame.sh[1]
    lev.r_plus = np.hypot(zeta * nu_u, nu_s)
    lev.r_minus = np.hypot(nu_u / zeta, nu_s)


# --------------------------------------------------------- the renderer ---

def profiles(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache, kernel):
    """tau (n_planes, n_v) of line_fast.line_profiles with the SGT kernel `kernel` (an SGTKernel); the arguments
    already validated by line_fast (absorption a name, cache a line_fast.SectionCache)."""
    from . import line_fast
    if not isinstance(kernel, SGTKernel):
        raise ValueError("kernel must be a line_sgt.SGTKernel, got %r" % (kernel,))
    zeta = kernel.zeta
    allow = 1.0 if zeta == 1.0 else max(STRETCH_ALLOWANCE, zeta, 1.0 / zeta)
    checked = {}

    def extra():
        checked["reach"] = _check_reach(cfg)
        checked["spf"] = _check_spf(cfg)

    st = line_fast._render_setup(tube, mu, ks, origin_xy, y_hat, e_u, v, n_avg, step, cfg, absorption, cache, True,
                                 reach=lambda: checked["reach"] * allow, extra_checks=extra)
    if st.ones:
        return np.ones((st.n_planes, st.n_v))
    return _deficit_read(st, kernel, checked["reach"], checked["spf"], allow, cfg)


def _levels_of(sig, h, spf):
    """Each pair's level from its FWHM alone (module docstring (2)): (J, n_planes) int."""
    w = FWHM_PER_SD * sig
    lev = np.zeros(sig.shape, dtype=np.int64)
    edge, l = OVERSAMPLE * spf * h, 1
    while True:
        m = w >= edge
        if not m.any():
            return lev
        lev[m] = l
        edge *= 2.0
        l += 1


def _read_strip(st):
    """The read's region in the frame (yh, sh) about the origin: the rectangle holding every sample point of the 2 n_avg
    + 1 lines, widened by 2.2 p, so that every pixel the bilinear read touches lies inside it (a touched pixel's
    centre is within p sqrt(2) of a sample point, its square within p / sqrt(2) more): (c1_lo, c1_hi, c2_lo, c2_hi)."""
    from . import line_render
    xs, ys = line_render._sample_points(st.o, st.yh, st.eu, st.v, st.n_avg, st.step)
    sh = np.array([st.yh[1], -st.yh[0]])
    c1 = (xs - st.o[0]) * st.yh[0] + (ys - st.o[1]) * st.yh[1]
    c2 = (xs - st.o[0]) * sh[0] + (ys - st.o[1]) * sh[1]
    m = 2.2 * st.p
    return float(c1.min()) - m, float(c1.max()) + m, float(c2.min()) - m, float(c2.max()) + m


def _strip_distance(st, strip, rs, cs):
    """Distance (um) of the kept lattice's samples [rs, cs] from the read's region (_read_strip)."""
    dx = st.X[rs, cs] - st.o[0]
    dy = st.Y[rs, cs] - st.o[1]
    c1 = dx * st.yh[0] + dy * st.yh[1]
    c2 = dx * st.yh[1] - dy * st.yh[0]
    e1 = np.maximum(np.maximum(strip[0] - c1, c1 - strip[1]), 0.0)
    e2 = np.maximum(np.maximum(strip[2] - c2, c2 - strip[3]), 0.0)
    return np.hypot(e1, e2)


def _slab_order(st):
    """The slabs' order in the loop: when dz is a whole number m of dzeta, slab j + m sees plane k + 1 at the depth
    offset (so with the sigma and the transfer) at which slab j sees plane k; taking the slabs as j, j + m, j + 2 m,
    ... (by j mod m) lets the transfer cache reuse all but one of each slab's transfers. Otherwise increasing j. The
    result does not depend on the order but for rounding."""
    m = st.dz / st.dzeta
    mi = int(round(m))
    J = st.zeta.size
    if mi < 2 or abs(m - mi) > 1e-9 * m:
        return None
    return sorted(range(J), key=lambda j: (j % mi, j))


def _deficit_read(st, kern, reach, spf, allow, cfg):
    """The SGT path after line_fast._render_setup: the levels, the slab loop, the output, the read."""
    from . import line_fast
    W = blur_kernels.sgt_unit_fwhm(kern.beta, kern.xi, kern.zeta)
    table = _table_for(kern, _rho_c(kern, reach), st.cache)
    frame = _Frame(st.yh)
    h, sig = st.h, st.sig
    levmap = _levels_of(sig, h, spf)
    w_A = OVERSAMPLE * spf * h
    sig_floor = min(float(np.min(cfg.renderer.kernel_table_sigma_um)), float(sig.min()))
    levels = {}
    for l in np.unique(levmap):
        l = int(l)
        M = 1 if l <= 1 else 2 ** (l - 1)
        lo = sig_floor if l == 0 else w_A * 2.0 ** (l - 1) / FWHM_PER_SD
        hi = w_A * 2.0 ** l / FWHM_PER_SD
        levels[l] = _make_level(l, M, st, sig[levmap == l], reach, allow, spf, None if kern.zeta == 1.0 else hi)
        levels[l].sig_range = (lo, hi)
    transfers = _Transfers()
    strip = _read_strip(st)
    true_R = reach * sig                                     # the truncation radius of each pair (um)
    for j, zj, rs, cs, dA, dmin in line_fast._slab_maps(st, _slab_order(st)):
        dist = None
        for l, lev in levels.items():
            kk = np.flatnonzero((levmap[j] == l) & (dmin <= st.R_jk[j]))
            if kk.size == 0:
                continue
            if dist is None:
                dist = _strip_distance(st, strip, rs, cs)
            Rl = allow * float(true_R[j, kk].max())
            m = (dist <= Rl) & (dA != 0.0)
            if not m.any():
                continue
            mr = np.flatnonzero(m.any(axis=1))
            mc = np.flatnonzero(m.any(axis=0))
            rr = slice(int(mr[0]), int(mr[-1]) + 1)
            cc = slice(int(mc[0]), int(mc[-1]) + 1)
            g = np.where(m[rr, cc], dA[rr, cc], 0.0)
            g, r0, c0 = _source(st, lev, rs.start + rr.start, cs.start + cc.start, g)
            B = _forward(g, r0, c0, lev)
            if kern.zeta != 1.0 and lev.odd is None:
                lev.odd = _odd_nodes(lev, st, table, kern, W, reach, allow, frame, *lev.sig_range)
                _stretched_radii(lev, frame, kern.zeta)
            for k in kk:
                sg = float(sig[j, k])
                nu_c = spf / (2.0 * FWHM_PER_SD * sg)
                nxq = min(int(math.floor(nu_c * lev.Nx * lev.H)), lev.nx)
                nyq = min(int(math.floor(nu_c * lev.Ny * lev.H)), lev.ny)
                T = transfers.get((l, round(sg, 12)), lambda: _transfer(lev, sg, nyq, nxq, table, kern, W, frame))
                rows = slice(lev.ny - nyq, lev.ny + nyq + 1)
                acc = lev.acc.get(int(k))
                if acc is None:
                    acc = lev.acc[int(k)] = np.zeros((lev.ky.size, lev.kx.size), dtype=complex)
                sub = acc[rows, :nxq + 1]
                sub += B[rows, :nxq + 1] * T
    deficit = np.zeros((st.n_planes, st.n_rows, st.n_cols))
    for lev in levels.values():
        for k, acc in lev.acc.items():
            deficit[k] += lev.norm * (lev.Ey @ (acc @ lev.Ex.T)).real
    return line_fast._read(st, deficit)
