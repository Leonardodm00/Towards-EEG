#!/usr/bin/env python3
"""Standalone check (NOT pipeline code), 2026-11-23 (D-221, D-222): the 1-D line model of a straight tube
invariant along its axis, with the defocus kernel either the Gaussian law of Block 4 or the skewed
generalized-t (SGT) family of Block 12 at the Gaussian law's FWHM.

The model is what the line renderers (Blocks 26, 35) compute for a flat tube along the pixel grid, reduced to
one dimension. Frame: the measuring line runs along y through a pixel centre (the origin); pixel n covers
[(n - 1/2) p, (n + 1/2) p] and is read at v_n = n p (Block 5's offsets, step p: the bilinear read of a pixel
centre is the pixel value, and the 2 A + 1 lines along the tube read equal pixels). The tube's axis lies along
x at lateral offset v0 (um) and depth z_ax (um), diameter d (um), coefficient mu (1/um).

  (1) Fine lattice across the line, as Block 26's: sample centres y_s = (i_s + 1/2) h - p/2, h = p / f
      (f = inference.line_fine_factor); the column through y_s is inside the tube over
      [z_lo, z_hi] = z_ax -+ sqrt(r^2 - (y_s - v0)^2) (Block 2 column_interval, tilt 0).
  (2) Slabs (Block 2 slab_centres over the columns' depth range, thickness renderer.dzeta_um) and the
      partition absorption: Delta A_j(y_s) = T_<j (1 - exp(-a_j)) (Block 2 slab_absorbance,
      transmitted_before, acquisition.light_direction).
  (3) Each slab-plane pair (j, k) at defocus delta = zeta_j - z_k is blurred by the 1-D profile kernel K_delta:
      the Gaussian of SD sigma_r(delta) (Block 4 kernel.sigma_r), or the SGT g_{w, kappa} of FWHM
      w(delta) = 2 sqrt(2 ln 2) sigma_r(delta) and shape kappa = (beta, xi, zeta) (Block 12, (H1)-(H3);
      blur_kernels). For a tube invariant along x the 2-D blur of an isotropic PSF reduces to its line
      projection: for the Gaussian that is the 1-D Gaussian of the same SD; for the SGT option the SGT is
      taken as the profile kernel itself (D-222).
  (4) Pixel mean of the blurred deficit, sample by sample with the kernel's exact CDF over the pixel (Block 26's
      rule: "integrates each sample's Gaussian over the pixel with ndtr"):
          1 - tau_k(n) = sum_j sum_s Delta A_j(y_s) (h / p) [K_{delta_jk}((n + 1/2) p - y_s) - K_{delta_jk}((n - 1/2) p - y_s)]
      with K the kernel's CDF.

What the 1-D reduction leaves out of Block 35 (stated, checked in validate_model.py): the tube's caps and the
reach rule (line_reach_sigmas) along the tube, both beyond 6 kernel SDs here.

The SGT's CDF (closed form, the Block 12 test oracle's): with s = zeta + 1/zeta, the core's half mass
H(y) = int_0^y h_{beta,xi} = (1/2) I_t(1/beta, 1/xi), t = xi y^beta / (1 + xi y^beta) (xi > SGT_XI_ZERO), or
(1/2) P(1/beta, y^beta) (xi = 0), and
    F(x | kappa) = 1/(1 + zeta^2) + (2 zeta^2/(1 + zeta^2)) H(x / zeta)     (x >= 0)
    F(x | kappa) = (2/(1 + zeta^2)) (1/2 - H(zeta |x|))                     (x < 0),
    K(t) = F(t W(kappa) / w | kappa),  W(kappa) = blur_kernels.sgt_unit_fwhm.
At (2, 0, 1) it is the Gaussian CDF Phi(t / sigma_r) (checked to rounding).

Two evaluations of H: "exact" (scipy.special.betainc / gammainc at every point) and "fast" (a cubic Hermite
interpolant of H on a fixed grid, the derivative h exact from blur_kernels.sgt_unit_pdf; the grid fixed, so the
model stays smooth in the parameters). The fast one is used for the recovery fits; its difference from the exact
one is reported by validate_model.py.

Custom: the 1-D reduction and the slab-plane sum (no library renders a depth-dependent blur; the pipeline's
line renderer is 2-D and 5 to 100 times slower, Block 35); the SGT's CDF (scipy.stats has no SGT, Block 12's
note; the closed form uses scipy.special.betainc / gammainc). The sum over (slab, plane, sample, pixel) is one
2-D correlation (scipy.signal.correlate) because the defocus delta_jk depends on j - R k only, R = dz / dzeta
an integer (14 at the defaults; asserted). Checked: scipy.signal.correlate, scipy.interpolate.CubicHermiteSpline.

Pure ASCII, LF.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import interpolate, signal, special

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
if str(WS / "src") not in sys.path:
    sys.path.insert(0, str(WS / "src"))

from allen_diameter.analysis import profiles          # noqa: E402
from allen_diameter.config import default_config      # noqa: E402
from allen_diameter.model import blur_kernels, geometry, kernel   # noqa: E402

FWHM_PER_SD = 2.0 * math.sqrt(2.0 * math.log(2.0))
GAUSS = (2.0, 0.0, 1.0)


# ---------------------------------------------------------------- the setup ---

@dataclass(frozen=True)
class Setup:
    """The line, the planes and the discretisation (all from the configuration)."""

    p: float                 # pixel pitch (um), acquisition.res0_um
    f: int                   # fine samples per pixel side, inference.line_fine_factor
    h: float                 # fine lattice spacing p / f (um)
    dzeta: float             # slab thickness (um), renderer.dzeta_um
    dz: float                # plane spacing (um), acquisition.dz_um
    R: int                   # dz / dzeta (integer)
    ks: np.ndarray           # plane indices (n_k,)
    z_planes: np.ndarray     # plane depths ks * dz (um)
    n_px: np.ndarray         # pixel rows read: v_n = n p
    v: np.ndarray            # line offsets (um), profiles.profile_offsets
    light: int               # acquisition.light_direction
    rcfg: object             # RendererConfig (kernel law)
    B_gl: float              # background (gl) of every plane
    sigma_gl: float          # noise SD of one line sample (gl)
    cfg: object = field(repr=False, default=None)


def make_setup(cfg=None, plane_half=None, B_gl=150.0, sigma_gl=1.1, k_centre=0):
    """Setup from a DiameterConfig (default: the deliverable defaults). plane_half: planes k_centre +- this
    (default cleaning.plane_half_range, Block 36). B_gl, sigma_gl: the background and the noise SD of a line
    sample (the check's choices, README)."""
    cfg = default_config() if cfg is None else cfg
    p = float(cfg.acquisition.res0_um)
    f = int(cfg.inference.line_fine_factor)
    dzeta = float(cfg.renderer.dzeta_um)
    dz = float(cfg.acquisition.dz_um)
    R = int(round(dz / dzeta))
    if abs(R * dzeta - dz) > 1e-12:
        raise ValueError("dz / dzeta must be an integer for the grouped slab-plane sum")
    half = int(cfg.cleaning.plane_half_range) if plane_half is None else int(plane_half)
    ks = np.arange(k_centre - half, k_centre + half + 1)
    v = profiles.profile_offsets(cfg.measure)
    n_px = np.rint(v / p).astype(int)
    if not np.allclose(n_px * p, v, rtol=0, atol=1e-12):
        raise ValueError("the line's samples must fall on pixel centres (profile_step_um = res0_um)")
    return Setup(p=p, f=f, h=p / f, dzeta=dzeta, dz=dz, R=R, ks=ks, z_planes=ks * dz, n_px=n_px, v=v,
                 light=int(cfg.acquisition.light_direction), rcfg=cfg.renderer, B_gl=float(B_gl),
                 sigma_gl=float(sigma_gl), cfg=cfg)


# ------------------------------------------------------------ the SGT CDF ---

def core_half_cdf(y, beta, xi):
    """H(y) = int_0^y h_{beta,xi}(u) du for y >= 0 (in [0, 1/2)), exact (scipy.special)."""
    y = np.asarray(y, dtype=float)
    if xi <= blur_kernels.SGT_XI_ZERO:
        return 0.5 * special.gammainc(1.0 / beta, y ** beta)
    yb = xi * y ** beta
    return 0.5 * special.betainc(1.0 / beta, 1.0 / xi, yb / (1.0 + yb))


def sgt_unit_cdf(x, beta, xi, zeta, H=None):
    """F(x | kappa), the CDF of blur_kernels.sgt_unit_pdf (unit integral, mode 0). H: a callable H(y) for
    y >= 0 (default core_half_cdf, exact)."""
    beta, xi, zeta = blur_kernels.check_sgt_shape(beta, xi, zeta)
    x = np.asarray(x, dtype=float)
    pos = x >= 0.0
    y = np.where(pos, x / zeta, -x * zeta)
    Hy = core_half_cdf(y, beta, xi) if H is None else H(y)
    z2 = zeta * zeta
    return np.where(pos, 1.0 / (1.0 + z2) + (2.0 * z2 / (1.0 + z2)) * Hy, (2.0 / (1.0 + z2)) * (0.5 - Hy))


class HermiteCore(object):
    """H(y) on a fixed grid by scipy.interpolate.CubicHermiteSpline, values exact (core_half_cdf) and
    derivatives h exact (blur_kernels.sgt_unit_pdf at zeta = 1, which is h on y >= 0). Grid: 0..0.02 step
    2e-5, 0.02..8 step 2e-3, then geometric (ratio 1.004) to Y_MAX; beyond Y_MAX the exact H is used."""

    Y_MAX = 600.0
    _grid = None

    @classmethod
    def grid(cls):
        if cls._grid is None:
            g1 = np.arange(0.0, 0.02, 2e-5)
            g2 = np.arange(0.02, 8.0, 2e-3)
            n3 = int(math.ceil(math.log(cls.Y_MAX / 8.0) / math.log(1.004)))
            g3 = 8.0 * 1.004 ** np.arange(0, n3 + 1)
            cls._grid = np.unique(np.concatenate([g1, g2, g3]))
        return cls._grid

    def __init__(self, beta, xi):
        self.beta, self.xi = float(beta), float(xi)
        y = self.grid()
        Hv = core_half_cdf(y, self.beta, self.xi)
        hv = blur_kernels.sgt_unit_pdf(y, self.beta, self.xi, 1.0)
        self.y_max = float(y[-1])
        self.spline = interpolate.CubicHermiteSpline(y, Hv, hv, extrapolate=False)

    def __call__(self, y):
        y = np.asarray(y, dtype=float)
        out = self.spline(np.minimum(y, self.y_max))
        far = y > self.y_max
        if np.any(far):
            out = np.where(far, core_half_cdf(y, self.beta, self.xi), out)
        return out


class Kernel(object):
    """The profile kernel's CDF K(t; sd) at the Gaussian law's SD sd (um): 'gauss' (ndtr) or the SGT of shape
    (beta, xi, zeta) at FWHM 2 sqrt(2 ln 2) sd, evaluated 'exact' or 'fast'."""

    def __init__(self, shape=GAUSS, method="exact", gauss_exact=True):
        self.shape = blur_kernels.check_sgt_shape(*shape)
        self.method = method
        self.is_gauss = gauss_exact and self.shape == GAUSS
        self.W = blur_kernels.sgt_unit_fwhm(*self.shape)
        self.H = None
        if not self.is_gauss and method == "fast":
            self.H = HermiteCore(self.shape[0], self.shape[1])
        elif method not in ("exact", "fast"):
            raise ValueError("method must be 'exact' or 'fast'")

    def cdf(self, t, sd):
        """K(t) for t (..., n) um and sd broadcastable (um)."""
        t = np.asarray(t, dtype=float)
        sd = np.asarray(sd, dtype=float)
        if self.is_gauss:
            return special.ndtr(t / sd)
        x = t * (self.W / (FWHM_PER_SD * sd))
        return sgt_unit_cdf(x, *self.shape, H=self.H)


# ---------------------------------------------------------- the forward model ---

def columns(d, v0, z_ax, setup):
    """The fine samples across the tube: (i_first, y_s (n_s,), z_lo, z_hi, inside) (Block 2 column_interval,
    an uncapped flat tube along x)."""
    r = 0.5 * float(d)
    h, p = setup.h, setup.p
    i_lo = int(math.floor((v0 - r + 0.5 * p) / h - 0.5)) - 1
    i_hi = int(math.ceil((v0 + r + 0.5 * p) / h - 0.5)) + 1
    i_s = np.arange(i_lo, i_hi + 1)
    y = (i_s + 0.5) * h - 0.5 * p
    tube = geometry.Tube((0.0, float(v0), float(z_ax)), r, 0.0, 0.0, 1.0, None)
    z_lo, z_hi, inside = geometry.column_interval(np.zeros_like(y), y, tube)
    return i_lo, y, z_lo, z_hi, inside


def absorbed(d, mu, v0, z_ax, setup):
    """(i_first, zeta (J,), dA (J, n_s)): the partition absorption of every slab at every fine sample."""
    i_lo, y, z_lo, z_hi, inside = columns(d, v0, z_ax, setup)
    if not inside.any():
        raise ValueError("no fine sample inside the tube (d too small)")
    zeta = geometry.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), setup.dzeta)
    zl, zh, zc = z_lo[None, :], z_hi[None, :], zeta[:, None]
    a = geometry.slab_absorbance(zl, zh, zc, setup.dzeta, mu)
    T = geometry.transmitted_before(zl, zh, zc, setup.dzeta, mu, setup.light)
    dA = np.where(inside[None, :], T * (-np.expm1(-a)), 0.0)
    return i_lo, zeta, dA


def kernel_table(zeta, i_first, n_s, setup, kern):
    """D[q, c]: (h/p) x the pixel integral of the kernel at defocus delta_q for the lattice offset index
    m = m_lo + c; q = j + R (k_max - k) groups the pairs of equal defocus (delta_jk = zeta_j - z_k).
    Returns (D, delta_q, m_lo)."""
    f, h, p, R = setup.f, setup.h, setup.p, setup.R
    ks = setup.ks
    k_max, k_min = int(ks.max()), int(ks.min())
    J = zeta.size
    n_q = J + R * (k_max - k_min)
    q = np.arange(n_q)
    # for every pair with q = j + R (k_max - k): zeta_j - k dz = zeta_0 + q dzeta - k_max dz (R dzeta = dz);
    # rows with no pair (q mod R >= J when J < R) are never read for the planes' rows
    delta = float(zeta[0]) + q * setup.dzeta - k_max * setup.dz
    sd = kernel.sigma_r(delta, setup.rcfg)
    n_lo, n_hi = int(setup.n_px.min()), int(setup.n_px.max())
    m_lo = n_lo * f - i_first - (n_s - 1)
    m_hi = n_hi * f - i_first
    m = np.arange(m_lo, m_hi + f + 1)
    t = (m - 0.5) * h                                   # edge offset from a sample (um)
    G = kern.cdf(t[None, :], sd[:, None])
    D = (h / p) * (G[:, f:] - G[:, :-f])
    return D, delta, m_lo


def deficit(d, mu, v0, z_ax, setup, kern, method="fft"):
    """1 - tau_k(v_n) (n_k, n_v) of the 1-D model (module docstring)."""
    i_first, zeta, dA = absorbed(d, mu, v0, z_ax, setup)
    n_s = dA.shape[1]
    D, _delta, _m_lo = kernel_table(zeta, i_first, n_s, setup, kern)
    out = signal.correlate(D, dA[:, ::-1], mode="valid", method=method)
    k_max = int(setup.ks.max())
    rows = setup.R * (k_max - setup.ks)
    cols = (setup.n_px - int(setup.n_px.min())) * setup.f
    return out[np.ix_(rows, cols)]


def tau(d, mu, v0, z_ax, setup, kern, method="fft"):
    """tau_k(v_n) (n_k, n_v), dimensionless."""
    return 1.0 - deficit(d, mu, v0, z_ax, setup, kern, method=method)


def deficit_direct(d, mu, v0, z_ax, setup, kern):
    """The same sum without grouping or correlation (every (j, k) pair's own sigma_r, an explicit loop):
    the reference for deficit()."""
    i_first, zeta, dA = absorbed(d, mu, v0, z_ax, setup)
    n_s = dA.shape[1]
    i_s = i_first + np.arange(n_s)
    f, h, p = setup.f, setup.h, setup.p
    out = np.zeros((setup.ks.size, setup.n_px.size))
    for kk, zk in enumerate(setup.z_planes):
        sd = kernel.sigma_r(zeta - zk, setup.rcfg)          # (J,)
        for nn, n in enumerate(setup.n_px):
            lo = ((n - 0.5) * p - ((i_s + 0.5) * h - 0.5 * p))[None, :]
            hi = lo + p
            K = (h / p) * (kern.cdf(hi, sd[:, None]) - kern.cdf(lo, sd[:, None]))
            out[kk, nn] = float(np.sum(dA * K))
    return out


# ------------------------------------------------------- models and parameters ---

PARAMS_GEOM = ("d", "mu", "v0", "z_ax")


@dataclass(frozen=True)
class ModelSpec:
    """Which kernel parameters are free. free: () Gaussian or a fixed shape, ("beta", "xi"), or
    ("beta", "xi", "zeta"); fixed_shape: the shape's values for the parameters not free."""

    free: tuple = ()
    fixed_shape: tuple = GAUSS
    method: str = "exact"

    @property
    def names(self):
        return PARAMS_GEOM + tuple(self.free)

    def shape_of(self, phi):
        """(beta, xi, zeta) from the natural parameter vector phi."""
        b, x, z = self.fixed_shape
        vals = dict(zip(self.free, phi[4:]))
        return (vals.get("beta", b), vals.get("xi", x), vals.get("zeta", z))

    def kern(self, phi):
        # a model with a free shape evaluates every shape, (2, 0, 1) included, by the SGT's CDF, so that its
        # finite differences never mix two code paths; the fixed Gaussian model uses ndtr (Block 26's rule)
        return Kernel(self.shape_of(phi), method=self.method, gauss_exact=not self.free)


def tau_phi(phi, spec, setup):
    """tau (n_k, n_v) at the natural parameters phi = (d, mu, v0, z_ax[, beta, xi[, zeta]])."""
    return tau(phi[0], phi[1], phi[2], phi[3], setup, spec.kern(phi))


def project_background(I, t, w=None):
    """Per plane B_k = sum_n w I t / sum_n w t^2 (the one scale per plane solved exactly) -> (n_k,)."""
    if w is None:
        return np.sum(I * t, axis=1) / np.sum(t * t, axis=1)
    return np.sum(w * I * t, axis=1) / np.sum(w * t * t, axis=1)


# ------------------------------------------------------------ finite differences ---

def default_steps(spec):
    """Absolute central-difference steps in the natural parameters (um, 1/um, um, um, -, -, -)."""
    base = {"d": 2e-3, "mu": 1e-3, "v0": 2e-3, "z_ax": 2e-3, "beta": 2e-3, "xi": 1e-3, "zeta": 1e-3}
    return np.array([base[n] for n in spec.names])


def jacobian_tau(phi, spec, setup, steps=None, lower=None):
    """d tau / d phi (n_k, n_v, p) by central differences; a parameter whose phi - step would cross its lower
    bound (lower[i], e.g. xi >= 0) uses the one-sided 3-point rule (-3 f0 + 4 f1 - f2) / (2 step)."""
    phi = np.asarray(phi, dtype=float)
    steps = default_steps(spec) if steps is None else np.asarray(steps, dtype=float)
    lower = np.full(phi.size, -np.inf) if lower is None else np.asarray(lower, dtype=float)
    t0 = tau_phi(phi, spec, setup)
    J = np.empty(t0.shape + (phi.size,))
    for i in range(phi.size):
        e = np.zeros_like(phi)
        e[i] = steps[i]
        if phi[i] - steps[i] < lower[i]:
            t1 = tau_phi(phi + e, spec, setup)
            t2 = tau_phi(phi + 2.0 * e, spec, setup)
            J[..., i] = (-3.0 * t0 + 4.0 * t1 - t2) / (2.0 * steps[i])
        else:
            J[..., i] = (tau_phi(phi + e, spec, setup) - tau_phi(phi - e, spec, setup)) / (2.0 * steps[i])
    return t0, J


def lower_bounds(spec):
    lo = {"d": 0.0, "mu": 0.0, "v0": -np.inf, "z_ax": -np.inf, "beta": 0.0, "xi": 0.0, "zeta": 0.0}
    return np.array([lo[n] for n in spec.names])


def fisher(phi, spec, setup, B=None, planes=None, steps=None, weights=None, jac=None):
    """The Fisher information of phi with the per-plane backgrounds profiled out (Schur complement), for
    I_kn = B_k tau_kn(phi) + noise of SD setup.sigma_gl, over the planes kept (bool (n_k,), default all).
    weights (n_v,) or None: with weights the sandwich covariance of the weighted fit is returned as 'cov'.
    jac: (tau, J_tau) of jacobian_tau over all planes, to reuse (default: computed here).
    Returns dict(F (p, p), cov, se, corr, cond_corr, vif, F_full, names, tau, J_tau, J_full)."""
    B = setup.B_gl if B is None else B
    if jac is None:
        t_all, J_all = jacobian_tau(phi, spec, setup, steps=steps, lower=lower_bounds(spec))
    else:
        t_all, J_all = jac
    keep = np.ones(t_all.shape[0], bool) if planes is None else np.asarray(planes, bool)
    t0, Jt = t_all[keep], J_all[keep]
    n_k, n_v, p = Jt.shape
    Bk = np.full(n_k, float(B)) if np.ndim(B) == 0 else np.asarray(B, float)[keep]
    s = setup.sigma_gl
    # full Jacobian of y = B_k tau_kn in (phi, B_1..B_K), in noise units
    Jphi = (Bk[:, None, None] * Jt).reshape(n_k * n_v, p) / s
    JB = np.zeros((n_k * n_v, n_k))
    for k in range(n_k):
        JB[k * n_v:(k + 1) * n_v, k] = t0[k] / s
    Jf = np.hstack([Jphi, JB])
    if weights is None:
        Ff = Jf.T @ Jf
        F = Ff[:p, :p] - Ff[:p, p:] @ np.linalg.solve(Ff[p:, p:], Ff[p:, :p])
        cov_full = np.linalg.inv(Ff)
        cov = cov_full[:p, :p]
    else:
        w = np.tile(np.asarray(weights, float), n_k)
        A = Jf.T @ (w[:, None] * Jf)
        Bm = Jf.T @ ((w * w)[:, None] * Jf)
        Ainv = np.linalg.inv(A)
        cov_full = Ainv @ Bm @ Ainv
        cov = cov_full[:p, :p]
        Ff = A
        F = Ff[:p, :p] - Ff[:p, p:] @ np.linalg.solve(Ff[p:, p:], Ff[p:, :p])
    se = np.sqrt(np.diag(cov))
    corr = cov / np.outer(se, se)
    ev = np.linalg.eigvalsh(corr)
    cond = float(ev.max() / ev.min()) if ev.min() > 0 else float("inf")
    vif = np.diag(cov) * np.diag(F) if weights is None else np.full(p, np.nan)
    # the backgrounds known (Block 30's fit_node holds B_k at the measured values): the phi block alone
    if weights is None:
        cov_bk = np.linalg.inv(Ff[:p, :p])
    else:
        A_k = Ff[:p, :p]
        B_k = (Jf[:, :p].T @ ((w * w)[:, None] * Jf[:, :p]))
        Ak_inv = np.linalg.inv(A_k)
        cov_bk = Ak_inv @ B_k @ Ak_inv
    se_bk = np.sqrt(np.diag(cov_bk))
    corr_bk = cov_bk / np.outer(se_bk, se_bk)
    ev_bk = np.linalg.eigvalsh(corr_bk)
    cond_bk = float(ev_bk.max() / ev_bk.min()) if ev_bk.min() > 0 else float("inf")
    return dict(F=F, cov=cov, se=se, corr=corr, cond_corr=cond, vif=vif, F_full=Ff, names=spec.names,
                tau=t0, J_tau=Jt, J_full=Jf, jac_all=(t_all, J_all),
                bk=dict(se=se_bk, corr=corr_bk, cond_corr=cond_bk, names=spec.names))


if __name__ == "__main__":
    import time
    st = make_setup()
    for d in (0.4, 0.8, 1.5, 2.5):
        for shape in (GAUSS, (2.0, 0.5, 1.0), (1.3, 0.0, 1.0), (2.0, 0.3, 1.3)):
            for meth in ("exact", "fast"):
                kern = Kernel(shape, method=meth)
                t0 = time.perf_counter()
                for _ in range(3):
                    tt = tau(d, 1.5, 0.04, 0.0, st, kern)
                dt = (time.perf_counter() - t0) / 3
                print("d %.1f shape %s %-5s  %.1f ms  min tau %.4f" % (d, shape, meth, 1e3 * dt, tt.min()))
