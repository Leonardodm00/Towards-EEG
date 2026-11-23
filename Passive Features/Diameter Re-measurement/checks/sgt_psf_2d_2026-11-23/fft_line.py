"""An FFT reference for Block 35's line read with an arbitrary 2-D blur kernel (wave 7 design check; not pipeline code).

The quantity is Block 26/35's: for plane k the pixel means of
    D_k(x) = sum_j (Delta A_j * K_{jk})(x),        tau_k(v_n) = 1 - read(D_k)(v_n)
with Delta A_j the partition absorption of slab j on the fine lattice x = (n + 1/2) h - p/2 (h = p / f), each
sample a point mass h^2 Delta A_j; K_{jk} any kernel given by its continuous Fourier transform Khat(nu) (unit
integral). The pixel mean of a point mass's blur is (K * box_p / p^2) at the pixel centre, whose transform is
Khat(nu) sinc(p nu_x) sinc(p nu_y). On a zero-padded periodic grid of the lattice:

    D_k(x_n + h/2) = ifft2( fft2(Delta A) * Khat * sinc * sinc * exp(2 pi i (nu_x + nu_y) h / 2) )_n

(f even: the half-sample shift puts pixel centres on the output lattice, n = c f + f/2 - 1). The aliases
Khat(nu + m / h) are dropped (checked against Block 35 for the Gaussian, where they are below 1e-12). The grid
is padded by `pad_um` around the union of the tube's footprint and the read's pixel box; pairs whose kernel SD
exceeds `sigma_cut_um` are left out for every kernel alike (the caller reports their share of the deficit).
"""
import math

import numpy as np
from scipy import fft as sfft

from allen_diameter.analysis.profiles import sample_profile
from allen_diameter.model import geometry, line_fast
from allen_diameter.model import line_render as lr
from allen_diameter.model.kernel import sigma_r


def render(tube, mu, ks, origin, y_hat, e_u, v, n_avg, step, cfg, transfer, pad_um, sigma_cut_um, workers=2):
    """tau (n_planes, n_v); transfer(nu_x, nu_y, sigma_um) -> Khat (broadcast arrays), sigma the law's SD of the pair."""
    acq, rcfg, icfg = cfg.acquisition, cfg.renderer, cfg.inference
    f = int(icfg.line_fine_factor)
    assert f % 2 == 0
    p, dz, dzeta, light = float(acq.res0_um), float(acq.dz_um), float(rcfg.dzeta_um), acq.light_direction
    h = p / f
    o, yh, eu, v, n_avg, step = lr._line_args(origin, y_hat, e_u, v, n_avg, step)
    ks = np.asarray(ks, dtype=np.int64)
    c_lo, c_hi, r_lo, r_hi, ex, ey = line_fast._pixel_box(o, yh, eu, v, n_avg, step, p)
    ix0, iy0, xs_all, ys_all, X, Y, z_lo, z_hi, inside = line_fast._footprint(tube, h, p)
    zeta = geometry.slab_centres(float(z_lo[inside].min()), float(z_hi[inside].max()), dzeta)
    sig = sigma_r(zeta[:, None] - (ks * dz)[None, :], rcfg)           # (J, n_planes)

    # the grid, in lattice indices: footprint and the pixels' output indices, padded
    out_cx = np.arange(c_lo, c_hi + 1) * f + f // 2 - 1
    out_ry = np.arange(r_lo, r_hi + 1) * f + f // 2 - 1
    npad = int(math.ceil(pad_um / h))
    gx0 = min(ix0, int(out_cx[0])) - npad
    gx1 = max(ix0 + xs_all.size, int(out_cx[-1]) + 1) + npad
    gy0 = min(iy0, int(out_ry[0])) - npad
    gy1 = max(iy0 + ys_all.size, int(out_ry[-1]) + 1) + npad
    Nx = sfft.next_fast_len(gx1 - gx0, real=True)
    Ny = sfft.next_fast_len(gy1 - gy0)
    nux = sfft.rfftfreq(Nx, h)[None, :]
    nuy = sfft.fftfreq(Ny, h)[:, None]
    common = np.sinc(p * nux) * np.sinc(p * nuy) * np.exp(1j * math.pi * (nux + nuy) * h)

    acc = np.zeros((ks.size, Ny, Nx // 2 + 1), dtype=complex)
    used = 0
    for j in range(zeta.size):
        zj = float(zeta[j])
        if not np.any(sig[j] <= sigma_cut_um):
            continue
        m = inside & (z_lo < zj + 0.5 * dzeta) & (z_hi > zj - 0.5 * dzeta)
        if not m.any():
            continue
        a = geometry.slab_absorbance(z_lo, z_hi, zj, dzeta, mu)
        dA = np.where(m, geometry.transmitted_before(z_lo, z_hi, zj, dzeta, mu, light) * (-np.expm1(-a)), 0.0)
        grid = np.zeros((Ny, Nx))
        grid[iy0 - gy0:iy0 - gy0 + ys_all.size, ix0 - gx0:ix0 - gx0 + xs_all.size] = dA
        A = sfft.rfft2(grid, workers=workers)
        for k in range(ks.size):
            s = float(sig[j, k])
            if s > sigma_cut_um:
                continue
            acc[k] += A * transfer(nux, nuy, s)
            used += 1
    out = np.empty((ks.size, v.size))
    frame = lr._Frame(c_lo, r_lo, p)
    for k in range(ks.size):
        field = sfft.irfft2(acc[k] * common, s=(Ny, Nx), workers=workers)
        deficit = field[np.ix_(out_ry - gy0, out_cx - gx0)]
        out[k] = 1.0 - sample_profile(deficit, frame, o, yh, eu, v, n_avg, step)
    return out, used


def gauss_transfer():
    def t(nux, nuy, s):
        return np.exp(-2.0 * math.pi ** 2 * s * s * (nux * nux + nuy * nuy))
    return t


def unit_cf_table(pdf, L=2000.0, n=2 ** 23, t_max=200.0):
    """The unit kernel's characteristic function ghat(t) = int pdf(x) exp(-2 pi i t x) dx on t in [0, t_max]
    (complex; by a 1-D FFT of the sampled pdf on [-L, L)): (t, ghat)."""
    dx = 2.0 * L / n
    x = -L + dx * np.arange(n)
    g = pdf(x)
    G = dx * np.fft.fft(np.fft.ifftshift(g))            # x = 0 at index 0 after the shift
    t = np.fft.fftfreq(n, dx)
    keep = (t >= 0) & (t <= t_max)
    order = np.argsort(t[keep])
    return t[keep][order], G[keep][order]


def _interp_cf(table, tq):
    t, G = table
    ta = np.abs(tq)
    re = np.interp(ta, t, G.real, right=0.0)
    im = np.interp(ta, t, G.imag, right=0.0) * np.sign(tq)   # ghat(-t) = conj(ghat(t)) for a real pdf
    return re + 1j * im if np.any(G.imag != 0) else re


def iso_transfer(table, unit_fwhm):
    """Isotropic PSF whose line projection is the SGT of FWHM w = 2 sqrt(2 ln 2) sigma: Khat = ghat(a |nu|)."""
    c = 2.0 * math.sqrt(2.0 * math.log(2.0))

    def t(nux, nuy, s):
        a = c * s / unit_fwhm
        return _interp_cf(table, a * np.hypot(nux, nuy))
    return t


def sep_transfer(table, unit_fwhm, y_hat, e_u):
    """SGT across the measuring line (along y_hat), the law's Gaussian along it (e_u):
    Khat = ghat(a nu.y_hat) exp(-2 pi^2 sigma^2 (nu.e_u)^2)."""
    c = 2.0 * math.sqrt(2.0 * math.log(2.0))
    yh = np.asarray(y_hat, dtype=float)
    eu = np.asarray(e_u, dtype=float)

    def t(nux, nuy, s):
        a = c * s / unit_fwhm
        nu_u = nux * yh[0] + nuy * yh[1]
        nu_s = nux * eu[0] + nuy * eu[1]
        return _interp_cf(table, a * nu_u) * np.exp(-2.0 * math.pi ** 2 * s * s * nu_s * nu_s)
    return t
