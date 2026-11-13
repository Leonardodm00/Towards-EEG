"""Independent references for the spec-derived tests of Blocks 23-31 (the
cleaning and the per-node inference; tester, 2026-11-13). Written from
specs/SPEC.md (sections 2.1, 2.3, Blocks 23-31) only: nothing here calls the
code under test.

  cfg_with           default_config() with overrides (Block 1 with_overrides)
  flat_columns       (S3) for a flat (phi = 0) capped tube: |v| <= r, |u| <= U,
                     z in c_z -+ sqrt(r^2 - v^2)
  ref_lattice        slab centres over [z_min, z_max] (Block 2's rule as section
                     2.3 / Block 24 state it: zeta_j = z_min + (j + 1/2) dzeta)
  ref_render_flat    Eqs. 5-6 by plain numpy + scipy.ndimage.gaussian_filter for
                     flat tubes, "linear" and "partition_vertical" (several
                     tubes in one slab lattice, Block 24's mathematics)
  fine_lattice       Block 4 / Block 25 lattice of a pixel block: pixel (row,
                     col) centred on ((left + col) p, (top + row) p), f x f
                     samples per pixel
  pixel_mean         mean over f x f samples
  mean_inv_sz        <1/s_z> over the condenser's disc (section 2.3):
                     2 (1 - sqrt(1 - s_m^2)) / s_m^2
  bilinear_read      Block 5's reading rule (bilinear between pixel centres,
                     mean over 2 A + 1 lines) written with scipy.ndimage

Pure ASCII.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", WS / "scripts", WS / "tests" / "smoke", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from spec_helpers import ref_sigma_r  # noqa: E402  (independent kernel law, earlier tester run)


def cfg_with(overrides=None):
    from allen_diameter import config as C
    cfg = C.default_config()
    return C.with_overrides(cfg, overrides) if overrides else cfg


def fine_lattice(left, top, width, height, p, f):
    """(xs (W f,), ys (H f,)) sample centres of a pixel block, section 2.1 / Block 4."""
    h = p / f
    xs = (left * p - 0.5 * p + 0.5 * h) + h * np.arange(width * f)
    ys = (top * p - 0.5 * p + 0.5 * h) + h * np.arange(height * f)
    return xs, ys


def pixel_mean(a, f):
    a = np.asarray(a, dtype=float)
    ny, nx = a.shape[-2], a.shape[-1]
    return a.reshape(a.shape[:-2] + (ny // f, f, nx // f, f)).mean(axis=(-3, -1))


def flat_columns(X, Y, c, r, theta, U):
    """(z_lo, z_hi, inside) of a flat capped round tube (phi = 0): (S1)-(S3)."""
    u = (X - c[0]) * math.cos(theta) + (Y - c[1]) * math.sin(theta)
    v = -(X - c[0]) * math.sin(theta) + (Y - c[1]) * math.cos(theta)
    inside = (np.abs(v) <= r) & (np.abs(u) <= U)
    half = np.sqrt(np.clip(r * r - v * v, 0.0, None))
    z_lo = np.where(inside, c[2] - half, c[2])
    z_hi = np.where(inside, c[2] + half, c[2])
    return z_lo, z_hi, inside


def ref_lattice(z_min, z_max, dzeta):
    J = max(1, int(math.ceil((z_max - z_min) / dzeta - 1e-12)))
    return z_min + (np.arange(J) + 0.5) * dzeta


def ref_render_flat(tubes, mus, xs, ys, z_planes, rcfg, absorption, light=+1, zeta=None):
    """tau_k (n_planes, ny, nx) on the sample lattice (xs, ys) of flat tubes
    [(c, r, theta, U)], Eqs. 5-6 with the slabs of all tubes in one lattice.
    Blur: scipy.ndimage.gaussian_filter, mode constant (linear convolution on
    the grid), truncate 8. Light +1: slabs numbered upward."""
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    h = float(xs[1] - xs[0])
    cols = [flat_columns(X, Y, *t) for t in tubes]
    if zeta is None:
        lo = min(float(z[0][z[2]].min()) for z in cols if z[2].any())
        hi = max(float(z[1][z[2]].max()) for z in cols if z[2].any())
        zeta = ref_lattice(lo, hi, rcfg.dzeta_um)
    dz = rcfg.dzeta_um
    a = np.zeros((zeta.size,) + X.shape)
    for (z_lo, z_hi, _ins), mu in zip(cols, mus):
        for j, zj in enumerate(zeta):
            a[j] += mu * np.clip(np.minimum(z_hi, zj + dz / 2) - np.maximum(z_lo, zj - dz / 2), 0.0, None)
    if absorption == "linear":
        dA = a
    elif absorption == "partition_vertical":
        order = np.arange(zeta.size) if light == +1 else np.arange(zeta.size)[::-1]
        dA = np.zeros_like(a)
        before = np.zeros(X.shape)
        for j in order:
            dA[j] = np.exp(-before) * (-np.expm1(-a[j]))
            before = before + a[j]
    else:
        raise ValueError(absorption)
    out = np.empty((len(z_planes),) + X.shape)
    for k, zk in enumerate(z_planes):
        tot = np.zeros(X.shape)
        for j, zj in enumerate(zeta):
            if not dA[j].any():
                continue
            s = float(ref_sigma_r(zj - zk, rcfg))
            tot += ndimage.gaussian_filter(dA[j], s / h, mode="constant", cval=0.0, truncate=8.0)
        out[k] = 1.0 - tot
    return out, zeta, dA


def mean_inv_sz(na, n_oil):
    s2 = (float(na) / float(n_oil)) ** 2
    return 2.0 * (1.0 - math.sqrt(1.0 - s2)) / s2


def bilinear_read(plane, left, top, p, origin, y_hat, e_u, v, n_avg, step):
    """Block 5's rule from section 2.2 / Block 26: bilinear between pixel
    centres, mean over 2 n_avg + 1 lines step um apart along e_u."""
    tot = np.zeros(len(v))
    for a in range(-n_avg, n_avg + 1):
        x = origin[0] + np.asarray(v) * y_hat[0] + a * step * e_u[0]
        y = origin[1] + np.asarray(v) * y_hat[1] + a * step * e_u[1]
        tot += ndimage.map_coordinates(np.asarray(plane, float), [y / p - top, x / p - left], order=1,
                                       mode="constant", cval=np.nan)
    return tot / (2 * n_avg + 1)
