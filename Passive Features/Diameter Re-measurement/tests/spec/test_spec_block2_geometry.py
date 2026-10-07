"""Block 2 (geometry): closed forms (S1)-(S5) checked against independent
membership sampling written here from SPEC.md section 2.3 / Block 2."""
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from allen_diameter.model import geometry as G


def _member(x, y, z, c, r, phi, theta, U=None, cut="axial"):
    """(S1)-(S2) at k = 1, written from the spec: membership of points."""
    u = (x - c[0]) * math.cos(theta) + (y - c[1]) * math.sin(theta)
    v = -(x - c[0]) * math.sin(theta) + (y - c[1]) * math.cos(theta)
    w = z - c[2]
    ok = v ** 2 + (w * math.cos(phi) - u * math.sin(phi)) ** 2 <= r ** 2
    if U is not None:
        if cut == "vertical":
            ok &= np.abs(u) <= U
        else:
            ok &= np.abs(u * math.cos(phi) + w * math.sin(phi)) <= U
    return ok


@pytest.mark.parametrize("cut", ["axial", "vertical"])
def test_column_interval_vs_sampled_membership(cut):
    rng = np.random.default_rng(1)
    for _ in range(6):
        c = rng.uniform(-1, 1, 3)
        r = rng.uniform(0.2, 1.5)
        phi = math.radians(rng.uniform(0, 80))
        theta = rng.uniform(-math.pi, math.pi)
        tube = G.Tube(tuple(c), r, phi, theta, 1.0, 3.0, cut)
        xy = c[:2] + rng.uniform(-4, 4, (40, 2))
        zlo, zhi, inside = G.column_interval(xy[:, 0], xy[:, 1], tube)
        zz = np.linspace(c[2] - 30, c[2] + 30, 200001)   # step 3e-4 um
        dzz = zz[1] - zz[0]
        for n in range(xy.shape[0]):
            m = _member(xy[n, 0], xy[n, 1], zz, c, r, phi, theta, 3.0, cut)
            if not m.any():
                assert not inside[n] or zhi[n] - zlo[n] < 2 * dzz
                continue
            assert inside[n]
            # interval ends to within two sampling steps of the brute-force membership
            assert abs(zz[m].min() - zlo[n]) <= 2 * dzz and abs(zz[m].max() - zhi[n]) <= 2 * dzz


def test_central_chord_and_misses():
    for k in (0.5, 1.0, 1.6):
        for phi_deg in (0, 30, 70):
            phi = math.radians(phi_deg)
            tube = G.Tube((0.3, -0.2, 1.0), 0.7, phi, 0.4, k, None)
            zlo, zhi, inside = G.column_interval(0.3, -0.2, tube)
            assert_allclose(zhi - zlo, 2 * 0.7 * math.sqrt(k * k + math.tan(phi) ** 2), rtol=1e-12)
    tube = G.Tube((0, 0, 2.0), 0.5, 0.3, 0.0, 1.0, 5.0)
    zlo, zhi, inside = G.column_interval(np.array([0.0]), np.array([3.0]), tube)
    assert not inside[0] and zlo[0] == 2.0 and zhi[0] == 2.0 and np.isfinite(zlo).all()


def test_slab_absorbance_and_transmission_identities():
    rng = np.random.default_rng(2)
    zlo = rng.uniform(-1, 0, 50)
    zhi = zlo + rng.uniform(0, 2, 50)
    mu, dz = 1.7, 0.02
    zeta = G.slab_centres(zlo.min(), zhi.max(), dz)
    a = np.array([G.slab_absorbance(zlo, zhi, zj, dz, mu) for zj in zeta])
    assert_allclose(a.sum(0), mu * (zhi - zlo), rtol=1e-12, atol=1e-14)
    for ld in (+1, -1):
        T = np.array([G.transmitted_before(zlo, zhi, zj, dz, mu, ld) for zj in zeta])
        order = np.arange(len(zeta)) if ld == +1 else np.arange(len(zeta))[::-1]
        cum = np.exp(-np.concatenate([np.zeros((1, 50)), np.cumsum(a[order], 0)[:-1]]))
        assert_allclose(T[order], cum, rtol=1e-12, atol=1e-14)
        # C1: partition sums to Beer-Lambert of the whole column
        assert_allclose((T * (1 - np.exp(-a))).sum(0), 1 - np.exp(-a.sum(0)), atol=1e-12)


def test_depth_extent_formulas():
    for cut in ("axial", "vertical"):
        for k in (0.6, 1.0, 1.4):
            phi, U, r = math.radians(40), 3.0, 0.6
            tube = G.Tube((0, 0, 0.5), r, phi, 0.7, k, U, cut)
            lo, hi = G.depth_extent(tube)
            phi0 = math.atan(math.tan(phi) / k)
            h = (U * math.tan(phi) + r * math.sqrt(k * k + math.tan(phi) ** 2)) if cut == "vertical" \
                else k * (U * math.sin(phi0) + r * math.cos(phi0))
            assert_allclose((lo, hi), (0.5 - h, 0.5 + h), rtol=1e-12)
            xs = np.linspace(-6, 6, 601)
            X, Y = np.meshgrid(xs, xs)
            zl, zh, ins = G.column_interval(X, Y, tube)
            assert zl[ins].min() >= lo - 1e-9 and zh[ins].max() <= hi + 1e-9


def test_theta_plus_pi_symmetry_and_line_interval_axis():
    rng = np.random.default_rng(3)
    X, Y = rng.uniform(-3, 3, (2, 300))
    t1 = G.Tube((0.1, 0.2, 0.0), 0.5, 0.0, 0.3, 1.0, 2.0)
    t2 = G.Tube((0.1, 0.2, 0.0), 0.5, 0.0, 0.3 + math.pi, 1.0, 2.0)
    a, b = G.column_interval(X, Y, t1), G.column_interval(X, Y, t2)
    for p, q in zip(a, b):
        assert_allclose(p, q, atol=1e-12)
    # a line along the axis with axial caps (k = 1) has length 2U
    tube = G.Tube((0, 0, 0), 0.4, 0.5, 1.1, 1.0, 2.5, "axial")
    t = G.axis_direction(0.5, 1.1)
    t1_, t2_, hit = G.line_interval(np.zeros(3), t * 3.0, tube)
    assert hit and abs((t2_ - t1_) * 3.0 - 5.0) < 1e-9


@pytest.mark.parametrize("kw", [dict(r=0.0), dict(r=-1.0), dict(phi=math.pi / 2), dict(phi=-0.1),
                                dict(aspect=0.0), dict(half_length=0.0), dict(r=float("nan")),
                                dict(phi=float("nan"))])
def test_invalid_tubes(kw):
    base = dict(c=(0, 0, 0), r=0.5, phi=0.2, theta=0.0)
    base.update(kw)
    with pytest.raises(ValueError):
        G.Tube(**base)
