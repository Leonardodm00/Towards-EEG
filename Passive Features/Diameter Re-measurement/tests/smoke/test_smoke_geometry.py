"""Smoke test for the tube geometry -- Block 2 in specs/SPEC.md.

Checks
    test_known_answer   width along e_v is d for every phi and aspect (handoff
                        Eq. 7); oblique horizontal cut 2r / sqrt(cos^2 a sin^2 phi
                        + sin^2 a) (handoff Eq. 8; the 1.265 d example); the
                        vertical line through c has length 2 r sqrt(k^2 + tan^2 phi)
                        (= d / cos phi at k = 1, handoff Eq. 10); a line along the
                        axis has length 2U / |S^-1 t_hat| with axial caps
    test_reference      column_interval vs an independent local-frame membership
                        test (S2) and vs inside_tube (handoff Eq. 6): 0 mismatches;
                        slab chords and line chords vs brute-force sampling; and
                        agreement with checks/stack_geometry_check.py (k = 1)
    test_convergence    skipped: closed forms (the slab sum is exact at every dzeta)
    test_invariants     sum_j a_j = mu * length; T_<j equals the cumulative
                        product of exp(-a) for both light directions; the
                        absorbed fractions telescope (C1); the analytic depth
                        extent bounds every column and is attained; rotating the
                        points by pi about the vertical through c with theta ->
                        theta + pi changes nothing (S3)
    test_contract       shapes, dtypes, no NaN, zero length outside
    test_determinism    skipped: deterministic closed forms
    test_edge_cases     phi = 0 with axial caps; tangent line; invalid tubes

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_geometry.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.model import geometry as G  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy",)


def make_rng(seed=SEED):
    return np.random.default_rng(seed)


def random_tube(rng, cut=None, aspect=None, phi_max_deg=80.0):
    r = rng.uniform(0.1, 2.0)
    phi = math.radians(rng.uniform(0.0, phi_max_deg))
    theta = rng.uniform(-math.pi, math.pi)
    c = tuple(rng.uniform(-1, 1, 3))
    k = rng.uniform(0.4, 1.6) if aspect is None else aspect
    if cut is None:
        return G.Tube(c, r, phi, theta, k, None)
    return G.Tube(c, r, phi, theta, k, rng.uniform(1.0, 6.0), cut)


def local_membership(x, y, z, tube):
    """Independent (S2) membership in unsquashed space, with the end cut,
    written from the spec without calling the module's own formulas."""
    dx, dy, dz = np.asarray(x) - tube.c[0], np.asarray(y) - tube.c[1], np.asarray(z) - tube.c[2]
    u = dx * math.cos(tube.theta) + dy * math.sin(tube.theta)
    v = -dx * math.sin(tube.theta) + dy * math.cos(tube.theta)
    w0 = dz / tube.aspect
    p0 = math.atan(math.tan(tube.phi) / tube.aspect)
    ok = v ** 2 + (w0 * math.cos(p0) - u * math.sin(p0)) ** 2 <= tube.r ** 2
    if tube.half_length is not None:
        if tube.end_cut == "axial":
            ok &= np.abs(u * math.cos(p0) + w0 * math.sin(p0)) <= tube.half_length
        else:
            ok &= np.abs(u) <= tube.half_length
    return ok


def load_check_script():
    path = WS / "checks" / "stack_geometry_check.py"
    spec = importlib.util.spec_from_file_location("stack_geometry_check", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    rng = make_rng()
    for _ in range(60):
        cut = rng.choice([None, "axial", "vertical"])
        t = random_tube(rng, cut=cut, phi_max_deg=85.0)
        P = np.asarray(t.c)
        # handoff Eq. 7: width along e_v through the node is d, any phi, any aspect
        t1, t2, hit = G.line_interval(P, t.e_v, t)
        assert hit and abs((t2 - t1) - t.d) < 1e-12, (t, t2 - t1)
        # vertical line through c: 2 r sqrt(k^2 + tan^2 phi), unless an axial cap cuts it
        z_lo, z_hi, ins = G.column_interval(t.c[0], t.c[1], t)
        expect = 2 * t.r * math.sqrt(t.aspect ** 2 + math.tan(t.phi) ** 2)
        if cut == "axial":
            # cap at |w0 sin(phi0)| <= U  ->  |w| <= k U / sin(phi0)
            p0 = t.phi0
            if math.sin(p0) > 0:
                expect = min(expect, 2 * t.aspect * t.half_length / math.sin(p0))
        assert ins and abs((z_hi - z_lo) - expect) < 1e-10, (cut, z_hi - z_lo, expect)
        t1, t2, hit = G.line_interval(P, np.array([0.0, 0.0, 1.0]), t)
        assert hit and abs((t2 - t1) - expect) < 1e-10
    # handoff Eq. 8, k = 1: horizontal line at angle a to e_u
    for _ in range(40):
        t = random_tube(rng, aspect=1.0)
        a = rng.uniform(-math.pi, math.pi)
        s = np.array([math.cos(t.theta + a), math.sin(t.theta + a), 0.0])
        t1, t2, hit = G.line_interval(np.asarray(t.c), s, t)
        expect = 2 * t.r / math.sqrt(math.cos(a) ** 2 * math.sin(t.phi) ** 2 + math.sin(a) ** 2)
        assert hit and abs((t2 - t1) - expect) < 1e-10 * expect, (t2 - t1, expect)
    # the handoff's example: pixel row, theta = 45 deg, phi = 30 deg -> 1.265 d
    t = G.Tube((0.0, 0.0, 0.0), 0.5, math.radians(30), math.radians(45))
    t1, t2, _ = G.line_interval(np.zeros(3), np.array([1.0, 0.0, 0.0]), t)
    assert abs((t2 - t1) / t.d - 1.2649) < 1e-4, (t2 - t1) / t.d
    # a line along the observed axis with axial caps: length 2U / |S^-1 t_hat|
    for _ in range(30):
        t = random_tube(rng, cut="axial")
        th = t.t_hat
        s0n = math.sqrt(th[0] ** 2 + th[1] ** 2 + (th[2] / t.aspect) ** 2)
        t1, t2, hit = G.line_interval(np.asarray(t.c), th, t)
        assert hit and abs((t2 - t1) - 2 * t.half_length / s0n) < 1e-9, (t2 - t1, 2 * t.half_length / s0n)


def test_reference():
    rng = make_rng(1)
    # (a) column interval vs two independent membership tests
    n_mis = 0
    n_tot = 0
    for _ in range(40):
        t = random_tube(rng, cut=rng.choice([None, "axial", "vertical"]))
        x = t.c[0] + rng.uniform(-8, 8, 3000)
        y = t.c[1] + rng.uniform(-8, 8, 3000)
        z_lo, z_hi, ins = G.column_interval(x, y, t)
        zmin, zmax = G.depth_extent(t)
        if not np.isfinite(zmin):
            zmin, zmax = t.c[2] - 20, t.c[2] + 20
        z = rng.uniform(zmin - 0.5, zmax + 0.5, x.size)
        by_col = ins & (z >= z_lo) & (z <= z_hi)
        by_eq6 = G.inside_tube(x, y, z, t)
        by_loc = local_membership(x, y, z, t)
        safe = (np.abs(z - z_lo) > 1e-9) & (np.abs(z - z_hi) > 1e-9)
        n_mis += int(np.sum((by_col != by_eq6) & safe)) + int(np.sum((by_col != by_loc) & safe))
        n_tot += int(safe.sum())
        # points just inside / just outside the interval ends
        sel = ins & (z_hi - z_lo > 1e-6)
        for zz, expect in ((z_lo + 1e-8, True), (z_hi - 1e-8, True), (z_lo - 1e-6, False), (z_hi + 1e-6, False)):
            n_mis += int(np.sum(local_membership(x[sel], y[sel], zz[sel], t) != expect))
    assert n_mis == 0, "%d membership mismatches over %d points" % (n_mis, n_tot)
    # (b) slab chords vs brute force z-sampling
    worst = 0.0
    for _ in range(15):
        t = random_tube(rng, cut=rng.choice(["axial", "vertical"]))
        x = t.c[0] + rng.uniform(-3, 3, 400)
        y = t.c[1] + rng.uniform(-3, 3, 400)
        z_lo, z_hi, ins = G.column_interval(x, y, t)
        idx = np.flatnonzero(ins)[:25]
        dz = 0.05
        for i in idx:
            zetas = G.slab_centres(z_lo[i] - 0.1, z_hi[i] + 0.1, dz)
            for zj in zetas[::3]:
                a = G.slab_absorbance(z_lo[i], z_hi[i], zj, dz, 1.0)
                zs = np.linspace(zj - dz / 2, zj + dz / 2, 20001)
                frac = local_membership(np.full_like(zs, x[i]), np.full_like(zs, y[i]), zs, t).mean()
                worst = max(worst, abs(frac * dz - float(a)))
    assert worst <= 2 * 0.05 / 20000, "slab chord vs brute force: worst %.2e um" % worst
    # (c) line chords vs brute force sampling along the line
    worst = 0.0
    step = 2e-4
    for _ in range(30):
        t = random_tube(rng, cut=rng.choice(["axial", "vertical"]), phi_max_deg=80)
        P = np.asarray(t.c) + rng.uniform(-0.5, 0.5, 3) * t.r
        s = rng.normal(size=3)
        s /= np.linalg.norm(s)
        t1, t2, hit = G.line_interval(P, s, t)
        tt = np.arange(-40.0, 40.0, step)
        pts = P[None, :] + tt[:, None] * s[None, :]
        L_bf = local_membership(pts[:, 0], pts[:, 1], pts[:, 2], t).sum() * step
        worst = max(worst, abs(L_bf - (t2 - t1)))
    assert worst <= 4 * step, "line chord vs brute force: worst %.2e um" % worst
    # (d) the theory chat's own implementation (k = 1)
    sg = load_check_script()
    for _ in range(20):
        t = random_tube(rng, aspect=1.0, cut=rng.choice([None, "vertical"]))
        x = t.c[0] + rng.uniform(-6, 6, 2000)
        y = t.c[1] + rng.uniform(-6, 6, 2000)
        lo_ref, hi_ref, in_ref = sg.ray_interval(x, y, np.asarray(t.c), t.r, t.phi, t.theta, t.half_length)
        lo, hi, ins = G.column_interval(x, y, t)
        assert np.array_equal(in_ref, ins)
        assert_allclose(lo[ins], lo_ref[ins], rtol=0, atol=1e-12)
        assert_allclose(hi[ins], hi_ref[ins], rtol=0, atol=1e-12)
        z = rng.uniform(-8, 8, x.size)
        p = np.stack([x, y, z], -1)
        if t.half_length is None:
            assert np.array_equal(sg.inside_3d(p, np.asarray(t.c), t.r, t.t_hat), G.inside_tube(x, y, z, t))


def test_convergence():
    raise unittest.SkipTest("closed forms; the slab sum is exact at every dzeta (test_invariants)")


def test_invariants():
    rng = make_rng(2)
    for _ in range(25):
        t = random_tube(rng, cut=rng.choice(["axial", "vertical"]))
        x = t.c[0] + rng.uniform(-4, 4, 500)
        y = t.c[1] + rng.uniform(-4, 4, 500)
        z_lo, z_hi, ins = G.column_interval(x, y, t)
        zmin, zmax = G.depth_extent(t)
        assert np.all(z_lo[ins] >= zmin - 1e-12) and np.all(z_hi[ins] <= zmax + 1e-12)
        mu, dz = rng.uniform(0.1, 5.0), 0.05
        zetas = G.slab_centres(zmin, zmax, dz)
        a = np.stack([G.slab_absorbance(z_lo, z_hi, zj, dz, mu) for zj in zetas])
        assert_allclose(a.sum(0), mu * (z_hi - z_lo), rtol=0, atol=1e-12)
        for light in (+1, -1):
            T = np.stack([G.transmitted_before(z_lo, z_hi, zj, dz, mu, light) for zj in zetas])
            if light == +1:
                cum = np.exp(-np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, 0)[:-1]], 0))
            else:
                rev = np.cumsum(a[::-1], 0)[::-1]
                cum = np.exp(-np.concatenate([rev[1:], np.zeros((1, a.shape[1]))], 0))
            assert_allclose(T, cum, rtol=0, atol=1e-12)
            dA = T * (1.0 - np.exp(-a))
            assert_allclose(dA.sum(0), 1.0 - np.exp(-a.sum(0)), rtol=0, atol=1e-12)   # C1
        # the extent is attained at its extreme point
        # (the vertical cut's extreme sits ON the cut |u| = U, so step 1e-9 inside it;
        #  the axial extreme is the cap rim, inside the cut)
        if t.end_cut == "vertical":
            u_star = t.half_length - 1e-9
        else:
            u_star = t.half_length * math.cos(t.phi0) - t.r * math.sin(t.phi0)
        xs = t.c[0] + u_star * math.cos(t.theta)
        ys = t.c[1] + u_star * math.sin(t.theta)
        _, zh, ok = G.column_interval(xs, ys, t)
        tol = 1e-9 * (1.0 + math.tan(t.phi))
        assert ok and abs(float(zh) - zmax) < tol, (t.end_cut, float(zh), zmax)
        # (S3) symmetry: rotate by pi about the vertical through c, theta -> theta + pi
        t2 = G.Tube(t.c, t.r, t.phi, t.theta + math.pi, t.aspect, t.half_length, t.end_cut)
        xr, yr = 2 * t.c[0] - x, 2 * t.c[1] - y
        lo2, hi2, in2 = G.column_interval(xr, yr, t2)
        assert np.array_equal(in2, ins)
        assert_allclose(lo2, z_lo, rtol=0, atol=1e-12)
        assert_allclose(hi2, z_hi, rtol=0, atol=1e-12)


def test_contract():
    t = G.Tube((0.0, 0.0, 0.0), 0.5, math.radians(20), 0.3, 1.0, 5.0, "axial")
    x, y = np.meshgrid(np.linspace(-6, 6, 41), np.linspace(-6, 6, 37))
    z_lo, z_hi, ins = G.column_interval(x, y, t)
    assert z_lo.shape == x.shape and z_hi.shape == x.shape and ins.shape == x.shape
    assert z_lo.dtype == np.float64 and ins.dtype == bool
    assert np.all(np.isfinite(z_lo)) and np.all(np.isfinite(z_hi))
    assert np.all(z_hi - z_lo >= 0) and np.all((z_hi - z_lo)[~ins] == 0)
    assert 0 < ins.sum() < ins.size
    t1, t2, hit = G.line_interval(np.zeros(3), np.random.default_rng(0).normal(size=(50, 3)), t)
    assert t1.shape == (50,) and hit.dtype == bool and np.all(np.isfinite(t1)) and np.all(t2 >= t1)
    zeta = G.slab_centres(-1.0, 1.0, 0.3)
    assert zeta.size == 7 and abs(zeta[0] + 0.85) < 1e-12 and zeta[-1] + 0.15 >= 1.0


def test_determinism():
    raise unittest.SkipTest("deterministic closed forms, no randomness")


def test_edge_cases():
    # phi = 0 with axial caps: the caps are the vertical planes |u| = U
    t = G.Tube((0.0, 0.0, 0.0), 0.5, 0.0, 0.0, 0.7, 3.0, "axial")
    u = np.array([-3.5, -2.9, 0.0, 2.9, 3.5])
    z_lo, z_hi, ins = G.column_interval(u, np.zeros_like(u), t)
    assert list(ins) == [False, True, True, True, False]
    assert_allclose((z_hi - z_lo)[ins], 2 * 0.5 * 0.7, rtol=0, atol=1e-12)
    assert G.depth_extent(t) == (-0.35, 0.35)
    # a tangent line, a line on the surface parallel to the axis, and a missing line do not hit
    t = G.Tube((0.0, 0.0, 0.0), 0.5, 0.0, 0.0)
    for P, s in (((0.0, 0.5, 0.0), (0.0, 0.0, 1.0)), ((0.0, 0.5, 0.0), (1.0, 0.0, 0.0)),
                 ((0.0, 0.6, 0.0), (0.0, 0.0, 1.0))):
        _, _, hit = G.line_interval(np.array(P), np.array(s), t)
        assert not hit, (P, s)
    # a line parallel to an infinite tube's axis, inside it, hits everywhere
    t1, t2, hit = G.line_interval(np.array([0.0, 0.1, 0.0]), np.array([1.0, 0.0, 0.0]), t)
    assert hit and t1 == -math.inf and t2 == math.inf
    # invalid tubes and arguments are refused
    for kw in (dict(r=0.0), dict(phi=math.pi / 2), dict(phi=-0.1), dict(aspect=0.0),
               dict(half_length=0.0), dict(end_cut="slanted")):
        base = dict(c=(0.0, 0.0, 0.0), r=0.5, phi=0.2, theta=0.0, aspect=1.0, half_length=2.0, end_cut="axial")
        base.update(kw)
        try:
            G.Tube(**base)
        except ValueError:
            continue
        raise AssertionError("Tube accepted %r" % (kw,))
    for bad in ((0.0, 1.0, 0.0), (1.0, 0.0, 0.1), (0.0, math.inf, 0.1)):
        try:
            G.slab_centres(*bad)
        except ValueError:
            continue
        raise AssertionError("slab_centres accepted %r" % (bad,))
    try:
        G.transmitted_before(0.0, 1.0, 0.5, 0.1, 1.0, light_direction=0)
    except ValueError:
        pass
    else:
        raise AssertionError("transmitted_before accepted light_direction = 0")
    assert G.depth_extent(G.Tube((0.0, 0.0, 0.0), 0.5, 0.3, 0.0)) == (-math.inf, math.inf)


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
