"""Smoke test for the fast line renderer -- Block 35 in specs/SPEC.md (D-137, D-138).

line_fast.line_profiles is Block 26's line_profiles (Eqs. 5-6 of SPEC section
2.3 read along Block 5's measuring line) with the hybrid's ray history h read
off one cross-section of the infinite tube, cached, instead of being traced
at every slab point; partition_vertical and linear use Block 2's closed forms
(the same as Block 26).

Checks, from the strongest oracle to the weakest
    test_known_answer   a long flat tube read across on a column of pixel
                        centres: sum_n (1 - tau(v_n)) p equals the mass per
                        unit length booked on the fine lattice (Block 26's
                        known answer; hybrid under a near-vertical condenser)
    test_reference      Block 26 on the same tube with the same directions
                        (4 x 8 in both), the line through the tube's middle:
                        d 0.3, 1 um x tilt 0, 30, 60 deg and d 2 um x tilt 0,
                        30 deg, axis depth 0 and 0.14 um, planes -3..3, n_avg 0
                        and the configured value, the three absorptions:
                        max |difference| <= 2e-4 (D-138.1), at half-length
                        10 um (flat) and 40 um (tilted; d 2 um at 30 deg in one
                        configuration), where the caps do not reach (SPEC
                        Block 35, amendment 1 (B), D-140). Prints the time per
                        call (Block 26, fast cold, warm)
    test_caps           reported: the tilted cases at half-length 10, 20, 40 um
    test_cap_margin     reported: the line 1.5, 3, 5 um from a cap
    test_convergence    the section's read of h against Block 21's ray_history
                        at interior points falls as the lattice is refined
    test_invariants     lateral shift by whole pixels; a depth shift reuses
                        the cached section; cached = uncached; mirror symmetry
                        in v; the order of ks
    test_contract       shape, dtype, finiteness; absorption None; the cache
    test_determinism    bit-identical repeat
    test_edge_cases     mu = 0 gives exactly 1; extrusion_valid False and
                        ExtrusionInvalid 0.5 um from a cap (also at mu = 0),
                        True in the middle of a 10 um half-length and at the
                        margin; ValueErrors as Block 26's
    test_directions     reported, no bound: 8 x 16 against 16 x 32 directions
                        on the fast renderer for the cases of test_reference
    test_timing         reported, no bound: time per call at 8 x 16 directions,
                        half-length 5 and 10 um, d 0.3, 1, 2 um, flat and 30
                        deg, cold and warm; Block 26 beside the flat 10 um ones

Run
    python3 tests/smoke/test_smoke_line_fast.py
        (from the code folder "Passive Features/Diameter Re-measurement")
    python3 -m pytest tests/smoke/test_smoke_line_fast.py -q

Debug
    A failed check_close() saves both arrays to
        tests/smoke/artifacts/<check name>.npz

Keep this file pure ASCII so it survives transfer to the cluster.
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
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import profiles as PR  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import hybrid as H  # noqa: E402
from allen_diameter.model import line_fast as LF  # noqa: E402
from allen_diameter.model import line_render as LR  # noqa: E402
from allen_diameter.model import ray_world as RW  # noqa: E402

SEED = 12345  # no randomness in the block; kept for the runner's header
REPORT_PACKAGES = ("numpy", "scipy")
MODES = ("linear", "partition_vertical", "hybrid")
TOL_B26 = 2e-4  # D-138.1 [user]
REF_CASES = ((0.3, 0.0), (0.3, 30.0), (0.3, 60.0), (1.0, 0.0), (1.0, 30.0), (1.0, 60.0), (2.0, 0.0), (2.0, 30.0))
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure."""
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
    """Least-squares slope p of log e(h) = log C + p * log h."""
    h = np.asarray(h, dtype=float)
    err = np.asarray(err, dtype=float)
    if np.any(err <= 0.0):
        raise ValueError("observed_order needs strictly positive errors")
    slope, _intercept = np.polyfit(np.log(h), np.log(err), 1)
    return float(slope)


def make_cfg(n_rho=None, n_psi=None, **renderer):
    """The default configuration; renderer fields replaced; the directions of
    both renderers (renderer.ray_* for Block 26, inference.ray_* for this
    block) set to n_rho x n_psi when given."""
    c = default_config()
    if n_rho is not None:
        renderer = dict(renderer, ray_n_rho=n_rho, ray_n_psi=n_psi)
        c = dataclasses.replace(c, inference=dataclasses.replace(c.inference, ray_n_rho=n_rho, ray_n_psi=n_psi))
    return dataclasses.replace(c, renderer=dataclasses.replace(c.renderer, **renderer))


def line_geometry(theta, origin):
    o = np.asarray(origin, dtype=float)
    return o, np.array([-math.sin(theta), math.cos(theta)]), np.array([math.cos(theta), math.sin(theta)])


def fast_unchecked(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, mode, cache=None):
    """The fast renderer with the validity test bypassed (measurements only)."""
    return LF._profiles(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, mode, cache, check=False)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    # Block 26's known answer (see test_smoke_line_render.py): a tube along x
    # through a pixel centre (phi = 0, U = 10 um), read across at v = n p,
    # n_avg = 0: sum_n (1 - tau(v_n)) p is the mass per unit length of the
    # lattice's sources. The hybrid under a near-vertical condenser has
    # h(P) = exp(-mu (z - z_lo)) / s_z with s_z = 1 to 1e-12; the slabs summed by
    # hand. Tolerance 1e-8 relative for linear and partition (the dropped tails,
    # 6 SD reach, and the profile's ends). The hybrid reads h off the section's
    # bicubic, and z - z_lo = w + sqrt(r^2 - v^2) has a square-root edge at the
    # rim, so its tolerance is the read's own error, 1e-5 relative (the mean
    # read error at 256 intervals, test_convergence).
    p, ks = 0.1144, np.arange(-3, 4)
    d, mu = 0.6, 1.0
    cfg = make_cfg(n_rho=2, n_psi=4, ray_na=1e-6)
    f = cfg.inference.line_fine_factor
    h, r, dzeta = p / f, d / 2, cfg.renderer.dzeta_um
    o, yh, eu = line_geometry(0.0, (437 * p, 211 * p))
    v = np.arange(-60, 61) * p
    tube = G.Tube((o[0], o[1], 0.0), r, 0.0, 0.0, 1.0, 10.0, "axial")
    assert LF.extrusion_valid(tube, ks, o, yh, eu, v, 0, p, cfg)
    iy = np.arange(int(math.floor((o[1] - r + 0.5 * p) / h)) - 10, int(math.ceil((o[1] + r + 0.5 * p) / h)) + 10)
    yl = (iy + 0.5) * h - 0.5 * p - o[1]
    inside = np.abs(yl) <= r
    ell = 2.0 * np.sqrt(np.clip(r * r - yl ** 2, 0.0, None)) * inside
    zlo, zhi = -0.5 * ell, 0.5 * ell
    zeta = G.slab_centres(float(zlo[inside].min()), float(zhi[inside].max()), dzeta)
    hyb_col = np.zeros(yl.size)
    for i in np.flatnonzero(inside):
        for zj in zeta:
            lo, hi = max(zlo[i], zj - 0.5 * dzeta), min(zhi[i], zj + 0.5 * dzeta)
            if hi > lo:
                hyb_col[i] += mu * (hi - lo) * math.exp(-mu * (0.5 * (lo + hi) - zlo[i]))
    expected = {"linear": h * np.sum(mu * ell), "partition_vertical": h * np.sum(-np.expm1(-mu * ell)),
                "hybrid": h * np.sum(hyb_col)}
    for mode in MODES:
        tau = LF.line_profiles(tube, mu, ks, o, yh, eu, v, 0, p, cfg, absorption=mode)
        mass = (1.0 - tau).sum(axis=1) * p
        check_close("known_answer_%s" % mode, mass, np.full(ks.size, expected[mode]),
                    rtol=1e-5 if mode == "hybrid" else 1e-8)
        print("  test_known_answer %s: relative error %.2e" % (mode, float(np.max(np.abs(mass / expected[mode] - 1)))))


def _ref_geometry():
    cfg = make_cfg(n_rho=4, n_psi=8)
    v = PR.profile_offsets(cfg.measure)
    theta = 0.37
    o, yh, eu = line_geometry(theta, (50.03, 40.71))
    return cfg, v, PR.n_along(cfg.measure), cfg.measure.profile_step_um, np.arange(-3, 4), theta, o, yh, eu


def _ref_tube(o, theta, d, phi_deg, cz, U):
    return G.Tube((o[0] + 0.02, o[1] - 0.01, cz), d / 2, math.radians(phi_deg), theta, 1.0, U, "axial")


CAP_U40 = {}  # (d, phi) -> hybrid difference at U 40 um, cz 0, n_avg configured (test_reference, reused by test_caps)


def test_reference():
    # Block 26 as the reference (D-138.1: 2e-4), 4 x 8 directions in both
    # (SPEC Block 35 allows it for time), mu = 1 /um, the line through the
    # tube's middle. Amendment 1 (B), D-140: the block is the tube without
    # caps, so the oracle is judged where the caps do not reach: half-length
    # 10 um for the flat cases, 40 um for the tilted ones; d 2 um at 30 deg in
    # one configuration (cz 0, the configured n_avg), for time.
    cfg, v, A, step, ks, theta, o, yh, eu = _ref_geometry()
    mu = 1.0
    rows, failures = [], []
    for d, phi_deg in REF_CASES:
        U = 10.0 if phi_deg == 0.0 else 40.0
        one = (d == 2.0 and phi_deg == 30.0)
        for cz in ((0.0,) if one else (0.0, 0.14)):
            tube = _ref_tube(o, theta, d, phi_deg, cz, U)
            assert LF.extrusion_valid(tube, ks, o, yh, eu, v, A, step, cfg)
            for mode in MODES:
                err, t = {0: float("nan")}, {}
                for n_avg in ((A,) if one else (0, A)):
                    t0 = time.perf_counter()
                    ref = LR.line_profiles(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, absorption=mode)
                    t["b26"] = time.perf_counter() - t0
                    cache = LF.SectionCache()
                    t0 = time.perf_counter()
                    got = LF.line_profiles(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, absorption=mode, cache=cache)
                    t["cold"] = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    LF.line_profiles(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, absorption=mode, cache=cache)
                    t["warm"] = time.perf_counter() - t0
                    err[n_avg] = float(np.max(np.abs(got - ref)))
                    if err[n_avg] > TOL_B26:
                        failures.append("d %.1f phi %g U %g cz %.2f %s n_avg %d: %.2e" % (d, phi_deg, U, cz, mode,
                                                                                       n_avg, err[n_avg]))
                if mode == "hybrid" and phi_deg > 0 and cz == 0.0:
                    CAP_U40[(d, phi_deg)] = err[A]
                rows.append((d, phi_deg, U, cz, mode, err[0], err[A], t["b26"], t["cold"], t["warm"]))
    print("  test_reference: max |tau_fast - tau_Block26| (4 x 8 directions, mu 1/um, 7 planes, %d offsets);"
          " times at n_avg %d (s)" % (v.size, A))
    print("  %4s %4s %4s %5s %-18s %9s %9s %8s %8s %8s" % ("d", "phi", "U", "cz", "mode", "err_A0",
                                                         "err_A%d" % A, "block26", "cold", "warm"))
    for row in rows:
        print("  %4.1f %4.0f %4.0f %5.2f %-18s %9.2e %9.2e %8.3f %8.3f %8.3f" % row)
    assert not failures, "above 2e-4 (D-138.1) in %d comparisons: %s" % (len(failures), "; ".join(failures))


def test_caps():
    # Reported, no bound (amendment 1 (B)): the hybrid's difference to Block 26
    # for the tilted cases at half-length 10, 20, 40 um (cz 0, n_avg
    # configured; 40 um from test_reference). Validity bypassed at 10 and 20 um
    # is not needed: the line is at the middle.
    cfg, v, A, step, ks, theta, o, yh, eu = _ref_geometry()
    print("  test_caps: max |tau_fast - tau_Block26|, hybrid, tilted, line at the middle, cz 0, n_avg %d" % A)
    for d, phi_deg in REF_CASES:
        if phi_deg == 0.0:
            continue
        out = []
        for U in (10.0, 20.0):
            tube = _ref_tube(o, theta, d, phi_deg, 0.0, U)
            ref = LR.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, step, cfg, absorption="hybrid")
            got = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, step, cfg, absorption="hybrid")
            out.append(float(np.max(np.abs(got - ref))))
        u40 = CAP_U40.get((d, phi_deg), float("nan"))
        print("  d %.1f phi %2.0f: U 10 %.2e, U 20 %.2e, U 40 %.2e" % (d, phi_deg, out[0], out[1], u40))


def test_cap_margin():
    # Reported, no bound (amendment 1 (C)): the hybrid's difference to Block 26
    # with the line 1.5, 3.0 and 5.0 um along the axis from one cap of a tube
    # of half-length 10 um (1.5 um is below the margin: the unchecked path).
    cfg, v, A, step, ks, theta, o, yh, eu = _ref_geometry()
    print("  test_cap_margin: max |tau_fast - tau_Block26|, hybrid, U 10 um, cz 0, n_avg %d, line x um from a cap"
          " (margin %.1f um)" % (A, cfg.inference.extrusion_cap_margin_um))
    for d in (0.3, 1.0):
        for phi_deg in (0.0, 30.0):
            out = []
            for dist in (1.5, 3.0, 5.0):
                tube0 = _ref_tube(o, theta, d, phi_deg, 0.0, 10.0)
                s_in = 10.0                                       # axial cut, aspect 1: |s| <= U
                a = tube0.t_hat
                c = np.array([o[0], o[1], 0.0]) - (s_in - dist) * a  # the line's origin is dist from the cap
                tube = G.Tube((c[0] + 0.02, c[1] - 0.01, c[2]), d / 2, math.radians(phi_deg), theta, 1.0, 10.0,
                              "axial")
                valid = LF.extrusion_valid(tube, ks, o, yh, eu, v, A, step, cfg)
                assert valid == (dist >= cfg.inference.extrusion_cap_margin_um), (d, phi_deg, dist, valid)
                ref = LR.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, step, cfg, absorption="hybrid")
                got = fast_unchecked(tube, 1.0, ks, o, yh, eu, v, A, step, cfg, "hybrid")
                out.append(float(np.max(np.abs(got - ref))))
            print("  d %.1f phi %2.0f: 1.5 um %.2e, 3.0 um %.2e, 5.0 um %.2e" % (d, phi_deg, out[0], out[1], out[2]))


def test_convergence():
    # The section's read of h (bicubic on the lattice) against Block 21's
    # ray_history on the infinite tube at interior points: the mean relative
    # error falls as the lattice is refined (64, 128, 256 intervals). h is
    # smooth inside the section but has square-root edges at the rim (a ray
    # entering tangentially), so the order is below 4; stated, not bounded.
    S, W = RW.directions(8, 16, 1.4, 1.515)
    tube = G.Tube((0.0, 0.0, 0.0), 0.5, math.radians(30.0), 0.37, 1.0, None)
    rng = make_rng()
    e1, e2, b1, b2 = LF._section_basis(tube)
    a = rng.uniform(-1.0, 1.0, (6000, 2))
    a = a[(a ** 2).sum(axis=1) <= 1.0]
    P = (a[:, :1] * b1) * e1 + (a[:, 1:] * b2) * e2 + rng.uniform(-3.0, 3.0, (a.shape[0], 1)) * tube.t_hat
    exact = H.ray_history(P, tube, 1.0, S, W, 1)
    saved = LF.SECTION_NODES
    errs, ns = [], (64, 128, 256)
    try:
        for n in ns:
            LF.SECTION_NODES = n
            sec = LF._build_section(tube, 1.0, S, W, 1)
            errs.append(float(np.mean(np.abs(sec.h(P) / exact - 1.0))))
    finally:
        LF.SECTION_NODES = saved
    order = observed_order(1.0 / np.array(ns, dtype=float), errs)
    print("  test_convergence: mean |h_read / h - 1| %s at %s intervals; observed order %.2f"
          % (", ".join("%.2e" % e for e in errs), list(ns), order))
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[2] < 1e-4, "the production lattice reads h to %.2e on average" % errs[2]


def test_invariants():
    p = 0.1144
    cfg = make_cfg(n_rho=4, n_psi=8)
    v = PR.profile_offsets(cfg.measure)
    A = PR.n_along(cfg.measure)
    ks = np.arange(-3, 4)
    # (1) a lateral shift of the tube and the line by whole pixels: the same
    # lattice, the same values [1e-12]; with a cache the section is reused.
    theta = 0.37
    o, yh, eu = line_geometry(theta, (50.03, 40.71))
    tube = G.Tube((o[0] + 0.02, o[1] - 0.01, 0.0), 0.15, 0.0, theta, 1.0, 10.0, "axial")
    shift = np.array([7 * p, -3 * p])
    tube_s = G.Tube((tube.c[0] + shift[0], tube.c[1] + shift[1], 0.0), 0.15, 0.0, theta, 1.0, 10.0, "axial")
    cache = LF.SectionCache()
    for mode in MODES:
        a = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, p, cfg, absorption=mode, cache=cache)
        b = LF.line_profiles(tube_s, 1.0, ks, o + shift, yh, eu, v, A, p, cfg, absorption=mode, cache=cache)
        check_close("shift_whole_pixels_%s" % mode, b, a, rtol=0.0, atol=1e-12)
    assert cache.misses == 1 and cache.hits == 1, (cache.misses, cache.hits)
    # (2) the tube shifted by 2 pixels along y_hat = (0, 1) (theta = 0), the line
    # kept: the profile's samples (step p) move by 2 [1e-12 on the shared samples]
    o2, yh2, eu2 = line_geometry(0.0, (300 * p, 120 * p))
    t0 = G.Tube((o2[0] + 0.01, o2[1] + 0.02, 0.0), 0.15, 0.0, 0.0, 1.0, 10.0, "axial")
    t2 = G.Tube((o2[0] + 0.01, o2[1] + 0.02 + 2 * p, 0.0), 0.15, 0.0, 0.0, 1.0, 10.0, "axial")
    for mode in MODES:
        a = LF.line_profiles(t0, 1.0, ks, o2, yh2, eu2, v, A, p, cfg, absorption=mode)
        b = LF.line_profiles(t2, 1.0, ks, o2, yh2, eu2, v, A, p, cfg, absorption=mode)
        check_close("shift_along_line_%s" % mode, b[:, 2:], a[:, :-2], rtol=0.0, atol=1e-12)
    # (3) a depth shift reuses the section, and a cached call equals an
    # uncached one [1e-12]
    cache = LF.SectionCache()
    tz = G.Tube((tube.c[0], tube.c[1], 0.14), 0.15, 0.0, theta, 1.0, 10.0, "axial")
    LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, p, cfg, absorption="hybrid", cache=cache)
    warm = LF.line_profiles(tz, 1.0, ks, o, yh, eu, v, A, p, cfg, absorption="hybrid", cache=cache)
    cold = LF.line_profiles(tz, 1.0, ks, o, yh, eu, v, A, p, cfg, absorption="hybrid")
    assert cache.hits == 1 and cache.misses == 1, (cache.hits, cache.misses)
    check_close("cached_vs_uncached", warm, cold, rtol=0.0, atol=1e-12)
    # (4) mirror symmetry in v: a tube along x through a pixel centre read
    # across; lattice, tube and condenser (psi -> -psi, even n_psi) symmetric
    # under y -> -y [1e-10]
    tube_m = G.Tube((o2[0], o2[1], 0.07), 0.4, math.radians(30.0), 0.0, 1.0, 10.0, "axial")
    for mode in MODES:
        tau = fast_unchecked(tube_m, 1.5, ks, o2, yh2, eu2, v, A, p, cfg, mode)
        check_close("mirror_%s" % mode, tau, tau[:, ::-1], rtol=0.0, atol=1e-10)
    # (5) the order of ks: each row is that plane's
    perm = np.array([4, 0, 6, 2, 1, 5, 3])
    for mode in MODES:
        a = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, p, cfg, absorption=mode)
        b = LF.line_profiles(tube, 1.0, ks[perm], o, yh, eu, v, A, p, cfg, absorption=mode)
        assert_array_equal(b, a[perm], err_msg="ks order, %s" % mode)


def test_contract():
    p = 0.1144
    cfg = make_cfg(n_rho=2, n_psi=4)
    v = PR.profile_offsets(cfg.measure)
    A = PR.n_along(cfg.measure)
    o, yh, eu = line_geometry(0.4, (12.0, 9.0))
    tube = G.Tube((12.0, 9.0, 0.0), 0.15, 0.0, 0.4, 1.0, 10.0, "axial")
    ks = [2, -1, 0]
    tau = LF.line_profiles(tube, 0.8, ks, o, yh, eu, v, A, p, cfg)
    assert tau.shape == (3, v.size), tau.shape
    assert tau.dtype == np.float64, tau.dtype
    assert np.all(np.isfinite(tau))
    assert cfg.inference.absorption == "hybrid"
    assert_array_equal(tau, LF.line_profiles(tube, 0.8, ks, o, yh, eu, v, A, p, cfg, absorption="hybrid"))
    assert np.all(tau < 1.0 + 1e-15) and tau.min() < 0.95, "a dip below 1, nothing above"
    assert issubclass(LF.ExtrusionInvalid, ValueError)
    cache = LF.SectionCache(max_entries=2)
    for mu in (0.5, 0.6, 0.7, 0.5):
        LF.line_profiles(tube, mu, ks, o, yh, eu, v, A, p, cfg, cache=cache)
    assert (cache.misses, cache.hits, len(cache)) == (4, 0, 2), (cache.misses, cache.hits, len(cache))
    LF.line_profiles(tube, 0.5, ks, o, yh, eu, v, A, p, cfg, cache=cache)
    assert cache.hits == 1
    assert LF.line_profiles(tube, 0.8, [], o, yh, eu, v, A, p, cfg).shape == (0, v.size)


def test_determinism():
    p = 0.1144
    cfg = make_cfg(n_rho=4, n_psi=8)
    v = PR.profile_offsets(cfg.measure)
    o, yh, eu = line_geometry(0.9, (20.0, 20.0))
    tube = G.Tube((20.0, 20.1, 0.05), 0.25, 0.4, 0.9, 1.0, 10.0, "axial")
    for mode in MODES:
        a = fast_unchecked(tube, 1.2, np.arange(-3, 4), o, yh, eu, v, 4, p, cfg, mode)
        b = fast_unchecked(tube, 1.2, np.arange(-3, 4), o, yh, eu, v, 4, p, cfg, mode)
        assert_array_equal(a, b, err_msg="repeat, %s" % mode)


def test_edge_cases():
    p = 0.1144
    cfg = make_cfg(n_rho=2, n_psi=4)
    v = PR.profile_offsets(cfg.measure)
    A = PR.n_along(cfg.measure)
    o, yh, eu = line_geometry(0.3, (10.0, 10.0))
    ks = np.arange(-3, 4)
    # the middle of a 10 um half-length: valid (d 0.3 um, flat)
    tube = G.Tube((10.0, 10.0, 0.0), 0.15, 0.0, 0.3, 1.0, 10.0, "axial")
    assert LF.extrusion_valid(tube, ks, o, yh, eu, v, A, p, cfg)
    for mode in MODES:
        # mu = 0: exactly 1 (SPEC Block 35)
        assert_array_equal(LF.line_profiles(tube, 0.0, ks, o, yh, eu, v, A, p, cfg, absorption=mode),
                           np.ones((ks.size, v.size)))
    # a line 0.5 um from a cap: not valid, ExtrusionInvalid (a ValueError)
    o_cap = o + (10.0 - 0.5) * eu
    assert not LF.extrusion_valid(tube, ks, o_cap, yh, eu, v, A, p, cfg)
    for mode in MODES:
        try:
            LF.line_profiles(tube, 1.0, ks, o_cap, yh, eu, v, A, p, cfg, absorption=mode)
        except LF.ExtrusionInvalid:
            pass
        else:
            raise AssertionError("no ExtrusionInvalid 0.5 um from a cap (%s)" % mode)
    try:  # checked before the mu = 0 shortcut (amendment 1 (C))
        LF.line_profiles(tube, 0.0, ks, o_cap, yh, eu, v, A, p, cfg)
    except LF.ExtrusionInvalid:
        pass
    else:
        raise AssertionError("no ExtrusionInvalid at mu = 0 0.5 um from a cap")
    m = cfg.inference.extrusion_cap_margin_um
    assert LF.extrusion_valid(tube, ks, o + (10.0 - m - 1e-9) * eu, yh, eu, v, A, p, cfg)
    assert not LF.extrusion_valid(tube, ks, o + (10.0 - m + 1e-6) * eu, yh, eu, v, A, p, cfg)
    # a tilted tube: the distance is along the axis (observed arc length)
    tilt = G.Tube((10.0, 10.0, 0.0), 0.15, math.radians(45.0), 0.3, 1.0, 10.0, "axial")
    s_at = 10.0 - m - 0.01
    assert LF.extrusion_valid(tilt, ks, o + s_at * math.cos(math.radians(45.0)) * eu, yh, eu, v, A, p, cfg)
    assert not LF.extrusion_valid(tilt, ks, o + (s_at + 0.02) * math.cos(math.radians(45.0)) * eu, yh, eu, v, A,
                                  p, cfg)
    bad = [
        dict(mu=-0.1), dict(mu=float("nan")), dict(y_hat=np.array([0.0, 1.01])),
        dict(e_u=np.array([0.5, 0.5])), dict(absorption="ray_world"), dict(absorption="unknown"),
        dict(tube=G.Tube((10.0, 10.0, 0.0), 0.15, 0.0, 0.3)), dict(cache="not a cache"),
    ]
    for change in bad:
        args = dict(tube=tube, mu=1.0, y_hat=yh, e_u=eu, absorption=None, cache=None)
        args.update(change)
        try:
            LF.line_profiles(args["tube"], args["mu"], ks, o, args["y_hat"], args["e_u"], v, A, p, cfg,
                             absorption=args["absorption"], cache=args["cache"])
        except ValueError:
            continue
        raise AssertionError("no ValueError for %r" % (change,))
    try:
        LF.SectionCache(max_entries=0)
    except ValueError:
        pass
    else:
        raise AssertionError("SectionCache(max_entries=0) accepted")


def test_directions():
    # Reported, no bound (SPEC Block 35): the fast renderer at 8 x 16 against
    # 16 x 32 directions, hybrid (the other two do not use directions), the
    # cases of test_reference at half-length 10 um.
    c8, c16 = make_cfg(n_rho=8, n_psi=16), make_cfg(n_rho=16, n_psi=32)
    v = PR.profile_offsets(c8.measure)
    A = PR.n_along(c8.measure)
    step = c8.measure.profile_step_um
    ks = np.arange(-3, 4)
    theta = 0.37
    o, yh, eu = line_geometry(theta, (50.03, 40.71))
    print("  test_directions: max |tau(8 x 16) - tau(16 x 32)|, fast renderer, hybrid, mu 1/um, U 10 um, n_avg %d" % A)
    for d, phi_deg in REF_CASES:
        out = []
        for cz in (0.0, 0.14):
            tube = G.Tube((o[0] + 0.02, o[1] - 0.01, cz), d / 2, math.radians(phi_deg), theta, 1.0, 10.0, "axial")
            a = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, step, c8, absorption="hybrid")
            b = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, step, c16, absorption="hybrid")
            out.append(float(np.max(np.abs(a - b))))
        print("  d %.1f phi %2.0f: cz 0 %.2e, cz 0.14 %.2e" % (d, phi_deg, out[0], out[1]))


def test_timing():
    # Reported, no bound: time per call at the production directions (8 x 16),
    # half-length 5 and 10 um, flat and 30 deg, hybrid, n_avg configured, the
    # line at the middle. Cold = new (d, mu); warm = the cached section (a depth
    # shift of 0.05 um). Block 26 at the same directions beside it for the flat
    # tubes at 10 um.
    cfg = make_cfg(n_rho=8, n_psi=16)
    v = PR.profile_offsets(cfg.measure)
    A = PR.n_along(cfg.measure)
    step = cfg.measure.profile_step_um
    ks = np.arange(-3, 4)
    theta = 0.37
    o, yh, eu = line_geometry(theta, (50.03, 40.71))
    print("  test_timing: s per call, 8 x 16 directions, hybrid, mu 1/um, 7 planes, n_avg %d" % A)
    print("  %4s %4s %4s %9s %9s %9s %9s" % ("d", "phi", "U", "cold", "warm", "block26", "err"))
    for U in (5.0, 10.0):
        for phi_deg in (0.0, 30.0):
            for d in (0.3, 1.0, 2.0):
                tube = _ref_tube(o, theta, d, phi_deg, 0.0, U)
                tz = _ref_tube(o, theta, d, phi_deg, 0.05, U)
                cache = LF.SectionCache()
                t0 = time.perf_counter()
                got = LF.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, step, cfg, absorption="hybrid", cache=cache)
                t_cold = time.perf_counter() - t0
                t0 = time.perf_counter()
                LF.line_profiles(tz, 1.0, ks, o, yh, eu, v, A, step, cfg, absorption="hybrid", cache=cache)
                t_warm = time.perf_counter() - t0
                t_b26, err = float("nan"), float("nan")
                if U == 10.0 and phi_deg == 0.0:
                    t0 = time.perf_counter()
                    ref = LR.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, step, cfg, absorption="hybrid")
                    t_b26 = time.perf_counter() - t0
                    err = float(np.max(np.abs(got - ref)))
                print("  %4.1f %4.0f %4.0f %9.2f %9.2f %9.2f %9.2e" % (d, phi_deg, U, t_cold, t_warm, t_b26, err))


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
