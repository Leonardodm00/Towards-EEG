"""Smoke test for the line renderer -- Block 26 in specs/SPEC.md (D-129.1, D-130.O10).

line_render.line_profiles evaluates Eqs. 5-6 of SPEC.md section 2.3 (one tube;
absorption linear, partition_vertical or hybrid) only where the measuring line
of Block 5 reads it: the pixel means of tau_k, read by
profiles.sample_profile (bilinear, mean over 2 n_avg + 1 lines).

Checks, from the strongest oracle to the weakest
    test_known_answer   a long flat tube read across on a column of pixel
                        centres: sum_n (1 - tau(v_n)) p equals the mass per
                        unit length booked on the fine lattice, h sum_y of the
                        column totals (linear: mu l; partition: 1 - exp(-mu l)
                        (C1); hybrid under a near-vertical condenser: the
                        slabs' a_j exp(-mu (zbar_j - z_lo)) summed by hand)
    test_reference      Block 4 (render_transmittance on a fine grid holding
                        the tube, pixel_integrate, sample_profile): d 0.3, 1,
                        2 um x tilt 0, 30, 60 deg x axis depth 0, 0.14 um,
                        planes -3..3, n_avg 0 and the configured value, the
                        three absorptions: max |difference| <= 1e-3; prints
                        the time per call of both
    test_convergence    the difference to Block 4 at the same fine factor
                        falls at order 2 in the factor (both integrate a
                        pixel; Block 4 by point samples, the midpoint rule)
    test_invariants     mirror symmetry in v; independence of the order of
                        ks; every pixel the read touches is computed
    test_contract       shape (n_planes, n_v), float64, finite; absorption
                        None = cfg.inference.absorption; pixels_needed
    test_determinism    bit-identical repeat (no randomness)
    test_edge_cases     mu = 0 and a tube beyond reach give exactly 1;
                        ValueError for mu < 0, a non-unit axis, an unknown
                        absorption, an uncapped tube

Test configuration (stated where it differs from the production defaults):
the reference and convergence checks use U_um = 2 um (Block 4 must hold the
whole tube on its grid for a like-for-like comparison) and 4 x 8 condenser
directions (both renderers take the same directions from the configuration;
the comparison is between two evaluations of one model, not a test of the
quadrature, which Block 21 checks).

Run
    python3 tests/smoke/test_smoke_line_render.py
        (from the code folder "Passive Features/Diameter Re-measurement")
        Prints the environment, the timing table and a PASS / FAIL / ERROR /
        TODO / SKIP table, then the full message of every failure. Exit code 1
        if any check failed, errored or is still TODO.
    python3 -m pytest tests/smoke/test_smoke_line_render.py -q

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
from allen_diameter.model import line_render as LR  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402

SEED = 12345  # no randomness in the block; kept for the runner's header
REPORT_PACKAGES = ("numpy", "scipy")
MODES = ("linear", "partition_vertical", "hybrid")
TIMINGS = []  # rows of test_reference's timing table
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run


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


# ---------------------------------------------------------------- checks ---

class _Frame(object):
    def __init__(self, left, top, res_um_px):
        self.left, self.top, self.res_um_px = left, top, res_um_px


def make_cfg(**renderer):
    """The default configuration with renderer fields replaced (see the docstring)."""
    c = default_config()
    return dataclasses.replace(c, renderer=dataclasses.replace(c.renderer, **renderer))


def with_fine(cfg, f):
    """Block 4's fine factor and the line renderer's both set to f samples per pixel side."""
    p = cfg.acquisition.res0_um
    return dataclasses.replace(cfg, renderer=dataclasses.replace(cfg.renderer, h_g_um_thin=p / f, h_g_um_thick=p / f),
                               inference=dataclasses.replace(cfg.inference, line_fine_factor=f))


def line_geometry(theta, origin):
    o = np.asarray(origin, dtype=float)
    return o, np.array([-math.sin(theta), math.cos(theta)]), np.array([math.cos(theta), math.sin(theta)])


def block4_profiles(tube, mu, ks, o, yh, eu, v, n_avgs, step, cfg, mode):
    """Block 4 on a pixel-aligned fine grid that holds the whole tube and every pixel
    the read touches; pixel_integrate; sample_profile. -> {n_avg: (n_planes, n_v)}."""
    p = cfg.acquisition.res0_um
    f = R.fine_factor(tube.d, cfg.renderer, p)
    assert f == cfg.inference.line_fine_factor, "the comparison needs one fine lattice"
    cols, rows = [], []
    for A in n_avgs:
        c, r = LR.pixels_needed(o, yh, eu, v, A, step, p)
        cols.append(c)
        rows.append(r)
    cols, rows = np.concatenate(cols), np.concatenate(rows)
    Eu, Ev = G.lateral_half_extent(tube)
    hx = abs(Eu * math.cos(tube.theta)) + abs(Ev * math.sin(tube.theta))
    hy = abs(Eu * math.sin(tube.theta)) + abs(Ev * math.cos(tube.theta))
    left = min(int(cols.min()), int(math.floor((tube.c[0] - hx) / p))) - 2
    right = max(int(cols.max()), int(math.ceil((tube.c[0] + hx) / p))) + 2
    top = min(int(rows.min()), int(math.floor((tube.c[1] - hy) / p))) - 2
    bottom = max(int(rows.max()), int(math.ceil((tube.c[1] + hy) / p))) + 2
    grid, _inner = R.block_fine_grid(left, top, right - left + 1, bottom - top + 1, p, f, 0)
    rc = dataclasses.replace(cfg.renderer, absorption=mode)
    res = R.render_transmittance(tube, mu, np.asarray(ks) * cfg.acquisition.dz_um, grid, rc,
                                 cfg.acquisition.light_direction, reduce=lambda pl: R.pixel_integrate(pl, f))
    fr = _Frame(left, top, p)
    out = {}
    for A in n_avgs:
        out[A] = np.stack([PR.sample_profile(res.tau[i], fr, o, yh, eu, v, A, step) for i in range(len(ks))])
    return out


def test_known_answer():
    # A tube along x (theta = 0, phi = 0, U = 10 um) through a pixel centre, read
    # across (y_hat = (0, 1)) at v = n p, n_avg = 0: the samples are pixel
    # centres, the read is exact, and sum_n (1 - tau(v_n)) p is the integral
    # over y of the pixel means, i.e. the mass per unit length of the lattice's
    # sources (the x-sum of the pixel kernel over a lattice of step h is 1 to
    # exp(-2 pi^2 sigma^2 / h^2): spectral). Expected values from the fine
    # lattice's columns, written here by hand. Tolerance 1e-8 relative: the
    # dropped tails (6 SD reach, 1e-9) and the profile's ends (8 SD).
    p, ks = 0.1144, np.arange(-3, 4)
    d, mu = 0.6, 1.0
    cfg = make_cfg(ray_na=1e-6, ray_n_rho=2, ray_n_psi=4)       # near-vertical condenser for the hybrid
    f = cfg.inference.line_fine_factor
    h, r, dzeta = p / f, d / 2, cfg.renderer.dzeta_um
    o, yh, eu = line_geometry(0.0, (437 * p, 211 * p))
    v = np.arange(-60, 61) * p
    tube = G.Tube((o[0], o[1], 0.0), r, 0.0, 0.0, 1.0, cfg.renderer.U_um, cfg.renderer.end_cut)
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
        tau = LR.line_profiles(tube, mu, ks, o, yh, eu, v, 0, p, cfg, absorption=mode)
        mass = (1.0 - tau).sum(axis=1) * p
        check_close("known_answer_%s" % mode, mass, np.full(ks.size, expected[mode]), rtol=1e-8)


def test_reference():
    # Block 4 as the reference, tolerance 1e-3 of the background (SPEC Block 26).
    cfg = make_cfg(U_um=2.0, ray_n_rho=4, ray_n_psi=8)
    v = PR.profile_offsets(cfg.measure)
    A = PR.n_along(cfg.measure)
    step = cfg.measure.profile_step_um
    ks = np.arange(-3, 4)
    theta = 0.37
    o, yh, eu = line_geometry(theta, (50.03, 40.71))
    mu = 1.0
    worst = {m: 0.0 for m in MODES}
    del TIMINGS[:]
    for d in (0.3, 1.0, 2.0):
        for phi_deg in (0.0, 30.0, 60.0):
            for cz in (0.0, 0.14):
                tube = G.Tube((o[0] + 0.02, o[1] - 0.01, cz), d / 2, math.radians(phi_deg), theta, 1.0,
                              cfg.renderer.U_um, cfg.renderer.end_cut)
                for mode in MODES:
                    t0 = time.perf_counter()
                    ref = block4_profiles(tube, mu, ks, o, yh, eu, v, (0, A), step, cfg, mode)
                    t_b4 = time.perf_counter() - t0
                    t_line = {}
                    for n_avg in (0, A):
                        t0 = time.perf_counter()
                        got = LR.line_profiles(tube, mu, ks, o, yh, eu, v, n_avg, step, cfg, absorption=mode)
                        t_line[n_avg] = time.perf_counter() - t0
                        err = float(np.max(np.abs(got - ref[n_avg])))
                        worst[mode] = max(worst[mode], err)
                        check_close("reference_d%g_phi%g_cz%g_%s_navg%d" % (d, phi_deg, cz, mode, n_avg),
                                    got, ref[n_avg], rtol=0.0, atol=1e-3)
                    TIMINGS.append((d, phi_deg, cz, mode, t_line[A], t_line[0], t_b4, err))
    print("  test_reference: largest |tau_line - tau_Block4| per mode: %s"
          % ", ".join("%s %.2e" % (m, worst[m]) for m in MODES))
    print("  time per call (s), test configuration (U 2 um, 4 x 8 directions, 7 planes, 53 offsets):")
    print("  %5s %5s %5s %-18s %9s %9s %9s %9s" % ("d", "phi", "cz", "mode", "line_A%d" % A, "line_A0",
                                                   "block4", "err_A%d" % A))
    for row in TIMINGS:
        print("  %5.1f %5.0f %5.2f %-18s %9.3f %9.3f %9.3f %9.2e" % row)


def test_convergence():
    # Both renderers sample the object on the same lattice; they differ in the
    # pixel integral (exact here, f x f point samples in Block 4: the midpoint
    # rule, error O(h^2)). The difference must fall at order 2 in h = p / f.
    cfg0 = make_cfg(U_um=2.0)
    v = PR.profile_offsets(cfg0.measure)
    A = PR.n_along(cfg0.measure)
    step = cfg0.measure.profile_step_um
    ks = np.arange(-3, 4)
    theta = 0.37
    o, yh, eu = line_geometry(theta, (50.03, 40.71))
    tube = G.Tube((o[0] + 0.02, o[1] - 0.01, 0.0), 0.5, math.radians(30.0), theta, 1.0, 2.0, "axial")
    fs = np.array([4, 8, 16])
    for mode in ("linear", "partition_vertical"):
        errs = []
        for f in fs:
            cfg = with_fine(cfg0, int(f))
            ref = block4_profiles(tube, 1.0, ks, o, yh, eu, v, (A,), step, cfg, mode)[A]
            got = LR.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, step, cfg, absorption=mode)
            errs.append(float(np.max(np.abs(got - ref))))
        order = observed_order(cfg0.acquisition.res0_um / fs, errs)
        print("  test_convergence %s: max diff %s at f = %s; observed order %.2f"
              % (mode, ", ".join("%.2e" % e for e in errs), list(fs), order))
        assert abs(order - 2.0) < 0.3, "observed order %.3f (expected 2)" % order


def test_invariants():
    p = 0.1144
    cfg = make_cfg(U_um=3.0, ray_n_rho=4, ray_n_psi=8)
    v = PR.profile_offsets(cfg.measure)
    A = PR.n_along(cfg.measure)
    # (1) mirror symmetry in v: a tube along x through a pixel centre, read
    # across; the lattice, the tube and the condenser (psi -> -psi, even
    # n_psi) are symmetric under y -> -y. 1e-10: roundoff of sums over ~1e5 terms.
    o, yh, eu = line_geometry(0.0, (300 * p, 120 * p))
    tube = G.Tube((o[0], o[1], 0.07), 0.4, math.radians(30.0), 0.0, 1.0, 3.0, "axial")
    for mode in MODES:
        tau = LR.line_profiles(tube, 1.5, np.arange(-3, 4), o, yh, eu, v, A, p, cfg, absorption=mode)
        check_close("mirror_%s" % mode, tau, tau[:, ::-1], rtol=0.0, atol=1e-10)
    # (2) the order of ks: each row is that plane's, whatever the order
    theta = 1.1
    o, yh, eu = line_geometry(theta, (31.3, 17.9))
    tube = G.Tube((o[0] + 0.05, o[1], 0.1), 0.35, math.radians(40.0), theta, 1.0, 3.0, "axial")
    ks = np.array([-3, -2, -1, 0, 1, 2, 3])
    perm = np.array([4, 0, 6, 2, 1, 5, 3])
    for mode in MODES:
        a = LR.line_profiles(tube, 1.0, ks, o, yh, eu, v, A, p, cfg, absorption=mode)
        b = LR.line_profiles(tube, 1.0, ks[perm], o, yh, eu, v, A, p, cfg, absorption=mode)
        assert_array_equal(b, a[perm], err_msg="ks order, %s" % mode)
    # (3) pixels_needed holds every pixel the bilinear read touches: a plane
    # that is NaN outside them reads finite everywhere (a sample touching a
    # missing pixel would read NaN).
    for A_ in (0, A):
        cols, rows = LR.pixels_needed(o, yh, eu, v, A_, p, p)
        left, top = int(cols.min()) - 3, int(rows.min()) - 3
        plane = np.full((int(rows.max()) - top + 4, int(cols.max()) - left + 4), np.nan)
        plane[rows - top, cols - left] = 1.0
        read = PR.sample_profile(plane, _Frame(left, top, p), o, yh, eu, v, A_, p)
        assert np.all(np.isfinite(read)), "pixels_needed misses a pixel the read touches (n_avg %d)" % A_
        assert np.unique(np.stack([cols, rows]), axis=1).shape[1] == cols.size, "pixels_needed repeats a pixel"


def test_contract():
    p = 0.1144
    cfg = make_cfg(U_um=2.0, ray_n_rho=2, ray_n_psi=4)
    v = PR.profile_offsets(cfg.measure)
    A = PR.n_along(cfg.measure)
    o, yh, eu = line_geometry(0.4, (12.0, 9.0))
    tube = G.Tube((12.0, 9.0, 0.0), 0.3, 0.2, 0.4, 1.0, 2.0, "axial")
    ks = [2, -1, 0]
    tau = LR.line_profiles(tube, 0.8, ks, o, yh, eu, v, A, p, cfg)
    assert tau.shape == (3, v.size), tau.shape
    assert tau.dtype == np.float64, tau.dtype
    assert np.all(np.isfinite(tau))
    assert cfg.inference.absorption == "hybrid"
    assert_array_equal(tau, LR.line_profiles(tube, 0.8, ks, o, yh, eu, v, A, p, cfg, absorption="hybrid"))
    assert np.all(tau < 1.0 + 1e-15) and tau.min() < 0.95, "a dip below 1, nothing above"
    cols, rows = LR.pixels_needed(o, yh, eu, v, A, p, p)
    assert cols.dtype.kind == "i" and rows.dtype.kind == "i" and cols.shape == rows.shape and cols.ndim == 1


def test_determinism():
    p = 0.1144
    cfg = make_cfg(U_um=2.0, ray_n_rho=4, ray_n_psi=8)
    v = PR.profile_offsets(cfg.measure)
    o, yh, eu = line_geometry(0.9, (20.0, 20.0))
    tube = G.Tube((20.0, 20.1, 0.05), 0.5, 0.4, 0.9, 1.0, 2.0, "axial")
    for mode in MODES:
        a = LR.line_profiles(tube, 1.2, np.arange(-3, 4), o, yh, eu, v, 4, p, cfg, absorption=mode)
        b = LR.line_profiles(tube, 1.2, np.arange(-3, 4), o, yh, eu, v, 4, p, cfg, absorption=mode)
        assert_array_equal(a, b, err_msg="repeat, %s" % mode)


def test_edge_cases():
    p = 0.1144
    cfg = make_cfg(U_um=2.0, ray_n_rho=2, ray_n_psi=4)
    v = PR.profile_offsets(cfg.measure)
    o, yh, eu = line_geometry(0.3, (10.0, 10.0))
    tube = G.Tube((10.0, 10.0, 0.0), 0.5, 0.3, 0.3, 1.0, 2.0, "axial")
    ks = np.arange(-3, 4)
    for mode in MODES:
        # mu = 0: exactly 1 (SPEC Block 26)
        assert_array_equal(LR.line_profiles(tube, 0.0, ks, o, yh, eu, v, 4, p, cfg, absorption=mode),
                           np.ones((ks.size, v.size)))
        # a tube beyond every slab's reach (40 um away): exactly 1
        far = G.Tube((50.0, 10.0, 0.0), 0.5, 0.3, 0.3, 1.0, 2.0, "axial")
        assert_array_equal(LR.line_profiles(far, 1.0, ks, o, yh, eu, v, 4, p, cfg, absorption=mode),
                           np.ones((ks.size, v.size)))
    bad = [
        dict(mu=-0.1), dict(mu=float("nan")), dict(y_hat=np.array([0.0, 1.01])),
        dict(e_u=np.array([0.5, 0.5])), dict(absorption="ray_world"), dict(absorption="unknown"),
        dict(tube=G.Tube((10.0, 10.0, 0.0), 0.5, 0.0, 0.3)),
    ]
    for change in bad:
        args = dict(tube=tube, mu=1.0, y_hat=yh, e_u=eu, absorption=None)
        args.update(change)
        try:
            LR.line_profiles(args["tube"], args["mu"], ks, o, args["y_hat"], args["e_u"], v, 4, p, cfg,
                             absorption=args["absorption"])
        except ValueError:
            continue
        raise AssertionError("no ValueError for %r" % (change,))


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
