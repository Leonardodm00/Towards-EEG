"""Smoke test for the image-based clearness screen -- Block 40 in specs/SPEC.md (D-194 [orchestrator]).

Checks
    test_known_answer   dip_features on a Gaussian dip (W = 2 sqrt(2 ln 2) s, S = 0, E = 1/16 at one width),
                        on a dip of two half-Gaussians (the skew by hand), on a dip with a second dip aside
                        (E = the second dip's height over the first's)
    test_reference      strip_profiles against profiles.sample_profile on the moved line, and against the
                        analytic band B - A g(v) within the bilinear read's error bound; screen_node with
                        and without the recorded line given
    test_convergence    the half-depth width of a Gaussian dip against the sampling step: order 2
    test_invariants     mirror and scale of a profile; a clean straight band is judged clean; a parallel
                        neighbour, a blob on the edge beside the line, a crossing branch, a taper and a bend
                        are each rejected for the stated reason; 200 seeded noisy clean bands
    test_contract       shapes, dtypes, keys, the judged planes, the offsets along the axis, inputs unchanged
    test_determinism    two identical calls are identical
    test_edge_cases     a dip that never returns on one side; a faint dip; k* outside the planes; NaN in the
                        line; lines that leave the patch; no line along the axis but the measuring line;
                        the refusals

Run (from the code folder "Passive Features/Diameter Re-measurement")
    python3 tests/smoke/test_smoke_clearness.py
    python3 -m pytest tests/smoke/test_smoke_clearness.py -q

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import platform
import sys
import time
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from allen_diameter.analysis import clearness as CL  # noqa: E402
from allen_diameter.analysis import profiles  # noqa: E402
from allen_diameter.config import ClearnessConfig  # noqa: E402

SEED = 20261119
REPORT_PACKAGES = ("numpy", "scipy")
_SAVED_ARTIFACTS = []
P_UM = 0.1144                               # um per pixel, the pilot's
V = P_UM * np.arange(-26, 27)               # um, the pilot's line: 53 samples
N_AVG, STEP = 4, P_UM                       # the pilot's read: 9 lines, one pixel apart
B_GL, A_GL, S0 = 200.0, 60.0, 0.35          # gl, gl, um: background, in-focus depth, in-focus SD of the band
LEFT, TOP, HW = 1000, 2000, 90              # the patch's corner (full-resolution px) and half-size (px)
THETA = 0.4                                 # rad, the axis' direction in the image
KS = np.arange(100, 109)                    # nine planes, k* = 104
K_STAR = 104
CFG = ClearnessConfig()
G_HALF = math.sqrt(2.0 * math.log(2.0))     # a Gaussian falls to 1/2 at this many SDs


def _geometry():
    e_u = np.array([math.cos(THETA), math.sin(THETA)])
    y_hat = np.array([-math.sin(THETA), math.cos(THETA)])
    origin = np.array([(LEFT + HW) * P_UM, (TOP + HW) * P_UM])
    return origin, y_hat, e_u


def _coords():
    """(u, v) in um of every pixel of the patch about the line's origin."""
    origin, y_hat, e_u = _geometry()
    rows, cols = np.mgrid[0:2 * HW + 1, 0:2 * HW + 1]
    dx, dy = (LEFT + cols) * P_UM - origin[0], (TOP + rows) * P_UM - origin[1]
    return dx * e_u[0] + dy * e_u[1], dx * y_hat[0] + dy * y_hat[1]


def band_block(extra=None, width=None, centre=None):
    """Nine planes of a straight dark band B - A_k exp(-(v - c(u))^2 / (2 s_k(u)^2)), s_k^2 = s(u)^2 +
    (0.3 |k - k*|)^2 and A_k = A s(u) / s_k (the band keeps its integral across the line); width(u) -> s (um)
    and centre(u) -> c (um) default to S0 and 0; extra(u, v) -> gl is subtracted from every plane."""
    u, v = _coords()
    s_u = np.full(u.shape, S0) if width is None else width(u)
    c_u = np.zeros(u.shape) if centre is None else centre(u)
    planes = []
    for k in KS:
        s_k = np.sqrt(s_u ** 2 + (0.3 * abs(int(k) - K_STAR)) ** 2)
        img = B_GL - A_GL * (s_u / s_k) * np.exp(-0.5 * ((v - c_u) / s_k) ** 2)
        if extra is not None:
            img = img - extra(u, v)
        planes.append(img)
    return np.stack(planes)


def screen(block, sigma=1.0, cfg=CFG, k_star=K_STAR, profiles_gl=None):
    origin, y_hat, e_u = _geometry()
    return CL.screen_node(block, KS, k_star, np.full(KS.size, B_GL), sigma, LEFT, TOP, P_UM, origin, y_hat, e_u,
                          V, N_AVG, STEP, cfg, profiles_gl=profiles_gl)


def _raises(exc, func, *args, **kw):
    try:
        func(*args, **kw)
    except exc:
        return
    raise AssertionError("%s not raised by %s" % (exc.__name__, getattr(func, "__name__", func)))


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    h = 0.001
    v = h * np.arange(-3000, 3001)              # a fine line, so that the sampling does not enter
    s, c0, A = 0.30, 0.137, 50.0
    f = CL.dip_features(A * np.exp(-0.5 * ((v - c0) / s) ** 2), v)
    # linear interpolation on a grid of 0.001 um: the crossing errs by about h^2 |D''| / (8 |D'|) < 1e-6 um
    assert_allclose(f["fwhm_um"], 2.0 * G_HALF * s, atol=1e-5)
    assert_allclose(f["centre_um"], c0, atol=1e-5)
    assert_allclose(f["depth_gl"], A, rtol=1e-5)          # the extreme lies within h / 2 of a sample
    assert abs(f["skew"]) < 1e-5
    # one width from the centre a Gaussian is at exp(-(2 g s)^2 / (2 s^2)) = 2^-4; the first sample outside lies
    # at most h beyond it, where the dip is lower by at most h W / s^2 = 0.8 % of itself
    assert_allclose(f["outside"], 1.0 / 16.0, rtol=1e-2)
    assert f["outside"] <= 1.0 / 16.0 + 1e-9

    # two half-Gaussians s_L, s_R about c0: x_L(f) = c0 - s_L g(f), x_R(f) = c0 + s_R g(f), g(f) = sqrt(2 ln(1/f))
    sL, sR = 0.25, 0.45
    D = A * np.exp(-0.5 * ((v - c0) / np.where(v < c0, sL, sR)) ** 2)
    f = CL.dip_features(D, v)
    g = lambda fr: math.sqrt(2.0 * math.log(1.0 / fr))    # noqa: E731
    W = (sL + sR) * g(0.5)
    S = 0.5 * (sR - sL) * max(abs(g(0.75) - g(0.5)), abs(g(0.25) - g(0.5))) / W
    assert_allclose(f["fwhm_um"], W, atol=1e-5)
    assert_allclose(f["centre_um"], c0 + 0.5 * (sR - sL) * g(0.5), atol=1e-5)
    assert_allclose(f["skew"], S, atol=1e-5)

    # a second dip of 0.4 A, 1.5 um aside (6 SDs of the first, whose own tail there is 1.5e-8 A): E = 0.4
    D = A * np.exp(-0.5 * (v / 0.25) ** 2) + 0.4 * A * np.exp(-0.5 * ((v - 1.5) / 0.25) ** 2)
    f = CL.dip_features(D, v)
    assert_allclose(f["outside"], 0.4, atol=1e-6)
    assert_allclose(f["fwhm_um"], 2.0 * G_HALF * 0.25, atol=1e-5)


def test_reference():
    origin, y_hat, e_u = _geometry()
    block = band_block()
    frame = SimpleNamespace(left=LEFT, top=TOP, res_um_px=P_UM)
    u = np.array([-2.0, -0.7, 0.0, 1.3])
    got = CL.strip_profiles(block[4], LEFT, TOP, P_UM, origin, y_hat, e_u, V, N_AVG, STEP, u)
    assert got.shape == (4, V.size)
    for i, uu in enumerate(u):                  # the library read on the moved line
        ref = profiles.sample_profile(block[4], frame, origin + uu * e_u, y_hat, e_u, V, N_AVG, STEP)
        assert_array_equal(got[i], ref)
    # the band does not change along the axis, so every line reads B - A g(v) up to the bilinear read's
    # error, at most (p^2 / 8) max|D''| per image direction = 2 x (0.1144^2 / 8) x A / s^2 = 1.6 gl
    bound = 2.0 * (P_UM ** 2 / 8.0) * A_GL / S0 ** 2
    assert_allclose(got, np.broadcast_to(B_GL - A_GL * np.exp(-0.5 * (V / S0) ** 2), got.shape), atol=bound)
    # the screen reads the measuring line itself when it is not given
    line = np.stack([profiles.sample_profile(block[i], frame, origin, y_hat, e_u, V, N_AVG, STEP)
                     for i in range(KS.size)])
    a, b = screen(block), screen(block, profiles_gl=line)
    for name in CL.FEATURES:
        assert_array_equal(a.line[name], b.line[name])
    assert a.score == b.score and a.clean == b.clean


def test_convergence():
    # W of a Gaussian dip by linear interpolation between samples h apart: the error falls as h^2. The largest
    # error over 25 positions of the dip within one sample, so that the dip's phase on the grid does not enter.
    s, errs, hs = 0.30, [], [0.2, 0.1, 0.05, 0.025]
    for h in hs:
        v = h * np.arange(-int(3.0 / h), int(3.0 / h) + 1)
        e = [abs(CL.dip_features(np.exp(-0.5 * ((v - c) / s) ** 2), v, window_um=0.5)["fwhm_um"] - 2.0 * G_HALF * s)
             for c in np.linspace(0.0, h, 25, endpoint=False)]
        errs.append(max(e))
    order = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.diff(errs) < 0), errs
    assert np.all((order > 1.5) & (order < 2.5)), (errs, order.tolist())       # 2 expected; 0.5 for the phase


def test_invariants():
    rng = np.random.default_rng(SEED)
    D = 40.0 * np.exp(-0.5 * ((V - 0.2) / np.where(V < 0.2, 0.3, 0.4)) ** 2) + 6.0 * np.exp(-0.5 * ((V + 1.9) / 0.2) ** 2)
    f = CL.dip_features(D, V)
    m = CL.dip_features(D[::-1], V)             # V is symmetric about 0: the mirrored profile on the same line
    for name in ("depth_gl", "fwhm_um", "skew", "outside"):
        assert_allclose(m[name], f[name], rtol=1e-12)
    assert_allclose(m["centre_um"], -f["centre_um"], atol=1e-12)
    sc = CL.dip_features(3.0 * D, V)
    for name in ("fwhm_um", "centre_um", "skew", "outside"):
        assert_allclose(sc[name], f[name], rtol=1e-12, atol=1e-12)
    assert_allclose(sc["depth_gl"], 3.0 * f["depth_gl"], rtol=1e-12)

    clean = screen(band_block())
    assert clean.clean and clean.reasons == () and clean.score <= 1.0, (clean.reasons, clean.ratios)
    # a straight uniform band: the ranges along the axis come from the bilinear read alone
    assert clean.width_range < 0.02 and clean.depth_range < 0.02 and clean.centre_range < 0.02, \
        (clean.width_range, clean.depth_range, clean.centre_range)
    assert_allclose(clean.line["fwhm_um"][4], 2.0 * G_HALF * S0, rtol=0.08)    # smoothing by one sample widens it
    assert np.all(np.diff(clean.line["fwhm_um"][4:]) > 0) and np.all(np.diff(clean.line["fwhm_um"][:5]) < 0)

    cases = {
        # a parallel neighbour 1.4 um aside, half as deep, in focus with the node (the pilot's node 2669)
        "outside": lambda u, v: 0.5 * A_GL * np.exp(-0.5 * ((v + 1.4) / 0.35) ** 2),
        # a branch that crosses the dendrite 1.5 um along the axis at 60 degrees
        "outside ": lambda u, v: 0.8 * A_GL * np.exp(-0.5 * (((u - 1.5) * math.sin(1.05) - v * math.cos(1.05)) / 0.35) ** 2),
    }
    for reason, extra in cases.items():
        r = screen(band_block(extra=extra))
        assert not r.clean and reason.strip() in r.reasons, (reason, r.reasons, r.ratios)
    # a dark blob on the dendrite's edge, 1.5 um along the axis: the measuring line itself stays clean
    blob = screen(band_block(extra=lambda u, v: 50.0 * np.exp(-0.5 * (((u - 1.5) / 0.4) ** 2 + ((v - 0.6) / 0.4) ** 2))))
    assert not blob.clean, blob.ratios
    assert blob.line["outside"][4] < CFG.outside_max and blob.line["skew"][4] < CFG.skew_max
    # a taper: the band's SD grows by 15 % per um along the axis
    taper = screen(band_block(width=lambda u: S0 * (1.0 + 0.15 * u)))
    assert not taper.clean and "width_range" in taper.reasons, (taper.reasons, taper.ratios)
    # a bend: the centre moves by 0.15 u^2 um, 0.6 um at |u| = 2 um
    bend = screen(band_block(centre=lambda u: 0.15 * u ** 2))
    assert not bend.clean and "centre_range" in bend.reasons, (bend.reasons, bend.ratios)

    # 200 clean bands under white pixel noise of SD 4 gl (depth 60 gl): the read's average of 9 bilinear samples
    # leaves about 0.3 of it on a line sample, i.e. a depth of some 50 noise SDs. None may be rejected: the
    # noise-free ratios are below 0.3 of every threshold.
    base = band_block()
    n_clean = sum(screen(base + rng.normal(0.0, 4.0, base.shape), sigma=1.2).clean for _ in range(200))
    assert n_clean == 200, n_clean


def test_contract():
    block = band_block()
    keep = block.copy()
    r = screen(block)
    assert_array_equal(block, keep)
    assert isinstance(r, CL.ClearnessResult) and isinstance(r.clean, bool) and isinstance(r.score, float)
    assert set(r.line) == set(CL.FEATURES) == set(r.along) and set(r.ratios) == set(CL.CRITERIA)
    for name in CL.FEATURES:
        assert r.line[name].shape == (KS.size,) and r.line[name].dtype == np.float64
        assert r.along[name].shape == (9,)
    assert r.k_index == 4 and r.focus.dtype == bool
    assert_array_equal(np.flatnonzero(r.focus), [3, 4, 5])
    assert_allclose(r.u_um, 0.5 * np.arange(-4, 5), atol=1e-12)
    assert r.strips_gl.shape == (9, V.size)
    assert_allclose(r.depth_sd, r.line["depth_gl"][4] / 1.0, rtol=1e-12)
    assert_allclose(r.score, max(r.ratios.values()), rtol=1e-12)
    assert_allclose(CL.along_offsets(2.4, 0.8), [-2.4, -1.6, -0.8, 0.0, 0.8, 1.6, 2.4], atol=1e-12)
    assert_allclose(CL.along_offsets(0.5, 1.0), [0.0], atol=1e-12)
    wide = screen(block, cfg=dataclasses.replace(CFG, focus_half_planes=2))
    assert_array_equal(np.flatnonzero(wide.focus), [2, 3, 4, 5, 6])
    D = np.array([[1.0, 5.0, 2.0, 1.0, 0.5]])
    sm = CL.smooth_dip(D, 1.0)
    assert sm.shape == D.shape and sm.dtype == np.float64 and sm.max() < 5.0
    assert_array_equal(CL.smooth_dip(D, 0.0), D)


def test_determinism():
    rng = np.random.default_rng(SEED)
    block = band_block() + rng.normal(0.0, 3.0, (KS.size, 2 * HW + 1, 2 * HW + 1))
    a, b = screen(block), screen(block)
    assert a.score == b.score and a.clean == b.clean and a.reasons == b.reasons
    for name in CL.FEATURES:
        assert_array_equal(a.line[name], b.line[name])
        assert_array_equal(a.along[name], b.along[name])
    assert_array_equal(a.strips_gl, b.strips_gl)


def test_edge_cases():
    # a dip that never returns to a quarter of its depth on the left: no skew, so no verdict of clean
    D = 50.0 * np.exp(-0.5 * (V / 0.3) ** 2) + np.where(V < 0, 20.0, 0.0)
    f = CL.dip_features(D, V)
    assert math.isfinite(f["fwhm_um"]) and math.isnan(f["skew"])
    shelf = screen(band_block(extra=lambda u, v: np.where(v < 0, 0.4 * A_GL, 0.0)))
    assert not shelf.clean and shelf.score == float("inf") and CL.REASON_NOT_FINITE in shelf.reasons
    # a clean band whose depth is below the floor in noise SDs
    faint = screen(band_block(), sigma=A_GL / 5.0)
    assert not faint.clean and faint.reasons == (CL.REASON_FAINT,) and faint.score <= 1.0
    # k* outside the recorded planes: the nearest plane
    assert screen(band_block(), k_star=500).k_index == KS.size - 1
    # NaN in a profile: its features are NaN
    Dn = 50.0 * np.exp(-0.5 * (V / 0.3) ** 2)
    Dn[10] = np.nan
    assert all(math.isnan(x) for x in CL.dip_features(Dn, V).values())
    # no dip (the profile is lighter than the background at the centre): the depth is kept, nothing else
    f = CL.dip_features(-np.ones(V.size), V)
    assert f["depth_gl"] == -1.0 and math.isnan(f["fwhm_um"])
    # lines along the axis that leave the patch: not finite, never clean
    far = screen(band_block(), cfg=dataclasses.replace(CFG, along_half_um=30.0, along_step_um=10.0))
    assert not far.clean and CL.REASON_NOT_FINITE in far.reasons
    # the measuring line alone: the ranges along the axis are 0
    one = screen(band_block(), cfg=dataclasses.replace(CFG, along_half_um=0.0))
    assert one.u_um.shape == (1,) and one.width_range == 0.0 and one.centre_range == 0.0 and one.clean
    origin, y_hat, e_u = _geometry()
    block = band_block()
    B = np.full(KS.size, B_GL)
    ok = (block, KS, K_STAR, B, 1.0, LEFT, TOP, P_UM, origin, y_hat, e_u, V, N_AVG, STEP, CFG)
    for i, bad in ((0, block[0]), (3, B[:3]), (3, -B), (4, 0.0), (4, float("nan")), (7, 0.0),
                   (8, np.array([1.0, np.nan])), (11, V[::-1]), (11, V[:3])):
        args = list(ok)
        args[i] = bad
        _raises(ValueError, CL.screen_node, *args)
    _raises(ValueError, CL.screen_node, *ok, profiles_gl=np.zeros((3, V.size)))
    _raises(ValueError, CL.dip_features, np.ones(4), V)
    _raises(ValueError, CL.dip_features, np.ones(V.size), V, window_um=0.0)
    _raises(ValueError, CL.dip_features, np.ones(V.size), V, skew_levels=(0.4, 0.25))
    _raises(ValueError, CL.dip_features, np.ones(V.size), V, outside_fwhm=0.0)
    _raises(ValueError, CL.smooth_dip, np.ones(5), -1.0)
    _raises(ValueError, CL.along_offsets, 1.0, 0.0)
    _raises(ValueError, CL.strip_profiles, block[0], LEFT, TOP, P_UM, origin, y_hat, e_u, V, N_AVG, STEP, [])
    _raises(ValueError, CL.strip_profiles, block[0], LEFT, TOP, P_UM, origin, y_hat, e_u, V, -1, STEP, [0.0])


# ------------------------------------------------------------------ main ---

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
