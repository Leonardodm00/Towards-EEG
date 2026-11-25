"""Smoke test for the soma fit over a grid of held depths and the far-background check (analysis/soma_profile.py)
-- Block 50 in specs/SPEC.md (D-252, D-254.3, D-256, D-257).

Checks (P1 to P5 are the entry's oracles)
    test_known_answer   P1: fit_c_grid with a stand-in fit_one whose chi2 is a parabola in the held c, its minimum
                        between two grid values, below the first and beyond the last: best_index, the neighbourhood
                        (interior; both ends mirrored; the clip into axis_bounds_um), the arguments fit_one received
                        (the holds {"c_um": c} with start and bounds None; the refinement's {} with the best fit's
                        theta and the neighbourhood), delivered and its source, each flag raised by a case built for
                        it and absent otherwise, in FLAGS order; the delivery rule's 1e-9 relative tolerance on both
                        sides; P3: far_mask against the distance written out pixel by pixel on a 5 x 7 crop, a pixel
                        exactly at far_um included, masked pixels excluded
    test_reference      P4: far_background on a known tilted plane per plane, noise-free [1e-9 gl, gl/um]; with
                        seeded Gaussian noise within 4 SE of the truth (SE from the design matrix); with 5 % of the
                        far pixels darkened by 30 gl the trimmed b0 moves by < 0.5 gl where an untrimmed
                        numpy.linalg.lstsq moves by > 1 gl; equal to calibration_scene.planar_background called
                        directly with a CropFrame [exact]
    test_convergence    not a discretisation: skipped
    test_invariants     P2: grid_table against the stand-ins' values (x0 to mu from theta in Block 46's PARAMS order,
                        area and r_eq from the attributes, NaN where absent), rms_gl = sqrt(chi2 / n_samples)
                        [1e-12 relative]; P5: background_check on arrays built by hand -- the near window's closed
                        ends (+- 1e-9 um), planes not ok or not finite left out, the mean and the largest |Delta|
                        [1e-12], the flag just below, at and just above the tolerance
    test_contract       types, shapes and dtypes of CGrid, grid_table, far_mask, far_background, background_check
    test_determinism    two calls give identical results; the inputs are not modified
    test_edge_cases     no usable fit (nothing delivered, no refinement call), the tie's rule (the lower c), one grid
                        value (no edge flag; refine -> ValueError before any fit), exceptions of fit_one pass through,
                        grids not ascending / not finite / outside the bounds (ValueError), no near plane
                        (status "no_planes", flag False, mean NaN), far_background's refusals (too few far pixels,
                        a plane of NaN), the ValueErrors of background_check, far_mask and far_background

The stand-in fit (Block 46 is amended in parallel and not imported): a frozen dataclass with the attributes the
contract names (theta (9,), chi2, status, at_bound, n_samples) and Block 46's area_um2, r_eq_um; theta[5] is the
held c, the other entries simple functions of c.

Run
    cd "Passive Features/Diameter Re-measurement"
    OMP_NUM_THREADS=1 python3 tests/smoke/test_smoke_soma_profile.py
        Prints the environment, the observed numbers, and a PASS / FAIL / ERROR / TODO / SKIP table, then the full
        message of every failure. Exit code 1 if any check failed, errored or is still TODO. A few seconds.
    python -m pytest tests/smoke/test_smoke_soma_profile.py -q
        The same checks through pytest.

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import copy
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
from typing import Tuple

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
if str(WS / "src") not in sys.path:
    sys.path.insert(0, str(WS / "src"))

from allen_diameter.analysis import soma_profile as SP  # noqa: E402
from allen_diameter.analysis import calibration_scene as CS  # noqa: E402
from allen_diameter.config import SomaFitConfig  # noqa: E402
from allen_image_io import CropFrame  # noqa: E402

SEED = 20261125
REPORT_PACKAGES = ("numpy", "scipy")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run

GRID = (1.5, 2.5, 3.5, 5.0)         # SomaFitConfig's c_grid_um (D-256.1)
BOUNDS = (1.0, 15.0)                # SomaFitConfig's axis_bounds_um
PARA_A, CHI2_0 = 1000.0, 5000.0     # the stand-in's chi2 = CHI2_0 + PARA_A (c - c_min)^2
N_SAMPLES = 7442 * 18               # the pilot's region (7442 px) on 18 planes: rms_gl of a few gl
P_UM = 0.25                         # P4's pixel (um)


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure (arrays saved to ARTIFACT_DIR)."""
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


@dataclasses.dataclass(frozen=True)
class Res:
    """Stand-in for Block 46's SomaFit: the attributes Block 50's contract reads."""
    theta: np.ndarray
    chi2: float
    status: str
    at_bound: Tuple[str, ...]
    n_samples: int
    area_um2: float = float("nan")
    r_eq_um: float = float("nan")


@dataclasses.dataclass(frozen=True)
class ResBare:
    """A result with only the five attributes of fit_one's contract (no area_um2, r_eq_um)."""
    theta: np.ndarray
    chi2: float
    status: str
    at_bound: Tuple[str, ...]
    n_samples: int


def theta_of(c):
    """The stand-in's theta at a held c: theta[5] = c, the others simple functions of c (PARAMS' units)."""
    return np.array([0.1 * c, -0.2 * c, 30.0 + 0.01 * c, 4.0 + 0.1 * c, 3.0 + 0.05 * c, c, 0.7, 0.12 / c, 0.79])


def area_of(c):
    return 100.0 + 10.0 * c


class StandIn:
    """fit_one stand-in: chi2 a parabola in the held c; per-c failures; a configurable refinement result."""

    def __init__(self, c_min, fail=(), inf=(), ref_chi2=None, ref_status="converged", ref_at_bound=(),
                 bare=(), raise_at=None):
        self.c_min, self.fail, self.inf = float(c_min), set(fail), set(inf)
        self.ref_chi2, self.ref_status, self.ref_at_bound = ref_chi2, ref_status, tuple(ref_at_bound)
        self.bare, self.raise_at = set(bare), raise_at
        self.calls = []

    def chi2(self, c):
        return CHI2_0 + PARA_A * (c - self.c_min) ** 2

    def __call__(self, hold, start, c_bounds_um):
        self.calls.append((copy.deepcopy(hold), None if start is None else np.array(start, copy=True),
                           c_bounds_um))
        if hold:
            c = float(hold["c_um"])
            if self.raise_at is not None and c == self.raise_at:
                raise RuntimeError("stand-in failure at c = %g" % c)
            if c in self.fail:
                return Res(np.full(9, np.nan), float("nan"), "failed", (), N_SAMPLES)
            chi2 = float("inf") if c in self.inf else self.chi2(c)
            if c in self.bare:
                return ResBare(theta_of(c), chi2, "converged", (), N_SAMPLES)
            return Res(theta_of(c), chi2, "converged", (), N_SAMPLES, area_of(c), math.sqrt(area_of(c) / (4 * math.pi)))
        th = np.array(start, dtype=float, copy=True)
        if self.ref_status == "failed":
            return Res(np.full(9, np.nan), float("nan"), "failed", (), N_SAMPLES)
        th[5] = self.c_min
        return Res(th, float(self.ref_chi2), self.ref_status, self.ref_at_bound, N_SAMPLES, 1.0, 2.0)


def plane_data(shape, left, top, p, centre, b, rng=None, sd=0.0):
    """(H, W) gl: b0 + b1 (x - x_s) + b2 (y - y_s) at the pixel centres, plus Gaussian noise of SD sd."""
    H, W = shape
    rows, cols = np.mgrid[0:H, 0:W]
    x, y = (left + cols) * p, (top + rows) * p
    D = b[0] + b[1] * (x - centre[0]) + b[2] * (y - centre[1])
    if sd > 0:
        D = D + sd * rng.standard_normal(shape)
    return D


def p4_scene():
    """The P4 crop: 80 x 72 px of 0.25 um, the traced centre near its middle, the far pixels at >= 6 um, a masked
    stripe; three planes with different tilts (gl, gl/um, gl/um)."""
    shape, left, top = (80, 72), 400, 1200
    centre = ((left + 36.3) * P_UM, (top + 39.6) * P_UM)
    mask = np.zeros(shape, dtype=bool)
    mask[5:9, :] = True
    far = SP.far_mask(shape, left, top, P_UM, centre, 6.0, mask)
    truth = np.array([[150.0, 0.3, -0.2], [148.5, -0.25, 0.1], [152.0, 0.0, 0.28]])
    return shape, left, top, centre, far, truth


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    # P1 (a): the minimum between two grid values (c_min 2.9: chi2 at 2.5 is 160 above CHI2_0, at 3.5 360).
    f = StandIn(2.9, ref_chi2=CHI2_0 - 1.0)
    g = SP.fit_c_grid(f, GRID, True, BOUNDS)
    assert g.best_index == 1, g.best_index
    assert g.neighbourhood_um == (1.5, 3.5), g.neighbourhood_um
    assert len(f.calls) == 5, len(f.calls)
    for (hold, start, cb), c in zip(f.calls[:4], GRID):
        assert hold == {"c_um": c} and start is None and cb is None, (hold, start, cb)
        assert type(hold["c_um"]) is float
    hold, start, cb = f.calls[4]
    assert hold == {} and cb == (1.5, 3.5), (hold, cb)
    assert_array_equal(start, g.fits[1].theta)
    assert g.delivered is g.refinement and g.delivered_source == "refinement", g.delivered_source
    assert g.flags == (), g.flags
    print(f"   P1 interior: best c {GRID[g.best_index]}, neighbourhood {g.neighbourhood_um}, delivered "
          f"{g.delivered_source}, flags {g.flags}")
    # refine False: no refinement call, no neighbourhood, the best grid fit delivered
    f = StandIn(2.9)
    g = SP.fit_c_grid(f, GRID, False, BOUNDS)
    assert len(f.calls) == 4 and g.neighbourhood_um is None and g.refinement is None
    assert g.delivered is g.fits[1] and g.delivered_source == "grid" and g.flags == ()

    # P1 (b): the minimum below the first value, mirrored (bounds 0.1 to 15) and clipped (the default bounds).
    f = StandIn(1.0, ref_chi2=CHI2_0)
    g = SP.fit_c_grid(f, GRID, True, (0.1, 15.0))
    assert g.best_index == 0 and g.neighbourhood_um == (0.5, 2.5), (g.best_index, g.neighbourhood_um)
    assert "best_at_grid_edge" in g.flags and f.calls[4][2] == (0.5, 2.5)
    f = StandIn(1.0, ref_chi2=CHI2_0)
    g = SP.fit_c_grid(f, GRID, True, BOUNDS)
    assert g.neighbourhood_um == (1.0, 2.5), g.neighbourhood_um
    assert g.flags == ("best_at_grid_edge",), g.flags
    # P1 (c): beyond the last value, mirrored (5.0 + 1.5) and clipped (bounds up to 6.0).
    f = StandIn(7.0, ref_chi2=CHI2_0)
    g = SP.fit_c_grid(f, GRID, True, BOUNDS)
    assert g.best_index == 3 and g.neighbourhood_um == (3.5, 6.5), (g.best_index, g.neighbourhood_um)
    f = StandIn(7.0, ref_chi2=CHI2_0)
    g = SP.fit_c_grid(f, GRID, True, (1.0, 6.0))
    assert g.neighbourhood_um == (3.5, 6.0), g.neighbourhood_um
    print(f"   P1 ends: low mirrored (0.5, 2.5) / clipped (1.0, 2.5); high mirrored (3.5, 6.5) / clipped (3.5, 6.0)")

    # P1 (d): the delivery rule -- within 1e-9 relative the refinement is delivered, beyond it the grid fit.
    best_chi2 = StandIn(2.9).chi2(2.5)
    g = SP.fit_c_grid(StandIn(2.9, ref_chi2=best_chi2 * (1 + 0.5e-9)), GRID, True, BOUNDS)
    assert g.delivered_source == "refinement" and g.flags == (), (g.delivered_source, g.flags)
    g = SP.fit_c_grid(StandIn(2.9, ref_chi2=best_chi2 * (1 + 2e-9)), GRID, True, BOUNDS)
    assert g.delivered is g.fits[1] and g.delivered_source == "grid", g.delivered_source
    assert g.flags == ("refinement_worse",), g.flags
    # each remaining flag by its own case
    g = SP.fit_c_grid(StandIn(2.9, ref_status="failed"), GRID, True, BOUNDS)
    assert g.flags == ("refinement_failed",) and g.delivered is g.fits[1], g.flags
    g = SP.fit_c_grid(StandIn(2.9, ref_chi2=CHI2_0, ref_at_bound=("c_um",)), GRID, True, BOUNDS)
    assert g.flags == ("refined_c_at_bound",) and g.delivered_source == "refinement", g.flags
    g = SP.fit_c_grid(StandIn(2.9, ref_chi2=CHI2_0, ref_at_bound=("a_um",)), GRID, True, BOUNDS)
    assert g.flags == (), g.flags
    g = SP.fit_c_grid(StandIn(2.9, fail=(1.5,)), GRID, False, BOUNDS)
    assert g.flags == ("grid_fit_failed",) and g.best_index == 1, g.flags
    g = SP.fit_c_grid(StandIn(2.9, inf=(2.5,)), GRID, False, BOUNDS)   # an infinite chi2 is not usable
    assert g.flags == ("grid_fit_failed",) and g.best_index == 2, (g.flags, g.best_index)
    # several flags at once, in FLAGS order
    g = SP.fit_c_grid(StandIn(7.0, fail=(2.5,), ref_chi2=1e9, ref_at_bound=("c_um",)), GRID, True, BOUNDS)
    assert g.flags == ("grid_fit_failed", "best_at_grid_edge", "refinement_worse", "refined_c_at_bound"), g.flags
    print(f"   P1 delivery: 0.5e-9 relative worse -> refinement, 2e-9 -> grid; flags one by one and together {g.flags}")

    # P3: far_mask against the distance written out, on a 5 x 7 crop; a pixel exactly at far_um included.
    shape, left, top, p = (5, 7), 3, -2, 0.5
    centre = ((3 + 2) * p, (-2 + 2) * p)    # the centre of pixel (row 2, col 2): (2.5, 0.0) um
    mask = np.zeros(shape, dtype=bool)
    mask[0, 6] = True
    mask[4, 0] = True
    got = SP.far_mask(shape, left, top, p, centre, 1.0, mask)
    exp = np.zeros(shape, dtype=bool)
    for r in range(shape[0]):
        for cidx in range(shape[1]):
            x, y = (left + cidx) * p, (top + r) * p
            exp[r, cidx] = math.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2) >= 1.0 and not mask[r, cidx]
    assert_array_equal(got, exp)
    assert got[2, 4] and got[0, 2] and not got[2, 3] and not got[0, 6] and not got[4, 0]
    print(f"   P3 far_mask: {int(got.sum())} of {got.size} pixels far; (2, 4) at exactly 1.0 um included")


def test_reference():
    shape, left, top, centre, far, truth = p4_scene()
    trim = 3.0   # calibration.background_trim_sigmas' default
    # noise-free: each plane's tilted background recovered [1e-9 gl, gl/um: float64 lstsq on ~4000 px]
    D = np.stack([plane_data(shape, left, top, P_UM, centre, b) for b in truth])
    b, ok, n_used = SP.far_background(D, far, left, top, P_UM, centre, trim)
    assert ok.all(), ok
    check_close("p4_noise_free", b, truth, rtol=0.0, atol=1e-9)
    print(f"   P4 noise-free: max |b - truth| {np.max(np.abs(b - truth)):.2e}; far pixels {int(far.sum())}, "
          f"used {n_used.tolist()}")
    # equal to planar_background called directly with Block 10.P's frame [exact]
    frame = CropFrame(left, top, 0, P_UM)
    rng = make_rng()
    Dn = np.stack([plane_data(shape, left, top, P_UM, centre, bb, rng, 2.0) for bb in truth])
    b, ok, n_used = SP.far_background(Dn, far, left, top, P_UM, centre, trim)
    for k in range(3):
        bk, nk, okk = CS.planar_background(Dn[k], far, frame, centre[0], centre[1], trim, 0.0)
        assert_array_equal(b[k], bk)
        assert nk == n_used[k] and okk == ok[k]
    # noisy (SD 2 gl): within 4 SE of the truth, SE = sd sqrt(diag((A^T A)^-1)) over the far pixels
    rows, cols = np.nonzero(far)
    x, y = (left + cols) * P_UM - centre[0], (top + rows) * P_UM - centre[1]
    A = np.column_stack([np.ones(x.size), x, y])
    se = 2.0 * np.sqrt(np.diag(np.linalg.inv(A.T @ A)))
    z = np.abs(b - truth) / se
    assert np.all(z < 4.0), z
    print(f"   P4 noisy (SD 2 gl): largest |b - truth| / SE {z.max():.2f} (SE b0 {se[0]:.3f} gl, b1 {se[1]:.4f})")
    # 5 % of the far pixels darkened by 30 gl (noise SD 1 gl): trimmed b0 moves < 0.5 gl, untrimmed lstsq > 1 gl
    rng = make_rng(SEED + 1)
    clean = plane_data(shape, left, top, P_UM, centre, truth[0], rng, 1.0)
    dirty = clean.copy()
    n_out = int(round(0.05 * far.sum()))
    pick = rng.choice(far.sum(), size=n_out, replace=False)
    dirty[rows[pick], cols[pick]] -= 30.0
    b_c = SP.far_background(clean[None], far, left, top, P_UM, centre, trim)[0][0]
    b_d = SP.far_background(dirty[None], far, left, top, P_UM, centre, trim)[0][0]
    l_c = np.linalg.lstsq(A, clean[rows, cols], rcond=None)[0]
    l_d = np.linalg.lstsq(A, dirty[rows, cols], rcond=None)[0]
    move_t, move_l = abs(b_d[0] - b_c[0]), abs(l_d[0] - l_c[0])
    assert move_t < 0.5 and move_l > 1.0, (move_t, move_l)
    print(f"   P4 outliers ({n_out} px at -30 gl): trimmed b0 moves {move_t:.3f} gl, untrimmed lstsq {move_l:.3f} gl")


def test_convergence():
    raise unittest.SkipTest("not a discretisation: a grid search with bookkeeping and a least-squares plane")


def test_invariants():
    # P2: grid_table against the stand-ins' values
    f = StandIn(2.9, fail=(5.0,), bare=(3.5,))
    g = SP.fit_c_grid(f, GRID, False, BOUNDS)
    t = SP.grid_table(g)
    c = np.array(GRID)
    chi2 = np.array([f.chi2(v) for v in GRID[:3]] + [np.nan])
    check_close("p2_c", t["c_um"], c, rtol=0.0)
    check_close("p2_chi2", t["chi2"], chi2, rtol=0.0)
    check_close("p2_rms", t["rms_gl"], np.sqrt(chi2 / N_SAMPLES), rtol=1e-12)   # its formula, float64 roundoff
    for name, i in (("x0_um", 0), ("y0_um", 1), ("z0_um", 2), ("a_um", 3), ("b_um", 4), ("phi_rad", 6),
                    ("mu_per_um", 7)):
        exp = np.array([theta_of(v)[i] for v in GRID[:3]] + [np.nan])
        check_close("p2_" + name, t[name], exp, rtol=0.0)
    area = np.array([area_of(1.5), area_of(2.5), np.nan, np.nan])   # 3.5: a bare result; 5.0: failed (no value)
    check_close("p2_area", t["area_um2"], area, rtol=0.0)
    check_close("p2_req", t["r_eq_um"], np.sqrt(area / (4 * math.pi)), rtol=1e-15)
    assert_array_equal(t["usable"], [True, True, True, False])
    print(f"   P2 grid_table: rms_gl {np.round(t['rms_gl'], 4).tolist()} gl, usable {t['usable'].tolist()}")

    # P5: background_check on arrays built by hand
    z0, near = 20.0, 0.56
    dz = np.array([-0.84, -0.56, -0.28, 0.0, 0.28, 0.56, 0.84, 0.56 + 1e-6, -0.56 - 2e-10])
    z = z0 + dz                                       # +-0.56 with float roundoff: inside by the 1e-9 tolerance
    K = z.size
    beta = np.column_stack([np.linspace(150, 158, K), np.full(K, 0.1), np.full(K, -0.1)])
    far_b = beta.copy()
    far_b[:, 0] -= np.array([9, 1.0, 2.0, 3.0, 0.5, -1.5, 9, 9, 4.0])
    far_ok = np.ones(K, dtype=bool)
    far_ok[2] = False                                  # not ok: out
    beta[4, 1] = np.nan                                # a coefficient not finite: out
    out = SP.background_check(beta, z, z0, far_b, far_ok, near, 1.0)
    exp_idx = np.array([1, 3, 5, 8])
    exp_d = np.array([1.0, 3.0, -1.5, 4.0])
    assert_array_equal(out["planes"], exp_idx)
    check_close("p5_delta", out["delta_gl"], exp_d, rtol=1e-12)
    check_close("p5_mean", out["mean_gl"], exp_d.mean(), rtol=1e-12)
    check_close("p5_max", out["max_abs_gl"], 4.0, rtol=1e-12)
    assert out["n_planes"] == 4 and out["status"] == "ok" and out["flag"] is True
    m = abs(out["mean_gl"])
    for tol, flag in ((m + 1e-9, False), (m, False), (m - 1e-9, True)):
        assert SP.background_check(beta, z, z0, far_b, far_ok, near, tol)["flag"] is flag, (tol, flag)
    print(f"   P5 background_check: planes {out['planes'].tolist()}, mean {out['mean_gl']:.4f} gl, "
          f"max {out['max_abs_gl']:.1f} gl; flag False at tol = |mean| + 1e-9 and = |mean|, True at - 1e-9")


def test_contract():
    g = SP.fit_c_grid(StandIn(2.9, ref_chi2=CHI2_0), list(GRID), True, BOUNDS)
    assert isinstance(g, SP.CGrid) and dataclasses.is_dataclass(g)
    try:
        g.best_index = 0
        raise AssertionError("CGrid is not frozen")
    except dataclasses.FrozenInstanceError:
        pass
    assert g.c_grid_um.dtype == np.float64 and g.c_grid_um.shape == (4,)
    assert isinstance(g.fits, tuple) and len(g.fits) == 4
    assert type(g.best_index) is int
    assert isinstance(g.neighbourhood_um, tuple) and all(type(v) is float for v in g.neighbourhood_um)
    assert g.delivered_source in SP.SOURCES and isinstance(g.flags, tuple)
    assert SP.FLAGS == ("grid_fit_failed", "best_at_grid_edge", "refinement_failed", "refinement_worse",
                        "refined_c_at_bound")
    t = SP.grid_table(g)
    assert list(t) == ["c_um", "chi2", "rms_gl", "x0_um", "y0_um", "z0_um", "a_um", "b_um", "phi_rad",
                       "mu_per_um", "area_um2", "r_eq_um", "usable"], list(t)
    for k, v in t.items():
        assert v.shape == (4,), (k, v.shape)
        assert v.dtype == (bool if k == "usable" else np.float64), (k, v.dtype)
    shape, left, top, centre, far, truth = p4_scene()
    assert far.shape == shape and far.dtype == bool
    D = np.stack([plane_data(shape, left, top, P_UM, centre, b) for b in truth]).astype(np.float32)
    b, ok, n = SP.far_background(D, far, left, top, P_UM, centre, 3.0)
    assert b.shape == (3, 3) and b.dtype == np.float64 and ok.shape == (3,) and ok.dtype == bool
    assert n.shape == (3,) and np.issubdtype(n.dtype, np.integer)
    out = SP.background_check(truth, np.array([0.0, 0.28, 0.56]), 0.28, b, ok, 1.2, 1.0)
    assert set(out) == {"planes", "delta_gl", "mean_gl", "max_abs_gl", "n_planes", "near_um", "tol_gl", "flag",
                        "status"}
    assert np.issubdtype(out["planes"].dtype, np.integer) and out["delta_gl"].dtype == np.float64
    assert type(out["flag"]) is bool and type(out["n_planes"]) is int and type(out["mean_gl"]) is float
    # the configuration's defaults run through fit_c_grid
    q = SomaFitConfig()
    g = SP.fit_c_grid(StandIn(2.9), q.c_grid_um, q.c_refine, q.axis_bounds_um)
    assert g.delivered_source == "grid" and g.best_index == 1


def test_determinism():
    shape, left, top, centre, far, truth = p4_scene()
    rng = make_rng()
    D = np.stack([plane_data(shape, left, top, P_UM, centre, bb, rng, 2.0) for bb in truth])
    grid = np.array(GRID)
    snap = (D.copy(), far.copy(), grid.copy())
    g1 = SP.fit_c_grid(StandIn(2.9, ref_chi2=CHI2_0), grid, True, BOUNDS)
    g2 = SP.fit_c_grid(StandIn(2.9, ref_chi2=CHI2_0), grid, True, BOUNDS)
    assert g1.best_index == g2.best_index and g1.neighbourhood_um == g2.neighbourhood_um and g1.flags == g2.flags
    assert_array_equal(g1.delivered.theta, g2.delivered.theta)
    assert g1.c_grid_um is not grid
    for name in ("c_um", "chi2", "rms_gl", "area_um2"):
        assert_array_equal(SP.grid_table(g1)[name], SP.grid_table(g2)[name])
    r1 = SP.far_background(D, far, left, top, P_UM, centre, 3.0)
    r2 = SP.far_background(D, far, left, top, P_UM, centre, 3.0)
    for a, b in zip(r1, r2):
        assert_array_equal(a, b)
    beta = truth.copy()
    beta_snap = beta.copy()
    o1 = SP.background_check(beta, np.zeros(3), 0.0, r1[0], r1[1], 1.2, 1.0)
    o2 = SP.background_check(beta, np.zeros(3), 0.0, r1[0], r1[1], 1.2, 1.0)
    assert_array_equal(o1["delta_gl"], o2["delta_gl"]) and o1["mean_gl"] == o2["mean_gl"]
    assert_array_equal(D, snap[0])
    assert_array_equal(far, snap[1])
    assert_array_equal(grid, snap[2])
    assert_array_equal(beta, beta_snap)


def _raises(exc, fn, *a):
    try:
        fn(*a)
    except exc:
        return
    raise AssertionError(f"{fn.__name__}{a!r:.80} did not raise {exc.__name__}")


def test_edge_cases():
    # no usable fit: nothing delivered, no refinement call
    f = StandIn(2.9, fail=GRID)
    g = SP.fit_c_grid(f, GRID, True, BOUNDS)
    assert g.best_index == -1 and g.delivered is None and g.delivered_source == "none"
    assert g.refinement is None and g.neighbourhood_um is None and len(f.calls) == 4
    assert g.flags == ("grid_fit_failed",), g.flags
    # the tie: chi2 equal at 2.5 and 3.5 (c_min 3.0) -> the lower c
    g = SP.fit_c_grid(StandIn(3.0), (2.5, 3.5), False, BOUNDS)
    assert g.fits[0].chi2 == g.fits[1].chi2 and g.best_index == 0
    # one grid value: no edge flag; refine -> ValueError before any fit
    g = SP.fit_c_grid(StandIn(2.9), (2.5,), False, BOUNDS)
    assert g.best_index == 0 and g.flags == () and g.delivered_source == "grid"
    f = StandIn(2.9)
    _raises(ValueError, SP.fit_c_grid, f, (2.5,), True, BOUNDS)
    assert f.calls == []
    # an exception of fit_one passes through
    _raises(RuntimeError, SP.fit_c_grid, StandIn(2.9, raise_at=3.5), GRID, False, BOUNDS)
    # bad grids
    for bad in ((2.5, 1.5), (1.5, 1.5), (1.5, float("nan")), (), (0.5, 2.0), (2.0, 16.0)):
        _raises(ValueError, SP.fit_c_grid, StandIn(2.9), bad, False, BOUNDS)
    g = SP.fit_c_grid(StandIn(2.9), (1.0, 15.0), False, BOUNDS)   # the bounds themselves are inside (closed)
    assert g.best_index == 0
    # background_check: no near plane
    zz = np.array([0.0, 0.28, 0.56])
    bb = np.ones((3, 3))
    out = SP.background_check(bb, zz, 5.0, bb, np.ones(3, bool), 1.2, 1.0)
    assert out["status"] == "no_planes" and out["flag"] is False and math.isnan(out["mean_gl"])
    assert out["n_planes"] == 0 and out["planes"].size == 0 and math.isnan(out["max_abs_gl"])
    out = SP.background_check(bb, zz, 0.28, bb, np.zeros(3, bool), 1.2, 1.0)   # none ok
    assert out["status"] == "no_planes"
    # background_check's ValueErrors
    ok3 = np.ones(3, bool)
    _raises(ValueError, SP.background_check, np.ones((2, 3)), zz, 0.0, bb, ok3, 1.2, 1.0)
    _raises(ValueError, SP.background_check, bb, zz, 0.0, np.ones((3, 2)), ok3, 1.2, 1.0)
    _raises(ValueError, SP.background_check, bb, zz, 0.0, bb, np.ones(2, bool), 1.2, 1.0)
    for z0, nr, tl in ((float("nan"), 1.2, 1.0), (0.0, float("inf"), 1.0), (0.0, -0.1, 1.0),
                       (0.0, 1.2, float("nan")), (0.0, 1.2, -1.0)):
        _raises(ValueError, SP.background_check, bb, zz, z0, bb, ok3, nr, tl)
    # far_mask / far_background refusals
    _raises(ValueError, SP.far_mask, (4, 5), 0, 0, 0.5, (1.0, 1.0), 1.0, np.zeros((5, 4), bool))
    _raises(ValueError, SP.far_mask, (4, 5), 0, 0, 0.5, (1.0, 1.0), 1.0, np.zeros((4, 5), int))
    _raises(ValueError, SP.far_background, np.zeros((2, 4, 5)), np.ones((5, 4), bool), 0, 0, 0.5, (1, 1), 3.0)
    _raises(ValueError, SP.far_background, np.zeros((4, 5)), np.ones((4, 5), bool), 0, 0, 0.5, (1, 1), 3.0)
    shape, left, top, centre, far, truth = p4_scene()
    D = np.stack([plane_data(shape, left, top, P_UM, centre, b) for b in truth])
    D[1] = np.nan
    b, ok, n = SP.far_background(D, far, left, top, P_UM, centre, 3.0)
    assert ok.tolist() == [True, False, True] and np.all(np.isnan(b[1])) and n[1] == 0
    b, ok, n = SP.far_background(D, far, left, top, P_UM, centre, 3.0, min_frac=0.99)   # too few far pixels
    assert not ok.any() and np.all(np.isnan(b)) and not n.any()


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
        print(f"-- {name}", flush=True)
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

    print()
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
