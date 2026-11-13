"""Smoke test for the shadow image of a node's neighbours (tier 1) -- Block 25 in
specs/SPEC.md, with its amendment 2 (D-132.1).

Checks, from the strongest oracle to the weakest
    test_known_answer   a flat segment against calibration_scene.shadow_depths
                        (pixel means of the continuous 1-D integral) at lateral
                        offsets across its middle, beyond its caps' reach
                        [1e-6 relative to the peak, as SPEC.md writes it];
                        before it, the same samples blurred by hand (the blur
                        exact, 1e-10); hybrid faint limit: at mubar dbar = 1e-3,
                        A_q / (1.4470 mubar l_q) within 5e-3 of 1
    test_reference      "hybrid" A_q equals the column sum of
                        render.absorbed_fractions(..., "hybrid", directions) on
                        the same grid [1e-10]; stretch_profiles equals
                        sample_profile on shadow_image's planes for the same
                        segments [1e-10], both absorptions; against Block 4's
                        render_transmittance of the same capped tube under the
                        same absorption, pixel-integrated: out of focus (axis
                        >= 0.84 um from the plane) max |S - (1 - tau)| <= 0.03 x
                        the plane's peak, in focus <= 0.15 x the peak, over
                        d in {0.3, 0.6, 1.0} um and tilt 0, 30 deg [PROVISIONAL:
                        the measured values and the time per segment printed]
    test_convergence    halving shadow_depth_bin_um moves a 45-degree segment's
                        image by less than 1e-3 x its peak (both absorptions)
    test_invariants     total per plane equal across planes and to the integral
                        of A_q [relative 1e-8] (segments and a soma); additivity
                        over segments [1e-12]; `kept` follows the floor and S
                        holds exactly the kept neighbours; `overlap` true for a
                        segment directly above the own stretch and false 3 um
                        aside; include=False leaves a neighbour out of S
    test_contract       ShadowResult shapes, dtypes, finiteness; neighbour_indices;
                        segment_absorbed shapes; stretch_profiles shape
    test_determinism    two runs bit-identical (both absorptions)
    test_edge_cases     a segment beyond shadow_reach_sigmas kernel SDs of the
                        patch contributes exactly 0; mu = 0 gives 0; ValueError
                        for an own stretch with no segment in the scene,
                        non-finite or empty planes, include of the wrong length,
                        B_gl <= 0, an unknown shadow_absorption

Run
    cd "Passive Features/Diameter Re-measurement"
    python3 tests/smoke/test_smoke_shadow.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    python3 -m pytest tests/smoke/test_smoke_shadow.py -q

Debug
    A failed check_close() saves both arrays to
        tests/smoke/artifacts/<check name>.npz

Keep this file pure ASCII (no Greek letters, dashes or arrows outside ASCII)
so it survives transfer to the cluster.
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
WS = HERE.parent.parent
SRC = WS / "src"
for _p in (str(SRC), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from allen_image_io import CropFrame  # noqa: E402
from allen_diameter.config import DiameterConfig  # noqa: E402
from allen_diameter.analysis import calibration_scene as CS  # noqa: E402
from allen_diameter.analysis.node_features import FeaturedScene  # noqa: E402
from allen_diameter.analysis.profiles import sample_profile  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import kernel as K  # noqa: E402
from allen_diameter.model import ray_world as RW  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402
from allen_diameter.model import shadow as SH  # noqa: E402

SEED = 12345  # no randomness in Block 25
REPORT_PACKAGES = ("numpy", "scipy")
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

P_X = 0.1144                      # AcquisitionConfig.res0_um (asserted below)
FAR_OWN = ((500.0, 500.0, 0.0), (501.0, 500.0, 0.0))   # an own stretch far from everything
OWN_IDS = {900, 901}
MEASURED = {}                      # printed by main() after the table


def cfg_for(absorption, **cleaning):
    cfg = DiameterConfig()
    assert cfg.acquisition.res0_um == P_X
    cl = dataclasses.replace(cfg.cleaning, shadow_absorption=absorption, **cleaning)
    return dataclasses.replace(cfg, cleaning=cl)


def make_scene(segs, soma=None, own=True):
    """segs: list of (p0, p1, d, mu[, (child, parent)]); d, mu scalars or pairs (parent end, child end).
    Ids 2i+1, 2i+2 unless given; with own=True the far own stretch (900, 901) is appended last."""
    rows = list(segs)
    if own:
        rows.append((FAR_OWN[0], FAR_OWN[1], 0.5, 1.0, (901, 900)))
    xyz, d, mu, ids = [], [], [], []
    for i, s in enumerate(rows):
        p0, p1, dd, mm = s[:4]
        xyz.append(list(p0) + list(p1))
        d.append(list(np.broadcast_to(np.asarray(dd, float), (2,))))
        mu.append(list(np.broadcast_to(np.asarray(mm, float), (2,))))
        ids.append(list(s[4]) if len(s) > 4 else [2 * i + 2, 2 * i + 1])
    soma_arr = np.zeros((0, 3)) if soma is None else np.asarray(soma, float).reshape(-1, 3)
    return FeaturedScene(seg_xyz=np.asarray(xyz, float), seg_d_um=np.asarray(d, float),
                         seg_mu_per_um=np.asarray(mu, float), seg_ids=np.asarray(ids, np.int64),
                         seg_type=np.full(len(rows), 3, np.int64), soma=soma_arr)


def tube_of(p0, p1, d):
    """Block 2 Tube of the segment p0 -> p1, built here independently of shadow.py."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = p1 - p0
    L = float(np.linalg.norm(t))
    if t[2] < 0:
        t = -t
    phi = math.atan2(t[2], math.hypot(t[0], t[1]))
    theta = math.atan2(t[1], t[0])
    c = 0.5 * (p0 + p1)
    return G.Tube(c=tuple(c), r=0.5 * d, phi=phi, theta=theta, half_length=0.5 * L, end_cut="axial")


def patch_grid(left, top, width, height, f=8):
    """The patch's whole fine grid (Block 4 block_fine_grid, no padding)."""
    return R.block_fine_grid(left, top, width, height, P_X, f, 0)[0]


def test_known_answer():
    cfg = cfg_for("partition_vertical")
    f = cfg.cleaning.shadow_fine_factor
    h = P_X / f
    top, height, left, width = -20, 41, -5, 11
    worst_spec, worst_discrete = 0.0, 0.0
    rows = []
    for d in (0.3, 1.0):
        r, mu = 0.5 * d, 1.0
        fs = make_scene([((-6.0, 0.0, 0.0), (6.0, 0.0, 0.0), d, mu)])
        planes = np.array([0.0, 0.84])
        res = SH.shadow_image(fs, OWN_IDS, left, top, width, height, planes, cfg)
        col = res.S[:, :, 5]                          # x = 0, 5.4 um from either cap
        y_pix = (top + np.arange(height)) * P_X
        y_sub = y_pix[:, None] - 0.5 * P_X + (np.arange(f)[None, :] + 0.5) * h      # the pixel's fine samples
        y_lat0 = top * P_X - 0.5 * P_X + 0.5 * h                 # the patch's fine lattice
        m = np.arange(int(math.floor((-r - y_lat0) / h)) - 1, int(math.ceil((r - y_lat0) / h)) + 2)
        y_src = y_lat0 + m * h
        y_src = y_src[np.abs(y_src) <= r]
        A_src = -np.expm1(-mu * 2.0 * np.sqrt(np.clip(r * r - y_src ** 2, 0.0, None)))
        for k, z in enumerate(planes):
            sig = float(K.sigma_r(z, cfg.renderer))
            # the same samples blurred by hand: pixel mean of sum_m h A(y_m) g(y_s - y_m)
            g = np.exp(-0.5 * ((y_sub[:, :, None] - y_src[None, None, :]) / sig) ** 2) / (math.sqrt(2 * math.pi) * sig)
            disc = (h * g * A_src).sum(-1).mean(-1)
            e_disc = float(np.max(np.abs(col[k] - disc)) / disc.max())
            # SPEC.md's oracle: shadow_depths (the continuous integral), pixel means
            ref = CS.shadow_depths(np.abs(y_sub), r, mu, sig, 1.0).mean(-1)
            e_spec = float(np.max(np.abs(col[k] - ref)) / ref.max())
            rows.append("d %.1f z %.2f: vs hand-blurred samples %.2e, vs shadow_depths %.2e" % (d, z, e_disc, e_spec))
            worst_discrete = max(worst_discrete, e_disc)
            worst_spec = max(worst_spec, e_spec)
    MEASURED["known_answer flat vs shadow_depths (relative to peak)"] = rows
    assert worst_discrete <= 1e-10, "blur of the samples: %.3e" % worst_discrete
    # hybrid faint limit (amendment 2): A_q / (1.4470 mubar l_q) within 5e-3 of 1
    cfgh = cfg_for("hybrid")
    p0, p1, d, mu = (0.0, 0.0, 0.0), (3.0, 0.4, 1.0), 1.0, 1e-3
    fs = make_scene([(p0, p1, d, mu)])
    grid = R.FineGrid(x0=-1.0, y0=-1.0, h=P_X / 8, nx=400, ny=160)
    A, _ = SH.segment_absorbed(fs, 0, grid, cfgh)
    X, Y = grid.mesh()
    zlo, zhi, ins = G.column_interval(X, Y, tube_of(p0, p1, d))
    ell = zhi - zlo
    msk = ins & (ell > 0)
    ratio = A[msk] / (1.4470 * mu * ell[msk])
    dev = float(np.max(np.abs(ratio - 1.0)))
    MEASURED["known_answer hybrid faint |A/(1.4470 mu l) - 1| max"] = "%.3e over %d columns" % (dev, int(msk.sum()))
    assert dev <= 5e-3, "faint limit %.3e" % dev
    assert np.all(A[~ins] == 0.0)
    assert worst_spec <= 1e-6, ("flat segment vs calibration_scene.shadow_depths: %.3e relative (SPEC 1e-6); "
                                "the samples blurred by hand agree to %.1e; per case: %s"
                                % (worst_spec, worst_discrete, "; ".join(rows)))


def test_reference():
    # (1) hybrid A_q = column sum of render.absorbed_fractions(..., "hybrid", directions) [1e-10]
    for light in (+1, -1):
        cfg = cfg_for("hybrid")
        cfg = dataclasses.replace(cfg, acquisition=dataclasses.replace(cfg.acquisition, light_direction=light))
        p0, p1, d, mu = (0.0, 0.0, 0.2), (1.6, 0.9, 1.2), 0.6, 2.0
        fs = make_scene([(p0, p1, d, mu)])
        grid = R.FineGrid(x0=-0.6, y0=-0.6, h=P_X / 8, nx=200, ny=130)
        A, _ = SH.segment_absorbed(fs, 0, grid, cfg)
        tube = tube_of(p0, p1, d)
        dz = cfg.renderer.dzeta_um
        zeta = R.slab_grid(tube, grid, dz)
        dirs = RW.directions(cfg.cleaning.shadow_ray_n_rho, cfg.cleaning.shadow_ray_n_psi,
                             cfg.renderer.ray_na, cfg.renderer.ray_n_oil)
        ref = R.absorbed_fractions(tube, mu, grid, zeta, dz, light, "hybrid", dirs).sum(axis=0)
        check_close("hybrid_colsum_light%+d" % light, A, ref, rtol=0.0, atol=1e-10)
        assert A.max() > 0.3
    # (2) stretch_profiles = sample_profile on shadow_image's planes, same segments [1e-10]
    for absorption in ("partition_vertical", "hybrid"):
        cfg = cfg_for(absorption)
        fs = make_scene([((0.0, 0.0, 0.0), (2.0, 0.5, 0.3), (0.6, 0.8), 1.5, (11, 10)),
                         ((2.0, 0.5, 0.3), (4.0, 0.2, 0.6), (0.8, 0.7), (1.5, 1.2), (12, 11))])
        ks = np.arange(-2, 4)
        th = 0.1
        yh, eu = (-math.sin(th), math.cos(th)), (math.cos(th), math.sin(th))
        v = np.arange(-20, 21) * 0.05
        origin = (2.0, 0.4)
        prof = SH.stretch_profiles(fs, {10, 11, 12}, ks, origin, yh, eu, v, 2, P_X, cfg)
        left, top, W, H = -20, -30, 80, 60
        res = SH.shadow_image(fs, OWN_IDS, left, top, W, H, ks * cfg.acquisition.dz_um, cfg)
        frame = CropFrame(left, top, 0, P_X)
        ref = np.stack([sample_profile(res.S[k], frame, origin, yh, eu, v, 2, P_X) for k in range(ks.size)])
        assert np.all(np.isfinite(ref)) and ref.max() > 0.05
        check_close("stretch_profiles_" + absorption, prof, ref, rtol=0.0, atol=1e-10)
    # (3) the two PROVISIONAL bounds against Block 4 under the same absorption
    rows, worst = [], {"out": 0.0, "in": 0.0}
    times = {"partition_vertical": [], "hybrid": []}
    for absorption in ("partition_vertical", "hybrid"):
        cfg = cfg_for(absorption)
        rc4 = dataclasses.replace(cfg.renderer, absorption=absorption, ray_n_rho=cfg.cleaning.shadow_ray_n_rho,
                                  ray_n_psi=cfg.cleaning.shadow_ray_n_psi)
        for d in (0.3, 0.6, 1.0):
            for tilt in (0.0, 30.0):
                out_r, in_r, ts, t4 = _block4_case(cfg, rc4, d, math.radians(tilt))
                times[absorption].append(ts)
                rows.append("%-18s d %.1f tilt %2d: out-of-focus %.4f, in focus %.4f (x peak); shadow %.2f s, "
                            "Block 4 %.2f s" % (absorption, d, tilt, out_r, in_r, ts, t4))
                worst["out"] = max(worst["out"], out_r)
                worst["in"] = max(worst["in"], in_r)
    MEASURED["Block 4 bounds (mu 1/um, segment 4 um)"] = rows
    MEASURED["time per segment (s), 4 um segment, d 0.3/0.6/1.0 x tilt 0/30"] = [
        "%s: %s" % (a, ", ".join("%.2f" % t for t in ts)) for a, ts in times.items()]
    assert worst["out"] <= 0.03 and worst["in"] <= 0.15, (
        "PROVISIONAL bounds: out-of-focus %.4f (<= 0.03), in focus %.4f (<= 0.15)" % (worst["out"], worst["in"]))


def _block4_case(cfg, rc4, d, phi, mu=1.0, U=2.0):
    """(out-of-focus ratio, in-focus ratio, shadow time, Block 4 time) for one capped tube."""
    t = np.array([math.cos(phi), 0.0, math.sin(phi)])
    p0, p1 = -U * t, U * t
    fs = make_scene([(tuple(p0), tuple(p1), d, mu)])
    tube = tube_of(p0, p1, d)
    left, top, W, H = -56, -45, 113, 91
    deltas = np.array([-1.68, -0.84, -0.28, 0.0, 0.28, 0.84, 1.68])
    zk = deltas.copy()
    t0 = time.perf_counter()
    S = SH.shadow_image(fs, OWN_IDS, left, top, W, H, zk, cfg).S
    ts = time.perf_counter() - t0
    grid = patch_grid(left, top, W, H)
    t0 = time.perf_counter()
    tau = R.render_transmittance(tube, mu, zk, grid, rc4, cfg.acquisition.light_direction,
                                 reduce=lambda pl: R.pixel_integrate(pl, 8)).tau
    t4 = time.perf_counter() - t0
    D4 = 1.0 - tau
    xc = (left + np.arange(W)) * P_X
    yc = (top + np.arange(H)) * P_X
    Xc, Yc = np.meshgrid(xc, yc)
    u = Xc * math.cos(tube.theta) + Yc * math.sin(tube.theta)
    z_ax = np.clip(u, -U * math.cos(phi), U * math.cos(phi)) * math.tan(phi)
    out_r, in_r = 0.0, float("nan")
    for k, z in enumerate(zk):
        peak = float(D4[k].max())
        diff = np.abs(S[k] - D4[k])
        far = np.abs(z_ax - z) >= 0.84 - 1e-9
        if far.any():
            out_r = max(out_r, float(diff[far].max()) / peak)
        if z == 0.0:
            in_r = float(diff.max()) / peak
    return out_r, in_r, ts, t4


def test_convergence():
    rows = []
    for absorption, d in (("partition_vertical", 0.6), ("hybrid", 0.3)):
        imgs = []
        for b in (0.07, 0.035):
            cfg = cfg_for(absorption, shadow_depth_bin_um=b)
            s = 3.0 / math.sqrt(2.0)
            fs = make_scene([((0.0, 0.0, 0.0), (s, 0.0, s), d, 1.5)])
            imgs.append(SH.shadow_image(fs, OWN_IDS, -40, -40, 100, 80, np.arange(-2, 11) * 0.28, cfg).S)
        peak = float(imgs[0].max())
        move = float(np.max(np.abs(imgs[1] - imgs[0]))) / peak
        rows.append("%s d %.1f: %.3e x peak" % (absorption, d, move))
        assert move < 1e-3, "%s: halving the depth bin moves the image by %.3e x peak" % (absorption, move)
    MEASURED["bin halving, 45-degree segment"] = rows


def test_invariants():
    # totals per plane equal across planes and to the integral of A_q [relative 1e-8]
    left, top, W, H = -110, -110, 220, 220
    planes = np.array([-0.56, -0.28, 0.0, 0.28, 0.56])
    for absorption, d in (("partition_vertical", 0.8), ("hybrid", 0.3)):
        cfg = cfg_for(absorption)
        p0, p1 = (0.0, 0.0, 0.0), (2.0, 1.0, 0.8)
        fs = make_scene([(p0, p1, d, 1.2)], soma=[(-3.0, 2.0, 1.5)])
        res = SH.shadow_image(fs, OWN_IDS, left, top, W, H, planes, cfg)
        grid = patch_grid(left, top, W, H)
        A_seg, _ = SH.segment_absorbed(fs, 0, grid, cfg)
        X, Y = grid.mesh()
        ell = 2.0 * np.sqrt(np.clip(1.5 ** 2 - (X + 3.0) ** 2 - (Y - 2.0) ** 2, 0.0, None))
        A_soma = -np.expm1(-cfg.cleaning.soma_mu_per_um * ell)
        integral = (A_seg.sum() + A_soma.sum()) * grid.h ** 2
        totals = res.S.sum(axis=(1, 2)) * P_X ** 2
        check_close("totals_" + absorption, totals, np.full(planes.size, integral), rtol=1e-8)
    # additivity over segments [1e-12]
    cfg = cfg_for("partition_vertical")
    fs = make_scene([((0.0, 0.0, 0.0), (2.0, 0.5, 0.3), 0.6, 1.5), ((0.5, 1.0, -0.6), (2.5, 1.4, 0.2), 0.9, 0.8)],
                    soma=[(-2.0, -1.0, 1.0)])
    planes = np.array([-0.28, 0.0, 0.56])
    both = SH.shadow_image(fs, OWN_IDS, -30, -30, 60, 60, planes, cfg)
    parts = [SH.shadow_image(fs, OWN_IDS, -30, -30, 60, 60, planes, cfg, include=m).S
             for m in ([True, False, False], [False, True, False], [False, False, True])]
    check_close("additivity", both.S, parts[0] + parts[1] + parts[2], rtol=0.0, atol=1e-12)
    assert both.kept.tolist() == [True, True, True]
    # include=False leaves the neighbour out of S, its peak still computed, not kept
    r1 = SH.shadow_image(fs, OWN_IDS, -30, -30, 60, 60, planes, cfg, include=[True, False, True])
    assert r1.kept.tolist() == [True, False, True]
    assert_array_equal(r1.per_neighbour_peak, both.per_neighbour_peak)
    check_close("include_false", r1.S, parts[0] + parts[2], rtol=0.0, atol=1e-12)
    # kept follows the floor (B_gl x peak >= shadow_floor_gl), S holds exactly the kept ones
    fs2 = make_scene([((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), 0.6, 1.5),
                      ((0.0, 2.0, 0.0), (2.0, 2.0, 0.0), 0.3, 1e-3),
                      ((0.0, -2.0, 4.0), (2.0, -2.0, 4.0), 0.6, 0.02)])
    B = 210.0
    rB = SH.shadow_image(fs2, OWN_IDS, -30, -30, 60, 60, planes, cfg, B_gl=B)
    floor = cfg.cleaning.shadow_floor_gl
    assert_array_equal(rB.kept, B * rB.per_neighbour_peak >= floor)
    assert rB.kept.tolist() == [True, False, False], (rB.kept, B * rB.per_neighbour_peak)
    rK = SH.shadow_image(fs2, OWN_IDS, -30, -30, 60, 60, planes, cfg, include=rB.kept)
    assert_array_equal(rB.S, rK.S)
    # overlap: true for a segment directly above the own stretch, false 3 um aside
    own = [((-3.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.6, 1.0, (2, 1)), ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), 0.6, 1.0, (3, 2))]
    above = make_scene(own + [((0.0, -3.0, 3.0), (0.0, 3.0, 3.0), 0.6, 1.0, (51, 50))], own=False)
    aside = make_scene(own + [((-3.0, 3.0, 0.0), (3.0, 3.0, 0.0), 0.6, 1.0, (51, 50))], own=False)
    for name, fsx, expect in (("above", above, True), ("aside", aside, False)):
        rx = SH.shadow_image(fsx, {1, 2, 3}, -40, -40, 80, 80, [0.0, 2.8], cfg)
        assert rx.kept.tolist() == [True]
        assert bool(rx.overlap.any()) is expect, name
        if expect:
            rr, cc = np.nonzero(rx.overlap)
            xs, ys = (cc - 40) * P_X, (rr - 40) * P_X
            assert np.all(np.abs(xs) <= 0.3 + P_X) and np.all(np.abs(ys) <= 0.3 + P_X), "overlap off the crossing"


def test_contract():
    cfg = cfg_for("partition_vertical")
    fs = make_scene([((0.0, 0.0, 0.0), (2.0, 0.5, 0.3), 0.6, 1.5, (2, 1)),
                     ((2.0, 0.5, 0.3), (3.0, 1.5, 0.3), 0.5, 1.5, (3, 2)),
                     ((0.0, 0.0, 0.0), (-1.0, 1.0, 0.2), 0.7, 1.0, (4, 1)),
                     ((-1.0, 1.0, 0.2), (-2.0, 2.0, 0.2), 0.7, 1.0, (5, 4))],
                    soma=[(-4.0, -3.0, 2.0)], own=False)
    segs, somas = SH.neighbour_indices(fs, {1, 2, 3})
    assert segs.tolist() == [2, 3] and somas.tolist() == [0], (segs, somas)
    assert segs.dtype.kind == "i" and somas.dtype.kind == "i"
    planes = np.array([-0.28, 0.0, 0.28, 0.56])
    res = SH.shadow_image(fs, {1, 2, 3}, -30, -20, 50, 40, planes, cfg, B_gl=210.0)
    assert isinstance(res, SH.ShadowResult)
    assert res.S.shape == (4, 40, 50) and res.S.dtype == np.float64 and np.all(np.isfinite(res.S))
    assert res.per_neighbour_peak.shape == (3,) and res.per_neighbour_peak.dtype == np.float64
    assert res.kept.shape == (3,) and res.kept.dtype == bool
    assert res.overlap.shape == (40, 50) and res.overlap.dtype == bool
    assert np.all(res.S >= 0.0) and np.all(res.per_neighbour_peak > 0.0)
    grid = R.FineGrid(x0=-1.0, y0=-1.0, h=P_X / 8, nx=50, ny=30)
    A, z = SH.segment_absorbed(fs, 0, grid, cfg)
    assert A.shape == (30, 50) and z.shape == (30, 50) and A.dtype == np.float64 and np.all(np.isfinite(z))
    prof = SH.stretch_profiles(fs, {1, 2, 3}, [0, 1, 2], (1.0, 0.25), (0.0, 1.0), (1.0, 0.0),
                               np.arange(-10, 11) * 0.05, 0, 0.0, cfg)
    assert prof.shape == (3, 21) and prof.dtype == np.float64 and np.all(np.isfinite(prof))


def test_determinism():
    for absorption in ("partition_vertical", "hybrid"):
        cfg = cfg_for(absorption)
        fs = make_scene([((0.0, 0.0, 0.0), (1.5, 0.5, 0.6), 0.4, 1.5)], soma=[(-2.0, 1.0, 0.8)])
        a = SH.shadow_image(fs, OWN_IDS, -20, -20, 40, 40, [0.0, 0.28], cfg, B_gl=210.0)
        b = SH.shadow_image(fs, OWN_IDS, -20, -20, 40, 40, [0.0, 0.28], cfg, B_gl=210.0)
        for f in ("S", "per_neighbour_peak", "kept", "overlap"):
            assert_array_equal(getattr(a, f), getattr(b, f))


def test_edge_cases():
    cfg = cfg_for("partition_vertical")
    planes = np.array([0.0, 0.28])
    sig_max = float(np.max(K.sigma_r(planes, cfg.renderer)))
    reach = cfg.cleaning.shadow_reach_sigmas * sig_max
    # a segment beyond reach of the patch contributes exactly 0 (patch x in [-2.35, 2.23] um)
    x_far = 2.23 + reach + 0.3 + 0.05
    fs = make_scene([((x_far, -1.0, 0.0), (x_far, 1.0, 0.0), 0.6, 2.0)])
    res = SH.shadow_image(fs, OWN_IDS, -20, -20, 40, 40, planes, cfg)
    assert np.all(res.S == 0.0) and res.per_neighbour_peak[0] == 0.0
    fs_near = make_scene([((x_far - 0.2, -1.0, 0.0), (x_far - 0.2, 1.0, 0.0), 0.6, 2.0)])
    assert SH.shadow_image(fs_near, OWN_IDS, -20, -20, 40, 40, planes, cfg).S.max() > 0.0
    for absorption in ("partition_vertical", "hybrid"):
        c = cfg_for(absorption)
        fz = make_scene([((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.6, 0.0)])
        rz = SH.shadow_image(fz, OWN_IDS, -20, -20, 40, 40, planes, c)
        assert np.all(rz.S == 0.0) and rz.per_neighbour_peak[0] == 0.0 and not rz.overlap.any()
    fs = make_scene([((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.6, 1.0)])
    bad = [
        (lambda: SH.shadow_image(fs, {7, 8}, -20, -20, 40, 40, planes, cfg), "own stretch without segment"),
        (lambda: SH.shadow_image(fs, {900}, -20, -20, 40, 40, planes, cfg), "own stretch of one node"),
        (lambda: SH.shadow_image(fs, OWN_IDS, -20, -20, 40, 40, [0.0, np.nan], cfg), "NaN plane"),
        (lambda: SH.shadow_image(fs, OWN_IDS, -20, -20, 40, 40, [np.inf], cfg), "inf plane"),
        (lambda: SH.shadow_image(fs, OWN_IDS, -20, -20, 40, 40, [], cfg), "no plane"),
        (lambda: SH.shadow_image(fs, OWN_IDS, -20, -20, 40, 40, planes, cfg, include=[True, True]), "include"),
        (lambda: SH.shadow_image(fs, OWN_IDS, -20, -20, 40, 40, planes, cfg, B_gl=0.0), "B_gl 0"),
        (lambda: SH.shadow_image(fs, OWN_IDS, -20, -20, 40, 40, planes, cfg_for("linear")), "absorption"),
        (lambda: SH.stretch_profiles(fs, {7, 8}, [0], (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), [0.0], 0, 0.0, cfg),
         "stretch without segment"),
    ]
    for fn, what in bad:
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("no ValueError for " + what)


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

    for key, val in MEASURED.items():
        print("\n-- measured: " + key)
        for line in ([val] if isinstance(val, str) else val):
            print("   " + line)

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
