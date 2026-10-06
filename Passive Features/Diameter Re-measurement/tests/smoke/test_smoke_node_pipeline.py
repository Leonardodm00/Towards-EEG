"""Smoke test for the per-node chain -- Block 5 in specs/SPEC.md
(handoff Eqs. 1-5, 9; D-018.1; D-023; D-024).

Checks
    test_known_answer   bilinear profiles of a linear ramp are exact, and NaN
                        off the block; Eq. 2 vertex on a sampled parabola;
                        plateau middle; TLS direction from noisy points;
                        y_hat perpendicular to the branch; mask distances;
                        GATE 1 (procedure s.3.10, first row): single-depth
                        phantoms, phi = 0, sigma_fit^2 = sigma_r(0)^2 + p_x^2/4
                        give d_hat / d within 2 % of 1
    test_reference      sample_profile vs allen_image_measure.line_profile at
                        the same points; focus_score vs the formula written out
    test_convergence    skipped: Block 5 has no discretisation parameter
    test_invariants     D6 guard (pixel/plane units change the tilt); rendered
                        phantoms with the default kernel: theta within 2 deg,
                        phi within 4 deg, d_hat / d in a plausible band
    test_contract       NodeResult carries the Block 8 columns with their types
    test_determinism    one seed, one result
    test_edge_cases     empty tissue -> faint; a second tube -> crossing; planes
                        missing below the node -> stack_edge; a near-vertical
                        tube -> vertical; no usable plane -> k_star -1; bad Branch

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_node_pipeline.py

Keep this file pure ASCII (hpc-python-compat).
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

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import allen_image_measure  # noqa: E402
from allen_image_io import CropFrame  # noqa: E402
from allen_diameter.analysis import background as BG  # noqa: E402
from allen_diameter.analysis import focus as FO  # noqa: E402
from allen_diameter.analysis import node_pipeline as NP  # noqa: E402
from allen_diameter.analysis import path as PA  # noqa: E402
from allen_diameter.analysis import profiles as PR  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import camera as C  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy")
PX = 0.1144


def small_cfg(**renderer):
    """Defaults with a smaller block and p_x/8 for thin tubes, to keep the rendered checks short."""
    base = default_config()
    rc = dataclasses.replace(base.renderer, h_g_um_thin=PX / 8, **renderer)
    return dataclasses.replace(base, renderer=rc, measure=dataclasses.replace(base.measure, block_half_um=3.5))


def gate_cfg():
    """Single-depth rendering (one slab, constant kernel sigma_r(0)), camera without noise or JPEG,
    sigma_fit^2 = sigma_r(0)^2 + p_x^2/4 (the heuristic budget of procedure s.3.10)."""
    base = small_cfg(dzeta_um=10.0, kernel_continuation="frozen", noise_sd_gl=0.0, jpeg=False)
    s0 = base.renderer.sigma_r0_um
    rc = dataclasses.replace(base.renderer, kernel_table_sigma_um=(s0,) * len(base.renderer.kernel_table_sigma_um))
    ms = dataclasses.replace(base.measure, sigma_fit_um=math.sqrt(s0 ** 2 + PX ** 2 / 4))
    return dataclasses.replace(base, renderer=rc, measure=ms)


def phantom_branch(c, phi, theta, r, n_each=4, step=1.18):
    t = np.array([math.cos(phi) * math.cos(theta), math.cos(phi) * math.sin(theta), math.sin(phi)])
    s = np.arange(-n_each, n_each + 1) * step
    return NP.Branch.from_points(np.asarray(c, float)[None, :] + s[:, None] * t[None, :], r)


def make_provider(tubes, mus, cfg, seed, pad_px=8, invalid_below=None):
    """A synthetic block provider for one or more tubes (transmittances multiply)."""
    rng = np.random.default_rng(seed)
    acq, rc = cfg.acquisition, cfg.renderer

    def provider(left, top, width, height, k_lo, k_hi):
        ks = np.arange(k_lo, k_hi + 1)
        f = R.fine_factor(min(t.d for t in tubes), rc, acq.res0_um)
        grid, inner = R.block_fine_grid(left, top, width, height, acq.res0_um, f, pad_px)
        tau = np.ones((ks.size, grid.ny, grid.nx))
        for t, mu in zip(tubes, mus):
            tau *= R.render_transmittance(t, mu, ks * acq.dz_um, grid, rc, acq.light_direction).tau
        block = C.camera_chain(R.pixel_integrate(tau[:, inner[0], inner[1]], f), rc, rng)
        valid = np.ones(ks.size, dtype=bool)
        if invalid_below is not None:
            valid[ks < invalid_below] = False
            block[ks < invalid_below] = 255
        return block, ks, valid, CropFrame(int(left), int(top), 0, acq.res0_um)
    return provider


def run_phantom(d, phi_deg, theta, cfg, seed=1, mu=1.0, c=(0.03, -0.02, 0.05), U=6.0, **kw):
    phi = math.radians(phi_deg)
    tube = G.Tube(c, 0.5 * d, phi, theta, 1.0, U, "axial")
    return NP.measure_node(phantom_branch(c, phi, theta, 0.5 * d), 4, make_provider([tube], [mu], cfg, seed, **kw), cfg)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    m = default_config().measure
    # bilinear profiles of a linear ramp are exact; samples off the block are NaN
    H, W = 40, 50
    rows, cols = np.mgrid[0:H, 0:W]
    plane = 100.0 + 0.7 * cols - 0.3 * rows
    frame = CropFrame(10, -5, 0, PX)
    o, th = (np.array([3.1, 1.2])), 0.4
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    v = np.linspace(-1.0, 1.0, 21)
    got = PR.sample_profile(plane, frame, o, y_hat, e_u, v, n_avg=2, step=PX)
    x, y = o[0] + v * y_hat[0], o[1] + v * y_hat[1]
    want = 100.0 + 0.7 * (x / PX - frame.left) - 0.3 * (y / PX - frame.top)
    assert np.max(np.abs(got - want)) <= 1e-9, "ramp"
    assert np.isnan(PR.sample_profile(plane, frame, o, y_hat, e_u, np.array([50.0]))[0]), "off-block sample"
    # Eq. 2: vertex of a sampled parabola within +-dz/2 of k*, exact
    dz = 0.28
    z = np.arange(-4, 5) * dz
    for z0 in (0.0, 0.05, -0.13, 0.139):
        k, zs, plateau, edge = FO.best_plane(5.0 - 10.0 * (z - z0) ** 2, z, m, dz)   # neighbours >= 15 % below
        assert abs(zs - z0) <= 1e-12 and abs(zs - z[k]) <= 0.5 * dz + 1e-12 and not plateau and not edge, (z0, zs)
    k, zs, plateau, _ = FO.best_plane(np.array([1.0, 2.0, 3.0, 3.0, 3.0, 2.0, 1.0]), np.arange(7) * dz, m, dz)
    assert plateau and abs(zs - 3 * dz) <= 1e-12, ("plateau", zs)
    _k, _z, _p, edge = FO.best_plane(np.array([np.nan, np.nan, 3.0, 2.0, 1.0]), np.arange(5) * dz, m, dz)
    assert edge, "k* at the first usable plane is at the edge"
    # TLS direction from noisy points (D6: um), and the axes of Eq. 5
    rng = np.random.default_rng(SEED)
    t_true = np.array([math.cos(0.3) * math.cos(2.0), math.cos(0.3) * math.sin(2.0), math.sin(0.3)])
    pts = np.linspace(-2, 2, 9)[:, None] * t_true[None, :] + rng.normal(0, 0.01, (9, 3))
    t = PA.orient(PA.tls_direction(pts), pts[0], pts[-1])
    assert math.degrees(math.acos(min(1.0, float(np.dot(t, t_true))))) < 0.5, t
    for theta in rng.uniform(-math.pi, math.pi, 20):
        tt = np.array([math.cos(theta), math.sin(theta), 0.7])
        th, phi, yh, eu = PA.angles(tt / np.linalg.norm(tt))
        assert abs(float(np.dot(yh, tt[:2]))) <= 1e-12 and abs(abs(float(np.dot(eu, tt[:2]))) - np.linalg.norm(tt[:2])) <= 1e-12
    # mask distances on a hand-made segment (0,0)-(2,0), radius 0.5, margin 1.0 -> 1.5 um
    pts_x = np.array([1.0, 1.0, 3.2, 3.6, -1.4, -1.6])
    pts_y = np.array([1.4, 1.6, 0.0, 0.0, 0.0, 0.0])
    dist = BG.segment_distance(pts_x, pts_y, (0.0, 0.0), (2.0, 0.0))
    assert np.allclose(dist, [1.4, 1.6, 1.2, 1.6, 1.4, 1.6], atol=1e-12)
    # pixel centres x = 0..5 on y = 0; segment (-1, 0)-(2.6, 0): distances 0, 0, 0, 0.4, 1.4, 2.4 vs 1.5
    mask = BG.mask_near_branch((1, 6), 0, 0, 1.0, np.array([[-1.0, 0, 0], [2.6, 0, 0]]), [0.5, 0.5], 1.0)
    assert list(mask[0]) == [True, True, True, True, True, False], mask
    # GATE 1: single-depth phantoms, phi = 0
    cfg = gate_cfg()
    rng = np.random.default_rng(SEED)
    ratios = []
    for d in (0.5, 0.75, 1.0):
        for rep in range(2):
            off = rng.uniform(-0.5, 0.5, 2) * PX
            res = run_phantom(d, 0.0, rng.uniform(0, math.pi), cfg, seed=rep, c=(off[0], off[1], rng.uniform(-0.14, 0.14)))
            ratios.append(res.d_hat_um / d)
    ratios = np.array(ratios)
    assert abs(ratios.mean() - 1) <= 0.02 and np.all(np.abs(ratios - 1) <= 0.03), ("gate 1", ratios)


def test_reference():
    rng = np.random.default_rng(SEED)
    plane = rng.uniform(50, 200, (60, 70))
    frame = CropFrame(0, 0, 0, PX)
    o, th = np.array([4.0, 3.4]), 1.1
    y_hat, e_u = np.array([-math.sin(th), math.cos(th)]), np.array([math.cos(th), math.sin(th)])
    v = PR.profile_offsets(default_config().measure)
    mine = PR.sample_profile(plane, frame, o, y_hat, e_u, v)
    c0, r0 = (o + v[0] * y_hat) / PX
    c1, r1 = (o + v[-1] * y_hat) / PX
    _t, theirs = allen_image_measure.line_profile(plane, (c0, r0), (c1, r1), PX, n=v.size)
    assert np.max(np.abs(mine - theirs)) <= 1e-4, np.max(np.abs(mine - theirs))   # they interpolate in float32
    m = default_config().measure
    I = 200.0 - 80.0 * np.exp(-0.5 * (v / 0.3) ** 2) + rng.normal(0, 2.0, v.size)
    F, I_min, B = FO.focus_score(I, v, m)
    w = np.exp(-0.5 * (np.arange(-4, 5) / m.focus_smooth_px) ** 2)
    smooth = np.convolve(np.pad(I, 4, mode="edge"), w / w.sum(), mode="valid")
    B_want = float(np.median(I[np.abs(v) > m.focus_bg_ends_um]))
    assert abs(B - B_want) <= 1e-12 and abs(I_min - smooth.min()) <= 1e-9 and abs(F + math.log(smooth.min() / B_want)) <= 1e-9


def test_convergence():
    raise unittest.SkipTest("Block 5 has no discretisation parameter; grid and slab convergence are Block 4's")


def test_invariants():
    # D6 guard: the same centres in pixel/plane units give a different tilt
    t = np.array([math.cos(math.radians(20.0)), 0.0, math.sin(math.radians(20.0))])
    pts = np.linspace(-2, 2, 5)[:, None] * t[None, :]
    phi_um = PA.angles(PA.tls_direction(pts))[1]
    idx = pts / np.array([PX, PX, 0.28])
    phi_idx = PA.angles(PA.tls_direction(idx))[1]
    assert abs(math.degrees(phi_um) - 20.0) <= 1e-9 and abs(math.degrees(phi_idx) - 20.0) > 5.0, (phi_um, phi_idx)
    # rendered phantoms, default Debye-core kernel: geometry recovered; d_hat within the known halo band
    cfg = small_cfg()
    for d, phi_deg, theta in ((1.0, 0.0, 0.6), (1.0, 15.0, 2.2)):
        res = run_phantom(d, phi_deg, theta, cfg)
        dth = (math.degrees(res.theta_rad - theta) + 90.0) % 180.0 - 90.0
        # the tilt rests on sub-plane depths (noise 3 gl + JPEG) over a 4 um window: a few degrees
        # (handoff Eq. 2 note); phi = arcsin|t_z| is folded at 0, so a flat tube reads slightly tilted
        assert abs(dth) <= 2.0 and abs(math.degrees(res.phi_rad) - phi_deg) <= 4.0, (d, phi_deg, dth, math.degrees(res.phi_rad))
        assert 0.95 <= res.d_hat_um / d <= 1.25 and res.fit_status == "converged", (d, phi_deg, res.d_hat_um, res.flags)


def test_contract():
    res = run_phantom(0.75, 0.0, 0.3, gate_cfg())
    cols = ("node_id, type, x_um, y_um, z_um, path_um, reg_verdict, s_star_um, dz_star_um, k_star, z_sub_um, cx_um, "
            "cy_um, cz_um, theta_rad, phi_rad, steep, vertical, B_bar, B_bar_region, d_hat_um, mu_hat_per_um, "
            "v0_hat_um, alpha_hat, fit_status, flags").split(", ")
    for name in cols:
        assert hasattr(res, name), name
    for name in ("x_um", "y_um", "z_um", "path_um", "z_sub_um", "cx_um", "cy_um", "cz_um", "theta_rad", "phi_rad",
                 "B_bar", "d_hat_um", "mu_hat_per_um", "v0_hat_um", "alpha_hat"):
        assert isinstance(getattr(res, name), float), name
    assert isinstance(res.k_star, int) and isinstance(res.flags, tuple) and all(isinstance(f, str) for f in res.flags)
    assert isinstance(res.steep, bool) and isinstance(res.vertical, bool) and res.B_bar_region == "block_masked"
    assert res.fit is not None and res.focus_F.ndim == 1 and math.isnan(res.s_star_um) and res.reg_verdict == ""


def test_determinism():
    a = run_phantom(0.75, 0.0, 0.3, gate_cfg(), seed=3)
    b = run_phantom(0.75, 0.0, 0.3, gate_cfg(), seed=3)
    assert (a.d_hat_um, a.cx_um, a.cy_um, a.cz_um, a.flags) == (b.d_hat_um, b.cx_um, b.cy_um, b.cz_um, b.flags)


def test_edge_cases():
    cfg = small_cfg()
    empty = run_phantom(0.8, 0.0, 0.5, cfg, mu=0.0)
    assert "faint" in empty.flags, empty.flags
    # a second, parallel tube 1.8 um away inside the profile window
    th = 0.5
    c1 = np.array([0.03, -0.02, 0.05])
    c2 = c1 + 1.8 * np.array([-math.sin(th), math.cos(th), 0.0])
    tubes = [G.Tube(tuple(c1), 0.4, 0.0, th, 1.0, 6.0, "axial"), G.Tube(tuple(c2), 0.4, 0.0, th, 1.0, 6.0, "axial")]
    res = NP.measure_node(phantom_branch(c1, 0.0, th, 0.4), 4, make_provider(tubes, [1.0, 1.0], cfg, 1), cfg)
    assert "crossing" in res.flags, res.flags
    # planes below the node missing: the focus search is cut short
    tube = G.Tube((0.0, 0.0, 0.05), 0.4, 0.0, th, 1.0, 6.0, "axial")
    br = phantom_branch((0.0, 0.0, 0.05), 0.0, th, 0.4)
    res = NP.measure_node(br, 4, make_provider([tube], [1.0], cfg, 1, invalid_below=0), cfg)
    assert "stack_edge" in res.flags, (res.flags, res.focus_F)
    res = NP.measure_node(br, 4, make_provider([tube], [1.0], cfg, 1, invalid_below=99), cfg)
    assert res.k_star == -1 and res.fit_status == "none"
    # a near-vertical tube
    res = run_phantom(0.8, 88.0, 0.5, small_cfg(), U=2.5)
    assert res.vertical and "vertical" in res.flags and math.degrees(res.phi_rad) > 85.0, (math.degrees(res.phi_rad), res.flags)
    for bad in (lambda: NP.Branch(np.arange(1), np.array([3]), np.zeros((1, 3)), np.ones(1), np.zeros(1)),
                lambda: NP.Branch(np.arange(2), np.array([3, 3]), np.zeros((2, 3)), np.ones(2), np.array([1.0, 0.0]))):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("invalid Branch accepted")


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
