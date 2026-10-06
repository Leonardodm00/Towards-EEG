"""Smoke test for the ray world -- Block 9 in specs/SPEC.md (impl-handoff (S6), (S10)).

Checks
    test_known_answer   the direction weights sum to 1 and <1/cos theta> equals
                        (2 / s_m^2)(1 - sqrt(1 - s_m^2)) = 1.447 (S10) to 1e-10;
                        an opaque thick flat tube is black at its centre in the
                        node plane
    test_reference      ray_transmittance equals checks/optics_points_check.py's
                        ray_models (its G column) on the same directions, to 1e-12
    test_convergence    <exp(-mu L)> at a point changes by < 1e-3 when the
                        direction grid is doubled
    test_invariants     faint limit (S10), against the analytic area mu pi r^2:
                        the ray world's dip area is <1/cos theta> mu pi r^2 and
                        the partition renderer's is mu pi r^2, each within 0.5 %;
                        a flat tube's dips at +-3 dz are equal (the ray world is
                        symmetric in depth); a point's value does not depend on
                        the grid: a tube whose footprint lies outside a small
                        grid still darkens it through oblique rays, exactly as
                        the same points of a grid that holds the tube
    test_contract       render_transmittance with absorption "ray_world" returns
                        planes of the grid's shape (and reduced ones)
    test_determinism    skipped: deterministic quadrature
    test_edge_cases     a tube without an end cut is refused

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_ray_world.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import importlib.util
import io
import contextlib
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

from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import ray_world as RW  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy")
PX = 0.1144


def rw_cfg(**changes):
    base = dict(absorption="ray_world")
    base.update(changes)
    return dataclasses.replace(default_config().renderer, **base)


def load_optics_check():
    path = WS / "checks" / "optics_points_check.py"
    spec = importlib.util.spec_from_file_location("optics_points_check", str(path))
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def profile_across(rc, tube, mu, zk, half=2.5, h=PX / 8, x_half=0.5):
    """Plane zk on a strip across a tube that runs along x (theta = 0): the middle
    column, as a 1-D profile in y. The partition renderer draws the object on the
    grid only, so its strip must be several kernel widths long in x (x_half); the
    ray world traces the whole tube whatever the grid."""
    ny = int(round(2 * half / h)) + 1
    grid = R.FineGrid(-x_half, -half, h, int(round(2 * x_half / h)) + 1, ny)
    tau = R.render_transmittance(tube, mu, [zk], grid, rc).tau[0]
    return grid.ys, tau[:, tau.shape[1] // 2]


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    rc = rw_cfg()
    S, W = RW.directions(rc.ray_n_rho, rc.ray_n_psi, rc.ray_na, rc.ray_n_oil)
    s_m = rc.ray_na / rc.ray_n_oil
    assert abs(W.sum() - 1) <= 1e-14 and np.allclose(np.linalg.norm(S, axis=1), 1.0, atol=1e-14)
    exact = 2.0 / s_m ** 2 * (1.0 - math.sqrt(1.0 - s_m ** 2))
    assert abs(RW.mean_inverse_cos(S, W) - exact) <= 1e-10 and abs(exact - 1.447) <= 5e-4, (RW.mean_inverse_cos(S, W), exact)
    tube = G.Tube((0.0, 0.0, 0.0), 0.5, 0.0, 0.0, 1.0, 4.0, "axial")
    ys, prof = profile_across(rw_cfg(ray_post_sigma_um=0.0), tube, 50.0, 0.0)
    assert prof[np.argmin(np.abs(ys))] < 1e-3, prof.min()


def test_reference():
    opc = load_optics_check()
    S, W = opc.directions(16, 32)
    for r, phi_deg, mu, v0, zk in ((0.5, 0.0, 1.0, 0.1, 0.2), (0.25, 20.0, 1.5, -0.2, -0.3), (0.4, 40.0, 0.7, 0.0, 0.0)):
        phi = math.radians(phi_deg)
        theirs = opc.ray_models(v0, zk, r, phi, mu, S, W, U=6.0)[3]
        tube = G.Tube((0.0, 0.0, 0.0), r, phi, 0.0, 1.0, 6.0, "vertical")
        mine = RW.ray_transmittance(np.array([[0.0, v0, zk]]), tube, mu, S, W)[0]
        assert abs(mine - theirs) <= 1e-12, (r, phi_deg, mine, theirs)


def test_convergence():
    rc = rw_cfg()
    tube = G.Tube((0.0, 0.0, 0.0), 0.4, math.radians(25.0), 0.3, 1.0, 4.0, "axial")
    pts = np.array([[0.05, 0.1, 0.0], [0.0, 0.6, 0.28], [-0.3, -0.2, -0.56]])
    a = RW.ray_transmittance(pts, tube, 1.0, *RW.directions(rc.ray_n_rho, rc.ray_n_psi, rc.ray_na, rc.ray_n_oil))
    b = RW.ray_transmittance(pts, tube, 1.0, *RW.directions(2 * rc.ray_n_rho, 2 * rc.ray_n_psi, rc.ray_na, rc.ray_n_oil))
    assert np.max(np.abs(a - b)) <= 1e-3, np.abs(a - b)


def test_invariants():
    # Faint limit: for every direction s, the integral over v of the chord of a
    # tube along x is pi r^2 / s_z, so the ray world's dip area is
    # <1/cos theta> mu pi r^2 (S10) at any plane; the partition renderer's kernels
    # are normalised, so its dip area is mu pi r^2. Measured at h = p_x/8:
    # ray / analytic = 1.44627 vs 1.44700 (0.05 %, the sqrt edges of the
    # unblurred chord profile sampled on the grid), partition / analytic =
    # 0.99991. 0.5 % leaves an order of magnitude of headroom. The partition
    # strip is 6 um long in x: the partition draws the object on the grid only,
    # and a 1 um strip loses ~1 % of the area to the kernels' tails.
    r, mu = 0.5, 1e-3
    tube = G.Tube((0.0, 0.0, 0.0), r, 0.0, 0.0, 1.0, 6.0, "axial")
    exact = mu * math.pi * r ** 2
    rc = rw_cfg()
    S, W = RW.directions(rc.ray_n_rho, rc.ray_n_psi, rc.ray_na, rc.ray_n_oil)
    ys, ray = profile_across(rc, tube, mu, 0.0, half=4.0)
    ys2, part = profile_across(dataclasses.replace(default_config().renderer), tube, mu, 0.0, half=4.0, x_half=3.0)
    a_ray, a_part = np.trapezoid(1 - ray, ys), np.trapezoid(1 - part, ys2)
    assert abs(a_ray / (RW.mean_inverse_cos(S, W) * exact) - 1) <= 5e-3, (a_ray / exact, RW.mean_inverse_cos(S, W))
    assert abs(a_part / exact - 1) <= 5e-3, a_part / exact
    _, lo = profile_across(rw_cfg(), tube, 1.0, -3 * 0.28)
    _, hi = profile_across(rw_cfg(), tube, 1.0, +3 * 0.28)
    assert np.max(np.abs(lo - hi)) <= 1e-12, np.max(np.abs(lo - hi))
    # grid independence (no post blur, which reads neighbours): footprint |y + 1| <= 0.3
    # is outside the small grid (y in [-0.5, 0.5]); the big grid (y in [-2, 0.5]) holds it.
    # Tolerance: the per-point direction sums are the same arithmetic in another
    # chunking (BLAS blocking may differ in the last bits).
    off = G.Tube((0.0, -1.0, 0.0), 0.3, 0.0, 0.0, 1.0, 3.0, "axial")
    small, big = R.FineGrid(-0.5, -0.5, 0.05, 21, 21), R.FineGrid(-0.5, -2.0, 0.05, 21, 51)
    rc0 = rw_cfg(ray_post_sigma_um=0.0)
    t_small = R.render_transmittance(off, 2.0, [1.0], small, rc0).tau[0]
    t_big = R.render_transmittance(off, 2.0, [1.0], big, rc0).tau[0]
    assert t_small.min() < 1 - 1e-3, t_small.min()
    assert np.max(np.abs(t_small - t_big[30:, :])) <= 1e-13, np.max(np.abs(t_small - t_big[30:, :]))


def test_contract():
    tube = G.Tube((0.0, 0.0, 0.0), 0.3, 0.2, 0.4, 1.0, 3.0, "axial")
    grid, inner = R.block_fine_grid(-10, -10, 21, 21, PX, 4, 2)
    res = R.render_transmittance(tube, 1.0, [0.0, 0.28], grid, rw_cfg())
    assert res.tau.shape == (2, grid.ny, grid.nx) and res.backend == "ray_world" and np.all(res.tau <= 1 + 1e-12)
    red = R.render_transmittance(tube, 1.0, [0.0, 0.28], grid, rw_cfg(),
                                 reduce=lambda p: R.pixel_integrate(p[inner[0], inner[1]], 4))
    assert red.tau.shape == (2, 21, 21)


def test_determinism():
    raise unittest.SkipTest("deterministic quadrature")


def test_edge_cases():
    grid = R.FineGrid(-1.0, -1.0, PX / 4, 71, 71)
    try:
        R.render_transmittance(G.Tube((0.0, 0.0, 0.0), 0.3, 0.2, 0.4), 1.0, [0.0], grid, rw_cfg())
    except ValueError:
        return
    raise AssertionError("an uncut tube must be refused (infinite depth extent)")


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
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
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
        print(("%-5s  %-" + str(width) + "s  %8.3fs  %s") % (status, name, seconds, lines[0] if lines else ""))
    for name, status, _s, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items()))
    return 1 if (counts["FAIL"] + counts["ERROR"] + counts["TODO"]) else 0


if __name__ == "__main__":
    sys.exit(main())
