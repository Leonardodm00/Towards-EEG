"""Smoke test for the cell level -- Block 8 in specs/SPEC.md (handoff step 7; D-013).

Checks
    test_known_answer   dendrite membrane area of hand-made cylinders and a cone;
                        doubling every radius of a cylinder chain doubles it
    test_reference      to_image_um equals allen_image_align.swc_to_full_px
                        times the pixel pitch (shift, flip); stretches of a
                        hand-made tree by hand
    test_convergence    skipped: no discretisation parameter
    test_invariants     the stretches partition the dendrite nodes; a stretch's
                        Branch starts with its parent dendrite node
    test_contract       a synthetic cell (straight dendrite rendered into planes
                        served through allen_image_io.fetch_zblock) measured,
                        corrected and filled end to end; the per-node rows carry
                        the Block 8 columns; the corrected SWC differs from the
                        original in the dendrite radius tokens only
    test_determinism    skipped: covered by Blocks 4-6
    test_edge_cases     a lone root node has no direction

Run
    cd "Passive Features/Diameter Re-measurement"
    python tests/smoke/test_smoke_cell.py

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import os
import platform
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for p in (str(SRC), str(WS / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import allen_image_io as aio  # noqa: E402
from allen_diameter.analysis import cell as CE  # noqa: E402
from allen_diameter.analysis.table import fit_table  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.loading import swc_io  # noqa: E402
from allen_diameter.model import camera, geometry, render  # noqa: E402

SEED = 20261006
REPORT_PACKAGES = ("numpy", "scipy", "pandas")
PX = 0.1144


def write_swc_text(path, rows):
    with open(path, "w") as f:
        f.write("# hand-made test SWC\n")
        for r in rows:
            f.write("%d %d %.4f %.4f %.4f %.4f %d\n" % r)


def tree_swc(tmp):
    """soma 1; chain 2-3-4 branching at 4 into 5-6 and 7 (types 3); unit steps along x."""
    rows = [(1, 1, 0.0, 0.0, 0.0, 5.0, -1), (2, 3, 6.0, 0.0, 0.0, 0.5, 1), (3, 3, 7.0, 0.0, 0.0, 0.5, 2),
            (4, 3, 8.0, 0.0, 0.0, 0.5, 3), (5, 3, 9.0, 0.0, 0.0, 0.5, 4), (6, 3, 10.0, 0.0, 0.0, 0.5, 5),
            (7, 3, 8.0, 1.0, 0.0, 0.5, 4)]
    path = os.path.join(tmp, "tree.swc")
    write_swc_text(path, rows)
    return swc_io.read_swc(path)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    with tempfile.TemporaryDirectory() as tmp:
        swc = tree_swc(tmp)
        # 5 unit segments between dendrite nodes (cylinders r = 0.5: pi each) + node 2's link to the
        # soma, 6 um long, a cylinder of node 2's own radius: 2 pi 0.5 6 = 6 pi
        assert abs(CE.dendrite_area_um2(swc) - (5 * math.pi + 6 * math.pi)) <= 1e-12
        assert abs(CE.area_ratio(swc, 2 * swc.radius) - 2.0) <= 1e-12
        r = swc.radius.copy()
        r[2] = 1.0                       # node 3: cone 2-3 and cone 3-4 (r 0.5 -> 1.0 over 1 um)
        cone = math.pi * 1.5 * math.sqrt(1.0 + 0.25)
        want = 6 * math.pi + 2 * cone + 3 * math.pi
        assert abs(CE.dendrite_area_um2(swc, r) - want) <= 1e-12, (CE.dendrite_area_um2(swc, r), want)


def test_reference():
    import pandas as pd
    import allen_image_align as aia
    rng = np.random.default_rng(SEED)
    xyz = rng.uniform(0, 200, (10, 3))
    df = pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
    for shift, flip in (((0.0, 0.0), None), ((12.5, -3.25), 3000.0)):
        x, y = aia.swc_to_full_px(df, PX, shift, flip)
        got = CE.to_image_um(xyz, PX, shift, flip, z0_um=1.5)
        assert np.allclose(got[:, 0], x * PX, atol=1e-9) and np.allclose(got[:, 1], y * PX, atol=1e-9)
        assert np.allclose(got[:, 2], xyz[:, 2] - 1.5, atol=0)
    with tempfile.TemporaryDirectory() as tmp:
        runs = [list(map(int, tree_swc(tmp).ids[r])) for r in CE.stretches(tree_swc(tmp))]
    assert sorted(runs) == [[2, 3, 4], [5, 6], [7]], runs


def test_convergence():
    raise unittest.SkipTest("no discretisation parameter in Block 8")


def test_invariants():
    with tempfile.TemporaryDirectory() as tmp:
        swc = tree_swc(tmp)
        runs = CE.stretches(swc)
        allr = np.sort(np.concatenate(runs))
        assert np.array_equal(allr, np.flatnonzero(swc.dendrite_mask())), allr
        xyz = CE.to_image_um(swc.xyz, PX)
        br, off = CE.stretch_branch(swc, runs[[list(swc.ids[r]) for r in runs].index([5, 6])], xyz)
        assert list(br.ids) == [4, 5, 6] and off == 1
        br, off = CE.stretch_branch(swc, runs[[list(swc.ids[r]) for r in runs].index([2, 3, 4])], xyz)
        assert list(br.ids) == [2, 3, 4] and off == 0          # the soma is not prepended
        br, off = CE.stretch_branch(swc, runs[[list(swc.ids[r]) for r in runs].index([7])], xyz)
        assert list(br.ids) == [4, 7] and off == 1


def _synthetic_cell(tmp, cfg, d_true=0.8, n_nodes=6, step=1.18, theta=0.3):
    """Soma + one straight flat dendrite; planes pre-rendered over the region and served by plane index."""
    c0 = np.array([6.0, 0.03, 0.05])
    t = np.array([math.cos(theta), math.sin(theta), 0.0])
    pts = c0[None, :] + (np.arange(n_nodes) * step)[:, None] * t[None, :]
    rows = [(1, 1, 0.0, 0.0, 0.0, 4.0, -1)] + [(k + 2, 3, p[0], p[1], p[2], 0.3, k + 1) for k, p in enumerate(pts)]
    path = os.path.join(tmp, "cell.swc")
    write_swc_text(path, rows)
    mid = pts.mean(axis=0)
    tube = geometry.Tube(tuple(mid), 0.5 * d_true, 0.0, theta, 1.0, 0.5 * (n_nodes - 1) * step + 6.0, "axial")
    lo = np.floor((pts[:, :2].min(axis=0) - 6.0) / PX).astype(int)
    hi = np.ceil((pts[:, :2].max(axis=0) + 6.0) / PX).astype(int)
    left, top, width, height = int(lo[0]), int(lo[1]), int(hi[0] - lo[0]), int(hi[1] - lo[1])
    ks = np.arange(-6, 7)
    rc = cfg.renderer
    f = render.fine_factor(d_true, rc, PX)
    grid, inner = render.block_fine_grid(left, top, width, height, PX, f, 8)
    res = render.render_transmittance(tube, 1.0, ks * cfg.acquisition.dz_um, grid, rc, +1,
                                      reduce=lambda pl: render.pixel_integrate(pl[inner[0], inner[1]], f))
    planes8 = camera.camera_chain(res.tau, rc, np.random.default_rng(SEED))

    class PlaneFetcher(aio.ImageFetcher):
        def get(self, image_id, left_, top_, width_, height_, downsample=0):
            out = np.full((height_, width_), 255, dtype=np.uint8)
            img = planes8[int(image_id) - int(ks[0])]
            x0, y0 = max(left_, left), max(top_, top)
            x1, y1 = min(left_ + width_, left + width), min(top_ + height_, top + height)
            if x1 > x0 and y1 > y0:
                out[y0 - top_:y1 - top_, x0 - left_:x1 - left_] = img[y0 - top:y1 - top, x0 - left:x1 - left]
            return out

    import pandas as pd
    planes = pd.DataFrame({"plane_index": ks, "id": ks})
    return swc_io.read_swc(path), PlaneFetcher(), planes


def test_contract():
    import run_cell
    base = default_config()
    cfg = dataclasses.replace(base, measure=dataclasses.replace(base.measure, block_half_um=3.5))
    rng = np.random.default_rng(SEED)
    n = 300                                            # an exact b = 1 table over the default design
    d = np.exp(rng.uniform(math.log(0.2), math.log(4.0), n))
    phi = np.radians(rng.uniform(0, 90, n))
    table = fit_table([dict(d_um=a, phi_rad=b, ratio=1.0, in_S=True) for a, b in zip(d, phi)],
                      dataclasses.replace(cfg, correction=dataclasses.replace(cfg.correction, spline_smoothing="1e-8")))
    with tempfile.TemporaryDirectory() as tmp:
        swc, fetcher, planes = _synthetic_cell(tmp, cfg)
        provider = run_cell.real_provider(fetcher, planes, PX)
        rows, radius_new, ratio = CE.measure_cell(swc, provider, table, cfg)
        assert len(rows) == 6 and all(tuple(r.keys()) == CE.CSV_COLUMNS for r in rows)
        kept = [r for r in rows if r["filled_from"] == "self"]
        assert len(kept) >= 4, [r["flags"] for r in rows]
        dh = np.array([r["d_hat_um"] for r in kept])
        assert np.all((dh > 0.75) & (dh < 1.0)), dh             # d = 0.8 um, flat: the halo reads a few % wide
        # Allen's 0.3 um against a measured radius of 0.375-0.5 um: the area scales with the radius here
        assert 1.25 < ratio < 1.67 and ratio == CE.area_ratio(swc, radius_new) and radius_new[0] == swc.radius[0], ratio
        summary = run_cell.write_outputs(tmp, "999", swc, rows, radius_new, ratio, cfg)
        out = swc_io.read_swc(os.path.join(tmp, "specimen_999", "reconstruction.swc"))
        assert np.array_equal(out.ids, swc.ids) and np.allclose(out.xyz, swc.xyz) and np.allclose(out.radius, radius_new, atol=1e-4)
        old_lines, new_lines = open(swc.path).read().splitlines(), open(out.path).read().splitlines()
        changed = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
        assert all(old_lines[i].split()[:5] == new_lines[i].split()[:5] and old_lines[i].split()[6:] == new_lines[i].split()[6:]
                   for i in changed) and len(old_lines) == len(new_lines)
        assert summary["n_nodes"] == 6 and abs(summary["area_ratio"] - ratio) <= 1e-12


def test_determinism():
    raise unittest.SkipTest("covered by Blocks 4-6 (rendering, chain, replicates)")


def test_edge_cases():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "lone.swc")
        write_swc_text(path, [(1, 3, 0.0, 0.0, 0.0, 0.5, -1)])
        swc = swc_io.read_swc(path)
        try:
            CE.stretch_branch(swc, CE.stretches(swc)[0], CE.to_image_um(swc.xyz, PX))
        except ValueError:
            return
        raise AssertionError("a lone root node must be refused")


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
