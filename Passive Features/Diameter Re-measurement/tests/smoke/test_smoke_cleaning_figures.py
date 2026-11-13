"""Smoke test for the cleaning figures -- Block 31 in specs/SPEC.md (plan step C6; D-129.4).

The module computes nothing scientific, so the oracles are about what is
drawn: the images shown are the input planes, the line drawn is line_xy, the
profiles drawn are the input profiles, one grey scale per row, the inputs are
not written to, one point per round, the thresholds at the configured values.

Test arrays (Blocks 25 and 28 are not imported): a Block 24 scene of a node
tube and a neighbour tube (model/scene_render.synthetic_scene_block, uint8
through the camera chain); the neighbour's predicted shadow B S = B (1 - tau)
with tau the neighbour rendered alone on the same fine grid
(scene_render.render_scene, pixel-integrated); the cleaned block = block + B S
in plain numpy (Block 28's rule); the profiles read by
analysis.profiles.sample_profile along the node's measuring line.

Checks
    test_known_answer   the images shown equal the input planes, their extent
                        is image_extent_um(frame), the line drawn is line_xy,
                        the profile lines are the input profiles
    test_reference      each row's vmin/vmax equals numpy's nanmin/nanmax over
                        the row's three planes
    test_convergence    skipped (no discretisation)
    test_invariants     the three images of a row share vmin/vmax; the input
                        arrays are unchanged (and read-only during the call)
    test_contract       3 image axes and 1 profile axis per plane shown, for
                        planes=None and a chosen subset; figure_rounds: one
                        point per round in each panel, stop_rel_d and
                        max_rounds drawn; save writes a non-empty PNG
    test_determinism    two calls give bit-identical rendered canvases
    test_edge_cases     ValueError for shapes that disagree, an empty planes,
                        a plane not in ks, an empty history, a missing key

Run (from "Passive Features/Diameter Re-measurement")
    python3 tests/smoke/test_smoke_cleaning_figures.py
    python -m pytest tests/smoke/test_smoke_cleaning_figures.py -q
Keep example PNGs: set CLEANING_FIGURES_OUT=<folder> (default: a temporary folder).

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import os
import platform
import re
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
for p in (str(SRC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from allen_diameter.analysis import profiles as PR  # noqa: E402
from allen_diameter.config import CleaningConfig, default_config  # noqa: E402
from allen_diameter.model import geometry as G  # noqa: E402
from allen_diameter.model import render as R  # noqa: E402
from allen_diameter.model import scene_render as SR  # noqa: E402
from allen_diameter.plotting import cleaning_figures as CF  # noqa: E402
from allen_diameter.plotting.figures import image_extent_um  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []
_SCENE = {}


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
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


def out_dir():
    d = os.environ.get("CLEANING_FIGURES_OUT")
    if d:
        os.makedirs(d, exist_ok=True)
        return d
    if "tmp" not in _SCENE:
        _SCENE["tmp"] = tempfile.mkdtemp(prefix="block31_")
    return _SCENE["tmp"]


def scene():
    """The synthetic node of the checks (cached): a flat node tube along x through the origin (d 1 um, mu 1/um) and
    a neighbour (d 0.8 um, mu 1.2/um) 2.2 um aside and 0.56 um deeper, crossing at 20 deg; planes -2..2."""
    if "arrays" in _SCENE:
        return _SCENE["arrays"]
    cfg = default_config()
    acq, rcfg = cfg.acquisition, cfg.renderer
    p = acq.res0_um
    node = G.Tube((0.0, 0.0, 0.0), 0.5, 0.0, 0.0, 1.0, None, "axial")
    nb = G.Tube((0.0, 2.2, 0.56), 0.4, 0.0, math.radians(20.0), 1.0, None, "axial")
    mus = [1.0, 1.2]
    ks = np.arange(-2, 3)
    W = H = 56
    left, top = -W // 2, -H // 2
    pad = 24
    block, ks_out, valid, frame = SR.synthetic_scene_block([node, nb], mus, ks, left, top, W, H, cfg, make_rng(), pad)
    # the neighbour alone on the same fine grid (synthetic_scene_block's own steps, before the camera chain)
    f = R.fine_factor(min(node.d, nb.d), rcfg, p)
    grid, inner = R.block_fine_grid(left, top, W, H, p, f, pad)
    res = SR.render_scene([nb], [mus[1]], ks * acq.dz_um, grid, rcfg, acq.light_direction,
                          reduce=lambda plane: R.pixel_integrate(plane[inner[0], inner[1]], f))
    B = rcfg.black_level_gl + rcfg.gain * rcfg.background_B_gl
    shadow = B * (1.0 - np.asarray(res.tau, dtype=float))
    cleaned = block.astype(float) + shadow
    v = PR.profile_offsets(cfg.measure)
    y_hat, e_u = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    origin = (0.0, 0.0)
    before = np.array([PR.sample_profile(block[j], frame, origin, y_hat, e_u, v) for j in range(ks.size)])
    after = np.array([PR.sample_profile(cleaned[j], frame, origin, y_hat, e_u, v) for j in range(ks.size)])
    line_xy = np.array([[origin[0] + v[0] * y_hat[0], origin[1] + v[0] * y_hat[1]],
                        [origin[0] + v[-1] * y_hat[0], origin[1] + v[-1] * y_hat[1]]])
    meta = {"node_id": 4711, "class": "close_in_focus", "round": 1,
            "features": {"d_um": 1.0, "mu_per_um": 1.0, "dx_um": 0.0, "dz_um": 0.0}}
    arrays = dict(block=block, shadow=shadow, cleaned=cleaned, ks=ks_out, frame=frame, line_xy=line_xy, v=v,
                  profiles_before=before, profiles_after=after, meta=meta)
    _SCENE["arrays"] = arrays
    return arrays


def call_node(arrays, planes=None):
    a = arrays
    return CF.figure_node(a["block"], a["shadow"], a["cleaned"], a["ks"], a["frame"], a["line_xy"], a["v"],
                          a["profiles_before"], a["profiles_after"], a["meta"], planes=planes)


def history(n=4):
    return [{"max_rel_d": 0.2 * 0.3 ** t, "max_line_change_gl": 9.0 * 0.4 ** t, "n_refined": 20 - 3 * t}
            for t in range(n)]


def split_axes(fig):
    """(image axes, profile axes) of a figure_node figure, in drawing order."""
    imgs = [ax for ax in fig.axes if len(ax.images) == 1]
    profs = [ax for ax in fig.axes if len(ax.images) == 0 and len(ax.lines) >= 2]
    return imgs, profs


def rows_of(planes, ks):
    ks = [int(k) for k in ks]
    return [ks.index(int(k)) for k in planes] if planes is not None else list(range(len(ks)))


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    a = scene()
    planes = [0, -2]
    fig = call_node(a, planes=planes)
    try:
        imgs, profs = split_axes(fig)
        ext = image_extent_um(a["frame"], a["block"].shape)
        for r, j in enumerate(rows_of(planes, a["ks"])):
            for c, src in enumerate(("block", "shadow", "cleaned")):
                ax = imgs[3 * r + c]
                im = ax.images[0]
                # the image shown is the input plane itself (exact: no computation on it)
                assert_array_equal(np.asarray(im.get_array()), a[src][j])
                check_close("extent r%d c%d" % (r, c), im.get_extent(), ext, rtol=0, atol=0)
                drawn = [ln for ln in ax.lines if np.allclose(np.column_stack(ln.get_data()), a["line_xy"],
                                                              rtol=0, atol=0)]
                assert len(drawn) == 1, "row %d image %d: measuring line not drawn at line_xy" % (r, c)
            ys = [np.asarray(ln.get_ydata(), dtype=float) for ln in profs[r].lines]
            xs = [np.asarray(ln.get_xdata(), dtype=float) for ln in profs[r].lines]
            assert_array_equal(xs[0], a["v"])
            assert_array_equal(ys[0], a["profiles_before"][j])
            assert_array_equal(ys[1], a["profiles_after"][j])
        # the cleaning really changed the line (the test scene is not trivial)
        assert np.nanmax(np.abs(a["profiles_after"] - a["profiles_before"])) > 3.0
    finally:
        plt.close(fig)


def test_reference():
    a = scene()
    fig = call_node(a)
    try:
        imgs, _ = split_axes(fig)
        for r, j in enumerate(rows_of(None, a["ks"])):
            stack = np.concatenate([np.asarray(a[s][j], dtype=float).ravel() for s in ("block", "shadow", "cleaned")])
            lo, hi = float(np.nanmin(stack)), float(np.nanmax(stack))
            for c in range(3):
                check_close("clim r%d c%d" % (r, c), imgs[3 * r + c].images[0].get_clim(), (lo, hi), rtol=0, atol=0)
    finally:
        plt.close(fig)


def test_convergence():
    raise unittest.SkipTest("plotting only: no discretisation")


def test_invariants():
    a = scene()
    copies = {k: np.array(a[k], copy=True) for k in ("block", "shadow", "cleaned", "ks", "line_xy", "v",
                                                     "profiles_before", "profiles_after")}
    for k in copies:
        a[k].flags.writeable = False     # any write inside figure_node raises
    try:
        fig = call_node(a, planes=[1, -1, 0])
        imgs, _ = split_axes(fig)
        for r in range(3):
            clims = [imgs[3 * r + c].images[0].get_clim() for c in range(3)]
            assert clims[0] == clims[1] == clims[2], "row %d: grey scales differ %r" % (r, clims)
        plt.close(fig)
    finally:
        for k in copies:
            a[k].flags.writeable = True
    for k, ref in copies.items():
        assert_array_equal(a[k], ref)
        assert a[k].dtype == ref.dtype, k


def test_contract():
    a = scene()
    for planes, n in ((None, a["ks"].size), ([0], 1), ([2, -2], 2)):
        fig = call_node(a, planes=planes)
        try:
            assert isinstance(fig, Figure)
            imgs, profs = split_axes(fig)
            assert len(imgs) == 3 * n, (planes, len(imgs))
            assert len(profs) == n, (planes, len(profs))
            assert len(fig.axes) == 4 * n, (planes, len(fig.axes))
        finally:
            plt.close(fig)
    ccfg = CleaningConfig()
    for n_rounds in (1, 3, 7):
        hist = history(n_rounds)
        fig = CF.figure_rounds(hist)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.axes) == 3
            keys = ("max_rel_d", "max_line_change_gl", "n_refined")
            for ax, key in zip(fig.axes, keys):
                data = [ln for ln in ax.lines if len(ln.get_xdata()) == n_rounds and ln.get_marker() == "o"]
                assert len(data) == 1, (key, n_rounds)
                assert_array_equal(np.asarray(data[0].get_xdata(), dtype=float), np.arange(1, n_rounds + 1))
                assert_array_equal(np.asarray(data[0].get_ydata(), dtype=float), [float(h[key]) for h in hist])
                vx = [ln for ln in ax.lines if list(ln.get_xdata()) == [ccfg.max_rounds, ccfg.max_rounds]]
                assert len(vx) == 1, "max_rounds not drawn in %s" % key
            hy = [ln for ln in fig.axes[0].lines if list(ln.get_ydata()) == [ccfg.stop_rel_d, ccfg.stop_rel_d]]
            assert len(hy) == 1, "stop_rel_d not drawn"
        finally:
            plt.close(fig)
    folder = out_dir()
    fig = call_node(a, planes=[-1, 0, 1])
    p1 = CF.save(fig, os.path.join(folder, "cleaning_node_example.png"), 110)
    plt.close(fig)
    fig = CF.figure_rounds(history(4))
    p2 = CF.save(fig, os.path.join(folder, "sub", "cleaning_rounds_example.png"), 110)
    plt.close(fig)
    for pth in (p1, p2):
        assert os.path.getsize(pth) > 1000, pth
        with open(pth, "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n", pth


def test_determinism():
    a = scene()
    bufs = []
    for _ in range(2):
        fig = call_node(a, planes=[0, 1])
        fig.canvas.draw()
        bufs.append(np.array(fig.canvas.buffer_rgba(), copy=True))
        plt.close(fig)
    assert_array_equal(bufs[0], bufs[1])
    bufs = []
    for _ in range(2):
        fig = CF.figure_rounds(history(3))
        fig.canvas.draw()
        bufs.append(np.array(fig.canvas.buffer_rgba(), copy=True))
        plt.close(fig)
    assert_array_equal(bufs[0], bufs[1])


def _raises(fn, what):
    try:
        fig = fn()
    except ValueError:
        return
    plt.close(fig)
    raise AssertionError("no ValueError for %s" % what)


def test_edge_cases():
    a = scene()

    def with_(**changes):
        b = dict(a)
        b.update(changes)
        return lambda: CF.figure_node(b["block"], b["shadow"], b["cleaned"], b["ks"], b["frame"], b["line_xy"],
                                      b["v"], b["profiles_before"], b["profiles_after"], b["meta"],
                                      planes=changes.get("planes"))

    _raises(with_(shadow=a["shadow"][:, :-1, :]), "shadow of another shape")
    _raises(with_(cleaned=a["cleaned"][:-1]), "cleaned with fewer planes")
    _raises(with_(block=a["block"][0]), "a 2-D block")
    _raises(with_(ks=a["ks"][:-1]), "ks of the wrong length")
    _raises(with_(line_xy=a["line_xy"][0]), "line_xy of shape (2,)")
    _raises(with_(profiles_before=a["profiles_before"][:, :-1]), "profiles_before with fewer offsets")
    _raises(with_(profiles_after=a["profiles_after"][:-1]), "profiles_after with fewer planes")
    _raises(with_(v=a["v"][:-1]), "v of the wrong length")
    _raises(with_(planes=[]), "an empty planes")
    _raises(with_(planes=[7]), "a plane not in ks")
    _raises(with_(meta="node 1"), "meta not a mapping")
    _raises(lambda: CF.figure_rounds([]), "an empty history")
    _raises(lambda: CF.figure_rounds([{"max_rel_d": 0.1, "n_refined": 3}]), "a round without max_line_change_gl")
    fig = plt.figure()
    try:
        _raises(lambda: CF.save(fig, os.path.join(out_dir(), "x.png"), 0), "dpi 0")
    finally:
        plt.close(fig)


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
    print(f"-- example PNGs in {out_dir()}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
