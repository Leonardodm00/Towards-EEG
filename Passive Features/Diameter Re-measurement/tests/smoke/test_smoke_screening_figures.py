"""Smoke test for the screening sheets -- Block 62 in specs/SPEC.md (section 5, "The screening export";
D-226.3, D-227, D-261.1).

Views are built here as plain objects with NodeView's fields (Block 61 is not imported, as the entry says):
synthetic uint8 stacks with a dark band along the measuring line, synthetic profiles and backgrounds. The marks
and the extent are built from the entry's definitions: marks["v_um"] = (-D/2, +D/2), marks["xy_um"] = (cx, cy)
-+ (D/2)(-sin theta, cos theta), one row per point; the extent the crop's outer pixel edges with pixel
(row, col) centred on ((left + col) p, (top + row) p).

Oracles (SPEC Block 62)
    S1  a near view gives 14 axes, a far view 66; each profile line's data equal v_um and profile_gl[q], each
        dashed line's background_gl[q], each image's array stack[q]           (test_known_answer)
    S2  only the focus pair carries the marks                                 (test_reference)
    S3  every image shares vmin, vmax (the 1st / 99th percentiles of the valid planes' pixels) and every
        profile the y-range                                                   (test_invariants)
    S4  the pair titles name the offsets -3 .. 3 (near) and a non-ok status   (test_contract)
    S5  write_node_sheet writes a non-empty PNG and leaves no open figure     (test_write_sheet)
    and: the same view gives the same PNG bytes twice                         (test_determinism)
    S6  a plane with valid False; ks without k_star raises ValueError         (test_edge_cases)

Run (from the package root)
    python3 tests/smoke/test_smoke_screening_figures.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    python3 -m pytest tests/smoke/test_smoke_screening_figures.py -q
        The same checks through pytest.
    SCREENING_SHEETS_DIR=/tmp/sheets python3 tests/smoke/test_smoke_screening_figures.py
        Also keeps one near and one far sheet as PNG in that folder, to look at.

Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import sys
import tempfile
import time
import traceback
import types
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from allen_diameter.plotting import screening_figures as SF  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "matplotlib")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run

DZ_UM = 0.28           # plane spacing (um), any positive value
RES_UM = 0.1144        # pixel size (um)
D_ALLEN = 0.72         # Allen's diameter (um)


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


# ------------------------------------------------------------ the inputs ---

def make_view(which, rng=None, theta=0.6, invalid=(), statuses=None, k_star=40):
    """A plain object with NodeView's fields: 7 planes (near) or 33 (far) about k_star."""
    rng = make_rng() if rng is None else rng
    half_pl = 3 if which == "near" else 16
    half_um = 4.0 if which == "near" else 11.0
    ks = np.arange(k_star - half_pl, k_star + half_pl + 1, dtype=np.int64)
    n = ks.size
    side = int(round(2 * half_um / RES_UM)) + 1
    left, top = 1200, 800
    cx = (left + (side - 1) / 2.0) * RES_UM
    cy = (top + (side - 1) / 2.0) * RES_UM
    rows, cols = np.mgrid[0:side, 0:side]
    x, y = (left + cols) * RES_UM, (top + rows) * RES_UM
    u = -(x - cx) * np.sin(theta) + (y - cy) * np.cos(theta)    # signed distance across the dendrite
    stack = np.empty((n, side, side), dtype=np.uint8)
    for q in range(n):
        depth = 60.0 * np.exp(-0.5 * ((ks[q] - k_star) / 4.0) ** 2)
        img = 180.0 - depth * np.exp(-0.5 * (u / 0.4) ** 2) + rng.normal(0.0, 4.0, u.shape)
        stack[q] = np.clip(np.round(img), 0, 255).astype(np.uint8)
    valid = np.ones(n, dtype=bool)
    for q in invalid:
        valid[q] = False
        stack[q] = 0
    v = np.arange(-3.0 if which == "near" else -10.0, 3.0001 if which == "near" else 10.0001, 0.1)
    bg = 180.0 + 0.5 * v[None, :] + rng.normal(0.0, 0.3, (n, v.size))
    w = (60.0 * np.exp(-0.5 * ((ks - k_star) / 4.0) ** 2))[:, None] * np.exp(-0.5 * (v[None, :] / 0.4) ** 2)
    prof = bg - w
    prof[0, :3] = np.nan                                      # NaN where the scan has none
    if statuses is None:
        statuses = ["ok"] * n
    return types.SimpleNamespace(
        cell_id=529878215, node_id=17 if which == "near" else 233, set=which, allen_diameter_um=D_ALLEN,
        k_star=int(k_star), ks=ks, stack=stack, valid=valid, left=left, top=top, res0_um=RES_UM,
        cx_um=cx, cy_um=cy, theta_rad=theta, phi_rad=0.1, v_um=v, profile_gl=prof, background_gl=bg,
        status=tuple(statuses), omega_um2=np.full(n, 0.1), B_bar=np.full(n, 180.0))


def marks_of(view):
    """SPEC "common definitions": v = -+ D/2; the points (cx, cy) -+ (D/2)(-sin theta, cos theta), one per row."""
    h = 0.5 * view.allen_diameter_um
    e = np.array([-np.sin(view.theta_rad), np.cos(view.theta_rad)])
    xy = np.vstack([[view.cx_um, view.cy_um] - h * e, [view.cx_um, view.cy_um] + h * e])
    return {"v_um": (-h, h), "xy_um": xy}


def extent_of(view):
    """Block 61's image_extent: (x0, x1, y0, y1) um, the crop's outer pixel edges."""
    p = view.res0_um
    H, W = view.stack.shape[1:]
    return ((view.left - 0.5) * p, (view.left + W - 0.5) * p, (view.top - 0.5) * p, (view.top + H - 0.5) * p)


def sheet(view, **kw):
    return SF.node_sheet(view, marks_of(view), extent_of(view), DZ_UM, **kw)


def pairs(fig):
    """[(image axes, profile axes)] in creation order: the module adds them pair by pair."""
    axes = fig.axes
    return [(axes[i], axes[i + 1]) for i in range(0, len(axes), 2)]


def lines(ax, gid):
    return [ln for ln in ax.get_lines() if ln.get_gid() == gid]


def _raises(exc, func, *args, **kw):
    try:
        func(*args, **kw)
    except exc:
        return
    raise AssertionError("%s not raised by %s" % (exc.__name__, getattr(func, "__name__", func)))


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    """S1: axis counts; every profile, background and image carries exactly the view's data, in ks order."""
    for which, n_axes, per_row in (("near", 14, 1), ("far", 66, 3)):
        view = make_view(which)
        fig = sheet(view)
        try:
            assert len(fig.axes) == n_axes, (which, len(fig.axes))
            ps = pairs(fig)
            for q, (a_im, a_pr) in enumerate(ps):
                (pl,) = lines(a_pr, "profile")
                (bl,) = lines(a_pr, "background")
                assert pl.get_linestyle() == "-" and bl.get_linestyle() == "--"
                assert_array_equal(pl.get_xdata(), view.v_um)
                assert_array_equal(pl.get_ydata(), view.profile_gl[q])      # bit for bit, NaN included
                assert_array_equal(bl.get_xdata(), view.v_um)
                assert_array_equal(bl.get_ydata(), view.background_gl[q])
                assert len(a_im.images) == 1
                assert_array_equal(np.asarray(a_im.images[0].get_array()), view.stack[q])
                # layout: row-major, per_row pairs per row, image left of its profile
                r, c = divmod(q, per_row)
                ss_im, ss_pr = a_im.get_subplotspec(), a_pr.get_subplotspec()
                assert (ss_im.rowspan.start, ss_im.colspan.start) == (r, 2 * c), (which, q)
                assert (ss_pr.rowspan.start, ss_pr.colspan.start) == (r, 2 * c + 1), (which, q)
            n_rows = ps[-1][0].get_subplotspec().rowspan.start + 1
            assert n_rows == (7 if which == "near" else 11), (which, n_rows)
            exp_title = "cell 529878215, node %d (%s), Allen's diameter 0.72 um" % (view.node_id, which)
            assert fig._suptitle.get_text() == exp_title, fig._suptitle.get_text()
        finally:
            plt.close(fig)
    view = make_view("near")
    fig = sheet(view, title="my title")
    try:
        assert fig._suptitle.get_text() == "my title"
    finally:
        plt.close(fig)


def test_reference():
    """S2: only the focus pair carries the marks: one artist at marks["xy_um"] on the image, two vertical lines
    at marks["v_um"] on the profile; the other axes none. Also the image's extent is the passed one."""
    for which in ("near", "far"):
        for theta in (0.0, 0.6, np.pi / 2):
            view = make_view(which, theta=theta)
            m = marks_of(view)
            x0, x1, y0, y1 = extent_of(view)
            fig = sheet(view)
            try:
                q_star = int(np.flatnonzero(view.ks == view.k_star)[0])
                for q, (a_im, a_pr) in enumerate(pairs(fig)):
                    assert_allclose(a_im.images[0].get_extent(), (x0, x1, y1, y0), rtol=0, atol=0)
                    assert a_im.get_ylim()[0] > a_im.get_ylim()[1]                # y increases downwards
                    others_im = [ln for ln in a_im.get_lines() if ln.get_gid() != "allen_marks"]
                    assert not others_im, "lines drawn over image %d" % q
                    assert not a_im.collections and not a_im.patches
                    mk, av = lines(a_im, "allen_marks"), lines(a_pr, "allen_v")
                    if q == q_star:
                        assert len(mk) == 1
                        assert_array_equal(np.column_stack([mk[0].get_xdata(), mk[0].get_ydata()]), m["xy_um"])
                        assert len(av) == 2
                        got = sorted(float(np.unique(ln.get_xdata())[0]) for ln in av)
                        assert got == sorted(m["v_um"]), got
                        assert all(np.unique(ln.get_xdata()).size == 1 for ln in av)    # vertical
                    else:
                        assert not mk and not av, (which, q)
                    # the frame colour singles out the focus pair only
                    framed = [to_hex(ax.spines["left"].get_edgecolor()) == SF.ORANGE for ax in (a_im, a_pr)]
                    assert framed == [q == q_star] * 2, (which, q, framed)
                    assert len(a_pr.get_lines()) == (4 if q == q_star else 2), (which, q)
            finally:
                plt.close(fig)


def test_convergence():
    raise unittest.SkipTest("not a discretisation: a figure")


def test_invariants():
    """S3: one vmin / vmax for every image, the 1st and 99th percentiles of the valid planes' pixels; one y-range
    for every profile, holding every finite value of both curves."""
    for which in ("near", "far"):
        view = make_view(which, invalid=(1,))
        fig = sheet(view)
        try:
            lo, hi = np.percentile(view.stack[view.valid].astype(np.float64), (1.0, 99.0))
            clims = {a_im.images[0].get_clim() for a_im, _ in pairs(fig) if a_im.images}
            assert len(clims) == 1, clims
            check_close("S3 clim " + which, list(clims)[0], (lo, hi), rtol=1e-12)    # float64 roundoff
            ylims = {a_pr.get_ylim() for _, a_pr in pairs(fig)}
            assert len(ylims) == 1, ylims
            ylo, yhi = list(ylims)[0]
            allv = np.concatenate([view.profile_gl.ravel(), view.background_gl.ravel()])
            allv = allv[np.isfinite(allv)]
            assert ylo <= allv.min() and yhi >= allv.max(), (ylo, yhi, allv.min(), allv.max())
            for a_im, _ in pairs(fig):
                assert a_im.get_aspect() == 1.0, a_im.get_aspect()                 # equal aspect
        finally:
            plt.close(fig)


def test_contract():
    """S4: the pair titles name the offsets -3 .. 3 (near) with the defocus o dz, and a non-ok status."""
    st = ["ok"] * 7
    st[5] = "d_at_bound"
    view = make_view("near", statuses=st)
    fig = sheet(view)
    try:
        titles = [a_im.get_title(loc="left") for a_im, _ in pairs(fig)]
        for q, o in enumerate(range(-3, 4)):
            head = "k* %+d (%+.2f um)" % (o, o * DZ_UM)
            assert titles[q].startswith(head), (titles[q], head)
            assert ("d_at_bound" in titles[q]) == (q == 5), titles[q]
        assert "ok" not in titles[0]
    finally:
        plt.close(fig)
    view = make_view("far")
    fig = sheet(view)
    try:
        titles = [a_im.get_title(loc="left") for a_im, _ in pairs(fig)]
        assert [int(t.split()[1]) for t in titles] == list(range(-16, 17)), titles
    finally:
        plt.close(fig)


def test_write_sheet():
    """S5: write_node_sheet writes a non-empty PNG and leaves no open figure."""
    keep = os.environ.get("SCREENING_SHEETS_DIR")
    plt.close("all")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(keep) if keep else Path(tmp)
        out.mkdir(parents=True, exist_ok=True)
        for which in ("near", "far"):
            st = None
            if which == "far":
                st = ["ok"] * 33
                st[2] = "iter_max_rounds"
            view = make_view(which, invalid=(0,) if which == "far" else (), statuses=st)
            path = out / ("%s_%d.png" % (which, view.node_id))
            SF.write_node_sheet(view, marks_of(view), extent_of(view), DZ_UM, path, dpi=110)
            data = path.read_bytes()
            assert len(data) > 1000 and data[:8] == b"\x89PNG\r\n\x1a\n", (which, len(data))
            assert plt.get_fignums() == [], plt.get_fignums()


def test_determinism():
    """The same view gives the same PNG bytes twice (no randomness in the module)."""
    view = make_view("near")
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "a.png", Path(tmp) / "b.png"
        SF.write_node_sheet(view, marks_of(view), extent_of(view), DZ_UM, a, dpi=40)
        SF.write_node_sheet(view, marks_of(view), extent_of(view), DZ_UM, b, dpi=40)
        assert a.read_bytes() == b.read_bytes()


def test_edge_cases():
    """S6: a plane with valid False shows "not fetched" and no image; ks without k_star raises ValueError; the
    inputs are not modified."""
    view = make_view("near", invalid=(2,))
    before = view.stack.copy()
    fig = sheet(view)
    try:
        for q, (a_im, a_pr) in enumerate(pairs(fig)):
            nf = [t for t in a_im.texts if t.get_gid() == "not_fetched"]
            if q == 2:
                assert not a_im.images and len(nf) == 1 and nf[0].get_text() == "not fetched"
                assert len(lines(a_pr, "profile")) == 1                    # its profile still drawn
            else:
                assert len(a_im.images) == 1 and not nf
        assert_array_equal(view.stack, before)
    finally:
        plt.close(fig)
    # the focus plane not fetched: still singled out, its marks drawn on the empty panel
    view = make_view("near", invalid=(3,))
    fig = sheet(view)
    try:
        a_im, a_pr = pairs(fig)[3]
        assert not a_im.images and len(lines(a_im, "allen_marks")) == 1 and len(lines(a_pr, "allen_v")) == 2
    finally:
        plt.close(fig)
    bad = make_view("near")
    bad.k_star = int(bad.ks.max()) + 5
    plt.close("all")
    _raises(ValueError, SF.node_sheet, bad, marks_of(bad), extent_of(bad), DZ_UM)
    _raises(ValueError, SF.write_node_sheet, bad, marks_of(bad), extent_of(bad), DZ_UM, "/tmp/never.png", 50)
    assert plt.get_fignums() == [] and not os.path.exists("/tmp/never.png")
    # no valid plane at all: the grey scale falls back, the sheet is still drawn
    view = make_view("near", invalid=tuple(range(7)))
    fig = sheet(view)
    try:
        assert len(fig.axes) == 14 and not any(a.images for a in fig.axes)
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
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
