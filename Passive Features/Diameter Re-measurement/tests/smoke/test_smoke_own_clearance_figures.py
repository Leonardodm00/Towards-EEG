"""Smoke test for the same-node clearance figures -- Block 38 in specs/SPEC.md (D-156.2).

The module computes nothing scientific, so the oracles are about what is
drawn: the curves drawn equal the inputs, the chord bar's ends, the shared y
range, the layers' image and its three lines, the inputs not written to.

Test inputs, plain numpy: per plane k = -3..3 two non-negative Gaussians
S_in (narrow) and S_out (wide) of the offset v; profiles
B_k (1 - S_in - S_out) plus seeded noise; an own_clearance.OwnClearance
(Block 37's dataclass, built by hand: Block 37's functions are not called)
whose arrays follow Block 37's formulas written in the test; the chord 0 on
four of the seven planes; depth layers of thickness 0.1 um with Gaussian weights.

Checks
    test_known_answer   figure_own: per panel shown, in the order of planes,
                        the profile, cleared, in-band and whole-model lines are
                        the inputs on v [exact]; the background line at B_k;
                        the chord bar from v0 - c_k/2 to v0 + c_k/2 at one y
                        [exact], absent where c_k = 0; titles carry k, c_k and
                        the share ("n/a" for NaN)
    test_reference      figure_layers: the image is layers_gl[:, j, :] of the
                        plane's position j [exact]; its cells are centred on
                        (v_n, layer_z_m) (cell edges read back from the mesh,
                        midpoints against the inputs, 1e-12 um: float64 sums)
    test_convergence    skipped (no discretisation)
    test_invariants     all panels share one y range that contains every drawn
                        curve and the bar; one legend for the figure and none
                        per axis; inputs unchanged (read-only during the calls)
    test_contract       panel count = planes shown (None, a subset, a single
                        plane); a Figure each; axis labels carry units (um,
                        gl); figure_layers: one image (n_layers, n_v), the
                        plane line at z_k and the band lines at z_k -+ beta, a
                        colour bar labelled in gl; save writes non-empty PNGs
    test_determinism    two calls give bit-identical rendered canvases
    test_edge_cases     ValueError for shapes that disagree, an empty planes,
                        a plane not in ks, layers_gl None, bad band, dpi 0;
                        beta = 0 still draws three lines

Run (from "Passive Features/Diameter Re-measurement")
    python3 tests/smoke/test_smoke_own_clearance_figures.py
    python -m pytest tests/smoke/test_smoke_own_clearance_figures.py -q
Keep example PNGs: set OWN_FIGURES_OUT=<folder> (default: a temporary folder).

Pure ASCII, for the cluster.
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
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
SRC = WS / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from allen_diameter.analysis.own_clearance import OwnClearance  # noqa: E402
from allen_diameter.plotting import own_clearance_figures as OF  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "matplotlib")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []
_CACHE = {}
CURVES = ("profile", "cleared", "model_in", "model")


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
    d = os.environ.get("OWN_FIGURES_OUT")
    if d:
        os.makedirs(d, exist_ok=True)
        return d
    if "tmp" not in _CACHE:
        _CACHE["tmp"] = tempfile.mkdtemp(prefix="block38_")
    return _CACHE["tmp"]


def inputs():
    """A node on planes k = -3..3 (dz 0.28 um), d = 0.9 um, axis at z = 0.05 um; fresh arrays each call."""
    rng = make_rng()
    ks = np.arange(-3, 4)
    dz = 0.28
    planes_z = ks * dz
    v = np.linspace(-2.5, 2.5, 81)
    v0 = 0.12
    B = 92.0 + 3.0 * np.arange(ks.size)
    z_ax = 0.05
    d = 0.9
    dzk = planes_z - z_ax
    a_in = 0.25 * np.exp(-0.5 * (dzk / 0.2) ** 2)
    a_out = 0.12 * np.exp(-0.5 * (dzk / 0.6) ** 2)
    s_in = a_in[:, None] * np.exp(-0.5 * ((v[None, :] - v0) / 0.35) ** 2)
    s_out = a_out[:, None] * np.exp(-0.5 * ((v[None, :] - v0) / 0.9) ** 2)
    s_out[0] = 0.0
    s_in[0] = 0.0                                     # plane k = -3: no shadow (share NaN, as Block 37 gives)
    prof = B[:, None] * (1.0 - s_in - s_out) + 1.5 * rng.standard_normal((ks.size, v.size))
    # Block 37's formulas, written here only to make consistent inputs
    cleared = prof + B[:, None] * s_out
    model_in = B[:, None] * (1.0 - s_in)
    model = B[:, None] * (1.0 - s_in - s_out)
    chord = 2.0 * np.sqrt(np.maximum(d * d / 4.0 - dzk ** 2, 0.0))   # 0 on k = -3, -2, 2, 3
    whole = s_in + s_out
    nstar = np.argmax(whole, axis=1)
    rows = np.arange(ks.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(whole[rows, nstar] > 0, s_in[rows, nstar] / whole[rows, nstar], np.nan)
    h = 0.1
    layer_z = z_ax + h * np.arange(-5, 6)
    w = np.exp(-0.5 * ((layer_z[:, None] - z_ax) / 0.25) ** 2)
    w = w / w.sum(axis=0)
    layers = w[:, :, None] * whole[None, :, :]          # (n_layers, n_planes, n_v), sums to the whole
    layers_gl = B[None, :, None] * layers
    clr = OwnClearance(cleared_gl=cleared, model_in_gl=model_in, model_gl=model, residual_gl=prof - model,
                       chord_um=chord, in_peak_gl=B * s_in.max(axis=1), out_peak_gl=B * s_out.max(axis=1),
                       in_share=share, residual_rms_gl=np.sqrt(np.mean((prof - model) ** 2, axis=1)),
                       layers_gl=layers_gl)
    meta = {"node_id": 4711, "class": "far_isolated", "round": 2, "d_um": d, "mu_per_um": 1.3,
            "band_half_um": 0.14}
    return dict(v=v, ks=ks, profiles_gl=prof, B_gl=B, clearance=clr, meta=meta, v0_um=v0, planes_z=planes_z,
                layer_z=layer_z, layers_gl=layers_gl, band_half_um=0.14)


def call_own(a, planes=None):
    return OF.figure_own(a["v"], a["ks"], a["profiles_gl"], a["B_gl"], a["clearance"], a["meta"], planes=planes,
                         v0_um=a["v0_um"])


def call_layers(a, plane=0):
    return OF.figure_layers(a["v"], a["layer_z"], a["layers_gl"], a["ks"], plane, a["planes_z"],
                            a["band_half_um"], a["meta"])


def panels(fig):
    return [ax for ax in fig.axes if ax.get_visible() and any(ln.get_gid() == "profile" for ln in ax.lines)]


def lines_by_gid(ax, gid):
    return [ln for ln in ax.lines if ln.get_gid() == gid]


def line_by_gid(ax, gid):
    found = lines_by_gid(ax, gid)
    assert len(found) == 1, "%s: %d lines" % (gid, len(found))
    return found[0]


def cols_of(planes, ks):
    ks = [int(k) for k in ks]
    return [ks.index(int(k)) for k in planes] if planes is not None else list(range(len(ks)))


def mesh_of(fig):
    m = [c for ax in fig.axes for c in ax.collections if isinstance(c, QuadMesh) and c.get_gid() == "layers"]
    assert len(m) == 1, len(m)
    return m[0]


def _raises(fn, what):
    try:
        fig = fn()
    except ValueError:
        return
    plt.close(fig)
    raise AssertionError("no ValueError for %s" % what)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    a = inputs()
    planes = [1, -3, 0, 3, -1]                         # out of order; -3 and 3 have c_k = 0, -3 share NaN
    fig = call_own(a, planes=planes)
    try:
        axs = panels(fig)
        assert len(axs) == len(planes), len(axs)
        c = a["clearance"]
        want = {"profile": a["profiles_gl"], "cleared": c.cleared_gl, "model_in": c.model_in_gl, "model": c.model_gl}
        bar_y = set()
        for ax, j in zip(axs, cols_of(planes, a["ks"])):
            for gid in CURVES:
                ln = line_by_gid(ax, gid)
                # exact: the module draws its inputs
                assert_array_equal(np.asarray(ln.get_xdata(), dtype=float), a["v"])
                assert_array_equal(np.asarray(ln.get_ydata(), dtype=float), want[gid][j])
            bg = line_by_gid(ax, "background")
            assert list(np.asarray(bg.get_ydata(), dtype=float)) == [a["B_gl"][j]] * 2
            ck = float(c.chord_um[j])
            bars = lines_by_gid(ax, "chord")
            if ck == 0.0:
                assert not bars, "a chord bar on plane %d with c_k = 0" % a["ks"][j]
            else:
                assert len(bars) == 1
                x = np.asarray(bars[0].get_xdata(), dtype=float)
                y = np.asarray(bars[0].get_ydata(), dtype=float)
                assert_array_equal(x, [a["v0_um"] - ck / 2.0, a["v0_um"] + ck / 2.0])
                assert y[0] == y[1]
                bar_y.add(float(y[0]))
            t = ax.get_title()
            assert ("k = %d," % a["ks"][j]) in t, t
            assert ("c_k = %.3g um" % ck) in t, t
            sh = c.in_share[j]
            assert (("%.3f" % sh) if np.isfinite(sh) else "n/a") in t, t
        assert len(bar_y) == 1, bar_y                  # one bar height across panels
        assert sum(1 for j in cols_of(planes, a["ks"]) if a["clearance"].chord_um[j] > 0) == 3   # k = 1, 0, -1
        st = fig._suptitle.get_text()
        for piece in ("node 4711", "class far_isolated", "round 2", "d = 0.9 um", "mu = 1.3 /um", "band +-0.14 um"):
            assert piece in st, (piece, st)
    finally:
        plt.close(fig)


def test_reference():
    a = inputs()
    for plane in (0, -2, 3):
        fig = call_layers(a, plane=plane)
        try:
            j = cols_of([plane], a["ks"])[0]
            m = mesh_of(fig)
            arr = np.asarray(m.get_array(), dtype=float).reshape(a["layer_z"].size, a["v"].size)
            assert_array_equal(arr, a["layers_gl"][:, j, :])
            xy = m.get_coordinates()                     # (n_layers + 1, n_v + 1, 2) cell corners
            xe = xy[0, :, 0]
            ye = xy[:, 0, 1]
            # 1e-12 um: midpoints of edges matplotlib built from the centres, float64 sums
            check_close("cell_x_%d" % plane, 0.5 * (xe[:-1] + xe[1:]), a["v"], rtol=0, atol=1e-12)
            check_close("cell_z_%d" % plane, 0.5 * (ye[:-1] + ye[1:]), a["layer_z"], rtol=0, atol=1e-12)
        finally:
            plt.close(fig)


def test_convergence():
    raise unittest.SkipTest("plotting only: no discretisation")


def test_invariants():
    a = inputs()
    c = a["clearance"]
    arrays = [a["v"], a["ks"], a["profiles_gl"], a["B_gl"], a["planes_z"], a["layer_z"], a["layers_gl"],
              c.cleared_gl, c.model_in_gl, c.model_gl, c.chord_um, c.in_share]
    copies = [np.array(x, copy=True) for x in arrays]
    for x in arrays:
        x.flags.writeable = False                      # a write inside the module raises
    fig = call_own(a)
    try:
        axs = panels(fig)
        lims = {tuple(ax.get_ylim()) for ax in axs}
        assert len(lims) == 1, lims
        lo, hi = lims.pop()
        for ax in axs:
            for ln in ax.lines:
                y = np.asarray(ln.get_ydata(), dtype=float)
                assert lo < y.min() and y.max() < hi, (ln.get_gid(), y.min(), y.max(), lo, hi)
            assert ax.get_legend() is None
        assert len(fig.legends) == 1
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        assert len(labels) == 6, labels
    finally:
        plt.close(fig)
    fig = call_layers(a, plane=1)
    plt.close(fig)
    for x, y in zip(arrays, copies):
        assert_array_equal(x, y)


def test_contract():
    a = inputs()
    for planes, n in ((None, 7), ([0], 1), ([2, -2, 0], 3)):
        fig = call_own(a, planes=planes)
        try:
            assert isinstance(fig, Figure)
            axs = panels(fig)
            assert len(axs) == n, (planes, len(axs))
            assert all("um" in ax.get_xlabel() for ax in axs if ax.get_xlabel())
            assert any(ax.get_xlabel() == "offset v (um)" for ax in axs)
            assert any(ax.get_ylabel() == "grey level (gl)" for ax in axs)
        finally:
            plt.close(fig)
    fig = call_layers(a, plane=1)
    try:
        assert isinstance(fig, Figure)
        m = mesh_of(fig)
        assert np.asarray(m.get_array()).reshape(-1).size == a["layer_z"].size * a["v"].size
        assert np.asarray(m.get_array()).shape in ((a["layer_z"].size, a["v"].size),
                                                   (a["layer_z"].size * a["v"].size,))
        ax = m.axes
        z_k = a["planes_z"][list(a["ks"]).index(1)]
        pl = line_by_gid(ax, "plane_z")
        assert list(np.asarray(pl.get_ydata(), dtype=float)) == [z_k, z_k]
        band = sorted(float(ln.get_ydata()[0]) for ln in lines_by_gid(ax, "band"))
        assert band == [z_k - a["band_half_um"], z_k + a["band_half_um"]], band
        assert ax.get_xlabel() == "offset v (um)" and ax.get_ylabel() == "layer depth z (um)"
        cbars = [x for x in fig.axes if x is not ax]
        assert len(cbars) == 1 and "(gl)" in cbars[0].get_ylabel(), [x.get_ylabel() for x in cbars]
    finally:
        plt.close(fig)
    folder = out_dir()
    paths = []
    fig = call_own(a)
    paths.append(OF.save(fig, os.path.join(folder, "own_node_example.png"), 110))
    plt.close(fig)
    fig = call_layers(a, plane=0)
    paths.append(OF.save(fig, os.path.join(folder, "sub", "own_layers_example.png"), 110))
    plt.close(fig)
    for pth in paths:
        assert os.path.getsize(pth) > 1000, pth
        with open(pth, "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n", pth


def test_determinism():
    makers = (lambda: call_own(inputs(), planes=[0, 2, -3]), lambda: call_layers(inputs(), plane=0))
    for make in makers:
        bufs = []
        for _ in range(2):
            fig = make()
            fig.canvas.draw()
            bufs.append(np.array(fig.canvas.buffer_rgba(), copy=True))
            plt.close(fig)
        assert_array_equal(bufs[0], bufs[1])


def test_edge_cases():
    import dataclasses
    a = inputs()
    c = a["clearance"]

    def own(**ch):
        b = dict(a)
        b.update(ch)
        return lambda: OF.figure_own(b["v"], b["ks"], b["profiles_gl"], b["B_gl"], b["clearance"], b["meta"],
                                     planes=ch.get("planes"), v0_um=b["v0_um"])

    _raises(own(profiles_gl=a["profiles_gl"][:, :-1]), "profiles of another length")
    _raises(own(profiles_gl=a["profiles_gl"][0]), "1-D profiles")
    _raises(own(B_gl=a["B_gl"][:-1]), "B of the wrong length")
    _raises(own(v=a["v"][:-1]), "v of the wrong length")
    _raises(own(ks=a["ks"][:-1]), "ks of the wrong length")
    for field in ("cleared_gl", "model_in_gl", "model_gl"):
        _raises(own(clearance=dataclasses.replace(c, **{field: getattr(c, field)[:-1]})), "%s short" % field)
    for field in ("chord_um", "in_share"):
        _raises(own(clearance=dataclasses.replace(c, **{field: getattr(c, field)[:-1]})), "%s short" % field)
    _raises(own(planes=[]), "an empty planes")
    _raises(own(planes=[9]), "a plane not in ks")
    _raises(own(meta="node 1"), "meta not a mapping")
    _raises(own(v0_um=float("nan")), "v0 NaN")

    def lay(**ch):
        b = dict(a)
        b.update(ch)
        return lambda: OF.figure_layers(b["v"], b["layer_z"], b["layers_gl"], b["ks"], ch.get("plane", 0),
                                        b["planes_z"], b["band_half_um"], b["meta"])

    _raises(lay(layers_gl=None), "layers_gl None")
    _raises(lay(layers_gl=a["layers_gl"][:, :-1]), "layers of the wrong plane count")
    _raises(lay(layers_gl=a["layers_gl"][:-1]), "layers and layer_z of different lengths")
    _raises(lay(layer_z=a["layer_z"][:0], layers_gl=a["layers_gl"][:0]), "no layer")
    _raises(lay(planes_z=a["planes_z"][:-1]), "planes_z of the wrong length")
    _raises(lay(v=a["v"][:-1]), "v of the wrong length")
    _raises(lay(plane=9), "a plane not in ks")
    _raises(lay(band_half_um=-0.1), "a negative band")
    _raises(lay(band_half_um=float("inf")), "an infinite band")
    _raises(lay(meta=None), "meta not a mapping")
    fig = plt.figure()
    try:
        _raises(lambda: OF.save(fig, os.path.join(out_dir(), "x.png"), 0), "dpi 0")
    finally:
        plt.close(fig)
    # beta = 0: the band lines fall on the plane line, all three still drawn
    fig = lay(band_half_um=0.0)()
    try:
        ax = mesh_of(fig).axes
        assert len(lines_by_gid(ax, "band")) == 2 and len(lines_by_gid(ax, "plane_z")) == 1
    finally:
        plt.close(fig)
    # every chord 0: no bar anywhere
    fig = own(clearance=dataclasses.replace(c, chord_um=np.zeros_like(c.chord_um)))()
    try:
        assert not any(ln.get_gid() == "chord" for ax in fig.axes for ln in ax.lines)
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
