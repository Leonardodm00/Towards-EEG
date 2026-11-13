"""Smoke test for the inference figures -- Block 34 in specs/SPEC.md (plan steps F5, E; D-129.4).

The module computes nothing scientific, so the oracles are about what is
drawn: the profiles, the model, the residual over sigma, the diameters and the
chi2 values drawn equal the inputs; unused planes are greyed; rows not
converged are hollow; the inputs are not written to; the axes' counts.

Test inputs, plain numpy: a Gaussian dip profile B (1 - a exp(-v^2 / 2 w^2))
per plane as the model, plus seeded noise as the cleaned profile; a NodeFit
(analysis/node_inference.py, Block 30) built by hand with one unused plane;
diameter rows of three classes, two of them not converged, some without
d_pilot_um; chi2 rows with NaN for unused planes.

Checks
    test_known_answer   figure_fit: the profile, model and residual lines are
                        profiles_gl, model_gl and (profiles_gl - model_gl) /
                        sigma_gl on v; the residual limits at +-2.
                        figure_diameters: per panel the drawn points are the
                        rows' (x, d_fit) by class and convergence; identity
                        y = x drawn. figure_chi2: one line per node equal to
                        (ks, chi2_per_plane), a line at 1
    test_reference      each errorbar segment spans d_fit -+ se exactly
    test_convergence    skipped (no discretisation)
    test_invariants     unused planes greyed and used ones not; hollow markers
                        exactly for the rows not converged; classes by
                        distinct markers; inputs unchanged (read-only during
                        the calls)
    test_contract       2 axes per plane shown (planes None, a subset, a
                        wrapped 7-plane case); figure_diameters 2 axes sharing
                        y; figure_chi2 1 axis; the d bounds drawn with cfg;
                        save writes a non-empty PNG
    test_determinism    two calls give bit-identical rendered canvases
    test_edge_cases     ValueError for shapes that disagree, an empty planes,
                        a plane not in ks, bad sigma, empty rows, a row
                        without a required key, dpi 0

Run (from "Passive Features/Diameter Re-measurement")
    python3 tests/smoke/test_smoke_inference_figures.py
    python -m pytest tests/smoke/test_smoke_inference_figures.py -q
Keep example PNGs: set INFERENCE_FIGURES_OUT=<folder> (default: a temporary folder).

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
for p in (str(SRC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from allen_diameter.analysis.node_inference import NodeFit  # noqa: E402
from allen_diameter.config import default_config  # noqa: E402
from allen_diameter.plotting import inference_figures as IF  # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []
_CACHE = {}


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
    d = os.environ.get("INFERENCE_FIGURES_OUT")
    if d:
        os.makedirs(d, exist_ok=True)
        return d
    if "tmp" not in _CACHE:
        _CACHE["tmp"] = tempfile.mkdtemp(prefix="block34_")
    return _CACHE["tmp"]


def fit_inputs(n_planes=5):
    """A node fitted in planes ks = -2..2 (n_planes = 5) or -3..3 (7); plane k = 2 (or the last) not used."""
    key = "fit%d" % n_planes
    if key in _CACHE:
        return _CACHE[key]
    rng = make_rng()
    half = n_planes // 2
    ks = np.arange(-half, half + 1)
    v = np.linspace(-3.0, 3.0, 61)
    B = 210.0
    sigma = 3.0
    amp = 0.35 * np.exp(-0.5 * (ks / 1.5) ** 2)
    width = 0.4 + 0.15 * np.abs(ks)
    model = B * (1.0 - amp[:, None] * np.exp(-0.5 * (v[None, :] / width[:, None]) ** 2))
    prof = model + sigma * rng.standard_normal(model.shape)
    used = np.ones(n_planes, dtype=bool)
    used[-1] = False
    chi2 = np.mean(((prof - model) / sigma) ** 2, axis=1)
    chi2[~used] = np.nan
    fit = NodeFit(theta=np.array([0.82, 1.3, 0.04, -0.1]), cov=np.diag([0.021, 0.08, 0.01, 0.05]) ** 2,
                  se=np.array([0.021, 0.08, 0.01, 0.05]), cost=float(np.nansum(chi2) * v.size),
                  chi2_per_plane=chi2, planes=used, status="converged", at_bound=(), n_starts=3, nfev=57)
    meta = {"node_id": 4711, "class": "far_isolated"}
    out = dict(profiles_gl=prof, model_gl=model, v=v, ks=ks, sigma_gl=sigma, fit=fit, meta=meta)
    _CACHE[key] = out
    return out


def call_fit(a, planes=None):
    return IF.figure_fit(a["profiles_gl"], a["model_gl"], a["v"], a["ks"], a["sigma_gl"], a["fit"], a["meta"],
                         planes=planes)


def diam_rows():
    """12 rows, three classes, statuses mixed, d_pilot_um absent, None or NaN on some."""
    rng = make_rng(7)
    classes = ("thin", "far", "crossing")
    rows = []
    for i in range(12):
        d_a = 0.4 + 0.12 * i
        row = {"node_id": 1000 + i, "d_allen_um": d_a, "d_fit_um": d_a * (0.8 + 0.1 * rng.standard_normal()),
               "se_um": 0.02 + 0.01 * (i % 3), "status": "converged", "class": classes[i % 3]}
        if i % 4 != 3:
            row["d_pilot_um"] = d_a * (0.9 + 0.05 * rng.standard_normal())
        if i == 5:
            row["d_pilot_um"] = float("nan")
        if i in (2, 7):
            row["status"] = "at_bound"
        if i == 9:
            row["status"] = "failed"
            row["se_um"] = float("nan")
        rows.append(row)
    return rows


def chi2_rows(n=6):
    rng = make_rng(11)
    rows = []
    for i in range(n):
        ks = np.arange(-3, 4)
        c = 1.0 + 0.2 * rng.standard_normal(ks.size) + 0.3 * (i == 2) * np.abs(ks)
        c[[0, -1]] = np.nan if i % 2 else c[[0, -1]]
        rows.append({"node_id": 2000 + i, "chi2_per_plane": c, "ks": ks})
    return rows


def plane_axes(fig):
    """(profile axes, residual axes) of a figure_fit figure, in drawing order."""
    profs = [ax for ax in fig.axes if any(ln.get_gid() == "profile" for ln in ax.lines)]
    resids = [ax for ax in fig.axes if any(ln.get_gid() == "residual" for ln in ax.lines)]
    return profs, resids


def line_by_gid(ax, gid):
    found = [ln for ln in ax.lines if ln.get_gid() == gid]
    assert len(found) == 1, "%s: %d lines" % (gid, len(found))
    return found[0]


def cols_of(planes, ks):
    ks = [int(k) for k in ks]
    return [ks.index(int(k)) for k in planes] if planes is not None else list(range(len(ks)))


def _sel(rows, xkey, cls, conv):
    out = []
    for r in rows:
        x = r.get(xkey)
        if x is None or not np.isfinite(float(x)):
            continue
        if r.get("class", "unclassified") == cls and (r["status"] == "converged") == conv:
            out.append(r)
    return out


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    a = fit_inputs()
    planes = [0, 2, -2]
    fig = call_fit(a, planes=planes)
    try:
        profs, resids = plane_axes(fig)
        for i, j in enumerate(cols_of(planes, a["ks"])):
            pl = line_by_gid(profs[i], "profile")
            ml = line_by_gid(profs[i], "model")
            rl = line_by_gid(resids[i], "residual")
            # exact: the module draws its inputs (and one display transform written the same way)
            assert_array_equal(np.asarray(pl.get_xdata(), dtype=float), a["v"])
            assert_array_equal(np.asarray(pl.get_ydata(), dtype=float), a["profiles_gl"][j])
            assert_array_equal(np.asarray(ml.get_ydata(), dtype=float), a["model_gl"][j])
            assert_array_equal(np.asarray(rl.get_ydata(), dtype=float),
                               (a["profiles_gl"][j] - a["model_gl"][j]) / a["sigma_gl"])
            lims = sorted(float(ln.get_ydata()[0]) for ln in resids[i].lines if ln.get_gid() == "resid_limit")
            assert lims == [-2.0, 2.0], lims
    finally:
        plt.close(fig)

    rows = diam_rows()
    fig = IF.figure_diameters(rows)
    try:
        for ax, xkey in zip(fig.axes, ("d_allen_um", "d_pilot_um")):
            n_drawn = 0
            for cls in ("thin", "far", "crossing"):
                for conv in (True, False):
                    grp = _sel(rows, xkey, cls, conv)
                    gid = "data|%s|%s" % (cls, "converged" if conv else "not_converged")
                    lines = [ln for ln in ax.lines if ln.get_gid() == gid]
                    if not grp:
                        assert not lines, gid
                        continue
                    assert len(lines) == 1, gid
                    assert_array_equal(np.asarray(lines[0].get_xdata(), dtype=float), [r[xkey] for r in grp])
                    assert_array_equal(np.asarray(lines[0].get_ydata(), dtype=float), [r["d_fit_um"] for r in grp])
                    n_drawn += len(grp)
            n_expect = sum(1 for r in rows if r.get(xkey) is not None and np.isfinite(float(r[xkey])))
            assert n_drawn == n_expect, (xkey, n_drawn, n_expect)
            ident = line_by_gid(ax, "identity")
            x, y = (np.asarray(t, dtype=float) for t in ident.get_data())
            assert_array_equal(x, y)
        assert n_expect < len(rows)          # the second panel really drops the rows without d_pilot_um
    finally:
        plt.close(fig)

    crow = chi2_rows()
    fig = IF.figure_chi2(crow)
    try:
        ax = fig.axes[0]
        for i, r in enumerate(crow):
            ln = line_by_gid(ax, "chi2|%d" % i)
            assert_array_equal(np.asarray(ln.get_xdata(), dtype=float), r["ks"])
            assert_array_equal(np.asarray(ln.get_ydata(), dtype=float), r["chi2_per_plane"])
        u = line_by_gid(ax, "unity")
        assert list(u.get_ydata()) == [1.0, 1.0]
    finally:
        plt.close(fig)


def test_reference():
    # the error bars, read back from matplotlib's LineCollection: each spans d_fit - se to d_fit + se at x
    rows = diam_rows()
    fig = IF.figure_diameters(rows)
    try:
        ax = fig.axes[0]
        segs = []
        for coll in ax.collections:
            segs.extend(np.asarray(s, dtype=float) for s in coll.get_segments())
        expect = [r for r in rows if np.isfinite(float(r["se_um"]))]
        assert len(segs) == len(expect), (len(segs), len(expect))
        got = sorted((float(s[0, 0]), float(min(s[:, 1])), float(max(s[:, 1]))) for s in segs)
        want = sorted((r["d_allen_um"], r["d_fit_um"] - r["se_um"], r["d_fit_um"] + r["se_um"]) for r in expect)
        # 1e-15 relative: matplotlib forms y -+ se in float64, the test the same sum
        check_close("errorbars", got, want, rtol=1e-15, atol=0)
    finally:
        plt.close(fig)


def test_convergence():
    raise unittest.SkipTest("plotting only: no discretisation")


def _rgb(c):
    return mcolors.to_hex(c)


def test_invariants():
    a = fit_inputs()
    keys = ("profiles_gl", "model_gl", "v", "ks")
    copies = {k: np.array(a[k], copy=True) for k in keys}
    fcopies = {k: np.array(getattr(a["fit"], k), copy=True) for k in ("theta", "se", "chi2_per_plane", "planes")}
    for k in keys:
        a[k].flags.writeable = False
    for k in fcopies:
        getattr(a["fit"], k).flags.writeable = False
    try:
        fig = call_fit(a)
        profs, resids = plane_axes(fig)
        used = a["fit"].planes
        for j in range(a["ks"].size):
            cols = {_rgb(ln.get_color()) for ln in profs[j].lines + resids[j].lines
                    if ln.get_gid() in ("profile", "model", "residual", "resid_limit")}
            if used[j]:
                assert IF.GREYED not in cols, (j, cols)
            else:
                assert cols == {IF.GREYED}, (j, cols)
                assert "not used" in profs[j].get_title()
        plt.close(fig)
    finally:
        for k in keys:
            a[k].flags.writeable = True
        for k in fcopies:
            getattr(a["fit"], k).flags.writeable = True
    for k, ref in copies.items():
        assert_array_equal(a[k], ref)
    for k, ref in fcopies.items():
        assert_array_equal(getattr(a["fit"], k), ref)

    rows = diam_rows()
    snapshot = [dict(r) for r in rows]
    fig = IF.figure_diameters(rows, default_config())
    try:
        markers = {}
        for ax in fig.axes:
            for ln in ax.lines:
                gid = ln.get_gid() or ""
                if not gid.startswith("data|"):
                    continue
                _, cls, conv = gid.split("|")
                hollow = ln.get_markerfacecolor() == "none"
                assert hollow == (conv == "not_converged"), gid
                markers.setdefault(cls, set()).add(ln.get_marker())
        assert all(len(m) == 1 for m in markers.values()), markers
        assert len({next(iter(m)) for m in markers.values()}) == len(markers), "classes share a marker"
        # every row not converged that is drawn is in a hollow group
        n_hollow = sum(len(ln.get_xdata()) for ln in fig.axes[0].lines
                       if (ln.get_gid() or "").endswith("not_converged"))
        assert n_hollow == sum(1 for r in rows if r["status"] != "converged")
    finally:
        plt.close(fig)
    for r, s in zip(rows, snapshot):
        assert r.keys() == s.keys()
        for k in r:
            assert (r[k] == s[k]) or (r[k] != r[k] and s[k] != s[k]), k

    crow = chi2_rows()
    ccopy = [np.array(r["chi2_per_plane"], copy=True) for r in crow]
    for r in crow:
        r["chi2_per_plane"].flags.writeable = False
    fig = IF.figure_chi2(crow)
    plt.close(fig)
    for r, c in zip(crow, ccopy):
        r["chi2_per_plane"].flags.writeable = True
        assert_array_equal(r["chi2_per_plane"], c)


def test_contract():
    a = fit_inputs()
    for planes, n in ((None, a["ks"].size), ([0], 1), ([2, -2], 2)):
        fig = call_fit(a, planes=planes)
        try:
            assert isinstance(fig, Figure)
            profs, resids = plane_axes(fig)
            assert len(profs) == n and len(resids) == n, (planes, len(profs), len(resids))
            assert len(fig.axes) == 2 * n, (planes, len(fig.axes))
        finally:
            plt.close(fig)
    a7 = fit_inputs(7)                       # wraps after IF.FIT_COLS planes
    fig = call_fit(a7)
    try:
        assert len(fig.axes) == 14
    finally:
        plt.close(fig)
    fig = IF.figure_diameters(diam_rows())
    try:
        assert isinstance(fig, Figure) and len(fig.axes) == 2
        assert fig.axes[0].get_shared_y_axes().joined(fig.axes[0], fig.axes[1])
        assert not [ln for ax in fig.axes for ln in ax.lines if ln.get_gid() == "d_bound"]
        assert fig.legends, "no legend"
    finally:
        plt.close(fig)
    cfg = default_config()
    fig = IF.figure_diameters(diam_rows(), cfg)   # bounds outside the drawn range: no line, no legend entry
    try:
        assert not [ln for ax in fig.axes for ln in ax.lines if ln.get_gid() == "d_bound"]
        assert "fit bounds on d" not in [t.get_text() for t in fig.legends[0].get_texts()]
    finally:
        plt.close(fig)
    rows_wide = diam_rows() + [{"node_id": 1, "d_allen_um": 0.05, "d_fit_um": 0.1, "se_um": 0.0, "status": "at_bound"}]
    fig = IF.figure_diameters(rows_wide, cfg)
    try:
        b = [float(ln.get_ydata()[0]) for ax in fig.axes for ln in ax.lines if ln.get_gid() == "d_bound"]
        assert cfg.inference.d_bounds_um[0] in b, b      # the lower bound 0.1 um lies in the drawn range
        assert "fit bounds on d" in [t.get_text() for t in fig.legends[0].get_texts()]
    finally:
        plt.close(fig)
    fig = IF.figure_chi2(chi2_rows())
    try:
        assert isinstance(fig, Figure) and len(fig.axes) == 1
        assert fig.axes[0].get_legend() is not None
    finally:
        plt.close(fig)
    folder = out_dir()
    paths = []
    fig = call_fit(a)
    paths.append(IF.save(fig, os.path.join(folder, "inference_fit_example.png"), 110))
    plt.close(fig)
    fig = IF.figure_diameters(rows_wide, cfg)
    paths.append(IF.save(fig, os.path.join(folder, "sub", "inference_diameters_example.png"), 110))
    plt.close(fig)
    fig = IF.figure_chi2(chi2_rows(20))
    paths.append(IF.save(fig, os.path.join(folder, "inference_chi2_example.png"), 110))
    plt.close(fig)
    for pth in paths:
        assert os.path.getsize(pth) > 1000, pth
        with open(pth, "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n", pth


def test_determinism():
    makers = (lambda: call_fit(fit_inputs(), planes=[0, 2]),
              lambda: IF.figure_diameters(diam_rows(), default_config()),
              lambda: IF.figure_chi2(chi2_rows()))
    for make in makers:
        bufs = []
        for _ in range(2):
            fig = make()
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
    a = fit_inputs()

    def with_(**changes):
        b = dict(a)
        b.update(changes)
        return lambda: IF.figure_fit(b["profiles_gl"], b["model_gl"], b["v"], b["ks"], b["sigma_gl"], b["fit"],
                                     b["meta"], planes=changes.get("planes"))

    import dataclasses
    _raises(with_(model_gl=a["model_gl"][:, :-1]), "model of another shape")
    _raises(with_(profiles_gl=a["profiles_gl"][0]), "1-D profiles")
    _raises(with_(v=a["v"][:-1]), "v of the wrong length")
    _raises(with_(ks=a["ks"][:-1]), "ks of the wrong length")
    _raises(with_(fit=dataclasses.replace(a["fit"], planes=a["fit"].planes[:-1])), "fit.planes of the wrong length")
    _raises(with_(fit=dataclasses.replace(a["fit"], chi2_per_plane=a["fit"].chi2_per_plane[:-1])),
            "fit.chi2_per_plane of the wrong length")
    _raises(with_(planes=[]), "an empty planes")
    _raises(with_(planes=[9]), "a plane not in ks")
    _raises(with_(sigma_gl=0.0), "sigma 0")
    _raises(with_(sigma_gl=float("nan")), "sigma NaN")
    _raises(with_(meta="node 1"), "meta not a mapping")
    _raises(lambda: IF.figure_diameters([]), "empty diameter rows")
    for key in IF.DIAM_KEYS:
        r = dict(diam_rows()[0])
        del r[key]
        _raises(lambda: IF.figure_diameters([r]), "a diameter row without %s" % key)
    _raises(lambda: IF.figure_chi2([]), "empty chi2 rows")
    for key in IF.CHI2_KEYS:
        r = dict(chi2_rows()[0])
        del r[key]
        _raises(lambda: IF.figure_chi2([r]), "a chi2 row without %s" % key)
    r = dict(chi2_rows()[0])
    r["ks"] = r["ks"][:-1]
    _raises(lambda: IF.figure_chi2([r]), "chi2 and ks of different lengths")
    fig = plt.figure()
    try:
        _raises(lambda: IF.save(fig, os.path.join(out_dir(), "x.png"), 0), "dpi 0")
    finally:
        plt.close(fig)
    # a row set where no row gives d_pilot_um still draws (the panel says so)
    rows = [{k: v for k, v in r.items() if k != "d_pilot_um"} for r in diam_rows()]
    fig = IF.figure_diameters(rows)
    try:
        assert any("no row gives" in t.get_text() for t in fig.axes[1].texts)
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
