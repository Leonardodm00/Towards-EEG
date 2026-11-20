"""Smoke test for the calibration's figures (plotting/sigma_figures.py) -- Block 43 in specs/SPEC.md (D-201).

The inputs are built from the contract alone: plain synthetic arrays and dicts shaped as the Block 42 and 43
entries state (Block 42's code is not imported: it is written in parallel). Plotting only, so the oracles are
the inputs themselves: what is drawn must be what was given.

Checks
    test_known_answer   figure_node_fit: one profile, model, background and residual line per plane whose data
                        equal the inputs [exact] (the residual equals profiles - model [exact]); one image per
                        plane on one grey range; the profile panels share one y range and the residual panels
                        another, with the zero line; the sharpest plane framed; the measuring line drawn from the
                        first to the last offset along y_hat; the view +-zoom_um; the text panel and the title.
                        figure_summary: the error bars equal the SEs (both panels of SE per node, sorted by
                        sigma0); the mean line at ivw_mean_um, the band at +-ivw_se_um, the median line; not-ok
                        nodes hollow in every panel; Allen's diameter marked at (d_allen, sigma0)
    test_reference      the grey range against a brute-force selection of the pixels in view (pixel by pixel,
                        from the extent) and numpy's percentiles; the error bars against a loop over the rows
    test_convergence    skipped (plotting only, no discretisation)
    test_invariants     a permutation of the rows draws the same sorted panels; pixels outside the view do not
                        change the grey range; every y range contains its data and the residual's is symmetric
    test_contract       Figure objects, the panels' count, the optional keys left out, zoom None, NaN samples
                        and an all-NaN model drawn as gaps, a summary of NaN (no ok node) drawn without its lines;
                        both figures saved by save() as PNGs (tests/smoke/artifacts/, git-ignored, to be viewed)
    test_determinism    the same inputs twice: the same pixels
    test_edge_cases     the ValueErrors of the contract (a node not a mapping or lacking a key, shapes that
                        disagree, an empty rows) and the guards next to them; inputs unchanged

Run (from the code folder "Passive Features/Diameter Re-measurement")
    python3 tests/smoke/test_smoke_sigma_figures.py
    python3 -m pytest tests/smoke/test_smoke_sigma_figures.py -q

Pure ASCII, for the cluster.
"""
from __future__ import annotations

import copy
import importlib.metadata
import math
import platform
import re
import sys
import time
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.colors import to_rgba  # noqa: E402

from allen_diameter.plotting import sigma_figures as SF  # noqa: E402
from allen_diameter.plotting.cleaning_figures import IMAGE_PERCENTILES, ORANGE  # noqa: E402
from allen_diameter.plotting.figures import image_extent_um  # noqa: E402

SEED = 20261120
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run

P_UM = 0.1144                       # the pixel pitch (SPEC 2.1)
KS = np.arange(40, 45)              # k* - 2 .. k* + 2 (D-201: 5 planes)
K_INDEX = 2
V = (np.arange(79) - 39) * P_UM     # the line +-4.46 um of Cell 23A (79 samples)
LEFT, TOP, HW = 3000, 5000, 52      # a patch of 105 x 105 pixels (12 um), origin at its centre
THETA = 0.35                        # the axis' heading in the image (rad)


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    """Fresh generator, to be passed explicitly to the code under test."""
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure (saved to ARTIFACT_DIR/<name>.npz)."""
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


def _raises(exc, func, *args, **kw):
    try:
        func(*args, **kw)
    except exc:
        return
    raise AssertionError("%s not raised by %s" % (exc.__name__, getattr(func, "__name__", func)))


def _close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def synthetic_node(rng=None, background=True):
    """A thick node on 5 planes, as the entry point would pass it (SPEC Block 43): a dark band along the axis
    through the line's origin, its width growing away from k*; the recorded profiles are the model plus noise,
    the model a straight background times a Gaussian dip (any shape serves: the figure only draws it)."""
    rng = make_rng() if rng is None else rng
    frame = SimpleNamespace(left=LEFT, top=TOP, res_um_px=P_UM, downsample=0)
    n = 2 * HW + 1
    extent = np.asarray(image_extent_um(frame, (KS.size, n, n)), dtype=float)
    origin = np.array([(LEFT + HW) * P_UM, (TOP + HW) * P_UM])
    e_u = np.array([math.cos(THETA), math.sin(THETA)])
    y_hat = np.array([-math.sin(THETA), math.cos(THETA)])
    rows, cols = np.mgrid[0:n, 0:n]
    dx, dy = (LEFT + cols) * P_UM - origin[0], (TOP + rows) * P_UM - origin[1]
    across = dx * y_hat[0] + dy * y_hat[1]
    b0, b1 = 150.0, -2.0
    width = 0.75 + 0.25 * np.abs(np.arange(KS.size) - K_INDEX)      # um
    depth = 0.55 / (1.0 + 0.4 * np.abs(np.arange(KS.size) - K_INDEX))
    images = np.stack([b0 * (1.0 - depth[i] * np.exp(-0.5 * (across / width[i]) ** 2)) for i in range(KS.size)])
    images = images + rng.normal(0.0, 1.1, images.shape)
    bg = np.tile(b0 + b1 * V, (KS.size, 1))
    model = np.stack([(b0 + b1 * V) * (1.0 - depth[i] * np.exp(-0.5 * (V / width[i]) ** 2)) for i in range(KS.size)])
    prof = model + rng.normal(0.0, 1.1, model.shape)
    node = dict(images=images, extent=extent, ks=KS.copy(), k_index=K_INDEX, origin_xy=origin, y_hat=y_hat,
                v_um=V.copy(), profiles_gl=prof, model_gl=model,
                title="node 4725 (synthetic)  |  k* = %d" % KS[K_INDEX],
                text="status    converged\nd       = 1.523 +- 0.012 um\nmu      = 0.251 +- 0.004 /um\n"
                     "v0      = 0.041 +- 0.003 um\nz_ax    = 0.100 +- 0.020 um\nsigma0  = 0.151 +- 0.004 um\n"
                     "c       = 1.31 +- 0.05\nchi2/dof = 1.02  (N 395)\nAllen d = 1.40 um")
    if background:
        node["background_gl"] = bg
    return node


def synthetic_rows(rng=None, n=20):
    """Rows as the entry point would build them from Block 42's SigmaFits (SPEC Block 43): 3 nodes not ok (one a
    failed fit with NaN values), and the summary computed here with numpy over the ok rows (the figure only
    draws it; Block 42's own summarize is tested in its block)."""
    rng = make_rng() if rng is None else rng
    sig = rng.normal(0.15, 0.02, n)
    se = rng.uniform(0.004, 0.02, n)
    c = rng.normal(1.2, 0.2, n)
    se_c = rng.uniform(0.04, 0.2, n)
    d_hat = rng.uniform(1.0, 2.6, n)
    d_allen = d_hat * rng.uniform(0.8, 1.1, n)
    depth = rng.uniform(10.0, 40.0, n)
    if n < 6:
        raise ValueError("synthetic_rows needs n >= 6")
    failed = n - 3                                   # 17 of 20
    ok = np.ones(n, dtype=bool)
    ok[[3, n // 2 + 1, failed]] = False              # 3, 11, 17 of 20
    rows = []
    for i in range(n):
        rows.append(dict(node_id=1000 + 37 * i, sigma0_um=float(sig[i]), se_sigma0_um=float(se[i]),
                         growth_scale=float(c[i]), se_growth_scale=float(se_c[i]), d_hat_um=float(d_hat[i]),
                         d_allen_um=float(d_allen[i]), depth_um=float(depth[i]), ok=bool(ok[i])))
    rows[failed].update(sigma0_um=float("nan"), se_sigma0_um=float("nan"), growth_scale=float("nan"),
                        se_growth_scale=None, d_hat_um=float("nan"))       # a failed fit
    s, e = sig[ok], se[ok]
    w = 1.0 / e ** 2
    mean = float(np.sum(w * s) / np.sum(w))
    med = float(np.median(s))
    summary = dict(n=int(ok.sum()), median_um=med, mad_um=float(np.median(np.abs(s - med))), ivw_mean_um=mean,
                   ivw_se_um=float(1.0 / math.sqrt(np.sum(w))), chi2_cons=float(np.sum((s - mean) ** 2 / e ** 2)),
                   dof=int(ok.sum()) - 1)
    return rows, summary


def _lines(fig, gid):
    return [ln for ax in fig.axes for ln in ax.get_lines() if ln.get_gid() == gid]


def _image_axes(fig):
    return [ax for ax in fig.axes if ax.images]


def _bars(fig, gid):
    """The (n, 2, 2) segments of the error bars whose LineCollection carries gid."""
    out = [np.asarray(col.get_segments()) for ax in fig.axes for col in ax.collections if col.get_gid() == gid]
    assert len(out) == 1, (gid, len(out))
    return out[0]


def _deep(obj):
    return copy.deepcopy(obj)


def _same(a, b):
    """Deep equality of nested dicts / lists / arrays (NaN equal to NaN)."""
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            _same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    elif isinstance(a, np.ndarray):
        assert_array_equal(a, b)
        assert a.dtype == b.dtype
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b)
    else:
        assert a == b, (a, b)


def _sorted_view(rows):
    """The rows in the order the sorted panels draw them: by sigma0, NaN last (stable)."""
    sig = np.array([float("nan") if r["sigma0_um"] is None else float(r["sigma0_um"]) for r in rows])
    return [rows[i] for i in np.argsort(sig, kind="stable")]


def _col(rows, key):
    return np.array([float("nan") if r[key] is None else float(r[key]) for r in rows])


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    node = synthetic_node()
    fig = SF.figure_node_fit(node, zoom_um=4.0)
    n_k = KS.size
    # the profile, model and background lines' data equal the inputs [exact: the arrays are drawn as given]
    for gid, key in (("profile", "profiles_gl"), ("model", "model_gl"), ("background", "background_gl")):
        lines = _lines(fig, gid)
        assert len(lines) == n_k, (gid, len(lines))
        for i, ln in enumerate(lines):
            assert_array_equal(ln.get_xdata(), V)
            assert_array_equal(ln.get_ydata(), node[key][i])
    # the residual is profile - model [exact: the same float64 subtraction]
    res = _lines(fig, "residual")
    assert len(res) == n_k
    for i, ln in enumerate(res):
        assert_array_equal(ln.get_ydata(), node["profiles_gl"][i] - node["model_gl"][i])
    # one image per plane, on one grey range
    imgs = _image_axes(fig)
    assert len(imgs) == n_k, len(imgs)
    assert len({im.get_clim() for ax in imgs for im in ax.images}) == 1
    for ax, plane in zip(imgs, node["images"]):
        assert_array_equal(np.asarray(ax.images[0].get_array()), plane)
    # the profile panels share one y range and the residual panels another, with the zero line in each
    p_axes = [ln.axes for ln in _lines(fig, "profile")]
    r_axes = [ln.axes for ln in res]
    assert len({ax.get_ylim() for ax in p_axes}) == 1 and len({ax.get_ylim() for ax in r_axes}) == 1
    assert p_axes[0].get_ylim() != r_axes[0].get_ylim()
    zeros = _lines(fig, "zero")
    assert len(zeros) == n_k and {ln.axes for ln in zeros} == set(r_axes)
    assert all(np.all(np.asarray(ln.get_ydata(), dtype=float) == 0.0) for ln in zeros)
    # the sharpest plane framed (orange, thicker), the others not
    for i, ax in enumerate(imgs):
        sp = ax.spines["left"]
        if i == K_INDEX:
            assert sp.get_edgecolor() == to_rgba(ORANGE) and sp.get_linewidth() == 2.0
        else:
            assert sp.get_edgecolor() != to_rgba(ORANGE) and sp.get_linewidth() < 1.0
        assert ("k*" in ax.get_title()) and (("(k*)" in ax.get_title()) == (i == K_INDEX))
    # the measuring line from the first to the last offset along y_hat, about the origin; the view +-zoom
    for ln, ax in zip(_lines(fig, "line"), imgs):
        check_close("line_x", ln.get_xdata(), [V[0] * node["y_hat"][0], V[-1] * node["y_hat"][0]], rtol=0, atol=1e-15)
        check_close("line_y", ln.get_ydata(), [V[0] * node["y_hat"][1], V[-1] * node["y_hat"][1]], rtol=0, atol=1e-15)
        assert ax.get_xlim() == (-4.0, 4.0) and ax.get_ylim() == (4.0, -4.0)    # y downwards, as in the image
    texts = [t for ax in fig.axes for t in ax.texts if t.get_gid() == "text"]
    assert len(texts) == 1 and texts[0].get_text() == node["text"]
    assert fig.get_suptitle() == node["title"]
    _close(fig)

    rows, summary = synthetic_rows()
    fig = SF.figure_summary(rows, summary)
    srt = _sorted_view(rows)
    # sigma0 +- SE per node sorted by value; c +- SE per node in the same order: the bars equal the SEs
    for gid, key, se_key in (("se|sigma0", "sigma0_um", "se_sigma0_um"), ("se|growth_scale", "growth_scale",
                                                                          "se_growth_scale")):
        y, se = _col(srt, key), _col(srt, se_key)
        has = np.isfinite(y) & np.isfinite(se)
        seg = _bars(fig, gid)
        assert seg.shape == (int(has.sum()), 2, 2)
        # matplotlib forms y -+ se itself (one float64 addition each): equal to roundoff
        check_close(gid + "_x", seg[:, :, 0], np.column_stack([np.flatnonzero(has)] * 2), rtol=0, atol=0)
        check_close(gid + "_lo", seg[:, 0, 1], y[has] - se[has], rtol=1e-15, atol=0)
        check_close(gid + "_hi", seg[:, 1, 1], y[has] + se[has], rtol=1e-15, atol=0)
    # the bars of the two panels against d_hat and depth (rows in their own order) equal the SEs too
    for gid, xkey in (("se|sigma0_vs_d", "d_hat_um"), ("se|sigma0_vs_depth", "depth_um")):
        xx, y, se = _col(rows, xkey), _col(rows, "sigma0_um"), _col(rows, "se_sigma0_um")
        has = np.isfinite(xx) & np.isfinite(y) & np.isfinite(se)
        seg = _bars(fig, gid)
        check_close(gid + "_x", seg[:, :, 0], np.column_stack([xx[has]] * 2), rtol=0, atol=0)
        check_close(gid + "_lo", seg[:, 0, 1], y[has] - se[has], rtol=1e-15, atol=0)
        check_close(gid + "_hi", seg[:, 1, 1], y[has] + se[has], rtol=1e-15, atol=0)
    # the points of the sorted panel ascend; the mean line at ivw_mean_um, the band +- ivw_se_um, the median
    a_s = _lines(fig, "ivw_mean")[0].axes
    pts = sorted([(x, y) for ln in a_s.get_lines() if (ln.get_gid() or "").startswith("sigma0|")
                  for x, y in zip(ln.get_xdata(), ln.get_ydata()) if np.isfinite(y)])
    ys = [y for _, y in pts]
    assert ys == sorted(ys) and len(ys) == sum(np.isfinite(_col(rows, "sigma0_um")))
    assert [x for x, _ in pts] == list(range(len(pts)))          # the finite values first, at 0, 1, ...
    assert len(_lines(fig, "ivw_mean")) == 1
    assert_array_equal(np.asarray(_lines(fig, "ivw_mean")[0].get_ydata(), dtype=float), [summary["ivw_mean_um"]] * 2)
    assert_array_equal(np.asarray(_lines(fig, "median")[0].get_ydata(), dtype=float), [summary["median_um"]] * 2)
    band = [p for p in a_s.patches if p.get_gid() == "ivw_band"]
    assert len(band) == 1
    # axhspan's patch (a Rectangle in matplotlib >= 3.9, a Polygon before) in its own (x axes, y data) frame;
    # a Rectangle stores y and height, so its top is y + (top - y): equal to a few ulps
    by = band[0].get_patch_transform().transform(band[0].get_path().vertices)[:, 1]
    check_close("band", [by.min(), by.max()], [summary["ivw_mean_um"] - summary["ivw_se_um"],
                                               summary["ivw_mean_um"] + summary["ivw_se_um"]], rtol=1e-14)
    # not-ok nodes hollow, ok nodes filled, in every panel; each set's data are its rows'
    ok = np.array([r["ok"] for r in rows])
    for name, xkey in (("sigma0", None), ("growth_scale", None), ("sigma0_vs_d", "d_hat_um"),
                       ("sigma0_vs_depth", "depth_um")):
        for tag, want in (("ok", True), ("not_ok", False)):
            lns = _lines(fig, "%s|%s" % (name, tag))
            assert len(lns) == 1, (name, tag)
            face = lns[0].get_markerfacecolor()
            assert (face == "none") == (not want), (name, tag, face)
            if xkey is not None:
                sel = [r for r, o in zip(rows, ok) if o == want]
                assert_array_equal(lns[0].get_xdata(), _col(sel, xkey))
                assert_array_equal(lns[0].get_ydata(), _col(sel, "sigma0_um"))
    # Allen's diameter marked: a cross at (d_allen, sigma0) for every node
    allen = _lines(fig, "allen")
    assert len(allen) == 1 and allen[0].get_marker() == "x"
    assert_array_equal(allen[0].get_xdata(), _col(rows, "d_allen_um"))
    assert_array_equal(allen[0].get_ydata(), _col(rows, "sigma0_um"))
    # the node ids below the sorted panels, in the sorted order; "n/a" at the foot where a value is missing
    labels = [t.get_text() for t in a_s.get_xticklabels()]
    assert labels == [str(r["node_id"]) for r in srt], labels
    a_c = _lines(fig, "c_one")[0].axes
    for ax, key in ((a_s, "sigma0_um"), (a_c, "growth_scale")):
        missing = [pos for pos, r in enumerate(srt) if not np.isfinite(_col([r], key)[0])]
        marks = [t.get_position()[0] for t in ax.texts if t.get_gid() == "no_value"]
        assert marks == missing and len(missing) == 1, (key, marks, missing)
    _close(fig)


def test_reference():
    # the grey range: brute force over the pixels (each pixel's edges from the extent, kept when it overlaps the
    # view), then numpy's percentiles of every plane's kept pixels together -- the rule of Block 41 that the
    # page follows
    node = synthetic_node()
    zoom = 3.0
    fig = SF.figure_node_fit(node, zoom_um=zoom)
    clim = _image_axes(fig)[0].images[0].get_clim()
    _close(fig)
    ext = node["extent"] - np.array([node["origin_xy"][0]] * 2 + [node["origin_xy"][1]] * 2)
    n_k, H, W = node["images"].shape
    px = (ext[1] - ext[0]) / W
    py = (ext[2] - ext[3]) / H
    kept = []
    for r in range(H):
        y0, y1 = ext[3] + r * py, ext[3] + (r + 1) * py
        for c in range(W):
            x0, x1 = ext[0] + c * px, ext[0] + (c + 1) * px
            if x1 > -zoom and x0 < zoom and y1 > -zoom and y0 < zoom:
                kept.append((r, c))
    rr, cc = np.array(kept).T
    vals = node["images"][:, rr, cc].ravel()
    expect = np.percentile(vals, IMAGE_PERCENTILES)
    # the pixel edges come from linspace in the module and from products here: equal to a few ulps
    check_close("grey_range", clim, expect, rtol=1e-12)
    assert 0 < len(kept) < H * W                                  # the view is inside the patch

    # the error bars: a loop over the rows in the sorted order, against the drawn segments
    rows, summary = synthetic_rows(make_rng(SEED + 1), n=9)
    fig = SF.figure_summary(rows, summary)
    seg = _bars(fig, "se|sigma0")
    k = 0
    for pos, r in enumerate(_sorted_view(rows)):
        y, se = r["sigma0_um"], r["se_sigma0_um"]
        if se is None or not (math.isfinite(y) and math.isfinite(se)):
            continue
        assert seg[k][0][0] == pos and seg[k][1][0] == pos
        assert abs(seg[k][1][1] - seg[k][0][1] - 2.0 * se) <= 4e-16 * max(1.0, abs(y)), (pos, seg[k], se)
        k += 1
    assert k == len(seg)
    _close(fig)


def test_convergence():
    raise unittest.SkipTest("plotting only: no discretisation")


def test_invariants():
    # a permutation of the rows draws the same sorted panels (the order is by sigma0's value)
    rows, summary = synthetic_rows()
    perm = [rows[i] for i in make_rng(7).permutation(len(rows))]
    figs = [SF.figure_summary(rows, summary), SF.figure_summary(perm, summary)]
    for gid in ("se|sigma0", "se|growth_scale"):
        assert_array_equal(_bars(figs[0], gid), _bars(figs[1], gid))
    for name in ("sigma0|ok", "sigma0|not_ok", "growth_scale|ok", "growth_scale|not_ok"):
        a, b = _lines(figs[0], name)[0], _lines(figs[1], name)[0]
        assert_array_equal(a.get_xdata(), b.get_xdata())
        assert_array_equal(a.get_ydata(), b.get_ydata())
    for fig in figs:
        _close(fig)

    # pixels outside the view do not change the grey range; pixels inside do
    node = synthetic_node()
    fig = SF.figure_node_fit(node, zoom_um=3.0)
    clim = _image_axes(fig)[0].images[0].get_clim()
    _close(fig)
    far = dict(node, images=node["images"].copy())
    far["images"][:, :5, :5] = 0.0                 # the corners lie 5.5 um from the origin, outside +-3 um
    far["images"][:, -5:, -5:] = 255.0
    fig = SF.figure_node_fit(far, zoom_um=3.0)
    assert _image_axes(fig)[0].images[0].get_clim() == clim
    _close(fig)
    fig = SF.figure_node_fit(far, zoom_um=None)     # the whole patch sees them
    assert _image_axes(fig)[0].images[0].get_clim() != clim
    _close(fig)

    # every y range contains its data; the residual's is symmetric about 0
    node = synthetic_node()
    node["profiles_gl"][1, 3] = 300.0              # an outlier far above the rest
    fig = SF.figure_node_fit(node)
    lo, hi = _lines(fig, "profile")[0].axes.get_ylim()
    for key in ("profiles_gl", "model_gl", "background_gl"):
        assert lo <= np.min(node[key]) and np.max(node[key]) <= hi, key
    rlo, rhi = _lines(fig, "residual")[0].axes.get_ylim()
    r = node["profiles_gl"] - node["model_gl"]
    assert rlo == -rhi and rlo <= r.min() and r.max() <= rhi
    _close(fig)


def test_contract():
    from matplotlib.figure import Figure
    node = synthetic_node()
    fig = SF.figure_node_fit(node)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3 * KS.size + 1                       # three rows of panels and the text panel
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out = SF.save(fig, ARTIFACT_DIR / "sigma_node_fit_example.png", dpi=110)
    assert Path(out).stat().st_size > 20000
    _close(fig)
    rows, summary = synthetic_rows()
    fig = SF.figure_summary(rows, summary)
    assert isinstance(fig, Figure) and len(fig.axes) == 4
    out = SF.save(fig, ARTIFACT_DIR / "sigma_summary_example.png", dpi=110)
    assert Path(out).stat().st_size > 20000
    _close(fig)
    # the optional keys left out (no background, no title, no text) and the whole patch
    bare = {k: node[k] for k in SF.NODE_REQUIRED}
    fig = SF.figure_node_fit(bare, zoom_um=None)
    assert not _lines(fig, "background") and len(_lines(fig, "profile")) == KS.size
    _close(fig)
    # NaN samples (a line leaving the patch) and an all-NaN model (a failed fit): gaps, finite ranges
    gaps = synthetic_node()
    gaps["profiles_gl"][:, :6] = np.nan
    gaps["model_gl"][:] = np.nan
    fig = SF.figure_node_fit(gaps)
    for gid in ("profile", "residual"):
        assert all(np.all(np.isfinite(ln.axes.get_ylim())) for ln in _lines(fig, gid))
    out = SF.save(fig, ARTIFACT_DIR / "sigma_node_fit_failed_example.png", dpi=80)
    _close(fig)
    # no node ok: the summary of NaN draws no mean, band or median, and the points still appear hollow
    none_ok = [dict(r, ok=False) for r in rows]
    nan_summary = dict(n=0, median_um=float("nan"), mad_um=float("nan"), ivw_mean_um=float("nan"),
                       ivw_se_um=float("nan"), chi2_cons=float("nan"), dof=float("nan"))
    fig = SF.figure_summary(none_ok, nan_summary)
    assert not _lines(fig, "ivw_mean") and not _lines(fig, "median")
    assert not [p for ax in fig.axes for p in ax.patches if p.get_gid() == "ivw_band"]
    assert not _lines(fig, "sigma0|ok") and len(_lines(fig, "sigma0|not_ok")) == 1
    fig.canvas.draw()
    _close(fig)
    # a single row
    fig = SF.figure_summary(rows[:1], dict(summary, n=1, chi2_cons=float("nan"), dof=0))
    assert len(fig.axes) == 4
    _close(fig)


def test_determinism():
    pix = []
    for _ in range(2):
        fig = SF.figure_node_fit(synthetic_node())
        fig.canvas.draw()
        pix.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        _close(fig)
    assert_array_equal(pix[0], pix[1])
    pix = []
    for _ in range(2):
        fig = SF.figure_summary(*synthetic_rows())
        fig.canvas.draw()
        pix.append(np.asarray(fig.canvas.buffer_rgba()).copy())
        _close(fig)
    assert_array_equal(pix[0], pix[1])


def test_edge_cases():
    node = synthetic_node()
    keep = _deep(node)
    # inputs unchanged (a uint8 patch as the real blocks are, and float64 arrays that numpy would not copy)
    node8 = dict(node, images=np.clip(np.rint(node["images"]), 0, 255).astype(np.uint8))
    keep8 = _deep(node8)
    _close(SF.figure_node_fit(node))
    _close(SF.figure_node_fit(node8))
    _same(node, keep)
    _same(node8, keep8)
    rows, summary = synthetic_rows()
    keep_rows, keep_summary = _deep(rows), _deep(summary)
    _close(SF.figure_summary(rows, summary))
    _same(rows, keep_rows)
    _same(summary, keep_summary)

    # a node that is not a mapping or lacks a key
    _raises(ValueError, SF.figure_node_fit, [node])
    _raises(ValueError, SF.figure_node_fit, None)
    for key in SF.NODE_REQUIRED:
        _raises(ValueError, SF.figure_node_fit, {k: v for k, v in node.items() if k != key})
    # shapes that disagree
    n_v = V.size
    for bad in (dict(profiles_gl=node["profiles_gl"][:, :-1]), dict(model_gl=node["model_gl"][:-1]),
                dict(background_gl=node["background_gl"].T), dict(images=node["images"][:-1]),
                dict(images=node["images"][0]), dict(ks=KS[:-1]), dict(ks=KS.reshape(1, -1)),
                dict(v_um=V[:-1]), dict(v_um=np.tile(V, (2, 1))), dict(extent=node["extent"][:3]),
                dict(origin_xy=np.zeros(3)), dict(y_hat=np.zeros((2, 1))),
                dict(profiles_gl=np.zeros((KS.size, n_v, 1)))):
        _raises(ValueError, SF.figure_node_fit, dict(node, **bad))
    # the guards beside them: non-finite geometry or offsets, k_index not an index, zoom_um not > 0
    for bad in (dict(extent=np.array([0.0, 1.0, np.nan, 0.0])), dict(origin_xy=np.array([np.inf, 0.0])),
                dict(v_um=np.where(np.arange(n_v) == 3, np.nan, V)), dict(k_index=KS.size), dict(k_index=-1),
                dict(k_index=1.5), dict(k_index="2"), dict(k_index=None)):
        _raises(ValueError, SF.figure_node_fit, dict(node, **bad))
    for zoom in (0.0, -1.0, float("nan"), "wide"):
        _raises(ValueError, SF.figure_node_fit, node, zoom_um=zoom)
    _close(SF.figure_node_fit(dict(node, k_index=np.int64(4), background_gl=None)))   # numpy int, None = absent

    # an empty rows, rows not a sequence of mappings, a row lacking a key
    _raises(ValueError, SF.figure_summary, [], summary)
    _raises(ValueError, SF.figure_summary, (), summary)
    _raises(ValueError, SF.figure_summary, rows[0], summary)                 # one mapping, not a sequence
    _raises(ValueError, SF.figure_summary, "rows", summary)
    _raises(ValueError, SF.figure_summary, [rows[0], [1, 2]], summary)
    for key in SF.ROW_KEYS:
        _raises(ValueError, SF.figure_summary, [rows[0], {k: v for k, v in rows[1].items() if k != key}], summary)
    # the guards beside them: ok not a bool, a value not a number, a negative SE, a bad summary
    _raises(ValueError, SF.figure_summary, [dict(rows[0], ok="yes")], summary)
    _raises(ValueError, SF.figure_summary, [dict(rows[0], ok=1)], summary)
    _raises(ValueError, SF.figure_summary, [dict(rows[0], sigma0_um="0.1")], summary)
    _raises(ValueError, SF.figure_summary, [dict(rows[0], se_sigma0_um=-0.01)], summary)
    _raises(ValueError, SF.figure_summary, [dict(rows[0], se_growth_scale=-1.0)], summary)
    _raises(ValueError, SF.figure_summary, rows, [summary])
    for key in SF.SUMMARY_KEYS:
        _raises(ValueError, SF.figure_summary, rows, {k: v for k, v in summary.items() if k != key})
    _close(SF.figure_summary([dict(rows[0], ok=np.bool_(True))], summary))     # numpy's bool is a bool
    _same(rows, keep_rows)


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
