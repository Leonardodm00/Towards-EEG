"""Smoke test for the soma fit's figures (plotting/soma_fit_figures.py) -- Block 48 in specs/SPEC.md (D-215, D-222.5).

Oracles of the entry: (T1) each figure returns a Figure with the expected number of axes on synthetic inputs and is
saved to a non-empty PNG (test_contract); (T2) no input mutated, (T3) sections that are None and an empty stem list
drawn without error, (T5) the ValueErrors for shapes that disagree (test_edge_cases); (T4) the residual panels share
one colour scale, symmetric about 0 (test_invariants). Beside them: known answers -- every array drawn is its input
(the residual the same float64 subtraction), the circles' centres and radii, the line styles, the rows' k and depth,
the tables' cells, the stems' markers and arrows, the flags in the legend (test_known_answer) -- and independent
references for the image frame (the region's contour about one marked pixel lies on ((left + col) p, (top + row) p);
a disk's contour within half a pixel of its circle) and for the residual's scale (its percentile by a loop over the
fitted pixels) (test_reference).

The inputs are synthetic, built here from the shapes SPEC Blocks 46-48 state (Block 47 is written in parallel, so its
outlines and stems are made here, not imported); the model planes are a smooth shadow, not Block 45's renderer: the
figures only draw them.

Checks, from the strongest oracle to the weakest
    test_known_answer   analytic or tabulated solution
    test_reference      independent implementation (library function, or
                        brute force on a tiny input)
    test_convergence    discretisations only: observed order p in
                        e(h) ~ C * h**p, where e(h) is the error at step h
    test_invariants     conservation laws, symmetries, bounds, normalisation
    test_contract       output shapes, dtypes and finiteness
    test_determinism    identical output for an identical seed
    test_edge_cases     degenerate or boundary inputs named in the spec

Run
    python tests/smoke/test_smoke_soma_fit_figures.py
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO. The example figures are written to
        tests/smoke/artifacts/soma_fit_*_example.png (not committed).
    python -m pytest tests/smoke/test_smoke_soma_fit_figures.py -q
        The same checks through pytest (TODO checks show up as failures).

Debug
    A failed check_close() saves both arrays to
        tests/smoke/artifacts/<check name>.npz
    Inspect it with
        d = np.load("tests/smoke/artifacts/<check name>.npz")
        diff = d["actual"] - d["expected"]
        np.unravel_index(np.argmax(np.abs(diff)), diff.shape)

Add a check
    Define a function test_<something>() with no arguments that raises
    AssertionError on failure; the runner below and pytest both pick it up.

Keep this file pure ASCII (no Greek letters, dashes or arrows outside ASCII)
so it survives transfer to the cluster.
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

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from allen_diameter.analysis.soma_fit import PARAMS  # noqa: E402  (Block 46, merged: the names the summary shows)
from allen_diameter.plotting import soma_fit_figures as SFF  # noqa: E402
from allen_diameter.plotting.cleaning_figures import IMAGE_PERCENTILES  # noqa: E402
from allen_diameter.plotting.figures import _DIV, image_extent_um  # noqa: E402

SEED = 20261122
REPORT_PACKAGES = ("numpy", "scipy", "matplotlib")
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
_SAVED_ARTIFACTS = []  # files written by check_close during this run

P_UM, DZ_UM = 0.1144, 0.28                    # the pixel pitch and the plane spacing (SPEC 2.1)
LEFT, TOP, H, W = 3000, 5000, 97, 121          # the crop: 97 rows x 121 columns, not square, so an x/y swap shows
KS = np.array([97, 105, 111, 117, 123, 137])   # the planes shown; the first and the last miss the fitted soma
# the fitted ellipsoid theta_g = (x0, y0, z0, a, b, c, phi, mu) (Block 45), the far slope and r_eq (Block 46)
X0, Y0, Z0 = (LEFT + 63) * P_UM + 0.031, (TOP + 47) * P_UM - 0.047, 117 * DZ_UM + 0.12
A, B, C, PHI, MU, GAMMA = 4.4, 3.2, 3.9, 0.62, 0.21, 0.93
R_EQ = 3.83
XS, YS, RS = X0 - 0.42, Y0 + 0.33, 3.5          # Allen's traced soma: its centre and radius
REGION_UM = 5.0                                 # the region's radius about the traced centre (fits in the crop)
NOISE_GL = 2.0


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


def _ellipse(cx, cy, a, b, phi, n=256):
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.vstack([cx + a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi),
                      cy + a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)])


def _centres():
    rows, cols = np.mgrid[0:H, 0:W]
    return (LEFT + cols) * P_UM, (TOP + rows) * P_UM          # pixel centres in the image frame (um)


def _segment_distance(x, y, p1, p2):
    d = np.subtract(p2, p1)
    t = np.clip(((x - p1[0]) * d[0] + (y - p1[1]) * d[1]) / float(d @ d), 0.0, 1.0)
    return np.hypot(x - p1[0] - t * d[0], y - p1[1] - t * d[1])


def planes_case(rng=None, uint8=False):
    """The planes figure's inputs as the entry point would build them (SPEC Block 48): a dark elliptical shadow whose
    depth falls and whose edge widens away from z0, a masked stem leaving it, a neighbour outside the region, noise."""
    rng = make_rng() if rng is None else rng
    frame = type("F", (), dict(left=LEFT, top=TOP, res_um_px=P_UM, downsample=0))()
    extent = np.asarray(image_extent_um(frame, (KS.size, H, W)), dtype=float)
    x, y = _centres()
    u = (x - X0) * math.cos(PHI) + (y - Y0) * math.sin(PHI)
    v = -(x - X0) * math.sin(PHI) + (y - Y0) * math.cos(PHI)
    rho = np.hypot(u / A, v / B)
    dz = KS * DZ_UM - Z0
    scale = 141.0 - 0.4 * np.arange(KS.size)
    model = np.stack([scale[j] * (1.0 - 0.55 * math.exp(-(dz[j] / C) ** 2)
                                  / (1.0 + np.exp((rho - 1.0) / (0.06 + 0.07 * abs(dz[j])))))
                      for j in range(KS.size)])
    stem = _segment_distance(x, y, (X0 + 2.0, Y0 + 0.8), (X0 + 6.5, Y0 + 3.4))
    neighbour = _segment_distance(x, y, (X0 - 6.8, Y0 - 5.6), (X0 - 2.0, Y0 - 5.2))
    data = model - 35.0 * (stem < 0.45) - 25.0 * (neighbour < 0.35) + rng.normal(0.0, NOISE_GL, model.shape)
    if uint8:
        data = np.clip(np.rint(data), 0, 255).astype(np.uint8)
    region = np.hypot(x - XS, y - YS) <= REGION_UM
    mask = stem < 0.45 + 0.6                                   # Allen's stem widened (Block 46's mask_widen)
    sil = _ellipse(X0, Y0, A, B, PHI)
    sections = [None if abs(d) >= C else _ellipse(X0, Y0, A * math.sqrt(1 - (d / C) ** 2),
                                                  B * math.sqrt(1 - (d / C) ** 2), PHI) for d in dz]
    return dict(data=data, model=model, region=region, mask=mask, ks=KS.copy(), dz_from_centre_um=dz,
                extent_um=extent, silhouette_xy=sil, sections_xy=sections, allen_circle=np.array([XS, YS, RS]),
                eq_circle=np.array([X0, Y0, R_EQ]), title="specimen 0 (synthetic)  |  the soma fit's planes")


def summary_case(held=False):
    """The summary's inputs (SPEC Blocks 46, 48): PARAMS, theta, se, fitted; the derived dict with the SomaFit's own
    field names (area_um2, se_area_um2, volume_um3, r_eq_um), Allen's radius and 4 pi r^2; 17 planes 0.56 um apart."""
    theta = np.array([X0, Y0, Z0, A, B, C, PHI, MU, GAMMA])
    se = np.array([0.012, 0.011, 0.034, 0.021, 0.018, 0.052, 0.0061, 0.0013, 0.017])
    fitted = PARAMS
    if held:                                                  # far_slope "fixed": gamma held, its SE NaN
        se[8] = np.nan
        fitted = PARAMS[:8]
    ks = 117 + 2 * np.arange(-8, 9)
    z = ks * DZ_UM
    return dict(names=PARAMS, theta=theta, se=se, fitted=fitted,
                derived=dict(area_um2=184.1, se_area_um2=1.94, volume_um3=230.0, r_eq_um=R_EQ),
                allen=dict(radius_um=RS, area_um2=4.0 * math.pi * RS ** 2),
                z_um=z, B_gl=141.0 - 0.05 * (ks - 117) + 0.3 * np.cos(ks),
                rms_gl=2.0 + 1.3 * np.exp(-((z - Z0) / 2.0) ** 2),
                noise_gl=NOISE_GL, title="specimen 0 (synthetic)  |  status converged, 64 evaluations")


STEM_SPECS = (  # (direction angle, |old - centre|, |new - centre|, flag, number of next nodes)
    (0.30, 5.6, 4.1, "moved", 3), (2.20, 2.2, 3.4, "moved", 3), (3.60, 3.5, 3.5, "on surface", 3),
    (4.40, 6.0, 6.0, "missed", 2), (5.40, 1.4, 1.4, "branch inside", 3), (1.20, 1.9, 1.9, "ends inside", 0))


def morphology_case():
    """The morphology figure's inputs (SPEC Block 48, from Block 47's outlines and SomaEdit's stems): both views, all
    five flags, a leaf stem with no next nodes."""
    def stems(cx, cy, sx, sy):
        out = []
        for ang, r_old, r_new, flag, m in STEM_SPECS:
            e = np.array([math.cos(ang) * sx, math.sin(ang) * sy])
            nxt = np.array([[cx + (r_old + 1.2 * (i + 1)) * e[0] for i in range(m)],
                            [cy + (r_old + 1.2 * (i + 1)) * e[1] for i in range(m)]]).reshape(2, m)
            out.append(dict(old=np.array([cx, cy]) + r_old * e, new=np.array([cx, cy]) + r_new * e, next=nxt,
                            flag=flag))
        return out
    top = dict(allen=_ellipse(XS, YS, RS, RS, 0.0, 181), silhouette=_ellipse(X0, Y0, A, B, PHI),
               sphere=_ellipse(X0, Y0, R_EQ, R_EQ, 0.0, 181), stems=stems(X0, Y0, 1.0, 1.0))
    side = dict(allen=_ellipse(-0.35, 0.4, RS, RS, 0.0, 181), silhouette=_ellipse(0.0, 0.0, A, C, 0.0),
                sphere=_ellipse(0.0, 0.0, R_EQ, R_EQ, 0.0, 181), stems=stems(0.0, 0.0, 1.0, 0.8))
    return dict(top=top, side=side, title="specimen 0 (synthetic)  |  the soma in the morphology")


def _lines(fig, gid):
    return [ln for ax in fig.axes for ln in ax.get_lines() if ln.get_gid() == gid]


def _images(fig, gid):
    return [im for ax in fig.axes for im in ax.images if im.get_gid() == gid]


def _patches(fig, gid):
    return [p for ax in fig.axes for p in ax.patches if p.get_gid() == gid]


def _contours(fig, gid):
    return [c for ax in fig.axes for c in ax.collections if c.get_gid() == gid]


def _table(fig, gid):
    out = [t for ax in fig.axes for t in ax.tables if t.get_gid() == gid]
    assert len(out) == 1, (gid, len(out))
    return out[0]


def _cell(tab, r, c):
    return tab.get_celld()[(r, c)].get_text().get_text()


def _arr(im):
    return np.ma.getdata(im.get_array())


def _vertices(cs):
    """The contour's vertices, without the repeated vertex of a CLOSEPOLY code."""
    out = []
    for path in cs.get_paths():
        v, codes = path.vertices, path.codes
        out.append(v if codes is None else v[codes != MplPath.CLOSEPOLY])
    return np.concatenate(out) if out else np.zeros((0, 2))


def _same(a, b):
    """Deep equality of nested dicts / lists / arrays (NaN equal to NaN, dtypes equal)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b)
        for k in a:
            _same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            _same(x, y)
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype
        assert_array_equal(a, b)
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b)
    else:
        assert a == b, (a, b)


# ---------------------------------------------------------------- checks ---

def test_known_answer():
    case = planes_case()
    K = KS.size
    fig = SFF.figure_soma_fit_planes(**case)
    D = case["data"].astype(float)
    data, model, resid = _images(fig, "data"), _images(fig, "model"), _images(fig, "residual")
    assert len(data) == len(model) == len(resid) == K
    for j in range(K):
        # the arrays drawn are the inputs, the residual the same float64 subtraction [exact]
        assert_array_equal(_arr(data[j]), D[j])
        assert_array_equal(_arr(model[j]), case["model"][j])
        assert_array_equal(_arr(resid[j]), D[j] - case["model"][j])
        # one row: data, model, residual, left to right, on the row's axes
        x_left = [im.axes.get_position().x0 for im in (data[j], model[j], resid[j])]
        assert x_left == sorted(x_left), x_left
        # data and model on one grey range: numpy's 1st and 99th percentiles of the two together
        lo, hi = np.percentile(np.concatenate([D[j].ravel(), case["model"][j].ravel()]), IMAGE_PERCENTILES)
        assert data[j].get_clim() == model[j].get_clim() == (lo, hi), (data[j].get_clim(), lo, hi)
        assert data[j].get_cmap().name == "gray"
    # every image (and overlay) on the crop's extent, origin upper; the view is the crop (y downwards)
    for im in data + model + resid + _images(fig, "masked"):
        assert tuple(im.get_extent()) == tuple(case["extent_um"]) and im.origin == "upper"
        assert im.axes.get_xlim() == tuple(case["extent_um"][:2])
        assert im.axes.get_ylim() == tuple(case["extent_um"][2:])
        assert im.axes.get_ylim()[0] > im.axes.get_ylim()[1]
    data_axes = [im.axes for im in data]
    # the silhouette (dashed) on every data panel, the section (solid) where the plane meets the soma, only there
    sil = _lines(fig, "silhouette")
    assert [ln.axes for ln in sil] == data_axes
    for ln in sil:
        assert_array_equal(ln.get_xdata(), case["silhouette_xy"][0])
        assert_array_equal(ln.get_ydata(), case["silhouette_xy"][1])
        assert ln.get_linestyle() == "--"
    sec = _lines(fig, "section")
    has = [s is not None for s in case["sections_xy"]]
    assert sum(has) == K - 2 and [ln.axes for ln in sec] == [ax for ax, h in zip(data_axes, has) if h]
    for ln, s in zip(sec, [s for s in case["sections_xy"] if s is not None]):
        assert_array_equal(ln.get_xdata(), s[0])
        assert_array_equal(ln.get_ydata(), s[1])
        assert ln.get_linestyle() == "-"
    # Allen's circle dotted, the sphere of equal area dash-dot, with the centres and radii passed [exact]
    for gid, circ, ls in (("allen", case["allen_circle"], ":"), ("sphere", case["eq_circle"], "-.")):
        ps = _patches(fig, gid)
        assert [p.axes for p in ps] == data_axes, gid
        for p in ps:
            assert tuple(p.center) == (circ[0], circ[1]) and p.radius == circ[2], (gid, p.center, p.radius)
            assert p.get_linestyle() == ls and not p.get_fill()
    # the outlines on the data panels only
    others = set(fig.axes) - set(data_axes)
    assert not any(ln.get_gid() in ("silhouette", "section") for ax in others for ln in ax.get_lines())
    assert not any(p.get_gid() in ("allen", "sphere") for ax in others for p in ax.patches)
    # the region's boundary on every panel; the masked pixels dimmed on every panel, exactly where the mask is
    image_axes = set(data_axes) | {im.axes for im in model + resid}
    assert {c.axes for c in _contours(fig, "region")} == image_axes and len(_contours(fig, "region")) == 3 * K
    dims = _images(fig, "masked")
    assert {im.axes for im in dims} == image_axes and len(dims) == 3 * K
    for im in dims:
        alpha = _arr(im)[..., 3]
        assert_array_equal(alpha > 0, case["mask"])
        check_close("dim_alpha", alpha[case["mask"]], SFF.MASK_DIM[1], rtol=0, atol=1e-15)
    # each row titled with k and its depth (z - z0); the misses named
    for j, ax in enumerate(data_axes):
        t = ax.get_title()
        assert ("k = %d " % KS[j]) in t and ("%+.2f um" % case["dz_from_centre_um"][j]) in t, t
        assert ("misses" in t) == (not has[j]), t
    _close(fig)

    s = summary_case(held=True)
    fig = SFF.figure_soma_fit_summary(**s)
    tab = _table(fig, "parameters")
    assert [_cell(tab, 0, c) for c in range(3)] == ["parameter", "value", "SE"]
    for i, name in enumerate(PARAMS):
        assert _cell(tab, i + 1, 0) == name
        assert _cell(tab, i + 1, 1) == SFF.VALUE_FMT % s["theta"][i]
        held = name not in s["fitted"]
        assert _cell(tab, i + 1, 2) == ("held" if held else SFF.SE_FMT % s["se"][i]), (name, _cell(tab, i + 1, 2))
        style = tab.get_celld()[(i + 1, 0)].get_text().get_fontstyle()
        assert (style == "italic") == held, (name, style)
    assert sum(name not in s["fitted"] for name in PARAMS) == 1          # gamma held
    der = _table(fig, "derived")
    d, al = s["derived"], s["allen"]
    want = [[SFF.DERIVED_FMT % d["area_um2"], SFF.SE_FMT % d["se_area_um2"], SFF.DERIVED_FMT % al["area_um2"]],
            [SFF.DERIVED_FMT % d["volume_um3"], SFF.NO_VALUE, SFF.NO_VALUE],      # SE and Allen's volume not given
            [SFF.DERIVED_FMT % d["r_eq_um"], SFF.NO_VALUE, SFF.DERIVED_FMT % al["radius_um"]]]
    for r in range(3):
        assert [_cell(der, r + 1, c) for c in (1, 2, 3)] == want[r], (r, [_cell(der, r + 1, c) for c in (1, 2, 3)])
    assert "4 pi r^2" in _cell(der, 1, 0) and "r_eq" in _cell(der, 3, 0)
    for gid, key in (("B", "B_gl"), ("rms", "rms_gl")):
        (ln,) = _lines(fig, gid)
        assert_array_equal(ln.get_xdata(), s["z_um"])
        assert_array_equal(ln.get_ydata(), s[key])
    (noise,) = _lines(fig, "noise")
    assert_array_equal(np.asarray(noise.get_ydata(), dtype=float), [NOISE_GL, NOISE_GL])
    assert noise.axes is _lines(fig, "rms")[0].axes and noise.axes is not _lines(fig, "B")[0].axes
    assert fig.get_suptitle() == s["title"]
    _close(fig)

    m = morphology_case()
    fig = SFF.figure_soma_morphology(**m)
    a_top, a_side = fig.axes
    for ax, view in ((a_top, m["top"]), (a_side, m["side"])):
        for gid in ("allen", "silhouette", "sphere"):
            (ln,) = [x for x in ax.get_lines() if x.get_gid() == gid]
            assert_array_equal(ln.get_xdata(), view[gid][0])
            assert_array_equal(ln.get_ydata(), view[gid][1])
            assert ln.get_linestyle() == SFF.STYLES[gid]
        old = [x for x in ax.get_lines() if x.get_gid() == "old"]
        new = [x for x in ax.get_lines() if x.get_gid() == "new"]
        nxt = [x for x in ax.get_lines() if x.get_gid() == "next"]
        arrows = [t for t in ax.texts if t.get_gid() == "arrow"]
        stems = view["stems"]
        assert len(old) == len(new) == len(stems)
        for ln_o, ln_n, st in zip(old, new, stems):
            assert_array_equal(np.column_stack([ln_o.get_xdata(), ln_o.get_ydata()])[0], st["old"])
            assert_array_equal(np.column_stack([ln_n.get_xdata(), ln_n.get_ydata()])[0], st["new"])
            assert ln_o.get_markerfacecolor() == "none" and ln_n.get_markerfacecolor() != "none"
            colour = SFF.MOVED_COLOUR if st["flag"] == "moved" else SFF.FLAG_COLOURS[st["flag"]]
            assert to_rgba(ln_n.get_markerfacecolor()) == to_rgba(colour), st["flag"]
        with_next = [st for st in stems if st["next"].shape[1]]
        assert len(nxt) == len(with_next) == len(stems) - 1
        for ln, st in zip(nxt, with_next):
            assert_array_equal(ln.get_xdata(), st["next"][0])
            assert_array_equal(ln.get_ydata(), st["next"][1])
        # an arrow from old to new exactly for the stems that moved
        moved = [st for st in stems if np.any(st["old"] != st["new"])]
        assert len(arrows) == len(moved) == 2
        for t, st in zip(arrows, moved):
            assert_array_equal(t.xy, st["new"])
            assert_array_equal(t.xyann, st["old"])
        assert ax.get_aspect() == 1.0
    # the top view in the image's orientation (y downwards), the side view not
    assert a_top.yaxis_inverted() and not a_side.yaxis_inverted()
    # each flag other than "moved" once in the legend, "moved" not as a flag
    labels = [t.get_text() for t in fig.legends[0].get_texts()]
    flags = [lab[len("flag: "):] for lab in labels if lab.startswith("flag: ")]
    assert flags == ["on surface", "missed", "branch inside", "ends inside"], labels
    assert fig.get_suptitle() == m["title"]
    _close(fig)


def test_reference():
    # the image frame: a region of one pixel at (row 13, col 88) is drawn around the pixel centre
    # ((LEFT + 88) p, (TOP + 13) p) -- the convention of image_extent_um, written out here
    case = planes_case()
    r0, c0 = 13, 88
    one = np.zeros((H, W), dtype=bool)
    one[r0, c0] = True
    mask = np.zeros((H, W), dtype=bool)
    mask[r0 + 40, c0 - 70] = True
    fig = SFF.figure_soma_fit_planes(**dict(case, region=one, mask=mask))
    centre = np.array([(LEFT + c0) * P_UM, (TOP + r0) * P_UM])
    for cs in _contours(fig, "region"):
        v = _vertices(cs)
        assert v.shape == (4, 2), v.shape                      # the four edge midpoints about the pixel's centre
        # coordinates of about 600 um carry float64 roundoff near 1e-13 um
        check_close("one_pixel_centre", v.mean(axis=0), centre, rtol=0, atol=1e-9)
        check_close("one_pixel_halfwidth", np.sort(np.abs(v - centre).max(axis=1)), [P_UM / 2] * 4, rtol=0,
                    atol=1e-9)
    for im in _images(fig, "masked"):
        a = _arr(im)[..., 3]
        assert a[r0 + 40, c0 - 70] > 0 and np.count_nonzero(a) == 1
    _close(fig)
    # a disk of pixel centres within R of a point: the boundary between an inside and an outside centre lies within
    # half a pixel of the circle of radius R (the triangle inequality), so every vertex does
    fig = SFF.figure_soma_fit_planes(**case)
    for cs in _contours(fig, "region"):
        dist = np.hypot(*(_vertices(cs) - [XS, YS]).T)
        assert dist.size > 100
        assert np.all(np.abs(dist - REGION_UM) <= P_UM / 2 + 1e-9), (dist.min(), dist.max())
    # the residual's scale: the 99th percentile of |data - model| over the fitted pixels, by a loop
    vals = []
    D, M = case["data"].astype(float), case["model"]
    for j in range(KS.size):
        for r in range(H):
            for c in range(W):
                if case["region"][r, c] and not case["mask"][r, c]:
                    vals.append(abs(D[j, r, c] - M[j, r, c]))
    half = np.percentile(np.array(vals), SFF.RESIDUAL_PERCENTILE)
    norm = _images(fig, "residual")[0].norm
    # the same float64 values in another order: numpy's percentile sorts them, so the result is identical
    assert (norm.vmin, norm.vmax) == (-half, half), (norm.vmin, norm.vmax, half)
    assert ("+-%.3g gl" % half) in fig.get_suptitle()
    _close(fig)


def test_convergence():
    raise unittest.SkipTest("plotting only: no discretisation")


def test_invariants():
    # (T4) the residual panels share one colour scale, symmetric about 0, on a diverging map neutral at 0
    case = planes_case()
    fig = SFF.figure_soma_fit_planes(**case)
    resid = _images(fig, "residual")
    norms = {id(im.norm) for im in resid}
    assert len(norms) == 1, "one Normalize for every residual panel"
    clims = {im.get_clim() for im in resid}
    assert len(clims) == 1
    (lo, hi), = clims
    assert lo == -hi and hi > 0
    cmaps = {im.get_cmap().name for im in resid}
    assert cmaps == {"soma_fit_residual"}
    cm, nm = resid[0].get_cmap(), resid[0].norm
    # LinearSegmentedColormap of 256 entries: the middle colour within one entry's step of the neutral grey
    check_close("neutral_at_0", cm(nm(0.0))[:3], to_rgba(_DIV[1])[:3], rtol=0, atol=0.02)
    check_close("blue_at_-h", cm(nm(lo))[:3], to_rgba(_DIV[0])[:3], rtol=0, atol=1e-12)
    check_close("red_at_+h", cm(nm(hi))[:3], to_rgba(_DIV[2])[:3], rtol=0, atol=1e-12)
    # the colour bar draws that scale
    cbar_axes = [ax for ax in fig.axes if not ax.images and not ax.get_lines()]
    assert len(cbar_axes) == 1
    assert any(getattr(im, "colorbar", None) is not None for im in resid)
    cb = [im.colorbar for im in resid if getattr(im, "colorbar", None) is not None][0]
    assert cb.ax is cbar_axes[0] and cb.norm is nm and cb.mappable is resid[0]
    _close(fig)
    # the scale reads the fitted pixels only: data changed on masked pixels or outside the region leave it; the
    # residual doubled on the fitted pixels doubles it
    D, M = case["data"].astype(float), case["model"]
    fitted = case["region"] & ~case["mask"]
    out = D.copy()
    out[:, ~fitted] += 500.0
    fig = SFF.figure_soma_fit_planes(**dict(case, data=out))
    assert _images(fig, "residual")[0].get_clim() == (lo, hi)
    _close(fig)
    twice = M + 2.0 * (D - M)
    fig = SFF.figure_soma_fit_planes(**dict(case, data=twice))
    # M + 2 (D - M) - M differs from 2 (D - M) by roundoff of 141 gl values: relative 1e-12 covers it
    check_close("scale_doubles", _images(fig, "residual")[0].get_clim()[1], 2.0 * hi, rtol=1e-12)
    _close(fig)
    # the grey range of a row reads that row only: another row's data changed leave it
    other = D.copy()
    other[0] = 0.0
    fig = SFF.figure_soma_fit_planes(**dict(case, data=other))
    ref = SFF.figure_soma_fit_planes(**case)
    for j in range(1, KS.size):
        assert _images(fig, "data")[j].get_clim() == _images(ref, "data")[j].get_clim()
    assert _images(fig, "data")[0].get_clim() != _images(ref, "data")[0].get_clim()
    _close(fig)
    _close(ref)


def test_contract():
    # (T1) a Figure with the expected number of axes, saved to a non-empty PNG
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    case = planes_case()
    fig = SFF.figure_soma_fit_planes(**case)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3 * KS.size + 1, len(fig.axes)           # 3 panels per plane and the colour bar
    for name, f in (("planes", fig), ("summary", SFF.figure_soma_fit_summary(**summary_case(held=True))),
                    ("morphology", SFF.figure_soma_morphology(**morphology_case()))):
        assert isinstance(f, Figure)
        assert len(f.axes) == {"planes": 3 * KS.size + 1, "summary": 4, "morphology": 2}[name], (name, len(f.axes))
        out = Path(SFF.save(f, ARTIFACT_DIR / ("soma_fit_%s_example.png" % name), dpi=90))
        with open(out, "rb") as fh:
            head = fh.read(8)
        assert head == b"\x89PNG\r\n\x1a\n" and out.stat().st_size > 20000, (name, out.stat().st_size)
        _close(f)
    # the residual drawn in float64 whatever the data's dtype (uint8, as Allen's blocks); one plane alone
    case8 = planes_case(uint8=True)
    fig = SFF.figure_soma_fit_planes(**case8)
    for im, j in zip(_images(fig, "residual"), range(KS.size)):
        r = _arr(im)
        assert r.dtype == np.float64
        assert_array_equal(r, case8["data"][j].astype(float) - case8["model"][j])    # no uint8 wrap-around
    _close(fig)
    one = dict(case, data=case["data"][3:4], model=case["model"][3:4], ks=KS[3:4],
               dz_from_centre_um=case["dz_from_centre_um"][3:4], sections_xy=case["sections_xy"][3:4])
    fig = SFF.figure_soma_fit_planes(**one)
    assert len(fig.axes) == 4
    _close(fig)
    # the summary with every parameter fitted and no noise line; the morphology without a title
    s = summary_case()
    fig = SFF.figure_soma_fit_summary(**dict(s, noise_gl=None, title=""))
    assert len(fig.axes) == 4 and not _lines(fig, "noise") and fig.get_suptitle() == ""
    assert all(_cell(_table(fig, "parameters"), i + 1, 2) != "held" for i in range(len(PARAMS)))
    _close(fig)
    fig = SFF.figure_soma_morphology(**dict(morphology_case(), title=""))
    assert len(fig.axes) == 2
    _close(fig)


def test_determinism():
    for build in (lambda: SFF.figure_soma_fit_planes(**planes_case()),
                  lambda: SFF.figure_soma_fit_summary(**summary_case(held=True)),
                  lambda: SFF.figure_soma_morphology(**morphology_case())):
        pix = []
        for _ in range(2):
            fig = build()
            fig.canvas.draw()
            pix.append(np.asarray(fig.canvas.buffer_rgba()).copy())
            _close(fig)
        assert_array_equal(pix[0], pix[1])


def test_edge_cases():
    # (T2) no input mutated: float64 and uint8 data, lists of sections, dicts and lists of stems
    for case in (planes_case(), planes_case(uint8=True)):
        keep = copy.deepcopy(case)
        _close(SFF.figure_soma_fit_planes(**case))
        _same(case, keep)
    for s in (summary_case(), summary_case(held=True)):
        keep = copy.deepcopy(s)
        _close(SFF.figure_soma_fit_summary(**s))
        _same(s, keep)
    m = morphology_case()
    keep = copy.deepcopy(m)
    _close(SFF.figure_soma_morphology(**m))
    _same(m, keep)

    # (T3) sections that are None (every plane, or the whole argument), empty sections, an empty stem list
    case = planes_case()
    for secs in ([None] * KS.size, None, [np.zeros((2, 0))] * KS.size):
        fig = SFF.figure_soma_fit_planes(**dict(case, sections_xy=secs))
        fig.canvas.draw()
        assert not _lines(fig, "section") and len(_lines(fig, "silhouette")) == KS.size
        titles = [im.axes.get_title() for im in _images(fig, "data")]
        assert all(("misses" in t) == (secs is not None) for t in titles), titles
        _close(fig)
    m = morphology_case()
    empty = dict(top=dict(m["top"], stems=[]), side=dict(m["side"], stems=[]))
    fig = SFF.figure_soma_morphology(**empty)
    fig.canvas.draw()
    assert not _lines(fig, "old") and not [t for ax in fig.axes for t in ax.texts if t.get_gid() == "arrow"]
    assert not [t for t in fig.legends[0].get_texts() if t.get_text().startswith("flag: ")]
    _close(fig)
    # a failed fit: a NaN model (the residual NaN, its scale 1 gl), NaN circles and outlines, NaN parameters
    fig = SFF.figure_soma_fit_planes(**dict(case, model=np.full(case["model"].shape, np.nan),
                                            eq_circle=np.array([np.nan] * 3),
                                            silhouette_xy=np.full((2, 256), np.nan),
                                            dz_from_centre_um=np.full(KS.size, np.nan)))
    fig.canvas.draw()
    assert _images(fig, "residual")[0].get_clim() == (-1.0, 1.0)
    _close(fig)
    s = summary_case()
    nan9 = np.full(len(PARAMS), np.nan)
    fig = SFF.figure_soma_fit_summary(**dict(s, theta=nan9, se=nan9, B_gl=np.full(s["z_um"].shape, np.nan),
                                             derived=dict(area_um2=np.nan, volume_um3=np.nan, r_eq_um=np.nan)))
    fig.canvas.draw()
    assert _cell(_table(fig, "parameters"), 1, 1) == "nan"
    _close(fig)
    # a region of every pixel or none, a mask of none: drawn without a boundary or a dimming layer
    for reg in (np.ones((H, W), dtype=bool), np.zeros((H, W), dtype=bool)):
        fig = SFF.figure_soma_fit_planes(**dict(case, region=reg, mask=np.zeros((H, W), dtype=bool)))
        fig.canvas.draw()
        assert not _contours(fig, "region") and not _images(fig, "masked")
        _close(fig)
    # every residual 0 (data = model): the scale falls back to 1 gl
    fig = SFF.figure_soma_fit_planes(**dict(case, data=case["model"].copy()))
    assert _images(fig, "residual")[0].get_clim() == (-1.0, 1.0)
    _close(fig)

    # (T5) the ValueErrors for shapes that disagree -- the planes. The first group would broadcast silently in
    # data - model or region & ~mask (one model plane drawn on every row, a row of the region repeated), so only the
    # module's own checks catch them; the rest would also fail later, inside numpy or matplotlib
    K = KS.size
    dz = case["dz_from_centre_um"]
    for bad in (dict(model=case["model"][:1]), dict(model=case["model"][0]), dict(data=case["data"][:1]),
                dict(region=case["region"][:1]), dict(mask=case["mask"][:, :1]),
                dict(dz_from_centre_um=np.append(dz, 0.0)), dict(ks=np.append(KS, 139)),
                dict(extent_um=np.append(case["extent_um"], 0.0))):
        _raises(ValueError, SFF.figure_soma_fit_planes, **dict(case, **bad))
    bad_planes = (dict(data=case["data"][0]), dict(data=case["data"][:, :, :-1]), dict(model=case["model"][:-1]),
                  dict(model=case["model"][:, :-1]), dict(data=case["data"][:0], model=case["model"][:0]),
                  dict(region=case["region"][:-1]), dict(region=case["region"].T), dict(mask=case["mask"][:, :-1]),
                  dict(ks=KS[:-1]), dict(ks=KS.reshape(1, -1)), dict(dz_from_centre_um=case["dz_from_centre_um"][1:]),
                  dict(extent_um=case["extent_um"][:3]), dict(silhouette_xy=case["silhouette_xy"].T),
                  dict(silhouette_xy=case["silhouette_xy"][0]), dict(sections_xy=case["sections_xy"][:-1]),
                  dict(sections_xy=case["sections_xy"] + [None]),
                  dict(sections_xy=[s if s is None else s.T for s in case["sections_xy"]]),
                  dict(allen_circle=case["allen_circle"][:2]), dict(eq_circle=np.zeros((3, 1))))
    for bad in bad_planes:
        _raises(ValueError, SFF.figure_soma_fit_planes, **dict(case, **bad))
    # the guards beside them: masks not bool, ks not integers, a non-finite or empty extent, a negative radius
    for bad in (dict(region=case["region"].astype(float)), dict(mask=case["mask"].astype(np.uint8)),
                dict(ks=KS + 0.5), dict(ks=np.ones(K, dtype=bool)), dict(extent_um=np.array([0.0, 1.0, np.nan, 0.0])),
                dict(extent_um=np.array([1.0, 1.0, 2.0, 0.0])), dict(allen_circle=np.array([XS, YS, -1.0])),
                dict(sections_xy="none"), dict(sections_xy=np.zeros((2, 5)))):
        _raises(ValueError, SFF.figure_soma_fit_planes, **dict(case, **bad))
    _close(SFF.figure_soma_fit_planes(**dict(case, ks=KS.astype(float))))     # integral floats are plane indices

    # the summary
    s = summary_case()
    for bad in (dict(theta=s["theta"][:-1]), dict(se=s["se"][:, None]), dict(names=PARAMS[:-1]),
                dict(B_gl=s["B_gl"][:-1]), dict(rms_gl=s["rms_gl"][1:]), dict(z_um=s["z_um"][:, None]),
                dict(z_um=np.zeros(0), B_gl=np.zeros(0), rms_gl=np.zeros(0)), dict(fitted=PARAMS + ("extra",)),
                dict(fitted="x0_um"), dict(names="x0_um"), dict(names=PARAMS[:-1] + ("x0_um",)),
                dict(derived=dict(area_um2=1.0, volume_um3=1.0)), dict(derived=[1.0, 2.0, 3.0]),
                dict(derived=dict(s["derived"], r_eq_um="large")), dict(allen=dict(radius_um=RS)),
                dict(allen=dict(s["allen"], area_um2=None)), dict(rms_gl=-s["rms_gl"]),
                dict(noise_gl=float("nan")), dict(noise_gl=-1.0), dict(noise_gl="2 gl")):
        _raises(ValueError, SFF.figure_soma_fit_summary, **dict(s, **bad))
    # the optional SEs and Allen's volume given are shown; None reads as not given
    fig = SFF.figure_soma_fit_summary(**dict(s, derived=dict(s["derived"], se_volume_um3=2.5, se_r_eq_um=None),
                                             allen=dict(s["allen"], volume_um3=179.6)))
    der = _table(fig, "derived")
    assert _cell(der, 2, 2) == SFF.SE_FMT % 2.5 and _cell(der, 3, 2) == SFF.NO_VALUE
    assert _cell(der, 2, 3) == SFF.DERIVED_FMT % 179.6
    _close(fig)

    # the morphology
    m = morphology_case()
    st0 = m["top"]["stems"][0]
    for bad in (dict(top=dict(m["top"], silhouette=m["top"]["silhouette"].T)),
                dict(top={k: v for k, v in m["top"].items() if k != "sphere"}),
                dict(side=dict(m["side"], allen=m["side"]["allen"][0])),
                dict(top=dict(m["top"], stems=[dict(st0, old=np.zeros(3))] + m["top"]["stems"][1:])),
                dict(top=dict(m["top"], stems=[dict(st0, new=np.zeros((2, 1)))] + m["top"]["stems"][1:])),
                dict(top=dict(m["top"], stems=[dict(st0, next=np.zeros((3, 2)))] + m["top"]["stems"][1:])),
                dict(top=dict(m["top"], stems=[{k: v for k, v in st0.items() if k != "flag"}] + m["top"]["stems"][1:])),
                dict(top=dict(m["top"], stems=[dict(st0, flag=1)] + m["top"]["stems"][1:])),
                dict(top=dict(m["top"], stems=m["top"]["stems"][:-1])),
                dict(side=dict(m["side"], stems=m["side"]["stems"][::-1])),
                dict(top=dict(m["top"], stems=st0)), dict(top=[m["top"]]), dict(side=None)):
        _raises(ValueError, SFF.figure_soma_morphology, **dict(m, **bad))
    # an unknown flag is drawn and named, in its own colour
    odd = [dict(st, flag="unforeseen") if i == 0 else st for i, st in enumerate(m["top"]["stems"])]
    odd_side = [dict(st, flag="unforeseen") if i == 0 else st for i, st in enumerate(m["side"]["stems"])]
    fig = SFF.figure_soma_morphology(top=dict(m["top"], stems=odd), side=dict(m["side"], stems=odd_side))
    assert "flag: unforeseen" in [t.get_text() for t in fig.legends[0].get_texts()]
    _close(fig)


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
