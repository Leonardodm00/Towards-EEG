"""Spec-derived tests, Block 48 (the soma fit's figures; D-215, D-222.5) -- tester 2026-11-23.

Oracles (SPEC Block 48): T1 each function returns a Figure with the expected number of axes, saved to a non-empty PNG;
T2 no input mutated; T3 sections that are None and an empty stem list are drawn without error; T4 the residual panels
share one colour scale, symmetric about 0; T5 the ValueErrors for shapes that disagree. Section 4 and "Figures only:
every number drawn is passed in": the arrays shown are the data, the model and data - model exactly; the image panels
use the extent passed (image frame, um); the outlines and circles drawn are the ones passed.
"""
from __future__ import annotations

import copy
import io
import math

import matplotlib
matplotlib.use("Agg")
import numpy as np                                     # noqa: E402
import pytest                                          # noqa: E402
from numpy.testing import assert_array_equal           # noqa: E402

import soma_spec_helpers as H                          # noqa: E402,F401  (paths)
from allen_diameter.plotting import soma_fit_figures as FF   # noqa: E402

K, HH, WW = 3, 24, 30
EXT = (-1.5, 1.8, 1.3, -1.4)                  # (left, right, bottom, top) um, y downwards


def _planes_inputs(seed=0):
    rng = np.random.default_rng(seed)
    data = 100 + rng.normal(0, 3, (K, HH, WW))
    model = 100 - 20 * rng.uniform(size=(K, HH, WW))
    yy, xx = np.mgrid[0:HH, 0:WW]
    region = (xx - 15) ** 2 + (yy - 12) ** 2 <= 100
    mask = (xx > 20) & (np.abs(yy - 12) < 3)
    t = np.linspace(0, 2 * math.pi, 50)
    sil = np.vstack([0.2 + 0.9 * np.cos(t), -0.1 + 0.7 * np.sin(t)])
    secs = [None, np.vstack([0.2 + 0.5 * np.cos(t), -0.1 + 0.4 * np.sin(t)]), np.zeros((2, 0))]
    return dict(data=data, model=model, region=region, mask=mask, ks=np.array([3, 5, 7]),
                dz_from_centre_um=np.array([-0.56, 0.0, 0.56]), extent_um=np.array(EXT), silhouette_xy=sil,
                sections_xy=secs, allen_circle=np.array([0.1, 0.0, 1.0]), eq_circle=np.array([0.2, -0.1, 0.75]))


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=40)
    return buf.getvalue()


def _images(fig, gid):
    return [im for ax in fig.axes for im in ax.images if im.get_gid() == gid]


def test_planes_figure_axes_arrays_scale_and_no_mutation():
    kw = _planes_inputs()
    snap = copy.deepcopy(kw)
    fig = FF.figure_soma_fit_planes(**kw, title="t")
    img_axes = [ax for ax in fig.axes if ax.images]
    assert len(img_axes) == 3 * K                                       # T1: one row of three per plane
    assert len(_png(fig)) > 1000
    for name, v in kw.items():                                          # T2
        if isinstance(v, np.ndarray):
            assert_array_equal(v, snap[name])
    for a, b in zip(kw["sections_xy"], snap["sections_xy"]):
        assert (a is None and b is None) or np.array_equal(a, b)
    data, model = kw["data"], kw["model"]
    D, M, R = _images(fig, "data"), _images(fig, "model"), _images(fig, "residual")
    assert len(D) == len(M) == len(R) == K
    for j in range(K):
        assert_array_equal(np.asarray(D[j].get_array()), data[j])
        assert_array_equal(np.asarray(M[j].get_array()), model[j])
        assert_array_equal(np.asarray(R[j].get_array()), data[j] - model[j])
        for im in (D[j], M[j], R[j]):
            assert tuple(im.get_extent()) == EXT                        # the extent passed, image frame
    # T4: one scale for every residual panel, symmetric about 0
    lims = {(im.norm.vmin, im.norm.vmax) for im in R}
    assert len(lims) == 1
    vmin, vmax = lims.pop()
    assert vmin == -vmax and vmax > 0


def test_planes_figure_draws_the_outlines_passed():
    kw = _planes_inputs()
    fig = FF.figure_soma_fit_planes(**kw)
    first = [ax for ax in fig.axes if ax.images and ax.images[0].get_gid() == "data"]
    sil_lines = [ln for ln in first[0].lines if ln.get_gid() == "silhouette"]
    assert len(sil_lines) == 1
    assert_array_equal(sil_lines[0].get_xydata().T, kw["silhouette_xy"])
    sec1 = [ln for ln in first[1].lines if ln.get_gid() == "section"]
    assert len(sec1) == 1 and np.array_equal(sec1[0].get_xydata().T, kw["sections_xy"][1])
    assert not [ln for ln in first[0].lines if ln.get_gid() == "section"]       # None: no section (T3)
    circles = {p.get_gid(): p for p in first[0].patches}
    assert tuple(circles["allen"].center) == (0.1, 0.0) and circles["allen"].radius == 1.0
    assert tuple(circles["sphere"].center) == (0.2, -0.1) and circles["sphere"].radius == 0.75


@pytest.mark.parametrize("change", ["model", "region", "mask", "ks", "dz", "extent", "sections", "allen", "eq"])
def test_planes_figure_refusals(change):
    kw = _planes_inputs()
    if change == "model":
        kw["model"] = kw["model"][:, :, :-1]
    elif change == "region":
        kw["region"] = kw["region"][:-1]
    elif change == "mask":
        kw["mask"] = kw["mask"].astype(int)
    elif change == "ks":
        kw["ks"] = kw["ks"][:2]
    elif change == "dz":
        kw["dz_from_centre_um"] = kw["dz_from_centre_um"][:2]
    elif change == "extent":
        kw["extent_um"] = kw["extent_um"][:3]
    elif change == "sections":
        kw["sections_xy"] = kw["sections_xy"][:2]
    elif change == "allen":
        kw["allen_circle"] = np.array([0.0, 1.0])
    else:
        kw["eq_circle"] = np.array([0.0, 0.0, -1.0])
    with pytest.raises(ValueError):
        FF.figure_soma_fit_planes(**kw)


def _summary_inputs():
    names = ("x0_um", "y0_um", "z0_um", "a_um", "b_um", "c_um", "phi_rad", "mu_per_um", "slope_um_per_um")
    theta = np.array([0.6, -0.4, 0.9, 4.4, 3.4, 3.0, 0.6, 0.15, 0.79])
    se = np.array([1e-3] * 8 + [np.nan])
    return dict(names=names, theta=theta, se=se, fitted=names[:8],
                derived=dict(area_um2=162.1, volume_um3=188.5, r_eq_um=3.59, se_area_um2=0.17),
                allen=dict(radius_um=4.0, area_um2=4 * math.pi * 16), z_um=np.array([-1.0, 0.0, 1.0]),
                B_gl=np.array([110.0, 111.0, 109.5]), rms_gl=np.array([3.0, 3.1, 2.9]))


def test_summary_figure():
    kw = _summary_inputs()
    snap = copy.deepcopy(kw)
    fig = FF.figure_soma_fit_summary(**kw, noise_gl=3.0, title="t")
    assert len(fig.axes) == 4 and len(_png(fig)) > 1000
    for name, v in kw.items():
        if isinstance(v, np.ndarray):
            assert_array_equal(v, snap[name])
    lines = {ln.get_gid(): ln for ax in fig.axes for ln in ax.lines}
    assert_array_equal(lines["B"].get_xydata(), np.column_stack([kw["z_um"], kw["B_gl"]]))
    assert_array_equal(lines["rms"].get_xydata(), np.column_stack([kw["z_um"], kw["rms_gl"]]))
    assert lines["noise"].get_ydata()[0] == 3.0
    texts = [c.get_text().get_text() for ax in fig.axes for t in ax.tables for c in t.get_celld().values()]
    assert "held" in texts                                               # gamma held
    for bad in (dict(theta=kw["theta"][:8]), dict(se=kw["se"][:3]), dict(B_gl=kw["B_gl"][:2]),
                dict(rms_gl=kw["rms_gl"][:2]), dict(derived=dict(area_um2=1.0)), dict(allen=dict(radius_um=1.0)),
                dict(fitted=("nope",))):
        k2 = dict(kw)
        k2.update(bad)
        with pytest.raises(ValueError):
            FF.figure_soma_fit_summary(**k2)


def _view(n_stems=2, flags=("moved", "missed")):
    t = np.linspace(0, 2 * math.pi, 40)
    circ = np.vstack([np.cos(t), np.sin(t)])
    stems = [dict(old=np.array([2.0 + j, 0.5]), new=np.array([1.0 + 0.1 * j, 0.4]) if flags[j] == "moved" else
                  np.array([2.0 + j, 0.5]), next=np.array([[3.0 + j, 4.0 + j], [0.6, 0.7]]), flag=flags[j])
             for j in range(n_stems)]
    return dict(allen=circ * 1.1, silhouette=circ * 0.9, sphere=circ, stems=stems)


def test_morphology_figure():
    top, side = _view(), _view()
    snap = copy.deepcopy((top, side))
    fig = FF.figure_soma_morphology(top, side, title="t")
    assert len(fig.axes) == 2 and len(_png(fig)) > 1000
    for v, s in zip((top, side), snap):
        for key in ("allen", "silhouette", "sphere"):
            assert_array_equal(v[key], s[key])
    ax = fig.axes[0]
    olds = [ln for ln in ax.lines if ln.get_gid() == "old"]
    news = [ln for ln in ax.lines if ln.get_gid() == "new"]
    assert len(olds) == len(news) == 2
    assert_array_equal(olds[0].get_xydata()[0], [2.0, 0.5])
    assert_array_equal(news[0].get_xydata()[0], [1.0, 0.4])
    arrows = [a for a in ax.texts if a.get_gid() == "arrow"]
    assert len(arrows) == 1                                              # only the stem that moved
    fig = FF.figure_soma_morphology(dict(_view(), stems=[]), dict(_view(), stems=[]))     # T3: no stem
    assert len(fig.axes) == 2
    with pytest.raises(ValueError):
        FF.figure_soma_morphology(_view(2), _view(1, flags=("moved",)))
    bad = _view()
    del bad["sphere"]
    with pytest.raises(ValueError):
        FF.figure_soma_morphology(bad, _view())


def test_region_boundary_between_pixel_centres():
    """The region's boundary lies half-way between the centres of the pixels in and out of it (the image frame of the
    extent passed): a band of columns 10..19 bounds at x = left + 10 w and left + 20 w, w the pixel width."""
    kw = _planes_inputs()
    region = np.zeros((HH, WW), bool)
    region[:, 10:20] = True
    kw["region"] = region
    kw["mask"] = np.zeros((HH, WW), bool)
    fig = FF.figure_soma_fit_planes(**kw)
    ax = [a for a in fig.axes if a.images and a.images[0].get_gid() == "data"][0]
    w = (EXT[1] - EXT[0]) / WW
    xs = []
    for coll in ax.collections:
        if coll.get_gid() == "region":
            for path in coll.get_paths():
                xs.extend(path.vertices[:, 0].tolist())
    xs = np.array(xs)
    assert xs.size > 0
    vertical = np.unique(np.round(xs, 9))
    assert np.any(np.isclose(vertical, EXT[0] + 10 * w, atol=1e-9)) and np.any(np.isclose(vertical, EXT[0] + 20 * w,
                                                                                           atol=1e-9)), vertical
