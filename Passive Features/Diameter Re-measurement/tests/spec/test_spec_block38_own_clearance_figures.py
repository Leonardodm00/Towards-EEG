"""Spec-derived tests of Block 38 (the same-node clearance figures; D-156.2) -- integration tester,
2026-11-17. Oracles from specs/SPEC.md Block 38 (and the "Accepted as part of the entry" items); the
OwnClearance is built by hand (the dataclass of Block 37 filled with test arrays), so nothing scientific is
computed by the code under test here.

Run: python3 -m pytest -q tests/spec/test_spec_block38_own_clearance_figures.py
Pure ASCII.
"""
from __future__ import annotations

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_array_equal  # noqa: E402

import nc_helpers  # noqa: F401,E402  (sys.path)
from allen_diameter.analysis.own_clearance import OwnClearance  # noqa: E402
from allen_diameter.plotting import own_clearance_figures as OF  # noqa: E402


def _data(n_planes=7, n_v=25, n_layers=9):
    rng = np.random.default_rng(3)
    v = np.linspace(-3, 3, n_v)
    ks = np.arange(10, 10 + n_planes)
    B = rng.uniform(90, 100, n_planes)
    prof = B[:, None] - rng.uniform(0, 30, (n_planes, n_v))
    chord = np.linspace(0, 0.8, n_planes)
    chord[3] = 0.0
    clr = OwnClearance(cleared_gl=prof + 3, model_in_gl=prof + 2, model_gl=prof + 1, residual_gl=prof * 0,
                       chord_um=chord, in_peak_gl=np.ones(n_planes), out_peak_gl=np.ones(n_planes),
                       in_share=np.linspace(0.1, 0.9, n_planes), residual_rms_gl=np.ones(n_planes),
                       layers_gl=rng.uniform(0, 5, (n_layers, n_planes, n_v)))
    layer_z = 2.8 + 0.1 * np.arange(n_layers)
    planes_z = 0.28 * ks
    return v, ks, prof, B, clr, layer_z, planes_z


def _by_gid(ax, gid):
    return [ln for ln in ax.get_lines() if ln.get_gid() == gid]


def test_figure_own_panels_data_chord_and_shared_y():
    """One panel per plane shown, in the order of `planes`; the drawn lines' data equal the inputs [exact];
    the chord bar from v0 - c/2 to v0 + c/2, absent where c = 0; one y range for all panels; inputs unchanged."""
    v, ks, prof, B, clr, _lz, _pz = _data()
    copies = [prof.copy(), B.copy(), clr.cleared_gl.copy()]
    planes = [13, 11, 15]
    v0 = 0.37
    fig = OF.figure_own(v, ks, prof, B, clr, {"node_id": 5, "class": "x"}, planes=planes, v0_um=v0)
    axes = [ax for ax in fig.axes if ax.get_visible() and ax.get_lines()]
    assert len(axes) == len(planes)
    ylims = {tuple(ax.get_ylim()) for ax in axes}
    assert len(ylims) == 1
    for ax, k in zip(axes, planes):
        j = list(ks).index(k)
        assert str(k) in ax.get_title()
        for gid, arr in (("profile", prof[j]), ("cleared", clr.cleared_gl[j]), ("model_in", clr.model_in_gl[j]),
                         ("model", clr.model_gl[j])):
            lines = _by_gid(ax, gid)
            assert len(lines) == 1, gid
            assert_array_equal(lines[0].get_xdata(), v)
            assert_array_equal(lines[0].get_ydata(), arr)
        bg = _by_gid(ax, "background")
        assert len(bg) == 1 and np.allclose(bg[0].get_ydata(), B[j])
        bars = _by_gid(ax, "chord")
        c = clr.chord_um[j]
        if c == 0:
            assert not bars
        else:
            assert len(bars) == 1
            assert_array_equal(bars[0].get_xdata(), [v0 - c / 2, v0 + c / 2])
    for a, b in zip((prof, B, clr.cleared_gl), copies):
        assert_array_equal(a, b)
    plt.close(fig)
    fig = OF.figure_own(v, ks, prof, B, clr, {}, planes=[13])            # chord 0 there: no bar
    ax = [a for a in fig.axes if a.get_visible() and a.get_lines()][0]
    assert not _by_gid(ax, "chord")
    plt.close(fig)
    fig = OF.figure_own(v, ks, prof, B, clr, {})                          # None: all planes
    assert len([a for a in fig.axes if a.get_visible() and a.get_lines()]) == len(ks)
    plt.close(fig)


def test_figure_layers_image_and_lines():
    """figure_layers draws an image of shape (n_layers, n_v) equal to layers_gl[:, plane, :] [exact] and the
    plane's depth and band z_k -+ beta as horizontal lines."""
    v, ks, prof, B, clr, lz, pz = _data()
    beta = 0.14
    fig = OF.figure_layers(v, lz, clr.layers_gl, ks, 12, pz, beta, {"node_id": 1})
    ax = fig.axes[0]
    meshes = [c for c in ax.collections if c.get_gid() == "layers"]
    assert len(meshes) == 1
    arr = np.asarray(meshes[0].get_array()).reshape(lz.size, v.size)
    assert_array_equal(arr, clr.layers_gl[:, 2, :])
    zk = pz[2]
    pl = _by_gid(ax, "plane_z")
    bd = _by_gid(ax, "band")
    assert len(pl) == 1 and np.allclose(pl[0].get_ydata(), zk)
    assert sorted(float(b.get_ydata()[0]) for b in bd) == pytest.approx([zk - beta, zk + beta], abs=1e-12)
    plt.close(fig)


def test_save_and_determinism():
    v, ks, prof, B, clr, lz, pz = _data()
    d = tempfile.mkdtemp(prefix="b38_")
    outs = []
    for i in range(2):
        fig = OF.figure_own(v, ks, prof, B, clr, {"node_id": 1}, planes=[11, 12])
        path = OF.save(fig, os.path.join(d, "sub", "own%d.png" % i), 80)
        plt.close(fig)
        assert os.path.getsize(path) > 0
        outs.append(open(path, "rb").read())
    assert outs[0] == outs[1]


def test_errors():
    """ValueError: shapes that disagree, an empty planes, a plane not in ks, layers_gl None in figure_layers."""
    v, ks, prof, B, clr, lz, pz = _data()
    with pytest.raises(ValueError):
        OF.figure_own(v[:-1], ks, prof, B, clr, {})
    with pytest.raises(ValueError):
        OF.figure_own(v, ks, prof, B[:-1], clr, {})
    with pytest.raises(ValueError):
        OF.figure_own(v, ks, prof, B, clr, {}, planes=[])
    with pytest.raises(ValueError):
        OF.figure_own(v, ks, prof, B, clr, {}, planes=[99])
    with pytest.raises(ValueError):
        OF.figure_layers(v, lz, None, ks, 12, pz, 0.14, {})
    with pytest.raises(ValueError):
        OF.figure_layers(v, lz[:-1], clr.layers_gl, ks, 12, pz, 0.14, {})
    with pytest.raises(ValueError):
        OF.figure_layers(v, lz, clr.layers_gl, ks, 99, pz, 0.14, {})
    plt.close("all")
