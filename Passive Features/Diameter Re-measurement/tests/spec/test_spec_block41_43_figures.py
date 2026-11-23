"""Spec-derived tests of Block 41 (the clearness figures) and Block 43 (the sigma calibration's figures);
tester, 2026-11-22. Written from specs/SPEC.md before the code was read. Plotting only: the oracles are the
entries' "drawn data equal the inputs", the shared ranges, the bars, the hollow markers, the refusals.

Pure ASCII.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402

HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
for _p in (WS / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from allen_diameter.plotting import clearness_figures as CF  # noqa: E402
from allen_diameter.plotting import sigma_figures as SF  # noqa: E402

P = 0.1144


def _lines(fig, gid):
    return [ln for ax in fig.axes for ln in ax.get_lines() if ln.get_gid() == gid]


def node41(n_k=5, seed=0):
    rng = np.random.default_rng(seed)
    v = np.arange(-26, 27) * P
    u = np.arange(-4, 5) * 0.5
    return dict(images=rng.uniform(100, 220, (n_k, 60, 70)), extent=(-4.0, 4.008, 3.432, -3.432), ks=np.arange(n_k) - 2,
                k_index=2, origin_xy=np.array([0.1, -0.2]), y_hat=np.array([0.0, 1.0]), e_u=np.array([1.0, 0.0]),
                v_um=v, profiles_gl=200 - 50 * np.exp(-0.5 * (v / 0.4) ** 2)[None, :] + rng.normal(0, 1, (n_k, v.size)),
                B_gl=np.full(n_k, 200.0), u_um=u, strips_gl=rng.uniform(150, 200, (u.size, v.size)),
                d_allen_um=0.6, x_lo_um=np.full(n_k, -0.45), x_hi_um=np.full(n_k, 0.47), focus=np.arange(n_k) < 3,
                title="node", text="numbers")


def test_figure_node_41_data_bars_ranges_and_inputs():
    n = node41()
    copies = {k: np.array(v, copy=True) for k, v in n.items() if isinstance(v, np.ndarray)}
    fig = CF.figure_node(n)
    prof = _lines(fig, "profile")
    assert len(prof) == 5
    for i, ln in enumerate(prof):
        assert_array_equal(ln.get_ydata(), n["profiles_gl"][i]) and assert_array_equal(ln.get_xdata(), n["v_um"])
    fw = _lines(fig, "fwhm")
    assert len(fw) == 5 and all(tuple(ln.get_xdata()) == (-0.45, 0.47) for ln in fw)
    pro_axes = {ln.axes for ln in prof}
    bars = [ln for ln in _lines(fig, "allen") if ln.axes in pro_axes]               # the bars of the profile row
    assert len(bars) == 5 and all(abs(ln.get_xdata()[1] - ln.get_xdata()[0] - 0.6) <= 1e-12 for ln in bars)
    edges = [ln for ln in _lines(fig, "allen") if ln.axes not in pro_axes]          # the dashed edges, D_A apart
    ys = sorted({round(float(ln.get_ydata()[0]), 12) for ln in edges})
    assert len(ys) == 2 and abs(ys[1] - ys[0] - 0.6) <= 1e-12
    imgs = [ax for ax in fig.axes if ax.get_images()]
    assert len(imgs) == 5 + 1                                        # one per plane plus the large one
    assert len({ax.get_images()[0].get_clim() for ax in imgs}) == 1  # one grey range
    pro_axes = [ax for ax in fig.axes if any(ln.get_gid() == "profile" for ln in ax.get_lines())]
    assert len({ax.get_ylim() for ax in pro_axes}) == 1
    strips = [ln for ln in fig.axes[1].get_lines() if str(ln.get_gid()).startswith("strip")]
    assert len(strips) == n["u_um"].size                              # one curve per line along the dendrite
    overlay = [ln for ln in fig.axes[2].get_lines() if str(ln.get_gid()).startswith("plane")]
    assert len(overlay) == 5
    for k, v in copies.items():
        assert_array_equal(n[k], v)
    plt.close(fig)
    m = node41()
    for k in ("d_allen_um", "x_lo_um", "x_hi_um", "focus", "title", "text"):
        m.pop(k)
    plt.close(CF.figure_node(m))                                      # the optional keys may be left out
    plt.close(CF.figure_sheet([node41(), node41(seed=1)]))


@pytest.mark.parametrize("bad", [dict(d_allen_um=0.0), dict(k_index=5), dict(B_gl=np.ones(4)),
                                 dict(strips_gl=np.ones((9, 5))), dict(images=np.ones((4, 6, 7)))])
def test_figure_node_41_refusals(bad):
    n = node41()
    n.update(bad)
    with pytest.raises(ValueError):
        CF.figure_node(n)


def test_figure_node_41_refusals_of_the_call():
    with pytest.raises(ValueError):
        CF.figure_node(node41(), zoom_um=0.0)
    with pytest.raises(ValueError):
        CF.figure_sheet([])
    n = node41()
    n.pop("strips_gl")
    with pytest.raises(ValueError):
        CF.figure_node(n)
    with pytest.raises(ValueError):
        CF.figure_node([1, 2])


def node43(n_k=5, seed=0):
    rng = np.random.default_rng(seed)
    v = np.arange(-39, 40) * P
    prof = (150 - 2 * v)[None, :] * (1 - 0.4 * np.exp(-0.5 * (v / 0.8) ** 2))[None, :] + rng.normal(0, 1, (n_k, v.size))
    model = (150 - 2 * v)[None, :] * (1 - 0.4 * np.exp(-0.5 * (v / 0.8) ** 2))[None, :] * np.ones((n_k, 1))
    return dict(images=rng.uniform(100, 220, (n_k, 60, 70)), extent=(-4.0, 4.008, 3.432, -3.432),
                ks=np.arange(n_k) - 2, k_index=2, origin_xy=np.array([0.1, -0.2]), y_hat=np.array([0.0, 1.0]),
                v_um=v, profiles_gl=prof, model_gl=model, background_gl=np.tile(150 - 2 * v, (n_k, 1)),
                title="t", text="x")


def test_figure_node_fit_43():
    n = node43()
    p0, m0 = n["profiles_gl"].copy(), n["model_gl"].copy()
    fig = SF.figure_node_fit(n)
    pr, mo, re = _lines(fig, "profile"), _lines(fig, "model"), _lines(fig, "residual")
    assert len(pr) == len(mo) == len(re) == 5
    for i in range(5):
        assert_array_equal(pr[i].get_ydata(), p0[i]) and assert_array_equal(mo[i].get_ydata(), m0[i])
        assert_allclose(re[i].get_ydata(), p0[i] - m0[i], rtol=0, atol=0)
    imgs = [ax for ax in fig.axes if ax.get_images()]
    assert len(imgs) == 5 and len({ax.get_images()[0].get_clim() for ax in imgs}) == 1
    p_axes = {ln.axes for ln in pr}
    r_axes = {ln.axes for ln in re}
    assert len({ax.get_ylim() for ax in p_axes}) == 1 and len({ax.get_ylim() for ax in r_axes}) == 1
    assert_array_equal(n["profiles_gl"], p0) and assert_array_equal(n["model_gl"], m0)
    plt.close(fig)
    for bad in (dict(k_index=7), dict(model_gl=np.ones((4, 79))), dict(origin_xy=np.array([np.nan, 0.0]))):
        m = node43()
        m.update(bad)
        with pytest.raises(ValueError):
            SF.figure_node_fit(m)
    with pytest.raises(ValueError):
        SF.figure_node_fit(node43(), zoom_um=-1.0)


def rows43():
    return [dict(node_id=i, sigma0_um=s, se_sigma0_um=e, growth_scale=c, se_growth_scale=ec, d_hat_um=d,
                 d_allen_um=da, depth_um=z, ok=ok)
            for i, (s, e, c, ec, d, da, z, ok) in enumerate([(0.10, 0.01, 1.1, 0.1, 1.5, 1.2, 3.0, True),
                                                              (0.14, 0.02, 0.9, 0.2, 2.0, 1.4, 5.0, True),
                                                              (0.30, 0.05, 2.0, 0.5, 1.2, 1.1, 7.0, False)])]


def test_figure_summary_43():
    from allen_diameter.analysis.sigma_calibration import summarize
    rows = rows43()
    s = summarize(np.array([r["sigma0_um"] for r in rows]), np.array([r["se_sigma0_um"] for r in rows]),
                  np.array([r["ok"] for r in rows]))
    fig = SF.figure_summary(rows, s)
    mean, = _lines(fig, "ivw_mean")
    assert_allclose(mean.get_ydata(), [s["ivw_mean_um"]] * 2, rtol=0, atol=0)
    filled, = [ln for ln in _lines(fig, "sigma0|ok")]
    hollow, = [ln for ln in _lines(fig, "sigma0|not_ok")]
    assert hollow.get_markerfacecolor() == "none" and filled.get_markerfacecolor() != "none"
    assert_allclose(sorted(filled.get_ydata()), [0.10, 0.14]) and assert_allclose(hollow.get_ydata(), [0.30])
    bars = [c for ax in fig.axes for c in ax.collections if c.get_gid() == "se|sigma0"]
    assert bars
    segs = bars[0].get_segments()
    half = sorted(abs(sg[1][1] - sg[0][1]) / 2 for sg in segs)
    assert_allclose(half, [0.01, 0.02, 0.05], rtol=1e-12)            # the bars are the SEs
    plt.close(fig)
    for bad in ([], [dict(rows[0], ok=1)], [{k: v for k, v in rows[0].items() if k != "depth_um"}],
                [dict(rows[0], se_sigma0_um=-0.1)]):
        with pytest.raises(ValueError):
            SF.figure_summary(bad, s)
    with pytest.raises(ValueError):
        SF.figure_summary(rows, {k: v for k, v in s.items() if k != "dof"})
