"""Spec-derived tests of Block 34 (the inference figures) -- integration tester, 2026-11-14. Oracles from
specs/SPEC.md Block 34: per plane shown a profile axis and a residual axis; the drawn data equal the inputs
and the residual is (profile - model) / sigma [exact]; lines at +-2; unused planes greyed; hollow markers for
rows not converged; inputs unchanged; ValueErrors. The NodeFit is built by hand (Block 30's dataclass).

Run: python3 -m pytest -q tests/spec/test_spec_block34_inference_figures.py
Pure ASCII.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import nc_helpers  # noqa: F401  (sys.path)
from allen_diameter.analysis.node_inference import NodeFit
from allen_diameter.plotting import inference_figures as IF


def make_fit(n_planes, used, status="converged", se_nan=False):
    se = np.full(4, np.nan) if se_nan else np.array([0.01, 0.02, 0.003, 0.004])
    chi2 = np.where(used, 1.1, np.nan)
    return NodeFit(theta=np.array([0.8, 1.0, 0.01, 0.3]), cov=np.diag(se ** 2), se=se, cost=12.0,
                   chi2_per_plane=chi2, planes=np.asarray(used, dtype=bool), status=status,
                   at_bound=("d_lo",) if status == "at_bound" else (), n_starts=1, nfev=10)


def data(n_planes=5, n_v=21, seed=0):
    rng = np.random.default_rng(seed)
    v = np.linspace(-3, 3, n_v)
    model = 210.0 * (1 - 0.3 * np.exp(-v[None, :] ** 2 / 0.2)) * np.ones((n_planes, 1))
    prof = model + rng.normal(0, 3.0, size=model.shape)
    return prof, model, v, np.arange(-2, -2 + n_planes)


def lines_by_gid(ax, gid):
    return [ln for ln in ax.get_lines() if ln.get_gid() == gid]


def test_figure_fit_drawn_data():
    prof, model, v, ks = data()
    used = np.array([True, True, False, True, True])
    fit = make_fit(5, used)
    p0, m0 = prof.copy(), model.copy()
    sigma = 3.0
    fig = IF.figure_fit(prof, model, v, ks, sigma, fit, {"node_id": 7, "class": "clean"})
    prof_axes = [ax for ax in fig.axes if lines_by_gid(ax, "profile")]
    res_axes = [ax for ax in fig.axes if lines_by_gid(ax, "residual")]
    assert len(prof_axes) == 5 and len(res_axes) == 5          # one of each per plane shown
    for j, (ap, ar) in enumerate(zip(prof_axes, res_axes)):
        assert_array_equal(lines_by_gid(ap, "profile")[0].get_ydata(), prof[j])
        assert_array_equal(lines_by_gid(ap, "model")[0].get_ydata(), model[j])
        assert_array_equal(lines_by_gid(ar, "residual")[0].get_ydata(), (prof[j] - model[j]) / sigma)
        lims = sorted(float(ln.get_ydata()[0]) for ln in lines_by_gid(ar, "resid_limit"))
        assert lims == [-2.0, 2.0]
        col = matplotlib.colors.to_hex(lines_by_gid(ap, "profile")[0].get_color())
        assert (col == IF.GREYED) == (not used[j])
    assert_array_equal(prof, p0)
    assert_array_equal(model, m0)
    sub = IF.figure_fit(prof, model, v, ks, sigma, fit, {}, planes=[0, -2])
    assert len([ax for ax in sub.axes if lines_by_gid(ax, "profile")]) == 2


def test_figure_fit_errors_and_nan_se():
    prof, model, v, ks = data()
    fit = make_fit(5, np.ones(5, bool), status="at_bound", se_nan=True)
    IF.figure_fit(prof, model, v, ks, 3.0, fit, {"node_id": 1})          # NaN se must not raise
    with pytest.raises(ValueError):
        IF.figure_fit(prof, model[:, :-1], v, ks, 3.0, fit, {})
    with pytest.raises(ValueError):
        IF.figure_fit(prof, model, v, ks, 3.0, fit, {}, planes=[])
    with pytest.raises(ValueError):
        IF.figure_fit(prof, model, v, ks, 0.0, fit, {})


def test_figure_diameters_hollow_and_errors(tmp_path):
    rows = [dict(node_id=1, d_allen_um=0.6, d_fit_um=0.8, se_um=0.01, status="converged", d_pilot_um=0.85,
                 **{"class": "clean"}),
            dict(node_id=2, d_allen_um=0.6, d_fit_um=0.9, se_um=float("nan"), status="at_bound",
                 **{"class": "clean"}),
            dict(node_id=3, d_allen_um=1.0, d_fit_um=1.2, se_um=0.02, status="failed", d_pilot_um=None)]
    fig = IF.figure_diameters(rows)
    assert len(fig.axes) == 2
    filled, hollow = [], []
    for ax in fig.axes:
        for ln in ax.get_lines():
            g = ln.get_gid() or ""
            if g.startswith("data|"):
                (filled if g.endswith("|converged") else hollow).append(ln)
                if g.endswith("not_converged"):
                    assert ln.get_markerfacecolor() == "none"
    xs_all = sorted(float(x) for ln in fig.axes[0].get_lines() if (ln.get_gid() or "").startswith("data|")
                    for x in ln.get_xdata())
    assert xs_all == [0.6, 0.6, 1.0]
    assert len(filled) >= 1 and len(hollow) >= 1
    out = IF.save(fig, str(tmp_path / "d.png"), 60)
    assert (tmp_path / "d.png").stat().st_size > 0 and str(out).endswith("d.png")
    with pytest.raises(ValueError):
        IF.figure_diameters([])
    with pytest.raises(ValueError):
        IF.figure_diameters([dict(node_id=1, d_allen_um=0.6, d_fit_um=0.8, se_um=0.01)])


def test_figure_chi2_and_determinism(tmp_path):
    rows = [dict(node_id=1, chi2_per_plane=np.array([1.0, np.nan, 0.9]), ks=np.array([0, 1, 2])),
            dict(node_id=2, chi2_per_plane=np.array([1.2, 1.1]), ks=np.array([3, 4]))]
    IF.save(IF.figure_chi2(rows), str(tmp_path / "a.png"), 60)
    IF.save(IF.figure_chi2(rows), str(tmp_path / "b.png"), 60)
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()
    with pytest.raises(ValueError):
        IF.figure_chi2([])
