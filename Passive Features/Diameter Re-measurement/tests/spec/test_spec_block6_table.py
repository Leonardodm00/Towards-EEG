"""Block 6 (phantoms and the bias table): oracles from SPEC.md section 2.4 and Block 6."""
import dataclasses
import math
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

from allen_diameter.config import default_config
from allen_diameter.analysis import phantoms as PH, table as TB
from allen_diameter.analysis.node_pipeline import NodeResult
from allen_diameter.loading import table_io

CFG = default_config()


def test_draws_reproducible_and_chunk_independent():
    a = [PH.draw_replicate(CFG, 11, n)[0] for n in range(20)]
    b = [PH.draw_replicate(CFG, 11, n)[0] for n in range(19, -1, -1)][::-1]
    assert a == b
    # replicate n uses default_rng([seed, n]) and draws d, phi, theta, mu in that order (spec)
    rng = np.random.default_rng([11, 7])
    lo, hi = CFG.phantom.d_range_um
    d = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    phi = math.radians(rng.uniform(*CFG.phantom.phi_range_deg))
    assert a[7].d_um == pytest.approx(d, rel=1e-15) and a[7].phi_rad == pytest.approx(phi, rel=1e-15)


def test_draw_laws_ks():
    D = [PH.draw_replicate(CFG, 5, n)[0] for n in range(2000)]
    ph, p = CFG.phantom, CFG.acquisition.res0_um
    ld = np.log([x.d_um for x in D])
    assert stats.kstest(ld, stats.uniform(math.log(ph.d_range_um[0]), math.log(ph.d_range_um[1] / ph.d_range_um[0])).cdf).pvalue > 1e-3
    assert stats.kstest(np.degrees([x.phi_rad for x in D]), stats.uniform(0, 90).cdf).pvalue > 1e-3
    assert stats.kstest([x.theta_rad for x in D], stats.uniform(0, math.pi).cdf).pvalue > 1e-3
    lm = np.log([x.mu_per_um for x in D])
    assert stats.kstest(lm, stats.uniform(math.log(0.3), math.log(10)).cdf).pvalue > 1e-3
    assert stats.kstest([x.cx_um for x in D], stats.uniform(-p / 2, p).cdf).pvalue > 1e-3
    assert stats.kstest([x.cz_um for x in D], stats.uniform(-0.14, 0.28).cdf).pvalue > 1e-3
    assert max(x.phi_rad for x in D) < math.pi / 2


def test_phantom_branch_on_axis():
    draw, rng = PH.draw_replicate(CFG, 3, 4)
    br = PH.phantom_branch(draw, CFG, rng)
    n = CFG.phantom.nodes_each_way
    assert len(br.ids) == 2 * n + 1
    assert_allclose(br.xyz_um[n], [draw.cx_um, draw.cy_um, draw.cz_um], atol=1e-15)
    t = np.diff(br.xyz_um, axis=0)
    assert_allclose(np.linalg.norm(t, axis=1), CFG.phantom.phantom_node_step_um, rtol=1e-12)
    tube = PH.phantom_tube(draw, CFG)
    assert_allclose(t / np.linalg.norm(t, axis=1)[:, None], np.tile(tube.t_hat, (2 * n, 1)), atol=1e-12)
    assert_allclose(br.radius_um, draw.d_um / 2)


def _result(**kw):
    base = dict(node_id=1, type=3, x_um=0.0, y_um=0.0, z_um=0.0, path_um=0.0, reg_verdict="", s_star_um=np.nan,
                dz_star_um=np.nan, k_star=0, z_sub_um=0.0, cx_um=0.0, cy_um=0.0, cz_um=0.0, theta_rad=0.0,
                phi_rad=0.1, steep=False, vertical=False, B_bar=200.0, B_bar_region="block_masked", d_hat_um=1.0,
                mu_hat_per_um=0.5, v0_hat_um=0.0, alpha_hat=0.5, fit_status="converged", flags=(), fit=None,
                focus_F=np.empty(0))
    base.update(kw)
    return NodeResult(**base)


def test_reject_reasons_rule():
    assert PH.reject_reasons(_result()) == []
    assert PH.reject_reasons(_result(flags=("steep", "vertical"), steep=True, vertical=True)) == []   # D-024
    for f in ("faint", "crossing", "stack_edge", "dark", "bbar_few", "profile_nan"):
        assert any(f in x for x in PH.reject_reasons(_result(flags=(f,))))
    assert PH.reject_reasons(_result(fit_status="at_bound", flags=("at_bound",)))
    assert PH.reject_reasons(_result(reg_verdict="ON THE PROCESS (lateral offset +0.00 um)")) == []
    assert PH.reject_reasons(_result(reg_verdict="NOT ON A VISIBLE PROCESS (no peak)"))
    assert PH.reject_reasons(_result(reg_verdict="ALONGSIDE: a ridge +0.80 um to the side"))


def _fixture_rows(n, seed, tau=0.03):
    rng = np.random.default_rng(seed)
    d = np.exp(rng.uniform(math.log(0.2), math.log(4.0), n))
    phi = np.radians(rng.uniform(0, 90, n))
    b = 1 + 0.06 / d + 0.15 * np.sin(phi) ** 2
    ratio = b + rng.normal(0, tau, n)
    rej = (phi > math.radians(45)) & (rng.random(n) < 0.5)
    dark = rng.random(n) < 0.1
    rows = []
    for i in range(n):
        reasons = (["crossing"] if rej[i] else []) + (["dark"] if dark[i] else [])
        rows.append(dict(d_um=d[i], phi_rad=phi[i], meas_phi_rad=phi[i], ratio=ratio[i], in_S=not reasons,
                         reject=";".join(reasons)))
    return rows, b


def test_table_estimator_on_fixture():
    rows, b = _fixture_rows(1500, 1)
    T = TB.fit_table(rows, CFG)
    qd = np.array([0.3, 0.5, 1.0, 2.0, 3.5])
    qp = np.radians([5, 20, 40, 60, 80])
    D, PHI = np.meshgrid(qd, qp)
    btrue = 1 + 0.06 / D + 0.15 * np.sin(PHI) ** 2
    bh, se = T.b_hat(D, PHI), T.se_hat(D, PHI)
    assert np.all(np.abs(bh - btrue) <= 4 * se)
    assert_allclose(T.m_hat(D, PHI), D * bh, rtol=1e-14)
    assert abs(np.median(T.tau_hat(D, PHI)) - 0.03) < 0.1 * 0.03
    # n_eff by hand, Gaussian weights with bandwidths (0.15 in ln d, 5 deg)
    X = T.X_kept
    w = np.exp(-0.5 * (((math.log(1.0) - X[:, 0]) / 0.15) ** 2 + ((math.radians(20) - X[:, 1]) / math.radians(5)) ** 2))
    assert T.n_eff(1.0, math.radians(20)) == pytest.approx(w.sum() ** 2 / (w ** 2).sum(), rel=1e-12)


def test_failure_rate_ignore_dark():
    rows, _ = _fixture_rows(1500, 2)
    T = TB.fit_table(rows, CFG)
    n_dark_only = sum(1 for r in rows if "dark" in r["reject"])
    assert T.X_all.shape[0] == len(rows) - n_dark_only
    assert T.n_rows == len(rows)
    # failure rate by hand at one point, counted rows only
    q = (1.0, math.radians(70))
    X = np.array([[math.log(r["d_um"]), r["phi_rad"]] for r in rows if "dark" not in r["reject"]])
    fail = np.array([not r["in_S"] for r in rows if "dark" not in r["reject"]])
    w = np.exp(-0.5 * (((math.log(q[0]) - X[:, 0]) / 0.15) ** 2 + ((q[1] - X[:, 1]) / math.radians(5)) ** 2))
    assert T.failure_rate(*q) == pytest.approx((w * fail).sum() / w.sum(), rel=1e-12)
    T0 = TB.fit_table(rows, dataclasses.replace(CFG, correction=dataclasses.replace(CFG.correction, failure_rate_ignore=())))
    assert T0.X_all.shape[0] == len(rows)


def test_files_round_trip_and_estimator_refusal(tmp_path):
    rows, _ = _fixture_rows(300, 3)
    p = str(tmp_path / "rows.csv")
    table_io.write_rows(rows, p)
    back = table_io.read_rows(p)
    assert [r["in_S"] for r in back] == [r["in_S"] for r in rows]          # booleans survive the CSV
    T1 = TB.fit_table(rows, CFG)
    T2 = TB.fit_table(back, CFG)
    q = (np.array([0.4, 1.5]), np.radians([10, 50]))
    assert_allclose(T2.b_hat(*q), T1.b_hat(*q), rtol=1e-12)
    stem = str(tmp_path / ("bias_table_" + CFG.signature_hash("estimator")))
    table_io.save_table(T1, stem)
    T3 = table_io.load_table(stem)
    assert np.array_equal(T3.b_hat(*q), T1.b_hat(*q))
    assert np.array_equal(T3.failure_rate(*q), T1.failure_rate(*q))
    T3.check_estimator(CFG)
    other = dataclasses.replace(CFG, measure=dataclasses.replace(CFG.measure, faint_min_dip_gl=7.0))
    with pytest.raises(ValueError):
        T3.check_estimator(other)
