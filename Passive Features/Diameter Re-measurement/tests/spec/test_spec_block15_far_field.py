"""Spec-derived tests of Block 15 (the far-field calibration) -- integration
tester run D4, 2026-11-13, commit 1555036. Written from specs/SPEC.md (15.1-15.6,
10.G as it reaches the far scan, 19.6-19.7 as they reach the far fit), not from
the code. Expected values come from spec_helpers_D4 (an independent iterative
core fit, the plane selection, the model SE, F1 data) or by hand.

Run (from the workstream directory):
    python -m pytest tests/spec/test_spec_block15_far_field.py -q
"""
from __future__ import annotations

import dataclasses
import json
import math
import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

import spec_helpers_D4 as D4

from allen_diameter.analysis import far_field as FF
from allen_diameter.loading import calibration_io

NAN = float("nan")
SIGMA0, GAMMA = 0.603, 0.79        # the default table's sigma_r(0.84 um) and continuation slope (Block 4)


def far_v(cfg):
    """15.1: the far line's offsets, every profile_step_um to +-far_profile_half_um."""
    step = cfg.measure.profile_step_um
    n = int(round(cfg.calibration.far_profile_half_um / step))
    return np.arange(-n, n + 1) * step


def core(cfg, v, w, depth, v_peak, sigma_p):
    cal = cfg.calibration
    return FF.core_fit_iterative(v, w, depth, v_peak, sigma_p, cal, cfg.measure)


def ref_core(cfg, v, w, depth, v_peak, sigma_p):
    cal = cfg.calibration
    return D4.ref_core_fit_iterative(v, w, depth, v_peak, sigma_p, cal.core_fit_fwhm_factor,
                                     cal.far_core_width_bounds, cal.min_window_samples, cal.far_core_max_passes,
                                     cal.far_core_stop_tol)


# ------------------------------------------------------------ 15.2 ----------

def test_core_fit_exact_gaussian_and_stop_at_pass_two():
    """O15.1: on a noise-free sampled Gaussian omega = s^2 to 1e-9 and the loop stops at pass 2."""
    cfg = D4.cfg_with()
    v = far_v(cfg)
    s, a, vc = 0.8, 40.0, 0.1
    w = D4.gauss_dip(v, a, vc, s)
    n0 = int(np.argmin(np.abs(v - vc)))
    out = core(cfg, v, w, a, float(v[n0]), 1.0)
    assert out["status"] == "ok"
    assert_allclose(out["omega_um2"], s ** 2, rtol=1e-9)
    assert out["passes"] == 2


@pytest.mark.parametrize("seed, s, a, sigma_p", [(1, 0.8, 20.0, 1.0), (2, 1.5, 12.0, 1.2), (3, 2.6, 8.0, 3.0)])
def test_core_fit_matches_independent_iteration_and_its_se(seed, s, a, sigma_p):
    """O15.2-3: on noisy dips (white noise SD 1 gl) the statistic equals an independent implementation of
    15.2 -- the same windows, passes, status, omega and SE(omega) = 2 s RMS [(J^T J)^-1]^(1/2)_ss -- to
    1e-6 relative (both solvers converge to the same least-squares minimum, tolerances 1e-12 and 1e-14)."""
    cfg = D4.cfg_with()
    v = far_v(cfg)
    rng = np.random.default_rng(seed)
    w = D4.gauss_dip(v, a, 0.15, s) + rng.normal(0.0, 1.0, v.size)
    n0 = int(np.argmin(np.abs(v - 0.15)))
    got = core(cfg, v, w, a, float(v[n0]), sigma_p)
    ref = ref_core(cfg, v, w, a, float(v[n0]), sigma_p)
    assert got["status"] == ref["status"]
    assert got["passes"] == ref["passes"] and got["n_win"] == ref["n_win"]
    assert_allclose(got["omega_um2"], ref["omega"], rtol=1e-6)
    assert_allclose(got["se_omega_um2"], ref["se_omega"], rtol=1e-6)
    assert_allclose(got["rms_gl"], ref["rms"], rtol=1e-6)
    assert_allclose(got["amp_gl"], ref["a"], rtol=1e-6)


def test_core_fit_statuses():
    """O15.3: at_bound (a dip six prior widths wide), few_samples (a window shorter than min_window_samples),
    not_converged (no stop allowed within the passes)."""
    cfg = D4.cfg_with()
    v = far_v(cfg)
    n0 = int(np.argmin(np.abs(v)))
    wide = D4.gauss_dip(v, 30.0, 0.0, 3.0)
    assert core(cfg, v, wide, 30.0, 0.0, 0.5)["status"] == "at_bound"
    few = D4.cfg_with(calibration=dict(min_window_samples=60))
    assert core(few, v, D4.gauss_dip(v, 30.0, 0.0, 0.3), 30.0, float(v[n0]), 0.3)["status"] == "few_samples"
    nc = D4.cfg_with(calibration=dict(far_core_max_passes=3, far_core_stop_tol=0.0))
    rng = np.random.default_rng(9)
    noisy = D4.gauss_dip(v, 20.0, 0.0, 0.8) + rng.normal(0.0, 1.0, v.size)
    out = core(nc, v, noisy, 20.0, float(v[n0]), 1.0)
    assert out["status"] == "not_converged" and out["passes"] == 3


def test_far_plane_statistics_no_dip_and_line_end():
    """O15.3: a flat flattened profile is no_dip; the line-end mean is the mean of w over |v - v_c| >= 2 FWHM
    when both sides hold >= 4 samples (here w carries +1 gl beyond 2.5 FWHM: the mean is computed by hand over
    the zone, 1e-6 relative for the Gaussian tail's 1e-5 a and the fitted centre); NaN when a side is short."""
    cfg = D4.cfg_with()
    m, cal = cfg.measure, cfg.calibration
    v = far_v(cfg)
    B0 = 180.0
    flat = FF.far_plane_statistics(v, np.full(v.size, B0), B0, 1.0, m, cal)
    assert flat["status"] == "no_dip" and np.isnan(flat["omega_um2"])
    s, a = 0.8, 30.0
    fwhm = D4.FW * s
    w = D4.gauss_dip(v, a, 0.0, s) + np.where(np.abs(v) >= 2.5 * fwhm, 1.0, 0.0)
    st = FF.far_plane_statistics(v, B0 - w, B0, 1.0, m, cal)
    assert st["status"] == "ok"
    zone = np.abs(v - st["v_c_um"]) >= cal.far_line_end_fwhm_factor * D4.FW * st["sigma_um"]
    assert_allclose(st["line_end_gl"], np.mean(w[zone]), rtol=1e-6)
    wide = D4.gauss_dip(v, a, 0.0, 2.5)          # 2 FWHM = 11.8 um > the line's 10 um: no zone
    st2 = FF.far_plane_statistics(v, B0 - wide, B0, 2.0, m, cal)
    assert st2["status"] == "ok" and math.isnan(st2["line_end_gl"])


# ------------------------------------------------------------ 15.3 ----------

def mixed_far_scans(cfg):
    """Six far scans: F1 widths, varied amplitudes and RMS, some not_converged / poor_fit / at_bound planes,
    one node without an area plane, one left with few planes by its RMS."""
    rng = np.random.default_rng(31)
    J = 6
    k_star = np.array([10, 20, 30, 40, 50, 60])
    z_p = k_star * D4.DZ + rng.uniform(-0.1, 0.1, J)
    scans = D4.f1_scans(rng.uniform(0, 0.04, J), np.zeros(J), z_p, k_star, SIGMA0, GAMMA, amp=30.0, rms=1.0)
    out = []
    for j, sc in enumerate(scans):
        n = sc.ks.size
        st = np.array(sc.status, dtype=object)
        amp = rng.uniform(15.0, 40.0, n)
        rms = rng.uniform(0.5, 1.5, n)
        if j == 1:
            st[rng.choice(n, 4, replace=False)] = "not_converged"
            st[rng.choice(n, 3, replace=False)] = "poor_fit"
        if j == 2:
            off = np.abs(sc.ks - sc.k_star)
            st[(off >= 4) & (off <= 6)] = "at_bound"            # no area plane: dropped
        if j == 3:
            rms = rms * 20.0                                    # the SNR rule keeps (almost) no plane
        out.append(dataclasses.replace(sc, status=tuple(st), amp_gl=amp, rms_gl=rms))
    return out


def test_select_far_planes_matches_spec_rule():
    """O15.4: the kept nodes, A_j, RMS_j, the planes kept, the dropped nodes and the counts equal 15.3 written
    from the spec (A_j to 1e-12 relative: the same medians of the same numbers)."""
    cfg = D4.cfg_with()
    r = cfg.renderer
    scans = mixed_far_scans(cfg)
    sel = FF.select_far_planes(scans, cfg)
    kept, dropped, n_nc, n_snr = D4.ref_select_far(scans, cfg.calibration, r.kernel_table_delta_um,
                                                   r.kernel_table_sigma_um, r.kernel_continuation_slope)
    assert list(sel.node_ids) == list(kept)
    assert set(sel.dropped) == dropped
    assert sel.n_not_converged == n_nc and sel.n_snr_dropped == n_snr
    for j, nid in enumerate(sel.node_ids):
        A, rms, ks = kept[int(nid)]
        assert_allclose(sel.area_gl_um[j], A, rtol=1e-12)
        assert_allclose(sel.rms_gl[j], rms, rtol=1e-12)
        assert_array_equal(np.sort(sel.obs_k[sel.obs_node == j]), np.sort(ks))
        sc = [s for s in scans if s.node_id == nid][0]
        want = {int(k): w for k, w in zip(sc.ks, sc.omega_um2)}
        assert_allclose(sel.omega_um2[sel.obs_node == j], [want[int(k)] for k in sel.obs_k[sel.obs_node == j]])
    assert {3, 4} <= dropped                      # node 3 (j = 2): no area plane; node 4 (j = 3): too few planes


def test_select_far_planes_refuses_near_scans():
    cfg = D4.cfg_with()
    scans = mixed_far_scans(cfg)
    with pytest.raises(ValueError):
        FF.select_far_planes([dataclasses.replace(scans[0], set="near")], cfg)


# ------------------------------------------------------------ 15.4 ----------

def f1_selection(J=8, seed=41, gamma=GAMMA, sigma0=SIGMA0, s_true=None, c_true=None, z_shift=0.0,
                 gamma_per_node=None, node_ids=None):
    """A FarSelection built by hand over |k - k*| >= 5 (both sides) from exact (F1) widths; the container
    only (no code under test)."""
    rng = np.random.default_rng(seed)
    k_star = np.arange(J) * 40
    z_p = k_star * D4.DZ + rng.uniform(-0.1, 0.1, J) + z_shift
    s = rng.uniform(-0.1, 0.1, J) if s_true is None else np.asarray(s_true, float)
    c = rng.uniform(-0.02, 0.05, J) if c_true is None else np.asarray(c_true, float)
    obs_node, obs_k, z, om = [], [], [], []
    for j in range(J):
        ks = k_star[j] + np.r_[np.arange(-16, -4), np.arange(5, 17)]
        zk = ks * D4.DZ
        g = gamma if gamma_per_node is None else gamma_per_node[j]
        obs_node.append(np.full(ks.size, j)); obs_k.append(ks); z.append(zk)
        om.append(D4.f1_omega(z_p[j], s[j], zk, c[j], sigma0, g))
    cat = [np.concatenate(x) for x in (obs_node, obs_k, z, om)]
    ids = np.arange(1, J + 1) if node_ids is None else np.asarray(node_ids)
    sel = FF.FarSelection(ids, np.full(J, 40.0), np.full(J, 1.0), z_p, cat[0].astype(int), cat[1].astype(int),
                          cat[2], cat[3], np.full(cat[3].size, 1.0), np.full(cat[3].size, 1.0), {}, 0, 0,
                          np.full(J, 24))
    return sel, s, c


@pytest.mark.parametrize("weights", ["none", "model"])
def test_far_fit_recovers_exact_f1(weights):
    """O15.5: gamma, c and s to 1e-8 by the unweighted and the IRLS fit (the spec's tolerance)."""
    cfg = D4.cfg_with()
    sel, s, c = f1_selection()
    fit = FF.fit_far_slope(sel, SIGMA0, cfg, weights=weights)
    assert fit.success and not fit.at_bound
    assert_allclose(fit.gamma, GAMMA, atol=1e-8)
    assert_allclose(fit.c_um2, c, atol=1e-8)
    assert_allclose(fit.s_um, s, atol=1e-8)


def test_far_fit_per_sign_and_quadratic_on_exact_f1():
    """O15.5: the per-sign fit returns gamma_+ = gamma_- (1e-8) and the quadratic comparison gamma_2 = 0
    (1e-6, the spec's figure)."""
    cfg = D4.cfg_with()
    sel, _s, _c = f1_selection()
    ps = FF.fit_far_slope(sel, SIGMA0, cfg, per_sign=True)
    assert_allclose([ps.gamma, ps.gamma_minus], [GAMMA, GAMMA], atol=1e-8)
    q = FF.fit_far_slope(sel, SIGMA0, cfg, quadratic=True)
    assert abs(q.gamma2) <= 1e-6
    assert_allclose(q.gamma, GAMMA, atol=1e-6)


def test_far_fit_invariances():
    """O15.6: a constant on every omega moves only the c_j; shifting every z_p by u moves the s_j by -u
    (1e-8, as O15.5: the same exact problem in shifted coordinates)."""
    cfg = D4.cfg_with()
    sel, s, c = f1_selection(s_true=np.linspace(-0.08, 0.08, 8))
    f0 = FF.fit_far_slope(sel, SIGMA0, cfg)
    f1 = FF.fit_far_slope(dataclasses.replace(sel, omega_um2=sel.omega_um2 + 0.03), SIGMA0, cfg)
    assert_allclose(f1.c_um2, f0.c_um2 + 0.03, atol=1e-8)
    assert_allclose(f1.gamma, f0.gamma, atol=1e-8)
    assert_allclose(f1.s_um, f0.s_um, atol=1e-8)
    u = 0.05
    f2 = FF.fit_far_slope(dataclasses.replace(sel, z_p_um=sel.z_p_um + u), SIGMA0, cfg)
    assert_allclose(f2.s_um, f0.s_um - u, atol=1e-8)
    assert_allclose(f2.gamma, f0.gamma, atol=1e-8)


def test_far_fit_at_bound_fails():
    """O15.5: gamma_hat at gamma_max (data at 6 > 5) fails the fit; so does gamma_hat at 0."""
    cfg = D4.cfg_with()
    sel, _s, _c = f1_selection(gamma=6.0)
    fit = FF.fit_far_slope(sel, SIGMA0, cfg)
    assert fit.at_bound and not fit.success
    sel0, _s, _c = f1_selection(gamma=-0.05, sigma0=1.5)     # widths shrinking with defocus
    fit0 = FF.fit_far_slope(sel0, 1.5, cfg)
    assert fit0.at_bound and not fit0.success


def test_far_fit_refuses_one_node():
    cfg = D4.cfg_with()
    sel, _s, _c = f1_selection(J=2)
    with pytest.raises(ValueError):
        FF.fit_far_slope(sel.subset([0]), SIGMA0, cfg)


@pytest.mark.parametrize("amp, sigma, rms", [(20.0, 0.6, 1.0), (5.0, 1.5, 2.0), (40.0, 3.0, 0.7), (8.0, 4.5, 1.0)])
def test_se_model_is_the_inverse_fisher_information(amp, sigma, rms):
    """O15.7: 2 w RMS [(J^T J)^-1]^(1/2)_ss on the samples every step over +-FWHM cut at the line's half
    (1e-10 relative: the same closed form evaluated independently)."""
    cfg = D4.cfg_with()
    step, half = cfg.measure.profile_step_um, cfg.calibration.far_profile_half_um
    got = FF.se_model(np.array([amp]), np.array([sigma]), np.array([rms]), step, half)
    assert_allclose(got[0], D4.ref_se_model(amp, sigma, rms, step, half), rtol=1e-10)


def test_anchor_sensitivity_sign_and_size():
    """O15.8 (oracle (d)): on F1 data with noise d gamma / d sigma_0 is negative and below 1 per um."""
    cfg = D4.cfg_with()
    sel, _s, _c = f1_selection(J=10, seed=43)
    rng = np.random.default_rng(44)
    noisy = dataclasses.replace(sel, omega_um2=sel.omega_um2 + rng.normal(0, 0.02, sel.omega_um2.size))
    dg = FF.anchor_sensitivity(noisy, SIGMA0, cfg)
    assert dg < 0 and abs(dg) < 1.0


def test_far_fit_summary_interval_quantile():
    """O15.9: SE^2 = SE_boot^2 + (dgamma/dsigma0)^2 SE(sigma0)^2 and the interval gamma +- t_{df, 0.975} SE with
    df = J - 1 under the node bootstrap and C - 1 when the reported unit is the cell (scipy.stats.t)."""
    cfg = D4.cfg_with()
    sel, _s, _c = f1_selection(J=8)
    fit = FF.fit_far_slope(sel, SIGMA0, cfg)
    se_b, dg, se0 = 0.03, -0.4, 0.02
    se = math.sqrt(se_b ** 2 + dg ** 2 * se0 ** 2)
    out = FF.far_fit_summary(sel, fit, None, None, se_b, 100, dg, SIGMA0, se0, cfg)
    t7 = stats.t.ppf(0.975, 7)
    assert_allclose(out["se"], se, rtol=1e-12)
    assert out["t_df"] == 7
    assert_allclose(out["interval"], [fit.gamma - t7 * se, fit.gamma + t7 * se], rtol=1e-12)
    boot = dict(rule="larger", unit="cell", n_cells=4, few_cells=True, se_node=0.02, se_cell=0.03, n_ok_node=100,
                n_ok_cell=100)
    out2 = FF.far_fit_summary(sel, fit, None, None, se_b, 100, dg, SIGMA0, se0, cfg, boot=boot)
    t3 = stats.t.ppf(0.975, 3)
    assert out2["t_df"] == 3
    assert_allclose(out2["interval"], [fit.gamma - t3 * se, fit.gamma + t3 * se], rtol=1e-12)
    json.dumps(out2, allow_nan=False)


def test_far_fit_and_selection_deterministic():
    """O15.11: identical calls give identical results."""
    cfg = D4.cfg_with()
    scans = mixed_far_scans(cfg)
    a, b = FF.select_far_planes(scans, cfg), FF.select_far_planes(scans, cfg)
    assert_array_equal(a.omega_um2, b.omega_um2) and a.dropped == b.dropped
    sel, _s, _c = f1_selection()
    f1, f2 = FF.fit_far_slope(sel, SIGMA0, cfg), FF.fit_far_slope(sel, SIGMA0, cfg)
    assert f1.gamma == f2.gamma and np.array_equal(f1.c_um2, f2.c_um2)


def test_bootstrap_gamma_uses_its_generator():
    """Silent-failure hunt: the bootstrap's draws repeat for one seed and change with the seed (the generator
    is used); a node resample is rng.integers(0, J, J)."""
    cfg = D4.cfg_with()
    sel, _s, _c = f1_selection(J=8, seed=45)
    rng = np.random.default_rng(46)
    noisy = dataclasses.replace(sel, omega_um2=sel.omega_um2 + rng.normal(0, 0.01, sel.omega_um2.size))
    s1 = FF.bootstrap_gamma(noisy, SIGMA0, cfg, np.random.default_rng(3), n_resamples=12)
    s2 = FF.bootstrap_gamma(noisy, SIGMA0, cfg, np.random.default_rng(3), n_resamples=12)
    s3 = FF.bootstrap_gamma(noisy, SIGMA0, cfg, np.random.default_rng(4), n_resamples=12)
    assert_array_equal(s1[2], s2[2])
    assert not np.array_equal(s1[2], s3[2])
    assert s1[1] == 12 and math.isfinite(s1[0])
    assert_allclose(s1[0], np.std(s1[2], ddof=1), rtol=1e-12)


# ------------------------------------------------------------ 15.6 ----------

def write_near_far(tmp, n_near, n_far, seed=50):
    """Exact near scans generated from the default table (so the anchor is sigma_r(0.84) = 0.603 um) and exact
    F1 far scans at gamma 0.79, in one scans file."""
    cfg = D4.cfg_with()
    r = cfg.renderer
    sig = D4.ref_sigma_r(np.arange(5) * D4.DZ, r.kernel_table_delta_um, r.kernel_table_sigma_um,
                         r.kernel_continuation_slope)
    G = sig ** 2 - r.sigma_r0_um ** 2
    rng = np.random.default_rng(seed)
    k = rng.integers(0, 40, n_near)
    z_ax = k * D4.DZ + rng.uniform(-0.1, 0.1, n_near)
    near = D4.eq4_scans(G, rng.uniform(0, 0.03, n_near), z_ax, z_ax + rng.uniform(-0.03, 0.03, n_near), k)
    kf = 100 + np.arange(n_far) * 40
    zp = kf * D4.DZ + rng.uniform(-0.1, 0.1, n_far)
    far = D4.f1_scans(rng.uniform(0, 0.03, n_far), rng.uniform(-0.05, 0.05, n_far), zp, kf, sig[3], GAMMA,
                      node_ids=list(range(1001, 1001 + n_far)), amp=60.0, rms=0.5)
    path = os.path.join(tmp, "scans.json")
    calibration_io.save_scans(near + far, path, dict(source="test"))
    cj = os.path.join(tmp, "cfg.json")
    full = json.loads(cfg.to_json())
    full["calibration"] = dict(bootstrap_resamples=20)
    with open(cj, "w") as f:
        json.dump(full, f)
    return path, cj, sig


def test_cli_fit_far_fit_production_kernel(tmp_path, capsys):
    """O15.10 (15.6, 19.2): with both sets sufficient the far-fit file holds the production kernel -- the near
    table cut at 0.84 um, continuation linear with slope gamma_hat -- which validates; gamma_hat = 0.79 on exact
    data (1e-6: the anchor is the fitted table's, itself recovered to 1e-9)."""
    import calibrate_kernel as CK
    from allen_diameter.config import config_from_dict
    scans, cj, sig = write_near_far(str(tmp_path), 12, 12)
    cal, ff = str(tmp_path / "C.json"), str(tmp_path / "F.json")
    CK.main(["fit", "--scans", scans, "--out", cal, "--config-json", cj, "--seed", "1"])
    CK.main(["far-fit", "--scans", scans, "--calibration", cal, "--out", ff, "--config-json", cj, "--seed", "1"])
    C, F = calibration_io.load_calibration(cal), calibration_io.load_calibration(ff)
    assert_allclose(C["anchor"]["sigma0_um"], sig[3], atol=1e-9)
    assert_allclose(F["far_fit"]["gamma"], GAMMA, atol=1e-6)
    k = F["kernel"]
    assert_allclose(k["kernel_table_delta_um"], [0.0, 0.28, 0.56, 0.84], atol=1e-12)
    assert k["kernel_continuation"] == "linear"
    assert k["kernel_continuation_slope"] == F["far_fit"]["gamma"]
    with open(cj) as f:
        full = json.load(f)
    full["renderer"].update(kernel_table_delta_um=k["kernel_table_delta_um"],
                            kernel_table_sigma_um=k["kernel_table_sigma_um"], sigma_r0_um=k["sigma_r0_um"],
                            kernel_continuation="linear", kernel_continuation_slope=k["kernel_continuation_slope"])
    config_from_dict(full).validate()
    assert C["meta"]["sufficiency"]["sufficient"] and F["meta"]["sufficiency"]["far"]["sufficient"]


def test_cli_insufficient_sets_give_no_production_kernel(tmp_path, capsys):
    """O19.3 / 19.2: 9 near nodes (< 10) -> fit records sufficient False and prints INSUFFICIENT; far-fit
    writes no production kernel."""
    import calibrate_kernel as CK
    scans, cj, _sig = write_near_far(str(tmp_path), 9, 12)
    cal, ff = str(tmp_path / "C.json"), str(tmp_path / "F.json")
    CK.main(["fit", "--scans", scans, "--out", cal, "--config-json", cj])
    out = capsys.readouterr().out
    assert "INSUFFICIENT: 9 near-field nodes < 10" in out
    CK.main(["far-fit", "--scans", scans, "--calibration", cal, "--out", ff, "--config-json", cj])
    C, F = calibration_io.load_calibration(cal), calibration_io.load_calibration(ff)
    assert C["meta"]["sufficiency"]["sufficient"] is False
    assert "kernel" not in F and "insufficient" in F["kernel_error"]
