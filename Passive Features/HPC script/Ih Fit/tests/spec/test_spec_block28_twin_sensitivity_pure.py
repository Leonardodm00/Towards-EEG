"""Spec-derived tests of Block 28 (run_twin_recovery.py: the bundle noise of 2.10
(d), the scatter tables, the merged tables, the truth string, the new extras
columns) that need no NEURON. Integration tester, 2026-11-25.

Oracles: the entry's definition of a bundle's noise written out through Block
10's own generators (`twin_noise.ls_sweep_noise`, `ss_sweep_noise`, the functions
the entry names) on a hand-built protocol and twins' table; Block 26's
`predicted_scatter` on residuals assembled here; file-system invariants of the
merge (one twin_id per twin, numeric order).
"""
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import run_twin_recovery as T
import twin_noise as TN
import twin_table as TT

SPEC_ID = 555
TWIN = 900000123
FS_SS, FS_LS = 50e3, 10e3
N_W = 600                       # samples of a stored SS window
PERIOD = 10025


def _table():
    rows = []
    for s, (sf, tilt) in {11: (0.10, 0.02), 12: (0.12, -0.03)}.items():
        rows.append(dict(specimen_id=SPEC_ID, protocol="SS", sweep_number=s, sigma_f_mV=sf, sigma_m_mV=0.05,
                         tau_m_s=0.004, sigma_s_mV=0.2, tau_s_s=0.3, tilt_mV_per_s=tilt))
    for s, (sf, tilt) in {54: (0.08, 0.05), 55: (0.09, 0.01), 56: (0.11, -0.02)}.items():
        rows.append(dict(specimen_id=SPEC_ID, protocol="LS", sweep_number=s, sigma_f_mV=sf, sigma_m_mV=0.04,
                         tau_m_s=0.006, sigma_s_mV=0.25, tau_s_s=0.5, tilt_mV_per_s=tilt))
    return pd.DataFrame(rows)


def _protocol():
    # sweep 11: positions 0..5 alternating dep/hyp; sweep 12: positions 0,2,3,5 (two dropped)
    sw = np.array([11] * 6 + [12] * 4)
    pos = np.array([0, 1, 2, 3, 4, 5, 0, 2, 3, 5])
    dep = (pos % 2 == 0)
    return types.SimpleNamespace(ss_sweep_number=sw, ss_position=pos, ss_polarity_is_dep=dep,
                                 ss_t_s=(np.arange(N_W) - 100) / FS_SS, ss_fs_Hz=FS_SS, ss_period_samples=PERIOD,
                                 ls_sweep_number=np.array([54, 55, 56]), ls_n_samples=np.array([3000, 3000, 2500]),
                                 ls_fs_Hz=FS_LS)


def _brief(dep, n_rep, sweeps=()):
    return types.SimpleNamespace(t=_protocol().ss_t_s.copy(), amplitude_pA=200.0 if dep else -200.0,
                                 n_repeats_averaged=n_rep, sweep_numbers=list(sweeps))


def _long(sweeps, n=3000):
    return types.SimpleNamespace(t=np.arange(n) / FS_LS, amplitude_pA=-50.0, n_repeats_averaged=len(sweeps),
                                 sweep_numbers=list(sweeps))


def test_long_bundle_noise_is_the_mean_of_its_sweeps_draws():
    tab, prot = _table(), _protocol()
    b = _long([54, 55])
    got = T.bundle_noise(b, False, prot, tab, SPEC_ID, twin_id=TWIN, base_seed=1000003)
    draws = []
    for s in (54, 55):
        law, tilt = TT.sweep_law(tab, SPEC_ID, "LS", s)
        draws.append(TN.ls_sweep_noise(law, tilt, 1.0 / FS_LS, 3000, base_seed=1000003, twin_id=TWIN, sweep_number=s))
    assert_allclose(got, (draws[0] + draws[1]) / 2.0, rtol=0, atol=1e-15)
    assert got.shape == b.t.shape and got.dtype == np.float64
    # deterministic; another realisation differs; another twin differs
    assert_array_equal(got, T.bundle_noise(b, False, prot, tab, SPEC_ID, twin_id=TWIN, base_seed=1000003))
    assert not np.array_equal(got, T.bundle_noise(b, False, prot, tab, SPEC_ID, twin_id=TWIN, base_seed=1000004))
    assert not np.array_equal(got, T.bundle_noise(b, False, prot, tab, SPEC_ID, twin_id=TWIN + 1, base_seed=1000003))


@pytest.mark.parametrize("dep", [True, False])
def test_brief_bundle_noise_is_the_mean_over_the_polarity_windows(dep):
    tab, prot = _table(), _protocol()
    rows = []
    for s in (11, 12):
        m = prot.ss_sweep_number == s
        law, tilt = TT.sweep_law(tab, SPEC_ID, "SS", s)
        full = TN.ss_sweep_noise(law, tilt, 1.0 / FS_SS, N_W, prot.ss_position[m], PERIOD, base_seed=1000000,
                                 twin_id=TWIN, sweep_number=s)
        rows.append(full[prot.ss_polarity_is_dep[m] == dep])
    rows = np.concatenate(rows)
    n_rep = rows.shape[0]
    got = T.bundle_noise(_brief(dep, n_rep, (11, 12)), True, prot, tab, SPEC_ID, twin_id=TWIN, base_seed=1000000)
    assert_allclose(got, rows.mean(axis=0), rtol=0, atol=1e-15)
    cache = {}
    a = T.bundle_noise(_brief(dep, n_rep), True, prot, tab, SPEC_ID, twin_id=TWIN, base_seed=1000000, ss_cache=cache)
    assert_array_equal(a, got)
    assert set(cache) == {11, 12}


def test_grid_mismatch_stops():
    tab, prot = _table(), _protocol()
    bad = [(_long([54], n=2999), False), (_long([56], n=3000), False), (_long([99]), False), (_long([]), False),
           (_brief(True, 4), True)]                     # 5 dep windows exist, the bundle says 4
    sh = _brief(True, 5); sh.t = sh.t + 1e-6
    bad.append((sh, True))
    for b, br in bad:
        with pytest.raises(T.BundleGridError):
            T.bundle_noise(b, br, prot, tab, SPEC_ID, twin_id=TWIN, base_seed=1000000)


def test_noise_realisations_seeds():
    tab, prot = _table(), _protocol()
    bs = [_long([54]), _brief(False, 5), _long([55, 56][:1])]
    got = list(T.noise_realisations(bs, [False, True, False], [2, 0], prot, tab, SPEC_ID, twin_id=TWIN, n_noise=3))
    assert len(got) == 3 and all(len(g) == 2 for g in got)
    for k in range(3):
        assert_array_equal(got[k][0], T.bundle_noise(bs[2], False, prot, tab, SPEC_ID, twin_id=TWIN,
                                                     base_seed=1000000 + k))
        assert_array_equal(got[k][1], T.bundle_noise(bs[0], False, prot, tab, SPEC_ID, twin_id=TWIN,
                                                     base_seed=1000000 + k))


def test_sensitivity_theta_of_reads_the_truth_columns():
    row = pd.Series(dict(specimen_id=1, cm_true=0.91, rm_true=23456.5, ra_true=150.25, ih_gihbar_S_cm2=1e-5,
                         ih_dvh_mV=-2.5, ih_kappa_tau=0.8))
    s = T.sensitivity_theta_of(row, "ih6_d207")
    d = dict(kv.split("=") for kv in s.split(","))
    assert list(d) == ["Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau"]
    assert [float(d[k]) for k in d] == [0.91, 23456.5, 150.25, 1e-5, -2.5, 0.8]
    assert list(dict(kv.split("=") for kv in T.sensitivity_theta_of(row, "ih5").split(","))) == \
        ["Cm", "Rm", "Ra", "gbar", "kappa_tau"]
    assert list(dict(kv.split("=") for kv in T.sensitivity_theta_of(row, "passive_fullstep").split(","))) == \
        ["Cm", "Rm", "Ra"]
    row2 = row.copy(); row2["ih_kappa_tau"] = np.nan
    with pytest.raises(ValueError):
        T.sensitivity_theta_of(row2, "ih5")


def test_gather_sensitivity_numeric_order_and_twin_id(tmp_path):
    out = tmp_path / "run"
    for tid in (10, 9, 100):
        d = T.sens_dir_of(out, tid)
        d.mkdir(parents=True)
        pd.DataFrame(dict(twin_id=[tid, tid], weighting=["W0", "W1"], sd_lin_Cm=[0.1 * tid, 0.2])).to_csv(
            d / "identifiability.csv", index=False)
        pd.DataFrame(dict(specimen_id=[tid] * 3, bundle=[0, 1, 2], max_abs_mV=[0.0, 1e-3, 2e-3])).to_csv(
            d / ("truth_trace_check_%d.csv" % tid), index=False)
    (out / "sensitivity" / "twin_x").mkdir()           # not a twin directory: ignored
    n = T.gather_sensitivity(out)
    assert n == {"identifiability": 3, "truth_trace_check": 3}
    idf = pd.read_csv(out / "identifiability.csv")
    assert idf["twin_id"].tolist() == [9, 9, 10, 10, 100, 100]
    ttc = pd.read_csv(out / "truth_trace_check.csv")
    assert ttc.columns[0] == "twin_id" and ttc["twin_id"].tolist() == [9] * 3 + [10] * 3 + [100] * 3
    assert T.sens_dir_of(out, 9) == out / "sensitivity" / "twin_9"
    empty = tmp_path / "none"
    empty.mkdir()
    assert T.gather_sensitivity(empty) == {"identifiability": 0, "truth_trace_check": 0}
    assert not (empty / "identifiability.csv").exists()


def test_extras_columns_follow_the_run_arm():
    for arm, names in (("ih6_d207", ["Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau"]),
                       ("ih5", ["Cm", "Rm", "Ra", "gbar", "kappa_tau"])):
        cols = T.extras_columns({"options": {"ih_arm": arm}})
        assert cols[-len(names):] == ["on_bound_" + n for n in names]
        for c in ("l2_after_search", "l2_after_refine", "refine_n_converged", "refine_n_failed",
                  "sqrt_l2_at_estimate", "sqrt_l2_at_truth"):
            assert c in cols


def test_parser_defaults_of_the_new_options():
    p = T.build_parser()
    a = p.parse_args(["sensitivity", "--run-dir", "/r", "--twin", "5"])
    assert (a.n_noise, a.h, a.check_h, a.n_workers) == (200, 1e-3, 1e-2, 1)
    assert T.REFINE_DEFAULT == "trf" and T.REFINE_MAX_NFEV_DEFAULT == 30 and T.NOISE_BASE_SEED_ROOT == 1000000
