# -*- coding: utf-8 -*-
"""
smoke_run_biological_fit.py
===========================

NEURON-free smoke test for the PURE helpers of run_biological_fit.py:
    parse_float_list / parse_window        (CLI list/window parsing)
    pick_winning_tau_w                     (per-cell tau_w selection)
    select_phase3_subset                   (subset grammar over a sid LIST)
    compute_absolute_gate                  (absolute-mV gate core, NEW)
    results_to_dataframe                   (result serialisation + new columns)

Two independent controls each. Run:  python smoke_run_biological_fit.py

The NEURON-side pieces (build_neuron_model, fit_one_cell, the sweep, Phase 2.5,
Phase 3, and the simulation half of the absolute gate) cannot be exercised
without NEURON + real archives; validate those with the 1-cell dry run in the
README (a --max-cells 1 --phase3-subset none job on one specimen).
"""

from collections import namedtuple

import numpy as np
import pandas as pd

from run_biological_fit import (
    parse_float_list, parse_window, pick_winning_tau_w,
    select_phase3_subset, compute_absolute_gate, results_to_dataframe,
)

# duck-typed fakes
Prof = namedtuple("Prof", ["tau_w_ms", "hw_rho", "kappa", "bias_log"])
Res = namedtuple("Res", [
    "specimen_id", "layer", "dendrite_type", "F", "fit_target",
    "cm_uF_per_cm2", "rm_Ohm_cm2", "ra_Ohm_cm", "validation_status",
    "validation_status_relative", "train_rel_loss",
    "train_rmsd_abs_mV", "valid_rmsd_abs_mV", "valid_to_train_ratio_abs",
    "tau_w_chosen_ms",
])


def _fake_classify(train_rmsd, valid_rmsd, k_good, k_fail, train_fail_mV,
                   valid_rmsd_good_mV=0.2):
    """Standalone re-implementation of the monolith's _classify_fit thresholds,
    so the gate's aggregation logic can be tested NEURON-free with a KNOWN
    classifier (the real gate injects mono._classify_fit)."""
    if not np.isfinite(train_rmsd) or train_rmsd > train_fail_mV:
        return "failed"
    if not np.isfinite(valid_rmsd):
        return "failed"
    if valid_rmsd <= valid_rmsd_good_mV:
        return "good"
    ratio = valid_rmsd / max(train_rmsd, 1e-9)
    if ratio <= k_good:
        return "good"
    if ratio <= k_fail:
        return "to_refine"
    return "failed"


def test_parsers():
    assert parse_float_list("2.0,5.0,10.0") == [2.0, 5.0, 10.0]
    assert parse_float_list("5.0") == [5.0]                       # single point
    assert parse_window("0.5,100.0") == (0.5, 100.0)
    try:
        parse_window("1,2,3"); raise AssertionError("should reject 3-tuple")
    except ValueError:
        pass
    print("[smoke] parse_float_list / parse_window (single point + bad input)  PASS")


def test_pick_winner():
    grid = [3.0, 5.0, 7.0]
    # control 1: smallest finite HW_rho wins
    profs = [Prof(3.0, 9.9, 1.0, None), Prof(5.0, 3.3, 2.0, None),
             Prof(7.0, 7.0, 1.5, None)]
    tau, reason, w = pick_winning_tau_w(profs, grid)
    assert tau == 5.0 and reason == "sharpest_hw_rho" and w.tau_w_ms == 5.0
    # control 2: single-point grid -> that point wins trivially
    one = [Prof(5.0, 0.4, 3.0, None)]
    assert pick_winning_tau_w(one, [5.0])[0] == 5.0
    # extra: all-inf -> flagged middle-of-grid fallback, winner None
    profs_inf = [Prof(t, np.inf, 0.0, None) for t in grid]
    tau2, reason2, w2 = pick_winning_tau_w(profs_inf, grid)
    assert tau2 == 5.0 and reason2.startswith("no_finite_hw_rho") and w2 is None
    print("[smoke] pick_winning_tau_w: sharpest / single-point / all-inf fallback  PASS")


def test_subset():
    sids = [900000000 + k for k in range(5)]
    assert select_phase3_subset(sids, "") == []
    assert select_phase3_subset(sids, "none") == []
    assert select_phase3_subset(sids, "all") == sids
    assert select_phase3_subset(sids, "first:2") == [900000000, 900000001]
    # control 1: frac:0.5 -> half (rounded), deterministic by sort.
    # k = int(round(0.5*5)) = int(round(2.5)) = 2 (Python banker's rounding).
    assert select_phase3_subset(sids, "frac:0.5") == sorted(sids)[:2]
    # a 6-id list gives a clean half (round(3.0)=3), no rounding ambiguity
    six = [900000000 + k for k in range(6)]
    assert select_phase3_subset(six, "frac:0.5") == sorted(six)[:3]
    # control 2: frac rounds to >=1 even for tiny fractions; bad spec rejected
    assert len(select_phase3_subset(sids, "frac:0.01")) == 1
    try:
        select_phase3_subset(sids, "bogus"); raise AssertionError("should reject")
    except ValueError:
        pass
    print("[smoke] select_phase3_subset: none/all/first:N/frac:F (+bad spec)  PASS")


def test_absolute_gate():
    kw = dict(classify_fn=_fake_classify, k_good=3.0, k_fail=10.0,
              train_fail_mV=2.0, valid_good_mV=0.2)
    # control 1: normal case -- means over finite bundles; NaN bundles dropped;
    # valid (0.5) > good (0.2), ratio 0.5/0.1 = 5 -> in (k_good, k_fail] -> to_refine
    tr = [0.10, 0.10, np.nan]
    va = [0.40, 0.60]
    train_abs, valid_abs, ratio, status = compute_absolute_gate(tr, va, **kw)
    assert abs(train_abs - 0.10) < 1e-12
    assert abs(valid_abs - 0.50) < 1e-12
    assert abs(ratio - 5.0) < 1e-9
    assert status == "to_refine"
    # control 2a: tiny valid (<= good) short-circuits to good regardless of ratio
    _, _, _, s_good = compute_absolute_gate([0.02], [0.10], **kw)
    assert s_good == "good"
    # control 2b: empty/all-NaN train -> NaN train, inf ratio, failed
    t_abs, v_abs, r_abs, s_fail = compute_absolute_gate([np.nan], [0.3], **kw)
    assert np.isnan(t_abs) and np.isinf(r_abs) and s_fail == "failed"
    # control 2c: train RMSD above the mV ceiling -> failed on train alone
    _, _, _, s_bigtrain = compute_absolute_gate([2.5], [0.1], **kw)
    assert s_bigtrain == "failed"
    print("[smoke] compute_absolute_gate: mean/ratio/NaN-drop/short-circuit/ceiling  PASS")


def test_results_df():
    rs = [Res(900000000, "2", "spiny", 1.9, "hyp",
              0.97, 12232.0, 150.0, "good", "good", 0.031,
              0.05, 0.12, 2.4, 5.0),
          Res(900000001, "3", "spiny", 1.9, "hyp",
              0.93, 13081.0, 151.0, "to_refine", "good", 0.028,
              0.06, 0.35, 5.8, 5.0)]
    df = results_to_dataframe(rs)
    # control 1: required + NEW columns present; values carried
    for c in ("specimen_id", "cm_uF_per_cm2", "validation_status",
              "validation_status_relative", "train_rel_loss",
              "train_rmsd_abs_mV", "valid_rmsd_abs_mV",
              "valid_to_train_ratio_abs", "tau_w_chosen_ms"):
        assert c in df.columns, c
    assert df.loc[0, "cm_uF_per_cm2"] == 0.97
    assert df.loc[1, "validation_status"] == "to_refine"
    assert df.loc[1, "valid_rmsd_abs_mV"] == 0.35
    # control 2: attrs absent on the object (e.g. Phase-2.5 *_phase2 not set) ->
    # NaN for numeric, '' for string; no crash
    assert df["cm_phase2"].isna().all()
    assert (df["error_message"] == "").all()
    assert len(df) == 2
    print("[smoke] results_to_dataframe: base+new+phase2.5 columns; absent -> NaN/''  PASS")


if __name__ == "__main__":
    test_parsers()
    test_pick_winner()
    test_subset()
    test_absolute_gate()
    test_results_df()
    print("\n[smoke] ALL CHECKS PASSED")
