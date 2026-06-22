# -*- coding: utf-8 -*-
"""
smoke_run_synth_benchmark.py
============================

NEURON-free smoke test for the PURE helpers of run_synth_benchmark.py
(tau_w winner selection, subset parsing, CLI list/window parsing, result
serialisation). Two independent controls each. Run: python smoke_run_synth_benchmark.py
"""

from collections import namedtuple

import numpy as np
import pandas as pd

from run_synth_benchmark import (
    parse_float_list, parse_window, pick_winning_tau_w,
    select_phase3_subset, results_to_dataframe,
)

# fakes (duck-typed)
Prof = namedtuple("Prof", ["tau_w_ms", "hw_rho", "kappa", "bias_log"])
Res = namedtuple("Res", ["specimen_id", "cm_uF_per_cm2", "rm_Ohm_cm2", "ra_Ohm_cm",
                         "validation_status", "tau_w_chosen_ms", "F"])


def test_parsers():
    assert parse_float_list("2.0,5.0,10.0") == [2.0, 5.0, 10.0]
    assert parse_float_list("-10,-30,-50") == [-10.0, -30.0, -50.0]   # negatives
    assert parse_window("0.5,100.0") == (0.5, 100.0)
    try:
        parse_window("1,2,3"); raise AssertionError("should reject 3-tuple")
    except ValueError:
        pass
    print("[smoke] parse_float_list / parse_window (incl negatives + bad input)  PASS")


def test_pick_winner():
    grid = [2.0, 5.0, 10.0]
    # control 1: smallest finite HW_rho wins
    profs = [Prof(2.0, 9.9, 1.0, 0.1), Prof(5.0, 3.3, 2.0, 0.0), Prof(10.0, 7.0, 1.5, 0.2)]
    tau, reason, w = pick_winning_tau_w(profs, grid)
    assert tau == 5.0 and reason == "sharpest_hw_rho" and w.tau_w_ms == 5.0
    # control 2: all-inf -> middle-of-grid fallback, flagged, winner None
    profs_inf = [Prof(t, np.inf, 0.0, np.nan) for t in grid]
    tau2, reason2, w2 = pick_winning_tau_w(profs_inf, grid)
    assert tau2 == 5.0 and reason2.startswith("no_finite_hw_rho") and w2 is None
    # a single finite among infs still wins
    mixed = [Prof(2.0, np.inf, 0, 0), Prof(5.0, np.inf, 0, 0), Prof(10.0, 4.2, 1, 0)]
    assert pick_winning_tau_w(mixed, grid)[0] == 10.0
    print("[smoke] pick_winning_tau_w: sharpest wins; all-inf -> flagged fallback  PASS")


def test_subset():
    gdf = pd.DataFrame({"specimen_id": [900000000 + k for k in range(5)]})
    assert select_phase3_subset(gdf, "") == []
    assert select_phase3_subset(gdf, "none") == []
    assert select_phase3_subset(gdf, "all") == gdf["specimen_id"].tolist()
    assert select_phase3_subset(gdf, "first:2") == [900000000, 900000001]
    # control 2: frac is deterministic and >=1 when fraction rounds down to 0
    assert select_phase3_subset(gdf, "frac:0.4") == [900000000, 900000001]
    assert len(select_phase3_subset(gdf, "frac:0.01")) == 1
    try:
        select_phase3_subset(gdf, "bogus"); raise AssertionError("should reject")
    except ValueError:
        pass
    print("[smoke] select_phase3_subset: none/all/first:N/frac:F + bad spec  PASS")


def test_results_df():
    rs = [Res(900000000, 0.97, 12232.0, 50.0, "good", 5.0, 1.9),
          Res(900000001, 0.93, 13081.0, 151.0, "to_refine", 10.0, 1.9)]
    df = results_to_dataframe(rs)
    # control 1: required columns present, values carried
    for c in ("specimen_id", "cm_uF_per_cm2", "rm_Ohm_cm2", "ra_Ohm_cm",
              "validation_status", "tau_w_chosen_ms"):
        assert c in df.columns, c
    assert df.loc[0, "cm_uF_per_cm2"] == 0.97 and df.loc[1, "validation_status"] == "to_refine"
    # control 2: missing attrs (e.g. Phase-2.5 *_phase2 not yet set) -> NaN, no crash
    assert df["cm_phase2"].isna().all()
    assert len(df) == 2
    print("[smoke] results_to_dataframe: fields carried; missing attrs -> NaN  PASS")


if __name__ == "__main__":
    test_parsers()
    test_pick_winner()
    test_subset()
    test_results_df()
    print("\n[smoke] ALL CHECKS PASSED")
