# -*- coding: utf-8 -*-
"""
smoke_gen_from_manifest.py
==========================

NEURON-free smoke test for the PURE manifest-row -> generation-config mapping in
gen_from_manifest.py. Validates against real manifest rows from synth_gt_grid
(no synthetic_ground_truth import, no NEURON). Two independent controls each.

Run:  python smoke_gen_from_manifest.py
Tail: [smoke] ALL CHECKS PASSED
"""

import numpy as np

from synth_gt_grid import draw_manifest
from gen_from_manifest import (
    row_to_gt_kwargs, row_to_noise_kwargs, use_builder_factory,
)

SWCS = [f"/m/specimen_{k}/reconstruction.swc" for k in range(4)]


def test_gt_kwargs_ih_on():
    df = draw_manifest(SWCS, draws_per_morph=1, seed=1, cells_per_cohort=4,
                       use_ih=True, ih_gihbar_nominal_S_cm2=2e-4, ih_ehcn_mV=-45.0)
    row = df.iloc[0]
    gt = row_to_gt_kwargs(row)
    # control 1: physical params copied verbatim, correct keys/units
    assert gt["cm_uF_cm2"] == float(row["cm_true"])
    assert gt["rm_Ohm_cm2"] == float(row["rm_true"])
    assert gt["ra_Ohm_cm"] == float(row["ra_true"])
    assert gt["e_pas_mV"] == float(row["e_pas_mV"]) and gt["spine_factor_F"] == float(row["F"])
    assert gt["active"] is False
    # control 2: I_h on -> nested IhConfig kwargs carry the row's gbar/ehcn/dist
    assert gt["ih"] is not None
    assert gt["ih"]["gIhbar_S_cm2"] == float(row["ih_gihbar_S_cm2"])
    assert gt["ih"]["ehcn_mV"] == -45.0
    assert gt["ih"]["distribution"] == "hay_exponential"
    print("[smoke] row_to_gt_kwargs (I_h on): params verbatim + IhConfig kwargs  PASS")


def test_gt_kwargs_ih_off():
    df = draw_manifest(SWCS, draws_per_morph=1, seed=1, cells_per_cohort=4, use_ih=False)
    gt = row_to_gt_kwargs(df.iloc[0])
    # control 1: passive baseline -> ih is None
    assert gt["ih"] is None
    # control 2: factory choice flips to the builder (pure passive)
    assert use_builder_factory(df.iloc[0]) is True
    print("[smoke] row_to_gt_kwargs (I_h off): ih None + builder factory  PASS")


def test_factory_choice():
    on = draw_manifest(SWCS, draws_per_morph=1, seed=2, cells_per_cohort=4, use_ih=True)
    off = draw_manifest(SWCS, draws_per_morph=1, seed=2, cells_per_cohort=4, use_ih=False)
    # control 1: I_h on -> bundled cell (factory False), so generator hosts I_h
    assert use_builder_factory(on.iloc[0]) is False
    # control 2: I_h off -> monolith builder (factory True)
    assert use_builder_factory(off.iloc[0]) is True
    print("[smoke] use_builder_factory: I_h on->bundled, off->builder  PASS")


def test_noise_kwargs():
    df = draw_manifest(SWCS, draws_per_morph=2, seed=3, cells_per_cohort=4,
                       noise_sigma_nominal_mV=0.05, noise_cv=0.3)
    r0, r1 = df.iloc[0], df.iloc[1]
    n0, n1 = row_to_noise_kwargs(r0), row_to_noise_kwargs(r1)
    # control 1: keys + values copied from the (jittered) manifest row
    assert set(n0) == {"sigma_mV", "baseline_sigma_mV", "drift_sigma_mV", "seed"}
    assert n0["sigma_mV"] == float(r0["noise_sigma_mV"])
    assert n0["seed"] == int(r0["noise_seed"]) and n0["seed"] != n1["seed"]
    # control 2: per-cell jitter is real -> sigma differs cell to cell
    assert n0["sigma_mV"] != n1["sigma_mV"]
    print("[smoke] row_to_noise_kwargs: per-cell jittered sigmas + distinct seeds  PASS")


if __name__ == "__main__":
    test_gt_kwargs_ih_on()
    test_gt_kwargs_ih_off()
    test_factory_choice()
    test_noise_kwargs()
    print("\n[smoke] ALL CHECKS PASSED")
