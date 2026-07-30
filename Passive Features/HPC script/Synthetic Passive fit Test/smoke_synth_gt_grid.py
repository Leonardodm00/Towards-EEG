# -*- coding: utf-8 -*-
"""
smoke_synth_gt_grid.py  (v2)
============================

NEURON-free smoke test for synth_gt_grid.py v2. Two independent controls per
property. Run:  python smoke_synth_gt_grid.py
Expected tail:  [smoke] ALL CHECKS PASSED
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import sys

from synth_gt_grid import (
    SearchBox, draw_manifest, save_manifest, load_manifest,
    list_groups, group_of, MANIFEST_COLUMNS,
    DEFAULT_CM_BOUNDS, DEFAULT_RM_BOUNDS, DEFAULT_RA_BOUNDS,
)
import synth_gt_grid as _sgg_module

FAKE_SWCS = [f"/morph/specimen_{k}/reconstruction.swc" for k in range(8)]


def _expect_raise(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing raised")


def test_determinism_and_stream_isolation():
    a = draw_manifest(FAKE_SWCS, draws_per_morph=3, seed=42, cells_per_cohort=5)
    b = draw_manifest(FAKE_SWCS, draws_per_morph=3, seed=42, cells_per_cohort=5)
    c = draw_manifest(FAKE_SWCS, draws_per_morph=3, seed=43, cells_per_cohort=5)
    # control 1: same seed -> identical GT
    for col in ("cm_true", "rm_true", "ra_true", "tau_m_true_ms"):
        assert np.array_equal(a[col].to_numpy(), b[col].to_numpy()), col
    assert not np.array_equal(a["cm_true"].to_numpy(), c["cm_true"].to_numpy())
    # control 2: toggling I_h / changing noise CV must NOT shift Cm/Rm/Ra (stream isolation)
    d = draw_manifest(FAKE_SWCS, draws_per_morph=3, seed=42, cells_per_cohort=5,
                      use_ih=False, noise_cv=0.9, ih_gihbar_cv=0.05)
    for col in ("cm_true", "rm_true", "ra_true"):
        assert np.array_equal(a[col].to_numpy(), d[col].to_numpy()), f"GT shifted by nuisance toggle: {col}"
    print("[smoke] determinism + GT/nuisance stream isolation  PASS")


def test_box_and_loguniform():
    df = draw_manifest(FAKE_SWCS, draws_per_morph=4000, seed=1, cells_per_cohort=64,
                       ra_mode="per_cell")  # per_cell so Ra has N independent draws
    for col, (lo, hi) in (("cm_true", DEFAULT_CM_BOUNDS),
                          ("rm_true", DEFAULT_RM_BOUNDS),
                          ("ra_true", DEFAULT_RA_BOUNDS)):
        v = df[col].to_numpy()
        assert v.min() >= lo - 1e-9 and v.max() <= hi + 1e-9, (col, v.min(), v.max())
        mid = 0.5 * (np.log(lo) + np.log(hi))
        assert abs(np.log(v).mean() - mid) < 0.05 * (np.log(hi) - np.log(lo)), col
    print("[smoke] draws inside box + log-uniform (mean log ~ midpoint)  PASS")


def test_cohort_ra():
    # per_cohort: Ra constant within each cohort, varies across cohorts
    df = draw_manifest(FAKE_SWCS, draws_per_morph=4, seed=7, cells_per_cohort=5)  # 32 cells
    per = df.groupby("cohort")["ra_true"].nunique()
    assert (per == 1).all(), per.to_dict()
    assert df.groupby("cohort")["ra_true"].first().nunique() >= 2
    # per_cell: Ra (almost surely) varies within a cohort
    dfc = draw_manifest(FAKE_SWCS, draws_per_morph=4, seed=7, cells_per_cohort=5,
                        ra_mode="per_cell")
    assert dfc.groupby("cohort")["ra_true"].nunique().max() > 1
    # group == cohort: one PBS group per cohort, sizes correct
    assert list_groups(df) == [f"cohort_{k:04d}" for k in range(df["cohort"].max() + 1)]
    print("[smoke] cohort-shared Ra (per_cohort) vs per_cell; group==cohort  PASS")


def test_ih_and_noise_jitter():
    N = 6000
    df = draw_manifest([f"/m/specimen_{k}/reconstruction.swc" for k in range(N)],
                       draws_per_morph=1, seed=3, cells_per_cohort=100,
                       ih_gihbar_nominal_S_cm2=2e-4, ih_gihbar_cv=0.5,
                       noise_sigma_nominal_mV=0.05, noise_cv=0.3)
    g = df["ih_gihbar_S_cm2"].to_numpy()
    s = df["noise_sigma_mV"].to_numpy()
    # control 1: positivity + median at nominal
    assert (g > 0).all() and (s > 0).all()
    assert abs(np.median(g) / 2e-4 - 1.0) < 0.05, np.median(g)
    assert abs(np.median(s) / 0.05 - 1.0) < 0.05, np.median(s)
    # control 2: realised CV matches the request (log-normal CV = exp draw)
    assert abs(g.std(ddof=1) / g.mean() - 0.5) < 0.05, g.std(ddof=1) / g.mean()
    assert abs(s.std(ddof=1) / s.mean() - 0.3) < 0.05, s.std(ddof=1) / s.mean()
    # noise level is one shared per-cell factor across sigma/baseline/drift:
    ratio = df["noise_baseline_sigma_mV"] / df["noise_sigma_mV"]
    assert np.allclose(ratio, 0.05 / 0.05) and np.allclose(
        df["noise_drift_sigma_mV"] / df["noise_sigma_mV"], 0.10 / 0.05)
    print("[smoke] I_h gbar + noise jitter: positive, median=nominal, CV=request  PASS")


def test_ih_off_baseline():
    off = draw_manifest(FAKE_SWCS, draws_per_morph=1, seed=0, cells_per_cohort=8,
                        use_ih=False)
    assert (~off["use_ih"]).all() and off["ih_gihbar_S_cm2"].isna().all()
    # noise still present + jittered when I_h is off
    assert (off["noise_sigma_mV"] > 0).all()
    print("[smoke] I_h off -> passive baseline (gbar NaN), noise intact  PASS")


def test_pairing_batching_tau_roundtrip():
    K = 3
    df = draw_manifest(FAKE_SWCS, draws_per_morph=K, seed=5, cells_per_cohort=5)
    assert len(df) == len(FAKE_SWCS) * K
    assert (df["morph_name"].value_counts() == K).all()
    sids = df["specimen_id"].to_numpy()
    assert len(set(sids)) == len(sids)
    # batching/cohorts partition all cells
    assert sum(len(group_of(df, g)) for g in list_groups(df)) == len(df)
    _expect_raise(KeyError, group_of, df, "cohort_9999")
    # tau identity
    tau = df["rm_true"].to_numpy() * df["cm_true"].to_numpy() * 1e-3
    assert np.allclose(df["tau_m_true_ms"].to_numpy(), tau, rtol=0, atol=1e-12)
    # round-trip
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "manifest.csv"
        save_manifest(df, p)
        df2 = load_manifest(p)
        assert list(df2.columns) == MANIFEST_COLUMNS
        for col in ("cm_true", "rm_true", "ra_true", "tau_m_true_ms",
                    "ih_gihbar_S_cm2", "noise_sigma_mV"):
            assert np.allclose(df[col].to_numpy(), df2[col].to_numpy(), equal_nan=True)
        assert df2["use_ih"].dtype == bool and df2["cohort"].dtype.kind == "i"
    print("[smoke] pairing + cohort partition + tau identity + round-trip  PASS")


def test_cli_forwards_every_flag():
    """Regression guard for the v70 bug: argparse accepted --ih-kinetics,
    --ra-phys-lo, --ra-phys-hi, but main() never passed them to
    draw_manifest(), so every manifest silently fell back to rodent Ih
    kinetics and the full [50,1000] Ra box regardless of the CLI flags
    (caught on-cluster, not by this suite -- because every other test here
    calls draw_manifest() directly and never exercises main()/argv at all).

    Every flag below is set to a value DIFFERENT from its argparse default.
    If any future flag is added to the parser but not forwarded to
    draw_manifest, the written manifest will show that flag's DEFAULT
    instead of the value asserted here, and this test fails loudly instead
    of silently shipping a wrong ground truth.
    """
    with tempfile.TemporaryDirectory() as td:
        morph_root = Path(td) / "morphs"
        for i in range(10):
            d = morph_root / f"specimen_{i:03d}"
            d.mkdir(parents=True)
            (d / "reconstruction.swc").write_text("")
        out_csv = Path(td) / "manifest.csv"

        argv = [
            "--morph-root", str(morph_root),
            "--morph-glob", "specimen_*/reconstruction.swc",
            "--out", str(out_csv),
            "--draws-per-morph", "2",          # default 1
            "--seed", "7",                     # default 0
            "--cells-per-cohort", "4",         # default 10
            "--ra-mode", "per_cell",           # default per_cohort
            "--e-pas", "-65.0",                # default -70.0
            "--F", "2.0",                      # default 1.9
            "--max-cells", "6",                # default None
            "--ih-gihbar", "1e-4",             # default 2e-4
            "--ih-gihbar-cv", "0.3",           # default 0.5
            "--ih-ehcn", "-49.85",             # default -45.0
            "--ih-dist", "uniform",            # default hay_exponential
            "--ih-kinetics", "Ih_human",       # default Ih  <- the bug
            "--ra-phys-lo", "100.0",           # default None <- the bug
            "--ra-phys-hi", "500.0",           # default None <- the bug
            "--noise-sigma", "0.07",           # default 0.05
            "--noise-baseline", "0.08",        # default 0.05
            "--noise-drift", "0.12",           # default 0.10
            "--noise-cv", "0.25",              # default 0.3
            "--cm-phys-lo", "0.35",            # default PHY_CM_LO_DEFAULT
            "--cm-phys-hi", "1.6",             # default PHY_CM_HI_DEFAULT
            "--tau-lo-ms", "2.5",              # default PHY_TAU_LO_DEFAULT
            "--tau-hi-ms", "45.0",             # default PHY_TAU_HI_DEFAULT
        ]

        old_argv = sys.argv
        try:
            sys.argv = ["synth_gt_grid.py"] + argv
            _sgg_module.main()
        finally:
            sys.argv = old_argv

        df = load_manifest(out_csv)

        # the two flags that actually broke:
        assert set(df["ih_kinetics"]) == {"Ih_human"}, \
            "CLI --ih-kinetics Ih_human not reflected in manifest"
        assert df["ra_true"].min() >= 100.0 - 1e-6 and df["ra_true"].max() <= 500.0 + 1e-6, \
            "CLI --ra-phys-lo/--ra-phys-hi window not respected: got [%.1f, %.1f]" % (
                df["ra_true"].min(), df["ra_true"].max())

        # everything else that main() is supposed to forward:
        assert set(df["ih_dist"]) == {"uniform"}
        assert set(df["ih_ehcn_mV"].dropna()) == {-49.85}
        assert np.isclose(df["ih_gihbar_S_cm2"].median(), 1e-4, rtol=0.5)
        assert set(df["e_pas_mV"]) == {-65.0}
        assert set(df["F"]) == {2.0}
        assert (df["noise_sigma_mV"] > 0).all()
        # max_cells caps the TOTAL cell count (n_morph*draws_per_morph, THEN
        # capped) -- not morphology count before multiplying.
        assert len(df) == 6              # max_cells=6 directly caps n
        assert (df["cm_true"] >= 0.35 - 1e-9).all() and (df["cm_true"] <= 1.6 + 1e-9).all()
        assert (df["tau_m_true_ms"] >= 2.5 - 1e-9).all() and (df["tau_m_true_ms"] <= 45.0 + 1e-9).all()
        # ra_mode=per_cell -> Ra is NOT shared within a cohort (unlike per_cohort)
        cohorts = list_groups(df)
        per_cell_varies = any(
            df.loc[df["group"] == g, "ra_true"].nunique() > 1 for g in cohorts
        )
        assert per_cell_varies, "ra_mode=per_cell not reflected (Ra looks cohort-shared)"

    print("[smoke] CLI (main/argv) forwards every flag to draw_manifest  PASS")


if __name__ == "__main__":
    test_determinism_and_stream_isolation()
    test_box_and_loguniform()
    test_cohort_ra()
    test_ih_and_noise_jitter()
    test_ih_off_baseline()
    test_pairing_batching_tau_roundtrip()
    test_cli_forwards_every_flag()
    print("\n[smoke] ALL CHECKS PASSED")
