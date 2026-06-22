# -*- coding: utf-8 -*-
"""
smoke_aggregate_synth_results.py
================================

NEURON-free smoke test for aggregate_synth_results.py: the pure join/ratio/
coverage maths (build_summary, build_coverage) AND an end-to-end render check
(every figure writes a file to a temp dir, headless). Two controls per property.

Run:  python smoke_aggregate_synth_results.py
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_synth_results import (
    build_summary, build_coverage, render_all, STYLES,
)


def _fake_manifest(n=12, n_cohorts=3):
    rng = np.random.default_rng(0)
    per = n // n_cohorts
    rows = []
    for i in range(n):
        coh = i // per
        rows.append(dict(
            specimen_id=900000000 + i, cohort=coh, morph_name=f"specimen_{i%4}",
            cm_true=float(np.exp(rng.uniform(np.log(0.3), np.log(3.0)))),
            rm_true=float(np.exp(rng.uniform(np.log(1e3), np.log(1e5)))),
            ra_true=float(150.0 + 100 * coh),                 # cohort-shared Ra
            ih_gihbar_S_cm2=2e-4, noise_sigma_mV=0.05, use_ih=True,
            ra_mode="per_cohort"))
    m = pd.DataFrame(rows)
    m["tau_m_true_ms"] = m["rm_true"] * m["cm_true"] * 1e-3
    return m


def _fake_recovered(m):
    # recover Cm x1.05, Rm x0.92; Ra fixed per cohort to ra_true x1.1 (post-2.5)
    r = m[["specimen_id", "cohort"]].copy()
    r["cm_uF_per_cm2"] = m["cm_true"].values * 1.05
    r["rm_Ohm_cm2"] = m["rm_true"].values * 0.92
    r["ra_Ohm_cm"] = (m.groupby("cohort")["ra_true"].transform("first") * 1.10).values
    r["validation_status"] = "good"
    r["valid_to_train_ratio"] = 2.0
    r["tau_w_chosen_ms"] = 5.0
    # pre-2.5 stash (free Ra, noisier)
    r["cm_phase2"] = m["cm_true"].values * 1.10
    r["rm_phase2"] = m["rm_true"].values * 0.85
    r["ra_phase2"] = m["ra_true"].values * 0.6
    return r


def test_build_summary():
    m = _fake_manifest()
    s = build_summary(m, _fake_recovered(m))
    # control 1: ratios computed correctly
    assert np.allclose(s["cm_ratio"], 1.05)
    assert np.allclose(s["rm_ratio"], 0.92)
    assert np.allclose(s["tau_ratio"], 1.05 * 0.92)           # product of factors
    assert np.allclose(s["cm_dex"], np.log10(1.05))
    # control 2: pre-2.5 columns present + Ra is cohort-constant post-2.5
    assert "cm_ratio_phase2" in s.columns and np.allclose(s["cm_ratio_phase2"], 1.10)
    per = s.groupby("cohort")["ra_Ohm_cm"].nunique()
    assert (per == 1).all()
    print("[smoke] build_summary: ratios/dex + pre-2.5 + cohort-Ra  PASS")


def test_build_coverage():
    m = _fake_manifest(n=4, n_cohorts=1)
    # Cm CI brackets truth (covered); Rm CI misses (not covered); Ra degenerate
    rows = []
    for _, r in m.iterrows():
        sid = int(r["specimen_id"])
        rows += [
            dict(specimen_id=sid, parameter="Cm", mle=r["cm_true"],
                 ci_bca_lo=r["cm_true"] * 0.9, ci_bca_hi=r["cm_true"] * 1.1,
                 ci_perc_lo=r["cm_true"] * 0.95, ci_perc_hi=r["cm_true"] * 1.05),
            dict(specimen_id=sid, parameter="Rm", mle=r["rm_true"] * 2,
                 ci_bca_lo=r["rm_true"] * 1.5, ci_bca_hi=r["rm_true"] * 2.5,
                 ci_perc_lo=r["rm_true"] * 1.5, ci_perc_hi=r["rm_true"] * 2.5),
            dict(specimen_id=sid, parameter="Ra", mle=r["ra_true"],
                 ci_bca_lo=r["ra_true"], ci_bca_hi=r["ra_true"],   # degenerate
                 ci_perc_lo=r["ra_true"], ci_perc_hi=r["ra_true"]),
        ]
    cov = build_coverage(pd.DataFrame(rows), m)
    # control 1: Cm covered, Rm not
    assert cov[cov.parameter == "Cm"]["covered_bca"].all()
    assert not cov[cov.parameter == "Rm"]["covered_bca"].any()
    # control 2: Ra flagged degenerate (fix_ra pin)
    assert cov[cov.parameter == "Ra"]["degenerate"].all()
    # empty phase3 -> empty coverage, no crash
    assert build_coverage(pd.DataFrame(), m).empty
    print("[smoke] build_coverage: covered/missed + degenerate Ra + empty  PASS")


def test_render_all_writes_files():
    m = _fake_manifest()
    s = build_summary(m, _fake_recovered(m))
    cov = build_coverage(pd.DataFrame([
        dict(specimen_id=int(m.iloc[0].specimen_id), parameter="Cm",
             mle=m.iloc[0].cm_true, ci_bca_lo=m.iloc[0].cm_true * 0.9,
             ci_bca_hi=m.iloc[0].cm_true * 1.1, ci_perc_lo=m.iloc[0].cm_true * 0.9,
             ci_perc_hi=m.iloc[0].cm_true * 1.1)]), m)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        render_all(s, cov, out, styles=("paper", "talk"))
        # control 1: paper produces PDF + PNG; talk produces PNG
        paper_pdf = list((out / "figures" / "paper").glob("*.pdf"))
        paper_png = list((out / "figures" / "paper").glob("*.png"))
        talk_png = list((out / "figures" / "talk").glob("*.png"))
        assert len(paper_pdf) >= 5 and len(paper_png) >= 5, (len(paper_pdf), len(paper_png))
        assert len(talk_png) >= 5, len(talk_png)
        # control 2: the key publication figures exist by name
        names = {p.stem for p in paper_pdf}
        for must in ("recovery_distributions", "bias_vs_injected",
                     "iso_tau_identifiability", "recovered_vs_injected"):
            assert must in names, (must, names)
    print(f"[smoke] render_all: paper(pdf+png) & talk(png) files written  PASS")


if __name__ == "__main__":
    test_build_summary()
    test_build_coverage()
    test_render_all_writes_files()
    print("\n[smoke] ALL CHECKS PASSED")
