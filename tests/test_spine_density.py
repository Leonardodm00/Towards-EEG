"""Smoke test for spine_density.build_phi.

Runs standalone (python3 test_spine_density.py) or under pytest. Every expected
value is recomputed INDEPENDENTLY from the raw synthetic geometry with a local
frustum function -- the test never calls the module's internal attribution -- so
agreement is a genuine cross-check, not a tautology.

Coverage
  A  analytic single spine on a straight dendrite: exact area, correct BASE
     attribution, exact phi value, f_implied
  B  branched tree, two spines on different branches, incl. a TIP spine:
     area conservation (sum phi*len == total spine area), f_implied, correct
     per-branch placement
  C  emergent cutoff: spine only beyond 60 um; assert proximal segments EXIST
     (no hard cutoff imposed) but carry phi == 0
  D  units: input_units='nm' vs pre-converted 'um' give identical um output;
     one shaft area matches an independent frustum / 1e6
  E  integrate_phi_over_segment: partial ranges sum to the full-branch integral
"""

import math

import numpy as np
import pandas as pd

import spine_density as sd


TOL = 1e-9


def _frus(r1, r2, L):
    """Independent copy of the frustum lateral area (nm in -> nm^2 out)."""
    return math.pi * (r1 + r2) * math.sqrt((r1 - r2) ** 2 + L ** 2)


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _mk(rows):
    """rows: list of (id, p, x, y, z, r, annotated_type)."""
    return pd.DataFrame(rows, columns=["id", "p", "x", "y", "z", "r", "annotated_type"])


# --------------------------------------------------------------------------- #
# A -- analytic single spine                                                  #
# --------------------------------------------------------------------------- #
def _neuron_A():
    return _mk([
        (0, -1,    0, 0,   0, 1000, "soma"),
        (1,  0, 1000, 0,   0,  500, "dendrite"),
        (2,  1, 2000, 0,   0,  500, "dendrite"),
        (3,  2, 3000, 0,   0,  500, "dendrite"),
        (4,  3, 4000, 0,   0,  500, "dendrite"),   # tip
        (5,  2, 2000, 0, 200,  100, "neck"),       # spine on node 2
        (6,  5, 2000, 0, 500,  300, "head"),
    ])


def test_A_analytic_single_spine():
    df = _neuron_A()
    phi = sd.build_phi(df, nid="A", input_units="nm")

    # independent expected areas (nm^2 -> um^2)
    a_spine = (_frus(500, 100, 200) + _frus(100, 300, 300)) / 1e6
    a_shaft = (_frus(1000, 500, 1000) + _frus(500, 500, 1000)
               + _frus(500, 500, 1000) + _frus(500, 500, 1000)) / 1e6

    assert len(phi) == 4, "expected 4 shaft segments, got %d" % len(phi)

    base_row = phi[phi["node_to"] == 2]
    assert len(base_row) == 1
    got_spine = float(base_row["spine_area_um2"].iloc[0])
    assert abs(got_spine - a_spine) < 1e-6, (got_spine, a_spine)
    # seg (1->2) is 1 um long, so phi == spine area numerically
    assert abs(float(base_row["phi_um"].iloc[0]) - a_spine / 1.0) < 1e-6

    # all other shaft segments carry no spine area
    other = phi[phi["node_to"] != 2]
    assert float(other["spine_area_um2"].abs().sum()) < TOL

    assert abs(float(phi["shaft_area_um2"].sum()) - a_shaft) < 1e-6
    f = sd.cell_f_implied_from_phi(phi)
    assert abs(f - (1.0 + a_spine / a_shaft)) < 1e-9, f

    # integrate over the whole branch recovers the spine area
    xmax = float(phi["x1_um"].max())
    integ = sd.integrate_phi_over_segment(phi, 0, 0.0, xmax)
    assert abs(integ - a_spine) < 1e-6, (integ, a_spine)


# --------------------------------------------------------------------------- #
# B -- branched tree, two spines (one on a tip)                               #
# --------------------------------------------------------------------------- #
def _neuron_B():
    return _mk([
        (0, -1,    0,    0,   0, 1000, "soma"),
        (1,  0, 1000,    0,   0,  500, "dendrite"),
        (2,  1, 2000,    0,   0,  500, "dendrite"),   # branch point
        # branch A
        (3,  2, 3000,    0,   0,  400, "dendrite"),
        (4,  3, 4000,    0,   0,  300, "dendrite"),   # tip
        (5,  3, 3000,    0, 200,  100, "neck"),       # spine on interior node 3
        (6,  5, 3000,    0, 500,  300, "head"),
        # branch B
        (7,  2, 2000, 1000,   0,  400, "dendrite"),
        (8,  7, 2000, 2000,   0,  300, "dendrite"),   # tip
        (9,  8, 2000, 2000, 200,  100, "neck"),       # spine on TIP node 8
        (10, 9, 2000, 2000, 500,  300, "head"),
    ])


def test_B_branched_conservation():
    df = _neuron_B()
    phi = sd.build_phi(df, nid="B", input_units="nm")

    spine1 = (_frus(400, 100, 200) + _frus(100, 300, 300)) / 1e6  # base node 3
    spine2 = (_frus(300, 100, 200) + _frus(100, 300, 300)) / 1e6  # base node 8 (tip)
    spine_total = spine1 + spine2

    shaft_total = (
        _frus(1000, 500, 1000) + _frus(500, 500, 1000)     # 0-1, 1-2
        + _frus(500, 400, 1000) + _frus(400, 300, 1000)    # 2-3, 3-4
        + _frus(500, 400, math.sqrt(1000**2))              # 2-7 (dy=1000)
        + _frus(400, 300, 1000)                            # 7-8
    ) / 1e6

    # 6 shaft segments over 3 branches
    assert len(phi) == 6, len(phi)
    assert phi["branch_id"].nunique() == 3

    # conservation: integral of phi over the whole cell == total spine area
    recovered = float((phi["phi_um"] * phi["seg_len_um"]).sum())
    assert abs(recovered - spine_total) < 1e-6, (recovered, spine_total)

    # f_implied matches the direct global ratio
    f = sd.cell_f_implied_from_phi(phi)
    assert abs(f - (1.0 + spine_total / shaft_total)) < 1e-6, f

    # each spine landed on the correct branch (base node's branch)
    br_of_3 = int(phi[phi["node_to"] == 3]["branch_id"].iloc[0])
    br_of_8 = int(phi[phi["node_to"] == 8]["branch_id"].iloc[0])
    assert abs(float(phi[phi["node_to"] == 3]["spine_area_um2"].iloc[0]) - spine1) < 1e-6
    assert abs(float(phi[phi["node_to"] == 8]["spine_area_um2"].iloc[0]) - spine2) < 1e-6
    # areas are confined to those two segments only
    mask = phi["node_to"].isin([3, 8])
    assert float(phi[~mask]["spine_area_um2"].abs().sum()) < TOL
    assert br_of_3 != br_of_8


# --------------------------------------------------------------------------- #
# C -- emergent cutoff (no hard 60 um exclusion)                              #
# --------------------------------------------------------------------------- #
def _neuron_C():
    rows = [(0, -1, 0, 0, 0, 1000, "soma")]
    prev = 0
    for k in range(1, 11):               # nodes at 10, 20, ..., 100 um
        rows.append((k, prev, k * 10000, 0, 0, 500, "dendrite"))
        prev = k
    # single spine at node 8 (80 um from soma)
    rows.append((11, 8, 80000, 0, 200, 100, "neck"))
    rows.append((12, 11, 80000, 0, 500, 300, "head"))
    return _mk(rows)


def test_C_emergent_cutoff():
    df = _neuron_C()
    phi = sd.build_phi(df, nid="C", input_units="nm")

    # the proximal shaft is fully represented -- nothing was excluded
    assert len(phi) == 10, len(phi)
    proximal = phi[phi["d_to_um"] <= 60.0]
    assert len(proximal) >= 5, "proximal segments must still be present"
    assert float(proximal["phi_um"].abs().sum()) < TOL, "proximal phi must be ~0"

    # the spine's own segment (base node 8) is distal and non-zero
    base_row = phi[phi["node_to"] == 8]
    assert float(base_row["phi_um"].iloc[0]) > 0.0


# --------------------------------------------------------------------------- #
# D -- units                                                                  #
# --------------------------------------------------------------------------- #
def test_D_units_nm_vs_um():
    df_nm = _neuron_A()
    df_um = df_nm.copy()
    for c in ("x", "y", "z", "r"):
        df_um[c] = df_um[c] / 1000.0

    phi_nm = sd.build_phi(df_nm, nid="A", input_units="nm").reset_index(drop=True)
    phi_um = sd.build_phi(df_um, nid="A", input_units="um").reset_index(drop=True)

    for col in ("x1_um", "seg_len_um", "shaft_area_um2", "spine_area_um2", "phi_um"):
        assert np.allclose(phi_nm[col].values, phi_um[col].values, atol=1e-9), col

    # one shaft area matches an independent frustum / 1e6
    row = phi_nm[phi_nm["node_to"] == 1].iloc[0]
    assert abs(float(row["shaft_area_um2"]) - _frus(1000, 500, 1000) / 1e6) < 1e-9


# --------------------------------------------------------------------------- #
# E -- integrator additivity                                                  #
# --------------------------------------------------------------------------- #
def test_E_integrator_additivity():
    df = _neuron_B()
    phi = sd.build_phi(df, nid="B", input_units="nm")
    for b in phi["branch_id"].unique():
        sub = phi[phi["branch_id"] == b]
        xmax = float(sub["x1_um"].max())
        mid = xmax / 2.0
        full = sd.integrate_phi_over_segment(phi, b, 0.0, xmax)
        left = sd.integrate_phi_over_segment(phi, b, 0.0, mid)
        right = sd.integrate_phi_over_segment(phi, b, mid, xmax)
        assert abs((left + right) - full) < 1e-9, (b, left, right, full)


# --------------------------------------------------------------------------- #
# F -- psi_vs_distance aggregation                                            #
# --------------------------------------------------------------------------- #
def test_F_psi_profile():
    df = _neuron_C()                       # 100 um dendrite, one spine at 80 um
    phi = sd.build_phi(df, nid="C", input_units="nm")
    prof = sd.psi_vs_distance(phi, bin_width_um=10.0)

    # 10 bins of 10 um covering 0..100 um
    assert len(prof) == 10, len(prof)
    assert abs(float(prof["d_hi_um"].max()) - 100.0) < 1e-9

    # conservation across bins
    assert abs(float(prof["spine_area_um2"].sum())
               - float(phi["spine_area_um2"].sum())) < 1e-9
    assert abs(float(prof["shaft_area_um2"].sum())
               - float(phi["shaft_area_um2"].sum())) < 1e-9
    assert abs(float(prof["shaft_len_um"].sum())
               - float(phi["seg_len_um"].sum())) < 1e-9

    # emergent proximal zero: every bin below 60 um has psi == 0
    prox = prof[prof["d_hi_um"] <= 60.0]
    assert len(prox) == 6, len(prox)
    assert float(prox["psi_mean"].abs().sum()) < TOL
    assert np.allclose(prox["F_bin"].values, 1.0)

    # the spine sits on segment 7->8, midpoint 75 um -> the [70,80) bin
    hot = prof[(prof["d_lo_um"] <= 75.0) & (prof["d_hi_um"] > 75.0)]
    assert len(hot) == 1
    assert float(hot["psi_mean"].iloc[0]) > 0.0
    assert abs(float(hot["F_bin"].iloc[0])
               - (1.0 + float(hot["psi_mean"].iloc[0]))) < 1e-12

    # area-weighted pooling identity: psi_bin == A_spine_bin / A_shaft_bin
    for r in prof.itertuples(index=False):
        if r.shaft_area_um2 > 0:
            assert abs(r.psi_mean - r.spine_area_um2 / r.shaft_area_um2) < 1e-12

    # per-segment psi definition: phi / (pi * diameter)
    row = phi[phi["node_to"] == 8].iloc[0]
    assert abs(float(row["psi"])
               - float(row["phi_um"]) / (math.pi * float(row["shaft_diam_um"]))
               ) < 1e-12

    # empty input is handled
    empty = sd.psi_vs_distance(pd.DataFrame())
    assert len(empty) == 0


# --------------------------------------------------------------------------- #
# G -- cell_f_beyond_cutoff (the literature-matched quantity)                 #
# --------------------------------------------------------------------------- #
def test_G_f_beyond_cutoff():
    df = _neuron_C()          # 100 um dendrite, 10 segs of 10 um, one spine at 80 um
    phi = sd.build_phi(df, nid="C", input_units="nm")

    # independent expectation: only the segment(s) with d_from >= 60 count.
    # d_from values are 0,10,...,90 -> segments with d_from in {60,70,80,90}
    # i.e. 4 segments, exactly one (base node 8, d_from=70) carries spine area.
    sub = phi[phi["d_from_um"] >= 60.0]
    assert len(sub) == 4
    a_sh = float(sub["shaft_area_um2"].sum())
    a_sp = float(sub["spine_area_um2"].sum())
    expect_F = 1.0 + a_sp / a_sh

    res = sd.cell_f_beyond_cutoff(phi, cutoff_um=60.0, by="d_from_um")
    assert res["n_segments_included"] == 4, res
    assert abs(res["F"] - expect_F) < 1e-12, (res["F"], expect_F)
    assert abs(res["A_shaft_um2"] - a_sh) < 1e-9
    assert abs(res["A_spine_um2"] - a_sp) < 1e-9

    # cutoff-restricted F must be >= whole-cell f_implied (proximal shaft with
    # ~no spine area dilutes the whole-cell ratio toward 1)
    f_whole = sd.cell_f_implied_from_phi(phi)
    assert res["F"] >= f_whole - 1e-12, (res["F"], f_whole)

    # cutoff_um=0 recovers the whole-cell quantity exactly
    res0 = sd.cell_f_beyond_cutoff(phi, cutoff_um=0.0, by="d_from_um")
    assert abs(res0["F"] - f_whole) < 1e-12
    assert abs(res0["frac_shaft_area_included"] - 1.0) < 1e-12

    # a cutoff beyond the whole cell excludes everything -> NaN, not a crash
    res_big = sd.cell_f_beyond_cutoff(phi, cutoff_um=1000.0, by="d_from_um")
    assert res_big["n_segments_included"] == 0
    assert math.isnan(res_big["F"])

    # 'by' selector sanity: d_to_um is never stricter than d_from_um for a
    # positive cutoff (every segment's d_to >= its d_from), so it should
    # include at least as many segments
    res_to = sd.cell_f_beyond_cutoff(phi, cutoff_um=60.0, by="d_to_um")
    assert res_to["n_segments_included"] >= res["n_segments_included"]

    empty = sd.cell_f_beyond_cutoff(pd.DataFrame())
    assert math.isnan(empty["F"]) and empty["n_segments_total"] == 0


# --------------------------------------------------------------------------- #
# runner                                                                      #
# --------------------------------------------------------------------------- #
def _run_all():
    tests = [
        ("A analytic single spine", test_A_analytic_single_spine),
        ("B branched conservation + tip spine", test_B_branched_conservation),
        ("C emergent cutoff (no hard exclusion)", test_C_emergent_cutoff),
        ("D units nm vs um", test_D_units_nm_vs_um),
        ("E integrator additivity", test_E_integrator_additivity),
        ("F psi-vs-distance profile", test_F_psi_profile),
        ("G f_beyond_cutoff (literature-matched)", test_G_f_beyond_cutoff),
    ]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print("[PASS] %s" % name)
            n_pass += 1
        except AssertionError as e:
            print("[FAIL] %s -- %s" % (name, e))
        except Exception as e:  # noqa
            print("[ERROR] %s -- %r" % (name, e))
    print("-" * 60)
    print("%d / %d passed" % (n_pass, len(tests)))
    return n_pass == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
