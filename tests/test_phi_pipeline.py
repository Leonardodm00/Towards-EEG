"""Smoke test for phi_pipeline_colab -- the Drive-facing driver.

Runs anywhere (no Colab, no Drive): it builds a temporary directory that looks
like the 'Reconstructed neurons' folder, injects a stub labeller with the same
call contract as label_dendritic_spines_robust, and exercises the full
compute_phi_factors path end to end.

The stub labeller is deliberately an IDENTITY labeller: the synthetic CSVs
already carry 'head'/'neck' labels, so this test exercises the PIPELINE
(discovery, plumbing, aggregation, persistence, provenance) and not the user's
spine-identification algorithm, which is separate code with its own behaviour.
What is checked about the labeller is that its parameters are forwarded
correctly -- in particular the 4000 nm threshold.

Coverage
  P1  discover_neuron_ids picks up neuron_<id>.csv and EXCLUDES the
      _spines.csv / _synapses.csv siblings
  P2  radius_report flags flat radii (the 50 nm fallback) and reports real ones
  P3  spine_node_counts counts spine nodes and roots independently
  P4  end-to-end compute_phi_factors: all four output files written, per-cell
      table correct, area conservation holds per cell, f_implied matches an
      independent recomputation
  P5  labeller kwargs forwarded (threshold 4000 nm reaches the labeller)
  P6  a neuron the labeller fails to return is skipped, not fatal
  P7  provenance JSON is valid, ASCII, and records the three design decisions
  P8  psi profile file is written and its pooled areas match the per-cell sum
"""

import json
import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import spine_density as sd
import phi_pipeline_colab as pipe


def _frus(r1, r2, L):
    return math.pi * (r1 + r2) * math.sqrt((r1 - r2) ** 2 + L ** 2)


# --------------------------------------------------------------------------- #
# Synthetic bank                                                              #
# --------------------------------------------------------------------------- #
def _neuron_frame(n_shaft=12, spine_at=(5, 9), radius_nm=500.0, flat=False):
    """Straight dendrite of n_shaft nodes at 10 um spacing, spines at given
    node indices. Returns an SWC-like frame in nm."""
    rows = [(0, -1, 0.0, 0.0, 0.0, 1000.0, "soma")]
    for k in range(1, n_shaft + 1):
        r = radius_nm if not flat else 50.0
        rows.append((k, k - 1, k * 10000.0, 0.0, 0.0, r, "dendrite"))
    nid_next = n_shaft + 1
    for base in spine_at:
        neck = nid_next
        head = nid_next + 1
        nid_next += 2
        x = base * 10000.0
        rows.append((neck, base, x, 0.0, 200.0,
                     50.0 if flat else 100.0, "neck"))
        rows.append((head, neck, x, 0.0, 500.0,
                     50.0 if flat else 300.0, "head"))
    return pd.DataFrame(
        rows, columns=["id", "p", "x", "y", "z", "r", "annotated_type"])


def _make_bank(root, ids=(101, 102, 103)):
    os.makedirs(root, exist_ok=True)
    frames = {}
    for i, nid in enumerate(ids):
        df = _neuron_frame(n_shaft=12 + i, spine_at=(5, 9), flat=(nid == 103))
        df.to_csv(os.path.join(root, "neuron_%d.csv" % nid), index=False)
        frames[nid] = df
    # decoys that discovery must ignore
    df.to_csv(os.path.join(root, "neuron_999_spines.csv"), index=False)
    df.to_csv(os.path.join(root, "neuron_999_synapses.csv"), index=False)
    with open(os.path.join(root, "notes.txt"), "w") as fh:
        fh.write("ignore me\n")
    return frames


class StubLabeller(object):
    """Identity labeller with label_dendritic_spines_robust's call contract."""

    __qualname__ = "StubLabeller"

    def __init__(self, drop_ids=()):
        self.calls = []
        self.drop_ids = set(drop_ids)

    def __call__(self, neuron_ids, input_dir=None, output_dir=None,
                 spine_length_threshold_nm=None, **kwargs):
        self.calls.append({
            "neuron_ids": list(neuron_ids),
            "input_dir": input_dir,
            "output_dir": output_dir,
            "spine_length_threshold_nm": spine_length_threshold_nm,
            "kwargs": dict(kwargs),
        })
        out = {}
        for nid in neuron_ids:
            if nid in self.drop_ids:
                continue
            path = os.path.join(input_dir, "neuron_%s.csv" % nid)
            if not os.path.exists(path):
                continue
            out[nid] = pd.read_csv(path)
        return out


# --------------------------------------------------------------------------- #
# P1 discovery                                                                #
# --------------------------------------------------------------------------- #
def test_P1_discovery():
    tmp = tempfile.mkdtemp()
    try:
        _make_bank(os.path.join(tmp, "bank"))
        ids = pipe.discover_neuron_ids(os.path.join(tmp, "bank"), verbose=False)
        assert ids == [101, 102, 103], ids
    finally:
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------- #
# P2 radius QC                                                                #
# --------------------------------------------------------------------------- #
def test_P2_radius_report():
    real = _neuron_frame(flat=False)
    rep = pipe.radius_report(real)
    assert rep["has_r_column"] is True
    assert rep["radius_suspect"] is False, rep
    assert rep["n_unique_r"] > 1
    assert rep["r_median_nm"] > 0

    # flat dendrite but a DISTINCT soma radius: the whole-cell uniqueness test
    # would see 2 values and pass; the non-soma test must still flag it.
    flat = _neuron_frame(flat=True)
    repf = pipe.radius_report(flat)
    assert repf["flat_radius"] is True, repf
    assert repf["radius_suspect"] is True, repf
    assert abs(repf["frac_at_default_r"] - 1.0) < 1e-12
    assert repf["n_nodes_nonsoma"] == len(flat) - 1

    # a cell whose radii are almost all at the 50 nm fallback but not constant
    dom = _neuron_frame(n_shaft=200, flat=True)   # realistic size for a 99% rule
    dom.loc[dom.index[-1], "r"] = 123.0
    repd = pipe.radius_report(dom)
    assert repd["flat_radius"] is False, repd
    assert repd["default_dominated"] is True, repd
    assert repd["radius_suspect"] is True, repd

    nocol = real.drop(columns=["r"])
    repn = pipe.radius_report(nocol)
    assert repn["has_r_column"] is False and repn["radius_suspect"] is True


# --------------------------------------------------------------------------- #
# P3 spine counts                                                             #
# --------------------------------------------------------------------------- #
def test_P3_spine_counts():
    df = _neuron_frame(spine_at=(5, 9))
    c = pipe.spine_node_counts(df)
    assert c["n_spine_nodes"] == 4, c    # 2 spines x (neck + head)
    assert c["n_spine_roots"] == 2, c    # 2 necks attached to shaft


# --------------------------------------------------------------------------- #
# P4/P5/P7/P8 end to end                                                      #
# --------------------------------------------------------------------------- #
def test_P4_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        bank = os.path.join(tmp, "bank")
        out = os.path.join(tmp, "phi_out")
        frames = _make_bank(bank)
        lab = StubLabeller()

        df_out, summary = pipe.compute_phi_factors(
            [101, 102, 103],
            output_path=out,
            output_filename="phi_test",
            input_dir=bank,
            label_fn=lab,
            verbose=False,
        )

        # --- P5 threshold forwarded ---
        assert len(lab.calls) == 1
        assert lab.calls[0]["spine_length_threshold_nm"] == 4000.0, lab.calls[0]
        assert lab.calls[0]["input_dir"] == bank

        # --- per-cell table ---
        assert list(df_out.index) == [101, 102, 103], df_out.index.tolist()
        assert summary["n_cells"] == 3
        assert summary["n_cells_radius_suspect"] == 1     # neuron 103
        assert bool(df_out.loc[103, "radius_suspect"]) is True
        assert bool(df_out.loc[101, "radius_suspect"]) is False

        # --- files written ---
        for fname in ("phi_test.csv", "phi_test.pkl",
                      "phi_test_psi_profile.csv", "phi_test_provenance.json"):
            p = os.path.join(out, fname)
            assert os.path.exists(p), "missing output %s" % fname
        for nid in (101, 102, 103):
            assert os.path.exists(os.path.join(out, "neuron_%d_phi.csv" % nid))

        # --- P4 independent recomputation of f_implied for neuron 101 ---
        df101 = frames[101]
        a_spine = 2 * (_frus(500, 100, 200) + _frus(100, 300, 300)) / 1e6
        a_shaft = (_frus(1000, 500, 10000)
                   + 11 * _frus(500, 500, 10000)) / 1e6
        expect_f = 1.0 + a_spine / a_shaft
        assert abs(float(df_out.loc[101, "f_implied"]) - expect_f) < 1e-9, (
            df_out.loc[101, "f_implied"], expect_f)
        assert abs(float(df_out.loc[101, "A_spine_um2"]) - a_spine) < 1e-9
        assert abs(float(df_out.loc[101, "A_shaft_um2"]) - a_shaft) < 1e-9
        assert int(df_out.loc[101, "n_spine_roots"]) == 2

        # --- F_lit: literature-comparable quantity, independently recomputed
        # from the persisted per-segment file using the same >=60um rule ---
        assert "F_lit" in df_out.columns
        assert float(df_out.loc[101, "F_lit_cutoff_um"]) == 60.0
        phi101_check = pd.read_csv(os.path.join(out, "neuron_101_phi.csv"))
        beyond = phi101_check[phi101_check["d_from_um"] >= 60.0]
        exp_F = 1.0 + beyond["spine_area_um2"].sum() / beyond["shaft_area_um2"].sum()
        assert abs(float(df_out.loc[101, "F_lit"]) - exp_F) < 1e-9, (
            df_out.loc[101, "F_lit"], exp_F)
        assert float(df_out.loc[101, "F_lit"]) >= float(df_out.loc[101, "f_implied"]) - 1e-9

        # --- conservation in the persisted per-cell phi file ---
        phi101 = pd.read_csv(os.path.join(out, "neuron_101_phi.csv"))
        recovered = float((phi101["phi_um"] * phi101["seg_len_um"]).sum())
        assert abs(recovered - a_spine) < 1e-9, (recovered, a_spine)
        assert abs(float(df_out.loc[101, "dropped_spine_area_um2"])) < 1e-12

        # --- P8 pooled psi profile ---
        prof = pd.read_csv(os.path.join(out, "phi_test_psi_profile.csv"))
        pooled_spine = float(prof["spine_area_um2"].sum())
        expect_pooled = float(df_out["A_spine_um2"].sum())
        assert abs(pooled_spine - expect_pooled) < 1e-9, (
            pooled_spine, expect_pooled)

        # --- P7 provenance ---
        with open(os.path.join(out, "phi_test_provenance.json"), "rb") as fh:
            raw = fh.read()
        assert all(b < 128 for b in raw), "provenance JSON must be ASCII"
        prov = json.loads(raw.decode("ascii"))
        assert prov["summary"]["proximal_cutoff_applied"] is False
        assert prov["summary"]["spine_area_attribution"] == "base_segment"
        assert prov["summary"]["phi_representation"] == \
            "piecewise_constant_per_shaft_segment"
        assert prov["summary"]["spine_length_threshold_nm"] == 4000.0
        assert "StubLabeller" in prov["phi_id"]
        assert prov["spine_density_module"]["module_version"] == sd.MODULE_VERSION
    finally:
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------- #
# P6 missing neuron is skipped, not fatal                                     #
# --------------------------------------------------------------------------- #
def test_P6_missing_neuron_skipped():
    tmp = tempfile.mkdtemp()
    try:
        bank = os.path.join(tmp, "bank")
        out = os.path.join(tmp, "phi_out")
        _make_bank(bank)
        lab = StubLabeller(drop_ids=(102,))
        df_out, summary = pipe.compute_phi_factors(
            [101, 102, 103], output_path=out, output_filename="phi_skip",
            input_dir=bank, label_fn=lab, verbose=False)
        assert list(df_out.index) == [101, 103], df_out.index.tolist()
        assert summary["n_cells"] == 2
        assert summary["n_cells_requested"] == 3
    finally:
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------- #
# runner                                                                      #
# --------------------------------------------------------------------------- #
def _run_all():
    tests = [
        ("P1 discovery excludes decoy files", test_P1_discovery),
        ("P2 radius QC flags flat radii", test_P2_radius_report),
        ("P3 spine node/root counts", test_P3_spine_counts),
        ("P4/P5/P7/P8 end-to-end + provenance", test_P4_end_to_end),
        ("P6 missing neuron skipped", test_P6_missing_neuron_skipped),
    ]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print("[PASS] %s" % name)
            n_pass += 1
        except AssertionError as e:
            print("[FAIL] %s -- %s" % (name, e))
        except Exception as e:  # noqa: BLE001
            print("[ERROR] %s -- %r" % (name, e))
    print("-" * 60)
    print("%d / %d passed" % (n_pass, len(tests)))
    return n_pass == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
