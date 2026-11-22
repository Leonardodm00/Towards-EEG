"""Spec-derived tests of Block 13 (twin_manifest.py; SPEC 2.7 (i) and the Block 13
entry). Integration tester, 2026-11-20.

Renewed 2026-11-22 (third incremental run) to the current SPEC: Block 16 adds a
fourth configuration (full_no_noise, n_no_noise = 5 by default, refused above
n_per_configuration), the arm of full and active_fit is ih5 (2.8), and the
truth of dv_h is 0 in every row with I_h unless dvh_truth_mV=None. The Block 13
oracles below are run on the three drawn configurations (n_no_noise=0) and,
for the drawn dv_h values, with dvh_truth_mV=None. Block 16's own oracles are in
test_spec_block16_twin_manifest_no_noise.py.

Expected values: the cell draw and the truth seed written out from the spec
with numpy's SeedSequence / default_rng; the truths from
synth_gt_grid.draw_manifest (an existing module outside the block, the
independent reference the spec names) called here with the arguments the spec
states.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import synth_gt_grid as G
import twin_manifest as tm

ID_BASE = 900_000_000


def _real(n=18):
    ids = np.array([508282493 + 1000 * k for k in range(n)], dtype=np.int64)[::-1]   # unsorted on purpose
    return pd.DataFrame({"specimen_id": ids, "swc_path": ["/x/specimen_%d/reconstruction.swc" % i for i in ids],
                         "v_rest_mV": np.linspace(-75.0, -65.0, n),
                         "cell_tau_s_above_window": [bool(k % 4 == 0) for k in range(n)]})


@pytest.mark.parametrize("seed,n", [(0, 10), (12345, 4)])
def test_draw_ids_cells_truths(seed, n):
    real = _real()
    kw = dict(ra_mode="per_cell", F=1.9, ih_gbar_floor_S_cm2=2e-6)
    man = tm.draw_twin_manifest(real, n_per_configuration=n, n_no_noise=0, dvh_truth_mV=None, seed=seed,
                                truth_kwargs=kw)
    assert len(man) == 3 * n
    assert list(man["configuration"]) == sum([[c] * n for c in ("full", "passive_fit", "active_fit")], [])
    by_id = real.set_index("specimen_id")
    sorted_ids = np.sort(real["specimen_id"].to_numpy())
    for q, conf in enumerate(("full", "passive_fit", "active_fit")):
        sub = man[man["configuration"] == conf].reset_index(drop=True)
        assert_array_equal(sub["specimen_id"], ID_BASE + 100 * q + np.arange(n))
        cells = np.random.default_rng(np.random.SeedSequence([seed, q])).choice(sorted_ids, size=n, replace=False)
        assert_array_equal(sub["twin_of_specimen_id"], cells)
        assert_array_equal(sub["v_rest_mV"], by_id.loc[cells, "v_rest_mV"].to_numpy())
        assert list(sub["real_swc_path"]) == list(by_id.loc[cells, "swc_path"])
        assert_array_equal(sub["cell_tau_s_above_window"], by_id.loc[cells, "cell_tau_s_above_window"].to_numpy())
        assert list(sub["arm"]) == [{"full": "ih5", "passive_fit": "passive_fullstep", "active_fit": "ih5"}[conf]] * n
        tseed = int(np.random.SeedSequence([seed, q, 1]).generate_state(1)[0])
        ref = G.draw_manifest(list(by_id.loc[cells, "swc_path"]), draws_per_morph=1, seed=tseed,
                              id_base=ID_BASE + 100 * q, use_ih=True, fp_control_frac=0.0, **kw)
        for col in ("cm_true", "rm_true", "ra_true"):
            assert_array_equal(sub[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float))
        if conf != "active_fit":
            for col in ("ih_gihbar_S_cm2", "ih_dvh_mV", "ih_kappa_tau"):
                assert_array_equal(sub[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float))
            assert not sub["is_fp_control"].any()
        else:
            assert (sub["ih_gihbar_S_cm2"] == 2e-6).all() and (sub["ih_dvh_mV"] == 0.0).all()
            assert (sub["ih_kappa_tau"] == 1.0).all() and sub["is_fp_control"].all()
    # no noise_, acq_ or e_pas column
    assert not [c for c in man.columns if c.startswith(("noise_", "acq_")) or c == "e_pas_mV"]
    assert (man["manifest_seed"] == seed).all()


def test_fp_values_match_draw_manifest_fp_rows():
    """active_fit's I_h truth = the values draw_manifest itself gives a
    false-positive row (fp_control_frac 0.5), with the default floor."""
    real = _real()
    man = tm.draw_twin_manifest(real, n_per_configuration=4, n_no_noise=0, seed=3)
    ref = G.draw_manifest(list(real["swc_path"][:4]), draws_per_morph=1, seed=1, fp_control_frac=0.5)
    fp = ref[ref["is_fp_control"].astype(bool)]
    assert len(fp) >= 1
    act = man[man["configuration"] == "active_fit"]
    for col in ("ih_gihbar_S_cm2", "ih_dvh_mV", "ih_kappa_tau"):
        assert set(act[col]) == set(fp[col].astype(float))


def test_cells_of_a_configuration_independent_of_others():
    real = _real()
    a = tm.draw_twin_manifest(real, n_per_configuration=3, n_no_noise=2, seed=9)
    b = tm.draw_twin_manifest(real, n_per_configuration=3, n_no_noise=2, seed=9,
                              truth_kwargs=dict(ra_mode="per_cell"))
    assert_array_equal(a["twin_of_specimen_id"], b["twin_of_specimen_id"])
    c = tm.draw_twin_manifest(real, n_per_configuration=3, n_no_noise=2, seed=10)
    assert not np.array_equal(a["twin_of_specimen_id"], c["twin_of_specimen_id"])


def test_round_trip_and_theta(tmp_path):
    real = _real()
    man = tm.draw_twin_manifest(real, n_per_configuration=3, n_no_noise=2, seed=1)
    p = tmp_path / "m.csv"
    tm.save_twin_manifest(man, p)
    back = tm.load_twin_manifest(p)
    pd.testing.assert_frame_equal(back, man)
    for _, r in back.iterrows():
        th = tm.twin_theta(r)
        assert th["Cm"] == r["cm_true"] and th["Rm"] == r["rm_true"] and th["Ra"] == r["ra_true"]
        if r["configuration"] == "active_fit":
            assert set(th) == {"Cm", "Rm", "Ra"}
        else:
            assert set(th) == {"Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau"}
            assert th["gbar"] == r["ih_gihbar_S_cm2"] and th["dv_h"] == r["ih_dvh_mV"]
            assert th["kappa_tau"] == r["ih_kappa_tau"]
        sp = tm.twin_spec_of(r, F=2.0)
        assert sp.twin_id == r["specimen_id"] and sp.twin_of_specimen_id == r["twin_of_specimen_id"]
        assert sp.F == 2.0 and sp.v_rest_mV == r["v_rest_mV"]


@pytest.mark.parametrize("case", ["n0", "n101", "n_too_many", "dup", "clash", "nan_vrest", "missing_col",
                                  "reserved", "n_float"])
def test_refusals(case):
    real = _real()
    kw = dict(n_per_configuration=2, n_no_noise=0, seed=0)
    if case == "n0":
        kw["n_per_configuration"] = 0
    elif case == "n101":
        real = _real(102)
        kw["n_per_configuration"] = 101
    elif case == "n_too_many":
        kw["n_per_configuration"] = 19
    elif case == "dup":
        real.loc[1, "specimen_id"] = real.loc[0, "specimen_id"]
    elif case == "clash":
        real.loc[0, "specimen_id"] = ID_BASE + 201
    elif case == "nan_vrest":
        real.loc[2, "v_rest_mV"] = np.nan
    elif case == "missing_col":
        real = real.drop(columns=["v_rest_mV"])
    elif case == "reserved":
        kw["truth_kwargs"] = dict(seed=3)
    elif case == "n_float":
        kw["n_per_configuration"] = 2.0
    with pytest.raises(ValueError):
        tm.draw_twin_manifest(real, **kw)
