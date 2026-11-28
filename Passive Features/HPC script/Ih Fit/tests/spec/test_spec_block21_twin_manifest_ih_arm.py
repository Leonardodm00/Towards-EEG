"""Spec-derived tests of Block 21 (twin_manifest.py; SPEC 2.9 (d) and the Block 21
entry). Integration tester, 2026-11-24.

Expected values: the arm table written out from the entry; the truth columns
from synth_gt_grid.draw_manifest called directly with the arguments the Block 13
entry states (seed SeedSequence([seed, q, 1]).generate_state(1)[0], id_base +
100 q, draws_per_morph=1, use_ih=True, fp_control_frac=0.0), the cells from
default_rng(SeedSequence([seed, q])).choice; M1's reference is twin_manifest.py
at the base commit of this run (c974a62, `git show`), the code the entry says
`ih_arm="ih5", dvh_truth_mV=0.0` must reproduce.
"""
import importlib.util
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import synth_gt_grid as G
import twin_manifest as tm

BASE = "c974a629c7230c94536489435591c67ca6dd2aab"
CODE_DIR = Path(__file__).resolve().parents[2]
ID_BASE = 900_000_000
TRUTH_COLS = ("cm_true", "rm_true", "ra_true", "ih_gihbar_S_cm2", "ih_dvh_mV", "ih_kappa_tau")


def _real(n=18):
    ids = np.array([508282493 + 1000 * k for k in range(n)], dtype=np.int64)[::-1]
    return pd.DataFrame({"specimen_id": ids, "swc_path": ["/x/specimen_%d/reconstruction.swc" % i for i in ids],
                         "v_rest_mV": np.linspace(-75.0, -65.0, n),
                         "cell_tau_s_above_window": [bool(k % 4 == 0) for k in range(n)]})


def _base_module(_c={}):
    if "m" in _c:
        return _c["m"]
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=CODE_DIR, capture_output=True, text=True,
                         check=True).stdout.strip()
    rel = (CODE_DIR.relative_to(top) / "twin_manifest.py").as_posix()
    src = subprocess.run(["git", "show", "%s:%s" % (BASE, rel)], cwd=top, capture_output=True, check=True).stdout
    p = Path(tempfile.mkdtemp()) / "twin_manifest_b21base.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("twin_manifest_b21base", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _c["m"] = m
    return m


def _draw_ref(real, n, seed, q, kw):
    """The cells and the truths of configuration q as the entries of Blocks 13 and 21 state them."""
    ids = np.sort(real["specimen_id"].to_numpy(dtype=np.int64))
    cells = np.random.default_rng(np.random.SeedSequence([seed, q])).choice(ids, size=n, replace=False)
    swc = dict(zip(real["specimen_id"].astype(np.int64), real["swc_path"]))
    tseed = int(np.random.SeedSequence([seed, q, 1]).generate_state(1)[0])
    t = G.draw_manifest([swc[int(c)] for c in cells], draws_per_morph=1, seed=tseed, id_base=ID_BASE + 100 * q,
                        use_ih=True, fp_control_frac=0.0, **kw)
    return cells, t.reset_index(drop=True)


def test_constants_and_arm_of():
    assert tm.IH_ARMS == ("ih5", "ih6_d207")
    for ih in tm.IH_ARMS:
        for conf in ("full", "active_fit", "full_no_noise"):
            assert tm.arm_of(conf, ih) == ih
        assert tm.arm_of("passive_fit", ih) == "passive_fullstep"
    for bad in [("full", "ih6"), ("full", "IH5"), ("nope", "ih5"), ("full", None), (None, "ih5")]:
        with pytest.raises(ValueError):
            tm.arm_of(*bad)
    # ARM_OF stays the ih5 table
    assert tm.ARM_OF == {"full": "ih5", "passive_fit": "passive_fullstep", "active_fit": "ih5",
                         "full_no_noise": "ih5"}


@pytest.mark.parametrize("seed,n,n0,kw", [(0, 10, 5, {}), (7, 3, 2, dict(ra_mode="per_cell")),
                                          (2, 4, 0, dict(ih_gbar_floor_S_cm2=1e-5))])
def test_M1_ih5_zero_reproduces_base(seed, n, n0, kw):
    real = _real()
    new = tm.draw_twin_manifest(real, n_per_configuration=n, n_no_noise=n0, seed=seed, ih_arm="ih5",
                                dvh_truth_mV=0.0, truth_kwargs=kw)
    old = _base_module().draw_twin_manifest(real, n_per_configuration=n, n_no_noise=n0, seed=seed,
                                            dvh_truth_mV=0.0, truth_kwargs=kw)
    pd.testing.assert_frame_equal(new, old)


def test_M2_defaults_with_range():
    real = _real()
    seed, n, n0 = 4, 6, 3
    kw = dict(ih_dvh_range_mV=(-5.0, 5.0), ih_gbar_floor_S_cm2=1e-5)
    df = tm.draw_twin_manifest(real, n_per_configuration=n, n_no_noise=n0, seed=seed, ih_arm="ih6_d207",
                               truth_kwargs=kw)   # renewed 2026-11-28 (Block 30, D-282): the default arm is ih5; this check is stated for ih6_d207, named here
    exp_arm = {"full": "ih6_d207", "passive_fit": "passive_fullstep", "active_fit": "ih6_d207",
               "full_no_noise": "ih6_d207"}
    assert list(df["arm"]) == [exp_arm[c] for c in df["configuration"]]
    for q, conf in enumerate(("full", "passive_fit", "active_fit")):
        cells, t = _draw_ref(real, n, seed, q, kw)
        sub = df[df["configuration"] == conf].reset_index(drop=True)
        assert_array_equal(sub["twin_of_specimen_id"].to_numpy(), cells)
        if conf != "active_fit":
            for col in TRUTH_COLS:
                assert_array_equal(sub[col].to_numpy(dtype=float), t[col].to_numpy(dtype=float), err_msg=col)
            dv = sub["ih_dvh_mV"].to_numpy(dtype=float)
            assert np.all((dv >= -5.0) & (dv <= 5.0)) and len(set(dv)) > 1
        else:
            assert np.all(sub["ih_dvh_mV"] == 0.0)
            assert np.all(sub["ih_gihbar_S_cm2"] == 1e-5)
            assert np.all(sub["ih_kappa_tau"] == 1.0)
            # the passive truths are the drawer's
            for col in ("cm_true", "rm_true", "ra_true"):
                assert_array_equal(sub[col].to_numpy(dtype=float), t[col].to_numpy(dtype=float), err_msg=col)
    full = df[df["configuration"] == "full"].reset_index(drop=True)
    nn = df[df["configuration"] == "full_no_noise"].reset_index(drop=True)
    assert len(nn) == n0
    for i in range(n0):
        for col in TRUTH_COLS + ("twin_of_specimen_id", "v_rest_mV", "real_swc_path"):
            assert nn.loc[i, col] == full.loc[i, col], col
        assert nn.loc[i, "paired_with"] == full.loc[i, "specimen_id"]
        assert nn.loc[i, "specimen_id"] == ID_BASE + 300 + i


def test_default_without_range_gives_zero_dvh():
    df = tm.draw_twin_manifest(_real(), n_per_configuration=3, n_no_noise=1, seed=1, ih_arm="ih6_d207")   # renewed 2026-11-28 (Block 30, D-282): the default arm is ih5; this check is stated for ih6_d207, named here
    assert set(df["arm"]) == {"ih6_d207", "passive_fullstep"}
    assert np.all(df["ih_dvh_mV"].to_numpy(dtype=float) == 0.0)


def test_float_dvh_sets_every_ih_row_and_active_stays_zero():
    df = tm.draw_twin_manifest(_real(), n_per_configuration=3, n_no_noise=2, seed=1, dvh_truth_mV=2.5,
                               ih_arm="ih6_d207",   # renewed 2026-11-28 (Block 30, D-282): the default arm is ih5; this check is stated for ih6_d207, named here
                               truth_kwargs=dict(ih_dvh_range_mV=(-5.0, 5.0)))
    has = df["configuration"].map(tm.HAS_IH).to_numpy(dtype=bool)
    assert np.all(df.loc[has, "ih_dvh_mV"] == 2.5)
    assert np.all(df.loc[~has, "ih_dvh_mV"] == 0.0)


def test_M3_ih5_refuses_nonzero_dvh():
    with pytest.raises(ValueError):
        tm.draw_twin_manifest(_real(), n_per_configuration=3, n_no_noise=1, seed=1, ih_arm="ih5",
                              truth_kwargs=dict(ih_dvh_range_mV=(-5.0, 5.0)))
    with pytest.raises(ValueError):
        tm.draw_twin_manifest(_real(), n_per_configuration=3, n_no_noise=1, seed=1, ih_arm="ih5",
                              dvh_truth_mV=1.0)
    df = tm.draw_twin_manifest(_real(), n_per_configuration=3, n_no_noise=1, seed=1, ih_arm="ih5",
                               dvh_truth_mV=0.0, truth_kwargs=dict(ih_dvh_range_mV=(-5.0, 5.0)))
    assert np.all(df["ih_dvh_mV"] == 0.0) and set(df["arm"]) == {"ih5", "passive_fullstep"}


@pytest.mark.parametrize("bad", ["ih6", "", None, 5, "passive_fullstep"])
def test_M4_unknown_arm(bad):
    with pytest.raises(ValueError):
        tm.draw_twin_manifest(_real(), n_per_configuration=2, n_no_noise=0, seed=0, ih_arm=bad)


def test_M5_round_trip(tmp_path):
    df = tm.draw_twin_manifest(_real(), n_per_configuration=4, n_no_noise=2, seed=9, ih_arm="ih6_d207",   # renewed 2026-11-28 (Block 30, D-282): the default arm is ih5; this check is stated for ih6_d207, named here
                               truth_kwargs=dict(ih_dvh_range_mV=(-5.0, 5.0)))
    p = tmp_path / "m.csv"
    tm.save_twin_manifest(df, p)
    back = tm.load_twin_manifest(p)
    pd.testing.assert_frame_equal(back, df)
    # twin_theta of a with-I_h row carries the drawn dv_h (six names)
    r = back.iloc[0]
    th = tm.twin_theta(r)
    assert set(th) == {"Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau"}
    assert th["dv_h"] == float(df.iloc[0]["ih_dvh_mV"])


def test_ascii_lf():
    raw = (CODE_DIR / "twin_manifest.py").read_bytes()
    assert all(c < 128 for c in raw) and b"\r" not in raw
