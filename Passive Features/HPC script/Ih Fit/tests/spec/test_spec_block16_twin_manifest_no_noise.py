"""Spec-derived tests of Block 16 (twin_manifest.py; SPEC 2.8 (e) and the Block 16
entry). Integration tester, 2026-11-22.

Expected values: the first 3 n rows are compared with twin_manifest.py as it
stood at the base commit 216ecf1 (`git show`; the spec's TM11 reference, "the
manifest of the contract commit's code for the same arguments"), apart from
`arm` (ih6 -> ih5) and `ih_dvh_mV` (0); the noise-free rows are written out
from the rule of the entry (the i-th a copy of the i-th full row, id
id_base + 300 + i).
"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import twin_manifest as tm

BASE = "216ecf13ccca455903b2cb05d740dd64f3feff08"
CODE_DIR = Path(__file__).resolve().parents[2]
ID_BASE = 900_000_000
KW = dict(ra_mode="per_cell", F=1.9, ih_dvh_range_mV=(-4.0, 4.0))   # a drawn dv_h, so that 0 is visible


def _real(n=18):
    ids = np.array([508282493 + 1000 * k for k in range(n)], dtype=np.int64)[::-1]
    return pd.DataFrame({"specimen_id": ids, "swc_path": ["/x/specimen_%d/reconstruction.swc" % i for i in ids],
                         "v_rest_mV": np.linspace(-75.0, -65.0, n),
                         "cell_tau_s_above_window": [bool(k % 4 == 0) for k in range(n)]})


def _old(_c={}):
    if "m" in _c:
        return _c["m"]
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=CODE_DIR, capture_output=True, text=True,
                         check=True).stdout.strip()
    rel = (CODE_DIR.relative_to(top) / "twin_manifest.py").as_posix()
    src = subprocess.run(["git", "show", "%s:%s" % (BASE, rel)], cwd=top, capture_output=True, check=True).stdout
    p = Path(tempfile.mkdtemp()) / "twin_manifest_base.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("twin_manifest_base", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _c["m"] = m
    return m


def test_constants():
    assert tm.CONFIGURATIONS == ("full", "passive_fit", "active_fit", "full_no_noise")
    assert tm.ARM_OF == {"full": "ih5", "passive_fit": "passive_fullstep", "active_fit": "ih5",
                         "full_no_noise": "ih5"}
    assert tm.HAS_IH == {"full": True, "passive_fit": True, "active_fit": False, "full_no_noise": True}
    assert tm.WITH_NOISE == {"full": True, "passive_fit": True, "active_fit": True, "full_no_noise": False}


@pytest.mark.parametrize("seed,n,n0", [(0, 10, 5), (3, 4, 4), (11, 3, 0), (5, 2, 1)])
def test_first_rows_equal_base_code(seed, n, n0):
    real = _real()
    new = tm.draw_twin_manifest(real, n_per_configuration=n, n_no_noise=n0, seed=seed, truth_kwargs=KW)
    old = _old().draw_twin_manifest(real, n_per_configuration=n, seed=seed, truth_kwargs=KW)
    assert len(new) == 3 * n + n0
    head = new.iloc[:3 * n].reset_index(drop=True)
    assert list(head.columns) == list(old.columns) + ["with_noise", "paired_with"]   # appended last
    for col in old.columns:
        if col == "arm":
            exp = old[col].replace({"ih6": "ih5"})
            assert list(head[col]) == list(exp)
        elif col == "ih_dvh_mV":
            has_ih = old["configuration"].isin(["full", "passive_fit"])
            assert (head.loc[has_ih, col] == 0.0).all()
            assert_array_equal(head.loc[~has_ih, col], old.loc[~has_ih, col])       # active_fit: the fp value
        else:
            pd.testing.assert_series_equal(head[col], old[col], check_names=False)
    assert old.loc[old["configuration"] == "full", "ih_dvh_mV"].abs().max() > 0   # the draw had nonzero dv_h
    # dvh_truth_mV=None: the drawn dv_h bit for bit
    keep = tm.draw_twin_manifest(real, n_per_configuration=n, n_no_noise=n0, seed=seed, dvh_truth_mV=None,
                                 truth_kwargs=KW)
    assert_array_equal(keep["ih_dvh_mV"].to_numpy()[:3 * n], old["ih_dvh_mV"].to_numpy())
    # another value is set on every row with I_h, the paired rows included
    sh = tm.draw_twin_manifest(real, n_per_configuration=n, n_no_noise=n0, seed=seed, dvh_truth_mV=-2.5,
                               truth_kwargs=KW)
    has = sh["configuration"].map(tm.HAS_IH).astype(bool)
    assert (sh.loc[has, "ih_dvh_mV"] == -2.5).all()


@pytest.mark.parametrize("n,n0", [(10, 5), (4, 4), (3, 1)])
def test_no_noise_rows(n, n0):
    real = _real()
    man = tm.draw_twin_manifest(real, n_per_configuration=n, n_no_noise=n0, seed=2, truth_kwargs=KW)
    full = man[man["configuration"] == "full"].reset_index(drop=True)
    nn = man[man["configuration"] == "full_no_noise"].reset_index(drop=True)
    assert len(nn) == n0
    assert list(man["configuration"]) == (["full"] * n + ["passive_fit"] * n + ["active_fit"] * n
                                          + ["full_no_noise"] * n0)
    assert_array_equal(nn["specimen_id"], ID_BASE + 300 + np.arange(n0))
    assert_array_equal(nn["paired_with"], full["specimen_id"].iloc[:n0])
    assert (nn["arm"] == "ih5").all() and (~nn["with_noise"].astype(bool)).all()
    other = man[man["configuration"] != "full_no_noise"]
    assert other["with_noise"].astype(bool).all() and (other["paired_with"] == -1).all()
    skip = {"specimen_id", "configuration", "arm", "with_noise", "paired_with"}
    for col in man.columns:
        if col in skip:
            continue
        pd.testing.assert_series_equal(nn[col], full[col].iloc[:n0], check_names=False)
    for i in range(n0):
        assert tm.twin_theta(nn.iloc[i]) == tm.twin_theta(full.iloc[i])
        assert tm.twin_spec_of(nn.iloc[i], F=1.9).configuration == "full_no_noise"
    assert man["specimen_id"].is_unique
    assert man["with_noise"].dtype == bool and man["paired_with"].dtype == np.int64


def test_round_trip_new_columns(tmp_path):
    man = tm.draw_twin_manifest(_real(), n_per_configuration=3, n_no_noise=2, seed=4)
    p = tmp_path / "m.csv"
    tm.save_twin_manifest(man, p)
    back = tm.load_twin_manifest(p)
    pd.testing.assert_frame_equal(back, man)
    assert back["with_noise"].dtype == bool and back["paired_with"].dtype == np.int64


@pytest.mark.parametrize("bad", [-1, 4, 2.0, True, "2", None])
def test_n_no_noise_refusals(bad):
    with pytest.raises(ValueError):
        tm.draw_twin_manifest(_real(), n_per_configuration=3, n_no_noise=bad, seed=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0", True])
def test_dvh_truth_refusals(bad):
    with pytest.raises(ValueError):
        tm.draw_twin_manifest(_real(), n_per_configuration=3, n_no_noise=1, seed=0, dvh_truth_mV=bad)


def test_default_n_no_noise_and_id_clash():
    real = _real()
    man = tm.draw_twin_manifest(real, seed=0)            # the defaults: 10 per configuration, 5 noise-free
    assert len(man) == 35 and (man["configuration"] == "full_no_noise").sum() == 5
    with pytest.raises(ValueError):                      # the default 5 > n = 3
        tm.draw_twin_manifest(real, n_per_configuration=3, seed=0)
    clash = real.copy()
    clash.loc[0, "specimen_id"] = ID_BASE + 301          # the second noise-free twin's id
    with pytest.raises(ValueError):
        tm.draw_twin_manifest(clash, n_per_configuration=3, n_no_noise=2, seed=0)
    tm.draw_twin_manifest(clash, n_per_configuration=3, n_no_noise=1, seed=0)   # that id is not used


def test_ascii_lf():
    raw = (CODE_DIR / "twin_manifest.py").read_bytes()
    assert all(c < 128 for c in raw) and b"\r" not in raw
