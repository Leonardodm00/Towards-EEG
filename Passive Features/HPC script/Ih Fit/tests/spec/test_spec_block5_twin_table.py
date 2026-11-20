"""Spec-derived tests of Block 5 (twin_table.py; SPEC 2.7 (a), (b), section 3 and
the Block 5 entry with amendments 1 and 3). Integration tester, 2026-11-20.

The input cell is written with smoke_noise_windows.write_fixture (the writer-like
fixture of Block 3); the two noise-fit tables are written here by hand in the
format of section 3. Every expected value comes from the planted fixture or the
spec's definitions (t_on / t_off as sample indices of the planted current over
f_s, the _ms columns over 1000, the strict '>' of the flag).

The real-data tests (nf2 tables, cell 508282493) run only when the environment
variables TEEG_NF2_DIR and TEEG_ARCHIVE_CELL name the directories (the
tester runs them with `python3 -I -m pytest`).
"""
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

import twin_table as tt
from twin_config import TwinConfig

FS = 50000.0
LS_AMPS = (-110.0, -90.0, 5.0, 20.0, 50.0, 0.0)   # 59 (0 pA) is no step; 56 at exactly 5 pA is a step, not tested
SPEC_ID = 900000003                                # the fixture writer's specimen id


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    import smoke_noise_windows as sw
    d = tmp_path_factory.mktemp("b5") / ("specimen_%d" % SPEC_ID)
    truth = sw.write_fixture(d, np.random.default_rng(7), ls_amps=LS_AMPS, ls_tail_n=sw.LS_LONG_TAIL_N)
    return d, truth


CELL = dict(noise_sigma_f_mV=0.034, noise_sigma_m_mV=0.017, noise_tau_m_ms=0.3, noise_sigma_s_mV=0.06,
            noise_tau_s_ms=265.0)


def _write_tables(dirp, *, cell=None, status="ok", rows=None):
    cell = dict(CELL if cell is None else cell)
    cells = dirp / "noise_cells_G.csv"
    sweeps = dirp / "noise_sweeps_G.csv"
    with open(cells, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["specimen_id", "status", "reason"] + list(cell))
        w.writerow([SPEC_ID, status, ""] + [repr(v) for v in cell.values()])
    if rows is None:
        rows = default_rows()
    with open(sweeps, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["specimen_id", "noise_sweep_protocol", "noise_sweep_number", "noise_tilt_mV_per_s",
                    "noise_sweep_sigma_f_mV", "noise_sweep_sigma_m_mV", "noise_sweep_sigma_s_mV",
                    "noise_sweep_flags"])
        for r in rows:
            w.writerow([SPEC_ID] + [r[0], r[1], repr(r[2]), repr(r[3]), repr(r[4]), repr(r[5]), r[6]])
    return cells, sweeps


def default_rows():
    """(protocol, sweep, tilt, sigma_f, sigma_m, sigma_s, flags); sweep 56 failed (cell law)."""
    rows = [("SS", 92, 0.0123, 0.0331, 0.0172, 0.071, ""), ("SS", 93, -0.004, 0.0352, 0.0165, 0.058, "")]
    for k, s in enumerate((54, 55, 56, 57, 58)):
        if s == 56:
            rows.append(("LS", s, 0.002, CELL["noise_sigma_f_mV"], CELL["noise_sigma_m_mV"],
                         CELL["noise_sigma_s_mV"], "v_refits_failed:6/50;sweep_fit_failed:ls:56:at_bound:log_sigma_m"))
        else:
            rows.append(("LS", s, 0.001 * (k + 1), 0.03 + 0.001 * k, 0.015 + 0.001 * k, 0.05 + 0.002 * k, ""))
    return rows


def _spike_fn_recorder(fixture_dir, reason_for):
    calls = []
    with np.load(fixture_dir / "ls_sweeps.npz") as z:
        stored = {k: np.array(z[k]) for k in z.files}

    def fn(v, fs):
        calls.append((np.array(v), fs))
        for k, val in stored.items():
            if k.startswith("v_") and val.shape == v.shape and np.array_equal(val, v):
                return reason_for.get(int(k[2:]), "")
        return "UNMATCHED"
    return fn, calls


def test_protocol_and_spike_calls(fx):
    """2.7 (a), amendment 1 and 3: the 0 pA sweep is left out, ls_index matched
    by sweep number; spike_reason_fn called exactly for amplitude > 5 pA (the
    exact-5 pA sweep not tested), on the stored v_<index> and its rate."""
    d, truth = fx
    reason_for = {4: "action_potential"}                    # index 4 = sweep 58 (50 pA)
    fn, calls = _spike_fn_recorder(d, reason_for)
    p = tt.load_cell_protocol(d, spike_reason_fn=fn)
    assert p.specimen_id == SPEC_ID
    assert_array_equal(p.ls_sweep_number, [54, 55, 56, 57, 58])
    assert_array_equal(p.ls_index, [0, 1, 2, 3, 4])
    assert_array_equal(p.ls_amplitude_pA, LS_AMPS[:5])
    assert len(calls) == 2 and all(c[1] == FS for c in calls)
    assert list(p.ls_spike_reason) == ["", "", "", "", "action_potential"]
    # onset and duration from the planted current: on at sample 13500, off (first after) 63500
    assert np.all(np.abs(p.ls_onset_s - 13500 / FS) < 1e-12)
    assert np.all(np.abs(p.ls_duration_s - (63500 / FS - 13500 / FS)) < 1e-12)
    assert_array_equal(p.ls_n_samples, 13500 + 50000 + 75000)
    # SS windows: planted positions, polarities and sweep numbers, archive order
    assert_array_equal(p.ss_position, truth["positions"])
    assert_array_equal(p.ss_sweep_number, truth["sweeps"])
    assert_array_equal(p.ss_polarity_is_dep, truth["pol"])
    assert p.ss_period_samples == 10025 and p.ss_pre_n == 500 and p.ss_t_s.size == 10500


def test_table_values_units_order(fx, tmp_path):
    d, truth = fx
    fn, _ = _spike_fn_recorder(d, {4: "action_potential"})
    p = tt.load_cell_protocol(d, spike_reason_fn=fn)
    cells, sweeps = tt.read_noise_tables(*_write_tables(tmp_path))
    tab = tt.build_twin_sweep_table([p], cells, sweeps)
    assert tuple(tab.columns) == tt.TWIN_SWEEP_COLUMNS
    assert list(zip(tab["protocol"], tab["sweep_number"])) == [("SS", 92), ("SS", 93)] + [
        ("LS", s) for s in (54, 55, 56, 57, 58)]
    rows = {(r[0], r[1]): r for r in default_rows()}
    for _, r in tab.iterrows():
        exp = rows[(r["protocol"], r["sweep_number"])]
        assert r["tilt_mV_per_s"] == exp[2]
        assert (r["sigma_f_mV"], r["sigma_m_mV"], r["sigma_s_mV"]) == exp[3:6]
        assert r["tau_m_s"] == CELL["noise_tau_m_ms"] / 1000.0
        assert r["tau_s_s"] == CELL["noise_tau_s_ms"] / 1000.0
        assert r["noise_flags"] == exp[6]
        assert r["law_source"] == ("cell" if r["sweep_number"] == 56 else "sweep")
        # tau_s = 265 ms exactly: NOT above the window (strict '>')
        assert r["cell_tau_s_above_window"] is False or r["cell_tau_s_above_window"] == False  # noqa: E712
    ss = tab[tab["protocol"] == "SS"]
    for s, k in zip(ss["sweep_number"], ss["n_windows"]):
        assert k == int(np.sum(truth["sweeps"] == s))
    assert ss["amplitude_pA"].isna().all() and ss["onset_s"].isna().all() and ss["duration_s"].isna().all()
    assert (ss["n_samples"] == 10500).all() and (ss["real_spike_reason"] == "").all() and ss["in_twin"].all()
    ls = tab[tab["protocol"] == "LS"]
    assert (ls["n_windows"] == 1).all()
    assert list(ls["in_twin"]) == [True, True, True, True, False]
    assert list(ls["real_spike_reason"]) == ["", "", "", "", "action_potential"]
    assert tab["specimen_id"].dtype == np.int64 and tab["sigma_f_mV"].dtype == np.float64
    # round trip
    path = tmp_path / "t.csv"
    tt.write_twin_sweep_table(tab, path)
    back = tt.read_twin_sweep_table(path)
    pd.testing.assert_frame_equal(back, tab)
    law, tilt = tt.sweep_law(back, SPEC_ID, "LS", 55)
    exp = rows[("LS", 55)]
    assert (law.sigma_f, law.sigma_m, law.tau_m, law.sigma_s, law.tau_s) == (exp[3], exp[4], 0.3 / 1000.0, exp[5],
                                                                              265.0 / 1000.0)
    assert tilt == exp[2]


def test_flag_strict_boundary(fx, tmp_path):
    cells, sweeps = tt.read_noise_tables(*_write_tables(tmp_path, cell=dict(CELL, noise_tau_s_ms=265.1)))
    laws = tt.sweep_laws_from_tables(cells, sweeps)
    assert laws["cell_tau_s_above_window"].all()


@pytest.mark.parametrize("case", ["status", "missing_row", "extra_row", "nan_tilt", "failed_altered",
                                  "bad_law", "dup_row"])
def test_refusals(fx, tmp_path, case):
    d, _ = fx
    p = tt.load_cell_protocol(d, spike_reason_fn=lambda v, fs: "")
    rows = default_rows()
    status = "ok"
    if case == "status":
        status = "failed"
    elif case == "missing_row":
        rows = [r for r in rows if r[1] != 57]
    elif case == "extra_row":
        rows = rows + [("LS", 99, 0.0, 0.03, 0.01, 0.05, "")]
    elif case == "nan_tilt":
        rows[3] = rows[3][:2] + (float("nan"),) + rows[3][3:]
    elif case == "failed_altered":
        i = [r[1] for r in rows].index(56)
        rows[i] = rows[i][:3] + (CELL["noise_sigma_f_mV"] * (1 + 1e-7),) + rows[i][4:]
    elif case == "bad_law":
        rows[0] = rows[0][:3] + (-0.01,) + rows[0][4:]
    elif case == "dup_row":
        rows = rows + [rows[0]]
    cells, sweeps = tt.read_noise_tables(*_write_tables(tmp_path, status=status, rows=rows))
    with pytest.raises(tt.TwinTableError) as ei:
        tt.build_twin_sweep_table([p], cells, sweeps)
    assert str(SPEC_ID) in str(ei.value)


def test_non_str_reason_refused(fx):
    d, _ = fx
    with pytest.raises(tt.TwinTableError):
        tt.load_cell_protocol(d, spike_reason_fn=lambda v, fs: None)


# ---------------------------------------------------------------------------
#  Real data (section 3, data inspected): nf2 tables and the archive cell
# ---------------------------------------------------------------------------
NF2 = os.environ.get("TEEG_NF2_DIR")
CELLDIR = os.environ.get("TEEG_ARCHIVE_CELL")


@pytest.mark.skipif(not NF2, reason="TEEG_NF2_DIR not set")
def test_nf2_laws_against_json():
    """TT8's first half from the spec: 589 rows (64 SS, 525 LS), 9 'cell'; each
    law equals the JSON's twin_laws to 1e-12 relative (a %.17g-printed float
    through a division by 1000: one or two ulps), each tilt exactly; the
    flagged cells are exactly D-182.3's four."""
    nf2 = Path(NF2)
    cells, sweeps = tt.read_noise_tables(nf2 / "noise_cells_L3_exc.csv", nf2 / "noise_sweeps_L3_exc.csv")
    laws = tt.sweep_laws_from_tables(cells, sweeps)
    assert len(laws) == 589
    assert (laws["protocol"] == "SS").sum() == 64 and (laws["protocol"] == "LS").sum() == 525
    assert (laws["law_source"] == "cell").sum() == 9
    flagged = sorted(set(int(s) for s, f in zip(laws["specimen_id"], laws["cell_tau_s_above_window"]) if f))
    assert flagged == sorted([571430815, 526785799, 571624532, 548408551])
    worst = 0.0
    for sid, grp in laws.groupby("specimen_id"):
        js = json.load(open(nf2 / "cells" / ("specimen_%d.json" % sid)))
        tw = js["sweep_stage"]["twin_laws"]
        tilts = {(r["protocol"], int(r["sweep"])): r["tilt_mV_per_s"] for r in js["sweep_stage"]["rows"]}
        for _, r in grp.iterrows():
            ref = tw["%s:%d" % (r["protocol"].lower(), r["sweep_number"])]
            for col in ("sigma_f_mV", "sigma_m_mV", "tau_m_s", "sigma_s_mV", "tau_s_s"):
                worst = max(worst, abs(r[col] - ref[col]) / abs(ref[col]))
            assert r["tilt_mV_per_s"] == tilts[(r["protocol"].lower(), r["sweep_number"])]
    assert worst < 1e-12, worst
    # the cells table's tau_s against the flag (strict '>' 0.265 s), independently
    tau_s = {int(a): b / 1000.0 for a, b in zip(cells["specimen_id"], cells["noise_tau_s_ms"])}
    assert flagged == sorted(k for k, v in tau_s.items() if v > 0.265)


@pytest.mark.skipif(not (NF2 and CELLDIR), reason="TEEG_NF2_DIR / TEEG_ARCHIVE_CELL not set")
def test_real_cell_table():
    """TT8's second half: 508282493 -> 2 SS rows (23, 17 windows, 10500 samples),
    22 LS rows (onset 0.27 s, duration 1.0 s, 363500 samples, -110 .. +310 pA by 20)."""
    nf2 = Path(NF2)
    cells, sweeps = tt.read_noise_tables(nf2 / "noise_cells_L3_exc.csv", nf2 / "noise_sweeps_L3_exc.csv")
    p = tt.load_cell_protocol(CELLDIR, spike_reason_fn=lambda v, fs: "")
    tab = tt.build_twin_sweep_table([p], cells, sweeps)
    ss = tab[tab["protocol"] == "SS"]
    ls = tab[tab["protocol"] == "LS"]
    assert list(ss["n_windows"]) == [23, 17] and (ss["n_samples"] == 10500).all()
    assert len(ls) == 22
    assert np.all(np.abs(ls["onset_s"] - 0.27) < 1e-12) and np.all(np.abs(ls["duration_s"] - 1.0) < 1e-12)
    assert (ls["n_samples"] == 363500).all()
    assert np.all(np.abs(np.sort(ls["amplitude_pA"].to_numpy()) - np.arange(-110, 311, 20)) < 1e-3)  # detected amplitudes
    # the current really is on 13500 .. 63499 in every stored sweep (section 3)
    with np.load(Path(CELLDIR) / "ls_sweeps.npz") as z:
        for k in p.ls_index:
            i = z["i_%d" % k]
            on = np.flatnonzero(np.abs(i) > 0.5 * np.abs(i).max())
            assert on[0] == 13500 and on[-1] == 63499
