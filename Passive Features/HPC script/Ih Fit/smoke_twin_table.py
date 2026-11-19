"""Smoke test for the twins' table -- Block 5 in specs/SPEC.md (TT1-TT9).

Checks (the oracles of SPEC, Block 5, "Test oracles")
    test_tt1_reference      two fixture cells and two hand-written noise-fit
                            tables: every cell of the built table equals the
                            value planted here (floats exactly)
    test_tt2_units_case     tau_*_s = the _ms column / 1000 exactly; protocol
                            upper case; sweep_law returns the row's law and
                            tilt exactly
    test_tt3_failed_sweep   law_source 'cell', the cell's law, an altered SD
                            refused (and one inside failed_law_rtol accepted)
    test_tt4_refusals       every refusal of the entry raises TwinTableError
                            naming the cell (and the sweep where there is one)
    test_tt5_round_trip     write / read: equal values, dtypes, column order
    test_tt6_slow_flag      tau_s 0.2650 s -> False, 0.2651 s -> True, on
                            every row of the cell
    test_tt7_protocol       load_cell_protocol returns the planted layout
    test_tt8_real_run       the nf2 tables and JSONs (--nf2-dir) and the
                            archive cell 508282493 (--archive-cell); SKIP
                            with the reason when a path is not given; its
                            third part (amendment 1) compares the in_twin
                            counts with the fitter's loader's ls_dep_screen
                            and needs passive_fitting_hpc_fixed (NEURON)
    test_tt9_ascii_lf       the two files are pure ASCII with LF line ends
    test_tt11_not_a_step    amendment 3 (D-191.4): listed LS sweeps that are
                            not steps (0 pA, a null amplitude, below 5 pA in
                            size) are left out without a row; the remaining
                            sweeps keep their own ls_index, number, amplitude
    test_tt10_spike_screen  amendment 1 (D-190): the spike test is called
                            exactly for the LS sweeps above
                            ls_dep_min_amplitude_pA (not at it), on their
                            whole stored voltage; the planted reasons and
                            in_twin come back; a non-str return is refused
    test_convergence, test_determinism: SKIP (no discretisation, no
                            randomness in Block 5)

Run (from the code folder)
    python smoke_twin_table.py
    python smoke_twin_table.py --nf2-dir /path/to/nf2 --archive-cell /path/to/specimen_508282493
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.
    python -m pytest smoke_twin_table.py -q
        The same checks through pytest (TT8 then SKIPs: no paths).

The fixture cells are written by smoke_noise_windows.write_fixture into a
temporary directory, removed at exit. The tables are written by hand with
run_noise_fit's columns and CSV writer (the formats of SPEC section 3).

Keep this file pure ASCII so it survives transfer to the cluster.
"""
from __future__ import annotations

import argparse
import atexit
import importlib.metadata
import json
import math
import platform
import re
import shutil
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal  # noqa: F401

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import twin_table as tt                                       # noqa: E402
from noise_law import NoiseLaw                                # noqa: E402
from run_noise_fit import CELL_COLUMNS, SWEEP_COLUMNS, write_csv_atomic   # noqa: E402
from twin_config import TwinConfig                            # noqa: E402
import smoke_noise_windows as snw                             # noqa: E402

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy", "pandas")
ARTIFACT_DIR = HERE / "tests" / "smoke" / "artifacts"
_SAVED_ARTIFACTS = []
ARGS = argparse.Namespace(nf2_dir=None, archive_cell=None)   # set by main(); pytest leaves them None

# --------------------------------------------------------------- fixture ---
# Two fixture cells (A, B) in the archive, a third (C) only in the tables, a
# refused cell (D) in the per-cell table. All values planted here.
ID_A, ID_B, ID_C, ID_D = 900000101, 900000102, 900000103, 900000104
ID_F = 900000106                     # TT10: LS sweeps around the depolarising threshold
ID_G = 900000107                     # TT11: listed LS sweeps that are not steps, between real steps
# written amplitudes; the metadata of sweep 57 (index 3) is then set to null; 55 (0 pA) and 58 (3 pA) are no steps
LS_AMPS_G = (-110.0, 0.0, -70.0, 30.0, 3.0, 90.0)
G_NULL_SWEEP = 57
G_STEPS = (54, 56, 59)               # sweep numbers left; their index in ls_sweeps.npz is sweep - 54
LS_AMPS_F = (-110.0, -30.0, 5.0, 20.0, 50.0, 130.0)   # 5.0 pA: exactly the threshold, not tested
# planted spike-test reasons (amendment 1, D-190): {(specimen, LS sweep): reason}; any other '' (passes)
REASONS = {(ID_A, 56): "peak_above_threshold", (ID_B, 57): "action_potential",
           (ID_F, 58): "action_potential", (ID_F, 59): "unusable"}
SWEEPS_A, DROPPED_A, LS_AMPS_A = (92, 93), snw.DROPPED, (-110.0, -90.0, 50.0)
SWEEPS_B, DROPPED_B, LS_AMPS_B = (7,), {7: (1, 4, 30)}, (-70.0, 30.0, 130.0, 230.0)
# per-cell laws (mV, mV, ms, mV, ms); B's tau_s 265.1 ms (flag True), A's 265.0 ms (flag False)
CELL_LAWS = {ID_A: (0.031, 0.017, 0.0713, 0.08, 265.0),
             ID_B: (0.0345, 0.0222, 0.0569, 0.2818, 265.1),
             ID_C: (0.032, 0.0165, 0.0698, 0.0859, 1277.0584282427603),
             ID_F: (0.033, 0.0175, 0.0701, 0.0901, 150.0),
             ID_G: (0.034, 0.0185, 0.0702, 0.0911, 140.0)}
FAILED_B = 55                        # B's LS sweep 55 failed: its row holds B's SDs
FAILED_FLAGS = "at_bound:log_sigma_m;sweep_fit_failed:ls:55:at_bound:log_sigma_m"
FLAG_A_SS93 = "some_note:93"         # a flag that is not a failure


def _planted_rows():
    """{(specimen, 'SS'|'LS', sweep): (sigma_f, sigma_m, sigma_s, tilt, flags)}."""
    rows = {}
    for n, s in enumerate(SWEEPS_A):
        rows[(ID_A, "SS", s)] = (0.0301 + 1e-4 * n, 0.0171 + 1e-4 * n, 0.0811 + 1e-3 * n, 0.0123 * (n + 1) - 0.02,
                                 FLAG_A_SS93 if s == 93 else "")
    for k in range(len(LS_AMPS_A)):
        rows[(ID_A, "LS", 54 + k)] = (0.0311 + 1e-4 * k, 0.0172 + 1e-4 * k, 0.0791 + 1e-3 * k, -0.0567 + 0.031 * k, "")
    rows[(ID_B, "SS", 7)] = (0.0347, 0.0223, 0.2901, 0.00777, "")
    for k in range(len(LS_AMPS_B)):
        s = 54 + k
        if s == FAILED_B:
            law = CELL_LAWS[ID_B]
            rows[(ID_B, "LS", s)] = (law[0], law[1], law[3], 0.0444, FAILED_FLAGS)
        else:
            rows[(ID_B, "LS", s)] = (0.0349 + 1e-4 * k, 0.0221 + 1e-4 * k, 0.2777 + 1e-3 * k, 0.0105 * k - 0.013, "")
    rows[(ID_F, "SS", 92)] = (0.0331, 0.0176, 0.0902, 0.003, "")
    for k in range(len(LS_AMPS_F)):
        rows[(ID_F, "LS", 54 + k)] = (0.0332 + 1e-4 * k, 0.0177, 0.0903 + 1e-3 * k, 0.004 * k, "")
    rows[(ID_G, "SS", 92)] = (0.0341, 0.0186, 0.0912, 0.005, "")
    for s in G_STEPS:
        rows[(ID_G, "LS", s)] = (0.0342 + 1e-4 * (s - 54), 0.0187, 0.0913 + 1e-3 * (s - 54), 0.006 * (s - 54), "")
    rows[(ID_C, "SS", 1)] = (0.0322, 0.0166, 0.0861, 0.001, "")
    rows[(ID_C, "LS", 3)] = (0.0323, 0.0167, 0.0862, -0.002, "")
    return rows


PLANTED = _planted_rows()
_STATE = {}


def _fixture():
    """Writes the archive cells and the two tables once; returns the state."""
    if _STATE:
        return _STATE
    root = Path(tempfile.mkdtemp(prefix="smoke_twin_table_"))
    atexit.register(shutil.rmtree, str(root), True)
    rng = np.random.default_rng(SEED)
    truth = {}
    for specimen, sweeps, dropped, amps in ((ID_A, SWEEPS_A, DROPPED_A, LS_AMPS_A),
                                            (ID_B, SWEEPS_B, DROPPED_B, LS_AMPS_B)):
        d = root / ("specimen_%d" % specimen)
        truth[specimen] = snw.write_fixture(d, rng, with_response=False, dropped=dropped, sweeps=sweeps,
                                            ls_amps=amps)
        _set_specimen(d, specimen)
    d_f = root / ("specimen_%d" % ID_F)
    truth[ID_F] = snw.write_fixture(d_f, rng, with_response=False, dropped={}, sweeps=(92,), ls_amps=LS_AMPS_F)
    _set_specimen(d_f, ID_F)
    # a cell with a real step (-90 pA, sweep 55) that load_ls_prewindows cannot return: its stored current
    # trace has no step (all zeros), so no onset can be detected
    d_e = root / "specimen_900000105"
    snw.write_fixture(d_e, rng, with_response=False, dropped={}, sweeps=(92,), ls_amps=(-110.0, -90.0, 50.0))
    _set_specimen(d_e, 900000105)
    with np.load(d_e / "ls_sweeps.npz") as z:
        arrs = {k: z[k] for k in z.files}
    arrs["i_1"] = np.zeros_like(arrs["i_1"])
    np.savez_compressed(d_e / "ls_sweeps.npz", **arrs)
    # TT11: a cell listing a 0 pA sweep, a null amplitude and a 3 pA sweep between real steps
    d_g = root / ("specimen_%d" % ID_G)
    truth[ID_G] = snw.write_fixture(d_g, rng, with_response=False, dropped={}, sweeps=(92,), ls_amps=LS_AMPS_G)
    _set_specimen(d_g, ID_G)
    meta = json.load(open(d_g / "metadata.json"))
    meta["ls_sweeps"][G_NULL_SWEEP - 54]["detected_amplitude_pA"] = None
    with open(d_g / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    cells_csv, sweeps_csv = root / "noise_cells_FIX.csv", root / "noise_sweeps_FIX.csv"
    write_csv_atomic(cells_csv, CELL_COLUMNS, _cell_table_rows())
    write_csv_atomic(sweeps_csv, SWEEP_COLUMNS, _sweep_table_rows())
    # the spike test's planted answers, keyed by the bytes of each stored LS voltage
    by_v = {}
    for specimen in (ID_A, ID_B, ID_F, ID_G):
        d = root / ("specimen_%d" % specimen)
        meta = json.load(open(d / "metadata.json"))
        with np.load(d / "ls_sweeps.npz") as z:
            for info in meta["ls_sweeps"]:
                v = np.asarray(z["v_%d" % info["index"]], dtype=np.float64)
                by_v[v.tobytes()] = (specimen, int(info["sweep_number"]), info["detected_amplitude_pA"],
                                     float(info["sampling_rate_Hz"]))
    _STATE.update(root=root, truth=truth, dir_e=d_e, dir_f=d_f, dir_g=d_g, cells_csv=cells_csv, sweeps_csv=sweeps_csv,
                  by_v=by_v)
    return _STATE


class PlantedSpikeTest:
    """A spike_reason_fn written here: identifies the sweep by its voltage,
    records the call, returns the planted reason ('' when none)."""

    _UNSET = object()

    def __init__(self, result=_UNSET):
        self.calls = []
        self.result = result          # given (even None): returned instead of the planted reason

    def __call__(self, v_mV, sampling_rate_Hz):
        key = _fixture()["by_v"].get(np.asarray(v_mV).tobytes())
        if key is None:
            raise AssertionError("spike_reason_fn called on a voltage that is no stored LS sweep (%d samples)"
                                 % np.asarray(v_mV).size)
        specimen, sweep, _amp, fs = key
        if sampling_rate_Hz != fs or np.asarray(v_mV).dtype != np.float64:
            raise AssertionError("spike_reason_fn: sweep %d called with fs %r (stored %r), dtype %s"
                                 % (sweep, sampling_rate_Hz, fs, np.asarray(v_mV).dtype))
        self.calls.append((specimen, sweep))
        return REASONS.get((specimen, sweep), "") if self.result is PlantedSpikeTest._UNSET else self.result


def _set_specimen(d, specimen):
    meta = json.load(open(d / "metadata.json"))
    meta["specimen_id"] = int(specimen)
    with open(d / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=1)


def _cell_table_rows():
    rows = []
    for specimen, law in CELL_LAWS.items():
        r = {c: None for c in CELL_COLUMNS}
        r.update({"specimen_id": specimen, "status": "ok", "reason": "",
                  "noise_sigma_f_mV": law[0], "noise_sigma_m_mV": law[1], "noise_tau_m_ms": law[2],
                  "noise_sigma_s_mV": law[3], "noise_tau_s_ms": law[4], "noise_flags": "no_plateau",
                  "noise_code_commit": "0" * 40})
        rows.append(r)
    r = {c: None for c in CELL_COLUMNS}
    r.update({"specimen_id": ID_D, "status": "refused", "reason": "ls_tilt_none:54:tail_short"})
    rows.append(r)
    return rows


def _sweep_table_rows():
    rows = []
    for (specimen, pr, s), (sf, sm, ss_, tilt, flags) in PLANTED.items():
        r = {c: None for c in SWEEP_COLUMNS}
        failed = "sweep_fit_failed:" in flags
        r.update({"specimen_id": specimen, "noise_sweep_protocol": pr, "noise_sweep_number": s,
                  "noise_tilt_mV_per_s": tilt, "noise_sweep_sigma_f_mV": sf, "noise_sweep_sigma_m_mV": sm,
                  "noise_sweep_sigma_s_mV": ss_, "noise_sweep_flags": flags,
                  "noise_sweep_sigma_m_raw_mV": None if failed else 0.5 * sm,
                  "noise_sweep_sigma_s_raw_mV": None if failed else 0.5 * ss_})
        rows.append(r)
    return rows


def _tables():
    st = _fixture()
    return tt.read_noise_tables(st["cells_csv"], st["sweeps_csv"])


def _protocols():
    st = _fixture()
    return [tt.load_cell_protocol(st["root"] / ("specimen_%d" % i), spike_reason_fn=PlantedSpikeTest())
            for i in (ID_A, ID_B)]


def _expected_table():
    """The twins' table of A and B written out from the planted values."""
    st = _fixture()
    rows = []
    for specimen, sweeps, amps in ((ID_A, SWEEPS_A, LS_AMPS_A), (ID_B, SWEEPS_B, LS_AMPS_B)):
        claw = CELL_LAWS[specimen]
        truth = st["truth"][specimen]
        common = dict(tau_m_s=claw[2] / 1000.0, tau_s_s=claw[4] / 1000.0,
                      cell_tau_s_above_window=claw[4] / 1000.0 > 0.265)
        for s in sweeps:
            sf, sm, ss_, tilt, flags = PLANTED[(specimen, "SS", s)]
            rows.append(dict(specimen_id=specimen, protocol="SS", sweep_number=s,
                             n_windows=int(np.sum(truth["sweeps"] == s)), amplitude_pA=np.nan, onset_s=np.nan,
                             duration_s=np.nan, n_samples=snw.PRE_N + snw.POST_N, fs_Hz=snw.FS,
                             sigma_f_mV=sf, sigma_m_mV=sm, sigma_s_mV=ss_, tilt_mV_per_s=tilt,
                             law_source="sweep", noise_flags=flags, real_spike_reason="", in_twin=True,
                             **common))
        for k, amp in enumerate(amps):
            s = 54 + k
            sf, sm, ss_, tilt, flags = PLANTED[(specimen, "LS", s)]
            rows.append(dict(specimen_id=specimen, protocol="LS", sweep_number=s, n_windows=1,
                             amplitude_pA=amp, onset_s=snw.LS_PRE_N / snw.FS,
                             duration_s=(snw.LS_PRE_N + snw.LS_STEP_N) / snw.FS - snw.LS_PRE_N / snw.FS,
                             n_samples=snw.LS_PRE_N + snw.LS_STEP_N + snw.LS_TAIL_N, fs_Hz=snw.FS,
                             sigma_f_mV=sf, sigma_m_mV=sm, sigma_s_mV=ss_, tilt_mV_per_s=tilt,
                             law_source="cell" if "sweep_fit_failed:" in flags else "sweep",
                             noise_flags=flags, real_spike_reason=REASONS.get((specimen, s), ""),
                             in_twin=(specimen, s) not in REASONS, **common))
    df = pd.DataFrame(rows)[list(tt.TWIN_SWEEP_COLUMNS)]
    for c in ("specimen_id", "sweep_number", "n_windows", "n_samples"):
        df[c] = df[c].astype(np.int64)
    for c in ("protocol", "law_source", "noise_flags", "real_spike_reason"):
        df[c] = df[c].astype(str)
    for c in ("cell_tau_s_above_window", "in_twin"):
        df[c] = df[c].astype(bool)
    return df


def _expect_refusal(label, func, *must_name):
    """func() must raise TwinTableError whose message names every item."""
    try:
        func()
    except tt.TwinTableError as exc:
        msg = str(exc)
        missing = [str(x) for x in must_name if str(x) not in msg]
        assert not missing, "%s: message %r does not name %s" % (label, msg, missing)
        return msg
    except Exception as exc:  # a different exception is a failure of the contract
        raise AssertionError("%s: raised %s, not TwinTableError: %s" % (label, type(exc).__name__, exc))
    raise AssertionError("%s: nothing raised" % label)


# --------------------------------------------------------------- helpers ---

def make_rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def check_close(name, actual, expected, rtol=1e-12, atol=0.0):
    """numpy.testing.assert_allclose that keeps the evidence on failure."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        path = ARTIFACT_DIR / f"{safe_name}.npz"
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


# ---------------------------------------------------------------- checks ---

def test_tt1_reference():
    cells, sweeps = _tables()
    built = tt.build_twin_sweep_table(_protocols(), cells, sweeps)
    expected = _expected_table()
    assert tuple(built.columns) == tt.TWIN_SWEEP_COLUMNS, list(built.columns)
    # exact: every value is planted here and passes through repr-CSV (exact) and one division by 1000
    pd.testing.assert_frame_equal(built, expected, check_exact=True)
    assert set(built["specimen_id"]) == {ID_A, ID_B}, "cells C, F (tables only) must not enter the table"
    # sorting: specimen, SS before LS, sweep
    keys = list(zip(built["specimen_id"], built["protocol"].map({"SS": 0, "LS": 1}), built["sweep_number"]))
    assert keys == sorted(keys), keys


def test_tt2_units_case():
    cells, sweeps = _tables()
    laws = tt.sweep_laws_from_tables(cells, sweeps)
    assert tuple(laws.columns) == tt.SWEEP_LAW_COLUMNS, list(laws.columns)
    assert len(laws) == len(sweeps) == len(PLANTED)
    cmap = cells.set_index("specimen_id")
    for r in laws.to_dict("records"):
        c = cmap.loc[r["specimen_id"]]
        assert r["tau_m_s"] == c["noise_tau_m_ms"] / 1000.0, ("tau_m", r)
        assert r["tau_s_s"] == c["noise_tau_s_ms"] / 1000.0, ("tau_s", r)
        assert r["protocol"] in ("SS", "LS"), r["protocol"]
    built = tt.build_twin_sweep_table(_protocols(), cells, sweeps)
    for r in built.to_dict("records"):
        law, tilt = tt.sweep_law(built, r["specimen_id"], r["protocol"], r["sweep_number"])
        assert isinstance(law, NoiseLaw)
        got = (law.sigma_f, law.sigma_m, law.tau_m, law.sigma_s, law.tau_s, tilt)
        want = (r["sigma_f_mV"], r["sigma_m_mV"], r["tau_m_s"], r["sigma_s_mV"], r["tau_s_s"], r["tilt_mV_per_s"])
        assert got == want, (got, want)
        sf, sm, ss_, tilt_p, _ = PLANTED[(r["specimen_id"], r["protocol"], r["sweep_number"])]
        assert (law.sigma_f, law.sigma_m, law.sigma_s, tilt) == (sf, sm, ss_, tilt_p), (r, (sf, sm, ss_, tilt_p))
    try:
        tt.sweep_law(built, ID_A, "SS", 999)
    except KeyError:
        pass
    else:
        raise AssertionError("sweep_law: an absent row did not raise KeyError")
    dup = pd.concat([built, built.iloc[[0]]], ignore_index=True)
    r0 = built.iloc[0]
    _expect_refusal("sweep_law duplicate", lambda: tt.sweep_law(dup, int(r0["specimen_id"]), r0["protocol"],
                                                                int(r0["sweep_number"])),
                    int(r0["specimen_id"]), int(r0["sweep_number"]))


def test_tt3_failed_sweep():
    cells, sweeps = _tables()
    built = tt.build_twin_sweep_table(_protocols(), cells, sweeps)
    sel = built[built["law_source"] == "cell"]
    assert list(zip(sel["specimen_id"], sel["protocol"], sel["sweep_number"])) == [(ID_B, "LS", FAILED_B)], sel
    law, _ = tt.sweep_law(built, ID_B, "LS", FAILED_B)
    cl = CELL_LAWS[ID_B]
    want = np.array([cl[0], cl[1], cl[2] / 1000.0, cl[3], cl[4] / 1000.0])
    assert np.array_equal(law.as_array(), want), (law, want)
    assert sel.iloc[0]["noise_flags"] == FAILED_FLAGS
    assert set(built.loc[built["law_source"] != "cell", "law_source"]) == {"sweep"}
    i = sweeps.index[(sweeps["specimen_id"] == ID_B) & (sweeps["noise_sweep_number"] == FAILED_B)
                     & (sweeps["noise_sweep_protocol"] == "LS")][0]
    rtol = TwinConfig().failed_law_rtol
    for col in ("noise_sweep_sigma_f_mV", "noise_sweep_sigma_m_mV", "noise_sweep_sigma_s_mV"):
        bad = sweeps.copy()
        bad.loc[i, col] = bad.loc[i, col] * (1.0 + 10.0 * rtol)       # outside the tolerance: refused
        _expect_refusal("failed row, %s altered" % col, lambda: tt.sweep_laws_from_tables(cells, bad),
                        ID_B, FAILED_B)
        ok = sweeps.copy()
        ok.loc[i, col] = ok.loc[i, col] * (1.0 + 0.1 * rtol)         # inside it: taken as it is
        laws = tt.sweep_laws_from_tables(cells, ok)
        j = laws.index[(laws["specimen_id"] == ID_B) & (laws["sweep_number"] == FAILED_B)
                       & (laws["protocol"] == "LS")][0]
        name = {"noise_sweep_sigma_f_mV": "sigma_f_mV", "noise_sweep_sigma_m_mV": "sigma_m_mV",
                "noise_sweep_sigma_s_mV": "sigma_s_mV"}[col]
        assert laws.loc[j, name] == ok.loc[i, col], "the failed row's SD must be taken as it is, not replaced"


def test_tt4_refusals():
    cells, sweeps = _tables()
    prots = _protocols()
    st = _fixture()
    build = tt.build_twin_sweep_table
    # a cell absent from the per-cell table
    _expect_refusal("cell absent", lambda: build(prots, cells[cells["specimen_id"] != ID_A], sweeps), ID_A)
    # a cell whose status is not ok
    c2 = cells.copy()
    c2.loc[c2["specimen_id"] == ID_B, "status"] = "failed"
    _expect_refusal("cell not ok", lambda: build(prots, c2, sweeps), ID_B)
    _expect_refusal("cell not ok (laws)", lambda: tt.sweep_laws_from_tables(c2, sweeps), ID_B)
    # a refused cell whose protocol is passed
    pd_ = tt.CellProtocol(**{**prots[0].__dict__, "specimen_id": ID_D})
    _expect_refusal("refused cell", lambda: build([pd_], cells, sweeps), ID_D)
    # a stored SS sweep without a row; an LS one without a row
    for pr, s in (("SS", 93), ("LS", 55)):
        m = ((sweeps["specimen_id"] == ID_A) & (sweeps["noise_sweep_protocol"] == pr)
             & (sweeps["noise_sweep_number"] == s))
        _expect_refusal("%s %d without a row" % (pr, s), lambda: build(prots, cells, sweeps[~m]), ID_A, s)
        # ... and with two rows
        _expect_refusal("%s %d with two rows" % (pr, s),
                        lambda: build(prots, cells, pd.concat([sweeps, sweeps[m]], ignore_index=True)), ID_A, s)
    # a stored SS sweep whose only row has the other protocol
    m = (sweeps["specimen_id"] == ID_B) & (sweeps["noise_sweep_protocol"] == "SS")
    s3 = sweeps.copy()
    s3.loc[m, "noise_sweep_protocol"] = "LS"
    _expect_refusal("SS row filed as LS", lambda: build(prots, cells, s3), ID_B, 7)
    # a row without a stored sweep
    extra = sweeps[(sweeps["specimen_id"] == ID_A)].iloc[[0]].copy()
    extra["noise_sweep_protocol"], extra["noise_sweep_number"] = "LS", 99
    _expect_refusal("row without a stored sweep",
                    lambda: build(prots, cells, pd.concat([sweeps, extra], ignore_index=True)), ID_A, 99)
    # a step of metadata.json['ls_sweeps'] that load_ls_prewindows does not return (sweep 55, -90 pA, whose
    # stored current has no step; amendment 3: a step, not a 0 pA sweep)
    _expect_refusal("LS sweep not loaded",
                    lambda: tt.load_cell_protocol(st["dir_e"], spike_reason_fn=lambda v, fs: ""), 900000105, 55)
    # a law that NoiseLaw refuses: a negative, a missing SD; a cell time scale missing or zero
    j = sweeps.index[(sweeps["specimen_id"] == ID_A) & (sweeps["noise_sweep_number"] == 92)][0]
    for col, val in (("noise_sweep_sigma_s_mV", -0.01), ("noise_sweep_sigma_f_mV", np.nan),
                     ("noise_sweep_sigma_m_mV", np.inf)):
        bad = sweeps.copy()
        bad.loc[j, col] = val
        _expect_refusal("law %s=%r" % (col, val), lambda: build(prots, cells, bad), ID_A, 92)
    for col, val in (("noise_tau_s_ms", np.nan), ("noise_tau_m_ms", 0.0)):
        bad = cells.copy()
        bad.loc[bad["specimen_id"] == ID_A, col] = val
        _expect_refusal("cell %s=%r" % (col, val), lambda: build(prots, bad, sweeps), ID_A)
    # a tilt that is not finite
    for val in (np.nan, np.inf):
        bad = sweeps.copy()
        bad.loc[j, "noise_tilt_mV_per_s"] = val
        _expect_refusal("tilt %r" % val, lambda: build(prots, cells, bad), ID_A, 92)
    # an empty tilt field in the CSV reads as missing, and is refused
    txt = Path(st["sweeps_csv"]).read_text().splitlines()
    hdr = txt[0].split(",")
    k_tilt = hdr.index("noise_tilt_mV_per_s")
    out = [txt[0]]
    for line in txt[1:]:
        f = line.split(",")
        if f[0] == str(ID_A) and f[1] == "LS" and f[2] == "54":
            f[k_tilt] = ""
        out.append(",".join(f))
    p_bad = Path(st["root"]) / "sweeps_empty_tilt.csv"
    p_bad.write_text("\n".join(out) + "\n")
    c_ok, s_bad = tt.read_noise_tables(st["cells_csv"], p_bad)
    _expect_refusal("empty tilt field", lambda: build(prots, c_ok, s_bad), ID_A, 54)
    # a failed row whose SD differs from the cell's
    i = sweeps.index[(sweeps["specimen_id"] == ID_B) & (sweeps["noise_sweep_number"] == FAILED_B)
                     & (sweeps["noise_sweep_protocol"] == "LS")][0]
    bad = sweeps.copy()
    bad.loc[i, "noise_sweep_sigma_s_mV"] = 0.5
    _expect_refusal("failed row altered", lambda: build(prots, cells, bad), ID_B, FAILED_B)


def test_tt5_round_trip():
    cells, sweeps = _tables()
    built = tt.build_twin_sweep_table(_protocols(), cells, sweeps)
    path = Path(_fixture()["root"]) / "twin_sweeps_FIX.csv"
    tt.write_twin_sweep_table(built, path)
    back = tt.read_twin_sweep_table(path)
    assert tuple(back.columns) == tt.TWIN_SWEEP_COLUMNS, list(back.columns)
    assert list(back.dtypes) == list(built.dtypes), (list(back.dtypes), list(built.dtypes))
    pd.testing.assert_frame_equal(back, built, check_exact=True)
    raw = path.read_bytes()
    assert b"\r" not in raw and all(b < 128 for b in raw), "the table file is not ASCII / LF"
    first_ss = raw.decode("ascii").splitlines()[1].split(",")
    assert first_ss[1] == "SS" and first_ss[4:7] == ["", "", ""], first_ss   # SS: NaN as empty fields
    assert first_ss[-2:] == ["", "True"], first_ss                         # SS: no reason, in the twin
    assert (~back["in_twin"]).sum() == 2 and set(back.loc[~back["in_twin"], "real_spike_reason"]) == \
        {"peak_above_threshold", "action_potential"}, back[["real_spike_reason", "in_twin"]]
    # a value that repr and %.17g both need all 17 digits for
    probe = built.copy()
    probe.loc[0, "tilt_mV_per_s"] = 0.1 + 0.2
    tt.write_twin_sweep_table(probe, path)
    assert tt.read_twin_sweep_table(path).loc[0, "tilt_mV_per_s"] == 0.1 + 0.2
    # a table with its columns out of order is refused
    swapped = built[list(tt.TWIN_SWEEP_COLUMNS[::-1])]
    _expect_refusal("columns out of order", lambda: tt.write_twin_sweep_table(swapped, path))


def test_tt6_slow_flag():
    cells, sweeps = _tables()
    built = tt.build_twin_sweep_table(_protocols(), cells, sweeps)
    for specimen, want in ((ID_A, False), (ID_B, True)):
        vals = set(built.loc[built["specimen_id"] == specimen, "cell_tau_s_above_window"])
        assert vals == {want}, (specimen, vals)
    tau = built.groupby("specimen_id")["tau_s_s"].first()
    assert tau[ID_A] == 0.2650 and tau[ID_B] == 0.2651, tau
    assert built["cell_tau_s_above_window"].dtype == bool
    # the threshold is the config's
    loose = tt.build_twin_sweep_table(_protocols(), cells, sweeps, TwinConfig(slow_flag_tau_s=0.3))
    assert not loose["cell_tau_s_above_window"].any()


def test_tt7_protocol():
    st = _fixture()
    for specimen, sweeps, amps in ((ID_A, SWEEPS_A, LS_AMPS_A), (ID_B, SWEEPS_B, LS_AMPS_B)):
        p = tt.load_cell_protocol(st["root"] / ("specimen_%d" % specimen), spike_reason_fn=PlantedSpikeTest())
        truth = st["truth"][specimen]
        K, K_LS = truth["K"], len(amps)
        assert p.specimen_id == specimen
        assert p.ss_sweep_number.shape == (K,) and p.ls_sweep_number.shape == (K_LS,)
        assert_array_equal(p.ss_position, truth["positions"])
        assert_array_equal(p.ss_sweep_number, truth["sweeps"])
        assert_array_equal(p.ss_polarity_is_dep, truth["pol"])
        assert_array_equal(p.ss_peak_pA, np.where(truth["pol"], 200.0, -200.0))
        assert_array_equal(p.ss_stim_duration_s, np.full(K, snw.PULSE_N / snw.FS))
        assert_array_equal(p.ss_t_s, truth["t_win"])
        assert (p.ss_fs_Hz, p.ss_pre_n, p.ss_period_samples, p.ss_train_length) == \
            (snw.FS, snw.PRE_N, snw.PERIOD, snw.TRAIN)
        assert_array_equal(p.ls_index, np.arange(K_LS))
        assert_array_equal(p.ls_sweep_number, 54 + np.arange(K_LS))
        assert_array_equal(p.ls_amplitude_pA, np.asarray(amps))
        assert_array_equal(p.ls_onset_s, np.full(K_LS, snw.LS_PRE_N / snw.FS))
        check_close("ls_duration_%d" % specimen, p.ls_duration_s, np.full(K_LS, snw.LS_STEP_N / snw.FS),
                    rtol=0.0, atol=1e-12)    # a difference of two sample times in s: float64 roundoff only
        assert_array_equal(p.ls_n_samples, np.full(K_LS, truth["ls_n_samples"]))
        assert p.ls_fs_Hz == snw.FS
        dtypes = {"ss_sweep_number": np.int64, "ss_position": np.int64, "ss_polarity_is_dep": np.bool_,
                  "ss_peak_pA": np.float64, "ss_stim_duration_s": np.float64, "ss_t_s": np.float64,
                  "ls_index": np.int64, "ls_sweep_number": np.int64, "ls_amplitude_pA": np.float64,
                  "ls_onset_s": np.float64, "ls_duration_s": np.float64, "ls_n_samples": np.int64}
        for name, dt in dtypes.items():
            assert getattr(p, name).dtype == dt, (name, getattr(p, name).dtype)
        assert p.ls_spike_reason.dtype == object and p.ls_spike_reason.shape == (K_LS,)
        want = [REASONS.get((specimen, 54 + k), "") for k in range(K_LS)]
        assert list(p.ls_spike_reason) == want and all(type(x) is str for x in p.ls_spike_reason), \
            (list(p.ls_spike_reason), want)


def test_tt8_real_run():
    if not ARGS.nf2_dir:
        raise unittest.SkipTest("no --nf2-dir given (the nf2 tables and cells/specimen_<id>.json)")
    nf2 = Path(ARGS.nf2_dir)
    cells, sweeps = tt.read_noise_tables(nf2 / "noise_cells_L3_exc.csv", nf2 / "noise_sweeps_L3_exc.csv")
    laws = tt.sweep_laws_from_tables(cells, sweeps)
    counts = laws["protocol"].value_counts().to_dict()
    n_cell = int((laws["law_source"] == "cell").sum())
    assert (len(laws), counts.get("SS"), counts.get("LS"), n_cell) == (589, 64, 525, 9), \
        (len(laws), counts, n_cell)
    worst = 0.0
    for specimen, grp in laws.groupby("specimen_id"):
        rec = json.load(open(nf2 / "cells" / ("specimen_%d.json" % specimen)))
        stage = rec["sweep_stage"]
        tilts = {(r["protocol"], int(r["sweep"])): r["tilt_mV_per_s"] for r in stage["rows"]}
        assert len(tilts) == len(grp), (specimen, len(tilts), len(grp))
        for r in grp.to_dict("records"):
            key = "%s:%d" % (r["protocol"].lower(), r["sweep_number"])
            j = stage["twin_laws"][key]
            got = np.array([r["sigma_f_mV"], r["sigma_m_mV"], r["tau_m_s"], r["sigma_s_mV"], r["tau_s_s"]])
            want = np.array([j["sigma_f_mV"], j["sigma_m_mV"], j["tau_m_s"], j["sigma_s_mV"], j["tau_s_s"]])
            # 1e-12 relative: float64 through a CSV of 17 digits and a division by 1000 (SPEC, TT8)
            check_close("tt8_law_%d_%s" % (specimen, key), got, want, rtol=1e-12)
            worst = max(worst, float(np.max(np.abs(got - want) / np.abs(want))))
            assert r["tilt_mV_per_s"] == tilts[(r["protocol"].lower(), r["sweep_number"])], (specimen, key)
    flagged = sorted(int(x) for x in laws.loc[laws["cell_tau_s_above_window"], "specimen_id"].unique())
    assert flagged == sorted([571430815, 526785799, 571624532, 548408551]), flagged
    print("   TT8: 589 rows, worst relative law difference %.3g" % worst)
    if not ARGS.archive_cell:
        raise unittest.SkipTest("first half PASSED; second half skipped: no --archive-cell given")
    # the layout half needs no spike test: a stub that passes every sweep (the third part uses the loader's)
    p = tt.load_cell_protocol(ARGS.archive_cell, spike_reason_fn=lambda v, fs: "")
    assert p.specimen_id == 508282493, p.specimen_id
    table = tt.build_twin_sweep_table([p], cells, sweeps)
    ss = table[table["protocol"] == "SS"]
    ls = table[table["protocol"] == "LS"]
    assert list(ss["sweep_number"]) == [92, 93] and list(ss["n_windows"]) == [23, 17], ss
    assert set(ss["n_samples"]) == {10500}, ss["n_samples"]
    assert len(ls) == 22, len(ls)
    check_close("tt8_onset", ls["onset_s"], np.full(22, 0.27), rtol=0.0, atol=1e-12)
    check_close("tt8_duration", ls["duration_s"], np.full(22, 1.0), rtol=0.0, atol=1e-12)
    assert set(ls["n_samples"]) == {363500}, set(ls["n_samples"])
    # the writer's detected amplitudes lie within 1e-5 pA of the nominal steps (-110.0000021...)
    check_close("tt8_amplitudes", np.sort(ls["amplitude_pA"].to_numpy()), np.arange(-110.0, 311.0, 20.0),
                rtol=0.0, atol=1e-3)
    meta = json.load(open(Path(ARGS.archive_cell) / "metadata.json"))
    amp_meta = {int(i["sweep_number"]): float(i["detected_amplitude_pA"]) for i in meta["ls_sweeps"]}
    for r in ls.to_dict("records"):
        assert r["amplitude_pA"] == amp_meta[r["sweep_number"]], r
    print("   TT8: 508282493 -> %d SS rows (%s windows), %d LS rows"
          % (len(ss), list(ss["n_windows"]), len(ls)))
    # third part (amendment 1, D-190): the loader's own screen as the reference
    try:
        import passive_fitting_hpc_fixed as pf
    except Exception as exc:  # NEURON or another dependency missing: reported, not hidden
        raise unittest.SkipTest("first two parts PASSED; third part skipped: import passive_fitting_hpc_fixed "
                                "failed (%s: %s)" % (type(exc).__name__, exc))
    p = tt.load_cell_protocol(ARGS.archive_cell, spike_reason_fn=pf.sweep_spike_reason)
    table = tt.build_twin_sweep_table([p], cells, sweeps)
    cd = pf.load_cell_from_archive(ARGS.archive_cell, load_depolarising_ls=True, ls_dep_max_amplitude_pA=None,
                                   verbose=False)
    screen = cd.ls_dep_screen
    dep = table[(table["protocol"] == "LS") & (table["amplitude_pA"] > TwinConfig().ls_dep_min_amplitude_pA)]
    assert len(dep) == screen["n_depolarising"], (len(dep), screen)
    assert int(dep["in_twin"].sum()) == screen["n_kept"], (int(dep["in_twin"].sum()), screen)
    for reason in pf.SPIKE_REASONS:
        n = int((dep["real_spike_reason"] == reason).sum())
        assert n == screen["n_" + reason], (reason, n, screen)
    hyp = table[(table["protocol"] == "LS") & ~(table["amplitude_pA"] > TwinConfig().ls_dep_min_amplitude_pA)]
    assert hyp["in_twin"].all() and (hyp["real_spike_reason"] == "").all(), hyp
    assert table.loc[table["protocol"] == "SS", "in_twin"].all()
    print("   TT8: 508282493 -> %d depolarising LS rows, %d in the twin; loader's screen %s"
          % (len(dep), int(dep["in_twin"].sum()),
             {k: screen[k] for k in ("n_depolarising", "n_kept") + tuple("n_" + r for r in pf.SPIKE_REASONS)}))


def test_tt10_spike_screen():
    st = _fixture()
    cells, sweeps = _tables()
    thr = TwinConfig().ls_dep_min_amplitude_pA
    assert thr == 5.0, thr
    fn = PlantedSpikeTest()
    p = tt.load_cell_protocol(st["dir_f"], spike_reason_fn=fn)
    above = [(ID_F, 54 + k) for k, a in enumerate(LS_AMPS_F) if a > thr]
    assert sorted(fn.calls) == above and len(fn.calls) == len(set(fn.calls)), (fn.calls, above)
    assert (ID_F, 54 + LS_AMPS_F.index(5.0)) not in fn.calls, "the sweep at exactly the threshold was tested"
    want = [REASONS.get((ID_F, 54 + k), "") for k in range(len(LS_AMPS_F))]
    assert list(p.ls_spike_reason) == want, (list(p.ls_spike_reason), want)
    table = tt.build_twin_sweep_table([p], cells, sweeps)
    ls = table[table["protocol"] == "LS"].set_index("sweep_number")
    for k, a in enumerate(LS_AMPS_F):
        s = 54 + k
        r = REASONS.get((ID_F, s), "")
        assert ls.loc[s, "real_spike_reason"] == r and ls.loc[s, "in_twin"] == (r == ""), (s, ls.loc[s].to_dict())
    assert len(table) == 1 + len(LS_AMPS_F), "one row per real sweep, in the twin or not"
    ssr = table[table["protocol"] == "SS"]
    assert list(ssr["real_spike_reason"]) == [""] and list(ssr["in_twin"]) == [True], ssr
    assert table["in_twin"].dtype == bool and table["real_spike_reason"].map(type).eq(str).all()
    # the threshold is the config's: at 60 pA only the 130 pA sweep is tested
    fn60 = PlantedSpikeTest()
    tt.load_cell_protocol(st["dir_f"], spike_reason_fn=fn60, twin_cfg=TwinConfig(ls_dep_min_amplitude_pA=60.0))
    assert fn60.calls == [(ID_F, 54 + LS_AMPS_F.index(130.0))], fn60.calls
    # a return value that is not a str is refused, naming the cell and the sweep
    for bad in (None, 0, False, b"", np.float64(0.0)):
        _expect_refusal("spike_reason_fn -> %r" % (bad,),
                        lambda: tt.load_cell_protocol(st["dir_f"], spike_reason_fn=PlantedSpikeTest(bad)),
                        ID_F, above[0][1])   # the first sweep tested: the 20 pA one
    # spike_reason_fn is required
    try:
        tt.load_cell_protocol(st["dir_f"])
    except TypeError:
        pass
    else:
        raise AssertionError("load_cell_protocol ran without spike_reason_fn")


def test_tt11_not_a_step():
    st = _fixture()
    cells, sweeps = _tables()
    fn = PlantedSpikeTest()
    p = tt.load_cell_protocol(st["dir_g"], spike_reason_fn=fn)
    steps = list(G_STEPS)
    assert list(p.ls_sweep_number) == steps, list(p.ls_sweep_number)
    # the index in ls_sweeps.npz of the remaining sweeps (not their position among the steps: 0, 1, 2)
    assert list(p.ls_index) == [s - 54 for s in steps], list(p.ls_index)
    assert list(p.ls_amplitude_pA) == [LS_AMPS_G[s - 54] for s in steps], list(p.ls_amplitude_pA)
    K_LS = len(steps)
    for name in ("ls_onset_s", "ls_duration_s", "ls_n_samples", "ls_spike_reason"):
        assert getattr(p, name).shape == (K_LS,), (name, getattr(p, name).shape)
    assert_array_equal(p.ls_onset_s, np.full(K_LS, snw.LS_PRE_N / snw.FS))
    assert_array_equal(p.ls_n_samples, np.full(K_LS, st["truth"][ID_G]["ls_n_samples"]))
    assert fn.calls == [(ID_G, 59)], fn.calls          # the spike test: only the step above 5 pA (90 pA)
    # the stored voltage the index points at is the sweep's own (each fixture sweep has its own noise)
    meta = json.load(open(st["dir_g"] / "metadata.json"))
    with np.load(st["dir_g"] / "ls_sweeps.npz") as z:
        for k, s in zip(p.ls_index, p.ls_sweep_number):
            info = [i for i in meta["ls_sweeps"] if int(i["sweep_number"]) == int(s)][0]
            assert int(info["index"]) == int(k), (s, k, info)
            assert st["by_v"][np.asarray(z["v_%d" % k], dtype=np.float64).tobytes()][1] == int(s), (s, k)
    table = tt.build_twin_sweep_table([p], cells, sweeps)
    ls = table[table["protocol"] == "LS"]
    assert list(ls["sweep_number"]) == steps and len(table) == 1 + K_LS, table[["protocol", "sweep_number"]]
    assert list(ls["amplitude_pA"]) == [LS_AMPS_G[s - 54] for s in steps]
    # a row of the per-sweep table for a sweep that is not a step is still refused (no stored step)
    extra = sweeps[sweeps["specimen_id"] == ID_G].iloc[[1]].copy()
    extra["noise_sweep_protocol"], extra["noise_sweep_number"] = "LS", 55
    _expect_refusal("row of a 0 pA sweep", lambda: tt.build_twin_sweep_table(
        [p], cells, pd.concat([sweeps, extra], ignore_index=True)), ID_G, 55)


def test_tt9_ascii_lf():
    for name in ("twin_table.py", "smoke_twin_table.py"):
        raw = (HERE / name).read_bytes()
        bad = [i for i, b in enumerate(raw) if b >= 128]
        assert not bad, "%s: non-ASCII byte at offset %d" % (name, bad[0])
        assert b"\r" not in raw, "%s: CR line ending" % name


def test_convergence():
    raise unittest.SkipTest("not a discretisation: Block 5 is a join and checks")


def test_determinism():
    raise unittest.SkipTest("no randomness in Block 5 (no generator is used)")


# ---------------------------------------------------------------- runner ---

def _environment():
    parts = [f"python {platform.python_version()}"]
    for package in REPORT_PACKAGES:
        try:
            parts.append(f"{package} {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            parts.append(f"{package} (not installed)")
    parts.append(platform.platform())
    parts.append(f"seed {SEED}")
    return " | ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nf2-dir", default=None, help="the nf2 run's tables and cells/ (TT8)")
    ap.add_argument("--archive-cell", default=None, help="the archive cell specimen_508282493/ (TT8)")
    args = ap.parse_args(argv)
    ARGS.nf2_dir, ARGS.archive_cell = args.nf2_dir, args.archive_cell
    checks = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    print(f"== {Path(__file__).name}")
    print(_environment())

    results = []
    for name, func in checks:
        start = time.perf_counter()
        try:
            func()
            status, detail = "PASS", ""
        except unittest.SkipTest as exc:
            status, detail = "SKIP", str(exc)
        except NotImplementedError as exc:
            status, detail = "TODO", str(exc)
        except AssertionError as exc:
            status, detail = "FAIL", str(exc)
        except Exception:  # unexpected errors are reported, never hidden
            status, detail = "ERROR", traceback.format_exc()
        results.append((name, status, time.perf_counter() - start, detail))

    width = max([len(name) for name, _, _, _ in results] + [4])
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(f"{status:<5}  {name:<{width}}  {seconds:8.3f}s  {headline}".rstrip())

    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print(f"\n---- {status}: {name}\n{detail.strip()}")

    counts = {s: sum(1 for r in results if r[1] == s)
              for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join(f"{n} {s.lower()}" for s, n in counts.items()))
    for path in _SAVED_ARTIFACTS:
        print(f"-- arrays from a failed comparison: {path}")
    failed = counts["FAIL"] + counts["ERROR"] + counts["TODO"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
