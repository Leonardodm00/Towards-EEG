"""Spec-derived tests of Block 8 (run_twin_recovery.py; the Block 8 entry with
amendment 7, SPEC 2.7 (i)). Integration tester, 2026-11-20.

Fixture: a group of 4 writer-like real cells (smoke_noise_windows.write_fixture,
ids rewritten in metadata.json) with ball-and-stick morphologies, hand-written
noise tables; the fifth cell is 'refused' in the per-cell table and must not
enter the cohort. Expected values: the cohort rule, the cell draw of 2.7 (i)
(SeedSequence written out), Block 5's table on the same inputs (the producer
the plan consumes), the loader's v_rest (the reference the spec names), the
exit codes of the entry.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("neuron")
pytest.importorskip("skopt")

import run_twin_recovery as rtr     # noqa: E402
import twin_table as tt             # noqa: E402

IDS = (500000011, 500000012, 500000013, 500000014)
REFUSED_ID = 500000015
LS_AMPS = (-110.0, -70.0, 0.0)
SMALL = ["--n-per-configuration", "1", "--n-calls", "1", "--n-initial", "1"]


@pytest.fixture(scope="module")
def group(tmp_path_factory):
    import smoke_noise_windows as sw
    import synthetic_ground_truth as sgt
    root = tmp_path_factory.mktemp("b8")
    gdir = root / "G"
    rng = np.random.default_rng(5)
    for sid in IDS + (REFUSED_ID,):
        d = gdir / ("specimen_%d" % sid)
        sw.write_fixture(d, rng, sweeps=(92,), dropped={92: (1, 4)}, ls_amps=LS_AMPS)
        meta = json.loads((d / "metadata.json").read_text())
        meta["specimen_id"] = sid
        meta["v_rest_mV"] = -60.0 - (sid % 10)
        (d / "metadata.json").write_text(json.dumps(meta))
        sgt.write_ball_and_stick_swc(d / "reconstruction.swc")
    ndir = root / "noise"
    ndir.mkdir()
    with open(ndir / "noise_cells_G.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["specimen_id", "status", "reason", "noise_sigma_f_mV", "noise_sigma_m_mV", "noise_tau_m_ms",
                    "noise_sigma_s_mV", "noise_tau_s_ms"])
        for k, sid in enumerate(IDS):
            w.writerow([sid, "ok", "", 0.034, 0.017, 0.07, 0.08, 300.0 if k == 1 else 200.0])
        w.writerow([REFUSED_ID, "refused", "ls_tilt_none", "", "", "", "", ""])
    with open(ndir / "noise_sweeps_G.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["specimen_id", "noise_sweep_protocol", "noise_sweep_number", "noise_tilt_mV_per_s",
                    "noise_sweep_sigma_f_mV", "noise_sweep_sigma_m_mV", "noise_sweep_sigma_s_mV", "noise_sweep_flags"])
        for sid in IDS:
            w.writerow([sid, "SS", 92, 0.01, 0.033, 0.018, 0.09, ""])
            for s in (54, 55):
                w.writerow([sid, "LS", s, -0.02, 0.031, 0.016, 0.07, ""])
    return dict(root=root, gdir=gdir, ndir=ndir)


def _plan(group, out, *extra):
    return rtr.main(["plan", "--group-dir", str(group["gdir"]), "--noise-dir", str(group["ndir"]), "--out", str(out)]
                    + SMALL + list(extra))


def test_plan_files_cohort_and_draw(group, tmp_path):
    import passive_fitting_hpc_fixed as mono
    out = tmp_path / "run"
    assert _plan(group, out, "--seed", "4") == 0
    for f in ("real_cells_G.csv", "twin_sweeps_G.csv", "manifest.csv", "run_config.json"):
        assert (out / f).is_file(), f
    rc = pd.read_csv(out / "real_cells_G.csv")
    assert list(rc["specimen_id"]) == list(IDS)                      # the refused cell is out of the cohort
    for _, r in rc.iterrows():
        d = group["gdir"] / ("specimen_%d" % r["specimen_id"])
        assert r["v_rest_mV"] == mono.load_cell_from_archive(d, verbose=False).v_rest_mV
        assert r["swc_path"] == str(d / "reconstruction.swc")
    assert list(rc["cell_tau_s_above_window"]) == [False, True, False, False]
    man = pd.read_csv(out / "manifest.csv")
    for q in range(3):
        cells = np.random.default_rng(np.random.SeedSequence([4, q])).choice(np.array(IDS), size=1, replace=False)
        sub = man.iloc[q:q + 1]
        assert_array_equal(sub["twin_of_specimen_id"], cells)
        assert int(sub["specimen_id"].iloc[0]) == 900000000 + 100 * q
    # the twins' table = Block 5 on the same inputs
    cells_df, sweeps_df = tt.read_noise_tables(group["ndir"] / "noise_cells_G.csv",
                                               group["ndir"] / "noise_sweeps_G.csv")
    prots = [tt.load_cell_protocol(group["gdir"] / ("specimen_%d" % s), spike_reason_fn=mono.sweep_spike_reason)
             for s in IDS]
    ref = tt.build_twin_sweep_table(prots, cells_df, sweeps_df)
    pd.testing.assert_frame_equal(tt.read_twin_sweep_table(out / "twin_sweeps_G.csv"), ref)
    # idempotent; another seed refused (exit 3) unless --force
    mt = {f: os.path.getmtime(out / f) for f in os.listdir(out)}
    assert _plan(group, out, "--seed", "4") == 0
    assert mt == {f: os.path.getmtime(out / f) for f in os.listdir(out)}
    assert _plan(group, out, "--seed", "5") == 3
    assert _plan(group, out, "--seed", "5", "--force") == 0
    # list: the manifest's ids in order
    man2 = pd.read_csv(out / "manifest.csv")
    assert list(man2["specimen_id"]) == [900000000, 900000100, 900000200]


def test_refusals_and_usage_codes(group, tmp_path):
    out = tmp_path / "r"
    assert _plan(group, out, "--cells", "%d,%d" % (IDS[0], REFUSED_ID)) == 3     # Block 5 refuses the cell
    assert rtr.main(["list", "--out", str(tmp_path / "none")]) == 2
    assert rtr.main(["merge", "--out", str(tmp_path / "none")]) == 2
    assert _plan(group, out, "--n-per-configuration", "5") == 3                   # cohort of 4 < 5
    assert _plan(group, out) == 0
    assert rtr.main(["twin", "--out", str(out), "--twin", "123"]) == 2
    assert rtr.main(["merge", "--out", str(out), "--no-plot"]) == 1                # no results at all


def test_one_twin_end_to_end_and_merge(group, tmp_path, capsys):
    """One twin (the passive_fit one: arm passive_fullstep) built and fitted at
    the smallest budget; its status ok, its results row present; the merge
    reports it and lists the other two twins as missing, exit 0."""
    out = tmp_path / "e2e"
    assert _plan(group, out, "--seed", "4") == 0
    tid = 900000100
    assert rtr.main(["twin", "--out", str(out), "--twin", str(tid)]) == 0
    st = json.loads((out / "status" / ("twin_%d.json" % tid)).read_text())
    assert st["status"] == "ok" and st["configuration"] == "passive_fit" and st["arm"] == "passive_fullstep"
    res = pd.read_csv(out / "fits" / ("twin_%d" % tid) / "phase2_results.csv")
    assert (res["specimen_id"] == tid).any()
    meta = json.loads((out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid) / "metadata.json").read_text())
    man = pd.read_csv(out / "manifest.csv")
    row = man[man["specimen_id"] == tid].iloc[0]
    assert meta["twin"]["twin_of_specimen_id"] == int(row["twin_of_specimen_id"])
    assert meta["v_rest_mV"] == float(row["v_rest_mV"])
    # resume: a second call neither rebuilds nor refits
    m0 = os.path.getmtime(out / "fits" / ("twin_%d" % tid) / "phase2_results.csv")
    m1 = os.path.getmtime(out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid) / "metadata.json")
    assert rtr.main(["twin", "--out", str(out), "--twin", str(tid)]) == 0
    assert m0 == os.path.getmtime(out / "fits" / ("twin_%d" % tid) / "phase2_results.csv")
    assert m1 == os.path.getmtime(out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid) / "metadata.json")
    assert rtr.main(["merge", "--out", str(out), "--no-plot"]) == 0
    ts = pd.read_csv(out / "twins_status.csv")
    assert list(ts["status"]) == ["missing", "ok", "missing"]
    byc = pd.read_csv(out / "recovery_by_configuration.csv")
    assert set(byc["specimen_id"]) == {tid} and set(byc["configuration"]) == {"passive_fit"}
    assert (out / "report").is_dir() and (out / "report_without_flagged").is_dir()


def test_same_seed_same_run(group, tmp_path):
    """Step 6: two runs with the same options in two run directories give the
    same manifest and twins' table (bytes) and the same twin archive (arrays)."""
    tid = 900000200        # the active_fit twin
    arrays = []
    for name in ("a", "b"):
        out = tmp_path / name
        assert _plan(group, out, "--seed", "4") == 0
        assert rtr.main(["twin", "--out", str(out), "--twin", str(tid)]) == 0
        d = out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid)
        with np.load(d / "ss_pulses.npz") as z, np.load(d / "ls_sweeps.npz") as y:
            arrays.append(({k: z[k] for k in z.files}, {k: y[k] for k in y.files}))
    for f in ("manifest.csv", "twin_sweeps_G.csv", "real_cells_G.csv"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()
    for part in (0, 1):
        assert arrays[0][part].keys() == arrays[1][part].keys()
        for k in arrays[0][part]:
            assert_array_equal(arrays[0][part][k], arrays[1][part][k])


def test_merge_with_only_the_fp_control_twin(group, tmp_path):
    """Block 8, merge: 'returns 0 when the reports are written, whatever the
    gate's verdict, and 1 when no twin has results'. With results for the
    active_fit twin alone (the ih6 arm's false-positive control: a pilot of the
    control, or a run whose full twins all failed) a report is due and the
    merge must return 0 (or 1), never raise. FAILS at 577445d (finding F1 of
    specs/test_reports/2026-11-20_1432_incremental.md): KeyError('arm') from
    ih_recovery_report.recovery_summary on a frame of false-positive rows only."""
    out = tmp_path / "fp"
    assert _plan(group, out, "--seed", "4") == 0
    assert rtr.main(["twin", "--out", str(out), "--twin", "900000200"]) == 0
    assert rtr.main(["merge", "--out", str(out), "--no-plot"]) == 0
    assert (out / "report").is_dir()
