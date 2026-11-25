"""Spec-derived tests of Block 8 (run_twin_recovery.py; the Block 8 entry with
amendment 7, SPEC 2.7 (i)). Integration tester, 2026-11-20.

Fixture: a group of 4 writer-like real cells (smoke_noise_windows.write_fixture,
ids rewritten in metadata.json) with ball-and-stick morphologies, hand-written
noise tables; the fifth cell is 'refused' in the per-cell table and must not
enter the cohort. Expected values: the cohort rule, the cell draw of 2.7 (i)
(SeedSequence written out), Block 5's table on the same inputs (the producer
the plan consumes), the loader's v_rest (the reference the spec names), the
exit codes of the entry.

Renewed 2026-11-22 (third incremental run) to the current SPEC (Block 17): plan
now draws a fourth configuration (full_no_noise, --n-no-noise 5 by default,
refused above --n-per-configuration), so the small runs below pass
--n-no-noise 0 to keep their three twins; the gate arm is ih5 with the required
axes Cm, gbar, kappa_tau (2.8 (g)), so the amendment-8 tests write ih5 results
(five names) and expect no dv_h row; the kappa_tau box is run_config.json's
tau_h_pk record. Block 17's own tests are at the end of this file.
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
SMALL = ["--n-per-configuration", "1", "--n-no-noise", "0", "--n-calls", "1", "--n-initial", "1"]


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
    assert (out / "report_no_noise" / "NO_RESULTS.txt").is_file()      # no full_no_noise twin in this run


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


# ===========================================================================
#  Amendment 8 (integration tester, 2026-11-20, second incremental run).
#  Oracles: the Block 8 entry's amendment 8 (8.1 the controls-only merge, 8.2 the
#  gate line in the submitter's header, 8.3 the base seed) and SPEC 2.7 (c), (e)
#  for the noise a twin carries. Expected values are computed here: the gated
#  axes from RecoveryTolerances.for_axis, the summary's columns from the spec's
#  list, the noise from SeedSequence([base, g, protocol, s(, pos)]) +
#  noise_law.draw_units (the draw the spec names) + an AR(1) recursion written
#  out here (L2').
# ===========================================================================
import dataclasses                        # noqa: E402

from twin_config import TwinConfig        # noqa: E402

SUMMARY_COLUMNS_SPEC = ["arm", "parameter", "n", "n_railed", "median_abs_err", "q25_abs_err", "q75_abs_err",
                        "median_err", "err_kind"]
TRUTH_COL = {"Cm": "cm_true", "Rm": "rm_true", "Ra": "ra_true", "gbar": "ih_gihbar_S_cm2", "dv_h": "ih_dvh_mV",
             "kappa_tau": "ih_kappa_tau"}
IH6 = ("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau")
IH5 = ("Cm", "Rm", "Ra", "gbar", "kappa_tau")
PASSIVE3 = ("Cm", "Rm", "Ra")
FLAGGED = {IDS[1]}                          # tau_s 300 ms > 265 ms in the fixture's per-cell table


def _seed_with_unflagged_control():
    """The smallest manifest seed whose active_fit draw (q = 2) is an unflagged cell (2.7 (i))."""
    for s in range(100):
        c = np.random.default_rng(np.random.SeedSequence([s, 2])).choice(np.array(IDS), size=1, replace=False)[0]
        if int(c) not in FLAGGED:
            return s
    raise AssertionError("no seed")


def _results(out, tid, names, values, arm):
    d = out / "fits" / ("twin_%d" % tid)
    d.mkdir(parents=True, exist_ok=True)
    rec = {"specimen_id": tid, "param_names": "|".join(names), "validation_status": "good", "arm": arm}
    rec.update(values)
    pd.DataFrame([rec]).to_csv(d / "phase2_results.csv", index=False)


def test_a8_1_merge_controls_only_reports(group, tmp_path):
    """8.1: only the active_fit twin has results -> exit 0; report/ and
    report_without_flagged/ hold the long table (the control's 6 rows), the FP
    summary, an EMPTY summary with exactly recovery_summary's columns, and a
    verdict with 'FAIL (axis absent)' on every gated axis; gate FAIL."""
    import ih_recovery_report as RR
    seed = _seed_with_unflagged_control()
    out = tmp_path / "a81"
    # renewed 2026-11-24 (Block 22): the default I_h arm is ih6_d207; this 8.1 check is written for ih5, and the
    # gbar box's low rail is 1e-5 since SPEC 2.9 (a)
    assert _plan(group, out, "--seed", str(seed), "--ih-arm", "ih5") == 0
    man = pd.read_csv(out / "manifest.csv")
    row = man[man["configuration"] == "active_fit"].iloc[0]
    tid = int(row["specimen_id"])
    vals = {n: float(row[TRUTH_COL[n]]) * 1.1 for n in IH5}
    vals["gbar"] = 1e-5                                     # the low rail: what a control should give
    _results(out, tid, IH5, vals, "ih5")
    assert rtr.main(["merge", "--out", str(out), "--no-plot"]) == 0
    gated = [a for a in IH5 if RR.RecoveryTolerances().for_axis(a) is not None]
    assert gated == ["Cm", "gbar", "kappa_tau"]             # the required axes of 2.8 (g)
    for sub in ("report", "report_without_flagged"):
        d = out / sub
        summ = pd.read_csv(d / "recovery_summary.csv")
        assert len(summ) == 0 and list(summ.columns) == SUMMARY_COLUMNS_SPEC, (sub, list(summ.columns))
        long_df = pd.read_csv(d / "recovery_per_cell.csv")
        assert len(long_df) == 5 and set(long_df["specimen_id"]) == {tid} and long_df["is_fp_control"].all()
        fp = pd.read_csv(d / "fp_control.csv")
        assert len(fp) >= 1 and set(fp["arm"]) == {"ih5"}
        ver = pd.read_csv(d / "gate_verdict.csv")
        assert "dv_h" not in set(ver["parameter"])         # not an axis of ih5, not required
        for a in gated:
            v = ver[ver["parameter"] == a]
            assert len(v) == 1 and v["verdict"].iloc[0] == "FAIL (axis absent)", (sub, a, ver.to_dict("records"))
        assert (d / "gate_verdict.txt").read_text().startswith("STAGE 7 GATE: FAIL")


def test_a8_1_merge_drop_railed_every_noncontrol_axis_railed(group, tmp_path):
    """Block 8, merge: 'returns 0 when the reports are written, whatever the
    gate's verdict, and 1 when no twin has results'. --drop-railed is one of
    plan's options. With a non-control twin whose every axis is on a bound of
    the box (all railed) and --drop-railed, recovery_summary's input after the
    drop is empty -- the same root cause as amendment 8.1 -- and a report is
    still due."""
    out = tmp_path / "a81r"
    assert _plan(group, out, "--seed", "4", "--drop-railed") == 0
    man = pd.read_csv(out / "manifest.csv")
    row = man[man["configuration"] == "full"].iloc[0]
    tid = int(row["specimen_id"])
    import ih_recovery_report as RR
    box = dict(RR.DEFAULT_BOX)
    rc = json.loads((out / "run_config.json").read_text())
    box["gbar"] = tuple(float(x) for x in rc["options"]["gbar_bounds"].split(","))
    box["kappa_tau"] = tuple(rc["tau_h_pk"]["bounds_kappa"])
    vals = {a: float(box[a][0]) for a in IH5}               # every estimate on the low bound -> rail
    _results(out, tid, IH5, vals, "ih5")
    try:
        code = rtr.main(["merge", "--out", str(out), "--no-plot"])
    except KeyError as exc:                                 # what the guard of 8.1 does not cover
        pytest.fail("merge raised KeyError(%s) with --drop-railed and every non-control axis railed" % exc)
    assert code == 0
    assert (out / "report" / "recovery_summary.csv").is_file()


def test_a8_2_gate_line_in_submitter_header():
    """8.2: the submitter's header gives the six suites' line exactly."""
    line = ("python smoke_twin_table.py && python smoke_twin_noise.py && python smoke_twin_manifest.py && "
            "python smoke_twin_response.py --build && python smoke_twin_build.py && python smoke_twin_recovery.py")
    text = (rtr.HERE / "submit_twin_recovery.sh").read_text()
    header = [ln for ln in text.splitlines() if ln.startswith("#")]
    assert any(ln.lstrip("# ").rstrip() == line for ln in header)
    for s in ("smoke_twin_table.py", "smoke_twin_noise.py", "smoke_twin_manifest.py", "smoke_twin_response.py",
              "smoke_twin_build.py", "smoke_twin_recovery.py"):
        assert (rtr.HERE / s).is_file(), s


def test_a8_3_plan_records_and_compares_base_seed(group, tmp_path):
    """8.3: default TwinConfig().base_seed; recorded as twin_config.base_seed
    (and among the options); a re-plan with another value is refused (exit 3)
    unless --force, which records the new one; the same value is a no-op."""
    out = tmp_path / "a83"
    assert _plan(group, out) == 0
    rc = json.loads((out / "run_config.json").read_text())
    assert rc["twin_config"]["base_seed"] == TwinConfig().base_seed == 20261119
    assert rc["twin_config"] == dataclasses.asdict(TwinConfig())
    assert _plan(group, out, "--base-seed", str(TwinConfig().base_seed)) == 0     # the same value: no-op
    assert _plan(group, out, "--base-seed", "7") == 3
    assert json.loads((out / "run_config.json").read_text())["twin_config"]["base_seed"] == 20261119
    assert _plan(group, out, "--base-seed", "7", "--force") == 0
    rc = json.loads((out / "run_config.json").read_text())
    assert rc["twin_config"]["base_seed"] == 7
    assert {k: v for k, v in rc["twin_config"].items() if k != "base_seed"} == \
        {k: v for k, v in dataclasses.asdict(TwinConfig()).items() if k != "base_seed"}


def test_a8_3_twin_refuses_bad_recorded_twin_config(group, tmp_path):
    """8.3: `twin` refuses (exit 3) a recorded twin_config whose keys are not
    exactly TwinConfig's fields: an unknown key, a missing key, no twin_config."""
    out = tmp_path / "a83k"
    assert _plan(group, out) == 0
    p = out / "run_config.json"
    good = json.loads(p.read_text())
    tid = int(pd.read_csv(out / "manifest.csv")["specimen_id"].iloc[0])
    for mutate in ("extra", "missing", "absent"):
        rc = json.loads(json.dumps(good))
        if mutate == "extra":
            rc["twin_config"]["no_such_field"] = 1
        elif mutate == "missing":
            del rc["twin_config"]["neuron_dt_ms"]
        else:
            del rc["twin_config"]
        p.write_text(json.dumps(rc))
        assert rtr.main(["twin", "--out", str(out), "--twin", str(tid)]) == 3, mutate
        assert not (out / "twins" / ("twin_%d" % tid)).exists(), mutate


def _ar1(xi, sigma, tau, dt):
    """(L2'): e_0 = sigma xi_0, e_j = rho e_{j-1} + sigma sqrt(1 - rho^2) xi_j, rho = exp(-dt / tau)."""
    rho = np.exp(-dt / tau)
    c = sigma * np.sqrt(1.0 - rho * rho)
    e = np.empty_like(xi)
    e[0] = sigma * xi[0]
    for j in range(1, xi.size):
        e[j] = rho * e[j - 1] + c * xi[j]
    return e


def _noise(seq, n, law, dt):
    """One draw of n samples of the stationary law (L1', no tilt), from SeedSequence seq (2.7 (e))."""
    import noise_law as NL
    u = NL.draw_units(np.random.default_rng(np.random.SeedSequence(seq)), 1, n)
    sf, sm, tm, ss, ts = law
    return sf * u.fast[0] + _ar1(u.mid[0], sm, tm, dt) + _ar1(u.slow[0], ss, ts, dt)


def _build_only(out, tid, monkeypatch, *extra):
    """`twin` with run_ih_fit.main replaced by a no-op: the build runs, the fit writes nothing."""
    import run_ih_fit
    monkeypatch.setattr(run_ih_fit, "main", lambda argv: None)
    return rtr.main(["twin", "--out", str(out), "--twin", str(tid)] + list(extra))


def test_a8_3_noise_depends_on_recorded_base_seed(group, tmp_path, monkeypatch):
    """8.3 with 2.7 (c), (e): one twin planned twice with base seeds 7 and 8
    (same options otherwise): the archives' SS windows and LS sweeps differ by
    exactly e_7 - e_8, the noises recomputed here from
    SeedSequence([b, g, 0, s, pos]) / ([b, g, 1, s]) (the clean response and
    the tilt line cancel); the archive records the base seed; twin uses the
    recorded neuron_dt_ms too. A reused archive whose base seed differs from
    the run's is refused (build_failed, exit 1) without --force."""
    import passive_fitting_hpc_fixed as mono
    tid = 900000200
    arch = {}
    for b in (7, 8):
        out = tmp_path / ("b%d" % b)
        assert _plan(group, out, "--seed", "4", "--base-seed", str(b)) == 0
        if b == 8:     # also: the recorded twin_config is used, not the defaults
            rc = json.loads((out / "run_config.json").read_text())
            rc["twin_config"]["neuron_dt_ms"] = 0.05
            (out / "run_config.json").write_text(json.dumps(rc))
        assert _build_only(out, tid, monkeypatch) == 1                    # fit_failed: the fit was a no-op
        st = json.loads((out / "status" / ("twin_%d.json" % tid)).read_text())
        assert st["status"] == "fit_failed" and st["built"] is True, st
        d = out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid)
        meta = json.loads((d / "metadata.json").read_text())
        assert meta["twin"]["base_seed"] == b
        if b == 8:
            assert meta["twin"]["neuron_dt_ms"] == 0.05
        with np.load(d / "ss_pulses.npz") as z, np.load(d / "ls_sweeps.npz") as y:
            arch[b] = (z["v"].copy(), z["sweep_number"].copy(), {k: y[k].copy() for k in y.files}, meta)
    # the neuron step differs between the two runs, so the clean parts differ too: compare the noise
    # through a third run at base seed 8 and the default step instead
    out = tmp_path / "b8d"
    assert _plan(group, out, "--seed", "4", "--base-seed", "8") == 0
    assert _build_only(out, tid, monkeypatch) == 1
    d = out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid)
    with np.load(d / "ss_pulses.npz") as z, np.load(d / "ls_sweeps.npz") as y:
        arch["8d"] = (z["v"].copy(), z["sweep_number"].copy(), {k: y[k].copy() for k in y.files},
                      json.loads((d / "metadata.json").read_text()))
    # expected noises
    man = pd.read_csv(tmp_path / "b7" / "manifest.csv")
    real = int(man[man["specimen_id"] == tid]["twin_of_specimen_id"].iloc[0])
    prot = tt.load_cell_protocol(group["gdir"] / ("specimen_%d" % real), spike_reason_fn=mono.sweep_spike_reason)
    table = tt.read_twin_sweep_table(tmp_path / "b7" / "twin_sweeps_G.csv")
    dt_ss = 1.0 / prot.ss_fs_Hz
    n_w = prot.ss_t_s.size
    v7, sw7, ls7, _ = arch[7]
    v8, sw8, ls8, meta8 = arch["8d"]
    assert_array_equal(sw7, prot.ss_sweep_number)
    for k in range(v7.shape[0]):
        s, pos = int(prot.ss_sweep_number[k]), int(prot.ss_position[k])
        r = table[(table["specimen_id"] == real) & (table["protocol"] == "SS") & (table["sweep_number"] == s)].iloc[0]
        law = (r["sigma_f_mV"], r["sigma_m_mV"], r["tau_m_s"], r["sigma_s_mV"], r["tau_s_s"])
        de = _noise([7, tid, 0, s, pos], n_w, law, dt_ss) - _noise([8, tid, 0, s, pos], n_w, law, dt_ss)
        # 1e-12 mV: float64 roundoff of sums of O(0.1 mV) terms over at most ~1e4 recursion steps
        np.testing.assert_allclose(v7[k] - v8[k], de, rtol=0, atol=1e-12)
    for ent in meta8["ls_sweeps"]:
        m, s = int(ent["index"]), int(ent["sweep_number"])
        r = table[(table["specimen_id"] == real) & (table["protocol"] == "LS") & (table["sweep_number"] == s)].iloc[0]
        law = (r["sigma_f_mV"], r["sigma_m_mV"], r["tau_m_s"], r["sigma_s_mV"], r["tau_s_s"])
        n = ls7["v_%d" % m].size
        dt = 1.0 / float(r["fs_Hz"])
        de = _noise([7, tid, 1, s], n, law, dt) - _noise([8, tid, 1, s], n, law, dt)
        np.testing.assert_allclose(ls7["v_%d" % m] - ls8["v_%d" % m], de, rtol=0, atol=1e-11)
    # a reused archive built under another base seed is refused, then rebuilt with --force
    out7 = tmp_path / "b7"
    assert _plan(group, out7, "--seed", "4", "--base-seed", "9", "--force") == 0
    assert _build_only(out7, tid, monkeypatch) == 1
    st = json.loads((out7 / "status" / ("twin_%d.json" % tid)).read_text())
    assert st["status"] == "build_failed" and "base_seed" in st["reason"], st
    assert _build_only(out7, tid, monkeypatch, "--force") == 1
    meta = json.loads((out7 / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid) / "metadata.json").read_text())
    assert meta["twin"]["base_seed"] == 9


def test_a8_3_negative_base_seed(group, tmp_path, monkeypatch):
    """Probe (spec silent): plan --base-seed -1 is accepted by the parser; numpy's
    SeedSequence refuses negative entries, so the noise of 2.7 (e) cannot be drawn.
    Records what each sub-command does: plan should refuse it, or twin must
    end build_failed (never an uncaught error)."""
    out = tmp_path / "neg"
    code = _plan(group, out, "--base-seed=-1")
    if code != 0:
        return                                   # refused at plan: fine
    tid = int(pd.read_csv(out / "manifest.csv")["specimen_id"].iloc[-1])
    assert _build_only(out, tid, monkeypatch) == 1
    st = json.loads((out / "status" / ("twin_%d.json" % tid)).read_text())
    assert st["status"] == "build_failed"
    pytest.xfail("plan accepts --base-seed -1 (exit 0); the twin then ends build_failed: %s" % st["reason"][:200])


# ===========================================================================
#  Block 17 (integration tester, 2026-11-22, third incremental run).
#  Oracles: the Block 17 entry and SPEC 2.8 (a), (e), (f), (g). Expected
#  values: tau_h^{base,pk} as the closed-form maximum of the mechanism's tau(v)
#  (constants read from mod/Ih_human.mod); m_inf(v) = 1 / (1 + exp((v + 90.87) /
#  8.05)); the noise of the full twin written out from 2.7 (c)-(e); the
#  loss_verdict rule of the entry (2 % of the other loss); evaluate_gate with
#  required_axes=None against ih_recovery_report.py at the base commit.
# ===========================================================================
import math                                    # noqa: E402
import re                                      # noqa: E402
import subprocess                              # noqa: E402

# renewed 2026-11-24 (Block 22): the Block 17 tests below pin the run's I_h arm to ih5 (the default is now
# ih6_d207) and keep the refinement's budget small (its default, nelder_mead with maxfev 100, is a plan option)
# [note 2026-11-25: since Block 28 the plan's default refinement is trf, --refine-max-nfev 30; --refine-maxfev is
# then unused by the fit (Block 27)]
B17_SMALL = ["--n-per-configuration", "1", "--n-no-noise", "1", "--n-calls", "1", "--n-initial", "1",
             "--ih-arm", "ih5", "--refine-maxfev", "4", "--refine-starts", "1"]
BASE17 = "216ecf13ccca455903b2cb05d740dd64f3feff08"


def _plan17(group, out, *extra):
    return rtr.main(["plan", "--group-dir", str(group["gdir"]), "--noise-dir", str(group["ndir"]), "--out", str(out)]
                    + B17_SMALL + list(extra))


def _tau_base_closed_form():
    txt = (rtr.HERE / "mod" / "Ih_human.mod").read_text()
    c = {k: float(re.search(r"^\s*%s\s*=\s*([-+0-9.eE]+)" % k, txt, re.M).group(1))
         for k in ("aa", "bb", "cc", "dd", "ff")}
    v = (math.log(c["bb"] / c["dd"]) + c["cc"] - c["aa"]) / (c["bb"] + c["dd"])
    return c["ff"] + 1.0 / (math.exp(-c["aa"] - c["bb"] * v) + math.exp(-c["cc"] + c["dd"] * v))


def _minf(v):
    return 1.0 / (1.0 + math.exp((v + 90.87) / 8.05))


def test_b17_plan_records_and_manifest(group, tmp_path):
    out = tmp_path / "p17"
    assert _plan17(group, out, "--seed", "4") == 0
    rc = json.loads((out / "run_config.json").read_text())
    tau = rc["tau_h_pk"]
    base = _tau_base_closed_form()
    assert abs(tau["base_ms"] - base) <= 1e-3                     # the 0.01 mV scan of the code
    assert tau["bounds_ms"] == [50.0, 1000.0] and tau["truth_ms"] == [150.0, 700.0]
    assert tau["bounds_kappa"] == [50.0 / tau["base_ms"], 1000.0 / tau["base_ms"]]
    assert tau["truth_kappa"] == [150.0 / tau["base_ms"], 700.0 / tau["base_ms"]]
    assert rc["dvh_truth_mV"] == 0.0 and rc["gate_arm"] == "ih5"
    assert rc["gate_required_axes"] == ["Cm", "gbar", "kappa_tau"]
    o = rc["options"]
    assert o["gbar_bounds"] == "1e-5,1e-3" and o["tau_h_pk_bounds_ms"] == "50,1000" and o["n_no_noise"] == 1
    assert "kappa_bounds" not in o and "kappa_range" not in o and "dvh_range" not in o
    assert rc["truth_kwargs"]["ih_kappa_range"] == tau["truth_kappa"] and rc["truth_kwargs"]["ih_dvh_range_mV"] is None
    import twin_manifest as TM
    man = TM.load_twin_manifest(out / "manifest.csv")
    assert list(man["configuration"]) == ["full", "passive_fit", "active_fit", "full_no_noise"]
    assert list(man["arm"]) == ["ih5", "passive_fullstep", "ih5", "ih5"]
    assert list(man["specimen_id"]) == [900000000, 900000100, 900000200, 900000300]
    ih = man["configuration"].isin(["full", "passive_fit", "full_no_noise"])
    assert (man.loc[ih, "ih_dvh_mV"] == 0.0).all()
    k = man.loc[ih, "ih_kappa_tau"].to_numpy(dtype=float)
    assert np.all((k >= tau["truth_kappa"][0]) & (k <= tau["truth_kappa"][1]))
    nn, fu = man.iloc[3], man.iloc[0]
    assert int(nn["paired_with"]) == 900000000 and not bool(nn["with_noise"])
    assert TM.twin_theta(nn) == TM.twin_theta(fu) and int(nn["twin_of_specimen_id"]) == int(fu["twin_of_specimen_id"])
    # the manifest is Block 16's on real_cells with the recorded truth kwargs
    rc_cells = pd.read_csv(out / "real_cells_G.csv")
    kw = dict(rc["truth_kwargs"])
    for key in ("ih_kappa_range", "ih_gbar_range_S_cm2"):
        if kw.get(key) is not None:
            kw[key] = tuple(kw[key])
    kw["ih_regions"] = tuple(kw["ih_regions"])
    ref = TM.draw_twin_manifest(rc_cells[list(TM.REAL_CELL_COLUMNS)], n_per_configuration=1, n_no_noise=1, seed=4,
                                id_base=900000000, ih_arm="ih5", dvh_truth_mV=0.0, truth_kwargs=kw)
    pd.testing.assert_frame_equal(man, ref)
    # list: the four twins (the submitter counts them for the cap)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert rtr.main(["list", "--out", str(out)]) == 0
    assert buf.getvalue().split() == ["900000000", "900000100", "900000200", "900000300"]


@pytest.mark.parametrize("old", [["--kappa-bounds", "0.5,2"], ["--kappa-range", "0.5,2"], ["--dvh-range=-5,5"]])
def test_b17_old_options_gone(group, tmp_path, old):
    assert _plan17(group, tmp_path / "o", *old) == 2


def test_b17_n_no_noise_above_n_refused(group, tmp_path):
    assert rtr.main(["plan", "--group-dir", str(group["gdir"]), "--noise-dir", str(group["ndir"]),
                     "--out", str(tmp_path / "x"), "--n-per-configuration", "1", "--n-calls", "1",
                     "--n-initial", "1"]) == 3                   # the default --n-no-noise 5 > 1


def test_b17_fit_command_and_noise_free_twin(group, tmp_path, monkeypatch):
    """TS11 and TS12's first half: the fit's command line per configuration (recorded), and the
    full_no_noise archive = its partner's archive minus the partner's noise (2.7 (c)-(e) written out)."""
    import run_ih_fit as R
    import twin_manifest as TM
    import passive_fitting_hpc_fixed as mono
    out = tmp_path / "c17"
    assert _plan17(group, out, "--seed", "4") == 0
    seen = {}
    monkeypatch.setattr(R, "main", lambda argv: seen.__setitem__(len(seen), list(argv)))
    man = TM.load_twin_manifest(out / "manifest.csv")
    for _, row in man.iterrows():
        tid = int(row["specimen_id"])
        n0 = len(seen)
        assert rtr.main(["twin", "--out", str(out), "--twin", str(tid)]) == 1   # no results written: fit_failed
        st = json.loads((out / "status" / ("twin_%d.json" % tid)).read_text())
        assert st["built"] is True and st["status"] == "fit_failed", st
        argv = seen[n0]
        assert argv[argv.index("--arm") + 1] == row["arm"]
        assert "--tau-h-pk-bounds-ms=50,1000" in argv and not any(a.startswith("--kappa-bounds") for a in argv)
        ev = [a for a in argv if a.startswith("--eval-theta")]
        conf = row["configuration"]
        if conf in ("full", "full_no_noise"):
            assert len(ev) == 1
            th = dict(kv.split("=") for kv in ev[0].split("=", 1)[1].split(","))
            assert list(th) == list(R.IH5_NAMES)
            truth = TM.twin_theta(row)
            for n in R.IH5_NAMES:
                assert float(th[n]) == float(truth[n]), n           # exactly: repr round-trips float64
        else:
            assert ev == [], conf
        a = R._parse_args(argv)                                   # the fit's own parser accepts it
        assert R.resolve_arm_config(a).name == row["arm"]
        meta = json.loads((out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid) / "metadata.json").read_text())
        assert meta["twin"]["with_noise"] is bool(row["with_noise"])
    # the noise-free twin: the full twin's traces minus its noise and tilt line
    full_id, nn_id = 900000000, 900000300
    rd = {}
    for t in (full_id, nn_id):
        d = out / "twins" / ("twin_%d" % t) / ("specimen_%d" % t)
        with np.load(d / "ss_pulses.npz") as z, np.load(d / "ls_sweeps.npz") as y:
            rd[t] = (z["v"].copy(), {k: y[k].copy() for k in y.files}, json.loads((d / "metadata.json").read_text()))
    real = int(man.iloc[0]["twin_of_specimen_id"])
    prot = tt.load_cell_protocol(group["gdir"] / ("specimen_%d" % real), spike_reason_fn=mono.sweep_spike_reason)
    table = tt.read_twin_sweep_table(out / "twin_sweeps_G.csv")
    base = json.loads((out / "run_config.json").read_text())["twin_config"]["base_seed"]
    dt = 1.0 / prot.ss_fs_Hz
    n_w = prot.ss_t_s.size
    vf, vn = rd[full_id][0], rd[nn_id][0]
    for s in np.unique(prot.ss_sweep_number):
        r = table[(table["specimen_id"] == real) & (table["protocol"] == "SS") & (table["sweep_number"] == s)].iloc[0]
        law = (r["sigma_f_mV"], r["sigma_m_mV"], r["tau_m_s"], r["sigma_s_mV"], r["tau_s_s"])
        m = np.flatnonzero(prot.ss_sweep_number == s)
        pos = prot.ss_position[m]
        u = (pos[:, None] * prot.ss_period_samples + np.arange(n_w)[None, :]) * dt
        for j, k in enumerate(m):
            e = _noise([base, full_id, 0, int(s), int(pos[j])], n_w, law, dt) + r["tilt_mV_per_s"] * (u[j] - u.mean())
            np.testing.assert_allclose(vf[k] - vn[k], e, rtol=0, atol=1e-11)   # roundoff of mV-sized sums
    for ent in rd[nn_id][2]["ls_sweeps"]:
        mi, s = int(ent["index"]), int(ent["sweep_number"])
        r = table[(table["specimen_id"] == real) & (table["protocol"] == "LS") & (table["sweep_number"] == s)].iloc[0]
        law = (r["sigma_f_mV"], r["sigma_m_mV"], r["tau_m_s"], r["sigma_s_mV"], r["tau_s_s"])
        n = rd[nn_id][1]["v_%d" % mi].size
        e = _noise([base, full_id, 1, s], n, law, dt) + r["tilt_mV_per_s"] * (np.arange(n) - (n - 1) / 2) * dt
        np.testing.assert_allclose(rd[full_id][1]["v_%d" % mi] - rd[nn_id][1]["v_%d" % mi], e, rtol=0, atol=1e-11)
    # reuse only when with_noise matches the manifest's
    d = out / "twins" / ("twin_%d" % nn_id) / ("specimen_%d" % nn_id)
    meta = json.loads((d / "metadata.json").read_text())
    meta["twin"]["with_noise"] = True
    (d / "metadata.json").write_text(json.dumps(meta))
    assert rtr.main(["twin", "--out", str(out), "--twin", str(nn_id)]) == 1
    st = json.loads((out / "status" / ("twin_%d.json" % nn_id)).read_text())
    assert st["status"] == "build_failed" and "with_noise" in st["reason"], st


@pytest.fixture(scope="module")
def e2e17(group, tmp_path_factory):
    """Plan, the four twins built and fitted at the smallest budget, merge; twice, in two run dirs."""
    runs = []
    for name in ("r1", "r2"):
        out = tmp_path_factory.mktemp("e17") / name
        assert _plan17(group, out, "--seed", "4") == 0
        codes = [rtr.main(["twin", "--out", str(out), "--twin", str(t)])
                 for t in (900000000, 900000100, 900000200, 900000300)]
        assert rtr.main(["merge", "--out", str(out), "--no-plot"]) == 0
        runs.append((out, codes))
    return runs


def test_b17_end_to_end_fits_and_records(e2e17):
    """Block 15's columns of 2.8 (f) on the fits Block 17 ran (I3, I5 in substance)."""
    import twin_manifest as TM
    out, codes = e2e17[0]
    assert codes == [0, 0, 0, 0], codes
    rc = json.loads((out / "run_config.json").read_text())
    base = rc["tau_h_pk"]["base_ms"]
    man = TM.load_twin_manifest(out / "manifest.csv")
    for _, row in man.iterrows():
        tid = int(row["specimen_id"])
        fd = out / "fits" / ("twin_%d" % tid)
        res = pd.read_csv(fd / "phase2_results.csv")
        r = res[res["specimen_id"] == tid].iloc[0]
        names = r["param_names"].split("|") if isinstance(r["param_names"], str) else []
        conf = row["configuration"]
        exp_names = ["Cm", "Rm", "Ra"] if conf == "passive_fit" else ["Cm", "Rm", "Ra", "gbar", "kappa_tau"]
        assert names == exp_names, (conf, names)
        assert (r["ls_role_rule"], r["weighting"], r["ls_sample_weight"]) == ("d207", "ls_amplitude", "peak_decay")
        assert r["ls_tau_d_ms"] == 500.0
        roles = pd.read_csv(fd / "ls_roles.csv")
        hyp = roles[roles["polarity"] == "hyp"].copy()
        hyp["a"] = hyp["amplitude_pA"].abs()
        hyp = hyp.sort_values("a")
        assert list(hyp["role"]) == ["validate", "train"]           # n = 2: h_1 validates, h_2 trains (2.8 (b))
        v_star = float(hyp.loc[hyp["role"] == "train", "v_trough_mV"].min())
        assert r["v_deepest_train_mV"] == pytest.approx(v_star, rel=1e-12)
        # loss_at_estimate is the training loss at the estimate: the optimiser's value (1e-9 relative)
        assert r["loss_at_estimate"] == pytest.approx(r["train_rel_loss"], rel=1e-9)
        if conf == "passive_fit":
            assert math.isnan(r["tau_h_pk_ms"]) and math.isnan(r["gh_at_v_deepest_S_cm2"])
            assert math.isnan(r["loss_at_eval_theta"])
        else:
            assert r["tau_h_base_pk_ms"] == pytest.approx(base, rel=1e-15)
            assert r["tau_h_pk_ms"] == pytest.approx(r["kappa_tau"] * base, rel=1e-12)
            assert r["gh_at_v_deepest_S_cm2"] == pytest.approx(r["gbar"] * _minf(v_star), rel=1e-12)
            if conf == "active_fit":
                assert math.isnan(r["loss_at_eval_theta"])
            else:
                assert np.isfinite(r["loss_at_eval_theta"]) and r["loss_at_eval_theta"] > 0


def test_b17_merge_reports_and_extras(e2e17):
    import twin_manifest as TM
    import run_ih_fit as R
    out, _ = e2e17[0]
    rc = json.loads((out / "run_config.json").read_text())
    base = rc["tau_h_pk"]["base_ms"]
    man = TM.load_twin_manifest(out / "manifest.csv")
    noisy = set(man.loc[man["with_noise"], "specimen_id"].astype(int))
    clean = set(man.loc[~man["with_noise"], "specimen_id"].astype(int))
    assert clean == {900000300}
    flagged = set(man.loc[man["cell_tau_s_above_window"].astype(bool), "specimen_id"].astype(int))
    for sub, want in (("report", noisy), ("report_without_flagged", noisy - flagged), ("report_no_noise", clean)):
        d = out / sub
        if not want:
            assert (d / "NO_RESULTS.txt").is_file(), sub
            continue
        long_df = pd.read_csv(d / "recovery_per_cell.csv")
        assert set(long_df["specimen_id"].astype(int)) == want, sub
        ver = pd.read_csv(d / "gate_verdict.csv")
        ih5 = ver[ver["arm"] == "ih5"]
        assert "dv_h" not in set(ih5["parameter"]), sub            # 2.8 (g): not an axis of ih5, not failed
        for a in ("Cm", "gbar", "kappa_tau"):
            assert (ih5["parameter"] == a).sum() == 1, (sub, a)
    byc = pd.read_csv(out / "recovery_by_configuration.csv")
    assert set(byc["configuration"]) == {"full", "passive_fit", "active_fit", "full_no_noise"}
    ex = pd.read_csv(out / "recovery_extras.csv", float_precision="round_trip")
    # updated 2026-11-25: SPEC Block 28 -- recovery_extras.csv gains l2_after_search, l2_after_refine,
    # refine_n_converged, refine_n_failed, sqrt_l2_at_estimate, sqrt_l2_at_truth and on_bound_<name> for the arm's
    # names (Fit options), and loss_verdict_on (loss_verdict amendment 3, D-250). The SPEC fixes no position for
    # these, so the earlier columns are checked in their order and the new ones by name.
    arm_names = set()
    for arm in set(man["arm"]):
        arm_names |= set(R.ARMS[arm].fit_params)
    new28 = ["l2_after_search", "l2_after_refine", "refine_n_converged", "refine_n_failed",
             "sqrt_l2_at_estimate", "sqrt_l2_at_truth", "loss_verdict_on"] + ["on_bound_" + n for n in arm_names]
    assert set(new28) <= set(ex.columns)
    assert {c for c in ex.columns if c.startswith("on_bound_")} == {"on_bound_" + n for n in arm_names}
    assert [c for c in ex.columns if c not in new28] == [
        "twin_id", "configuration", "twin_of_specimen_id", "paired_with", "with_noise",
        "cell_tau_s_above_window", "tau_h_pk_true_ms", "tau_h_pk_est_ms", "loss_at_truth",
        "loss_at_estimate", "loss_verdict", "v_deepest_train_mV", "gh_deep_true_S_cm2",
        "gh_deep_est_S_cm2", "ln_gh_deep_ratio",
        # renewed 2026-11-24: Block 22's columns (SPEC 2.9 (e))
        "loss_after_search", "refine_improved", "refine_n_evals", "refine_best_start",
        "dvh_true_mV", "dvh_est_mV"]
    assert list(ex["twin_id"]) == list(man["specimen_id"])
    for _, e in ex.iterrows():
        tid = int(e["twin_id"])
        row = man[man["specimen_id"] == tid].iloc[0]
        r = pd.read_csv(out / "fits" / ("twin_%d" % tid) / "phase2_results.csv", float_precision="round_trip").iloc[0]
        th = TM.twin_theta(row)
        conf = row["configuration"]
        assert e["configuration"] == conf and int(e["paired_with"]) == int(row["paired_with"])
        assert bool(e["with_noise"]) == bool(row["with_noise"])
        assert e["v_deepest_train_mV"] == pytest.approx(r["v_deepest_train_mV"], rel=1e-15)
        # SPEC Block 28: the new columns are the fit's (Block 27)
        for c in ("l2_after_search", "l2_after_refine"):
            assert e[c] == pytest.approx(r[c], rel=1e-15, nan_ok=True), c
        for c in ("refine_n_converged", "refine_n_failed"):
            assert int(e[c]) == int(r[c]), c
        fitted = set(r["param_names"].split("|"))
        for n in arm_names:                              # empty for a name the arm does not fit
            v = e["on_bound_" + n]
            if n not in fitted:
                assert pd.isna(v) or v == "", (n, v)
            else:
                assert (pd.isna(v) and pd.isna(r["on_bound_" + n])) or v == r["on_bound_" + n], (n, v)
        if conf == "active_fit":
            assert math.isnan(e["tau_h_pk_true_ms"]) and math.isnan(e["gh_deep_true_S_cm2"])
        else:
            assert e["tau_h_pk_true_ms"] == pytest.approx(th["kappa_tau"] * base, rel=1e-12)
            assert e["gh_deep_true_S_cm2"] == pytest.approx(th["gbar"] * _minf(e["v_deepest_train_mV"]), rel=1e-12)
        lt, le = e["loss_at_truth"], e["loss_at_estimate"]
        if conf in ("full", "full_no_noise"):
            # 1e-15 relative: merge reads phase2_results.csv with pandas' default float parser (last digit)
            assert lt == pytest.approx(r["loss_at_eval_theta"], rel=1e-15)
            assert le == pytest.approx(r["loss_at_estimate"], rel=1e-15)
            # updated 2026-11-25: SPEC Block 28, loss_verdict under trf (amendment 3, D-250): with finite
            # sqrt_l2_at_estimate and sqrt_l2_at_truth the verdict of 2.8 (f) compares THESE two by the same rule
            # and tolerance (a value at or above 1e3, a failed point, gives na); otherwise it compares L as before
            se, sq = e["sqrt_l2_at_estimate"], e["sqrt_l2_at_truth"]
            assert se == pytest.approx(r["sqrt_l2_at_estimate"], rel=1e-15)
            assert sq == pytest.approx(r["sqrt_l2_at_eval_theta"], rel=1e-15)   # Block 28: the fit's column
            if np.isfinite(se) and np.isfinite(sq):
                assert e["loss_verdict_on"] == "sqrt_l2"
                a_e, a_t = se, sq
            else:
                assert e["loss_verdict_on"] == "loss"
                a_e, a_t = le, lt
            if np.isfinite(se) and np.isfinite(sq) and max(se, sq) >= 1e3:
                exp = "na"
            else:
                exp = "optimiser" if a_e - a_t > 0.02 * a_e else ("data" if a_t - a_e > 0.02 * a_t else "tie")
            assert e["loss_verdict"] == exp
        else:
            assert math.isnan(lt) and e["loss_verdict"] == "na"
        if conf == "passive_fit":
            assert math.isnan(e["tau_h_pk_est_ms"]) and math.isnan(e["gh_deep_est_S_cm2"])
            assert math.isnan(e["ln_gh_deep_ratio"])
        elif conf != "active_fit":
            assert e["ln_gh_deep_ratio"] == pytest.approx(math.log(e["gh_deep_est_S_cm2"] / e["gh_deep_true_S_cm2"]),
                                                          rel=1e-12)
    ts = pd.read_csv(out / "twins_status.csv")
    assert list(ts["status"]) == ["ok"] * 4


def test_b17_two_runs_identical(e2e17):
    (a, _), (b, _) = e2e17
    for f in ("manifest.csv", "twin_sweeps_G.csv", "recovery_extras.csv"):
        pa, pb = pd.read_csv(a / f), pd.read_csv(b / f)
        pd.testing.assert_frame_equal(pa, pb)
    for t in (900000000, 900000100, 900000200, 900000300):
        ra = pd.read_csv(a / "fits" / ("twin_%d" % t) / "phase2_results.csv")
        rb = pd.read_csv(b / "fits" / ("twin_%d" % t) / "phase2_results.csv")
        num = [c for c in ra.columns if ra[c].dtype.kind == "f" and "seconds" not in c and "time" not in c]
        pd.testing.assert_frame_equal(ra[num], rb[num])
        for part in ("ss_pulses.npz", "ls_sweeps.npz"):
            za = np.load(a / "twins" / ("twin_%d" % t) / ("specimen_%d" % t) / part)
            zb = np.load(b / "twins" / ("twin_%d" % t) / ("specimen_%d" % t) / part)
            for k in za.files:
                assert_array_equal(za[k], zb[k])


def _old_report(_c={}):
    if "m" in _c:
        return _c["m"]
    import importlib.util
    import sys
    import tempfile
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=rtr.HERE, capture_output=True, text=True,
                         check=True).stdout.strip()
    rel = (Path(rtr.HERE).resolve().relative_to(top) / "ih_recovery_report.py").as_posix()
    src = subprocess.run(["git", "show", "%s:%s" % (BASE17, rel)], cwd=top, capture_output=True, check=True).stdout
    p = Path(tempfile.mkdtemp()) / "ih_recovery_report_base.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("ih_recovery_report_base", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ih_recovery_report_base"] = m
    spec.loader.exec_module(m)
    _c["m"] = m
    return m


def _summary(arm, errs):
    rows = []
    for p, e in errs.items():
        rows.append(dict(arm=arm, parameter=p, n=5, n_railed=0, median_abs_err=e, q25_abs_err=e, q75_abs_err=e,
                         median_err=e, err_kind="log_ratio" if p != "dv_h" else "diff"))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("errs", [
    dict(Cm=0.05, Rm=0.1, Ra=0.3, gbar=0.2, kappa_tau=0.1),                 # all within: pass on ih5
    dict(Cm=0.05, Rm=0.1, Ra=0.3, gbar=5.0, kappa_tau=0.1),                 # gbar out: fail
    dict(Cm=0.05, Rm=0.1, Ra=0.3, gbar=0.2),                                # kappa absent: fail (required)
    dict(Cm=0.05, Rm=0.1, Ra=0.3, gbar=0.2, dv_h=9.0, kappa_tau=0.1),       # dv_h present: judged as before
])
def test_b17_evaluate_gate_required_axes(errs):
    import ih_recovery_report as RR
    old = _old_report()
    summ = _summary("ih5", errs)
    fp = pd.DataFrame()
    tol = RR.RecoveryTolerances()
    ver, ok = RR.evaluate_gate(summ, fp, arm="ih5", tol=tol, required_axes=("Cm", "gbar", "kappa_tau"))
    # the axes that decide: the three required ones present and within, plus any present gated axis
    def within(a):
        t = tol.for_axis(a)
        return a in errs and errs[a] <= t
    exp_axes = all(within(a) for a in ("Cm", "gbar", "kappa_tau")) and \
        (("dv_h" not in errs) or within("dv_h"))
    rows = ver[ver["arm"] == "ih5"] if "arm" in ver.columns else ver
    if "dv_h" not in errs:
        assert "dv_h" not in set(rows["parameter"])
    if "kappa_tau" not in errs:
        assert rows[rows["parameter"] == "kappa_tau"]["verdict"].iloc[0] == "FAIL (axis absent)"
    # the axes rows decide as the tolerances say (the control rows may decide the rest)
    ax_rows = rows[rows["parameter"].isin(["Cm", "gbar", "kappa_tau", "dv_h"])]
    assert (not ax_rows["verdict"].astype(str).str.startswith("FAIL").any()) == exp_axes
    # required_axes=None: the old verdict table, exactly
    v_new, ok_new = RR.evaluate_gate(summ, fp, arm="ih5", tol=tol)
    v_old, ok_old = old.evaluate_gate(summ, fp, arm="ih5", tol=old.RecoveryTolerances())
    pd.testing.assert_frame_equal(v_new, v_old)
    assert ok_new == ok_old
    with pytest.raises(ValueError):
        RR.evaluate_gate(summ, fp, arm="ih5", tol=tol, required_axes=("Cm", "dvh"))


def test_b17_submitter_usage_names_new_options():
    text = (rtr.HERE / "submit_twin_recovery.sh").read_text()
    for opt in ("--n-no-noise", "--gbar-bounds", "--tau-h-pk-bounds-ms", "--tau-h-pk-truth-ms"):
        assert opt in text, opt
    raw = (rtr.HERE / "submit_twin_recovery.sh").read_bytes()
    assert all(c < 128 for c in raw) and b"\r" not in raw
    for f in ("run_twin_recovery.py", "ih_recovery_report.py"):
        raw = (rtr.HERE / f).read_bytes()
        assert all(c < 128 for c in raw) and b"\r" not in raw, f
    assert subprocess.run(["bash", "-n", str(rtr.HERE / "submit_twin_recovery.sh")]).returncode == 0
