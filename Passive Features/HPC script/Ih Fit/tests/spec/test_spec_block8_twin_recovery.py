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
    assert _plan(group, out, "--seed", str(seed)) == 0
    man = pd.read_csv(out / "manifest.csv")
    row = man[man["configuration"] == "active_fit"].iloc[0]
    tid = int(row["specimen_id"])
    vals = {n: float(row[TRUTH_COL[n]]) * 1.1 for n in IH6 if n != "dv_h"}
    vals["dv_h"] = 1.0
    vals["gbar"] = 1e-6                                     # the low rail: what a control should give
    _results(out, tid, IH6, vals, "ih6")
    assert rtr.main(["merge", "--out", str(out), "--no-plot"]) == 0
    gated = [a for a in IH6 if RR.RecoveryTolerances().for_axis(a) is not None]
    assert gated == ["Cm", "gbar", "dv_h", "kappa_tau"]
    for sub in ("report", "report_without_flagged"):
        d = out / sub
        summ = pd.read_csv(d / "recovery_summary.csv")
        assert len(summ) == 0 and list(summ.columns) == SUMMARY_COLUMNS_SPEC, (sub, list(summ.columns))
        long_df = pd.read_csv(d / "recovery_per_cell.csv")
        assert len(long_df) == 6 and set(long_df["specimen_id"]) == {tid} and long_df["is_fp_control"].all()
        fp = pd.read_csv(d / "fp_control.csv")
        assert len(fp) >= 1 and set(fp["arm"]) == {"ih6"}
        ver = pd.read_csv(d / "gate_verdict.csv")
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
    vals = {a: float(box[a][0]) for a in IH6}               # every estimate on the low bound -> rail
    _results(out, tid, IH6, vals, "ih6")
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
