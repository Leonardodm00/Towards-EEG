"""Smoke test for the Stage-7 run on the twins -- Blocks 8 and 17 in specs/SPEC.md (TS1-TS14).

The fixture: a group of 4 real-layout fixture cells (smoke_noise_windows.write_fixture:
one SS sweep with two trace-QC drops; LS sweeps of -110, -70, -30 and +30 pA)
under an archive root whose path holds a space, each with a ball-and-stick
reconstruction.swc (synthetic_ground_truth.write_ball_and_stick_swc), and the
noise-fit run's two tables written by hand in the formats of SPEC section 3
(run_noise_fit's columns; two of the four cells with tau_s above 265 ms, so
flagged by Block 5). Every run uses --n-per-configuration 1 --n-no-noise 1:
four twins, one per configuration (Block 17: full, passive_fit, active_fit,
full_no_noise).

Checks
    test_TS1_plan              the four files; manifest.csv = draw_twin_manifest called here with
                               truth kwargs recorded from run_ih_recovery.phase_manifest (Block 17: no
                               dv_h range, the kappa range = tau_h_pk's truth range / base, n_no_noise,
                               dv_h truth 0); the new options and defaults, the old ones gone; the
                               tau_h_pk record of run_config.json; the twins' table = Block 5's built
                               here; v_rest_mV = the loader's; a second plan changes nothing; another
                               seed refused without --force; the refusals; versions() with NEURON's pip
                               metadata hidden (amendment 10.2)
    test_TS2_fit_command       run_ih_fit.main replaced by a recorder: each twin's argv equals the one
                               run_ih_recovery.phase_fit builds from the same values (non-default and
                               negative values included) with Block 17's changes (the arm ih5,
                               --tau-h-pk-bounds-ms for --kappa-bounds, --eval-theta), with the twin's
                               own directories and arm
    test_TS3_end_to_end        each twin built and fitted (three as subprocesses, the active_fit one in
                               process) at the smallest budget run_ih_fit accepts (1 call, 1 initial
                               point): the loader reads the archive, phase2_results.csv holds the
                               twin's row, status ok; active_fit: no I_h in the cell, five parameters fitted
    test_TS4_model_identity    the IhSpec and ParamSpec handed to twin_build.build_twin equal
                               run_ih_fit.build_ih_spec / build_param_spec on the same options
    test_TS5_merge             report/ = ih_recovery_report called here on the same frames;
                               report_without_flagged/ leaves out exactly the flagged twins;
                               recovery_by_configuration.csv; twins_status.csv; 0 / 1 exit codes
    test_TS6_resume_failure    a second `twin` neither rebuilds nor refits; --force does both; a
                               missing real cell ends build_failed, exit 1, with a status file
    test_TS7_submitter         bash -n; dry run (one array qsub -J 0-2 ncpus=1 by default, one merge qsub
                               afterany, three twin commands and the merge command, each accepted by
                               the parser; nothing written or submitted); NCPUS=4: 31 x 4 cores refused (3);
                               MAX_TWINS=2; a stubbed submission; the job roles; the marker lines;
                               every call with a DISPLAY no X server answers (amendment 10)
    test_TS8_TS14_ascii_lf_frozen
                               (TS8 adapted = TS14) the four own files pure ASCII and LF; `git diff
                               <WAVE3A>` empty for the frozen files of amendment 2
    test_TS9_controls_only     amendment 8.1: only the active_fit twin has results (its real cell
                               unflagged): merge exits 0, no traceback; report/ and
                               report_without_flagged/ hold the long table and the FP summary, an
                               empty summary, "FAIL (axis absent)" on the gated axes; with
                               --drop-railed and every non-control row railed: exit 0, no summary rows
    test_TS10_base_seed        amendment 8.3: --base-seed recorded and compared at a re-plan; the same
                               twin under b1 != b2 has different noisy traces and identical noise-free
                               ones, under b1 twice identical ones; `twin` refuses an unknown
                               twin_config key (exit 3); a negative --base-seed refused (2)
    test_TS11_eval_theta       Block 17: per configuration, the arm, --tau-h-pk-bounds-ms (no
                               --kappa-bounds), --eval-theta only for full and full_no_noise with the
                               truth's five values exactly; run_ih_fit's parser accepts every argv and
                               gives the kappa box (LO / base, HI / base)
    test_TS12_no_noise_extras  Block 17, on TS3's run: the noise-free twin's archive has with_noise
                               False and equals its partner's noise-free traces (recomputed here), the
                               partner's differs; it is fitted; merge writes recovery_extras.csv with its
                               columns, loss_at_truth finite for full and full_no_noise and na verdicts
                               elsewhere, gh_deep_true_S_cm2, tau_h_pk_true_ms and the log ratio recomputed
                               here; loss_verdict on hand cases
    test_TS15_penalty_round_trip
                               amendment 3: loss_verdict is na when a loss is not finite or at or above
                               run_ih_fit's penalty 1e6, ordinary pairs unchanged; on TS3's run,
                               recovery_extras.csv's losses equal the fit's exactly (both round_trip)
    test_TS13_gate             Block 17: evaluate_gate with required_axes -- a summary without dv_h
                               passes or fails on Cm, gbar, kappa_tau alone; required_axes=None equals
                               the WAVE3A code's verdict tables; unknown axes refused; on TS5's run
                               report_no_noise/ holds exactly the noise-free twins, report/ and
                               report_without_flagged/ none of them, the gate on ih5 without a dv_h row

Run (from this folder: NEURON loads ./x86_64 from the working directory)
    python smoke_twin_recovery.py [--build]
        --build runs `nrnivmodl mod` first when ./x86_64/special is missing.
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table,
        then the full message of every failure. Exit code 1 if any check
        failed, errored or is still TODO.

Debug
    A failed check_close() saves both arrays to smoke_artifacts/<check name>.npz.

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import dataclasses
import importlib.metadata
import io
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                     # NEURON loads ./x86_64 from the CWD
sys.path.insert(0, str(HERE))

import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402
from numpy.testing import assert_allclose, assert_array_equal  # noqa: E402,F401

SEED = 12345
REPORT_PACKAGES = ("numpy", "scipy", "pandas", "scikit-optimize", "neuron")
#: TS7 runs every call of the submitter with this DISPLAY, which no X server answers (amendment 10):
#: NEURON exits on it ("can't open DISPLAY") unless the submitter unsets DISPLAY
NO_X_DISPLAY = "localhost:99.0"
ARTIFACT_DIR = HERE / "smoke_artifacts"
_SAVED_ARTIFACTS = []
#: the frozen check's reference (SPEC Block 17, amendment 2: the wave-3a merge) and its files: the old list
#: without ih_recovery_report.py (Block 17 owns it), with synth_gt_grid.py
CONTRACT_COMMIT = "06b5c46"
FROZEN = ("run_ih_recovery.py", "submit_ih_recovery.sh", "gen_from_manifest.py", "synth_gt_grid.py",
          "run_ih_fit.py")
OWN_FILES = ("run_twin_recovery.py", "submit_twin_recovery.sh", "smoke_twin_recovery.py", "ih_recovery_report.py")

# ---------------------------------------------------------------- fixture ---
GROUP = "FIX"
REAL_IDS = (500000001, 500000002, 500000003, 500000004)
FLAGGED = (500000001, 500000004)           # tau_s 300 ms > 265 ms (D-182.3)
SS_SWEEP, SS_DROPPED = 92, {92: (2, 3)}
LS_AMPS = (-110.0, -70.0, -30.0, 30.0)     # sweeps 54..57
SMALL = ["--n-per-configuration", "1", "--n-no-noise", "1", "--n-calls", "1", "--n-initial", "1"]
#: the smallest budget run_ih_fit accepts: its parser only asks n_initial <= n_calls, and gp_minimize
#: refuses n_initial_points = 0 without x0 (skopt), so 1 call with 1 initial point
NONDEFAULT_FIT = ["--gbar-bounds=2e-6,2e-3", "--vshift-base=-1.5", "--ehcn=-40.0", "--ih-regions", "soma,dend",
                  "--dvh-bounds=-8.0,8.0", "--F", "2.1", "--ss-tau-w-ms", "4.0", "--tau-h-pk-bounds-ms=60,900"]
IH6 = ("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau")
IH5 = ("Cm", "Rm", "Ra", "gbar", "kappa_tau")
PASSIVE = ("Cm", "Rm", "Ra")
CONFS = ("full", "passive_fit", "active_fit", "full_no_noise")                 # Block 16's order
#: the twin's model (2.7 (h), (i); Block 17: the six names as before, its truth's dv_h 0)
EXPECTED_NAMES = {"full": IH6, "passive_fit": IH6, "active_fit": PASSIVE, "full_no_noise": IH6}
EXPECTED_ARM = {"full": "ih5", "passive_fit": "passive_fullstep", "active_fit": "ih5", "full_no_noise": "ih5"}
FITTED = {"ih5": IH5, "passive_fullstep": PASSIVE}                             # the arms' names (Block 15)
WITH_EVAL = ("full", "full_no_noise")                                          # --eval-theta (Block 17)
TWIN_IDS = ["900000000", "900000100", "900000200", "900000300"]               # id_base + 100 q (+ 300 + i)
# the plan options the entry names that keep the name and default run_ih_recovery.py gives them
PLAN_OPTION_NAMES = ("seed", "id_base", "ra_mode", "cm_phys_lo", "cm_phys_hi", "tau_lo_ms", "tau_hi_ms",
                     "ra_phys_lo", "ra_phys_hi", "gbar_range", "gbar_floor", "F",
                     "fit_target", "ss_time_weight", "ss_tau_w_ms", "ss_window_ms", "n_calls", "n_initial",
                     "ih_mechanism", "ih_distribution", "ih_regions", "vshift_base", "ehcn", "dt_brief_ms",
                     "dt_long_ms", "phase3_subset", "dvh_bounds", "tol_cm_factor",
                     "tol_gbar_factor", "tol_dvh_mV", "tol_kappa_factor", "tol_fp_low_rail_frac", "tol_fp_cm_factor",
                     "lenient_rails", "drop_railed")
#: Block 17's new options and defaults; the options it removes
NEW_DEFAULTS = {"n_no_noise": 5, "gbar_bounds": "1e-6,1e-2", "tau_h_pk_bounds_ms": "50,1000",
                "tau_h_pk_truth_ms": "150,700"}
REMOVED_OPTIONS = ("--kappa-bounds=0.5,2.0", "--kappa-range=0.7,1.4", "--dvh-range=-5,5")

_STATE = {}


def _fixture():
    """The archive group and the two noise tables, written once."""
    if _STATE:
        return _STATE
    import smoke_noise_windows as snw
    import synthetic_ground_truth as sgt
    from run_noise_fit import CELL_COLUMNS, SWEEP_COLUMNS, write_csv_atomic
    root = Path(tempfile.mkdtemp(prefix="smoke_twin_recovery_"))
    atexit.register(shutil.rmtree, str(root), True)
    archive_root = root / "Human Neurons Fitting"
    group_dir = archive_root / GROUP
    rng = np.random.default_rng(SEED)
    for sid in REAL_IDS:
        d = group_dir / ("specimen_%d" % sid)
        snw.write_fixture(d, rng, with_response=True, dropped=SS_DROPPED, sweeps=(SS_SWEEP,), ls_amps=LS_AMPS)
        meta = json.loads((d / "metadata.json").read_text())
        meta["specimen_id"] = sid
        (d / "metadata.json").write_text(json.dumps(meta, indent=1))
        sgt.write_ball_and_stick_swc(d / "reconstruction.swc")
    output_base = archive_root / "pipeline_outputs_ih"
    noise_dir = output_base / "noise_fit" / "nfx" / GROUP
    noise_dir.mkdir(parents=True)
    cells, sweeps = [], []
    for k, sid in enumerate(REAL_IDS):
        r = {c: None for c in CELL_COLUMNS}
        r.update(specimen_id=sid, status="ok", reason="", noise_sigma_f_mV=0.030 + 1e-3 * k,
                 noise_sigma_m_mV=0.017, noise_tau_m_ms=0.07, noise_sigma_s_mV=0.08 + 1e-3 * k,
                 noise_tau_s_ms=(300.0 if sid in FLAGGED else 150.0), noise_flags="", noise_code_commit="0" * 40)
        cells.append(r)
        for pr, s in [("SS", SS_SWEEP)] + [("LS", 54 + j) for j in range(len(LS_AMPS))]:
            r = {c: None for c in SWEEP_COLUMNS}
            r.update(specimen_id=sid, noise_sweep_protocol=pr, noise_sweep_number=s,
                     noise_tilt_mV_per_s=0.01 * (s - 60), noise_sweep_sigma_f_mV=0.031 + 1e-4 * s,
                     noise_sweep_sigma_m_mV=0.018, noise_sweep_sigma_s_mV=0.081 + 1e-4 * s, noise_sweep_flags="")
            sweeps.append(r)
    write_csv_atomic(noise_dir / ("noise_cells_%s.csv" % GROUP), CELL_COLUMNS, cells)
    write_csv_atomic(noise_dir / ("noise_sweeps_%s.csv" % GROUP), SWEEP_COLUMNS, sweeps)
    _STATE.update(root=root, archive_root=archive_root, group_dir=group_dir, output_base=output_base,
                  noise_dir=noise_dir, n_runs=0)
    return _STATE


def _new_out(tag):
    st = _fixture()
    st["n_runs"] += 1
    return st["root"] / "runs" / ("%s_%d" % (tag, st["n_runs"]))


def _run(argv):
    """run_twin_recovery.main in process: (exit code, stdout, stderr)."""
    import run_twin_recovery as T
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = T.main([str(a) for a in argv])
    return code, out.getvalue(), err.getvalue()


def _plan(out, *extra):
    st = _fixture()
    return _run(["plan", "--group-dir", st["group_dir"], "--noise-dir", st["noise_dir"], "--out", out]
                + SMALL + list(extra))


def _sub(args, timeout=1800):
    """run_twin_recovery.py as a subprocess (one process per twin, as on the cluster)."""
    return subprocess.run([sys.executable, str(HERE / "run_twin_recovery.py")] + [str(a) for a in args],
                          cwd=str(HERE), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                          timeout=timeout)


def _manifest(out):
    import twin_manifest as TM
    return TM.load_twin_manifest(Path(out) / "manifest.csv")


def _clear():
    import synthetic_ground_truth as sgt
    sgt._clear_neuron_sections()


def _mtimes(paths):
    return {str(p): os.stat(p).st_mtime_ns for p in paths}


def _rir_defaults(**over):
    import run_ih_recovery as RIR
    ns = RIR._parse_args(["--output-dir", "-", "--code-dir", "-", "--morph-root", "-"])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _tau_base_ms(options):
    """tau_h^{base,pk} (ms) of the run's I_h spec, rebuilt here: run_ih_fit's parser on the I_h options
    (not run_twin_recovery's helpers), then run_ih_fit.tau_h_base_peak_ms (Block 15)."""
    import run_ih_fit as R
    argv = ["--archive-dir", "-", "--output-dir", "-", "--code-dir", "-", "--ih-mechanism", options["ih_mechanism"],
            "--ih-distribution", options["ih_distribution"], "--ih-regions", options["ih_regions"],
            "--vshift-base=%s" % options["vshift_base"]]
    if options.get("ehcn") not in (None, ""):
        argv.append("--ehcn=%s" % options["ehcn"])
    return float(R.tau_h_base_peak_ms(R.build_ih_spec(R._parse_args(argv))))


def _ms_pair(text):
    lo, hi = (float(x) for x in str(text).split(","))
    return lo, hi


def check_close(name, actual, expected, rtol=0.0, atol=0.0):
    """assert_allclose that keeps the evidence (smoke_artifacts/<name>.npz) on failure."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    try:
        assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
    except AssertionError:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / ("%s.npz" % re.sub(r"[^A-Za-z0-9_.-]+", "_", name))
        np.savez(path, actual=actual, expected=expected)
        _SAVED_ARTIFACTS.append(path)
        raise


def _fail_parts(parts):
    bad = [k for k, v in parts.items() if not v]
    assert not bad, "failed parts: " + "; ".join(bad)


# ---------------------------------------------------------------- checks ---
def test_TS1_plan():
    import passive_fitting_hpc_fixed as mono
    import run_ih_recovery as RIR
    import run_twin_recovery as T
    import synth_gt_grid as G
    import twin_config
    import twin_manifest as TM
    import twin_table as TT
    st = _fixture()
    out = _new_out("ts1")
    code, so, se = _plan(out)
    assert code == 0, "plan exit %d: %s" % (code, se)
    files = [out / ("real_cells_%s.csv" % GROUP), out / ("twin_sweeps_%s.csv" % GROUP), out / "manifest.csv",
             out / "run_config.json"]
    parts = {"the four files are written": all(p.is_file() for p in files)}

    # the plan's options: the entry's names, run_ih_recovery's defaults
    pa = T.build_parser().parse_args(["plan", "--group-dir", "g", "--noise-dir", "n", "--out", "o"])
    rd = vars(_rir_defaults())
    parts["plan's options carry run_ih_recovery's names and defaults"] = all(
        hasattr(pa, k) and getattr(pa, k) == rd[k] for k in PLAN_OPTION_NAMES) and pa.n_per_configuration == 10
    parts["Block 17's options and defaults (n_no_noise 5, gbar 1e-6,1e-2, tau_h_pk 50,1000 and 150,700)"] = all(
        getattr(pa, k, None) == v for k, v in NEW_DEFAULTS.items())
    gone = []
    for opt in REMOVED_OPTIONS:
        with contextlib.redirect_stderr(io.StringIO()):
            c = T.main(["plan", "--group-dir", "g", "--noise-dir", "n", "--out", str(_new_out("ts1x")), opt])
        gone.append(c == 2)
    parts["--kappa-bounds, --kappa-range, --dvh-range are no options of plan (exit 2)"] = all(gone) and not any(
        hasattr(pa, k) for k in ("kappa_bounds", "kappa_range", "dvh_range"))

    # real_cells: v_rest = the loader's; the swc; the flag
    rc = pd.read_csv(files[0])
    v_ok, swc_ok = True, True
    for _, r in rc.iterrows():
        d = st["group_dir"] / ("specimen_%d" % int(r["specimen_id"]))
        v_ok &= float(r["v_rest_mV"]) == float(mono.load_cell_from_archive(d, verbose=False).v_rest_mV)
        swc_ok &= str(r["swc_path"]) == str(d / "reconstruction.swc")
    parts["real_cells: the cohort, v_rest_mV = the loader's, swc_path"] = (
        list(rc["specimen_id"]) == list(REAL_IDS) and v_ok and swc_ok
        and list(rc["cell_tau_s_above_window"].astype(bool)) == [s in FLAGGED for s in REAL_IDS]
        and list(rc["n_ls_in_twin"]) == [4] * 4 and list(rc["n_ls_left_out"]) == [0] * 4)

    # the twins' table: Block 5's, built here
    cells_df, sweeps_df = TT.read_noise_tables(st["noise_dir"] / ("noise_cells_%s.csv" % GROUP),
                                               st["noise_dir"] / ("noise_sweeps_%s.csv" % GROUP))
    prots = [TT.load_cell_protocol(st["group_dir"] / ("specimen_%d" % s), spike_reason_fn=mono.sweep_spike_reason)
             for s in REAL_IDS]
    ref_table = TT.build_twin_sweep_table(prots, cells_df, sweeps_df, twin_config.TwinConfig())
    try:
        pd.testing.assert_frame_equal(TT.read_twin_sweep_table(files[1]), ref_table.reset_index(drop=True))
        parts["twin_sweeps = Block 5's table built here"] = True
    except AssertionError as exc:
        parts["twin_sweeps = Block 5's table built here (%s)" % str(exc).splitlines()[0]] = False

    # the manifest: draw_twin_manifest called here; truth kwargs recorded from run_ih_recovery.phase_manifest
    recorded = {}
    orig = G.draw_manifest

    class _Stop(Exception):
        pass

    def recorder(swcs, **kw):
        recorded.update(kw)
        raise _Stop()
    G.draw_manifest = recorder
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            RIR.phase_manifest(_rir_defaults(morph_root=str(st["group_dir"]), output_dir=str(out)), out, None)
    except _Stop:
        pass
    finally:
        G.draw_manifest = orig
    dropped = ("draws_per_morph", "seed", "cells_per_cohort", "e_pas_mV", "max_cells", "use_ih", "fp_control_frac",
               "id_base", "noise_table")
    ref_kwargs = {k: v for k, v in recorded.items() if k not in dropped and not k.startswith("noise_")}
    # Block 17: no dv_h range (dv_h's truth 0); the kappa range = tau_h_pk's truth range (150, 700 ms) / base
    cfg1 = json.loads(files[3].read_text())
    base = _tau_base_ms(cfg1["options"])
    ref_kwargs["ih_dvh_range_mV"] = None
    ref_kwargs["ih_kappa_range"] = (150.0 / base, 700.0 / base)
    real_cells = rc.copy()
    real_cells["v_rest_mV"] = real_cells["v_rest_mV"].astype(np.float64)
    real_cells["cell_tau_s_above_window"] = real_cells["cell_tau_s_above_window"].astype(bool)
    with contextlib.redirect_stdout(io.StringIO()):
        ref_man = TM.draw_twin_manifest(real_cells[list(TM.REAL_CELL_COLUMNS)], n_per_configuration=1, n_no_noise=1,
                                        seed=0, id_base=900_000_000, dvh_truth_mV=0.0, truth_kwargs=ref_kwargs)
    try:
        pd.testing.assert_frame_equal(_manifest(out), ref_man)
        parts["manifest = draw_twin_manifest here (truth kwargs of phase_manifest, cells_per_cohort %r)"
              % recorded.get("cells_per_cohort")] = recorded.get("cells_per_cohort") == 10
    except AssertionError as exc:
        parts["manifest = draw_twin_manifest here (%s)" % str(exc).splitlines()[0]] = False
    m1 = _manifest(out)
    ih_rows = m1["configuration"].isin(["full", "passive_fit", "full_no_noise"])
    parts["the manifest: four configurations in order, ids, with_noise, the pairing, dv_h truth 0"] = (
        list(m1["configuration"]) == list(CONFS) and [str(i) for i in m1["specimen_id"]] == TWIN_IDS
        and list(m1["with_noise"]) == [True, True, True, False]
        and int(m1["paired_with"].iloc[3]) == int(m1["specimen_id"].iloc[0])
        and bool((m1.loc[ih_rows, "ih_dvh_mV"] == 0.0).all()))
    rec = cfg1.get("tau_h_pk") or {}
    kap = m1.loc[m1["configuration"].isin(["full", "passive_fit"]), "ih_kappa_tau"].to_numpy(dtype=float)
    parts["run_config.json: tau_h_pk in ms and as kappa (base recomputed), gate arm ih5, its required axes; the "
          "truths' kappa inside the truth range"] = (
        rec.get("base_ms") == base and rec.get("bounds_ms") == [50.0, 1000.0]
        and rec.get("bounds_kappa") == [50.0 / base, 1000.0 / base] and rec.get("truth_ms") == [150.0, 700.0]
        and rec.get("truth_kappa") == [150.0 / base, 700.0 / base] and cfg1.get("gate_arm") == "ih5"
        and cfg1.get("gate_required_axes") == ["Cm", "gbar", "kappa_tau"]
        and cfg1["truth_kwargs"]["ih_kappa_range"] == [150.0 / base, 700.0 / base]
        and cfg1["truth_kwargs"]["ih_dvh_range_mV"] is None
        and bool(np.all((kap >= 150.0 / base) & (kap <= 700.0 / base)))
        and 340.0 <= base <= 346.0)                       # Block 15's I2 range for Ih_human at base shift 0

    # a second identical plan changes nothing; another seed refused; --force re-plans
    before = _mtimes(files)
    time.sleep(0.05)
    code2, so2, _ = _plan(out)
    parts["a second identical plan: exit 0, no file changed"] = code2 == 0 and _mtimes(files) == before
    code3, _, se3 = _plan(out, "--seed", "1")
    parts["another seed: exit 3, nothing changed"] = code3 == 3 and _mtimes(files) == before and "seed" in se3
    code4, _, _ = _plan(out, "--seed", "1", "--force")
    parts["--force with another seed: exit 0, run_config's seed 1"] = (
        code4 == 0 and json.loads(files[3].read_text())["options"]["seed"] == 1)

    # refusals: a cohort smaller than n; a cell refused by Block 5; list without run_config
    code5, _, se5 = _run(["plan", "--group-dir", st["group_dir"], "--noise-dir", st["noise_dir"], "--out",
                          _new_out("ts1r"), "--n-per-configuration", "5"])
    code6, _, se6 = _run(["plan", "--group-dir", st["group_dir"], "--noise-dir", st["noise_dir"], "--out",
                          _new_out("ts1r"), "--n-per-configuration", "1", "--cells", "500000001,123456789"])
    code7, _, _ = _run(["list", "--out", _new_out("ts1r")])
    parts["refused: cohort of 4 < 5 (3), an unknown cell named (3), list without run_config (2)"] = (
        code5 == 3 and "fewer" in se5 and code6 == 3 and "123456789" in se6 and code7 == 2)
    # versions(): a NEURON without pip metadata (davinci's env) -> its module's __version__ (amendment 10.2)
    import importlib.metadata as im
    import neuron
    real_version = im.version

    def no_neuron_metadata(name):
        if name == "neuron":
            raise im.PackageNotFoundError(name)
        return real_version(name)
    im.version = no_neuron_metadata
    try:
        v = T.versions()
    finally:
        im.version = real_version
    parts["versions(): NEURON without pip metadata -> its module's __version__ (amendment 10.2)"] = (
        v.get("neuron") == neuron.__version__ and v.get("numpy") == real_version("numpy"))
    print("\n  [TS1] run_config: %s" % ", ".join("%s=%r" % (k, json.loads(files[3].read_text()).get(k))
                                              for k in ("code_commit", "code_dirty", "versions")))
    _fail_parts(parts)


def _recording_run(tag, extra):
    """A planned run whose three twins go through `twin` with run_ih_fit.main replaced by a recorder and
    twin_build.build_twin wrapped (called through). Memoised per tag."""
    key = "rec_" + tag
    if key in _STATE:
        return _STATE[key]
    import run_ih_fit as R
    import twin_build as TB
    _fixture()
    out = _new_out(tag)
    code, _, se = _plan(out, *extra)
    assert code == 0, se
    fit_calls, build_calls = [], []
    orig_main, orig_build = R.main, TB.build_twin

    def main_recorder(argv=None):
        fit_calls.append(list(argv))

    def build_recorder(spec, real_spec_dir, table, group_dir, **kw):
        build_calls.append(dict(spec=spec, real_spec_dir=real_spec_dir, group_dir=group_dir, **kw))
        return orig_build(spec, real_spec_dir, table, group_dir, **kw)
    R.main, TB.build_twin = main_recorder, build_recorder
    codes = {}
    try:
        for tid in _manifest(out)["specimen_id"]:
            codes[int(tid)] = _run(["twin", "--out", out, "--twin", int(tid)])
            _clear()
    finally:
        R.main, TB.build_twin = orig_main, orig_build
    _STATE[key] = dict(out=out, fit_calls=fit_calls, build_calls=build_calls, codes=codes)
    return _STATE[key]


def _phase_fit_argv(options, archive, arm_name):
    """The argv run_ih_recovery.phase_fit builds from these values (its run_ih_fit.main recorded); the
    run's options no longer hold kappa_bounds (Block 17), so a placeholder is given and removed by
    _ref_fit_argv."""
    import run_ih_fit as R
    import run_ih_recovery as RIR
    got = []
    orig = R.main
    R.main = lambda argv=None: got.append(list(argv))
    try:
        ns = argparse.Namespace(**options)
        ns.code_dir, ns.overwrite, ns.fail_fast = str(HERE), True, False
        ns.kappa_bounds = "KAPPA_PLACEHOLDER"
        with contextlib.redirect_stdout(io.StringIO()):
            RIR.phase_fit(ns, Path(archive), _fixture()["root"] / "phase_fit_ref", [arm_name])
    finally:
        R.main = orig
    assert len(got) == 1, got
    return got[0]


def _ref_fit_argv(options, archive, fit_dir, row):
    """The twin's expected argv (Block 17): phase_fit's for the arm (ih6 standing in for ih5, which
    phase_fit does not know), --arm set to the configuration's arm, its own output directory,
    --kappa-bounds replaced in place by --tau-h-pk-bounds-ms, and --eval-theta with the truth's five
    values (repr) for full and full_no_noise."""
    import twin_manifest as TM
    conf = str(row["configuration"])
    arm = EXPECTED_ARM[conf]
    ref = _phase_fit_argv(options, archive, "ih6" if arm == "ih5" else arm)
    ref[ref.index("--arm") + 1] = arm
    ref[ref.index("--output-dir") + 1] = str(fit_dir)   # phase_fit's own <out>/arms/<arm> -> the twin's
    i = ref.index("--kappa-bounds=KAPPA_PLACEHOLDER")
    ref[i] = "--tau-h-pk-bounds-ms=%s" % options["tau_h_pk_bounds_ms"]
    if conf in WITH_EVAL:
        th = TM.twin_theta(row)
        ref.append("--eval-theta=" + ",".join("%s=%r" % (n, th[n]) for n in IH5))
    return ref


def test_TS2_fit_command():
    rec = _recording_run("ts2", NONDEFAULT_FIT)
    out = rec["out"]
    cfg = json.loads((out / "run_config.json").read_text())
    man = _manifest(out)
    parts = {"one fit call per twin": len(rec["fit_calls"]) == len(man) == 4}
    for (_, row), argv in zip(man.iterrows(), rec["fit_calls"]):
        tid, conf = int(row["specimen_id"]), str(row["configuration"])
        arm = EXPECTED_ARM[conf]
        twin_group = out / "twins" / ("twin_%d" % tid)
        fit_dir = out / "fits" / ("twin_%d" % tid)
        ref = _ref_fit_argv(cfg["options"], twin_group, fit_dir, row)
        parts["%s: argv = phase_fit's with Block 17's changes (arm %s, own directories)" % (conf, arm)] = (
            argv == ref and argv[argv.index("--arm") + 1] == arm
            and argv[argv.index("--archive-dir") + 1] == str(twin_group))
    neg = rec["fit_calls"][0] if rec["fit_calls"] else []
    parts["negative values in the '=' form"] = ("--vshift-base=-1.5" in neg and "--ehcn=-40.0" in neg
                                                and "--dvh-bounds=-8.0,8.0" in neg)
    print("\n  [TS2] full twin's argv: %s" % " ".join(neg))
    _fail_parts(parts)


def test_TS3_end_to_end():
    import passive_fitting_hpc_fixed as mono
    import twin_build as TB
    out = _new_out("ts3")
    code, _, se = _plan(out)
    assert code == 0, se
    man = _manifest(out)
    parts, notes = {}, []
    has_ih = {}
    for _, row in man.iterrows():
        tid, conf = int(row["specimen_id"]), str(row["configuration"])
        t0 = time.time()
        if conf == "active_fit":                 # in process: the cell is inspected
            orig = TB.build_twin_cell

            def spy(*a, **kw):
                cell = orig(*a, **kw)
                from neuron import h
                has_ih[tid] = sorted({m for s in h.allsec() for m in ("Ih_human", "Ih") if s.has_membrane(m)})
                return cell
            TB.build_twin_cell = spy
            try:
                rc, so, se = _run(["twin", "--out", out, "--twin", tid])
            finally:
                TB.build_twin_cell = orig
                _clear()
        else:
            p = _sub(["twin", "--out", out, "--twin", tid])
            rc, so, se = p.returncode, p.stdout, p.stderr
        secs = time.time() - t0
        status = json.loads((out / "status" / ("twin_%d.json" % tid)).read_text())
        cd = mono.load_cell_from_archive(out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid), verbose=False)
        res = pd.read_csv(out / "fits" / ("twin_%d" % tid) / "phase2_results.csv")
        r = res[res["specimen_id"] == tid]
        names = str(r["param_names"].iloc[0]).split("|") if len(r) else []
        est = {n: float(r[n].iloc[0]) for n in names if n in r.columns}
        parts["%s: exit 0, status ok, the loader reads the archive, phase2_results.csv holds its row"
              % conf] = (rc == 0 and status["status"] == "ok" and int(cd.specimen_id) == tid
                         and float(cd.v_rest_mV) == float(row["v_rest_mV"]) and len(r) == 1)
        parts["%s: the fit's parameters are the arm's (%s)" % (conf, EXPECTED_ARM[conf])] = (
            tuple(names) == FITTED[EXPECTED_ARM[conf]])
        notes.append("%s twin %d: %.1f s (build %.1f s, fit %.1f s); budget n_calls=%s n_initial=%s; returned %s; "
                     "validation_status %s" % (conf, tid, secs, status["build_seconds"], status["fit_seconds"],
                                               r["n_calls"].iloc[0] if len(r) else "?",
                                               r["n_initial"].iloc[0] if len(r) else "?",
                                               ", ".join("%s=%.4g" % kv for kv in est.items()),
                                               r["validation_status"].iloc[0] if len(r) else "?"))
    tid_a = int(man[man["configuration"] == "active_fit"]["specimen_id"].iloc[0])
    tid_f = int(man[man["configuration"] == "full"]["specimen_id"].iloc[0])
    meta_a = json.loads((out / "twins" / ("twin_%d" % tid_a) / ("specimen_%d" % tid_a) / "metadata.json").read_text())
    parts["active_fit: no I_h mechanism in any section of its cell, has_ih False (five parameters fitted: above)"] = (
        has_ih.get(tid_a) == [] and meta_a["twin"]["has_ih"] is False)
    meta_f = json.loads((out / "twins" / ("twin_%d" % tid_f) / ("specimen_%d" % tid_f) / "metadata.json").read_text())
    parts["full: has_ih True"] = meta_f["twin"]["has_ih"] is True
    print("\n  [TS3] " + "\n  [TS3] ".join(notes))
    _STATE["ts3_out"] = out
    _fail_parts(parts)


def test_TS4_model_identity():
    import ih_mechanism as IM
    import run_ih_fit as R
    import twin_manifest as TM
    rec = _recording_run("ts2", NONDEFAULT_FIT)
    out = rec["out"]
    man = _manifest(out)
    cfg = json.loads((out / "run_config.json").read_text())
    parts = {"one build and one fit per twin, every twin ok up to the (recorded) fit": (
        len(rec["build_calls"]) == 4 and len(rec["fit_calls"]) == 4
        and all(json.loads((out / "status" / ("twin_%d.json" % int(t))).read_text())["built"] is True
                for t in man["specimen_id"]))}
    for (_, row), call in zip(man.iterrows(), rec["build_calls"]):
        conf, tid = str(row["configuration"]), int(row["specimen_id"])
        # the options the fit receives, rebuilt here by run_ih_recovery.phase_fit (not read from the recorder)
        fa = R._parse_args(_ref_fit_argv(cfg["options"], out / "twins" / ("twin_%d" % tid),
                                         out / "fits" / ("twin_%d" % tid), row))
        ref_ih = R.build_ih_spec(fa)
        base = _tau_base_ms(cfg["options"])
        lo_ms, hi_ms = _ms_pair(cfg["options"]["tau_h_pk_bounds_ms"])
        ref_ps = R.build_param_spec(EXPECTED_NAMES[conf], cm_bounds=R.parse_window(fa.cm_bounds),
                                    rm_bounds=R.parse_window(fa.rm_bounds), ra_bounds=R.parse_window(fa.ra_bounds),
                                    gbar_bounds=R.parse_window(fa.gbar_bounds),
                                    dvh_bounds=R.parse_window(fa.dvh_bounds),
                                    kappa_bounds=(lo_ms / base, hi_ms / base))   # the fit's box, 2.8 (a)
        got_ih, apply = call["ih_spec"], call["apply_theta"]
        got_ps = getattr(apply, "__self__", None)
        diff = [f.name for f in dataclasses.fields(IM.IhSpec)
                if getattr(got_ih, f.name) != getattr(ref_ih, f.name)]
        parts["%s: IhSpec equal option by option%s" % (conf, (" (differ: %s)" % diff) if diff else "")] = (
            isinstance(got_ih, IM.IhSpec) and not diff)
        parts["%s: ParamSpec equal (names, axes, apply rule)" % conf] = (
            got_ps is not None and got_ps.names == EXPECTED_NAMES[conf] and got_ps.axes == ref_ps.axes
            and got_ps._apply_fn is ref_ps._apply_fn and getattr(apply, "__func__", None) is type(ref_ps).apply)
        spec = call["spec"]
        ref_spec = TM.twin_spec_of(row, F=float(cfg["options"]["F"]))
        parts["%s: TwinSpec = twin_spec_of(row, F=the fit's --F)" % conf] = (
            spec == ref_spec and spec.F == 2.1 and call["real_spec_dir"]
            == Path(cfg["group_dir"]) / ("specimen_%d" % int(row["twin_of_specimen_id"])))
    full_ih = rec["build_calls"][0]["ih_spec"] if rec["build_calls"] else None
    print("\n  [TS4] full twin's IhSpec: %r" % (full_ih,))
    _fail_parts(parts)


def _write_results(out, man, factors):
    """phase2_results.csv per twin, estimates = truth x factor (log axes) or truth + 1 mV (dv_h)."""
    truth_col = {"Cm": "cm_true", "Rm": "rm_true", "Ra": "ra_true", "gbar": "ih_gihbar_S_cm2", "dv_h": "ih_dvh_mV",
                 "kappa_tau": "ih_kappa_tau"}
    for _, row in man.iterrows():
        tid, conf = int(row["specimen_id"]), str(row["configuration"])
        names = FITTED[EXPECTED_ARM[conf]]
        rec = {"specimen_id": tid, "param_names": "|".join(names), "validation_status": "good",
               "arm": EXPECTED_ARM[conf]}
        for n in names:
            t = float(row[truth_col[n]])
            rec[n] = t + 1.0 if n == "dv_h" else t * factors[tid]
        if conf == "active_fit":
            rec["gbar"] = 1e-6                    # the low rail of the default box: the FP control passes it
        d = out / "fits" / ("twin_%d" % tid)
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([rec]).to_csv(d / "phase2_results.csv", index=False)


def test_TS5_merge():
    import ih_recovery_report as RR
    out = _new_out("ts5")
    code, _, se = _plan(out)
    assert code == 0, se
    man = _manifest(out)
    flags = {int(t): bool(f) for t, f in zip(man["specimen_id"], man["cell_tau_s_above_window"])}
    assert any(flags.values()) and not all(flags.values()), "the fixture's draw must hold flagged and unflagged twins"
    factors = {int(t): 1.05 + 0.1 * i for i, t in enumerate(man["specimen_id"])}
    _write_results(out, man, factors)
    tid0 = int(man["specimen_id"].iloc[0])
    status0 = dict(twin_id=tid0, twin_of_specimen_id=int(man["twin_of_specimen_id"].iloc[0]), configuration="full",
                   arm="ih6", status="ok", reason="", build_seconds=1.5, fit_seconds=2.5, code_commit="abc")
    (out / "status").mkdir()
    (out / "status" / ("twin_%d.json" % tid0)).write_text(json.dumps(status0))
    code, so, se = _run(["merge", "--out", out, "--no-plot"])
    parts = {"merge exit 0": code == 0}

    # the reference: ih_recovery_report called here, one frame per arm, as run_ih_recovery.phase_report does;
    # Block 17: the gate on ih5 with required axes Cm, gbar, kappa_tau; the kappa box = tau_h_pk's / base
    opts = json.loads((out / "run_config.json").read_text())["options"]
    box = dict(RR.DEFAULT_BOX)
    for axis, text in (("gbar", opts["gbar_bounds"]), ("dv_h", opts["dvh_bounds"])):
        box[axis] = tuple(float(x) for x in text.split(","))
    base = _tau_base_ms(opts)
    lo_ms, hi_ms = _ms_pair(opts["tau_h_pk_bounds_ms"])
    box["kappa_tau"] = (lo_ms / base, hi_ms / base)
    tol = RR.RecoveryTolerances(log_ratio_cm=float(np.log(1.25)), log_ratio_gbar=float(np.log(1.25)),
                                abs_dvh_mV=3.0, log_ratio_kappa=float(np.log(1.5)), fp_min_low_rail_frac=0.8,
                                fp_log_ratio_cm=float(np.log(1.25)))

    def reference(sub_man, dest):
        frames = []
        for arm in ("ih5", "passive_fullstep"):
            ids = [int(t) for t, c in zip(sub_man["specimen_id"], sub_man["configuration"]) if EXPECTED_ARM[c] == arm]
            if ids:
                # amendment 3: the merge reads each phase2_results.csv with round_trip floats
                frames.append(pd.concat([pd.read_csv(out / "fits" / ("twin_%d" % t) / "phase2_results.csv",
                                                     float_precision="round_trip").assign(arm=arm)
                                         for t in ids], ignore_index=True))
        long_df = RR.recovery_long(sub_man, frames, box=box, strict_rails=True)
        summ = RR.recovery_summary(long_df, drop_railed=False)
        infl, fp = RR.cm_inflation(long_df), RR.fp_control_summary(long_df)
        verdict, passed = RR.evaluate_gate(summ, fp, arm="ih5", tol=tol, required_axes=("Cm", "gbar", "kappa_tau"))
        RR.write_report(dest, long_df=long_df, summary=summ, inflation=infl, fp=fp, verdict=verdict, passed=passed,
                        tol=tol)
        return long_df

    names = ("recovery_per_cell.csv", "recovery_summary.csv", "cm_inflation.csv", "fp_control.csv",
             "gate_verdict.csv", "gate_verdict.txt")

    def same(a, b):
        for n in names:
            try:
                ta, tb = (a / n).read_text(), (b / n).read_text()
            except OSError:
                return False
            if ta == tb:                       # byte-identical (also an empty table, written without columns)
                continue
            if n.endswith(".txt") or not ta.strip() or not tb.strip():
                return False
            try:
                pd.testing.assert_frame_equal(pd.read_csv(a / n), pd.read_csv(b / n))
            except AssertionError:
                return False
        return True
    noisy = man[man["with_noise"].astype(bool)].reset_index(drop=True)
    quiet = man[~man["with_noise"].astype(bool)].reset_index(drop=True)
    ref_dir = _new_out("ts5ref")
    ref_long = reference(noisy, ref_dir)
    parts["report/ = ih_recovery_report on the same frames of the noisy twins (six files)"] = same(out / "report",
                                                                                                   ref_dir)
    ref_dir_0 = _new_out("ts5ref0")
    ref_long_0 = reference(quiet, ref_dir_0)
    parts["report_no_noise/ = ih_recovery_report on the noise-free twins alone (six files)"] = same(
        out / "report_no_noise", ref_dir_0)
    unf = noisy[~noisy["specimen_id"].map(flags)].reset_index(drop=True)
    ref_dir_u = _new_out("ts5refu")
    reference(unf, ref_dir_u)
    per_u = pd.read_csv(out / "report_without_flagged" / "recovery_per_cell.csv")
    parts["report_without_flagged/ = the reference on the unflagged twins; exactly the unflagged ids"] = (
        same(out / "report_without_flagged", ref_dir_u)
        and set(per_u["specimen_id"]) == {int(t) for t in unf["specimen_id"]}
        and set(per_u["specimen_id"]) == {t for t, f in flags.items() if not f} - set(int(t) for t in quiet["specimen_id"]))
    byc = pd.read_csv(out / "recovery_by_configuration.csv")
    conf_of = {int(t): c for t, c in zip(man["specimen_id"], man["configuration"])}
    parts["recovery_by_configuration: the long table, each row's configuration, real cell and flag"] = (
        len(byc) == len(ref_long) + len(ref_long_0) and set(byc["configuration"]) == set(CONFS)
        and all(conf_of[int(t)] == c for t, c in zip(byc["specimen_id"], byc["configuration"]))
        and all(EXPECTED_ARM[c] == a for a, c in zip(byc["arm"], byc["configuration"]))
        and all(flags[int(t)] == bool(f) for t, f in zip(byc["specimen_id"], byc["cell_tau_s_above_window"])))
    ts = pd.read_csv(out / "twins_status.csv")
    st_of = dict(zip(ts["twin_id"].astype(int), ts["status"]))
    parts["twins_status: one row per twin, the status file's fields, else missing"] = (
        list(ts["twin_id"]) == list(man["specimen_id"]) and st_of[tid0] == "ok"
        and float(ts.loc[ts["twin_id"] == tid0, "fit_seconds"].iloc[0]) == 2.5
        and all(st_of[int(t)] == "missing" for t in man["specimen_id"] if int(t) != tid0))

    # a twin without results: merge still 0, its rows gone; no results at all: 1
    tid_x = int(man["specimen_id"].iloc[1])
    (out / "fits" / ("twin_%d" % tid_x) / "phase2_results.csv").unlink()
    code2, _, _ = _run(["merge", "--out", out, "--no-plot"])
    per2 = pd.read_csv(out / "report" / "recovery_per_cell.csv")
    parts["one twin without results: exit 0, none of its rows"] = code2 == 0 and tid_x not in set(per2["specimen_id"])
    for t in man["specimen_id"]:
        p = out / "fits" / ("twin_%d" % int(t)) / "phase2_results.csv"
        if p.exists():
            p.unlink()
    code3, _, se3 = _run(["merge", "--out", out, "--no-plot"])
    code4, _, _ = _run(["merge", "--out", _new_out("ts5none")])
    parts["no results at all: exit 1; no run_config.json: exit 2"] = code3 == 1 and code4 == 2
    print("\n  [TS5] flags %s; gate line: %s" % (flags, [ln for ln in so.splitlines() if "VERDICT" in ln][:2]))
    _STATE["ts5_out"] = out
    _fail_parts(parts)


def test_TS6_resume_failure():
    out = _STATE.get("ts3_out")
    if out is None:
        test_TS3_end_to_end()
        out = _STATE["ts3_out"]
    man = _manifest(out)
    tid = int(man[man["configuration"] == "passive_fit"]["specimen_id"].iloc[0])
    meta = out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid) / "metadata.json"
    csv = out / "fits" / ("twin_%d" % tid) / "phase2_results.csv"
    before = _mtimes([meta, csv])
    time.sleep(0.05)
    p = _sub(["twin", "--out", out, "--twin", tid])
    st1 = json.loads((out / "status" / ("twin_%d.json" % tid)).read_text())
    parts = {"second call: exit 0, ok, neither rebuilt nor refitted (mtimes unchanged)": (
        p.returncode == 0 and _mtimes([meta, csv]) == before and st1["status"] == "ok"
        and st1["built"] is False and st1["fitted"] is False)}
    t0 = time.time()
    p2 = _sub(["twin", "--out", out, "--twin", tid, "--force"])
    after = _mtimes([meta, csv])
    st2 = json.loads((out / "status" / ("twin_%d.json" % tid)).read_text())
    parts["--force: exit 0, rebuilt and refitted (both mtimes changed)"] = (
        p2.returncode == 0 and after[str(meta)] != before[str(meta)] and after[str(csv)] != before[str(csv)]
        and st2["built"] is True and st2["fitted"] is True)
    # Block 17: a recorded archive is reused only if its with_noise equals the manifest's
    tid0 = int(man[man["configuration"] == "full_no_noise"]["specimen_id"].iloc[0])
    meta0 = out / "twins" / ("twin_%d" % tid0) / ("specimen_%d" % tid0) / "metadata.json"
    saved = meta0.read_bytes()
    m0 = json.loads(saved)
    m0["twin"]["with_noise"] = True
    meta0.write_text(json.dumps(m0, indent=1))
    try:
        cw, _, _ = _run(["twin", "--out", out, "--twin", tid0])
        stw = json.loads((out / "status" / ("twin_%d.json" % tid0)).read_text())
    finally:
        meta0.write_bytes(saved)
    cr, _, _ = _run(["twin", "--out", out, "--twin", tid0])
    _clear()
    str_ = json.loads((out / "status" / ("twin_%d.json" % tid0)).read_text())
    parts["an archive whose with_noise differs from the manifest's: build_failed (exit 1), not reused; restored: "
          "ok, neither rebuilt nor refitted"] = (
        cw == 1 and stw["status"] == "build_failed" and "with_noise" in stw["reason"] and cr == 0
        and str_["status"] == "ok" and str_["built"] is False and str_["fitted"] is False)
    # a twin whose real cell directory is missing: build_failed, exit 1, a status file
    st = _fixture()
    group_copy = st["root"] / "group_copy" / GROUP
    shutil.copytree(str(st["group_dir"]), str(group_copy))
    out2 = _new_out("ts6")
    code, _, se = _run(["plan", "--group-dir", group_copy, "--noise-dir", st["noise_dir"], "--out", out2,
                        "--group", GROUP] + SMALL)
    assert code == 0, se
    man2 = _manifest(out2)
    row = man2.iloc[0]
    shutil.rmtree(str(group_copy / ("specimen_%d" % int(row["twin_of_specimen_id"]))))
    code3, _, _ = _run(["twin", "--out", out2, "--twin", int(row["specimen_id"])])
    sp = out2 / "status" / ("twin_%d.json" % int(row["specimen_id"]))
    st3 = json.loads(sp.read_text()) if sp.is_file() else {}
    code4, _, _ = _run(["twin", "--out", out2, "--twin", 123])
    code5, _, _ = _run(["twin", "--out", _new_out("ts6none"), "--twin", int(row["specimen_id"])])
    parts["missing real cell: build_failed, exit 1, a status file; an id not in the manifest: exit 2; "
          "no run_config.json: exit 2"] = (
        code3 == 1 and st3.get("status") == "build_failed" and bool(st3.get("reason")) and code4 == 2
        and code5 == 2)
    print("\n  [TS6] --force rebuild and refit: %.1f s; build_failed reason: %s"
          % (time.time() - t0, str(st3.get("reason"))[:120]))
    _fail_parts(parts)


# ------------------------------------------------------------ the submitter ---
def _stubs(tmp):
    stubs = tmp / "stubs"
    (stubs / "bin").mkdir(parents=True)
    (stubs / "condabase" / "etc" / "profile.d").mkdir(parents=True)
    files = {
        "bin/conda": "#!/bin/bash\n"
                     "if [ \"$1\" = info ] && [ \"$2\" = --base ]; then echo \"$FAKE_CONDA_BASE\"; exit 0; fi\n"
                     "exit 0\n",
        "condabase/etc/profile.d/conda.sh":
            "conda() { case \"$1\" in activate) export CONDA_DEFAULT_ENV=\"$2\" ;; esac; return 0; }\n",
        "bin/qsub": "#!/bin/bash\n"
                    "n=$(cat \"$QSUB_LOG.n\" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > \"$QSUB_LOG.n\"\n"
                    "{ printf 'CALL'; printf ' [%s]' \"$@\"; printf '\\n'; } >> \"$QSUB_LOG\"\n"
                    "if [ $n -eq 1 ]; then echo '5151[].testserver'; else echo '5152.testserver'; fi\n",
        "bin/python": "#!/bin/bash\n"
                      "if [ \"${1##*/}\" = run_twin_recovery.py ]; then\n"
                      "    case \"$2\" in\n"
                      "    twin|merge)\n"
                      "        { printf 'RUN'; printf ' [%s]' \"$@\"; "
                      "printf ' ENV[%s/%s/%s]\\n' \"$OMP_NUM_THREADS\" \"$OPENBLAS_NUM_THREADS\" \"$MKL_NUM_THREADS\"; }"
                      " >> \"$PY_LOG\"\n"
                      "        exit \"${PY_STUB_EXIT:-0}\" ;;\n"
                      "    list) if [ -n \"${PY_STUB_LIST:-}\" ]; then\n"
                      "            i=0; while [ $i -lt \"$PY_STUB_LIST\" ]; do echo $((910000000 + i)); i=$((i + 1)); done\n"
                      "            exit 0; fi ;;\n"
                      "    plan) if [ -n \"${PY_STUB_LIST:-}\" ]; then exit 0; fi ;;\n"
                      "    esac\n"
                      "fi\n"
                      "exec \"$REAL_PYTHON\" \"$@\"\n",
    }
    for rel, text in files.items():
        p = stubs / rel
        p.write_text(text)
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return stubs


def test_TS7_submitter():
    import run_twin_recovery as T
    st = _fixture()
    script = HERE / "submit_twin_recovery.sh"
    text = script.read_text()
    tmp = st["root"] / "ts7"
    tmp.mkdir()
    stubs = _stubs(tmp)
    home, tmpdir = tmp / "home", tmp / "tmpdir"
    home.mkdir()
    tmpdir.mkdir()
    qlog, pylog = tmp / "qsub.log", tmp / "py.log"
    out_base = st["output_base"]

    def env(**extra):
        e = {k: v for k, v in os.environ.items()
             if not (k.startswith(("TWIN_", "PBS_", "ROADMAP_", "IH_", "PY_STUB")) or k in (
                 "RUN_TAG", "NOISE_TAG", "GROUP", "TWINS", "MAX_TWINS", "NCPUS", "WALLTIME", "CONDA_ENV", "DRY_RUN",
                 "PLAN_ARGS"))}
        e.update({"PATH": "%s:%s" % (stubs / "bin", e.get("PATH", "")), "HOME": str(home), "TMPDIR": str(tmpdir),
                  "FAKE_CONDA_BASE": str(stubs / "condabase"), "REAL_PYTHON": sys.executable,
                  "DISPLAY": NO_X_DISPLAY,
                  "QSUB_LOG": str(qlog), "PY_LOG": str(pylog), "IH_CODE_DIR": str(HERE),
                  "IH_ARCHIVE_ROOT": str(st["archive_root"]), "GROUP": GROUP, "NOISE_TAG": "nfx",
                  "PLAN_ARGS": " ".join(SMALL), "ROADMAP_DISABLE": "1"})
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def reset():
        for p in (qlog, Path(str(qlog) + ".n"), pylog):
            if p.exists():
                p.unlink()

    def sh(e, path=script):
        return subprocess.run(["bash", str(path)], cwd=str(tmp), env=e, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=900)

    def lines_of(p):
        return p.read_text().splitlines() if p.exists() else []
    parser = T.build_parser()

    def parsed(line, marker):
        """The printed command after `marker`, read back by shlex and by run_twin_recovery.py's
        own parser: the namespace, or None when refused or not a call of run_twin_recovery.py."""
        if marker not in line:
            return None
        words = shlex.split(line.split(marker, 1)[1])
        if len(words) < 3 or words[0] != "python" or words[1] != str(HERE / "run_twin_recovery.py"):
            return None
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                return parser.parse_args(words[2:])
        except SystemExit:
            return None

    parts = {"bash -n": subprocess.run(["bash", "-n", str(script)]).returncode == 0}
    parts["the marker lines of submit_noise_fit.sh (stage S7), the cap"] = all(s in text for s in (
        "export ROADMAP_STAGE=S7", "export ROADMAP_N_TASKS=", "specs/roadmap/hpc_status.sh", "roadmap_job_start",
        "trap 'roadmap_job_end $?' EXIT", "MAX_CORES=120", "set -u relaxed", "conda activate \"$1\"",
        "JOB_NAME_TWIN=twin_recovery", "JOB_NAME_MERGE=twin_merge"))
    usage = "\n".join(ln for ln in text.splitlines() if ln.startswith("#"))
    parts["the usage text names Block 17's plan options"] = all(o in usage for o in (
        "--n-no-noise", "--gbar-bounds", "--tau-h-pk-bounds-ms", "--tau-h-pk-truth-ms"))

    # the dry run (the run not planned yet: it plans into a temporary directory, removed)
    out_dir = out_base / "twin_recovery" / "t7" / GROUP
    reset()
    p = sh(env(RUN_TAG="t7", DRY_RUN=1))
    lines = p.stdout.splitlines()
    q = [ln for ln in lines if "DRY RUN:" in ln and " qsub " in ln]
    sub = [ln for ln in lines if "DRY RUN:   subjob " in ln]
    mrg = [ln for ln in lines if "DRY RUN:   merge:" in ln]
    ids = TWIN_IDS
    parts["dry run: one array qsub (-J 0-3: four twins, the noise-free one counted; the default ncpus=1, "
          "amendment 7), one merge qsub (afterany)"] = (
        p.returncode == 0 and len(q) == 2 and " -J 0-3 " in q[0] and " select=1:ncpus=1 " in q[0]
        and " walltime=24:00:00 " in q[0] and " -N twin_recovery " in q[0] and "depend=afterany:" in q[1]
        and " -N twin_merge " in q[1])
    ns_t = [parsed(ln, "subjob %d:" % i) for i, ln in enumerate(sub)]
    ns_m = parsed(mrg[0], "merge:") if mrg else None
    parts["dry run: four twin commands (manifest order) and the merge, each accepted by the parser"] = (
        len(sub) == 4 and len(mrg) == 1 and all(n is not None for n in ns_t) and ns_m is not None
        and [(n.command, n.out, n.twin, n.force) for n in ns_t] == [("twin", str(out_dir), int(i), False)
                                                                    for i in ids]
        and (ns_m.command, ns_m.out, ns_m.no_plot) == ("merge", str(out_dir), False))
    parts["dry run: nothing written (no run dir, no run file, no temporary dir left), nothing submitted"] = (
        not out_dir.exists() and not (home / "twin_recovery_runs").exists() and not qlog.exists()
        and not pylog.exists() and not any(tmpdir.iterdir()) and "CLI contract" in p.stdout)
    parts["DISPLAY set to a display no X server answers, in every call (amendment 10): the dry run completes, "
          "no \"can't open DISPLAY\""] = (
        p.returncode == 0 and "can't open DISPLAY" not in p.stdout + p.stderr and "\nunset DISPLAY\n" in text)
    dry_text = p.stdout

    # the cap, NCPUS=4 given explicitly: 31 x 4 = 124 refused, 30 x 4 = 120 accepted
    reset()
    p31 = sh(env(RUN_TAG="t7c", PY_STUB_LIST=31, NCPUS=4))
    p30 = sh(env(RUN_TAG="t7c", PY_STUB_LIST=30, NCPUS=4, DRY_RUN=1))
    parts["31 twins x 4 cores refused (exit 3), 30 x 4 = 120 accepted"] = (
        p31.returncode == 3 and "124 cores > 120" in p31.stderr and not qlog.exists() and p30.returncode == 0)

    # MAX_TWINS=2: a stubbed submission of 2 subjobs (it plans the run for real)
    reset()
    p2 = sh(env(RUN_TAG="t7", MAX_TWINS=2))
    calls = lines_of(qlog)
    run_files = sorted((home / "twin_recovery_runs").glob("t7_%s_*.env" % GROUP))
    rf = run_files[-1] if run_files else None
    parts["MAX_TWINS=2: 2 subjobs (-J 0-1), then the merge after the array; the run file; the plan written"] = (
        p2.returncode == 0 and len(calls) == 2 and rf is not None and "[-J] [0-1]" in calls[0]
        and "[select=1:ncpus=1]" in calls[0] and ("[TWIN_ROLE=twin,TWIN_RUN_FILE=%s,ROADMAP_DISABLE=1]" % rf) in calls[0]
        and "[depend=afterany:5151[].testserver]" in calls[1]
        and ("[TWIN_ROLE=merge,TWIN_RUN_FILE=%s,ROADMAP_DISABLE=1]" % rf) in calls[1]
        and "TWIN_IDS=( 900000000 900000100 )" in rf.read_text() and (out_dir / "manifest.csv").is_file()
        and (out_dir / "logs" / rf.name).is_file())
    # a real (stubbed) submission of the whole run: the plan a no-op, qsub twice
    reset()
    man_before = _mtimes([out_dir / "manifest.csv", out_dir / "run_config.json"])
    p3 = sh(env(RUN_TAG="t7"))
    calls3 = lines_of(qlog)
    parts["a stubbed submission: plan a no-op, the run file, qsub twice (-J 0-3)"] = (
        p3.returncode == 0 and len(calls3) == 2 and "[-J] [0-3]" in calls3[0]
        and _mtimes([out_dir / "manifest.csv", out_dir / "run_config.json"]) == man_before
        and "already planned" in p3.stdout)
    # the dry run on a planned run: plan on it writes nothing; another seed refused
    reset()
    p4 = sh(env(RUN_TAG="t7", DRY_RUN=1, PLAN_ARGS=" ".join(SMALL + ["--seed", "5"])))
    parts["planned run, other options: refused (3), nothing submitted"] = p4.returncode == 3 and not qlog.exists()
    # one twin: a plain job; an unknown twin refused
    reset()
    p5 = sh(env(RUN_TAG="t7", TWINS="900000100"))
    calls5 = lines_of(qlog)
    p6 = sh(env(RUN_TAG="t7", TWINS="900000100,123", DRY_RUN=1))
    parts["TWINS=one id: a plain job with TWIN_INDEX=0; an unknown id refused (3)"] = (
        p5.returncode == 0 and len(calls5) == 2 and "[-J]" not in calls5[0] and "TWIN_INDEX=0" in calls5[0]
        and p6.returncode == 3 and "123" in p6.stderr)
    # the job roles, from a spool copy, with the run file of the MAX_TWINS submission
    spool = tmp / "spool"
    spool.mkdir()
    job = spool / "5151.SC"
    shutil.copy2(str(script), str(job))
    reset()
    pt = sh(env(PBS_JOBID="5151[1].testserver", PBS_ARRAY_INDEX=1, TWIN_ROLE="twin", TWIN_RUN_FILE=rf or "",
                OMP_NUM_THREADS=9), job)
    pm = sh(env(PBS_JOBID="5152.testserver", TWIN_ROLE="merge", TWIN_RUN_FILE=rf or ""), job)
    runs = lines_of(pylog)
    exp_t = "RUN [%s] [twin] [--out] [%s] [--twin] [900000100] ENV[1/1/1]" % (HERE / "run_twin_recovery.py", out_dir)
    exp_m = "RUN [%s] [merge] [--out] [%s] ENV[1/1/1]" % (HERE / "run_twin_recovery.py", out_dir)
    parts["the twin role: the index-th twin, threads NCPUS; the merge role: threads 1; logs teed"] = (
        pt.returncode == 0 and pm.returncode == 0 and runs == [exp_t, exp_m]
        and (out_dir / "logs" / "twin_900000100_5151_1_.testserver.log").is_file())
    pq = sh(env(PBS_JOBID="99.testserver", RUN_TAG="t7"), job)
    pn = sh(env(RUN_TAG="t7", NOISE_TAG=""))
    parts["refused: qsub'd without a role (2); no NOISE_TAG (2)"] = pq.returncode == 2 and pn.returncode == 2
    print("\n  [TS7] dry-run qsub lines:\n    " + "\n    ".join(q))
    print("  [TS7] dry-run commands:\n    " + "\n    ".join(sub + mrg))
    _STATE["dry_text"] = dry_text
    _fail_parts(parts)


def test_TS8_TS14_ascii_lf_frozen():
    bad = []
    for name in OWN_FILES:
        raw = (HERE / name).read_bytes()
        if b"\r" in raw or any(b > 127 for b in raw):
            bad.append(name)
    assert not bad, "not pure ASCII / LF: %s" % bad
    r = subprocess.run(["git", "-C", str(HERE), "diff", "--stat", CONTRACT_COMMIT, "--"] + list(FROZEN),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        raise unittest.SkipTest("git diff unavailable: %s" % r.stderr.strip())
    assert r.stdout.strip() == "", "frozen files differ from %s: %s" % (CONTRACT_COMMIT, r.stdout)


def test_TS9_controls_only():
    import twin_manifest as TM
    sorted_ids = np.array(sorted(REAL_IDS), dtype=np.int64)
    seed = next(s for s in range(200) if int(TM.configuration_cells(sorted_ids, 1, s, 2)[0]) not in FLAGGED)
    out = _new_out("ts9")
    code, _, se = _plan(out, "--seed", seed)
    assert code == 0, se
    man = _manifest(out)
    act = man[man["configuration"] == "active_fit"].reset_index(drop=True)
    assert not bool(act["cell_tau_s_above_window"].iloc[0]), "the fixture's draw must give an unflagged control"
    _write_results(out, act, {int(act["specimen_id"].iloc[0]): 1.1})
    try:
        code, so, se = _run(["merge", "--out", out, "--no-plot"])
    except Exception:                      # noqa: BLE001 -- an exception out of merge is this check's failure
        code, so, se = None, "", traceback.format_exc()
    parts = {"merge exit 0, no traceback%s" % (("" if code == 0 else " (%s)" % se.strip().splitlines()[-1])
                                               if se.strip() else ""): code == 0 and "Traceback" not in so + se}
    gated = {"Cm", "gbar", "kappa_tau"}              # Block 17: the required axes of ih5; dv_h skipped
    for sub in ("report", "report_without_flagged"):
        d = out / sub
        per = pd.read_csv(d / "recovery_per_cell.csv") if (d / "recovery_per_cell.csv").is_file() else pd.DataFrame()
        fp = pd.read_csv(d / "fp_control.csv") if (d / "fp_control.csv").is_file() else pd.DataFrame()
        summ_ok = False
        if (d / "recovery_summary.csv").is_file():
            summ = pd.read_csv(d / "recovery_summary.csv")
            summ_ok = len(summ) == 0 and list(summ.columns) == ["arm", "parameter", "n", "n_railed", "median_abs_err",
                                                                 "q25_abs_err", "q75_abs_err", "median_err", "err_kind"]
        ver = pd.read_csv(d / "gate_verdict.csv") if (d / "gate_verdict.csv").is_file() else pd.DataFrame()
        absent = set(ver.loc[ver["verdict"] == "FAIL (axis absent)", "parameter"]) if len(ver) else set()
        parts["%s/: the long table (the control's rows), the FP summary, an empty summary with its columns, "
              "FAIL (axis absent) on the gated axes, the two control rows" % sub] = (
            len(per) > 0 and set(per["specimen_id"]) == {int(act["specimen_id"].iloc[0])} and len(fp) == 1
            and summ_ok and absent == gated and len(ver) == len(gated) + 2
            and set(ver["arm"]) == {"ih5"})
    parts["report_no_noise/: no noise-free twin has results -> NO_RESULTS.txt"] = (
        (out / "report_no_noise" / "NO_RESULTS.txt").is_file())
    # amendment 9.1: --drop-railed and every non-control row railed (estimates on the box's bounds)
    out2 = _new_out("ts9r")
    code, _, se = _plan(out2, "--drop-railed")
    assert code == 0, se
    man2 = _manifest(out2)
    _write_results(out2, man2, {int(t): 1.0 for t in man2["specimen_id"]})
    opts2 = json.loads((out2 / "run_config.json").read_text())["options"]
    base2 = _tau_base_ms(opts2)
    box = {"Cm": (0.3, 3.0), "Rm": (1000.0, 100000.0), "Ra": (50.0, 1000.0), "gbar": (1e-6, 1e-2),
           "dv_h": (-10.0, 10.0), "kappa_tau": (50.0 / base2, 1000.0 / base2)}   # the default boxes (Block 17)
    for t, c in zip(man2["specimen_id"], man2["configuration"]):
        if c == "active_fit":
            continue
        f = out2 / "fits" / ("twin_%d" % int(t)) / "phase2_results.csv"
        df = pd.read_csv(f)
        for n in str(df["param_names"].iloc[0]).split("|"):
            df[n] = box[n][1]                                                    # every axis on its upper bound
        df.to_csv(f, index=False)
    try:
        code2, so2, se2 = _run(["merge", "--out", out2, "--no-plot"])
    except Exception:                      # noqa: BLE001
        code2, so2, se2 = None, "", traceback.format_exc()
    summ2 = (pd.read_csv(out2 / "report" / "recovery_summary.csv")
             if (out2 / "report" / "recovery_summary.csv").is_file() else None)
    per2 = (pd.read_csv(out2 / "report" / "recovery_per_cell.csv")
            if (out2 / "report" / "recovery_per_cell.csv").is_file() else pd.DataFrame())
    parts["--drop-railed, every non-control row railed: merge exit 0, no traceback, the summary has no rows%s"
          % ((" (%s)" % se2.strip().splitlines()[-1]) if code2 != 0 and se2.strip() else "")] = (
        code2 == 0 and "Traceback" not in so2 + se2 and summ2 is not None and len(summ2) == 0
        and len(per2) > 0 and bool(per2.loc[~per2["is_fp_control"].astype(bool), "rail"].astype(bool).all()))
    _fail_parts(parts)


def _build_only(out, tid):
    """`twin` with run_ih_fit.main replaced by a no-op: the build alone (exit 1: no fit result)."""
    import run_ih_fit as R
    orig = R.main
    R.main = lambda argv=None: None
    try:
        return _run(["twin", "--out", out, "--twin", tid])
    finally:
        R.main = orig
        _clear()


def test_TS10_base_seed():
    import twin_build as TB
    import twin_config
    import twin_manifest as TM
    import twin_table as TT
    import ih_mechanism as IM
    import param_spec as PS
    b0 = twin_config.TwinConfig().base_seed
    b1, b2 = b0 + 1, b0 + 2
    runs = {}
    for tag, b in (("a", b1), ("b", b2), ("c", b1)):
        out = _new_out("ts10" + tag)
        code, _, se = _plan(out, "--base-seed", b)
        assert code == 0, se
        runs[tag] = out
    cfgs = {k: json.loads((v / "run_config.json").read_text()) for k, v in runs.items()}
    parts = {"run_config.json records the base seed (twin_config.base_seed and the option)": all(
        cfgs[k]["twin_config"]["base_seed"] == b and cfgs[k]["options"]["base_seed"] == b
        for k, b in (("a", b1), ("b", b2), ("c", b1)))}
    code, _, _ = _plan(_new_out("ts10d"))
    parts["--base-seed defaults to TwinConfig().base_seed"] = code == 0 and json.loads(
        (_STATE["root"] / "runs" / ("ts10d_%d" % _STATE["n_runs"]) / "run_config.json").read_text()
    )["twin_config"]["base_seed"] == b0
    before = _mtimes([runs["a"] / "manifest.csv", runs["a"] / "run_config.json"])
    code_r, _, se_r = _plan(runs["a"], "--base-seed", b2)
    parts["a re-plan with another base seed: refused (3), nothing changed"] = (
        code_r == 3 and "base_seed" in se_r and _mtimes([runs["a"] / "manifest.csv", runs["a"] / "run_config.json"])
        == before)
    mans = {k: _manifest(v) for k, v in runs.items()}
    row = mans["a"][mans["a"]["configuration"] == "full"].iloc[0]
    tid = int(row["specimen_id"])
    same_twin = all(
        TM.twin_theta(mans[k][mans[k]["specimen_id"] == tid].iloc[0]) == TM.twin_theta(row)
        and int(mans[k][mans[k]["specimen_id"] == tid].iloc[0]["twin_of_specimen_id"]) == int(row["twin_of_specimen_id"])
        for k in runs)
    tables_equal = all(TT.read_twin_sweep_table(runs[k] / ("twin_sweeps_%s.csv" % GROUP))
                       .equals(TT.read_twin_sweep_table(runs["a"] / ("twin_sweeps_%s.csv" % GROUP))) for k in runs)
    parts["the same twin in the three runs (id, real cell, truth) and the same table"] = same_twin and tables_equal
    traces, seeds = {}, {}
    for k, out in runs.items():
        _build_only(out, tid)
        d = out / "twins" / ("twin_%d" % tid) / ("specimen_%d" % tid)
        with np.load(d / "ss_pulses.npz") as z:
            ss = np.array(z["v"])
        with np.load(d / "ls_sweeps.npz") as z:
            ls = np.concatenate([np.asarray(z[n]) for n in sorted(z.files) if n.startswith("v_")])
        traces[k] = (ss, ls)
        seeds[k] = json.loads((d / "metadata.json").read_text())["twin"]["base_seed"]
    parts["the archive is built with the recorded base seed"] = seeds == {"a": b1, "b": b2, "c": b1}
    parts["b1 != b2: different noisy traces; b1 twice: identical noisy traces"] = (
        not np.array_equal(traces["a"][0], traces["b"][0]) and not np.array_equal(traces["a"][1], traces["b"][1])
        and np.array_equal(traces["a"][0], traces["c"][0]) and np.array_equal(traces["a"][1], traces["c"][1]))
    # the noise-free traces under each run's recorded TwinConfig (Block 12's simulate_twin, with_noise=False)
    import passive_fitting_hpc_fixed as mono
    st = _fixture()
    real = st["group_dir"] / ("specimen_%d" % int(row["twin_of_specimen_id"]))
    prot = TT.load_cell_protocol(real, spike_reason_fn=mono.sweep_spike_reason)
    table = TT.read_twin_sweep_table(runs["a"] / ("twin_sweeps_%s.csv" % GROUP))
    clean = {}
    for k in ("a", "b"):
        cfg_k = twin_config.TwinConfig(**cfgs[k]["twin_config"])
        spec = TM.twin_spec_of(mans[k][mans[k]["specimen_id"] == tid].iloc[0], F=float(cfgs[k]["options"]["F"]))
        _clear()
        cell = TB.build_twin_cell(real / "reconstruction.swc", spec, ih_spec=IM.IhSpec(),
                                  apply_theta=PS.make_ih_spec().apply, cfg=cfg_k)
        tr = TB.simulate_twin(cell, spec, prot, table, cfg=cfg_k, with_noise=False)
        clean[k] = (tr.ss_v_mV, np.concatenate(tr.ls_v_mV))
        _clear()
    parts["b1 != b2: identical noise-free traces"] = (np.array_equal(clean["a"][0], clean["b"][0])
                                                      and np.array_equal(clean["a"][1], clean["b"][1]))
    # amendment 9.2: a negative base seed refused by the parser (2), nothing written
    out_neg = _new_out("ts10neg")
    try:
        code_n, _, se_n = _plan(out_neg, "--base-seed=-1")
    except SystemExit as exc:                  # argparse's exit
        code_n, se_n = exc.code, ""
    parts["plan --base-seed=-1: exit 2, the run directory not written"] = code_n == 2 and not out_neg.exists()
    # an unknown key in the recorded twin_config: twin refused (3), nothing built
    cfg_bad = dict(cfgs["b"])
    cfg_bad["twin_config"] = dict(cfg_bad["twin_config"], not_a_field=1)
    (runs["b"] / "run_config.json").write_text(json.dumps(cfg_bad, indent=1))
    other = int(mans["b"][mans["b"]["configuration"] == "passive_fit"]["specimen_id"].iloc[0])
    code_k, _, se_k = _run(["twin", "--out", runs["b"], "--twin", other])
    parts["twin_config with an unknown key: twin refused (3), nothing built"] = (
        code_k == 3 and "not_a_field" in se_k and not (runs["b"] / "twins" / ("twin_%d" % other)).exists())
    _fail_parts(parts)


def test_TS11_eval_theta():
    import run_ih_fit as R
    import twin_manifest as TM
    rec = _recording_run("ts2", NONDEFAULT_FIT)
    out = rec["out"]
    man = _manifest(out)
    cfg = json.loads((out / "run_config.json").read_text())
    base = _tau_base_ms(cfg["options"])
    parts = {"four configurations, one fit call each": list(man["configuration"]) == list(CONFS)
             and len(rec["fit_calls"]) == 4}
    for (_, row), argv in zip(man.iterrows(), rec["fit_calls"]):
        conf = str(row["configuration"])
        ev = [a for a in argv if a.startswith("--eval-theta")]
        fa = R._parse_args(argv)                       # run_ih_fit's own parser accepts the argv (else exit 2)
        kb = R.kappa_bounds_from_args(fa)
        ok = (argv[argv.index("--arm") + 1] == EXPECTED_ARM[conf]
              and "--tau-h-pk-bounds-ms=60,900" in argv and not any(a.startswith("--kappa-bounds") for a in argv)
              and kb == (60.0 / base, 900.0 / base))
        if conf in WITH_EVAL:
            th = TM.twin_theta(row)
            got = R.parse_eval_theta(fa.eval_theta, IH5) if fa.eval_theta is not None else {}
            ok = ok and len(ev) == 1 and list(got) == list(IH5) and all(got[n] == th[n] for n in IH5) \
                and th["dv_h"] == 0.0
        else:
            ok = ok and not ev and fa.eval_theta is None
        parts["%s: arm %s, --tau-h-pk-bounds-ms (kappa box 60/base, 900/base), %s" % (
            conf, EXPECTED_ARM[conf], "--eval-theta = the truth's five values exactly" if conf in WITH_EVAL
            else "no --eval-theta")] = ok
    print("\n  [TS11] full_no_noise argv tail: %s" % " ".join(rec["fit_calls"][3][-3:] if len(rec["fit_calls"]) > 3
                                                            else []))
    _fail_parts(parts)


def test_TS12_no_noise_extras():
    import human_ih_params as HIP
    import passive_fitting_hpc_fixed as mono
    import run_ih_fit as R
    import run_twin_recovery as T
    import twin_build as TB
    import twin_config
    import twin_manifest as TM
    import twin_table as TT
    out = _STATE.get("ts3_out")
    if out is None:
        test_TS3_end_to_end()
        out = _STATE["ts3_out"]
    man = _manifest(out)
    cfg = json.loads((out / "run_config.json").read_text())
    row0 = man[man["configuration"] == "full_no_noise"].iloc[0]
    rowf = man[man["configuration"] == "full"].iloc[0]
    t0, tf = int(row0["specimen_id"]), int(rowf["specimen_id"])
    d0 = out / "twins" / ("twin_%d" % t0) / ("specimen_%d" % t0)
    df_ = out / "twins" / ("twin_%d" % tf) / ("specimen_%d" % tf)
    meta0 = json.loads((d0 / "metadata.json").read_text())

    def traces(d):
        with np.load(d / "ss_pulses.npz") as z:
            ss = np.array(z["v"])
        with np.load(d / "ls_sweeps.npz") as z:
            ls = [np.asarray(z["v_%d" % m]) for m in range(len([n for n in z.files if n.startswith("v_")]))]
        return ss, ls
    ss0, ls0 = traces(d0)
    ssf, lsf = traces(df_)
    # the partner's noise-free traces, recomputed here with the fit's own model (run_ih_fit's builders)
    real = Path(cfg["group_dir"]) / ("specimen_%d" % int(rowf["twin_of_specimen_id"]))
    prot = TT.load_cell_protocol(real, spike_reason_fn=mono.sweep_spike_reason)
    table = TT.read_twin_sweep_table(out / ("twin_sweeps_%s.csv" % GROUP))
    tcfg = twin_config.TwinConfig(**cfg["twin_config"])
    fa = R._parse_args(_ref_fit_argv(cfg["options"], out / "twins" / ("twin_%d" % tf), out / "fits" / "x", rowf))
    ih_spec = R.build_ih_spec(fa)
    pspec = R.build_param_spec(IH6, cm_bounds=R.parse_window(fa.cm_bounds), rm_bounds=R.parse_window(fa.rm_bounds),
                               ra_bounds=R.parse_window(fa.ra_bounds), gbar_bounds=R.parse_window(fa.gbar_bounds),
                               dvh_bounds=R.parse_window(fa.dvh_bounds), kappa_bounds=R.kappa_bounds_from_args(fa))
    spec_f = TM.twin_spec_of(rowf, F=float(cfg["options"]["F"]))
    _clear()
    try:
        cell = TB.build_twin_cell(real / "reconstruction.swc", spec_f, ih_spec=ih_spec, apply_theta=pspec.apply,
                                  cfg=tcfg)
        clean = TB.simulate_twin(cell, spec_f, prot, table, cfg=tcfg, with_noise=False)
    finally:
        _clear()
    parts = {"the noise-free twin: metadata with_noise False, noise_laws_applied False, the partner's real cell "
             "and truth": (meta0["twin"]["with_noise"] is False and meta0["twin"].get("noise_laws_applied") is False
                           and int(row0["paired_with"]) == tf and TM.twin_theta(row0) == TM.twin_theta(rowf)
                           and int(row0["twin_of_specimen_id"]) == int(rowf["twin_of_specimen_id"]))}
    # exact: the same NEURON computation (same cell, theta, protocol, dt) on both sides
    parts["its archive's SS and LS traces equal the partner's noise-free traces, bit for bit"] = (
        np.array_equal(ss0, clean.ss_v_mV) and len(ls0) == len(clean.ls_v_mV)
        and all(np.array_equal(a, b) for a, b in zip(ls0, clean.ls_v_mV)))
    parts["the partner's (noisy) traces differ from them"] = (
        not np.array_equal(ssf, ss0) and not all(np.array_equal(a, b) for a, b in zip(lsf, ls0)))
    st0 = json.loads((out / "status" / ("twin_%d.json" % t0)).read_text())
    parts["the noise-free twin is fitted (status ok, its row)"] = st0["status"] == "ok" and T._results_row(
        out / "fits" / ("twin_%d" % t0) / "phase2_results.csv", t0) is not None

    code, so, se = _run(["merge", "--out", out, "--no-plot"])
    # round_trip: written with 17 significant digits, read back as the same float64
    ex = pd.read_csv(out / "recovery_extras.csv", float_precision="round_trip")
    parts["merge exit 0; recovery_extras.csv: its columns in order, one row per twin with results"] = (
        code == 0 and list(ex.columns) == list(T.EXTRAS_COLUMNS) and list(ex["twin_id"]) == list(man["specimen_id"]))
    base = _tau_base_ms(cfg["options"])
    ok_rows, notes = True, []
    for _, e in ex.iterrows():
        tid = int(e["twin_id"])
        m = man[man["specimen_id"] == tid].iloc[0]
        conf = str(m["configuration"])
        res = pd.read_csv(out / "fits" / ("twin_%d" % tid) / "phase2_results.csv", float_precision="round_trip")
        r = res[res["specimen_id"] == tid].iloc[0]
        th = TM.twin_theta(m)
        has_ih = conf != "active_fit"
        v = float(r["v_deepest_train_mV"])
        # independent: human_ih_params.minf directly, with the truth's dv_h (2.8 (f))
        gh_t = (th["gbar"] * float(HIP.minf(v, ih_spec.mechanism, vshift_mV=float(ih_spec.vshift_base_mV),
                                             vshift_minf_mV=th["dv_h"])) if has_ih else float("nan"))
        lt, le = float(r["loss_at_eval_theta"]), float(r["loss_at_estimate"])
        row_ok = (str(e["configuration"]) == conf and int(e["paired_with"]) == int(m["paired_with"])
                  and bool(e["with_noise"]) == bool(m["with_noise"])
                  and bool(e["cell_tau_s_above_window"]) == bool(m["cell_tau_s_above_window"])
                  and (float(e["loss_at_estimate"]) == le or (np.isnan(le) and np.isnan(e["loss_at_estimate"]))))
        if conf in WITH_EVAL:
            row_ok = row_ok and np.isfinite(e["loss_at_truth"]) and float(e["loss_at_truth"]) == lt \
                and e["loss_verdict"] in ("optimiser", "data", "tie")
        else:
            row_ok = row_ok and np.isnan(e["loss_at_truth"]) and e["loss_verdict"] == "na"
        if has_ih:
            # 1e-12 relative: float64 products of the same factors, in another order at most
            row_ok = row_ok and np.isfinite(v) and np.isclose(e["gh_deep_true_S_cm2"], gh_t, rtol=1e-12, atol=0.0) \
                and np.isclose(e["tau_h_pk_true_ms"], th["kappa_tau"] * base, rtol=1e-12, atol=0.0)
        else:
            row_ok = row_ok and np.isnan(e["gh_deep_true_S_cm2"]) and np.isnan(e["tau_h_pk_true_ms"])
        if EXPECTED_ARM[conf] == "ih5":
            ge = float(r["gh_at_v_deepest_S_cm2"])
            row_ok = row_ok and float(e["gh_deep_est_S_cm2"]) == ge and float(e["tau_h_pk_est_ms"]) == float(
                r["tau_h_pk_ms"])
            if has_ih:
                row_ok = row_ok and np.isclose(e["ln_gh_deep_ratio"], np.log(ge / gh_t), rtol=1e-12, atol=1e-15)
        else:
            row_ok = row_ok and np.isnan(e["gh_deep_est_S_cm2"]) and np.isnan(e["ln_gh_deep_ratio"])
        if not row_ok:
            notes.append("row of %s FAILED" % conf)
        ok_rows = ok_rows and bool(row_ok)
        notes.append("%s: L(truth)=%.6g L(est)=%.6g verdict %s v*=%.3f gh_true=%.4g gh_est=%.4g tau_pk true %.4g est "
                     "%.4g" % (conf, e["loss_at_truth"], e["loss_at_estimate"], e["loss_verdict"],
                               e["v_deepest_train_mV"], e["gh_deep_true_S_cm2"], e["gh_deep_est_S_cm2"],
                               e["tau_h_pk_true_ms"], e["tau_h_pk_est_ms"]))
    parts["every row: loss_at_truth finite for full and full_no_noise (= loss_at_eval_theta), na elsewhere; "
          "gh_deep_true, tau_h_pk_true and the log ratio recomputed here"] = ok_rows
    hand = [((1.0, 2.0), "optimiser"), ((2.0, 1.0), "data"), ((1.0, 1.01), "tie"), ((1.01, 1.0), "tie"),
            ((0.97, 1.0), "optimiser"), ((1.0, 0.97), "data"), ((float("nan"), 1.0), "na"),
            ((1.0, float("inf")), "na"), ((0.0, 0.0), "tie")]
    parts["loss_verdict on hand cases (2 %% of the other loss)"] = all(T.loss_verdict(*a) == w for a, w in hand)
    print("\n  [TS12] " + "\n  [TS12] ".join(notes))
    _fail_parts(parts)


def _old_report_module():
    """ih_recovery_report.py at the contract commit (WAVE3A), loaded from `git show` as a module."""
    import importlib.util
    rel = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-prefix"], stdout=subprocess.PIPE,
                         universal_newlines=True, check=True).stdout.strip()
    src = subprocess.run(["git", "-C", str(HERE), "show", "%s:%sih_recovery_report.py" % (CONTRACT_COMMIT, rel)],
                         stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    d = Path(tempfile.mkdtemp(prefix="old_report_"))
    atexit.register(shutil.rmtree, str(d), True)
    (d / "ih_recovery_report_wave3a.py").write_text(src)
    spec = importlib.util.spec_from_file_location("ih_recovery_report_wave3a", str(d / "ih_recovery_report_wave3a.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_TS13_gate():
    import ih_recovery_report as RR
    old = _old_report_module()
    tol = RR.RecoveryTolerances()
    req = ("Cm", "gbar", "kappa_tau")

    def summary(arm, errs):
        return pd.DataFrame([dict(arm=arm, parameter=k, n=3, n_railed=0, median_abs_err=v, q25_abs_err=v,
                                  q75_abs_err=v, median_err=v, err_kind="difference" if k == "dv_h" else "log_ratio")
                             for k, v in errs.items()])
    good = {"Cm": 0.1, "Rm": 0.5, "Ra": 0.5, "gbar": 0.1, "kappa_tau": 0.2}     # inside D-011's tolerances
    fp_pass = pd.DataFrame([dict(arm="ih5", n_fp_cells=2, n_gbar_low_rail=2, n_gbar_high_rail=0, low_rail_frac=1.0,
                                 median_cm_abs_log_ratio=0.05, median_cm_factor=1.05)])
    fp_none = fp_pass.iloc[0:0]
    v1, p1 = RR.evaluate_gate(summary("ih5", good), fp_pass, arm="ih5", tol=tol, required_axes=req)
    parts = {"no dv_h, the three required axes inside: pass, no dv_h row, the control rows": (
        p1 is True and "dv_h" not in set(v1["parameter"]) and len(v1) == 5 + 2
        and set(v1.loc[v1["parameter"].isin(req), "verdict"]) == {"pass"})}
    bad_k = dict(good, kappa_tau=float(np.log(1.5)) + 1e-3)
    v2, p2 = RR.evaluate_gate(summary("ih5", bad_k), fp_none, arm="ih5", tol=tol, required_axes=req)
    no_g = {k: v for k, v in good.items() if k != "gbar"}
    v3, p3 = RR.evaluate_gate(summary("ih5", no_g), fp_none, arm="ih5", tol=tol, required_axes=req)
    bad_r = dict(good, Rm=50.0, Ra=50.0)                                          # not gated axes
    v4, p4 = RR.evaluate_gate(summary("ih5", bad_r), fp_none, arm="ih5", tol=tol, required_axes=req)
    parts["fails on kappa_tau beyond its tolerance; fails with gbar absent (FAIL (axis absent)); R_m, R_a "
          "never decide"] = (
        p2 is False and v2.loc[v2["parameter"] == "kappa_tau", "verdict"].iloc[0] == "FAIL"
        and p3 is False and v3.loc[v3["parameter"] == "gbar", "verdict"].iloc[0] == "FAIL (axis absent)"
        and p4 is True)
    # None: the WAVE3A code's verdict tables, exactly, on several summaries
    cases = [(summary("ih5", good), fp_pass, "ih5"), (summary("ih5", good), fp_none, "ih5"),
             (summary("ih6", dict(good, dv_h=1.0)), fp_none, "ih6"), (summary("ih6", dict(good, dv_h=5.0)), fp_none,
                                                                      "ih6"),
             (summary("ih5", bad_k), fp_pass, "ih5"), (summary("ih5", no_g), fp_pass, "ih5"),
             (pd.DataFrame(columns=list(summary("ih5", good).columns)), fp_pass, "ih5")]
    same = True
    for sm, fp, arm in cases:
        vn, pn = RR.evaluate_gate(sm, fp, arm=arm, tol=tol, required_axes=None)
        vo, po = old.evaluate_gate(sm, fp, arm=arm, tol=old.RecoveryTolerances())
        vd, pd_ = RR.evaluate_gate(sm, fp, arm=arm, tol=tol)
        try:
            pd.testing.assert_frame_equal(vn, vo)
            pd.testing.assert_frame_equal(vd, vo)
        except AssertionError:
            same = False
        same = same and pn == po == pd_
    parts["required_axes=None (and omitted) = the WAVE3A code's verdict table and verdict, seven summaries "
          "(dv_h absent fails there)"] = same and old.evaluate_gate(
        summary("ih5", good), fp_none, arm="ih5", tol=old.RecoveryTolerances())[1] is False
    try:
        RR.evaluate_gate(summary("ih5", good), fp_none, arm="ih5", tol=tol, required_axes=("Cm", "tau"))
        refused = False
    except ValueError:
        refused = True
    parts["an unknown axis in required_axes: ValueError"] = refused
    # the merge's reports (TS5's run): the noise-free twins in report_no_noise/ only
    out = _STATE.get("ts5_out")
    if out is None:
        test_TS5_merge()
        out = _STATE["ts5_out"]
    man = _manifest(out)
    quiet = set(int(t) for t, w in zip(man["specimen_id"], man["with_noise"]) if not bool(w))
    ids = {}
    for sub in ("report", "report_without_flagged", "report_no_noise"):
        f = out / sub / "recovery_per_cell.csv"
        ids[sub] = set(int(t) for t in pd.read_csv(f)["specimen_id"]) if f.is_file() else None
    ver = pd.read_csv(out / "report" / "gate_verdict.csv")
    ver0 = pd.read_csv(out / "report_no_noise" / "gate_verdict.csv")
    parts["report_no_noise/ holds exactly the noise-free twins; report/ and report_without_flagged/ none of them"] = (
        bool(quiet) and ids["report_no_noise"] == quiet and ids["report"] is not None and not (ids["report"] & quiet)
        and (ids["report_without_flagged"] is None or not (ids["report_without_flagged"] & quiet)))
    parts["the gate in report/ and report_no_noise/ is on ih5, without a dv_h row"] = (
        set(ver["arm"]) == {"ih5"} and "dv_h" not in set(ver["parameter"]) and set(ver0["arm"]) == {"ih5"}
        and "dv_h" not in set(ver0["parameter"]))
    _fail_parts(parts)


def test_TS15_penalty_round_trip():
    import run_ih_fit as R
    import run_twin_recovery as T
    na = [(1e6, 0.3), (0.3, 1e6), (float("nan"), 0.3), (0.3, float("inf")), (2e6, 1e7), (1e6, 1e6)]
    ordinary = [((1.0, 2.0), "optimiser"), ((2.0, 1.0), "data"), ((1.0, 1.01), "tie"), ((0.0, 0.0), "tie"),
                ((0.97, 1.0), "optimiser"), ((1.0, 0.97), "data"), ((999999.0, 0.3), "data")]
    parts = {"the penalty is run_ih_fit's 1e6": R._LOSS_CRASH_PENALTY == 1e6,
             "na at or above the penalty and for non-finite losses": all(T.loss_verdict(*a) == "na" for a in na),
             "ordinary pairs unchanged": all(T.loss_verdict(*a) == w for a, w in ordinary)}
    out = _STATE.get("ts3_out")
    if out is None:
        test_TS3_end_to_end()
        out = _STATE["ts3_out"]
    code, _, se = _run(["merge", "--out", out, "--no-plot"])
    ex = pd.read_csv(out / "recovery_extras.csv", float_precision="round_trip")
    ok = code == 0 and len(ex) == 4
    for _, e in ex.iterrows():
        tid = int(e["twin_id"])
        res = pd.read_csv(out / "fits" / ("twin_%d" % tid) / "phase2_results.csv", float_precision="round_trip")
        r = res[res["specimen_id"] == tid].iloc[0]
        for a, b in (("loss_at_truth", "loss_at_eval_theta"), ("loss_at_estimate", "loss_at_estimate")):
            x, y = float(e[a]), float(r[b])
            ok = ok and (x == y or (np.isnan(x) and np.isnan(y)))
    parts["recovery_extras.csv: loss_at_truth / loss_at_estimate = the fit's loss_at_eval_theta / "
          "loss_at_estimate exactly (round_trip on both sides)"] = ok
    _fail_parts(parts)


def test_convergence():
    raise unittest.SkipTest("not a discretisation: wiring of Blocks 5, 12, 13, run_ih_fit and ih_recovery_report")


def test_determinism():
    raise unittest.SkipTest("covered by TS1: a second identical plan writes nothing, and the manifest equals "
                            "draw_twin_manifest recomputed here")


# ---------------------------------------------------------------- runner ---
def _environment():
    parts = ["python %s" % platform.python_version()]
    for package in REPORT_PACKAGES:
        try:
            parts.append("%s %s" % (package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            parts.append("%s (not installed)" % package)
    parts.append(platform.platform())
    parts.append("seed %d" % SEED)
    return " | ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true", help="run `nrnivmodl mod` when ./x86_64/special is missing")
    args = ap.parse_args(argv)
    if args.build and not (HERE / "x86_64" / "special").exists():
        subprocess.run(["nrnivmodl", "mod"], cwd=str(HERE), check=True)
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print(_environment())
    results = []
    t_all = time.perf_counter()
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
    print()
    for name, status, seconds, detail in results:
        lines = detail.strip().splitlines()
        headline = lines[0] if lines else ""
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, headline[:160])).rstrip())
    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items())
          + "; %.1f s in all" % (time.perf_counter() - t_all))
    for path in _SAVED_ARTIFACTS:
        print("-- arrays from a failed comparison: %s" % path)
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
