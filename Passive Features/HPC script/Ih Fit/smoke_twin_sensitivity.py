"""Smoke test for Block 28 of specs/SPEC.md: the twins' run with trf, the sensitivity sub-command and its noise (V1-V7).

The fixture is smoke_twin_recovery.py's (imported, not copied): a group of 4 real-layout fixture cells (one
SS sweep, 92, with two trace-QC drops; LS sweeps 54..57 of -110, -70, -30 and +30 pA), each with a
ball-and-stick reconstruction.swc, and the noise-fit run's two tables. Every run uses
--n-per-configuration 1 --n-no-noise 1: four twins (full, passive_fit, active_fit, full_no_noise; the
noise-free twin copies the full twin's real cell and truth).

Checks
    test_V1_plan_trf          plan's defaults: refine trf, refine_max_nfev 30 (= run_ih_fit's parser default)
                              recorded in run_config.json; fit_argv carries --refine trf --refine-max-nfev N
                              (30, and a non-default 7), accepted by run_ih_fit's parser with those values;
                              --refine nelder_mead and none still accepted
    test_V5_twin_trf          `twin` on the full twin with the default trf at a reduced budget (1 call, 1
                              initial point, --refine-max-nfev 3, a subprocess): status ok; merge writes
                              recovery_extras.csv with the Block 28 columns after the Block 22 ones, the L_2
                              columns equal to the fit's (sqrt_l2_at_truth = sqrt_l2_at_eval_theta), the
                              on_bound_<name> flags the fit's
    test_V2_sensitivity       `sensitivity --n-noise 20` on the noisy full twin (built by V5: reused, not
                              rebuilt) and on its noise-free partner (built here): exit 0; Block 27's four
                              files and identifiability.csv exist; sd_lin_<name> finite and > 0 for every
                              name, both weightings; the rows equal identifiability_<id>.csv's on the shared
                              columns; the sensitivities S of the two twins are identical; with the noisy
                              twin's seeds and layout the noise-free twin's S gives the noisy twin's sd_lin
                              (1e-9 relative); the difference as built (own seeds and layouts) reported on
                              the PASS line (amendment 2, D-249)
    test_V3_bundle_noise      the bundle noise: length = the bundle's; two calls identical; a one-sweep long
                              bundle = ls_sweep_noise of that sweep (exactly); a two-sweep long bundle = the
                              mean of the two sweeps' draws; a brief bundle = the mean of the rows of its
                              polarity (recomputed here), its SD below one window's; a bundle on another
                              grid raises BundleGridError (the entry's STOP case)
    test_V4_merge             merge gathers <run>/identifiability.csv (two rows, W0 and W1, per twin) and
                              <run>/truth_trace_check.csv (one row per bundle per twin, twin_id first),
                              one twin_id per twin run; no <group> level under the run directory
                              (amendment 2, D-249)
    test_V8_verdict_sqrt_l2   amendment 3 (D-250): the entry's hand case (two bundles of weight 0.5; estimate L
                              0.1414, L_2 0.0400; truth L 0.1732, L_2 0.0300) gives 'optimiser' on sqrt_l2 and
                              'data' on L; with the sqrt_l2 values nan the verdict is the one on L ('loss'); a
                              failed point's sqrt(L_2) = sqrt(1e6) gives 'na'; on V5's run the trf twin's row
                              has loss_verdict_on 'sqrt_l2' and the verdict recomputed from the fit's columns
    test_V6_submitter         DRY_RUN=1 MODE=sensitivity prints one sensitivity command per twin with
                              --n-workers NCPUS --n-noise N_NOISE (each accepted by the parser), the merge,
                              and PLAN_ARGS' --refine-max-nfev in the plan command; MODE unset prints the twin
                              commands as before; MODE=bogus and N_NOISE=1 refused (2); a job of the array
                              with a run file holding TWIN_MODE=sensitivity runs the sensitivity command
    test_V7_ascii_lf_bash     the own files pure ASCII and LF; bash -n on the submitter; its roadmap marker
                              lines unchanged

Run (from this folder: NEURON loads ./x86_64 from the working directory)
    python smoke_twin_sensitivity.py [--build]
        --build runs `nrnivmodl mod` first when ./x86_64/special is missing.
        Prints the environment and a PASS / FAIL / ERROR / TODO / SKIP table, then the full message of every
        failure. Exit code 1 if any check failed, errored or is still TODO.

Keep this file pure ASCII (hpc-python-compat).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                     # NEURON loads ./x86_64 from the CWD
sys.path.insert(0, str(HERE))

import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402

import smoke_twin_recovery as STR  # noqa: E402  (the fixture, the stubs, the runner's helpers)

OWN_FILES = ("run_twin_recovery.py", "submit_twin_recovery.sh", "smoke_twin_sensitivity.py")
N_NOISE = 20
#: the reduced budget of V5's fit: trf's max_nfev per start (the plan's --refine-max-nfev)
V5_MAX_NFEV = 3
IH6 = ("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau")
B22_EXTRAS = ("loss_after_search", "refine_improved", "refine_n_evals", "refine_best_start", "dvh_true_mV",
              "dvh_est_mV")
B28_EXTRAS = ("l2_after_search", "l2_after_refine", "refine_n_converged", "refine_n_failed", "sqrt_l2_at_estimate",
              "sqrt_l2_at_truth")
#: the identities below compare float64 results of the same operations on equal inputs: 1e-9 is the
#: entry's tolerance and far above roundoff of a 20 x 6 least-squares solve
REL_TOL = 1e-9

_S = {}


def _fail_parts(parts):
    bad = [k for k, v in parts.items() if not v]
    assert not bad, "failed parts: " + "; ".join(bad)


def _run_dir():
    """One planned run (the fixture, --refine-max-nfev V5_MAX_NFEV), shared by V2-V5."""
    if "out" not in _S:
        out = STR._new_out("sens")
        code, so, se = STR._plan(out, "--refine-max-nfev", str(V5_MAX_NFEV))
        assert code == 0, "plan exited %d: %s" % (code, se[-2000:])
        man = STR._manifest(out)
        tid = {str(c): int(t) for c, t in zip(man["configuration"], man["specimen_id"])}
        _S.update(out=out, man=man, tid=tid, cfg=json.loads((out / "run_config.json").read_text()))
    return _S


def _sens(tid):
    """`sensitivity --n-noise N_NOISE` on twin tid (a subprocess, as on the cluster), once."""
    key = "sens_%d" % tid
    if key not in _S:
        st = _run_dir()
        t0 = time.time()
        r = STR._sub(["sensitivity", "--run-dir", st["out"], "--twin", tid, "--n-noise", N_NOISE])
        _S[key] = (r, time.time() - t0)
    return _S[key]


def _sdir(tid):
    import run_twin_recovery as T
    st = _run_dir()
    return T.sens_dir_of(st["out"], tid)


def _protocol_table(real_id):
    import passive_fitting_hpc_fixed as mono
    import run_twin_recovery as T
    import twin_table as TT
    st = _run_dir()
    table = TT.read_twin_sweep_table(st["out"] / ("twin_sweeps_%s.csv" % st["cfg"]["group"]))
    prot = TT.load_cell_protocol(Path(st["cfg"]["group_dir"]) / ("specimen_%d" % real_id),
                                 spike_reason_fn=mono.sweep_spike_reason,
                                 twin_cfg=T.recorded_twin_config(st["cfg"]))
    return prot, table


# ---------------------------------------------------------------- checks ---
def test_V1_plan_trf():
    import run_ih_fit as R
    import run_twin_recovery as T
    out = STR._new_out("v1")
    code, _so, se = STR._plan(out)
    cfg = json.loads((out / "run_config.json").read_text()) if code == 0 else {}
    o = cfg.get("options", {})
    fit_default = R._parse_args(["--archive-dir", "-", "--output-dir", "-", "--code-dir", "-"]).refine_max_nfev
    parts = {"plan exits 0": code == 0,
             "run_config.json: refine 'trf', refine_max_nfev 30": o.get("refine") == "trf"
             and o.get("refine_max_nfev") == 30,
             "REFINE_MAX_NFEV_DEFAULT = run_ih_fit's parser default (%r)" % fit_default:
             T.REFINE_MAX_NFEV_DEFAULT == fit_default == 30,
             "REFINE_CHOICES hold trf, the default": "trf" in T.REFINE_CHOICES and T.REFINE_DEFAULT == "trf"}
    for nfev in (30, 7):
        oo = dict(o, refine_max_nfev=nfev)
        argv = T.fit_argv(oo, "/a", "/o", "ih6_d207", n_workers=2)
        i = argv.index("--refine")
        fa = R._parse_args(argv)
        parts["fit_argv carries --refine trf ... --refine-max-nfev %d; run_ih_fit parses it" % nfev] = (
            argv[i:i + 8] == ["--refine", "trf", "--refine-starts", str(o.get("refine_starts")), "--refine-maxfev",
                              str(o.get("refine_maxfev")), "--refine-max-nfev", str(nfev)]
            and fa.refine == "trf" and fa.refine_max_nfev == nfev)
    for alt in ("nelder_mead", "none"):
        out2 = STR._new_out("v1" + alt)
        c2, _s, e2 = STR._plan(out2, "--refine", alt)
        o2 = json.loads((out2 / "run_config.json").read_text())["options"] if c2 == 0 else {}
        parts["plan --refine %s accepted and recorded" % alt] = c2 == 0 and o2.get("refine") == alt
    # a plan written before Block 28 (no refine_max_nfev): fit_argv passes no --refine-max-nfev
    old = {k: v for k, v in o.items() if k != "refine_max_nfev"}
    parts["an older plan's options: no --refine-max-nfev"] = "--refine-max-nfev" not in T.fit_argv(old, "/a", "/o",
                                                                                                     "ih5")
    print("\n  [V1] options: refine %r, refine_max_nfev %r, refine_starts %r, refine_maxfev %r"
          % (o.get("refine"), o.get("refine_max_nfev"), o.get("refine_starts"), o.get("refine_maxfev")))
    _fail_parts(parts)


def test_V5_twin_trf():
    import run_twin_recovery as T
    st = _run_dir()
    out, tf = st["out"], st["tid"]["full"]
    t0 = time.time()
    r = STR._sub(["twin", "--out", out, "--twin", tf])
    secs = time.time() - t0
    stj = json.loads((out / "status" / ("twin_%d.json" % tf)).read_text()) if (
        out / "status" / ("twin_%d.json" % tf)).is_file() else {}
    argv = json.loads((out / "fits" / ("twin_%d" % tf) / "fit_argv.json").read_text())["argv"]
    _S["v5_meta_mtime"] = os.stat(out / "twins" / ("twin_%d" % tf) / ("specimen_%d" % tf) / "metadata.json"
                                  ).st_mtime_ns
    code, so, se = STR._run(["merge", "--out", out, "--no-plot"])
    ex = pd.read_csv(out / "recovery_extras.csv", float_precision="round_trip", keep_default_na=False,
                     na_values=[""]) if code == 0 else pd.DataFrame()
    fit = pd.read_csv(out / "fits" / ("twin_%d" % tf) / "phase2_results.csv", float_precision="round_trip")
    fr = fit.iloc[0]
    parts = {"twin exits 0, status ok (%.1f s)" % secs: r.returncode == 0 and stj.get("status") == "ok",
             "the fit ran with --refine trf --refine-max-nfev %d" % V5_MAX_NFEV:
             argv[argv.index("--refine") + 1] == "trf"
             and argv[argv.index("--refine-max-nfev") + 1] == str(V5_MAX_NFEV),
             "the fit's row: refine trf": str(fr.get("refine")) == "trf",
             "merge exits 0": code == 0}
    cols = list(ex.columns)
    want = list(T.EXTRAS_COLUMNS) + ["on_bound_" + n for n in IH6]
    parts["recovery_extras.csv: the Block 28 columns after the Block 22 ones, then on_bound_<six names>"] = (
        cols == want and cols[cols.index("dvh_est_mV") + 1:cols.index("dvh_est_mV") + 7] == list(B28_EXTRAS))
    if code == 0 and len(ex):
        e = ex[ex["twin_id"] == tf].iloc[0]

        def same(a, b):
            a, b = float(a), float(b)
            return (np.isnan(a) and np.isnan(b)) or a == b
        parts["the L_2 columns are the fit's (sqrt_l2_at_truth = sqrt_l2_at_eval_theta), finite"] = (
            all(same(e[c], fr[c]) for c in ("l2_after_search", "l2_after_refine", "refine_n_converged",
                                             "refine_n_failed", "sqrt_l2_at_estimate"))
            and same(e["sqrt_l2_at_truth"], fr["sqrt_l2_at_eval_theta"])
            and all(np.isfinite(float(e[c])) for c in B28_EXTRAS))
        flags = {n: ("" if pd.isna(e["on_bound_" + n]) else str(e["on_bound_" + n])) for n in IH6}
        fit_flags = {n: ("" if pd.isna(fr["on_bound_" + n]) else str(fr["on_bound_" + n])) for n in IH6}
        parts["on_bound_<name>: the fit's flags, each in {'', lo, hi}"] = (
            flags == fit_flags and all(v in ("", "lo", "hi") for v in flags.values()))
        print("\n  [V5] twin %d: %.1f s; l2 %r -> %r, converged %r, failed %r, sqrt_l2 est %r truth %r; flags %s"
              % (tf, secs, e["l2_after_search"], e["l2_after_refine"], e["refine_n_converged"], e["refine_n_failed"],
                 e["sqrt_l2_at_estimate"], e["sqrt_l2_at_truth"], flags))
    # a twin of an arm without some names (passive_fit): the names it does not fit are ''
    tp = st["tid"]["passive_fit"]
    fake = fit.copy()
    fake["specimen_id"] = tp
    fake["param_names"] = "Cm|Rm|Ra"
    fake["on_bound_Cm"] = "lo"
    frames = {tf: T.load_results_round_trip(out / "fits" / ("twin_%d" % tf) / "phase2_results.csv", "ih6_d207"),
              tp: fake}
    ex2 = T.recovery_extras(st["man"], frames, st["cfg"])
    e2 = ex2[ex2["twin_id"] == tp].iloc[0]
    parts["a passive_fit twin: on_bound_ of gbar, dv_h, kappa_tau ''; its own flags kept"] = (
        all(e2["on_bound_" + n] == "" for n in ("gbar", "dv_h", "kappa_tau")) and e2["on_bound_Cm"] == "lo")
    _fail_parts(parts)


def _ensure_v5():
    if "v5_meta_mtime" not in _S:
        test_V5_twin_trf()


def test_V2_sensitivity():
    import identifiability as IDF
    import run_ih_fit as R
    import run_twin_recovery as T
    import trace_sensitivity as TS
    _ensure_v5()
    st = _run_dir()
    out, tf, t0_ = st["out"], st["tid"]["full"], st["tid"]["full_no_noise"]
    man = st["man"]
    rf = man[man["specimen_id"] == tf].iloc[0]
    r0 = man[man["specimen_id"] == t0_].iloc[0]
    parts = {"the noise-free twin copies the full twin's real cell and truth (the oracle's premise)":
             int(r0["twin_of_specimen_id"]) == int(rf["twin_of_specimen_id"])
             and all(float(r0[c]) == float(rf[c]) for c in ("cm_true", "rm_true", "ra_true", "ih_gihbar_S_cm2",
                                                             "ih_dvh_mV", "ih_kappa_tau"))}
    runs = {}
    for tid in (tf, t0_):
        r, secs = _sens(tid)
        runs[tid] = r
        d = _sdir(tid)
        files = ["sensitivity_%d.npz" % tid, "sensitivity_%d.json" % tid, "bundles_%d.npz" % tid,
                 "identifiability_%d.csv" % tid, "truth_trace_check_%d.csv" % tid, "identifiability.csv"]
        parts["twin %d: sensitivity exits 0 (%.1f s), its files exist" % (tid, secs)] = (
            r.returncode == 0 and all((d / f).is_file() for f in files))
        if r.returncode != 0:
            print("\n  [V2] twin %d stderr:\n%s" % (tid, r.stderr[-3000:]))
    parts["the full twin's archive reused, not rebuilt (metadata.json unchanged)"] = (
        os.stat(out / "twins" / ("twin_%d" % tf) / ("specimen_%d" % tf) / "metadata.json").st_mtime_ns
        == _S["v5_meta_mtime"])
    _fail_parts(parts)

    tab, sens, meta, z = {}, {}, {}, {}
    for tid in (tf, t0_):
        d = _sdir(tid)
        tab[tid] = pd.read_csv(d / "identifiability.csv", float_precision="round_trip")
        base = pd.read_csv(d / ("identifiability_%d.csv" % tid), float_precision="round_trip")
        sens[tid] = TS.load_sensitivities(d / ("sensitivity_%d.npz" % tid))
        meta[tid] = R._read_meta(d / ("sensitivity_%d.json" % tid))
        z[tid] = R._read_bundles_npz(d / ("bundles_%d.npz" % tid))
        t = tab[tid]
        names = list(meta[tid]["names"])
        sd_cols = ["sd_lin_" + n for n in names]
        corr_cols = ["corr_%s_%s" % (names[i], names[k]) for i in range(len(names)) for k in range(i + 1, len(names))]
        parts["twin %d: rows W0, W1; twin_id, configuration, n_noise %d" % (tid, N_NOISE)] = (
            list(t["weighting"]) == ["W0", "W1"] and list(t.columns[:3]) == ["twin_id", "configuration", "n_noise"]
            and set(t["twin_id"]) == {tid} and set(t["n_noise"]) == {N_NOISE})
        parts["twin %d: sd_lin_<name> finite and > 0 for the six names, both weightings; corr in [-1, 1]" % tid] = (
            names == list(IH6) and all(c in t.columns for c in sd_cols + corr_cols)
            and bool(np.all(np.isfinite(t[sd_cols].to_numpy())) and np.all(t[sd_cols].to_numpy() > 0))
            and bool(np.all(np.abs(t[corr_cols].to_numpy()) <= 1.0 + 1e-12)))
        shared = [c for c in base.columns]
        parts["twin %d: identifiability.csv = identifiability_%d.csv on its columns" % (tid, tid)] = all(
            np.array_equal(t[c].to_numpy(), base[c].to_numpy()) if t[c].dtype.kind in "biuO" else
            np.allclose(t[c].to_numpy(dtype=float), base[c].to_numpy(dtype=float), rtol=0, atol=0, equal_nan=True)
            for c in shared)
        for _, row in t.iterrows():
            print("\n  [V2] twin %d (%s) %s: %s" % (tid, row["configuration"], row["weighting"],
                                                    ", ".join("%s %.6g" % (c[7:], row[c]) for c in sd_cols)), end="")
    print()

    # the reason the two tables differ, and that it is the only one: S identical; the noise-free twin's S
    # with the noisy twin's layouts (a_{b,s}, D_b of its recorded traces) and seeds gives the noisy twin's sd_lin
    s_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(sens[tf]["S"], sens[t0_]["S"]))
    s_max = max(float(np.max(np.abs(a))) for a in sens[tf]["S"])
    parts["the sensitivities S of the two twins identical (max |dS| %.3g, max |S| %.3g)" % (s_diff, s_max)] = (
        s_diff <= REL_TOL * s_max)
    prot, table = _protocol_table(int(rf["twin_of_specimen_id"]))
    bundles = R.stored_bundles(z[tf])
    brief = [bool(x) for x in np.asarray(z[tf]["is_brief"], dtype=bool)]
    for tag, tau in (("W0", None), ("W1", R.SENS_W1_TAU_D_MS)):
        lay, tix = R.layout_from_stored(z[tf], meta[tf], ls_tau_d_ms=tau)
        J0 = IDF.jacobian_from_sensitivities([sens[t0_]["S"][i] for i in tix], lay)
        Rr = np.asarray([IDF.noise_residuals(lay, n) for n in T.noise_realisations(
            bundles, brief, tix, prot, table, int(rf["twin_of_specimen_id"]), twin_id=tf, n_noise=N_NOISE)])
        sd0 = IDF.predicted_scatter(J0, Rr)["sd"]
        sdf = tab[tf].loc[tab[tf]["weighting"] == tag, ["sd_lin_" + n for n in IH6]].to_numpy()[0]
        rel = float(np.max(np.abs(sd0 - sdf) / sdf))
        parts["%s: the noise-free twin's S with the noisy twin's layout and seeds -> its sd_lin (max rel %.2g)"
              % (tag, rel)] = rel <= REL_TOL
        sdn = tab[t0_].loc[tab[t0_]["weighting"] == tag, ["sd_lin_" + n for n in IH6]].to_numpy()[0]
        _S.setdefault("v2_as_built", []).append((tag, float(np.max(np.abs(sdn - sdf) / sdf))))
        # the two causes apart (printed, not gated): the seeds alone (the noisy twin's layout, the noise-free
        # twin's seeds) and the layout alone (the noise-free twin's layout, the noisy twin's seeds)
        lay0, tix0 = R.layout_from_stored(z[t0_], meta[t0_], ls_tau_d_ms=tau)
        J00 = IDF.jacobian_from_sensitivities([sens[t0_]["S"][i] for i in tix0], lay0)
        b0 = R.stored_bundles(z[t0_])
        br0 = [bool(x) for x in np.asarray(z[t0_]["is_brief"], dtype=bool)]
        Rs = np.asarray([IDF.noise_residuals(lay, n) for n in T.noise_realisations(
            bundles, brief, tix, prot, table, int(rf["twin_of_specimen_id"]), twin_id=t0_, n_noise=N_NOISE)])
        Rl = np.asarray([IDF.noise_residuals(lay0, n) for n in T.noise_realisations(
            b0, br0, tix0, prot, table, int(rf["twin_of_specimen_id"]), twin_id=tf, n_noise=N_NOISE)])
        seeds_only = float(np.max(np.abs(IDF.predicted_scatter(J0, Rs)["sd"] - sdf) / sdf))
        layout_only = float(np.max(np.abs(IDF.predicted_scatter(J00, Rl)["sd"] - sdf) / sdf))
        print("\n  [V2] %s: max |dS| %.3g (max |S| %.3g); shared seeds and layout: max rel %.2g; as built: %.3g; "
              "seeds alone %.3g; layout alone %.3g; ||J0 - Jf|| / ||Jf|| %.3g"
              % (tag, s_diff, s_max, rel, _S["v2_as_built"][-1][1], seeds_only, layout_only,
                 IDF.relative_difference(J00, J0)))
    _fail_parts(parts)
    # amendment 2 (D-249): the difference as built is reported, not gated; it is the PASS line's detail
    return ("sd_lin as built, noise-free twin vs its partner (own seeds and layouts), max relative difference: "
            + ", ".join("%s %.3g" % kv for kv in _S["v2_as_built"]))


def test_V3_bundle_noise():
    import twin_noise as TN
    import twin_table as TT
    import run_ih_fit as R
    import run_twin_recovery as T
    if "sens_%d" % _run_dir()["tid"]["full"] not in _S:
        test_V2_sensitivity()
    st = _run_dir()
    tf = st["tid"]["full"]
    real = int(st["man"].loc[st["man"]["specimen_id"] == tf, "twin_of_specimen_id"].iloc[0])
    prot, table = _protocol_table(real)
    z = R._read_bundles_npz(_sdir(tf) / ("bundles_%d.npz" % tf))
    bundles = R.stored_bundles(z)
    brief = [bool(x) for x in np.asarray(z["is_brief"], dtype=bool)]
    bs = 1000007
    parts = {}
    for i, (b, br) in enumerate(zip(bundles, brief)):
        a1 = T.bundle_noise(b, br, prot, table, real, twin_id=tf, base_seed=bs)
        a2 = T.bundle_noise(b, br, prot, table, real, twin_id=tf, base_seed=bs, ss_cache={})
        parts["bundle %d (%s %+.0f pA): length %d = len(t); two calls identical" % (
            i, "brief" if br else "long", b.amplitude_pA, b.t.size)] = (
            a1.shape == b.t.shape and a1.dtype == np.float64 and np.array_equal(a1, a2))
        if not br:
            assert len(b.sweep_numbers) == 1, "the fixture's long bundles are one sweep each"
            s = int(b.sweep_numbers[0])
            law, tilt = TT.sweep_law(table, real, "LS", s)
            ref = TN.ls_sweep_noise(law, tilt, 1.0 / float(prot.ls_fs_Hz), int(b.t.size), base_seed=bs, twin_id=tf,
                                    sweep_number=s)
            parts["bundle %d: one-sweep long bundle = ls_sweep_noise of sweep %d (exact)" % (i, s)] = (
                np.array_equal(a1, ref))
        else:
            dep = b.amplitude_pA > 0
            pol = np.asarray(prot.ss_polarity_is_dep, dtype=bool)
            sw = np.asarray(prot.ss_sweep_number)
            rows = []
            for s in np.unique(sw):
                law, tilt = TT.sweep_law(table, real, "SS", int(s))
                m = sw == s
                full = TN.ss_sweep_noise(law, tilt, 1.0 / float(prot.ss_fs_Hz), int(b.t.size),
                                         np.asarray(prot.ss_position)[m], int(prot.ss_period_samples), base_seed=bs,
                                         twin_id=tf, sweep_number=int(s))
                rows.append(full[pol[m] == dep])
            rows = np.concatenate(rows, axis=0)
            ref = rows.mean(axis=0)
            sd_b, sd_w = float(np.std(a1)), float(np.std(rows[0]))
            parts["bundle %d: brief = the mean of its %d polarity rows (1e-15 mV); SD %.4g < one window's %.4g"
                  % (i, rows.shape[0], sd_b, sd_w)] = (
                rows.shape[0] == b.n_repeats_averaged and np.max(np.abs(a1 - ref)) <= 1e-15 and sd_b < sd_w)
    # two sweeps in one long bundle: the mean of the two sweeps' draws (the fixture has none: built here on
    # two stored sweeps of equal length, their amplitude irrelevant to the noise)
    longs = [b for b, br in zip(bundles, brief) if not br]
    two = SimpleNamespace(t=longs[0].t, amplitude_pA=-1.0, n_repeats_averaged=2,
                          sweep_numbers=[int(longs[0].sweep_numbers[0]), int(longs[1].sweep_numbers[0])])
    a = T.bundle_noise(two, False, prot, table, real, twin_id=tf, base_seed=bs)
    one = [T.bundle_noise(SimpleNamespace(t=longs[0].t, amplitude_pA=-1.0, n_repeats_averaged=1, sweep_numbers=[s]),
                          False, prot, table, real, twin_id=tf, base_seed=bs) for s in two.sweep_numbers]
    parts["a two-sweep long bundle = the mean of its sweeps' draws"] = np.array_equal(a, (one[0] + one[1]) / 2.0)
    # the entry's STOP case: a bundle grid that is not the stored one
    short = SimpleNamespace(t=longs[0].t[:-1], amplitude_pA=-1.0, n_repeats_averaged=1,
                            sweep_numbers=list(longs[0].sweep_numbers))
    bb = [b for b, br in zip(bundles, brief) if br][0]
    shifted = SimpleNamespace(t=bb.t + 1e-6, amplitude_pA=bb.amplitude_pA, n_repeats_averaged=bb.n_repeats_averaged,
                              sweep_numbers=list(bb.sweep_numbers))
    # amendment 4 (D-260): a brief bundle that names only some of its polarity's sweeps (the loader's
    # mis-indexed sweep_numbers; run tw5) is accepted and gives the same noise; a wrong window COUNT stops
    subset = SimpleNamespace(t=bb.t, amplitude_pA=bb.amplitude_pA, n_repeats_averaged=bb.n_repeats_averaged,
                             sweep_numbers=list(bb.sweep_numbers)[:1])
    parts["brief, a subset of its sweeps named: same noise"] = np.array_equal(
        T.bundle_noise(subset, True, prot, table, real, twin_id=tf, base_seed=bs),
        T.bundle_noise(bb, True, prot, table, real, twin_id=tf, base_seed=bs))
    miscount = SimpleNamespace(t=bb.t, amplitude_pA=bb.amplitude_pA,
                               n_repeats_averaged=int(bb.n_repeats_averaged) - 1,
                               sweep_numbers=list(bb.sweep_numbers))
    for lab, bad, isb in (("long, one sample short", short, False), ("brief, times shifted", shifted, True),
                          ("brief, one window fewer", miscount, True)):
        try:
            T.bundle_noise(bad, isb, prot, table, real, twin_id=tf, base_seed=bs)
            parts["%s: BundleGridError" % lab] = False
        except T.BundleGridError:
            parts["%s: BundleGridError" % lab] = True
    print("\n  [V3] " + "\n  [V3] ".join("%s: %s" % (k, "ok" if v else "FAILED") for k, v in parts.items()))
    _fail_parts(parts)


def test_V4_merge():
    if "sens_%d" % _run_dir()["tid"]["full_no_noise"] not in _S:
        test_V2_sensitivity()
    import run_ih_fit as R
    st = _run_dir()
    out = st["out"]
    tids = sorted((st["tid"]["full"], st["tid"]["full_no_noise"]))
    code, so, se = STR._run(["merge", "--out", out, "--no-plot"])
    idf = pd.read_csv(out / "identifiability.csv") if (out / "identifiability.csv").is_file() else None
    ttc = pd.read_csv(out / "truth_trace_check.csv") if (out / "truth_trace_check.csv").is_file() else None
    parts0 = {"no <group> level under the run directory (amendment 2)": not (out / st["cfg"]["group"]).exists()
              and (out / "sensitivity").is_dir()}
    nb = {t: int(np.asarray(R._read_bundles_npz(_sdir(t) / ("bundles_%d.npz" % t))["is_train"]).size) for t in tids}
    parts = {"merge exits 0": code == 0, **parts0,
             "identifiability.csv: twin_ids %s, rows W0 and W1 for each" % tids:
             idf is not None and sorted(set(int(x) for x in idf["twin_id"])) == tids
             and all(list(idf.loc[idf["twin_id"] == t, "weighting"]) == ["W0", "W1"] for t in tids),
             "truth_trace_check.csv: twin_id first, one row per bundle per twin (%s)" % nb:
             ttc is not None and ttc.columns[0] == "twin_id" and sorted(set(int(x) for x in ttc["twin_id"])) == tids
             and all(int((ttc["twin_id"] == t).sum()) == nb[t] for t in tids)
             and bool(np.all(ttc["twin_id"] == ttc["specimen_id"]))}
    # merge on a run with no fit and no sensitivity: exit 1 as before; with sensitivity only: exit 0
    out2 = STR._new_out("v4empty")
    STR._plan(out2)
    c2, _s, _e = STR._run(["merge", "--out", out2, "--no-plot"])
    parts["merge with neither results nor sensitivity files: exit 1 (unchanged)"] = c2 == 1
    if ttc is not None:
        print("\n  [V4] truth_trace_check max_abs_mV per twin: %s" % {
            int(t): float(ttc.loc[ttc["twin_id"] == t, "max_abs_mV"].max()) for t in tids})
    _fail_parts(parts)


def test_V8_verdict_sqrt_l2():
    import run_twin_recovery as T
    # the entry's hand case, built from per-bundle ratios RMSD_b / D_b with c_b = 0.5 (2.10 (a)):
    # L = sum c_b x_b, L_2 = sum c_b x_b^2; estimate x = (0.4 / sqrt(2), 0), truth x = (sqrt(0.03), sqrt(0.03))
    c = np.array([0.5, 0.5])
    x_est, x_tru = np.array([0.4 / np.sqrt(2.0), 0.0]), np.array([np.sqrt(0.03), np.sqrt(0.03)])
    L_e, L2_e = float(c @ x_est), float(c @ x_est ** 2)
    L_t, L2_t = float(c @ x_tru), float(c @ x_tru ** 2)
    nan = float("nan")
    parts = {"hand case values: L_est %.4f, L2_est %.4f, L_truth %.4f, L2_truth %.4f" % (L_e, L2_e, L_t, L2_t): (
        abs(L_e - 0.1414) < 5e-5 and abs(L2_e - 0.04) < 1e-12 and abs(L_t - 0.1732) < 5e-5
        and abs(L2_t - 0.03) < 1e-12)}
    parts["sqrt_l2 finite: ('optimiser', 'sqrt_l2')"] = (
        T.verdict_pair(L_t, L_e, np.sqrt(L2_t), np.sqrt(L2_e)) == ("optimiser", "sqrt_l2"))
    parts["the same pair on L alone: 'data'"] = T.loss_verdict(L_t, L_e) == "data"
    parts["sqrt_l2 nan (either or both): the verdict on L, ('data', 'loss')"] = all(
        T.verdict_pair(L_t, L_e, a, b) == ("data", "loss") for a, b in ((nan, nan), (nan, np.sqrt(L2_e)),
                                                                        (np.sqrt(L2_t), nan)))
    parts["a failed point's sqrt(L_2) = sqrt(1e6): ('na', 'sqrt_l2')"] = (
        T.verdict_pair(L_t, L_e, 1e3, np.sqrt(L2_e)) == ("na", "sqrt_l2"))
    parts["a tie within 2 % on sqrt_l2"] = T.verdict_pair(L_t, L_e, 0.1, 0.101) == ("tie", "sqrt_l2")
    # on V5's run: the trf twin's row
    _ensure_v5()
    st = _run_dir()
    out, tf = st["out"], st["tid"]["full"]
    code, _so, _se = STR._run(["merge", "--out", out, "--no-plot"])
    ex = pd.read_csv(out / "recovery_extras.csv", float_precision="round_trip")
    fr = pd.read_csv(out / "fits" / ("twin_%d" % tf) / "phase2_results.csv", float_precision="round_trip").iloc[0]
    e = ex[ex["twin_id"] == tf].iloc[0]
    want = T.loss_verdict(float(fr["sqrt_l2_at_eval_theta"]), float(fr["sqrt_l2_at_estimate"]),
                          penalty=float(np.sqrt(1e6)))
    parts["V5's trf twin: loss_verdict_on 'sqrt_l2' after loss_verdict, verdict %r = the one on the fit's "
          "sqrt_l2 columns" % (e["loss_verdict"],)] = (
        code == 0 and list(ex.columns).index("loss_verdict_on") == list(ex.columns).index("loss_verdict") + 1
        and e["loss_verdict_on"] == "sqrt_l2" and e["loss_verdict"] == want)
    print("\n  [V8] " + "\n  [V8] ".join(parts))
    _fail_parts(parts)


def _stubs(tmp):
    """smoke_twin_recovery's stubs, with the python stub also recording `sensitivity`."""
    stubs = STR._stubs(tmp)
    py = stubs / "bin" / "python"
    py.write_text(py.read_text().replace("    twin|merge)\n", "    twin|merge|sensitivity)\n"))
    return stubs


def test_V6_submitter():
    import run_twin_recovery as T
    st = STR._fixture()
    script = HERE / "submit_twin_recovery.sh"
    tmp = st["root"] / "v6"
    tmp.mkdir()
    stubs = _stubs(tmp)
    home, tmpdir = tmp / "home", tmp / "tmpdir"
    home.mkdir()
    tmpdir.mkdir()
    pylog = tmp / "py.log"

    def env(**extra):
        e = {k: v for k, v in os.environ.items()
             if not (k.startswith(("TWIN_", "PBS_", "ROADMAP_", "IH_", "PY_STUB")) or k in (
                 "RUN_TAG", "NOISE_TAG", "GROUP", "TWINS", "MAX_TWINS", "NCPUS", "WALLTIME", "CONDA_ENV", "DRY_RUN",
                 "PLAN_ARGS", "MODE", "N_NOISE"))}
        e.update({"PATH": "%s:%s" % (stubs / "bin", e.get("PATH", "")), "HOME": str(home), "TMPDIR": str(tmpdir),
                  "FAKE_CONDA_BASE": str(stubs / "condabase"), "REAL_PYTHON": sys.executable,
                  "QSUB_LOG": str(tmp / "qsub.log"), "PY_LOG": str(pylog), "IH_CODE_DIR": str(HERE),
                  "IH_ARCHIVE_ROOT": str(st["archive_root"]), "GROUP": STR.GROUP, "NOISE_TAG": "nfx",
                  "PLAN_ARGS": " ".join(STR.SMALL + ["--refine-max-nfev", "4"]), "ROADMAP_DISABLE": "1"})
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def sh(e):
        return subprocess.run(["bash", str(script)], cwd=str(tmp), env=e, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=900)
    parser = T.build_parser()

    def parsed(line, marker):
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
    out_dir = st["output_base"] / "twin_recovery" / "v6" / STR.GROUP
    parts = {}
    for mode in ("sensitivity", ""):
        extra = dict(RUN_TAG="v6", DRY_RUN=1, NCPUS=2)
        if mode:
            extra.update(MODE=mode, N_NOISE=7)
        p = sh(env(**extra))
        lines = p.stdout.splitlines()
        sub = [parsed(ln, "DRY RUN:   subjob %d:" % k) for k, ln in enumerate(
            [x for x in lines if "DRY RUN:   subjob " in x])]
        mrg = [parsed(ln, "DRY RUN:   merge:") for ln in lines if "DRY RUN:   merge:" in ln]
        plan = [ln for ln in lines if "] plan" in ln and " plan " in ln]
        if mode:
            ok = (len(sub) == 4 and all(a is not None and a.command == "sensitivity" and a.n_workers == 2
                                        and a.n_noise == 7 and Path(a.run_dir) == out_dir for a in sub)
                  and sorted(a.twin for a in sub) == sorted(int(t) for t in STR.TWIN_IDS))
            label = "DRY_RUN=1 MODE=sensitivity N_NOISE=7 NCPUS=2: four sensitivity commands (--n-workers 2 --n-noise 7)"
        else:
            ok = (len(sub) == 4 and all(a is not None and a.command == "twin" and a.n_workers == 2
                                        and Path(a.out) == out_dir for a in sub))
            label = "DRY_RUN=1, MODE unset: four twin commands as before"
        parts[label + ", the merge, exit 0"] = (p.returncode == 0 and ok and len(mrg) == 1 and mrg[0] is not None
                                                and mrg[0].command == "merge")
        parts["%s: the plan command carries PLAN_ARGS' --refine-max-nfev 4" % (mode or "MODE unset")] = (
            len(plan) == 1 and "--refine-max-nfev 4" in plan[0])
        if mode:
            print("\n  [V6] " + "\n  [V6] ".join(ln for ln in lines if "subjob 0" in ln or "mode:" in ln))
        if p.returncode != 0:
            print("\n  [V6] stderr: %s" % p.stderr[-2000:])
    for bad, lab in ((dict(MODE="bogus"), "MODE=bogus"), (dict(MODE="sensitivity", N_NOISE=1), "N_NOISE=1")):
        p = sh(env(RUN_TAG="v6", DRY_RUN=1, **bad))
        parts["%s refused (exit 2)" % lab] = p.returncode == 2
    # a job of the array: the run file says TWIN_MODE=sensitivity
    rf = tmp / "run.env"
    rf.write_text("TWIN_CODE_DIR=%s\nTWIN_OUT_DIR=%s\nTWIN_GROUP=FIX\nTWIN_NCPUS=3\nTWIN_CONDA_ENV=prova\n"
                  "TWIN_MODE=sensitivity\nTWIN_N_NOISE=11\nTWIN_IDS=( 900000000 900000300 )\n"
                  % (shlex.quote(str(HERE)), shlex.quote(str(tmp / "jobout"))))
    if pylog.exists():
        pylog.unlink()
    p = subprocess.run(["bash", str(script)], cwd=str(tmp), env=env(TWIN_ROLE="twin", TWIN_INDEX=1,
                                                                    TWIN_RUN_FILE=rf),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=300)
    log = pylog.read_text() if pylog.exists() else ""
    parts["a job with TWIN_MODE=sensitivity runs `sensitivity --run-dir ... --twin 900000300 --n-workers 3 "
          "--n-noise 11`, threads 1"] = (
        p.returncode == 0 and "[sensitivity] [--run-dir] [%s] [--twin] [900000300] [--n-workers] [3] [--n-noise] [11]"
        % (tmp / "jobout") in log and "ENV[1/1/1]" in log
        and (tmp / "jobout" / "logs").is_dir()
        and any(f.name.startswith("sensitivity_900000300_") for f in (tmp / "jobout" / "logs").iterdir()))
    _fail_parts(parts)


def test_V7_ascii_lf_bash():
    bad = [n for n in OWN_FILES if (b"\r" in (HERE / n).read_bytes() or any(b > 127 for b in (HERE / n).read_bytes()))]
    r = subprocess.run(["bash", "-n", str(HERE / "submit_twin_recovery.sh")], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True)
    text = (HERE / "submit_twin_recovery.sh").read_text()
    markers = ("export ROADMAP_STAGE=S7", "export ROADMAP_N_TASKS=\"$TASKS\"",
               "export ROADMAP_SCRIPT=\"Passive Features/HPC script/Ih Fit/submit_twin_recovery.sh\"",
               ". \"$REPO_ROOT/specs/roadmap/hpc_status.sh\"", "roadmap_job_start", "trap 'roadmap_job_end $?' EXIT",
               "JOB_NAME_TWIN=twin_recovery   # roadmap allen-mtg-fit S7")
    missing = [m for m in markers if m not in text]
    assert r.returncode == 0 and not bad and not missing, "bash -n %d %s; not ASCII/LF: %s; markers missing: %s" % (
        r.returncode, r.stderr, bad, missing)


def test_convergence():
    raise unittest.SkipTest("not a discretisation: wiring of Blocks 10, 26 and 27; the finite-difference step "
                            "is checked by Block 27's fd_step_check")


# ---------------------------------------------------------------- runner ---
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true", help="run `nrnivmodl mod` when ./x86_64/special is missing")
    args = ap.parse_args(argv)
    if args.build and not (HERE / "x86_64" / "special").exists():
        subprocess.run(["nrnivmodl", "mod"], cwd=str(HERE), check=True)
    checks = [(name, obj) for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    print("== %s" % Path(__file__).name)
    print(STR._environment())
    results = []
    t_all = time.perf_counter()
    for name, func in checks:
        start = time.perf_counter()
        try:
            ret = func()
            status, detail = "PASS", (ret if isinstance(ret, str) else "")
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
        print(("%-5s  %-*s  %8.3fs  %s" % (status, width, name, seconds, headline[:400])).rstrip())
    for name, status, _seconds, detail in results:
        if status in ("FAIL", "ERROR"):
            print("\n---- %s: %s\n%s" % (status, name, detail.strip()))
    counts = {s: sum(1 for r in results if r[1] == s) for s in ("PASS", "FAIL", "ERROR", "TODO", "SKIP")}
    print("\n-- " + ", ".join("%d %s" % (n, s.lower()) for s, n in counts.items())
          + "; %.1f s in all" % (time.perf_counter() - t_all))
    return 1 if counts["FAIL"] + counts["ERROR"] + counts["TODO"] else 0


if __name__ == "__main__":
    sys.exit(main())
