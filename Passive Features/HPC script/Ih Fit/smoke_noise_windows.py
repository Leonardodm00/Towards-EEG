"""
smoke_noise_windows.py -- smoke suite for noise_windows and noise_cell_stats (W1-W15)
=====================================================================================

Block 3 of specs/SPEC.md (D-051, D-072, D-073, D-076; the window start and
the subtraction D-095, D-103, D-104, D-106; the per-sweep tilts D-102,
D-107; the tilt offsets removed before the median and the SS tilt read at
the 80 ms floor, D-115). Builds writer-like archive cells in a temporary folder (two SS
sweeps of 37 pulses at the real onset period, with an RC pulse response and
Block-2 noise, eleven pulses dropped by a planted QC table; six Long Square
sweeps) so that every check has a known answer: the base cell (0.5 s LS
tail), a tilted cell (per-sweep tilts at sigma_b = 0.1 mV/s, 1.5 s LS tail)
and a cell whose sweep 93 keeps three hyp windows. About a minute on one
core; W4's oracle imports the monolith (NEURON) and W12's imports
ls_baseline_qc (pandas); each is reported SKIPPED when its import fails.

  W1  layout (D-072): the stored windows come back with the planted positions,
      sweep numbers and polarities and the period 10025 exactly; a QC table
      with a wrong row count, a swapped polarity, a non-alternating train and a
      wrong configured period each raise LayoutError; a false head rejected
  W2  start (D-103, D-104, D-110, D-111): on noise-free planted groups
      (B + A exp(-t/tau) for t >= 0, white noise of a known SD before t = 0)
      A, tau, B come back to 1e-9 relative, sigma_SS_hat is the median
      pre-pulse SD (1e-15) and t_sp lies within one sample of tau ln(A /
      (kappa sigma_SS_hat)); a fast decay floors at 80 ms (ss_start_floor);
      groups at 100, 120, 140, 160 ms give 130 ms (the median); all at 151 ms
      fall back to 80 ms (ss_start_fallback, short_window), all at 149 ms are
      kept; a ramp fails every fit at the upper bound (no_valid_group, all
      windows kept); a fit double failing the first group leaves it out of
      the median with its windows kept; on the base cell's noisy windows every
      fit is valid, tau_hat within 25 % of the RC 28.143 ms, and exact:
      stationary (ST0), the derivative formula = finite differences,
      curve_fit started there stays, no dense-grid point lower
  W3  subtraction (D-103.1): the per-sample median of the subtracted windows
      is 0 per group (1e-12); the medians equal numpy.median per group from an
      independent loop (exactly); a different common vector added to each
      group leaves X_tilde unchanged (1e-12) and the statistics of response
      + noise after the subtraction equal those of the noise alone (1e-10)
  W4  statistics: window_stats equals np.std(ddof=1) and sample_acf per
      window (1e-14); the 15 labels, blocks, n and K of SPEC 2.2; the LS
      pre-window equals noise_calibration.pre_window_samples(mono, bundle) on
      the same sweep through the monolith (skipped without NEURON)
  W5  per-sweep vectors through ss_noise_windows: keys are the sweep numbers,
      each the median over that sweep's kept windows; the cell vector is the
      median over all kept windows
  W6  bootstrap SE: K = 60 iid Gaussian values, B = 2000, 40 repeats: the mean
      SE within 20 % of 1.2533 sigma / sqrt(K); bit-identical for one seed
  W7  [legacy (T1)] semivariogram: 400 sweeps simulated at a known law with
      the fixture's layout: the pair count is the layout's, and V_hat equals
      noise_law.semivariogram(law, dt, m, m, 4 P) within 3.5 SE
  W8  enlarged window: m = n_w - pre_n - round(t_SS f_s / 1e3), first sample at
      t >= t_SS, ends 0.5 ms before the next onset
  W9  LS: 13250-sample pre-windows (265 ms at 50 kHz), hyp and dep both read,
      the 0 pA sweep skipped, segments of 2500 and 6625 samples; end windows of
      25000 samples after the step with the 1.5 s tail ('ok'), 'tail_short'
      with the 0.5 s tail
  W10 ASCII and LF on noise_windows.py, noise_cell_stats.py and this file
  W11 SS tilt (TL1), the reading of D-115.2 (ss_noise_windows' tilts, on the
      kept windows at the 80 ms floor): equals the within-polarity closed form
      (1e-12 mV/s), on the floor windows with and without each group's median
      subtracted alike; on the tilted cell within 4 exact SDs of each planted
      tilt; one polarity takes one intercept; two windows give NaN with
      ss_tilt_none
  W12 LS tilt (TL2): equals ls_baseline_qc.baseline_record's delta_end_mV over
      the distance of its window centres (1e-12 mV/s); on the tilted cell
      within 4 exact SDs of each planted tilt; with the 0.5 s tail every tilt
      is NaN with ls_tilt_none:<sweep>:tail_short, as the oracle's flag
  W13 small groups (D-106): sweep 93 with 3 hyp windows: flagged few_windows,
      absent from the kept windows, X_tilde, the statistic vector's K, sweep
      93's per-sweep vector and its tilt reading; sigma_SS_hat over the kept
      windows
  W14 [legacy] transient_start (noise-free time exact, floor 80 ms,
      ss_enlarged_none at 190 ms) and subtract_mean_response (zero mean per
      polarity) as at commit 1aabe3e
  W15 the offsets (D-115): on a noise-free cell without response, with a
      planted tilt per sweep, read_ss_tilts returns each tilt to 1e-9
      relative and ss_noise_windows' X_tilde is 0 (1e-12) where the plain
      subtraction of the uncorrected windows leaves the offsets (> 0.01 mV);
      on the noisy tilted cell the tilts are read_ss_tilts on the kept
      windows, the offsets equal an independent loop (1e-15), X_tilde an
      independent re-implementation (1e-12), the start that of decay_start on
      the corrected windows; a NaN tilt gives zero offsets

Run:  python smoke_noise_windows.py
Expected last line: "SMOKE noise_windows: 15/15 passed". Exit 1 on any FAIL.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import noise_law as nl                                   # noqa: E402
import noise_stats                                       # noqa: E402
from noise_windows import (DecayFit, LayoutError, WindowConfig, decay_start, enlarged_windows,  # noqa: E402
                           fit_decay, floor_index, keep_mask, load_ls_prewindows, load_ss_windows, ls_segments,
                           measure_period, profile_decay, read_ss_tilts, select_windows, ss_noise_windows,
                           subtract_mean_response, subtract_median_response, tilt_offsets, transient_start)
from noise_cell_stats import (LS_FULL_LAGS_MS, LS_LAST_LAGS_MS, SS_LAGS_MS, STAT_BLOCK,  # noqa: E402
                              STAT_LABELS, bootstrap_median_se, ls_sweep_tilts, ls_window_stats,
                              per_sweep_vectors, ss_sweep_tilts, ss_window_stats, statistic_vector,
                              window_mean_semivariogram, window_stats)

RESULTS = []
FS = 50000.0
DT = 1.0 / FS
PERIOD = 10025                     # samples, the archive's onset period (G12, 2026-10-20)
TRAIN = 37
PRE_N, POST_N = 500, 10000         # the writer's (-10, +200) ms window
PULSE_N = 25                       # 0.5 ms
REST_N = 25000                     # 0.5 s before the first pulse
TAIL_N = 11000
LS_PRE_N, LS_STEP_N = 13500, 50000  # LS sweep: 270 ms before the step, a 1 s step
LS_TAIL_N = 25000                  # 0.5 s after the step: the end window starts AT the step's end ('tail_short')
LS_LONG_TAIL_N = 75000             # 1.5 s after the step: the end window starts 1 s after it ('ok')
THETA_CF = nl.NoiseLaw(0.0380, 0.175, 0.0508, 0.0366)
SILENT = nl.NoiseLaw(0.0, 0.5, 0.0, 1.0)      # W15: no noise at all, only the planted tilts
R_IN_MOHM, TAU_M_MS = 113.0, 28.143
DROPPED = {92: (2, 3, 11, 13, 20, 29, 36), 93: (0, 5, 6, 27)}   # planted trace-QC drops
SMALL_KEEP_HYP_93 = (1, 17, 33)                                  # W13: sweep 93 keeps three hyp windows
SMALL_DROPPED = {92: DROPPED[92],
                 93: tuple(sorted(set(DROPPED[93]) | {p for p in range(1, TRAIN, 2) if p not in SMALL_KEEP_HYP_93}))}
Z_SE = 3.5
N_SD = 4.0                         # W11, W12: a planted tilt must lie within 4 exact SDs (two-sided 6e-5)
T_FLOOR_MS = 80.0                  # W11: the floor of the SS reading (D-115.2, SPEC 2.4), written out here


def report(name, ok, note=""):
    RESULTS.append((name, bool(ok), note))
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + note) if note else ""))


class SkipTest(Exception):
    pass


# ---------------------------------------------------------------------------
#  Fixture: a writer-like archive cell
# ---------------------------------------------------------------------------
def rc_response(i_pA, r_mohm=R_IN_MOHM, tau_ms=TAU_M_MS):
    a = np.exp(-DT / (tau_ms * 1e-3))
    return lfilter([r_mohm * 1e-3 * (1.0 - a)], [1.0, -a], i_pA)      # MOhm * pA = uV -> mV


def make_ss_sweep(law, rng, with_response=True, sigma_b=0.0):
    """One continuous SS sweep: (v, i, onsets, polarity list, noise, response,
    realised tilt b in mV/s); sigma_b (mV/s) plants a per-sweep tilt (0 = none)."""
    n = REST_N + (TRAIN - 1) * PERIOD + TAIL_N
    onsets = REST_N + PERIOD * np.arange(TRAIN)
    pol = [k % 2 == 0 for k in range(TRAIN)]                     # dep first, alternating
    i = np.zeros(n)
    for s, dep in zip(onsets, pol):
        i[s:s + PULSE_N] = 200.0 if dep else -200.0
    resp = rc_response(i) if with_response else np.zeros(n)
    units = nl.draw_units(rng, 1, n)
    noise = nl.simulate_sweep(law, DT, n, units, 0, sigma_b=sigma_b, sigma_a=0.0)
    return -60.0 + resp + noise, i, onsets, pol, noise, resp, float(sigma_b * units.tilt[0])


def write_fixture(d, rng, law=THETA_CF, with_response=True, dropped=DROPPED, sweeps=(92, 93),
                  ls_amps=(-110.0, -90.0, -70.0, 20.0, 50.0, 0.0), sigma_b=0.0, ls_tail_n=LS_TAIL_N):
    """Writes ss_pulses.npz (Format A), trace_qc_results.csv, metadata.json and
    ls_sweeps.npz as the Phase-0 writer does. Returns the planted truth.
    sigma_b (mV/s): a per-sweep tilt in the SS and LS sweeps, its realised
    values returned ('ss_tilt' per SS sweep, 'ls_tilt' per LS sweep);
    ls_tail_n: samples stored after the LS step (the default keeps the
    random stream of the cells written before 2026-10-29)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    t_win = (np.arange(PRE_N + POST_N) - PRE_N) / FS
    v_rows, i_rows, pol_rows, sw_rows, pos_rows, noise_rows, resp_rows, qc_rows = [], [], [], [], [], [], [], []
    ss_tilt = {}
    for sw in sweeps:
        v, i, onsets, pol, noise, resp, b = make_ss_sweep(law, rng, with_response, sigma_b)
        ss_tilt[int(sw)] = b
        drop = set(dropped.get(sw, ()))
        for k, (s, dep) in enumerate(zip(onsets, pol)):
            passed = k not in drop
            qc_rows.append((len(qc_rows), "dep" if dep else "hyp", 200.0 if dep else -200.0, passed))
            if passed:
                sl = slice(s - PRE_N, s + POST_N)
                v_rows.append(v[sl]); i_rows.append(i[sl]); noise_rows.append(noise[sl]); resp_rows.append(resp[sl])
                pol_rows.append(dep); sw_rows.append(sw); pos_rows.append(k)
    K = len(v_rows)
    np.savez_compressed(d / "ss_pulses.npz", t=t_win, v=np.array(v_rows), i_pA=np.array(i_rows),
                        polarity_is_dep=np.array(pol_rows, dtype=bool),
                        peak_pA=np.array([200.0 if p else -200.0 for p in pol_rows]),
                        stim_duration_s=np.full(K, PULSE_N / FS), sampling_rate_Hz=np.full(K, FS),
                        sweep_number=np.array(sw_rows, dtype=np.int64))
    with open(d / "trace_qc_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pulse_index", "polarity", "peak_pA", "amp_band_pA", "qced", "passed", "reason",
                    "p_combined", "p_min", "winning_scale_ms", "fit_A_mV", "fit_tau_ms", "fit_c_mV", "spike_ratio"])
        for idx, pol, peak, passed in qc_rows:
            w.writerow([idx, pol, "%g" % peak, "%g" % peak, "True", str(passed),
                        "ok" if passed else "pathological", "", "", "", "", "", "", ""])
    # Long Square sweeps: 270 ms pre, 1 s step, ls_tail_n samples after it
    n_pre, n_step, n_tail = LS_PRE_N, LS_STEP_N, int(ls_tail_n)
    n_ls = n_pre + n_step + n_tail
    units = nl.draw_units(rng, len(ls_amps), n_ls)
    ls_noise = nl.simulate_traces(law, DT, units.fast, units.slow, xi_tilt=units.tilt, sigma_b=sigma_b)
    ls_data, ls_info = {}, []
    for k, amp in enumerate(ls_amps):
        i = np.zeros(n_ls); i[n_pre:n_pre + n_step] = amp
        v = -60.0 + rc_response(i) + ls_noise[k]
        ls_data["v_%d" % k] = v; ls_data["i_%d" % k] = i
        ls_info.append({"index": k, "sweep_number": 54 + k, "detected_amplitude_pA": amp,
                        "sampling_rate_Hz": FS, "stimulus_name": "Long Square", "n_samples": n_ls})
    np.savez_compressed(d / "ls_sweeps.npz", **ls_data)
    meta = {"specimen_id": 900000003, "layer": "3", "dendrite_type": "spiny", "rin_MOhm": R_IN_MOHM,
            "tau_ms": TAU_M_MS, "v_rest_mV": -60.0, "ljp_correction_mV": 14.0,
            "ss_extraction": {"sweep_numbers": list(sweeps), "n_pulses_raw": len(qc_rows),
                              "n_pulses_ampdur_qc": len(qc_rows), "n_pulses_qc": K,
                              "n_dep": int(sum(pol_rows)), "n_hyp": int(K - sum(pol_rows)),
                              "pre_ms": 10.0, "post_ms": 200.0,
                              "trace_qc": {"enabled": True, "engine": "v2", "alpha": 0.5}},
            "ls_sweeps": ls_info}
    json.dump(meta, open(d / "metadata.json", "w"), indent=1)
    return dict(K=K, positions=np.array(pos_rows), sweeps=np.array(sw_rows), pol=np.array(pol_rows, dtype=bool),
                noise=np.array(noise_rows), resp=np.array(resp_rows), ls_noise=ls_noise, ls_amps=ls_amps,
                n_pre_ls=n_pre, t_win=t_win, ss_tilt=ss_tilt,
                ls_tilt={54 + k: float(sigma_b * units.tilt[k]) for k in range(len(ls_amps))},
                ls_n_samples=n_ls, dir=d)


def expected_t_tr(resp_window, threshold, bin_ms=5.0):
    """[legacy: W14 and the dev checks] First 5 ms boundary after which every later bin's |mean response| < threshold."""
    t_ms = (np.arange(resp_window.size) - PRE_N) / FS * 1e3
    edges = np.arange(0.0, 200.0 + 1e-9, bin_ms)
    exceeds = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t_ms >= a) & (t_ms < b)
        exceeds.append(abs(np.mean(resp_window[m])) >= threshold)
    bad = np.flatnonzero(exceeds)
    return float(edges[bad[-1] + 1]) if bad.size else 0.0


def single_pulse_response(dep, tau_ms=TAU_M_MS):
    """The RC response of ONE pulse in a stored window (no cross-talk)."""
    i = np.zeros(PRE_N + POST_N)
    i[PRE_N:PRE_N + PULSE_N] = 200.0 if dep else -200.0
    return rc_response(i, tau_ms=tau_ms)


# ---------------------------------------------------------------------------
#  Checks
# ---------------------------------------------------------------------------
def _w1(d, truth):
    ss = load_ss_windows(str(d))
    ok = (np.array_equal(ss.position, truth["positions"]) and np.array_equal(ss.sweep_number, truth["sweeps"])
          and np.array_equal(ss.polarity_is_dep, truth["pol"]) and ss.period_samples == PERIOD
          and ss.train_length == TRAIN and ss.n_windows == truth["K"] and ss.pre_n == PRE_N)
    notes = ["K=%d period=%d train=%d" % (ss.n_windows, ss.period_samples, ss.train_length)]
    # tampered tables
    rows = list(csv.reader(open(d / "trace_qc_results.csv")))
    hdr, body = rows[0], rows[1:]
    def write_rows(rs):
        with open(d / "trace_qc_results.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(hdr); w.writerows(rs)
    caught = []
    for label, mut in (("wrong count", lambda rs: rs[:-1]),
                       ("swapped polarity", lambda rs: [r[:1] + ["hyp" if r[1] == "dep" else "dep"] + r[2:]
                                                        if r[5] == "True" and r[0] == "0" else r for r in rs]),
                       ("non-alternating train", lambda rs: [r[:1] + ["dep"] + r[2:] if r[0] in ("40",) else r for r in rs])):
        try:
            write_rows(mut([list(r) for r in body]))
            load_ss_windows(str(d))
            caught.append("%s: NOT refused" % label)
        except LayoutError:
            caught.append("%s refused" % label)
        finally:
            write_rows(body)
    try:
        load_ss_windows(str(d), WindowConfig(ss_period_samples=10000))
        caught.append("wrong period: NOT refused")
    except LayoutError:
        caught.append("wrong period refused")
    ok = ok and all("NOT" not in c for c in caught)
    # measure_period must not stop at a shift where only the 16-sample head recurs: two windows of
    # a quantised trace whose head pattern appears at shift 300 but whose full overlap matches at 9000
    n_w = ss.n_w
    rngq = np.random.default_rng(1)
    base = np.round(rngq.standard_normal(n_w + 9000) * 4) / 4
    va, vb = base[:n_w].copy(), base[9000:9000 + n_w].copy()
    va[300:316] = vb[:16]                                   # a false head at shift 300
    fake = load_ss_windows(str(d))
    fake.v = np.stack([va, vb]); fake.sweep_number = np.array([1, 1]); fake.position = np.array([0, 1])
    fake.polarity_is_dep = np.array([True, False])
    period_ok = measure_period(fake, 100) == 9000
    ok = ok and period_ok
    report("W1 layout from the QC table: positions, sweeps, polarities, period %d; four tampers refused; false head rejected" % PERIOD,
           ok, "; ".join(notes + caught + ["false-head shift rejected: %s" % period_ok]))


def planted_ss(ss, seed, params, sigma_pre=0.04, base=-60.0):
    """A copy of `ss` with noise-free planted windows: base + sgn A exp(-t / tau)
    for t >= 0 (sgn +1 dep, -1 hyp) and white noise of SD sigma_pre before
    t = 0, so that sigma_SS_hat is known; plus a 1 mV artefact on [0, 5) ms,
    which the fit must not see (fit_start_ms = 5). params(key, sigma_hat) ->
    (A, tau_ms) per group key (sweep, is_dep), sigma_hat the median pre-pulse
    SD of the planted windows (drawn first, with `seed`)."""
    rng = np.random.default_rng(seed)
    v = np.empty_like(ss.v)
    v[:, :ss.pre_n] = base + sigma_pre * rng.standard_normal((ss.n_windows, ss.pre_n))
    sigma_hat = float(np.median(v[:, :ss.pre_n].std(axis=1, ddof=1)))
    t_post = ss.t[ss.pre_n:] * 1e3
    for k in range(ss.n_windows):
        key = (int(ss.sweep_number[k]), bool(ss.polarity_is_dep[k]))
        A, tau = params(key, sigma_hat)
        v[k, ss.pre_n:] = base + (1.0 if key[1] else -1.0) * A * np.exp(-t_post / tau)
        v[k, ss.pre_n:ss.pre_n + int(round(5e-3 * FS))] += 1.0            # outside the fitted range
    return replace(ss, v=v), sigma_hat


def _w2(d, truth):
    ss = load_ss_windows(str(d))
    cfg = WindowConfig()
    kap = cfg.kappa_tr
    notes, ok = [], True
    # (a) a planted decay: parameters back to 1e-9 (D-110: the exact fit), sigma_SS_hat, the start within one sample
    A0, tau0 = 0.4, TAU_M_MS
    sa, sig = planted_ss(ss, 21, lambda key, s: (A0, tau0))
    st = decay_start(sa)
    t_exp = tau0 * np.log(A0 / (kap * sig))
    sgn = np.where(st.group_is_dep, 1.0, -1.0)
    ok_a = (all(x == "ok" for x in st.group_status)
            and np.allclose(st.A_mV, sgn * A0, rtol=1e-9, atol=0) and np.allclose(st.tau_ms, tau0, rtol=1e-9, atol=0)
            and np.allclose(st.B_mV, -60.0, rtol=1e-9, atol=0) and abs(st.sigma_ss_mV - sig) <= 1e-15
            and np.all(np.abs(st.t_group_ms - t_exp) <= 1e3 / FS) and abs(st.t_ss_ms - st.t_star_ms) == 0.0
            and abs(st.t_ss_ms - t_exp) <= 1e3 / FS and st.flags == () and st.fallback_reason == "")
    ok &= ok_a
    notes.append("planted: t_SS %.3f ms (expected %.3f), max |tau_hat/tau - 1| %.1e"
                 % (st.t_ss_ms, t_exp, float(np.max(np.abs(st.tau_ms / tau0 - 1.0)))))
    # (b) a fast decay: the floor binds
    sb, _ = planted_ss(ss, 22, lambda key, s: (0.4, 5.0))
    stb = decay_start(sb)
    ok_b = stb.t_ss_ms == 80.0 and "ss_start_floor" in stb.flags and "ss_start_fallback" not in stb.flags
    ok &= ok_b
    notes.append("tau 5 ms: t_sp %.1f, t_SS %.0f, flags %s" % (float(np.max(stb.t_group_ms)), stb.t_ss_ms, stb.flags))
    # (c) the median (D-111): groups at 100, 120, 140, 160 ms -> 130 ms; all at 151 ms fall back, all at 149 ms kept
    keys = [(int(gs), bool(gd)) for gs, gd in zip(st.group_sweep, st.group_is_dep)]
    def at(times):
        table = dict(zip(keys, times))
        return lambda key, s: (kap * s * np.exp(table[key] / tau0), tau0)
    smed, _ = planted_ss(ss, 23, at((100.0, 120.0, 140.0, 160.0)))
    s151, _ = planted_ss(ss, 23, at((151.0,) * 4))
    s149, _ = planted_ss(ss, 23, at((149.0,) * 4))
    stm, st151, st149 = decay_start(smed), decay_start(s151), decay_start(s149)
    ok_c = (len(keys) == 4 and abs(stm.t_ss_ms - 130.0) <= 1e3 / FS
            and st151.t_ss_ms == 80.0 and "ss_start_fallback" in st151.flags and st151.fallback_reason == "short_window"
            and abs(st151.t_star_ms - 151.0) <= 1e3 / FS
            and abs(st149.t_ss_ms - 149.0) <= 1e3 / FS and "ss_start_fallback" not in st149.flags
            and st149.fallback_reason == "")
    ok &= ok_c
    notes.append("groups at 100/120/140/160 -> t_SS %.2f; all 151 -> %.0f (%s); all 149 -> %.2f"
                 % (stm.t_ss_ms, st151.t_ss_ms, st151.fallback_reason, st149.t_ss_ms))
    # (d) a ramp, no decay: every group's minimum at the upper bound -> failed fits (D-110.2), the fallback,
    # every window kept
    ramp = sa.v.copy()
    t_post = ss.t[ss.pre_n:] * 1e3
    for k in range(ss.n_windows):
        ramp[k, ss.pre_n:] = -60.0 + (1.0 if ss.polarity_is_dep[k] else -1.0) * 0.002 * t_post
    std = decay_start(replace(sa, v=ramp))
    ok_d = (all(x == "fit" for x in std.group_status) and std.t_ss_ms == 80.0 and np.isnan(std.t_star_ms)
            and std.fallback_reason == "no_valid_group" and "ss_start_fallback" in std.flags
            and sum(f.endswith(":fit") for f in std.flags) == std.n_groups and bool(np.all(std.keep)))
    ok &= ok_d
    # (e) a fit double failing the first group: out of the median, its windows kept
    calls = {"n": 0}
    def first_fails(t, y, c):
        calls["n"] += 1
        if calls["n"] == 1:
            nan = float("nan")
            return DecayFit(nan, nan, nan, nan, False, False, "test double")
        return fit_decay(t, y, c)
    se_, _ = planted_ss(ss, 24, at((140.0, 100.0, 110.0, 120.0)))
    ste = decay_start(se_, fit=first_fails)
    tag = "ss_group_dropped:%d:%s:fit" % (keys[0][0], "dep" if keys[0][1] else "hyp")
    ok_e = (ste.group_status[0] == "fit" and tag in ste.flags and bool(np.all(ste.keep))
            and abs(ste.t_ss_ms - 110.0) <= 1e3 / FS and all(x == "ok" for x in ste.group_status[1:]))
    ok &= ok_e
    notes.append("first group failed: t_SS %.2f (110 without it, 115 with it)" % ste.t_ss_ms)
    # (f) the base cell's noisy windows (RC 113 MOhm, 28.143 ms, theta_cf): every fit valid, tau_hat near tau_m
    stf = decay_start(ss)
    ok_f = (all(x == "ok" for x in stf.group_status) and np.all(np.abs(stf.tau_ms / TAU_M_MS - 1.0) < 0.25)
            and 80.0 <= stf.t_ss_ms <= 150.0)
    ok &= ok_f
    notes.append("noisy cell: tau_hat %s ms, t_sp %s ms, t_SS %.1f ms, flags %s"
                 % (np.round(stf.tau_ms, 2).tolist(), np.round(stf.t_group_ms, 1).tolist(), stf.t_ss_ms, stf.flags))
    # (g) the fit is the exact least-squares optimum (D-110) on each noisy group: stationary (ST0), the
    # derivative formula against finite differences, curve_fit started there stays, no dense-grid point lower
    sel = np.arange(ss.n_w) >= ss.pre_n + int(round(cfg.fit_start_ms * 1e-3 * FS))
    tt = ss.t[sel] * 1e3
    worst = dict(stat=0.0, fd=0.0, lm_tau=0.0, lm_rss=0.0, grid=0.0)
    for g, key in enumerate(keys):
        idx = np.flatnonzero((ss.sweep_number == key[0]) & (ss.polarity_is_dep == key[1]))
        y = np.median(ss.v[idx], axis=0)[sel]
        f = fit_decay(tt, y, cfg)
        _, _, rss_hat = profile_decay(tt, y, f.tau_ms)
        h = 1e-5
        dfun = lambda tau: (profile_decay(tt, y, tau * (1 + h))[2] - profile_decay(tt, y, tau * (1 - h))[2]) / (2 * tau * h)
        worst["stat"] = max(worst["stat"], abs(_drss(tt, y, f.tau_ms)) * f.tau_ms / rss_hat)
        worst["fd"] = max(worst["fd"], abs(_drss(tt, y, 0.8 * f.tau_ms) / dfun(0.8 * f.tau_ms) - 1.0))
        popt = curve_fit(lambda x, B, A, tau: B + A * np.exp(-x / tau), tt, y, p0=(f.B_mV, f.A_mV, f.tau_ms))[0]
        rss_lm = float(np.sum((y - popt[0] - popt[1] * np.exp(-tt / popt[2])) ** 2))
        worst["lm_tau"] = max(worst["lm_tau"], abs(popt[2] / f.tau_ms - 1.0))
        worst["lm_rss"] = max(worst["lm_rss"], (rss_hat - rss_lm) / rss_hat)
        dense = min(profile_decay(tt, y, tau)[2] for tau in np.geomspace(f.tau_ms / 2, f.tau_ms * 2, 20001))
        worst["grid"] = max(worst["grid"], (rss_hat - dense) / rss_hat)
    ok_g = (worst["stat"] < 1e-8 and worst["fd"] < 1e-6 and worst["lm_tau"] < 1e-7 and worst["lm_rss"] < 1e-12
            and worst["grid"] < 1e-12)
    ok &= ok_g
    notes.append("exactness: |dRSS/dtau| tau / RSS %.1e, formula vs finite difference %.1e, curve_fit tau %.1e, "
                 "RSS %.1e, dense grid %.1e" % (worst["stat"], worst["fd"], worst["lm_tau"], worst["lm_rss"], worst["grid"]))
    report("W2 start (D-103, D-104, D-110, D-111): planted decay exact, floor, median, fallback at 151 / kept at 149, "
           "a ramp fails, failed fits, noisy cell, the fit exact", ok,
           "; ".join(notes) + "; parts a-g %s" % [ok_a, ok_b, ok_c, ok_d, ok_e, ok_f, ok_g])


def _drss(t, y, tau):
    """(ST0) written out here (oracle): dRSS/dtau = -(2 A / tau^2) sum r e t at the profiled (A, B)."""
    A, B, _ = profile_decay(t, y, tau)
    e = np.exp(-t / tau)
    r = y - B - A * e
    return -2.0 * A / tau ** 2 * float(r @ (e * t))


def _w3(d, truth):
    ss = load_ss_windows(str(d))
    X, j0 = enlarged_windows(ss, 80.0)
    Xt, med = subtract_median_response(X, ss.polarity_is_dep, ss.sweep_number)
    groups = {}
    for k, key in enumerate(zip(ss.sweep_number.tolist(), ss.polarity_is_dep.tolist())):
        groups.setdefault((int(key[0]), bool(key[1])), []).append(k)
    zero = max(float(np.abs(np.median(Xt[idx], axis=0)).max()) for idx in groups.values())
    same = set(med) == set(groups) and all(np.array_equal(med[g], np.median(X[idx], axis=0)) for g, idx in groups.items())
    rng = np.random.default_rng(3)
    shift = np.zeros_like(X)
    for idx in groups.values():
        shift[idx] = rng.standard_normal(X.shape[1])
    Xt2, _ = subtract_median_response(X + shift, ss.polarity_is_dep, ss.sweep_number)
    inv = float(np.abs(Xt2 - Xt).max())
    Xn = truth["noise"][:, j0:]
    Xnt, _ = subtract_median_response(Xn, ss.polarity_is_dep, ss.sweep_number)
    Xct, _ = subtract_median_response(Xn + shift, ss.polarity_is_dep, ss.sweep_number)
    dev = float(np.nanmax(np.abs(ss_window_stats(Xct, FS) - ss_window_stats(Xnt, FS))))
    report("W3 subtraction (D-103.1): zero median per group; medians = numpy per group; common vectors leave X_tilde "
           "and the statistics", zero < 1e-12 and same and inv < 1e-12 and dev < 1e-10,
           "%d groups (K %s); max |median| %.1e; medians identical %s; max |X_tilde change| %.1e; "
           "max |stat change| %.1e" % (len(groups), [len(i) for i in groups.values()], zero, same, inv, dev))


def _w4(d, truth):
    rng = np.random.default_rng(4)
    X = rng.standard_normal((7, 3000))
    st = window_stats(X, FS, SS_LAGS_MS)
    lags = [int(round(l * 1e-3 * FS)) for l in SS_LAGS_MS]
    ref = np.array([[x.std(ddof=1)] + list(noise_stats.sample_acf(x, lags)) for x in X])
    ok_stats = np.allclose(st, ref, atol=1e-14, rtol=0)
    from noise_cell_stats import lags_in_samples
    ok_stats = ok_stats and list(lags_in_samples((0.02, 0.1, 1.0, 10.0), 44100.0)) == [1, 4, 44, 441]   # round, not truncate
    ls = load_ls_prewindows(str(d))
    ls_st = ls_window_stats(ls.pre, FS)
    sv = statistic_vector(np.zeros((3, 5)), 6000, ls_st, 13250)
    ok_labels = (sv.d == 15 and sv.labels == STAT_LABELS and sv.block == STAT_BLOCK
                 and sv.block[:5] == ("ss",) * 5 and sv.block[5:] == ("ls",) * 10
                 and list(sv.n) == [6000] * 5 + [13250] * 4 + [2500] + [6625] * 5
                 and list(sv.K) == [3] * 5 + [len(ls.pre)] * 10
                 and STAT_LABELS[5] == "ls_sd" and STAT_LABELS[9] == "ls_first_sd" and STAT_LABELS[10] == "ls_last_sd")
    # oracle: the monolith's pre-window for the same sweep
    note = "stats 1e-14 %s; labels/n/K %s" % (ok_stats, ok_labels)
    try:
        import passive_fitting_hpc_fixed as mono           # noqa: WPS433
        import noise_calibration as nc
        cd = nc.load_for_noise(d, mono=mono)
        singles = [b for b in cd.long_square_subthreshold if int(getattr(b, "n_repeats_averaged", 1)) == 1]
        matched, worst = 0, 0.0
        for b in singles:
            sn = int(b.sweep_numbers[0]) if b.sweep_numbers else None
            k = np.flatnonzero(ls.sweep_number == sn)
            if k.size != 1:
                continue
            ref_pre = nc.pre_window_samples(mono, b)
            mine = ls.pre[int(k[0])]
            if ref_pre.size == mine.size:
                worst = max(worst, float(np.abs(ref_pre - mine).max()))
                matched += 1
            else:
                worst = float("inf")
        ok_oracle = matched >= 3 and worst < 1e-12
        note += "; monolith oracle: %d hyp sweeps matched, max |diff| %.1e" % (matched, worst)
    except ImportError as exc:
        ok_oracle = True
        note += "; monolith oracle SKIPPED (%s)" % exc
    report("W4 statistics: window_stats = numpy + sample_acf; the 15 labels, blocks, n, K; LS pre-window = the fitter's",
           ok_stats and ok_labels and ok_oracle, note)


def _w5(d, truth):
    ss = load_ss_windows(str(d))
    nwin = ss_noise_windows(ss)
    st = ss_window_stats(nwin.X_tilde, FS)
    per = per_sweep_vectors(st, nwin.ss.sweep_number)
    ok = set(per) == {92, 93} and nwin.X_tilde.shape == (nwin.ss.n_windows, nwin.m)
    for s, vec in per.items():
        ok = ok and np.allclose(vec, np.median(st[nwin.ss.sweep_number == s], axis=0))
    sv = statistic_vector(st, nwin.m, np.zeros((0, 10)), 13250)
    ok = ok and np.allclose(sv.q[:5], np.median(st, axis=0)) and np.all(np.isnan(sv.q[5:])) and list(sv.K[5:]) == [0] * 10
    report("W5 per-sweep vectors via ss_noise_windows: keys 92, 93; medians over the sweep's kept windows; cell vector; "
           "empty LS -> NaN, K 0", ok, "t_SS %.1f ms, m %d, K per sweep %s"
           % (nwin.start.t_ss_ms, nwin.m, {s: int((nwin.ss.sweep_number == s).sum()) for s in per}))


def _w6():
    rng = np.random.default_rng(6)
    K, sigma = 60, 0.7
    ses = []
    for _ in range(40):
        A = sigma * rng.standard_normal((K, 2))
        ses.append(bootstrap_median_se(A, 2000, rng))
    mean_se = np.mean(ses, axis=0)
    expect = 1.2533 * sigma / np.sqrt(K)
    A = sigma * np.random.default_rng(1).standard_normal((K, 3))
    a1 = bootstrap_median_se(A, 500, np.random.default_rng(11))
    a2 = bootstrap_median_se(A, 500, np.random.default_rng(11))
    ok = np.all(np.abs(mean_se / expect - 1) < 0.2) and np.array_equal(a1, a2)
    report("W6 bootstrap SE of the median: within 20 %% of 1.2533 sigma/sqrt(K); deterministic for one seed",
           ok, "mean SE / expected = %s" % np.round(mean_se / expect, 3))


def _w7(truth):
    rng = np.random.default_rng(7)
    law = nl.NoiseLaw(0.03, 0.2, 0.06, 0.04)
    ss = load_ss_windows(str(truth["dir"]))
    X, j0 = enlarged_windows(ss, 80.0)
    m = X.shape[1]
    # 400 sweeps with the fixture's layout: sweep 92's positions, polarity from the position
    pos = ss.position[ss.sweep_number == 92]
    n = REST_N + (TRAIN - 1) * PERIOD + TAIL_N
    V_hats, pairs = [], 0
    one = load_ss_windows(str(truth["dir"]))
    sel = ss.sweep_number == 92
    one.v = one.v[sel]; one.sweep_number = one.sweep_number[sel]; one.position = pos; one.polarity_is_dep = one.polarity_is_dep[sel]
    onsets = REST_N + PERIOD * pos
    for r in range(400):
        units = nl.draw_units(rng, 1, n)
        tr = nl.simulate_sweep(law, DT, n, units, 0, sigma_b=0.0, sigma_a=0.0)
        W = np.stack([tr[s - PRE_N + j0:s + POST_N] for s in onsets])
        Wt, _, _ = subtract_mean_response(W, one.polarity_is_dep)
        v, npairs = window_mean_semivariogram(Wt, one, 4)
        V_hats.append(v); pairs = npairs
    V_hats = np.array(V_hats)
    exact = nl.semivariogram(law, DT, m, m, 4 * PERIOD)
    se = V_hats.std(ddof=1) / np.sqrt(V_hats.size)
    exp_pairs = sum(1 for p in pos if (p + 4) in set(pos))
    # pairs never cross sweeps: the full two-sweep fixture pairs only within each sweep
    Xf, _ = enlarged_windows(ss, 80.0)
    Xft, _, _ = subtract_mean_response(Xf, ss.polarity_is_dep)
    _, pairs_two = window_mean_semivariogram(Xft, ss, 4)
    exp_two = sum(sum(1 for p in ss.position[ss.sweep_number == sw] if (p + 4) in set(ss.position[ss.sweep_number == sw].tolist()))
                  for sw in (92, 93))
    ok = abs(V_hats.mean() - exact) < Z_SE * se and pairs == exp_pairs and pairs_two == exp_two
    report("W7 [legacy (T1)] semivariogram at 4 periods: pair count %d; Monte Carlo mean within 3.5 SE of the exact value; no cross-sweep pair" % exp_pairs,
           ok, "V_hat %.3e vs exact %.3e, SE %.1e, pairs %d; two sweeps %d (expected %d)" % (V_hats.mean(), exact, se, pairs, pairs_two, exp_two))


def _w8(d):
    ss = load_ss_windows(str(d))
    X, j0 = enlarged_windows(ss, 80.0)
    m_exp = ss.n_w - PRE_N - int(round(80.0 * 1e-3 * FS))
    first_t = ss.t[j0]
    end_before_next = (ss.n_w - PRE_N) < ss.period_samples          # window ends before the next onset
    ok = X.shape == (ss.n_windows, m_exp) and abs(first_t - 0.080) < 0.5 / FS and end_before_next
    report("W8 enlarged window: m = %d, first sample at t = 80 ms, ends %.1f ms before the next onset" % (
        m_exp, (ss.period_samples - (ss.n_w - PRE_N)) / FS * 1e3), ok, "shape %s, t[j0] = %.6f s" % (X.shape, first_t))


def _w9(d, d_long, truth_long):
    ls = load_ls_prewindows(str(d))
    ok = (len(ls.pre) == 5 and all(n == 13250 for n in ls.n_pre) and set(ls.polarity) == {"hyp", "dep"}
          and np.all(np.abs(ls.amplitude_pA) >= 5.0))
    first, last = ls_segments(ls.pre[0], FS)
    ok = ok and first.size == 2500 and last.size == 6625 and np.array_equal(last, ls.pre[0][-6625:])
    hyp_only = load_ls_prewindows(str(d), WindowConfig(ls_polarities=("hyp",)))
    ok = ok and len(hyp_only.pre) == 3
    # end windows: 'tail_short' with the 0.5 s tail (the window starts AT the step's end), 'ok' with 1.5 s
    step_end = (LS_PRE_N + LS_STEP_N) / FS
    ok_short = (all(s == "tail_short" for s in ls.end_status) and all(e.size == 0 for e in ls.end)
                and np.all(np.isnan(ls.end_start_s)) and np.allclose(ls.step_end_s, step_end, rtol=0, atol=1e-12))
    lsl = load_ls_prewindows(str(d_long))
    n_tot = truth_long["ls_n_samples"]
    with np.load(Path(d_long) / "ls_sweeps.npz") as z:
        v0 = np.asarray(z["v_0"])
    ok_long = (all(s == "ok" for s in lsl.end_status) and all(e.size == 25000 for e in lsl.end)
               and np.allclose(lsl.end_start_s, (n_tot - 25000) / FS, rtol=0, atol=1e-12)
               and np.all(lsl.end_start_s > lsl.step_end_s) and np.all(lsl.n_samples == n_tot)
               and np.array_equal(lsl.end[0], v0[-25000:]))
    report("W9 LS: 5 pre-windows of 13250 samples (0 pA skipped), hyp and dep, segments 2500 and 6625, hyp-only 3; "
           "end windows of 25000 samples after the step ('ok') or 'tail_short'",
           ok and ok_short and ok_long,
           "n_pre %s, polarities %s; 0.5 s tail: %s; 1.5 s tail: end start %.3f s, step end %.3f s"
           % (sorted(set(ls.n_pre.tolist())), list(ls.polarity), sorted(set(ls.end_status.tolist())),
              float(lsl.end_start_s[0]), float(lsl.step_end_s[0])))


def _w10():
    bad = []
    for name in ("noise_windows.py", "noise_cell_stats.py", "smoke_noise_windows.py"):
        raw = (HERE / name).read_bytes()
        if b"\r" in raw or any(b > 126 or (b < 32 and b not in (9, 10)) for b in raw):
            bad.append(name)
    report("W10 ASCII + LF on the three files", not bad, "bad: %s" % bad if bad else "ok")


def _fw_slope(t, y, dep):
    """Within-polarity (Frisch-Waugh) closed form of (TL1)'s slope."""
    num = den = 0.0
    for p in (True, False):
        g = dep == p
        if g.any():
            dt_, dy_ = t[g] - t[g].mean(), y[g] - y[g].mean()
            num += float(dt_ @ dy_); den += float(dt_ @ dt_)
    return num / den


def _tl1_exact_sd(ss_kept, j0, m, law=THETA_CF):
    """Exact SD of (TL1)'s slope under the law's stationary part, per sweep:
    sqrt([A Sigma A^T]_00), A = (D^T D)^-1 D^T, Sigma the window means'
    covariance (noise_law.var_window_mean, window_mean_cov at d P)."""
    v0 = nl.var_window_mean(law, DT, m)
    cov = {dd: nl.window_mean_cov(law, DT, m, m, dd * PERIOD) for dd in range(1, TRAIN)}
    out = {}
    for s in dict.fromkeys(int(x) for x in ss_kept.sweep_number):
        g = ss_kept.sweep_number == s
        pos, dep = ss_kept.position[g], ss_kept.polarity_is_dep[g]
        t = (pos * PERIOD + j0 + 0.5 * (m - 1)) * DT
        D = np.column_stack([t - t.mean()] + [(dep == p).astype(float) for p in (True, False) if np.any(dep == p)])
        dd = np.abs(pos[:, None] - pos[None, :])
        Sig = np.where(dd == 0, v0, 0.0)
        for lag, c in cov.items():
            Sig = np.where(dd == lag, c, Sig)
        A = np.linalg.solve(D.T @ D, D.T)
        out[s] = float(np.sqrt((A @ Sig @ A.T)[0, 0]))
    return out


def _w11(d_tilt, truth_tilt):
    ss = load_ss_windows(str(d_tilt))
    nwin = ss_noise_windows(ss)
    til = nwin.tilts                                   # the reading of D-115.2
    k = nwin.ss
    jf = PRE_N + int(round(T_FLOOR_MS * 1e-3 * FS))   # independent of noise_windows.floor_index
    mf = k.n_w - jf
    ok_jf = floor_index(k.pre_n, FS) == jf
    Xf = k.v[:, jf:]
    means = Xf.mean(axis=1)
    centres = (k.position * PERIOD + jf + 0.5 * (mf - 1)) * DT
    fw = {s: _fw_slope(centres[k.sweep_number == s], means[k.sweep_number == s], k.polarity_is_dep[k.sweep_number == s])
          for s in (92, 93)}
    b = til.as_dict()
    err_fw = max(abs(b[s] - fw[s]) for s in fw)
    Xf_sub, _ = subtract_median_response(Xf, k.polarity_is_dep, k.sweep_number)
    b_sub = ss_sweep_tilts(Xf_sub, k, jf).as_dict()
    err_sub = max(abs(b[s] - b_sub[s]) for s in b)
    sd = _tl1_exact_sd(k, jf, mf)
    z = {s: float((b[s] - truth_tilt["ss_tilt"][s]) / sd[s]) for s in b}
    ok_prec = all(abs(v) < N_SD for v in z.values())
    # one polarity: the dep windows only -> one intercept
    dep_only = select_windows(k, k.polarity_is_dep)
    b_dep = read_ss_tilts(dep_only).as_dict()
    md = dep_only.v[:, jf:].mean(axis=1)
    cd = (dep_only.position * PERIOD + jf + 0.5 * (mf - 1)) * DT
    ok_one = all(abs(b_dep[s] - _fw_slope(cd[dep_only.sweep_number == s], md[dep_only.sweep_number == s],
                                          np.ones(int((dep_only.sweep_number == s).sum()), dtype=bool))) < 1e-12
                 for s in b_dep)
    # two windows of a sweep: no more windows than columns -> NaN, flagged
    two = select_windows(k, np.isin(np.arange(k.n_windows), np.flatnonzero(k.sweep_number == 92)[:2])
                         | (k.sweep_number == 93))
    til2 = read_ss_tilts(two)
    ok_two = np.isnan(til2.as_dict()[92]) and "ss_tilt_none:92" in til2.flags and np.isfinite(til2.as_dict()[93])
    b_xt = ss_sweep_tilts(nwin.X_tilde, k, nwin.j0).as_dict()     # reported: after (OF1) X_tilde holds no tilt
    ok = (err_fw < 1e-12 and err_sub < 1e-12 and ok_prec and ok_one and ok_two and np.all(til.se_mV_per_s > 0)
          and ok_jf)
    report("W11 SS tilt (TL1) at the 80 ms floor (D-115.2): = within-polarity closed form; = after each group's median; "
           "within 4 exact SDs of the planted tilts; one intercept; two windows -> NaN", ok,
           "t_SS %.1f ms; b_hat %s vs planted %s mV/s; exact SD %s; z %s; OLS SE %s; |closed form| %.1e, |subtracted| %.1e; "
           "re-read on X_tilde (no tilt left) %s"
           % (nwin.start.t_ss_ms, {s: round(v, 5) for s, v in b.items()},
              {s: round(v, 5) for s, v in truth_tilt["ss_tilt"].items()}, {s: round(v, 5) for s, v in sd.items()},
              {s: round(v, 2) for s, v in z.items()}, np.round(til.se_mV_per_s, 5).tolist(), err_fw, err_sub,
              {s: round(v, 5) for s, v in b_xt.items()}))


def _w12(d, d_tilt, truth_tilt):
    notes = []
    ls = load_ls_prewindows(str(d_tilt))
    til = ls_sweep_tilts(ls)
    b = til.as_dict()
    # (a) the oracle: ls_baseline_qc.baseline_record on the same stored sweeps
    try:
        import ls_baseline_qc as lbq                       # noqa: WPS433 (imports pandas)
    except ImportError as exc:
        lbq = None
        notes.append("oracle SKIPPED (%s)" % exc)
    ok_oracle, worst = True, 0.0
    meta = json.load(open(Path(d_tilt) / "metadata.json"))
    index_of = {int(r["sweep_number"]): int(r["index"]) for r in meta["ls_sweeps"]}
    with np.load(Path(d_tilt) / "ls_sweeps.npz") as z:
        for k, s in enumerate(ls.sweep_number):
            v = np.asarray(z["v_%d" % index_of[int(s)]])
            t = np.arange(v.size) / FS
            pm = np.arange(v.size) < int(ls.n_pre[k])
            i0 = v.size - 25000
            dt_c = float(t[i0:].mean() - t[pm].mean())
            if lbq is not None:
                rec = lbq.baseline_record(t, v, t_on_s=float(ls.onset_s[k]), t_off_s=float(ls.step_end_s[k]),
                                          pre_mask=pm, fs_Hz=FS)
                ok_oracle &= rec["flag"] == "ok"
                worst = max(worst, abs(rec["delta_end_mV"] / dt_c - b[int(s)]))
    # a sweep that stopped early (5000 trailing NaN samples): the end window ends at its last finite sample
    d_nan = Path(d_tilt).parent / (Path(d_tilt).name + "_nan_tail")
    if d_nan.exists():
        shutil.rmtree(d_nan)
    shutil.copytree(d_tilt, d_nan)
    with np.load(Path(d_tilt) / "ls_sweeps.npz") as z:
        data = {key: np.asarray(z[key]).copy() for key in z.files}
    data["v_0"][-5000:] = np.nan
    np.savez_compressed(d_nan / "ls_sweeps.npz", **data)
    ls_n = load_ls_prewindows(str(d_nan))
    b_n = ls_sweep_tilts(ls_n).as_dict()
    v0 = data["v_0"]
    last = int(np.flatnonzero(np.isfinite(v0))[-1]) + 1
    ok_nan = (ls_n.end_status[0] == "ok" and np.array_equal(ls_n.end[0], v0[last - 25000:last])
              and np.isfinite(b_n[54]))
    if lbq is not None:
        t0 = np.arange(v0.size) / FS
        pm0 = np.arange(v0.size) < int(ls_n.n_pre[0])
        rec0 = lbq.baseline_record(t0, v0, t_on_s=float(ls_n.onset_s[0]), t_off_s=float(ls_n.step_end_s[0]),
                                   pre_mask=pm0, fs_Hz=FS)
        dtc0 = float(t0[last - 25000:last].mean() - t0[pm0].mean())
        ok_nan &= rec0["flag"] == "ok" and abs(rec0["delta_end_mV"] / dtc0 - b_n[54]) < 1e-12
        worst = max(worst, abs(rec0["delta_end_mV"] / dtc0 - b_n[54]))
    ok_oracle &= worst < 1e-12 and ok_nan
    if lbq is not None:
        notes.append("oracle: max |b_hat - delta_end / dt_c| %.1e mV/s (a NaN-tailed sweep included: %s)" % (worst, ok_nan))
    # (b) the planted tilts: within 4 exact SDs, sqrt(V_pre + V_end - 2 C) / dt_c
    n_pre, n_end, n_tot = int(ls.n_pre[0]), 25000, int(ls.n_samples[0])
    i0 = n_tot - n_end
    dt_c = (i0 + 0.5 * (n_end - 1) - 0.5 * (n_pre - 1)) * DT
    sd = np.sqrt(nl.var_window_mean(THETA_CF, DT, n_pre) + nl.var_window_mean(THETA_CF, DT, n_end)
                 - 2.0 * nl.window_mean_cov(THETA_CF, DT, n_pre, n_end, i0)) / dt_c
    z_ = {s: float((b[s] - truth_tilt["ls_tilt"][s]) / sd) for s in b}
    ok_prec = all(abs(v) < N_SD for v in z_.values()) and til.flags == ()
    notes.append("dt_c %.4f s, exact SD %.4f mV/s, z %s" % (dt_c, sd, {s: round(v, 2) for s, v in z_.items()}))
    # (c) the 0.5 s tail: every tilt NaN, flagged as the oracle flags it
    ls_s = load_ls_prewindows(str(d))
    til_s = ls_sweep_tilts(ls_s)
    ok_short = (np.all(np.isnan(til_s.b_mV_per_s))
                and set(til_s.flags) == {"ls_tilt_none:%d:tail_short" % int(s) for s in ls_s.sweep_number})
    if lbq is not None:
        meta_s = json.load(open(Path(d) / "metadata.json"))
        idx_s = {int(r["sweep_number"]): int(r["index"]) for r in meta_s["ls_sweeps"]}
        with np.load(Path(d) / "ls_sweeps.npz") as z:
            for k, s in enumerate(ls_s.sweep_number):
                v = np.asarray(z["v_%d" % idx_s[int(s)]])
                rec = lbq.baseline_record(np.arange(v.size) / FS, v, t_on_s=float(ls_s.onset_s[k]),
                                          t_off_s=float(ls_s.step_end_s[k]),
                                          pre_mask=np.arange(v.size) < int(ls_s.n_pre[k]), fs_Hz=FS)
                ok_short &= rec["flag"] == "tail_short"
    report("W12 LS tilt (TL2): = ls_baseline_qc's delta_end / centre distance; within 4 exact SDs of the planted tilts; "
           "0.5 s tail -> NaN, tail_short", ok_oracle and ok_prec and ok_short,
           "; ".join(notes + ["b_hat %s vs planted %s" % ({s: round(v, 4) for s, v in b.items()},
                                                         {s: round(truth_tilt["ls_tilt"][s], 4) for s in b})]))


def _w13(d_small, truth_small):
    ss = load_ss_windows(str(d_small))
    nwin = ss_noise_windows(ss)
    st = nwin.start
    small = (ss.sweep_number == 93) & ~ss.polarity_is_dep
    g93h = [g for g in range(st.n_groups) if st.group_sweep[g] == 93 and not st.group_is_dep[g]]
    ok_flag = (int(small.sum()) == 3 and "ss_group_dropped:93:hyp:few_windows" in st.flags and len(g93h) == 1
               and st.group_status[g93h[0]] == "few_windows" and np.array_equal(st.keep, ~small))
    k = nwin.ss
    ok_kept = (k.n_windows == ss.n_windows - 3 and not np.any((k.sweep_number == 93) & ~k.polarity_is_dep)
               and nwin.X_tilde.shape[0] == k.n_windows)
    sig = float(np.median(ss.v[~small, :ss.pre_n].std(axis=1, ddof=1)))
    ok_sig = abs(st.sigma_ss_mV - sig) <= 1e-15
    stats = ss_window_stats(nwin.X_tilde, FS)
    sv = statistic_vector(stats, nwin.m, np.zeros((0, 10)), 13250)
    per = per_sweep_vectors(stats, k.sweep_number)
    dep93 = k.sweep_number == 93
    ok_stats = (list(sv.K[:5]) == [k.n_windows] * 5 and np.allclose(per[93], np.median(stats[dep93], axis=0)))
    til = nwin.tilts                                   # the reading of D-115.2, on the kept windows only
    n93 = int(til.n_windows[list(til.sweep_number).index(93)])
    ok_tilt = n93 == int(dep93.sum()) and np.isfinite(til.as_dict()[93])
    report("W13 small groups (D-106): sweep 93 with 3 hyp windows flagged and absent from the kept windows, X_tilde, "
           "K, sweep 93's vector and tilt; sigma_SS_hat over the kept windows",
           ok_flag and ok_kept and ok_sig and ok_stats and ok_tilt,
           "K %d -> %d; flags %s; sweep 93 tilt from %d windows" % (ss.n_windows, k.n_windows, st.flags, n93))


def _w14(d, truth):
    """Legacy functions as at commit 1aabe3e (the old W2 (a) and W3 (a))."""
    ss = load_ss_windows(str(d))
    rng = np.random.default_rng(14)
    sigma_ss = float(np.median(ss.v[:, :PRE_N].std(axis=1, ddof=1)))
    clean = np.stack([single_pulse_response(p) for p in truth["pol"]])
    pre_noise = sigma_ss * rng.standard_normal((clean.shape[0], PRE_N))
    pre_noise -= pre_noise.mean(axis=1, keepdims=True)     # zero-mean per window: r_odd = the response exactly
    clean[:, :PRE_N] += pre_noise
    ss_clean = replace(ss, v=-60.0 + clean)
    tr = transient_start(ss_clean, WindowConfig(t_ss_floor_ms=0.0))
    t_exp = expected_t_tr(single_pulse_response(True), 0.1 * tr.sigma_ss_mV)
    tr_floor = transient_start(ss_clean)
    ok_t = (abs(tr.t_tr_ms - t_exp) <= 1e-9 and tr.t_tr_ms > 0 and tr_floor.t_ss_ms == max(80.0, tr.t_tr_ms)
            and ("floor_binds" in tr_floor.flags) == (tr.t_tr_ms < 80.0))
    long_resp = np.array([rc_response(np.where((np.arange(PRE_N + POST_N) >= PRE_N) & (np.arange(PRE_N + POST_N) < PRE_N + PULSE_N),
                                               200.0 if p else -200.0, 0.0), tau_ms=200.0) for p in truth["pol"]])
    tr_long = transient_start(replace(ss, v=-60.0 + 5.0 * long_resp + 1e-6 * rng.standard_normal(long_resp.shape)))
    ok_long = "ss_enlarged_none" in tr_long.flags and tr_long.t_ss_ms >= 190.0
    X, _ = enlarged_windows(ss, 80.0)
    Xt, mu_dep, mu_hyp = subtract_mean_response(X, ss.polarity_is_dep)
    zero = max(np.abs(Xt[ss.polarity_is_dep].mean(axis=0)).max(), np.abs(Xt[~ss.polarity_is_dep].mean(axis=0)).max())
    report("W14 [legacy] transient_start: noise-free time exact, floor 80 ms, ss_enlarged_none at 190 ms; "
           "subtract_mean_response: zero mean per polarity", ok_t and ok_long and zero < 1e-12,
           "t_tr %.0f (expected %.0f) ms; long-response flags %s; max |mean| %.1e" % (tr.t_tr_ms, t_exp, tr_long.flags, zero))


def _w15(tmp, d_tilt):
    """D-115: the SS tilt read at the floor and its offsets removed before the median."""
    # (a), (b) a noise-free cell without response and a planted tilt per sweep
    d0 = Path(tmp) / "specimen_900000006"
    truth0 = write_fixture(d0, np.random.default_rng(17), law=SILENT, with_response=False, sigma_b=0.1,
                           ls_tail_n=LS_LONG_TAIL_N)
    ss0 = load_ss_windows(str(d0))
    kept0 = select_windows(ss0, keep_mask(ss0))
    b0 = read_ss_tilts(kept0).as_dict()
    err_a = max(abs(b0[s] / truth0["ss_tilt"][s] - 1.0) for s in b0)
    nwin0 = ss_noise_windows(ss0)
    zero_b = float(np.abs(nwin0.X_tilde).max())
    Xp, _ = enlarged_windows(kept0, nwin0.start.t_ss_ms)
    Xpt, _ = subtract_median_response(Xp, kept0.polarity_is_dep, kept0.sweep_number)
    plain_b = float(np.abs(Xpt).max())
    # (c) the noisy tilted cell against an independent re-implementation
    ss = load_ss_windows(str(d_tilt))
    nwin = ss_noise_windows(ss)
    kept = select_windows(ss, keep_mask(ss))
    rt = read_ss_tilts(kept)
    same_tilts = (np.array_equal(nwin.tilts.b_mV_per_s, rt.b_mV_per_s)
                  and list(nwin.tilts.sweep_number) == list(rt.sweep_number))
    b = rt.as_dict()
    off = np.zeros(kept.n_windows)
    for s_ in b:
        g = kept.sweep_number == s_
        tk = kept.position[g] * PERIOD * DT
        off[g] = b[s_] * (tk - tk.mean())
    err_off = float(np.abs(tilt_offsets(kept.position, kept.sweep_number, PERIOD, DT, b) - off).max())
    corr = kept.v - off[:, None]
    X = corr[:, nwin.j0:]
    Xt2 = X.copy()
    for s_ in (92, 93):
        for p in (True, False):
            idx = np.flatnonzero((kept.sweep_number == s_) & (kept.polarity_is_dep == p))
            Xt2[idx] = X[idx] - np.median(X[idx], axis=0)
    err_xt = float(np.abs(nwin.X_tilde - Xt2).max())
    st2 = decay_start(replace(kept, v=corr))
    same_start = (st2.t_ss_ms == nwin.start.t_ss_ms
                  and np.allclose(st2.tau_ms, nwin.start.tau_ms, rtol=0, atol=1e-9, equal_nan=True)
                  and np.allclose(st2.A_mV, nwin.start.A_mV, rtol=0, atol=1e-12, equal_nan=True))
    # (d) a NaN tilt gives zero offsets for its sweep
    zz = tilt_offsets(kept.position, kept.sweep_number, PERIOD, DT, {92: float("nan"), 93: b[93]})
    ok_nan = (np.all(zz[kept.sweep_number == 92] == 0.0)
              and np.allclose(zz[kept.sweep_number == 93], off[kept.sweep_number == 93], rtol=0, atol=1e-15))
    ok = (err_a < 1e-9 and zero_b < 1e-12 and plain_b > 0.01 and same_tilts and err_off < 1e-15 and err_xt < 1e-12
          and same_start and ok_nan)
    report("W15 offsets (D-115): noise-free tilts read exactly, X_tilde 0 (plain subtraction leaves the offsets); "
           "tilts, offsets, X_tilde and the start equal independent re-implementations; NaN tilt -> zero offsets", ok,
           "planted %s, |b/b - 1| %.1e; max|X_tilde| %.1e (plain %.3f mV); same tilts %s; |offsets| %.1e; |X_tilde| %.1e; "
           "same start %s (t_SS %.1f ms); NaN ok %s"
           % ({s_: round(v, 4) for s_, v in truth0["ss_tilt"].items()}, err_a, zero_b, plain_b, same_tilts, err_off,
              err_xt, same_start, nwin.start.t_ss_ms, ok_nan))


def main():
    rng = np.random.default_rng(20261023)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "specimen_900000003"
        truth = write_fixture(d, rng)
        truth["dir"] = d
        d_tilt = Path(tmp) / "specimen_900000004"
        truth_tilt = write_fixture(d_tilt, np.random.default_rng(11), sigma_b=0.1, ls_tail_n=LS_LONG_TAIL_N)
        d_small = Path(tmp) / "specimen_900000005"
        truth_small = write_fixture(d_small, np.random.default_rng(13), dropped=SMALL_DROPPED)
        for fn, args in ((_w1, (d, truth)), (_w2, (d, truth)), (_w3, (d, truth)), (_w4, (d, truth)),
                         (_w5, (d, truth)), (_w6, ()), (_w7, (truth,)), (_w8, (d,)), (_w9, (d, d_tilt, truth_tilt)),
                         (_w10, ()), (_w11, (d_tilt, truth_tilt)), (_w12, (d, d_tilt, truth_tilt)),
                         (_w13, (d_small, truth_small)), (_w14, (d, truth)), (_w15, (tmp, d_tilt))):
            try:
                fn(*args)
            except BaseException as exc:          # SystemExit included (D-010 (iii) lineage)
                traceback.print_exc()
                report(fn.__name__ + " raised", False, repr(exc))
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print("SMOKE noise_windows: %d/%d passed" % (n_ok, len(RESULTS)))
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
