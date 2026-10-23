"""
smoke_noise_windows.py -- smoke suite for noise_windows and noise_cell_stats (W1-W10)
=====================================================================================

Block 3 of specs/SPEC.md (D-051, D-072, D-073, D-076). Builds a writer-like
archive cell in a temporary folder (two SS sweeps of 37 pulses at the real
onset period, with an RC pulse response and Block-2 noise, eleven pulses
dropped by a planted QC table; six Long Square sweeps) so that every check
has a known answer. About 30 s on one core; W4's oracle imports the monolith
(NEURON) and is skipped if that import fails.

  W1  layout (D-072): the stored windows come back with the planted positions,
      sweep numbers and polarities and the period 10025 exactly; a QC table
      with a wrong row count, a swapped polarity, a non-alternating train and a
      wrong configured period each raise LayoutError
  W2  transient test: on the noise-free single-pulse response the first clean
      5 ms bin boundary (bin MEAN of r_odd below kappa sigma_SS) is exactly the
      expected one, floored at 80 ms; a response lasting past 190 ms flags
      ss_enlarged_none; with Block-2 noise and 63 windows the test as written
      (kappa 0.1) is noise-limited (reported, not asserted), and the
      noise-floor option (z = 3 x the even part's SD) gives the response's
      own time at that threshold within two bins
  W3  subtraction (D-051.1): the per-sample mean of the subtracted windows is
      0 per polarity (1e-12); the statistics of response + noise after the
      subtraction equal those of the noise alone after the subtraction to
      1e-10 (the response is a common vector per polarity)
  W4  statistics: window_stats equals np.std(ddof=1) and sample_acf per
      window (1e-14); the 15 labels, blocks, n and K of SPEC 2.2; the LS
      pre-window equals noise_calibration.pre_window_samples(mono, bundle) on
      the same sweep through the monolith (SkipTest without NEURON)
  W5  per-sweep vectors: keys are the sweep numbers, each the median over that
      sweep's windows; the cell vector is the median over all windows
  W6  bootstrap SE: K = 60 iid Gaussian values, B = 2000, 40 repeats: the mean
      SE within 20 % of 1.2533 sigma / sqrt(K); bit-identical for one seed
  W7  semivariogram: 400 sweeps simulated at a known law with the fixture's
      layout: the pair count is the layout's, and V_hat equals
      noise_law.semivariogram(law, dt, m, m, 4 P) within 3.5 SE
  W8  enlarged window: m = n_w - pre_n - round(t_SS f_s / 1e3), first sample at
      t >= t_SS, ends 0.5 ms before the next onset
  W9  LS: six sweeps give 13250-sample pre-windows (265 ms at 50 kHz), hyp and
      dep both read, the 0 pA sweep skipped, segments of 2500 and 6625 samples
  W10 ASCII and LF on noise_windows.py, noise_cell_stats.py and this file

Run:  python smoke_noise_windows.py
Expected last line: "SMOKE noise_windows: 10/10 passed". Exit 1 on any FAIL.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import noise_law as nl                                   # noqa: E402
import noise_stats                                       # noqa: E402
from noise_windows import (LayoutError, WindowConfig, enlarged_windows, load_ls_prewindows,  # noqa: E402
                           load_ss_windows, ls_segments, measure_period, subtract_mean_response,
                           transient_start)
from noise_cell_stats import (LS_FULL_LAGS_MS, LS_LAST_LAGS_MS, SS_LAGS_MS, STAT_BLOCK,  # noqa: E402
                              STAT_LABELS, bootstrap_median_se, ls_window_stats, per_sweep_vectors,
                              ss_window_stats, statistic_vector, window_mean_semivariogram, window_stats)

RESULTS = []
FS = 50000.0
DT = 1.0 / FS
PERIOD = 10025                     # samples, the archive's onset period (G12, 2026-10-20)
TRAIN = 37
PRE_N, POST_N = 500, 10000         # the writer's (-10, +200) ms window
PULSE_N = 25                       # 0.5 ms
REST_N = 25000                     # 0.5 s before the first pulse
TAIL_N = 11000
THETA_CF = nl.NoiseLaw(0.0380, 0.175, 0.0508, 0.0366)
R_IN_MOHM, TAU_M_MS = 113.0, 28.143
DROPPED = {92: (2, 3, 11, 13, 20, 29, 36), 93: (0, 5, 6, 27)}   # planted trace-QC drops
Z_SE = 3.5


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
    """One continuous SS sweep: (v, i, onsets, polarity list, noise, response);
    sigma_b (mV/s) plants a per-sweep tilt (0 = none, as Block 3's checks use)."""
    n = REST_N + (TRAIN - 1) * PERIOD + TAIL_N
    onsets = REST_N + PERIOD * np.arange(TRAIN)
    pol = [k % 2 == 0 for k in range(TRAIN)]                     # dep first, alternating
    i = np.zeros(n)
    for s, dep in zip(onsets, pol):
        i[s:s + PULSE_N] = 200.0 if dep else -200.0
    resp = rc_response(i) if with_response else np.zeros(n)
    units = nl.draw_units(rng, 1, n)
    noise = nl.simulate_sweep(law, DT, n, units, 0, sigma_b=sigma_b, sigma_a=0.0)
    return -60.0 + resp + noise, i, onsets, pol, noise, resp


def write_fixture(d, rng, law=THETA_CF, with_response=True, dropped=DROPPED, sweeps=(92, 93),
                  ls_amps=(-110.0, -90.0, -70.0, 20.0, 50.0, 0.0), sigma_b=0.0):
    """Writes ss_pulses.npz (Format A), trace_qc_results.csv, metadata.json and
    ls_sweeps.npz as the Phase-0 writer does. Returns the planted truth.
    sigma_b (mV/s): a per-sweep tilt in the SS and LS sweeps (Block 4's smoke)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    t_win = (np.arange(PRE_N + POST_N) - PRE_N) / FS
    v_rows, i_rows, pol_rows, sw_rows, pos_rows, noise_rows, resp_rows, qc_rows = [], [], [], [], [], [], [], []
    for sw in sweeps:
        v, i, onsets, pol, noise, resp = make_ss_sweep(law, rng, with_response, sigma_b)
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
    # Long Square sweeps: 270 ms pre, 1 s step, 0.5 s tail
    n_pre, n_step, n_tail = 13500, 50000, 25000
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
                n_pre_ls=n_pre, t_win=t_win)


def expected_t_tr(resp_window, threshold, bin_ms=5.0):
    """First 5 ms boundary after which every later bin's |mean response| < threshold."""
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


def _w2(d, truth, rng):
    ss = load_ss_windows(str(d))
    sigma_ss = float(np.median(ss.v[:, :PRE_N].std(axis=1, ddof=1)))
    resp_dep = single_pulse_response(True)
    # (a) a noise-free cell: the pure single-pulse response after t = 0, white noise of SD sigma_SS
    # before it (so that sigma_SS is the real one and the response is all the test sees)
    ss_clean = load_ss_windows(str(d))
    clean = np.stack([single_pulse_response(p) for p in truth["pol"]])
    clean[:, :PRE_N] += sigma_ss * rng.standard_normal((clean.shape[0], PRE_N))
    ss_clean.v = -60.0 + clean
    tr_clean = transient_start(ss_clean, WindowConfig(t_ss_floor_ms=0.0))
    t_exp_clean = expected_t_tr(resp_dep, 0.1 * tr_clean.sigma_ss_mV)
    ok_a = abs(tr_clean.t_tr_ms - t_exp_clean) <= 1e-9 and tr_clean.t_tr_ms > 0
    tr_floor = transient_start(ss_clean)
    ok_floor = tr_floor.t_ss_ms == max(80.0, tr_clean.t_tr_ms) and (
        ("floor_binds" in tr_floor.flags) == (tr_clean.t_tr_ms < 80.0))
    # (b) a response lasting past 190 ms: tau_m = 200 ms
    long_resp = np.array([rc_response(np.where((np.arange(PRE_N + POST_N) >= PRE_N) & (np.arange(PRE_N + POST_N) < PRE_N + PULSE_N),
                                               200.0 if p else -200.0, 0.0), tau_ms=200.0) for p in truth["pol"]])
    ss_long = load_ss_windows(str(d))
    ss_long.v = -60.0 + 5.0 * long_resp + 1e-6 * rng.standard_normal(long_resp.shape)
    tr_long = transient_start(ss_long)
    ok_b = "ss_enlarged_none" in tr_long.flags and tr_long.t_ss_ms >= 190.0
    # (c) with the real noise: as written (reported), and with the noise-floor option (asserted:
    # the response's own time at the raised threshold, within one bin, no fail flag)
    tr_noisy = transient_start(ss, WindowConfig(t_ss_floor_ms=0.0))
    tr_fz = transient_start(ss, WindowConfig(t_ss_floor_ms=0.0, transient_floor_z=3.0))
    t_exp_fz = expected_t_tr(resp_dep, tr_fz.threshold_mV)
    ok_c = abs(tr_fz.t_tr_ms - t_exp_fz) <= 10.0 + 1e-9 and "transient_fail" not in tr_fz.flags   # two bins: noise near the threshold
    tr_fz80 = transient_start(ss, WindowConfig(transient_floor_z=3.0))              # default floor 80 ms
    ok_c = ok_c and tr_fz80.t_ss_ms == max(80.0, tr_fz.t_tr_ms) and (("floor_binds" in tr_fz80.flags) == (tr_fz.t_tr_ms < 80.0))
    report("W2 transient test: noise-free time exact, floor at 80 ms, long response flagged, noise-floor option",
           ok_a and ok_floor and ok_b and ok_c,
           "clean t_tr %.0f (expected %.0f) ms; floored t_SS %.0f; long-response flags %s; "
           "with noise, kappa 0.1 as written: t_tr %.0f ms, flags %s, noise floor / sigma_SS = %.2f; "
           "with z = 3 (threshold %.2f sigma_SS): t_tr %.0f (response's %.0f) ms, flags %s"
           % (tr_clean.t_tr_ms, t_exp_clean, tr_floor.t_ss_ms, tr_long.flags, tr_noisy.t_tr_ms,
              tr_noisy.flags, tr_noisy.floor_mV / tr_noisy.sigma_ss_mV,
              tr_fz.threshold_mV / tr_fz.sigma_ss_mV, tr_fz.t_tr_ms, t_exp_fz, tr_fz.flags))


def _w3(d, truth):
    ss = load_ss_windows(str(d))
    X, j0 = enlarged_windows(ss, 80.0)
    Xt, mu_dep, mu_hyp = subtract_mean_response(X, ss.polarity_is_dep)
    zero = max(np.abs(Xt[ss.polarity_is_dep].mean(axis=0)).max(), np.abs(Xt[~ss.polarity_is_dep].mean(axis=0)).max())
    Xn = truth["noise"][:, j0:]
    Xnt, _, _ = subtract_mean_response(Xn, ss.polarity_is_dep)
    s_noise = ss_window_stats(Xnt, FS)
    # exact: a response that is a common vector per polarity (the single-pulse response)
    common = np.stack([single_pulse_response(p)[j0:] for p in ss.polarity_is_dep])
    Xc_t, _, _ = subtract_mean_response(Xn + common, ss.polarity_is_dep)
    dev_exact = np.nanmax(np.abs(ss_window_stats(Xc_t, FS) - s_noise))
    # realistic: the fixture's stored windows carry the previous pulses' tails (dropped pulses make
    # them differ between windows), so the residual is small but not zero
    dev_real = np.nanmax(np.abs(ss_window_stats(Xt, FS) - s_noise))
    mu_check = np.abs(mu_dep - (truth["resp"][ss.polarity_is_dep][:, j0:].mean(axis=0) - 60.0
                                + Xn[ss.polarity_is_dep].mean(axis=0))).max()
    report("W3 subtraction (D-051.1): zero per-sample mean per polarity; a common response leaves the noise's statistics",
           zero < 1e-12 and dev_exact < 1e-10 and dev_real < 1e-3 and mu_check < 1e-10,
           "max |mean| %.1e; common response: max |stat - stat(noise)| %.1e; fixture windows (pulse tails differ): %.1e; "
           "mean window check %.1e" % (zero, dev_exact, dev_real, mu_check))


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
    X, _ = enlarged_windows(ss, 80.0)
    Xt, _, _ = subtract_mean_response(X, ss.polarity_is_dep)
    st = ss_window_stats(Xt, FS)
    per = per_sweep_vectors(st, ss.sweep_number)
    ok = set(per) == {92, 93}
    for s, vec in per.items():
        ok = ok and np.allclose(vec, np.median(st[ss.sweep_number == s], axis=0))
    sv = statistic_vector(st, Xt.shape[1], np.zeros((0, 10)), 13250)
    ok = ok and np.allclose(sv.q[:5], np.median(st, axis=0)) and np.all(np.isnan(sv.q[5:])) and list(sv.K[5:]) == [0] * 10
    report("W5 per-sweep vectors: keys 92, 93; medians over the sweep's windows; cell vector over all; empty LS -> NaN, K 0",
           ok, "K per sweep %s" % {s: int((ss.sweep_number == s).sum()) for s in per})


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
    report("W7 semivariogram at 4 periods: pair count %d; Monte Carlo mean within 3.5 SE of the exact value; no cross-sweep pair" % exp_pairs,
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


def _w9(d, truth):
    ls = load_ls_prewindows(str(d))
    ok = (len(ls.pre) == 5 and all(n == 13250 for n in ls.n_pre) and set(ls.polarity) == {"hyp", "dep"}
          and np.all(np.abs(ls.amplitude_pA) >= 5.0))
    first, last = ls_segments(ls.pre[0], FS)
    ok = ok and first.size == 2500 and last.size == 6625 and np.array_equal(last, ls.pre[0][-6625:])
    hyp_only = load_ls_prewindows(str(d), WindowConfig(ls_polarities=("hyp",)))
    ok = ok and len(hyp_only.pre) == 3
    report("W9 LS pre-windows: 5 sweeps of 13250 samples (0 pA skipped), hyp and dep, segments 2500 and 6625; hyp-only 3",
           ok, "n_pre %s, polarities %s" % (sorted(set(ls.n_pre.tolist())), list(ls.polarity)))


def _w10():
    bad = []
    for name in ("noise_windows.py", "noise_cell_stats.py", "smoke_noise_windows.py"):
        raw = (HERE / name).read_bytes()
        if b"\r" in raw or any(b > 126 or (b < 32 and b not in (9, 10)) for b in raw):
            bad.append(name)
    report("W10 ASCII + LF on the three files", not bad, "bad: %s" % bad if bad else "ok")


def main():
    rng = np.random.default_rng(20261023)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "specimen_900000003"
        truth = write_fixture(d, rng)
        truth["dir"] = d
        for fn, args in ((_w1, (d, truth)), (_w2, (d, truth, rng)), (_w3, (d, truth)), (_w4, (d, truth)),
                         (_w5, (d, truth)), (_w6, ()), (_w7, (truth,)), (_w8, (d,)), (_w9, (d, truth)), (_w10, ())):
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
