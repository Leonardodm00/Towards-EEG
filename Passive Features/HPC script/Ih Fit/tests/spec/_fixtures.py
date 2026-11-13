"""Independent writer-like archive fixtures and reference computations for the
tester's spec tests (written by the integration tester, not by the author).

What is written follows SPEC.md section 3 (data contracts): per SS sweep a
train of 37 pulses, polarity alternating from dep, onset to onset P = 10025
samples at 50 kHz, each stored window (-10, +200) ms around its onset, the
stored windows being the train's pulses that passed the trace QC, in order;
trace_qc_results.csv with one row per detected pulse (pulse_index, polarity,
passed, reason); metadata.json's ss_extraction counts and ls_sweeps; LS sweeps
with a pre-window [t_0, t_on - 5 ms) and a tail after the step.

The archive KEY NAMES (ss_pulses.npz: t, v, polarity_is_dep, sweep_number,
sampling_rate_Hz; metadata ls_sweeps entries: index, sweep_number,
detected_amplitude_pA, sampling_rate_Hz; ls_sweeps.npz: v_<index>, i_<index>,
index) are not stated in SPEC.md; they were read from the loader
(noise_windows.load_ss_windows / load_ls_prewindows) -- see the report's
"Problems in the spec".

Noise: the law (L1)-(L2) simulated here with scipy.signal.lfilter from my own
standard-normal draws (e_0 = sigma xi_0, e_j = rho e_{j-1} + sigma
sqrt(1 - rho^2) xi_j), independently of noise_law.
"""
import csv
import json
import math
import os
from fractions import Fraction

import numpy as np
from scipy.signal import lfilter

FS = 50000.0
DT = 1.0 / FS
PRE_N = 500            # 10 ms before the onset
POST_N = 10000         # 200 ms after it
N_W = PRE_N + POST_N   # 10500 samples per stored window
P = 10025              # onset period, samples (200.5 ms)
TRAIN = 37
THETA_CF = (0.0380, 0.175, 0.0508, 0.0366)   # mV, --, mV, s (SPEC Block 2, M8)


# ---------------------------------------------------------------------------
#  Independent simulation of the law
# ---------------------------------------------------------------------------
def ar1_from_units(xi, sigma, rho):
    """Stationary AR(1) of SD sigma, lag-1 rho from unit draws xi (n_tr, n)."""
    xi = np.atleast_2d(np.asarray(xi, dtype=float))
    if sigma == 0.0:
        return np.zeros_like(xi)
    eta = sigma * math.sqrt(1.0 - rho * rho) * xi
    eta[:, 0] = sigma * xi[:, 0]
    return lfilter([1.0], [1.0, -rho], eta, axis=1)


def law_noise(rng, n, law, n_traces=1):
    """(n_traces, n) noise of law (sigma_f, rho_f, sigma_s, tau_s) at DT."""
    sf, rf, ss, ts = law
    xf = rng.standard_normal((n_traces, n))
    xs = rng.standard_normal((n_traces, n))
    return ar1_from_units(xf, sf, rf) + ar1_from_units(xs, ss, math.exp(-DT / ts))


def gamma(law, dt, k):
    """(L3) autocovariance at lags k (samples)."""
    sf, rf, ss, ts = law
    k = np.abs(np.asarray(k, dtype=float))
    return sf ** 2 * rf ** k + ss ** 2 * np.exp(-k * dt / ts)


def demeaned_moments(law, dt, n, lags):
    """Own O(n) closed forms: Var(xbar), E[S_0], E[S_L] of a demeaned window
    (derived from E[x_i x_j] = gamma(|i-j|)): E[S_0] = n g0 - n Var(xbar);
    E[S_L] = (n-L) g(L) - (1/n) sum_{j<=n-L} (z_j + z_{j+L}) + (n-L) Var(xbar),
    z_i = sum_k gamma(|i-k|)."""
    g = gamma(law, dt, np.arange(n))
    k = np.arange(n)
    var_mean = (g[0] * n + 2.0 * np.sum((n - k[1:]) * g[1:])) / n ** 2
    G = np.cumsum(g)
    i = np.arange(n)
    z = G[i] + G[n - 1 - i] - g[0]
    es0 = n * g[0] - n * var_mean
    out = []
    for L in np.atleast_1d(lags):
        L = int(L)
        if L < 1 or L > n - 1:
            out.append(np.nan)
            continue
        out.append((n - L) * g[L] - (z[:n - L].sum() + z[L:].sum()) / n + (n - L) * var_mean)
    return var_mean, es0, np.array(out)


# ---------------------------------------------------------------------------
#  Writer-like specimen directories
# ---------------------------------------------------------------------------
def ss_sweep_trace(rng, law, tilt_mV_per_s=0.0, A_mV=0.4, tau_ms=28.143, v_rest=-70.3,
                   artefact_mV=0.0, train=TRAIN):
    """One continuous SS sweep: rest + noise + tilt (centred) + an exponential
    response of +A (dep, even positions) / -A (hyp, odd) after each onset,
    plus an optional artefact on [0, 5) ms after each onset."""
    L = (train - 1) * P + N_W
    x = v_rest + law_noise(rng, L, law)[0]
    t = np.arange(L) * DT
    x = x + tilt_mV_per_s * (t - t.mean())
    for k in range(train):
        on = k * P + PRE_N
        sign = 1.0 if k % 2 == 0 else -1.0
        tt_ms = np.arange(L - on) * DT * 1e3
        x[on:] += sign * A_mV * np.exp(-tt_ms / tau_ms)
        if artefact_mV:
            x[on:on + int(round(5e-3 * FS))] += sign * artefact_mV
    return x


def ls_sweep(rng, law, amp_pA, tail_s=1.5, pre_s=0.270, step_s=1.0, tilt_mV_per_s=0.0,
             v_rest=-70.3, R_MOhm=113.0, tau_ms=28.143, onset_index=None):
    """(v, i) of one LS sweep: rest + noise + tilt + an RC step response; the
    current is amp on [onset, onset + step) and 0 elsewhere."""
    on = int(round(pre_s * FS)) if onset_index is None else int(onset_index)
    n_step = int(round(step_s * FS))
    n = on + n_step + int(round(tail_s * FS))
    i = np.zeros(n)
    i[on:on + n_step] = amp_pA
    v = v_rest + law_noise(rng, n, law)[0]
    t = np.arange(n) * DT
    v = v + tilt_mV_per_s * (t - t.mean())
    dV = amp_pA * 1e-12 * R_MOhm * 1e6 * 1e3      # mV
    tt = np.arange(n_step) * DT * 1e3
    v[on:on + n_step] += dV * (1.0 - np.exp(-tt / tau_ms))
    end_level = dV * (1.0 - math.exp(-step_s * 1e3 / tau_ms))
    t2 = np.arange(n - on - n_step) * DT * 1e3
    v[on + n_step:] += end_level * np.exp(-t2 / tau_ms)
    return v, i


def write_cell(root, specimen_id=111, sweeps=(31, 35), law=THETA_CF, ss_tilts=(0.0, 0.0),
               drop=(), A_mV=0.4, tau_ms=28.143, artefact_mV=0.0,
               ls=((60, -50.0), (61, 30.0), (62, 0.0), (63, -90.0)), ls_tail_s=1.5,
               ls_tilts=None, seed=12345, listed_sweeps=True, ls_onset_index=None,
               ls_v_nan_tail=0, ls_i_nan_tail=0):
    """Write a specimen directory specimen_<id>/ under root and return its path.

    drop: iterable of (sweep_index, position) pulses failing the trace QC.
    ls: iterable of (sweep_number, amplitude_pA); ls_tilts: mV/s per LS sweep.
    """
    rng = np.random.default_rng(seed)
    d = os.path.join(root, "specimen_%d" % specimen_id)
    os.makedirs(d, exist_ok=True)
    drop = set((int(a), int(b)) for a, b in drop)
    rows, V, I, pol, swn = [], [], [], [], []
    r = 0
    for s_i, s in enumerate(sweeps):
        x = ss_sweep_trace(rng, law, ss_tilts[s_i], A_mV, tau_ms, artefact_mV=artefact_mV)
        cur = np.zeros(x.size)                         # 0.5 ms pulses of +/-200 pA at each onset
        for k in range(TRAIN):
            on = k * P + PRE_N
            cur[on:on + 25] = 200.0 if k % 2 == 0 else -200.0
        for k in range(TRAIN):
            passed = (s_i, k) not in drop
            rows.append({"pulse_index": r, "polarity": "dep" if k % 2 == 0 else "hyp",
                         "passed": "True" if passed else "False", "reason": "" if passed else "trace_qc"})
            if passed:
                V.append(x[k * P:k * P + N_W])
                I.append(cur[k * P:k * P + N_W])
                pol.append(k % 2 == 0)
                swn.append(s)
            r += 1
    V = np.asarray(V, dtype=np.float64)
    t = (np.arange(N_W) - PRE_N) / FS
    np.savez(os.path.join(d, "ss_pulses.npz"), t=t, v=V, i_pA=np.asarray(I, dtype=np.float64),
             polarity_is_dep=np.asarray(pol, dtype=bool),
             sweep_number=np.asarray(swn, dtype=np.int64), sampling_rate_Hz=np.full(V.shape[0], FS))
    with open(os.path.join(d, "trace_qc_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pulse_index", "polarity", "passed", "reason"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    ls = list(ls)
    if ls_tilts is None:
        ls_tilts = [0.0] * len(ls)
    ls_arrays, ls_meta = {}, []
    for j, ((s, amp), b) in enumerate(zip(ls, ls_tilts)):
        v, i = ls_sweep(rng, law, amp, tail_s=ls_tail_s, tilt_mV_per_s=b, onset_index=ls_onset_index)
        if ls_v_nan_tail:
            v[-ls_v_nan_tail:] = np.nan
        if ls_i_nan_tail:
            i[-ls_i_nan_tail:] = np.nan
        ls_arrays["v_%d" % j] = v
        ls_arrays["i_%d" % j] = i
        ls_meta.append({"index": j, "sweep_number": int(s), "detected_amplitude_pA": float(amp),
                        "sampling_rate_Hz": FS})
    ls_arrays["index"] = np.arange(len(ls))
    np.savez(os.path.join(d, "ls_sweeps.npz"), **ls_arrays)
    ext = {"n_pulses_raw": r, "n_pulses_ampdur_qc": r, "n_pulses_qc": int(V.shape[0]),
           "trace_qc": {"engine": "v2", "alpha": 0.5}}
    if listed_sweeps:
        ext["sweep_numbers"] = [int(s) for s in sweeps]
    meta = {"specimen_id": int(specimen_id), "ss_extraction": ext, "ls_sweeps": ls_meta}
    with open(os.path.join(d, "metadata.json"), "w") as fh:
        json.dump(meta, fh)
    return d


# ---------------------------------------------------------------------------
#  Reference computations
# ---------------------------------------------------------------------------
def tl1_reference(means, centres, sweeps, dep):
    """(TL1) by numpy.linalg.lstsq per sweep (design [t - tbar, 1_dep, 1_hyp]
    with the columns of the polarities present) -> {sweep: (b, se)}."""
    out = {}
    for s in dict.fromkeys(int(x) for x in sweeps):
        g = np.flatnonzero(np.asarray(sweeps) == s)
        t, y, p = np.asarray(centres)[g], np.asarray(means)[g], np.asarray(dep)[g]
        cols = [t - t.mean()]
        for val in (True, False):
            if np.any(p == val):
                cols.append((p == val).astype(float))
        X = np.column_stack(cols)
        n, c = X.shape
        if n <= c or np.linalg.matrix_rank(X) < c:
            out[s] = (np.nan, np.nan)
            continue
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ coef
        s2 = res @ res / (n - c)
        out[s] = (coef[0], math.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0]))
    return out


def dl_exact(y, v):
    """(P3)-(P4)-(D-076.1) in exact rational arithmetic on the float inputs:
    returns floats (Q, tau2_raw, tau2, mu, lam list, y_shrunk list)."""
    y = [Fraction(float(a)) for a in y]
    v = [Fraction(float(a)) for a in v]
    k = len(y)
    w = [1 / a for a in v]
    W = sum(w)
    yfe = sum(wi * yi for wi, yi in zip(w, y)) / W
    Q = sum(wi * (yi - yfe) ** 2 for wi, yi in zip(w, y))
    tau2_raw = (Q - (k - 1)) / (W - sum(wi * wi for wi in w) / W)
    tau2 = max(Fraction(0), tau2_raw)
    wre = [1 / (vi + tau2) for vi in v]
    mu = sum(a * b for a, b in zip(wre, y)) / sum(wre)
    lam = [tau2 / (tau2 + vi) for vi in v]
    ysh = [mu + l * (yi - mu) for l, yi in zip(lam, y)]
    return (float(Q), float(tau2_raw), float(tau2), float(mu), [float(a) for a in lam], [float(a) for a in ysh],
            float(abs(Q) + k - 1), float(W - sum(wi * wi for wi in w) / W))
