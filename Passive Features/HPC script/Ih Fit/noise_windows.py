"""noise_windows -- Block 3 (data layer) of specs/SPEC.md: the archive's SS
layout, the enlarged SS windows, the LS pre-windows, the transient test and
the per-polarity mean-response subtraction (D-051, D-072, D-073, D-076).

What this module does, in plain words. A real cell's Square Subthreshold
(SS) sweep is a train of brief pulses, one every P samples (200.5 ms at
50 kHz). The archive stores, for each pulse that passed the writer's trace
QC, a (-10, +200) ms window around it, with the sweep number but not the
pulse's place in the train. The place is recovered from the writer's own
per-pulse table, `trace_qc_results.csv`, whose rows are the trains in order
and whose `passed` rows are the stored windows (D-072). The noise window of
a pulse is the late part of its stored window, [t_SS(c), 200) ms, which ends
0.5 ms before the next pulse; t_SS(c) is the later of 80 ms (D-076) and the
transient test's time (D-051 (iv)). Every window of a polarity has that
polarity's MEAN WINDOW (the per-sample mean over the windows, a vector over
time) subtracted (D-051.1): that removes the pulse response and leaves the
noise. The Long Square (LS) pre-window of every sweep, [t_0, t_on - 5 ms),
is read as the fitter reads it.

Only loading, geometry and the subtraction live here; the statistics are in
noise_cell_stats.py, the fit in noise_fit.py. No NEURON import: the LS
pre-window is cut from ls_sweeps.npz with the fitter's own rule (the smoke
test checks it against the monolith's loader).

Units: mV, s and Hz inside; ms only in config fields ending in _ms and in
printed text. Positions and periods in samples. Pure ASCII, LF.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


class LayoutError(RuntimeError):
    """The archive's SS files disagree with each other or with the D-072 layout."""


# ===========================================================================
#  Configuration (every knob of Block 3; D-051's provisional values labelled)
# ===========================================================================
@dataclass(frozen=True)
class WindowConfig:
    t_ss_floor_ms: float = 80.0        # D-076 (ii): the SS noise window starts at or after this
    kappa_tr: float = 0.1              # D-051 (iv), PROVISIONAL: |r_odd| < kappa_tr * sigma_SS
    transient_bin_ms: float = 5.0      # D-051 (iv), PROVISIONAL: bins of the transient test
    fallback_t_ms: float = 190.0       # D-051 (i): t_SS >= this -> flag ss_enlarged_none
    transient_floor_z: Optional[float] = None   # None = D-051 as written (threshold kappa_tr sigma_SS);
                                       # z: the threshold is at least z x the noise floor of r_odd, read
                                       # from the EVEN part r_even = (r_dep + r_hyp) / 2, which carries the
                                       # same noise and no pulse response (assistant's proposal 2026-10-23,
                                       # the user's call: with few windows the test's own noise fails it)
    ss_pre_ms: float = 10.0            # the writer's pre-pulse part of each stored window
    ls_pre_gap_ms: float = 5.0         # the LS pre-window ends this long before the step onset
    ls_first_ms: float = 50.0          # D-051: LS first segment
    ls_last_ms: float = 132.5          # D-051: LS last segment
    ls_polarities: Tuple[str, ...] = ("hyp", "dep")   # D-073: a twin of every LS sweep (assistant's choice)
    ls_amp_min_pA: float = 5.0         # a sweep with |amplitude| below this is not a step
    ss_period_samples: Optional[int] = None   # None: measured from adjacent stored windows
    min_overlap: int = 100             # samples two adjacent windows must share to prove adjacency


# ===========================================================================
#  Containers
# ===========================================================================
@dataclass
class SSWindows:
    specimen_id: int
    fs: float                      # Hz
    dt: float                      # s
    t: np.ndarray                  # (n_w,) s, t = 0 at the pulse onset
    v: np.ndarray                  # (K, n_w) mV, stored windows in the archive's order
    polarity_is_dep: np.ndarray    # (K,) bool
    sweep_number: np.ndarray       # (K,) int
    position: np.ndarray           # (K,) int, 0-based place in the sweep's train
    train_length: int              # pulses per sweep (37 in L3_exc)
    period_samples: int            # onset to onset, samples (10025 at 50 kHz)
    pre_n: int                     # samples before the onset in a stored window
    n_raw: int                     # pulses detected by the writer (all sweeps)
    n_sweeps: int

    @property
    def n_windows(self) -> int:
        return int(self.v.shape[0])

    @property
    def n_w(self) -> int:
        return int(self.v.shape[1])


@dataclass
class LSWindows:
    specimen_id: int
    fs: float
    dt: float
    sweep_number: np.ndarray       # (K_LS,) int
    amplitude_pA: np.ndarray       # (K_LS,) float, the writer's detected amplitude
    polarity: np.ndarray           # (K_LS,) 'hyp' | 'dep'
    pre: List[np.ndarray]          # K_LS arrays (n_pre_k,) mV: [t_0, t_on - gap)
    n_pre: np.ndarray              # (K_LS,) int
    onset_s: np.ndarray            # (K_LS,) s, the step onset in the stored sweep


@dataclass
class TransientResult:
    t_tr_ms: float                 # the transient test's own time (first clean bin boundary)
    t_ss_ms: float                 # max(floor, t_tr), the window start used
    sigma_ss_mV: float             # median pre-pulse SD of the cell
    r_odd: np.ndarray              # (n_w,) mV, eq. (27) of the topic doc
    bin_edges_ms: np.ndarray       # (n_bins + 1,)
    bin_exceeds: np.ndarray        # (n_bins,) bool: |bin mean of r_odd| >= the threshold
    flags: Tuple[str, ...]
    threshold_mV: float = float("nan")   # kappa_tr sigma_SS, or z x the noise floor when that is larger
    floor_mV: float = float("nan")       # per-sample noise floor of r_odd, read from r_even (t >= 0)


# ===========================================================================
#  Loading: the SS windows and their layout (D-072)
# ===========================================================================
def _read_qc_table(path: str):
    """Rows of trace_qc_results.csv as (pulse_index, polarity, passed)."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["pulse_index"]), r["polarity"].strip(),
                         r["passed"].strip().lower() == "true"))
    if [r[0] for r in rows] != list(range(len(rows))):
        raise LayoutError("trace_qc_results.csv: pulse_index is not 0..N-1 in order")
    return rows


def load_ss_windows(spec_dir: str, cfg: WindowConfig = WindowConfig()) -> SSWindows:
    """Read ss_pulses.npz (Format A) and place every stored window in its
    sweep's train from trace_qc_results.csv and metadata.json (D-072).

    Raises LayoutError with the reason when the three files disagree.
    """
    npz_path = os.path.join(spec_dir, "ss_pulses.npz")
    qc_path = os.path.join(spec_dir, "trace_qc_results.csv")
    meta_path = os.path.join(spec_dir, "metadata.json")
    for p in (npz_path, qc_path, meta_path):
        if not os.path.isfile(p):
            raise LayoutError("missing %s" % p)
    with np.load(npz_path) as z:
        if "variable_length" in z and bool(np.asarray(z["variable_length"]).ravel()[0]):
            raise LayoutError("ss_pulses.npz is Format B (variable length); Block 3 reads Format A")
        if "sweep_number" not in z:
            raise LayoutError("ss_pulses.npz carries no sweep_number")
        t = np.asarray(z["t"], dtype=np.float64)
        v = np.asarray(z["v"], dtype=np.float64)
        pol = np.asarray(z["polarity_is_dep"], dtype=bool)
        sweep = np.asarray(z["sweep_number"], dtype=np.int64)
        fs_all = np.unique(np.asarray(z["sampling_rate_Hz"], dtype=np.float64))
    if fs_all.size != 1:
        raise LayoutError("more than one sampling rate in ss_pulses.npz: %s" % fs_all)
    fs = float(fs_all[0])
    if v.ndim != 2 or v.shape[0] != pol.size or v.shape[0] != sweep.size or t.size != v.shape[1]:
        raise LayoutError("ss_pulses.npz: inconsistent shapes v %s, t %s, polarity %s, sweep %s"
                          % (v.shape, t.shape, pol.shape, sweep.shape))
    if not np.all(np.isfinite(v)):
        raise LayoutError("ss_pulses.npz: non-finite voltage sample")
    K = v.shape[0]
    with open(meta_path) as fh:
        meta = json.load(fh)
    ext = meta.get("ss_extraction") or {}
    n_raw = int(ext.get("n_pulses_raw", -1))
    n_ampdur = int(ext.get("n_pulses_ampdur_qc", n_raw))
    n_qc = int(ext.get("n_pulses_qc", K))
    rows = _read_qc_table(qc_path)
    if len(rows) != n_raw or len(rows) != n_ampdur:
        raise LayoutError("QC table has %d rows; metadata n_pulses_raw %d, n_pulses_ampdur_qc %d"
                          % (len(rows), n_raw, n_ampdur))
    passed_idx = [r[0] for r in rows if r[2]]
    if len(passed_idx) != K or n_qc != K:
        raise LayoutError("QC table passes %d pulses, metadata n_pulses_qc %d, npz holds %d windows"
                          % (len(passed_idx), n_qc, K))
    # the trains, in the writer's visiting order: metadata's sweep_numbers list when present
    # (a sweep that lost every window is still a train), else the npz's sweep numbers ascending
    listed = ext.get("sweep_numbers")
    if listed:
        sweeps_sorted = np.asarray([int(x) for x in listed], dtype=np.int64)
        if not set(np.unique(sweep).tolist()) <= set(sweeps_sorted.tolist()):
            raise LayoutError("npz sweep numbers %s are not all in metadata sweep_numbers %s"
                              % (np.unique(sweep).tolist(), sweeps_sorted.tolist()))
    else:
        sweeps_sorted = np.unique(sweep)
    n_sweeps = int(sweeps_sorted.size)
    if n_raw % n_sweeps:
        raise LayoutError("n_pulses_raw %d is not a multiple of the %d sweeps" % (n_raw, n_sweeps))
    train = n_raw // n_sweeps
    # Custom: layout from the QC table (no library reads the writer's files).
    # Row r lies in train r // train, at position r % train.
    row_pol_dep = np.array([r[1] == "dep" for r in rows])
    for s_i in range(n_sweeps):
        seg = row_pol_dep[s_i * train:(s_i + 1) * train]
        if np.any(seg[1:] == seg[:-1]):
            raise LayoutError("train %d (sweep %d): polarity does not alternate" % (s_i, sweeps_sorted[s_i]))
    row_sweep = sweeps_sorted[np.asarray(passed_idx) // train]
    row_pos = np.asarray(passed_idx) % train
    if not np.array_equal(row_sweep, sweep):
        raise LayoutError("the passed rows' sweeps (trains in order) differ from the npz sweep numbers")
    if not np.array_equal(row_pol_dep[passed_idx], pol):
        raise LayoutError("the passed rows' polarities differ from the npz polarities")
    pre_n = int(round(cfg.ss_pre_ms * 1e-3 * fs))
    if abs(t[pre_n] - 0.0) > 0.25 / fs or abs(t[0] + pre_n / fs) > 0.25 / fs:
        raise LayoutError("the time axis does not put t = 0 at sample %d (ss_pre_ms = %g)" % (pre_n, cfg.ss_pre_ms))
    ss = SSWindows(specimen_id=int(meta.get("specimen_id", -1)), fs=fs, dt=1.0 / fs, t=t, v=v,
                   polarity_is_dep=pol, sweep_number=sweep, position=row_pos.astype(np.int64),
                   train_length=int(train), period_samples=0, pre_n=pre_n, n_raw=n_raw,
                   n_sweeps=n_sweeps)
    ss.period_samples = measure_period(ss, cfg.min_overlap)
    if cfg.ss_period_samples is not None and int(cfg.ss_period_samples) != ss.period_samples:
        raise LayoutError("measured onset period %d samples, configured %d"
                          % (ss.period_samples, cfg.ss_period_samples))
    return ss


def measure_period(ss: SSWindows, min_overlap: int = 100) -> int:
    """The onset period P (samples) from every pair of stored windows that
    are consecutive in a train: window k+1 reproduces window k shifted by P
    exactly on their n_w - P common samples. Every pair must agree."""
    # Custom: the exact-overlap test of noise_fit_gates.onset_spacing, re-implemented
    # so that the data layer imports nothing from the tools module.
    n_w = ss.n_w
    periods = set()
    for a in range(ss.n_windows - 1):
        b = a + 1
        if ss.sweep_number[a] != ss.sweep_number[b] or ss.position[b] != ss.position[a] + 1:
            continue
        va, vb = ss.v[a], ss.v[b]
        found = None
        # candidate shifts d: where vb's first 16 samples recur in va (vectorised
        # pre-filter, needed on ADC-quantised traces where single values recur),
        # then the full overlap va[d:] == vb[:n_w - d] decides
        head = vb[:16]
        d_max = n_w - max(int(min_overlap), 16)
        views = np.lib.stride_tricks.sliding_window_view(va[:d_max + 16], 16)
        for d in np.flatnonzero(np.all(views == head, axis=1)):
            d = int(d)
            if d >= 1 and np.array_equal(va[d:], vb[:n_w - d]):
                found = d
                break
        if found is None:
            raise LayoutError("windows %d and %d (sweep %d, positions %d, %d) share no overlap of at least %d samples"
                              % (a, b, ss.sweep_number[a], ss.position[a], ss.position[b], min_overlap))
        periods.add(found)
    if not periods:
        raise LayoutError("no adjacent stored windows: the onset period cannot be measured")
    if len(periods) > 1:
        raise LayoutError("adjacent windows disagree on the onset period: %s" % sorted(periods))
    return int(periods.pop())


# ===========================================================================
#  Windows: the transient test, the enlarged window, the subtraction
# ===========================================================================
def transient_start(ss: SSWindows, cfg: WindowConfig = WindowConfig()) -> TransientResult:
    """D-051 (iv) with D-076 (ii): t_SS(c) = max(t_floor, t_tr(c)).

    Each window minus its own pre-pulse mean; r_odd = (mean over dep -
    mean over hyp) / 2 (the pulse response, eq. (27) of the topic doc);
    t_tr is the first bin boundary after which every later 5 ms bin has
    |mean of r_odd over the bin| below the threshold kappa_tr * sigma_SS,
    sigma_SS the median pre-pulse SD. The bin MEAN is the reading of
    D-051's "|r_odd| in 5 ms bins" (a smoothed response; the max over 250
    correlated samples would exceed any threshold near the noise level).
    With cfg.transient_floor_z the threshold is raised to z x the per-sample
    SD of r_even = (mean over dep + mean over hyp) / 2 over t >= 0, the odd
    part's own noise floor (r_even carries the same noise and no response). Flags: 'transient_fail' (no clean bin boundary before the
    end), 'ss_enlarged_none' (t_SS >= fallback_t_ms), 'one_polarity' (r_odd
    from one polarity only), 'floor_binds' (t_tr < t_floor),
    'noise_floor_threshold' (cfg.transient_floor_z raised the threshold).
    """
    # Custom: the transient test (D-051 (iv)).
    pre = ss.v[:, :ss.pre_n]
    r = ss.v - pre.mean(axis=1, keepdims=True)
    sigma_ss = float(np.median(pre.std(axis=1, ddof=1)))
    dep, hyp = ss.polarity_is_dep, ~ss.polarity_is_dep
    flags: List[str] = []
    if dep.any() and hyp.any():
        r_odd = 0.5 * (r[dep].mean(axis=0) - r[hyp].mean(axis=0))
        r_even = 0.5 * (r[dep].mean(axis=0) + r[hyp].mean(axis=0))
        floor = float(np.std(r_even[ss.pre_n:], ddof=1))
    else:
        flags.append("one_polarity")
        r_odd = r.mean(axis=0) * (1.0 if dep.any() else -1.0)
        floor = float(np.std(r_odd[:ss.pre_n], ddof=1)) if ss.pre_n > 1 else 0.0
    t_ms = ss.t * 1e3
    t_end = float(t_ms[-1] + 1e3 / ss.fs)
    edges = np.arange(0.0, t_end + 1e-9, cfg.transient_bin_ms)
    edges = edges[edges <= t_end + 1e-9]
    thr = cfg.kappa_tr * sigma_ss
    if cfg.transient_floor_z is not None and cfg.transient_floor_z * floor > thr:
        thr = cfg.transient_floor_z * floor
        flags.append("noise_floor_threshold")
    exceeds = np.zeros(edges.size - 1, dtype=bool)
    for i in range(edges.size - 1):
        m = (t_ms >= edges[i]) & (t_ms < edges[i + 1])
        exceeds[i] = bool(m.any() and abs(float(np.mean(r_odd[m]))) >= thr)
    # first boundary after which no later bin exceeds
    bad = np.flatnonzero(exceeds)
    if bad.size == 0:
        t_tr = float(edges[0])
    elif bad[-1] + 1 >= exceeds.size:
        t_tr = float(edges[-1])
        flags.append("transient_fail")
    else:
        t_tr = float(edges[bad[-1] + 1])
    t_ss = max(float(cfg.t_ss_floor_ms), t_tr)
    if t_tr < cfg.t_ss_floor_ms:
        flags.append("floor_binds")
    if t_ss >= cfg.fallback_t_ms:
        flags.append("ss_enlarged_none")
    return TransientResult(t_tr_ms=t_tr, t_ss_ms=t_ss, sigma_ss_mV=sigma_ss, r_odd=r_odd,
                           bin_edges_ms=edges, bin_exceeds=exceeds, flags=tuple(flags),
                           threshold_mV=thr, floor_mV=floor)


def enlarged_windows(ss: SSWindows, t_ss_ms: float) -> Tuple[np.ndarray, int]:
    """The enlarged window of every stored pulse: the samples with
    t >= t_SS, i.e. [t_SS, 200) ms after the onset. Returns (X (K, m), j0)."""
    j0 = ss.pre_n + int(round(t_ss_ms * 1e-3 * ss.fs))
    if j0 >= ss.n_w - 1:
        raise ValueError("t_SS = %g ms leaves no window" % t_ss_ms)
    return ss.v[:, j0:].copy(), j0


def subtract_mean_response(X: np.ndarray, polarity_is_dep: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(D-051.1): x_tilde_k = x_k - mean window of x_k's polarity, the mean
    taken per sample over that polarity's windows. Returns (X_tilde, mean_dep,
    mean_hyp); a polarity with no window gets a zero mean."""
    X = np.asarray(X, dtype=np.float64)
    dep = np.asarray(polarity_is_dep, dtype=bool)
    out = np.empty_like(X)
    means = []
    for mask in (dep, ~dep):
        if mask.any():
            mu = X[mask].mean(axis=0)
            out[mask] = X[mask] - mu
        else:
            mu = np.zeros(X.shape[1])
        means.append(mu)
    return out, means[0], means[1]


# ===========================================================================
#  Loading: the LS pre-windows, as the fitter reads them
# ===========================================================================
def _step_onset_index(i_pA: np.ndarray) -> Optional[int]:
    """First sample where |i| exceeds half its maximum (the loader's rule for
    the averaged current; one sweep here)."""
    a = np.abs(np.asarray(i_pA, dtype=np.float64))
    if a.size == 0 or not np.isfinite(a).all() or a.max() <= 0:
        return None
    active = a > 0.5 * a.max()
    return int(np.argmax(active)) if active.any() else None


def load_ls_prewindows(spec_dir: str, cfg: WindowConfig = WindowConfig()) -> LSWindows:
    """One pre-window per Long Square sweep of the polarities in
    cfg.ls_polarities: the samples with t < t_on - ls_pre_gap_ms, t_on the
    step onset detected on that sweep's current (`_estimate_noise`'s rule,
    `_pre_window_mask` of noise_calibration)."""
    meta_path = os.path.join(spec_dir, "metadata.json")
    npz_path = os.path.join(spec_dir, "ls_sweeps.npz")
    with open(meta_path) as fh:
        meta = json.load(fh)
    infos = meta.get("ls_sweeps") or []
    if not infos or not os.path.isfile(npz_path):
        raise LayoutError("no Long Square sweeps in %s" % spec_dir)
    sweeps, amps, pols, pres, n_pre, onsets, fs_seen = [], [], [], [], [], [], set()
    with np.load(npz_path) as z:
        for info in infos:
            amp = info.get("detected_amplitude_pA")
            if amp is None or not np.isfinite(amp) or abs(amp) < cfg.ls_amp_min_pA:
                continue
            polarity = "hyp" if amp < 0 else "dep"
            if polarity not in cfg.ls_polarities:
                continue
            k = int(info["index"])
            v = np.asarray(z["v_%d" % k], dtype=np.float64)
            i = np.asarray(z["i_%d" % k], dtype=np.float64)
            fs = float(info["sampling_rate_Hz"])
            on = _step_onset_index(i)
            if on is None:
                continue
            t_on = on / fs
            end = max(0.0, t_on - cfg.ls_pre_gap_ms * 1e-3)
            n = int(np.count_nonzero(np.arange(v.size) / fs < end))
            if n < 2:
                continue
            fs_seen.add(fs)
            sweeps.append(int(info["sweep_number"])); amps.append(float(amp)); pols.append(polarity)
            pres.append(v[:n].copy()); n_pre.append(n); onsets.append(t_on)
    if not pres:
        raise LayoutError("no usable Long Square pre-window in %s" % spec_dir)
    if len(fs_seen) != 1:
        raise LayoutError("more than one LS sampling rate: %s" % sorted(fs_seen))
    fs = fs_seen.pop()
    return LSWindows(specimen_id=int(meta.get("specimen_id", -1)), fs=fs, dt=1.0 / fs,
                     sweep_number=np.asarray(sweeps, dtype=np.int64),
                     amplitude_pA=np.asarray(amps, dtype=np.float64),
                     polarity=np.asarray(pols), pre=pres,
                     n_pre=np.asarray(n_pre, dtype=np.int64),
                     onset_s=np.asarray(onsets, dtype=np.float64))


def ls_segments(pre: np.ndarray, fs: float, cfg: WindowConfig = WindowConfig()
                ) -> Tuple[np.ndarray, np.ndarray]:
    """(first ls_first_ms, last ls_last_ms) of one pre-window (D-051 (i))."""
    n1 = int(round(cfg.ls_first_ms * 1e-3 * fs))
    n2 = int(round(cfg.ls_last_ms * 1e-3 * fs))
    pre = np.asarray(pre, dtype=np.float64)
    if n1 > pre.size or n2 > pre.size:
        raise ValueError("pre-window of %d samples shorter than the segments (%d, %d)" % (pre.size, n1, n2))
    return pre[:n1].copy(), pre[pre.size - n2:].copy()
