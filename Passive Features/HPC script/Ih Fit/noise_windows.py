"""noise_windows -- Block 3 (data layer) of specs/SPEC.md: the archive's SS
layout, the SS window start and the enlarged windows, the subtraction of the
group median window, and the LS pre-windows and end windows (D-051, D-072,
D-073, D-076, D-095, D-103, D-104, D-106, D-107, D-110, D-111).

What this module does, in plain words. A real cell's Square Subthreshold
(SS) sweep is a train of brief pulses, one every P samples (200.5 ms at
50 kHz). The archive stores, for each pulse that passed the writer's trace
QC, a (-10, +200) ms window around it, with the sweep number but not the
pulse's place in the train. The place is recovered from the writer's own
per-pulse table, `trace_qc_results.csv`, whose rows are the trains in order
and whose `passed` rows are the stored windows (D-072).

The noise window of a pulse is the late part of its stored window,
[t_SS(c), 200) ms. The windows are grouped by sweep and polarity, and a
group of fewer than 5 windows leaves everything (D-106). Each group's MEDIAN
WINDOW -- the per-sample median over its windows -- shows the pulse
response; an exponential fitted exactly to its decay (least squares by
variable projection, D-110) says when the response has fallen to kappa_tr
times the pre-pulse noise SD, and the cell's start is the median of those
times over its groups (D-111), at least 80 ms, and 80 ms again when the
window left would be shorter than 50 ms (D-103, D-104). Every window then has
its group's median window subtracted, which removes the response and leaves
the noise (D-103.1). The Long Square (LS) pre-window of every sweep, [t_0,
t_on - 5 ms), is read as the fitter reads it, and the sweep's last 500 ms
(the end window) for its tilt (D-107).

Only loading, geometry, the start and the subtraction live here; the
statistics and the tilts are in noise_cell_stats.py, the fit in noise_fit.py.
No NEURON import: the LS windows are cut from ls_sweeps.npz with the fitter's
own rule (the smoke test checks it against the monolith's loader).

Units: mV, s and Hz inside; ms in config fields ending in _ms, in the
start's results and in printed text. Positions and periods in samples. Pure
ASCII, LF.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, minimize_scalar


class LayoutError(RuntimeError):
    """The archive's SS files disagree with each other or with the D-072 layout."""


# ===========================================================================
#  Configuration (every knob of Block 3; provisional values labelled)
# ===========================================================================
@dataclass(frozen=True)
class WindowConfig:
    # --- the SS window start, SPEC 2.2 (a)-(e) ---
    t_ss_floor_ms: float = 80.0        # D-076 (ii), kept by D-103.3: t_SS(c) >= this
    kappa_tr: float = 0.1              # D-051, PROVISIONAL: a group's start is where its decay is within
                                       # kappa_tr * sigma_SS_hat(c) (ST1)
    fit_start_ms: float = 5.0          # D-103.2, PROVISIONAL: the decay fit uses the samples with t >= this
    decay_tau_bounds_ms: Tuple[float, float] = (1.0, 1000.0)   # D-110, PROVISIONAL: tau's search interval
    decay_grid_n: int = 301            # D-110, PROVISIONAL: log-spaced grid points over the bounds
    min_window_ms: float = 50.0        # D-104.1: a window [t_star, 200) ms shorter than this -> the fallback
    fallback_t_ms: float = 80.0        # D-103.3 (was 190 ms, D-051 (i)): the fallback start, flagged
    min_windows_per_group: int = 5     # D-104.2, D-106.1, PROVISIONAL: a smaller group leaves everything
    # --- the stored SS windows ---
    ss_pre_ms: float = 10.0            # the writer's pre-pulse part of each stored window
    ss_period_samples: Optional[int] = None   # None: measured from adjacent stored windows
    min_overlap: int = 100             # samples two adjacent windows must share to prove adjacency
    # --- the LS sweeps ---
    ls_pre_gap_ms: float = 5.0         # the LS pre-window ends this long before the step onset
    ls_first_ms: float = 50.0          # D-051: LS first segment
    ls_last_ms: float = 132.5          # D-051: LS last segment
    ls_end_ms: float = 500.0           # D-107.1: the end window of the LS tilt (Allen's 500 ms; length open)
    ls_polarities: Tuple[str, ...] = ("hyp", "dep")   # D-073: a twin of every LS sweep (assistant's choice)
    ls_amp_min_pA: float = 5.0         # a sweep with |amplitude| below this is not a step
    # --- LEGACY: read only by transient_start (D-051 (iv), superseded by D-103) ---
    transient_bin_ms: float = 5.0      # bins of the legacy transient test
    transient_floor_z: Optional[float] = None   # the legacy test's noise-floor option (SPEC at 1aabe3e, 2.2)
    transient_fallback_t_ms: float = 190.0      # the legacy test's ss_enlarged_none (D-051 (i))


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
    end: List[np.ndarray]          # K_LS arrays (n_end,) mV: the end window of (TL2); empty when unusable
    end_start_s: np.ndarray        # (K_LS,) s from the sweep's first sample; NaN when unusable
    step_end_s: np.ndarray         # (K_LS,) s, t_off: the first sample after the step
    n_samples: np.ndarray          # (K_LS,) int, samples stored in the sweep
    end_status: np.ndarray         # (K_LS,) 'ok' | 'tail_short'


@dataclass
class TransientResult:
    """LEGACY: the result of transient_start (superseded by D-103)."""
    t_tr_ms: float                 # the transient test's own time (first clean bin boundary)
    t_ss_ms: float                 # max(floor, t_tr), the window start used
    sigma_ss_mV: float             # median pre-pulse SD of the cell
    r_odd: np.ndarray              # (n_w,) mV, eq. (27) of the topic doc
    bin_edges_ms: np.ndarray       # (n_bins + 1,)
    bin_exceeds: np.ndarray        # (n_bins,) bool: |bin mean of r_odd| >= the threshold
    flags: Tuple[str, ...]
    threshold_mV: float = float("nan")   # kappa_tr sigma_SS, or z x the noise floor when that is larger
    floor_mV: float = float("nan")       # per-sample noise floor of r_odd, read from r_even (t >= 0)


@dataclass
class DecayFit:
    """One median window's exact least-squares fit B + A exp(-t / tau) (D-103.2, D-110)."""
    A_mV: float                    # amplitude at the onset (t = 0) of the fitted exponential
    tau_ms: float
    B_mV: float                    # the baseline
    rss: float                     # residual sum of squares over the fitted samples, mV^2
    ok: bool                       # an interior minimum of RSS(tau) was found within the bounds
    at_bound: bool                 # the grid's minimum lay at a bound: no resolvable decay (D-110.2)
    message: str


@dataclass
class StartResult:
    """The SS window start of one cell, SPEC 2.2 (a)-(e)."""
    t_ss_ms: float                 # the start used (ST2)
    t_star_ms: float               # max(t_floor, median over valid groups of t_sp) (D-111); NaN when no group is valid
    sigma_ss_mV: float             # median pre-pulse SD over the kept windows (D-095.2)
    keep: np.ndarray               # (K,) bool: False for the windows of the small groups (D-106.1)
    group_sweep: np.ndarray        # (G,) int
    group_is_dep: np.ndarray       # (G,) bool
    group_K: np.ndarray            # (G,) int
    group_status: Tuple[str, ...]  # (G,) 'ok' | 'few_windows' | 'fit'
    A_mV: np.ndarray               # (G,) NaN where no valid fit
    tau_ms: np.ndarray             # (G,)
    B_mV: np.ndarray               # (G,)
    t_group_ms: np.ndarray         # (G,) t_sp of (ST1); NaN where no valid fit
    flags: Tuple[str, ...]
    fallback_reason: str           # '' | 'short_window' | 'no_valid_group'

    @property
    def n_groups(self) -> int:
        return int(self.group_sweep.size)


@dataclass
class SSNoiseWindows:
    """The SS noise windows of one cell, ready for the statistics (SPEC 2.2 (a)-(g))."""
    start: StartResult
    ss: SSWindows                  # the kept windows (D-106.1)
    X_tilde: np.ndarray            # (K_kept, m) mV, each minus its group's median window
    j0: int                        # first sample of the enlarged window in a stored window

    @property
    def m(self) -> int:
        return int(self.X_tilde.shape[1])


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
#  Windows: the start (D-103, D-104, D-106), the enlarged window, the subtraction
# ===========================================================================
def profile_decay(t_ms: np.ndarray, y_mV: np.ndarray, tau_ms: float) -> Tuple[float, float, float]:
    """(A, B, RSS) of the linear least-squares fit y = B + A exp(-t / tau) at
    a fixed tau: the closed form of the centred regression of y on e =
    exp(-t / tau). RSS in mV^2; (NaN, NaN, inf) when e is constant."""
    A, B, rss, _, _ = _profile(np.asarray(t_ms, dtype=np.float64), np.asarray(y_mV, dtype=np.float64), float(tau_ms))
    return A, B, rss


def _profile(t: np.ndarray, y: np.ndarray, tau: float):
    e = np.exp(-t / tau)
    ec = e - e.mean()
    see = float(ec @ ec)
    if not see > 0.0:
        return float("nan"), float("nan"), float("inf"), None, e
    A = float(ec @ (y - y.mean())) / see
    B = float(y.mean() - A * e.mean())
    r = y - B - A * e
    return A, B, float(r @ r), r, e


def _drss_dtau(t: np.ndarray, y: np.ndarray, tau: float) -> float:
    """(ST0): dRSS/dtau of the profiled residual, -(2 A / tau^2) sum r e t
    (the envelope theorem: A and B are optimal at every tau)."""
    A, _, _, r, e = _profile(t, y, tau)
    if r is None:
        return float("nan")
    return -2.0 * A / tau ** 2 * float(r @ (e * t))


def fit_decay(t_ms: np.ndarray, y_mV: np.ndarray, cfg: WindowConfig = WindowConfig()) -> DecayFit:
    """(D-103.2, D-110) The exact least-squares fit of y = B + A exp(-t / tau)
    to the samples given (t in ms from the onset, y in mV), tau in
    cfg.decay_tau_bounds_ms, by variable projection: (A, B) in closed form
    at each tau (profile_decay); tau at the minimum of the profiled RSS on a
    grid of cfg.decay_grid_n log-spaced points, refined by a bounded Brent
    minimisation between the grid minimum's neighbours and polished to the
    root of dRSS/dtau (ST0). No starting value enters. ok = False, at_bound
    = True when the grid's minimum is its first or last point (D-110.2)."""
    # Custom: variable projection of the separable model (no SciPy routine profiles out the linear
    # parameters); the 1-D steps are scipy.optimize.minimize_scalar and brentq.
    t = np.asarray(t_ms, dtype=np.float64)
    y = np.asarray(y_mV, dtype=np.float64)
    nan = float("nan")
    if t.ndim != 1 or t.shape != y.shape or t.size < 4:
        raise ValueError("fit_decay: t and y must be 1-D of equal length >= 4")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        return DecayFit(nan, nan, nan, nan, False, False, "non-finite sample")
    lo, hi = (float(b) for b in cfg.decay_tau_bounds_ms)
    if not 0.0 < lo < hi or int(cfg.decay_grid_n) < 3:
        raise ValueError("fit_decay: need 0 < tau_lo < tau_hi and decay_grid_n >= 3")
    grid = np.geomspace(lo, hi, int(cfg.decay_grid_n))
    rss = np.array([_profile(t, y, g)[2] for g in grid])
    i = int(np.argmin(rss))
    if not np.isfinite(rss[i]):
        return DecayFit(nan, nan, nan, nan, False, False, "no finite residual on the grid")
    if i == 0 or i == grid.size - 1:
        return DecayFit(nan, nan, nan, float(rss[i]), False, True,
                        "the minimum of RSS(tau) lies at the bound tau = %g ms: no resolvable decay" % grid[i])
    res = minimize_scalar(lambda lt: _profile(t, y, float(np.exp(lt)))[2],
                          bounds=(float(np.log(grid[i - 1])), float(np.log(grid[i + 1]))),
                          method="bounded", options={"xatol": 1e-12, "maxiter": 500})
    tau = float(np.exp(res.x))
    a, b = tau * (1.0 - 1e-6), tau * (1.0 + 1e-6)
    ga, gb = _drss_dtau(t, y, a), _drss_dtau(t, y, b)
    note = "bounded Brent"
    if np.isfinite(ga) and np.isfinite(gb) and ga * gb < 0.0:
        tau = float(brentq(lambda x: _drss_dtau(t, y, x), a, b, xtol=1e-13, maxiter=200))
        note = "bounded Brent, polished to the root of dRSS/dtau"
    A, B, r_ss, _, _ = _profile(t, y, tau)
    return DecayFit(A, tau, B, r_ss, True, False, note)


def _groups(ss: SSWindows) -> List[Tuple[Tuple[int, bool], np.ndarray]]:
    """((sweep, is_dep), window indices) of every group present, in the order
    of the sweeps' first windows, dep before hyp."""
    out = []
    for s in dict.fromkeys(int(x) for x in ss.sweep_number):
        for dep in (True, False):
            idx = np.flatnonzero((ss.sweep_number == s) & (ss.polarity_is_dep == dep))
            if idx.size:
                out.append(((s, dep), idx))
    return out


def decay_start(ss: SSWindows, cfg: WindowConfig = WindowConfig(),
                fit: Callable[[np.ndarray, np.ndarray, WindowConfig], DecayFit] = fit_decay) -> StartResult:
    """The SS window start of the cell, SPEC 2.2 (a)-(e) (D-095, D-103,
    D-104, D-106, D-110, D-111): small groups out of everything;
    sigma_SS_hat(c) over the kept windows; per kept group the median window
    and its exact decay fit (`fit`, swappable); t_sp of (ST1); t_SS(c) of
    (ST2): the MEDIAN of the valid groups' t_sp, the floor and the fallback. Flags: 'ss_group_dropped:<sweep>:<polarity>:few_windows' or
    ':fit', 'ss_start_floor', 'ss_start_fallback' (reason in
    fallback_reason)."""
    # Custom: the start rule (ST1)-(ST2) (no library function).
    groups = _groups(ss)
    G = len(groups)
    keep = np.ones(ss.n_windows, dtype=bool)
    for (_, _), idx in groups:
        if idx.size < cfg.min_windows_per_group:
            keep[idx] = False
    pre = ss.v[keep, :ss.pre_n]
    sigma_ss = float(np.median(pre.std(axis=1, ddof=1))) if pre.shape[0] else float("nan")
    thr = float(cfg.kappa_tr) * sigma_ss
    t_ms = np.asarray(ss.t, dtype=np.float64) * 1e3
    fit_sel = np.arange(ss.n_w) >= ss.pre_n + int(round(cfg.fit_start_ms * 1e-3 * ss.fs))
    A, tau, B, t_g = (np.full(G, np.nan) for _ in range(4))
    status: List[str] = []
    flags: List[str] = []
    for g, ((s, dep), idx) in enumerate(groups):
        pol = "dep" if dep else "hyp"
        if idx.size < cfg.min_windows_per_group:
            status.append("few_windows")
            flags.append("ss_group_dropped:%d:%s:few_windows" % (s, pol))
            continue
        med = np.median(ss.v[idx], axis=0)
        f = fit(t_ms[fit_sel], med[fit_sel], cfg)
        if not f.ok:
            status.append("fit")
            flags.append("ss_group_dropped:%d:%s:fit" % (s, pol))
            continue
        A[g], tau[g], B[g] = f.A_mV, f.tau_ms, f.B_mV
        with np.errstate(divide="ignore"):
            t_g[g] = f.tau_ms * np.log(abs(f.A_mV) / thr) if abs(f.A_mV) > thr else 0.0    # (ST1)
        status.append("ok")
    valid = np.array([st == "ok" for st in status], dtype=bool)
    t_end = (ss.n_w - ss.pre_n) / ss.fs * 1e3                      # 200 ms
    reason = ""
    if valid.any():
        t_med = float(np.median(t_g[valid]))                         # D-111: the median over valid groups
        t_star = max(float(cfg.t_ss_floor_ms), t_med)
        if t_end - t_star >= cfg.min_window_ms:                      # (ST2)
            t_ss = t_star
            if t_med < cfg.t_ss_floor_ms:
                flags.append("ss_start_floor")
        else:
            t_ss, reason = float(cfg.fallback_t_ms), "short_window"
    else:
        t_star, t_ss, reason = float("nan"), float(cfg.fallback_t_ms), "no_valid_group"
    if reason:
        flags.append("ss_start_fallback")
    return StartResult(t_ss_ms=float(t_ss), t_star_ms=float(t_star), sigma_ss_mV=sigma_ss, keep=keep,
                       group_sweep=np.array([k[0] for k, _ in groups], dtype=np.int64),
                       group_is_dep=np.array([k[1] for k, _ in groups], dtype=bool),
                       group_K=np.array([idx.size for _, idx in groups], dtype=np.int64),
                       group_status=tuple(status), A_mV=A, tau_ms=tau, B_mV=B, t_group_ms=t_g,
                       flags=tuple(flags), fallback_reason=reason)


def select_windows(ss: SSWindows, keep: np.ndarray) -> SSWindows:
    """A copy of `ss` holding the windows where keep is True (D-106.1); the
    layout fields (period, train length, pre_n, n_raw, n_sweeps) unchanged."""
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (ss.n_windows,):
        raise ValueError("keep must be (%d,), got %s" % (ss.n_windows, keep.shape))
    return replace(ss, v=ss.v[keep].copy(), polarity_is_dep=ss.polarity_is_dep[keep].copy(),
                   sweep_number=ss.sweep_number[keep].copy(), position=ss.position[keep].copy())


def enlarged_windows(ss: SSWindows, t_ss_ms: float) -> Tuple[np.ndarray, int]:
    """The enlarged window of every stored pulse: the samples with
    t >= t_SS, i.e. [t_SS, 200) ms after the onset. Returns (X (K, m), j0)."""
    j0 = ss.pre_n + int(round(t_ss_ms * 1e-3 * ss.fs))
    if j0 >= ss.n_w - 1:
        raise ValueError("t_SS = %g ms leaves no window" % t_ss_ms)
    return ss.v[:, j0:].copy(), j0


def subtract_median_response(X: np.ndarray, polarity_is_dep: np.ndarray, sweep_number: np.ndarray
                             ) -> Tuple[np.ndarray, Dict[Tuple[int, bool], np.ndarray]]:
    """(D-103.1, SPEC 2.2 (g)): x_tilde_k = x_k - m_sp, m_sp the per-sample
    median over the windows of k's group (its sweep and polarity). Returns
    (X_tilde, {(sweep, is_dep): m_sp}). Applied alike to the real windows and
    to the fit's simulated ones (D-051 (iv), D-033)."""
    X = np.asarray(X, dtype=np.float64)
    dep = np.asarray(polarity_is_dep, dtype=bool)
    sw = np.asarray(sweep_number)
    if X.ndim != 2 or dep.shape != (X.shape[0],) or sw.shape != (X.shape[0],):
        raise ValueError("X must be (K, m) with polarity_is_dep and sweep_number of shape (K,)")
    out = np.empty_like(X)
    medians: Dict[Tuple[int, bool], np.ndarray] = {}
    for s in dict.fromkeys(int(x) for x in sw):
        for p in (True, False):
            idx = np.flatnonzero((sw == s) & (dep == p))
            if idx.size:
                m_sp = np.median(X[idx], axis=0)
                out[idx] = X[idx] - m_sp
                medians[(s, p)] = m_sp
    return out, medians


def ss_noise_windows(ss: SSWindows, cfg: WindowConfig = WindowConfig(),
                     fit: Callable[[np.ndarray, np.ndarray, WindowConfig], DecayFit] = fit_decay
                     ) -> SSNoiseWindows:
    """SPEC 2.2 (a)-(g) in order: the start (decay_start), the kept windows
    (select_windows), their enlarged parts (enlarged_windows) and the
    subtraction of each group's median window (subtract_median_response)."""
    start = decay_start(ss, cfg, fit)
    kept = select_windows(ss, start.keep)
    X, j0 = enlarged_windows(kept, start.t_ss_ms)
    X_tilde, _ = subtract_median_response(X, kept.polarity_is_dep, kept.sweep_number)
    return SSNoiseWindows(start=start, ss=kept, X_tilde=X_tilde, j0=j0)


# ===========================================================================
#  LEGACY (superseded by D-103; unused by the pipeline): kept so the dev
#  checks of 2026-10-23 to 2026-10-29 re-run on the current code
# ===========================================================================
def transient_start(ss: SSWindows, cfg: WindowConfig = WindowConfig()) -> TransientResult:
    """LEGACY (superseded by D-095 / D-103; unused by the pipeline, kept so the
    dev checks of 2026-10-23 to 2026-10-29 re-run). D-051 (iv) with D-076
    (ii): t_SS(c) = max(t_floor, t_tr(c)).

    Each window minus its own pre-pulse mean; r_odd = (mean over dep -
    mean over hyp) / 2 (the pulse response, eq. (27) of the topic doc);
    t_tr is the first bin boundary after which every later 5 ms bin has
    |mean of r_odd over the bin| below the threshold kappa_tr * sigma_SS,
    sigma_SS the median pre-pulse SD. The bin MEAN is the reading of
    D-051's "|r_odd| in 5 ms bins" (a smoothed response; the max over 250
    correlated samples would exceed any threshold near the noise level).
    With cfg.transient_floor_z the threshold is raised to z x the per-sample
    SD of r_even = (mean over dep + mean over hyp) / 2 over t >= 0, the odd
    part's own noise floor (r_even carries the same noise and no response --
    except in the window's last ~40 ms and for the tilt's ramp, corrected
    2026-10-23, SPEC 2.2). Flags: 'transient_fail' (no clean bin boundary before the
    end), 'ss_enlarged_none' (t_SS >=
    transient_fallback_t_ms, 190 ms), 'one_polarity' (r_odd
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
    if t_ss >= cfg.transient_fallback_t_ms:
        flags.append("ss_enlarged_none")
    return TransientResult(t_tr_ms=t_tr, t_ss_ms=t_ss, sigma_ss_mV=sigma_ss, r_odd=r_odd,
                           bin_edges_ms=edges, bin_exceeds=exceeds, flags=tuple(flags),
                           threshold_mV=thr, floor_mV=floor)


def subtract_mean_response(X: np.ndarray, polarity_is_dep: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LEGACY (superseded by D-103.1, the group median window: see
    subtract_median_response; unused by the pipeline, kept for the dev checks).
    (D-051.1): x_tilde_k = x_k - mean window of x_k's polarity, the mean
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
#  Loading: the LS pre-windows, as the fitter reads them, and their end windows
# ===========================================================================
def _step_onset_index(i_pA: np.ndarray) -> Optional[int]:
    """First sample where |i| exceeds half its maximum (the loader's rule for
    the averaged current; one sweep here)."""
    a = np.abs(np.asarray(i_pA, dtype=np.float64))
    if a.size == 0 or not np.isfinite(a).all() or a.max() <= 0:
        return None
    active = a > 0.5 * a.max()
    return int(np.argmax(active)) if active.any() else None


def _step_end_index(i_pA: np.ndarray) -> Optional[int]:
    """t_off of (TL2): the first sample after the last one where |i| exceeds
    half its maximum."""
    a = np.abs(np.asarray(i_pA, dtype=np.float64))
    if a.size == 0 or not np.isfinite(a).all() or a.max() <= 0:
        return None
    active = np.flatnonzero(a > 0.5 * a.max())
    return int(active[-1]) + 1 if active.size else None


def load_ls_prewindows(spec_dir: str, cfg: WindowConfig = WindowConfig()) -> LSWindows:
    """One pre-window per Long Square sweep of the polarities in
    cfg.ls_polarities: the samples with t < t_on - ls_pre_gap_ms, t_on the
    step onset detected on that sweep's current (`_estimate_noise`'s rule,
    `_pre_window_mask` of noise_calibration); and its end window for (TL2):
    the last ls_end_ms of the stored sweep, ending at its last finite sample,
    usable ('ok') only when it starts after t_off (the rule of
    ls_baseline_qc.baseline_record), else 'tail_short' with an empty array."""
    meta_path = os.path.join(spec_dir, "metadata.json")
    npz_path = os.path.join(spec_dir, "ls_sweeps.npz")
    with open(meta_path) as fh:
        meta = json.load(fh)
    infos = meta.get("ls_sweeps") or []
    if not infos or not os.path.isfile(npz_path):
        raise LayoutError("no Long Square sweeps in %s" % spec_dir)
    sweeps, amps, pols, pres, n_pre, onsets, fs_seen = [], [], [], [], [], [], set()
    ends, end_start, step_end, n_samples, end_status = [], [], [], [], []
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
            off = _step_end_index(i)
            if on is None or off is None:
                continue
            t_on = on / fs
            end = max(0.0, t_on - cfg.ls_pre_gap_ms * 1e-3)
            n = int(np.count_nonzero(np.arange(v.size) / fs < end))
            if n < 2:
                continue
            fs_seen.add(fs)
            sweeps.append(int(info["sweep_number"])); amps.append(float(amp)); pols.append(polarity)
            pres.append(v[:n].copy()); n_pre.append(n); onsets.append(t_on)
            # Custom: the end window of (TL2), the rule of ls_baseline_qc.baseline_record (re-implemented so
            # the data layer imports no pandas); the smoke compares the two.
            finite = np.flatnonzero(np.isfinite(v))
            last = int(finite[-1]) + 1 if finite.size else 0
            n_end = int(round(cfg.ls_end_ms * 1e-3 * fs))
            i0 = last - n_end
            if n_end >= 3 and i0 >= 0 and i0 > off:
                ends.append(v[i0:last].copy()); end_start.append(i0 / fs); end_status.append("ok")
            else:
                ends.append(np.zeros(0)); end_start.append(float("nan")); end_status.append("tail_short")
            step_end.append(off / fs); n_samples.append(int(v.size))
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
                     onset_s=np.asarray(onsets, dtype=np.float64),
                     end=ends, end_start_s=np.asarray(end_start, dtype=np.float64),
                     step_end_s=np.asarray(step_end, dtype=np.float64),
                     n_samples=np.asarray(n_samples, dtype=np.int64), end_status=np.asarray(end_status))


def ls_segments(pre: np.ndarray, fs: float, cfg: WindowConfig = WindowConfig()
                ) -> Tuple[np.ndarray, np.ndarray]:
    """(first ls_first_ms, last ls_last_ms) of one pre-window (D-051 (i))."""
    n1 = int(round(cfg.ls_first_ms * 1e-3 * fs))
    n2 = int(round(cfg.ls_last_ms * 1e-3 * fs))
    pre = np.asarray(pre, dtype=np.float64)
    if n1 > pre.size or n2 > pre.size:
        raise ValueError("pre-window of %d samples shorter than the segments (%d, %d)" % (pre.size, n1, n2))
    return pre[:n1].copy(), pre[pre.size - n2:].copy()
