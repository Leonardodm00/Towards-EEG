"""noise_fit -- Block 4 of specs/SPEC.md: the per-cell fit of the noise law
by indirect inference with the simulated inverse-covariance weight (D-033,
D-048 to D-051, D-056; the layout and the simulator D-103, D-106, D-107.2;
section 2.3 of the spec, items (a)-(h)).

In plain words: the real cell gives one vector of 15 noise statistics,
q_hat(c). For a candidate law theta we simulate a cell with exactly the real
cell's layout (its SS sweeps as continuous 37-pulse trains, its kept windows
at their positions, its LS pre-windows), give every simulated sweep the tilt
read on its real sweep (D-107.2), subtract from each window its group's
median window as the data had it subtracted (D-103.1), reduce it with the
same estimators, and move theta until the simulated vector matches the real
one, each statistic weighted by how much it would scatter if the law were
true (the inverse of the simulated covariance). Nothing about the tilt is
estimated here. The per-sweep stage of D-076 (i) is Block 4b
(noise_sweep_fit).

Layers: model and analysis only -- no file I/O (an optional `log` callable
is the only output channel besides the result), no plotting. Units: theta in
mV, --, mV, s; tilts in mV/s; the fit coordinates u = (log sigma_f, rho_f,
log sigma_s, log tau_s). Pure ASCII, LF.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg, optimize
from scipy.stats import chi2
from scipy.stats.mstats import hdquantiles

import noise_cell_stats as ncs
import noise_law
from noise_law import NoiseLaw
from noise_cell_stats import SweepTilts
from noise_windows import LSWindows, SSNoiseWindows, SSWindows, WindowConfig, subtract_median_response

LOG_1E4 = math.log(1e-4)
LOG_10 = math.log(10.0)
N_STATS = len(ncs.STAT_LABELS)       # 15
N_SS = ncs.N_SS_STATS                # 5
N_LS = ncs.N_LS_STATS                # 10


# ===========================================================================
#  Configuration and the fit coordinates
# ===========================================================================
@dataclass(frozen=True)
class NoiseFitConfig:
    """Section 2.3 and the Block 4 entry. Defaults are D-048 to D-051 and
    D-056 (M, R, B, k_max, tol_u, three starts); the rest are the assistant's
    choices flagged in 2.3 (c)-(e)."""
    M_binding: int = 3000
    R_iter: int = 600
    R_final: int = 3000
    B_boot: int = 2000
    k_max: int = 6
    tol_u: float = 0.01
    starts_after_first: int = 1          # 1: warm start from theta_hat_k; 3: plus tau_s / 2 and 2 tau_s
    max_nfev: int = 15                   # trf iterations; each costs 5 evaluations of q_sim (2-point Jacobian)
    ftol: float = 1e-4
    xtol: float = 1e-3
    gtol: float = 1e-8
    fd_rel_step: Tuple[float, float, float, float] = (1e-3, 1e-2, 1e-3, 1e-3)
    log_sigma_bounds: Tuple[float, float] = (LOG_1E4, LOG_10)
    rho_bounds: Tuple[float, float] = (1e-3, 0.999)
    log_tau_bounds: Tuple[float, float] = (LOG_1E4, LOG_10)
    chunk_windows: int = 500
    seed: int = 20261023
    window_cfg: WindowConfig = WindowConfig()

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([self.log_sigma_bounds[0], self.rho_bounds[0], self.log_sigma_bounds[0], self.log_tau_bounds[0]])
        hi = np.array([self.log_sigma_bounds[1], self.rho_bounds[1], self.log_sigma_bounds[1], self.log_tau_bounds[1]])
        return lo, hi


def law_to_u(law: NoiseLaw) -> np.ndarray:
    """u = (log sigma_f, rho_f, log sigma_s, log tau_s)."""
    return np.array([math.log(law.sigma_f), law.rho_f, math.log(law.sigma_s), math.log(law.tau_s)])


def u_to_law(u: np.ndarray) -> NoiseLaw:
    u = np.asarray(u, dtype=float)
    return NoiseLaw(math.exp(u[0]), float(u[1]), math.exp(u[2]), math.exp(u[3]))


def _clip_to_bounds(u: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = hi - lo
    return np.clip(np.asarray(u, dtype=float), lo + 1e-6 * span, hi - 1e-6 * span)


def _child_seeds(seed: int, specimen_id: int, n: int) -> List[int]:
    """Per-cell child seeds: SeedSequence([seed, specimen_id]).spawn(n)."""
    seq = np.random.SeedSequence([int(seed), int(specimen_id)])
    return [int(c.generate_state(1, dtype=np.uint64)[0]) for c in seq.spawn(n)]


# ===========================================================================
#  The cell's layout and target
# ===========================================================================
@dataclass(frozen=True, eq=False)
class CellLayout:
    """Everything the simulator needs to reproduce the real cell's windows.

    SS: n_sweeps continuous sweeps of L_s = (train_length - 1) P + n_w
    samples; window k (0-based position p_k in sweep s_k) is the enlarged
    window [p_k P + j0, p_k P + n_w) of that sweep's trace, m = n_w - j0
    samples; the windows are the KEPT ones (D-106.1). LS: K_LS pre-windows of
    n_ls samples each. Every sweep carries its real sweep's tilt (mV/s,
    D-107.2): ss_tilt[i] for sweep_numbers[i], ls_tilt[k] for
    ls_sweep_number[k].
    """
    specimen_id: int
    fs: float
    dt: float
    n_w: int
    pre_n: int
    j0: int
    period_samples: int
    train_length: int
    sweep_numbers: Tuple[int, ...]         # SS sweeps with kept windows, in the data's order
    sweep_index: np.ndarray                # (K,) index into sweep_numbers, per window
    position: np.ndarray                   # (K,) position in the train, per window
    polarity_is_dep: np.ndarray            # (K,)
    sweep_number: np.ndarray               # (K,) sweep number per window
    n_ls: int                              # samples of an LS pre-window (0 when the cell has none)
    ls_sweep_number: np.ndarray            # (K_LS,)
    ss_tilt: np.ndarray                    # (n_sweeps,) mV/s, the tilt of each SS sweep (TL1)
    ls_tilt: np.ndarray                    # (K_LS,) mV/s, the tilt of each LS sweep (TL2)

    @property
    def K(self) -> int:
        return int(self.position.size)

    @property
    def K_LS(self) -> int:
        return int(self.ls_sweep_number.size)

    @property
    def m(self) -> int:
        return int(self.n_w - self.j0)

    @property
    def n_sweeps(self) -> int:
        return len(self.sweep_numbers)

    @property
    def L_s(self) -> int:
        return int((self.train_length - 1) * self.period_samples + self.n_w)

    @classmethod
    def from_windows(cls, ss: SSWindows, ls: Optional[LSWindows], t_ss_ms: float,
                     ss_tilts: SweepTilts, ls_tilts: Optional[SweepTilts]) -> "CellLayout":
        """The layout of the kept windows `ss` (Block 3's SSNoiseWindows.ss) and
        the LS pre-windows, with every sweep's tilt from Block 3's readings.
        Refused (ValueError, with the reason) when a sweep has no tilt or a
        NaN one (an unreadable tilt: the reading's flag is quoted)."""
        j0 = ss.pre_n + int(round(t_ss_ms * 1e-3 * ss.fs))
        if j0 >= ss.n_w - 1:
            raise ValueError("t_SS = %g ms leaves no enlarged window" % t_ss_ms)
        if ss.n_windows == 0:
            raise ValueError("no kept SS window: the cell cannot be fitted")
        sweep_numbers = tuple(int(s) for s in dict.fromkeys(int(x) for x in ss.sweep_number))
        index_of = {s: i for i, s in enumerate(sweep_numbers)}
        sweep_index = np.array([index_of[int(s)] for s in ss.sweep_number], dtype=int)
        position = np.asarray(ss.position, dtype=int)
        if position.min() < 0 or position.max() >= ss.train_length:
            raise ValueError("a stored position lies outside the train")
        ss_tilt = _tilts_for(sweep_numbers, ss_tilts, "SS")
        if ls is None or len(ls.pre) == 0:
            n_ls, ls_sweeps, ls_tilt = 0, np.zeros(0, dtype=int), np.zeros(0)
        else:
            lengths = np.array([int(p.size) for p in ls.pre])
            if np.unique(lengths).size != 1:
                raise ValueError("LS pre-windows of unequal length %s; Block 4 needs one n_LS per cell"
                                 % sorted(set(lengths.tolist())))
            n_ls, ls_sweeps = int(lengths[0]), np.asarray(ls.sweep_number, dtype=int)
            if ls_tilts is None:
                raise ValueError("the LS sweeps need their tilts (Block 3's ls_sweep_tilts)")
            ls_tilt = _tilts_for(tuple(int(x) for x in ls_sweeps), ls_tilts, "LS")
        return cls(specimen_id=int(ss.specimen_id), fs=float(ss.fs), dt=float(ss.dt), n_w=int(ss.n_w),
                   pre_n=int(ss.pre_n), j0=int(j0), period_samples=int(ss.period_samples),
                   train_length=int(ss.train_length), sweep_numbers=sweep_numbers, sweep_index=sweep_index,
                   position=position, polarity_is_dep=np.asarray(ss.polarity_is_dep, dtype=bool),
                   sweep_number=np.asarray(ss.sweep_number, dtype=int), n_ls=n_ls, ls_sweep_number=ls_sweeps,
                   ss_tilt=ss_tilt, ls_tilt=ls_tilt)

    @classmethod
    def from_cell(cls, nwin: SSNoiseWindows, ls: Optional[LSWindows]) -> "CellLayout":
        """from_windows on Block 3's outputs: the kept windows of `nwin`, its
        t_SS(c), the SS tilts read on its X_tilde (TL1) and the LS tilts (TL2)."""
        ss_t = ncs.ss_sweep_tilts(nwin.X_tilde, nwin.ss, nwin.j0)
        ls_t = ncs.ls_sweep_tilts(ls) if ls is not None and len(ls.pre) else None
        return cls.from_windows(nwin.ss, ls, nwin.start.t_ss_ms, ss_t, ls_t)


def _tilts_for(sweeps: Tuple[int, ...], tilts: SweepTilts, protocol: str) -> np.ndarray:
    """The tilt (mV/s) of each sweep, in the order given; ValueError when one
    is missing or NaN (its reading's flag quoted)."""
    b = tilts.as_dict()
    out = np.empty(len(sweeps))
    for i, s in enumerate(sweeps):
        if s not in b:
            raise ValueError("%s sweep %d has no tilt reading" % (protocol, s))
        if not np.isfinite(b[s]):
            why = [f for f in tilts.flags if (":%d" % s) in f]
            raise ValueError("the tilt of %s sweep %d could not be read (%s); the fit refuses the cell"
                             % (protocol, s, ", ".join(why) or "NaN"))
        out[i] = float(b[s])
    return out


@dataclass(eq=False)
class CellTarget:
    """The data side: q_hat(c) and the per-window matrices behind it (for W_0's
    bootstrap)."""
    q_hat: np.ndarray            # (15,)
    per_window_ss: np.ndarray    # (K, 5)
    per_window_ls: np.ndarray    # (K_LS, 10)
    labels: Tuple[str, ...] = ncs.STAT_LABELS

    @classmethod
    def from_stats(cls, ss_stats: np.ndarray, ls_stats: np.ndarray, layout: CellLayout,
                   window_cfg: WindowConfig = WindowConfig()) -> "CellTarget":
        ss_stats = np.asarray(ss_stats, dtype=float).reshape(-1, N_SS)
        ls_stats = np.asarray(ls_stats, dtype=float).reshape(-1, N_LS)
        sv = ncs.statistic_vector(ss_stats, layout.m, ls_stats, max(layout.n_ls, 1), window_cfg, layout.fs)
        return cls(q_hat=sv.q, per_window_ss=ss_stats, per_window_ls=ls_stats)

    @classmethod
    def from_cell(cls, nwin: SSNoiseWindows, ls: Optional[LSWindows], layout: CellLayout,
                  window_cfg: WindowConfig = WindowConfig()) -> "CellTarget":
        """The statistics of Block 3's kept, subtracted windows and of the LS
        pre-windows (the same estimators the simulator's windows go through)."""
        ss_stats = ncs.ss_window_stats(nwin.X_tilde, nwin.ss.fs)
        ls_stats = ncs.ls_window_stats(ls.pre, ls.fs, window_cfg) if ls is not None and len(ls.pre) else np.zeros((0, N_LS))
        return cls.from_stats(ss_stats, ls_stats, layout, window_cfg)


# ===========================================================================
#  The simulator of one replicate cell at the real layout (D-056 (i))
# ===========================================================================
def simulate_cell_windows(law: NoiseLaw, layout: CellLayout, rng: np.random.Generator,
                          *, parts: Sequence[str] = ("ss", "ls"), window_cfg: WindowConfig = WindowConfig(),
                          return_windows: bool = False):
    """One replicate cell: (ss_stats (K, 5), ls_stats (K_LS, 10)); with
    return_windows also the subtracted SS windows X_tilde (K, m) and the LS
    traces (K_LS, n_ls).

    SS: each sweep is one continuous trace of L_s samples from
    `noise_law.simulate_sweep` (fast + slow, and the real sweep's tilt
    layout.ss_tilt as a fixed number, D-107.2; the offset is invisible and
    left out); the enlarged windows are cut at the kept positions and each
    group's median window -- the per-sample median over this replicate's own
    windows of that sweep and polarity -- is subtracted (D-103.1, D-051 (iv)).
    LS: K_LS traces of n_ls samples, trace k with the tilt layout.ls_tilt[k].
    Draw order: the SS sweeps in layout order, then the LS windows (the unit
    tilt draws are made and left unused) -- fixed, so a seed gives the same
    draws for any law.
    """
    ss_stats = np.zeros((0, N_SS))
    ls_stats = np.zeros((0, N_LS))
    X_tilde = None
    ls_traces = None
    if "ss" in parts and layout.K > 0:
        X = np.empty((layout.K, layout.m))
        P, j0, n_w = layout.period_samples, layout.j0, layout.n_w
        for s_idx in range(layout.n_sweeps):
            units = noise_law.draw_units(rng, 1, layout.L_s)
            trace = noise_law.simulate_sweep(law, layout.dt, layout.L_s, units, 0,
                                             tilt_mV_per_s=float(layout.ss_tilt[s_idx]))
            # Custom: layout-true cutting -- window at position p is [p P + j0, p P + n_w).
            for k in np.flatnonzero(layout.sweep_index == s_idx):
                p = int(layout.position[k])
                X[k] = trace[p * P + j0:p * P + n_w]
        X_tilde, _ = subtract_median_response(X, layout.polarity_is_dep, layout.sweep_number)
        ss_stats = ncs.ss_window_stats(X_tilde, layout.fs)
    if "ls" in parts and layout.K_LS > 0:
        units = noise_law.draw_units(rng, layout.K_LS, layout.n_ls)
        ls_traces = noise_law.simulate_traces(law, layout.dt, units.fast, units.slow,
                                              tilt_mV_per_s=layout.ls_tilt)
        ls_stats = ncs.ls_window_stats(list(ls_traces), layout.fs, window_cfg)
    if return_windows:
        return ss_stats, ls_stats, X_tilde, ls_traces
    return ss_stats, ls_stats


# ===========================================================================
#  The binding function q_sim(theta) with common random numbers (D-056 (iii))
# ===========================================================================
def hd_median(A: np.ndarray) -> np.ndarray:
    """Harrell-Davis median of each column of A (N, d), ignoring NaN;
    `scipy.stats.mstats.hdquantiles(prob=0.5)` per column."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("hd_median: A must be (N, d)")
    out = np.full(A.shape[1], np.nan)
    for j in range(A.shape[1]):
        col = A[:, j]
        col = col[np.isfinite(col)]
        if col.size >= 2:
            out[j] = float(np.asarray(hdquantiles(col, prob=[0.5]))[0])
        elif col.size == 1:
            out[j] = float(col[0])
    return out


def binding_function(law: NoiseLaw, layout: CellLayout, cfg: NoiseFitConfig, seed: int,
                     *, which: str = "both", return_pool: bool = False):
    """q_sim(theta) (15,): the SS part on M_cells = ceil(M / K) replicate cells
    at the real layout and tilts (pooled windows), the LS part on N_LS =
    ceil(M / K_LS) K_LS pre-windows, window i with the tilt of the cell's LS
    sweep i mod K_LS; the Harrell-Davis median per statistic. Common random
    numbers: `default_rng` is rebuilt from `seed` (two children: SS, LS) at
    every call, so the same (layout, seed) gives the same draws for any law.
    which: 'ss', 'ls' or 'both' (the other part stays NaN). With return_pool
    also (pool_ss (M_cells K, 5), pool_ls (N_LS, 10))."""
    ss_seed, ls_seed = [int(c.generate_state(1, dtype=np.uint64)[0])
                        for c in np.random.SeedSequence(int(seed)).spawn(2)]
    q = np.full(N_STATS, np.nan)
    pool_ss = np.zeros((0, N_SS))
    pool_ls = np.zeros((0, N_LS))
    if which in ("ss", "both") and layout.K > 0:
        rng = np.random.default_rng(ss_seed)
        m_cells = int(math.ceil(cfg.M_binding / layout.K))
        pools = []
        for _ in range(m_cells):
            s, _l = simulate_cell_windows(law, layout, rng, parts=("ss",), window_cfg=cfg.window_cfg)
            pools.append(s)
        pool_ss = np.vstack(pools)
        q[:N_SS] = hd_median(pool_ss)
    if which in ("ls", "both") and layout.K_LS > 0:
        rng = np.random.default_rng(ls_seed)
        # Custom: the LS tilts cycled through the pool -- replicate sets of the cell's LS sweeps (D-107.2).
        n_total = int(math.ceil(cfg.M_binding / layout.K_LS)) * layout.K_LS
        tilt_of = layout.ls_tilt[np.arange(n_total) % layout.K_LS]
        pools = []
        start = 0
        while start < n_total:
            n = min(n_total - start, int(cfg.chunk_windows))
            units = noise_law.draw_units(rng, n, layout.n_ls)
            traces = noise_law.simulate_traces(law, layout.dt, units.fast, units.slow,
                                               tilt_mV_per_s=tilt_of[start:start + n])
            pools.append(ncs.ls_window_stats(list(traces), layout.fs, cfg.window_cfg))
            start += n
        pool_ls = np.vstack(pools)
        q[N_SS:] = hd_median(pool_ls)
    if return_pool:
        return q, pool_ss, pool_ls
    return q


def closed_form_vector(law: NoiseLaw, layout: CellLayout, window_cfg: WindowConfig = WindowConfig()) -> np.ndarray:
    """q_cf(theta | n_m) (15,): sqrt(E[S_0] / (n - 1)) for an SD and
    E[S_L] / E[S_0] for an autocorrelation (eq. 10 of the topic doc, Block 2),
    at the cell's window lengths; the fit's starting point only."""
    dt, fs = layout.dt, layout.fs
    m = layout.m
    n_ls = max(layout.n_ls, 2)
    n1 = int(round(window_cfg.ls_first_ms * 1e-3 * fs))
    n2 = int(round(window_cfg.ls_last_ms * 1e-3 * fs))
    q = [noise_law.closed_form_sd(law, dt, m)]
    q += list(noise_law.closed_form_acf(law, dt, m, ncs.lags_in_samples(ncs.SS_LAGS_MS, fs)))
    q += [noise_law.closed_form_sd(law, dt, n_ls)]
    q += list(noise_law.closed_form_acf(law, dt, n_ls, ncs.lags_in_samples(ncs.LS_FULL_LAGS_MS, fs)))
    q += [noise_law.closed_form_sd(law, dt, n1)]
    q += [noise_law.closed_form_sd(law, dt, n2)]
    q += list(noise_law.closed_form_acf(law, dt, n2, ncs.lags_in_samples(ncs.LS_LAST_LAGS_MS, fs)))
    return np.array(q, dtype=float)


# ===========================================================================
#  Sigma_hat: the simulated covariance of the cell's statistic vector (D-049)
# ===========================================================================
@dataclass(eq=False)
class SigmaResult:
    Sigma: np.ndarray                      # (15, 15), NaN rows/columns for statistics without data
    vectors: np.ndarray                    # (R, 15)
    R: int
    per_sweep: Dict[int, np.ndarray] = field(default_factory=dict)   # sweep -> (R, 5) SS or (R, 10) LS


def simulated_covariance(law: NoiseLaw, layout: CellLayout, R: int, rng: np.random.Generator,
                         *, per_sweep: bool = False, window_cfg: WindowConfig = WindowConfig()) -> SigmaResult:
    """R replicate cells at the real layout and tilts, each reduced by
    `noise_cell_stats.statistic_vector`; sample covariance with ddof 1. With
    per_sweep, also every sweep's vector per replicate (SS: the medians over
    that sweep's windows; LS: its one window's 10 values) for Block 4b."""
    R = int(R)
    if R < 2:
        raise ValueError("simulated_covariance: R >= 2")
    vectors = np.empty((R, N_STATS))
    sweep_rows: Dict[int, List[np.ndarray]] = {}
    for r in range(R):
        ss_stats, ls_stats = simulate_cell_windows(law, layout, rng, window_cfg=window_cfg)
        sv = ncs.statistic_vector(ss_stats, layout.m, ls_stats, max(layout.n_ls, 1), window_cfg, layout.fs)
        vectors[r] = sv.q
        if per_sweep:
            if layout.K > 0:
                for s, vec in ncs.per_sweep_vectors(ss_stats, layout.sweep_number).items():
                    sweep_rows.setdefault(int(s), []).append(vec)
            for i, s in enumerate(layout.ls_sweep_number):
                sweep_rows.setdefault(int(s), []).append(ls_stats[i])
    Sigma = np.cov(vectors, rowvar=False, ddof=1)
    return SigmaResult(Sigma=Sigma, vectors=vectors, R=R,
                       per_sweep={s: np.vstack(rows) for s, rows in sweep_rows.items()})


def _cholesky_lower(Sigma: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Lower Cholesky factor; on failure a diagonal jitter of 1e-10 x the
    mean variance is added (flagged)."""
    try:
        return linalg.cholesky(Sigma, lower=True), False
    except linalg.LinAlgError:
        jitter = 1e-10 * float(np.trace(Sigma)) / Sigma.shape[0]
        return linalg.cholesky(Sigma + jitter * np.eye(Sigma.shape[0]), lower=True), True


def correlation_condition_number(Sigma: np.ndarray) -> float:
    """kappa of the correlation matrix of Sigma (`numpy.linalg.cond`)."""
    sd = np.sqrt(np.diag(Sigma))
    corr = Sigma / np.outer(sd, sd)
    return float(np.linalg.cond(corr))


# ===========================================================================
#  The inner problem: argmin of (F1) at a fixed weight
# ===========================================================================
@dataclass(eq=False)
class InnerResult:
    law: NoiseLaw
    u: np.ndarray
    J: float
    nfev: int
    start_index: int
    status: int
    message: str
    starts: List[Tuple[np.ndarray, float]]     # (u0, J at the solution) per start
    at_bound: Tuple[str, ...]


def _whitened_residual(u: np.ndarray, q_hat: np.ndarray, C: np.ndarray, mask: np.ndarray,
                       layout: CellLayout, cfg: NoiseFitConfig, seed: int) -> np.ndarray:
    q = binding_function(u_to_law(u), layout, cfg, seed)
    r = q[mask] - q_hat[mask]
    return linalg.solve_triangular(C, r, lower=True)


def solve_inner(q_hat: np.ndarray, C: np.ndarray, starts: Sequence[NoiseLaw],
                layout: CellLayout, cfg: NoiseFitConfig, seed: int, mask: np.ndarray,
                residual: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> InnerResult:
    """argmin_u J_W = |C^{-1} (q_sim(u) - q_hat)|^2 over the kept statistics
    (mask), Sigma_hat = C C^T, by `least_squares(method='trf')` with the
    bounds, steps and tolerances of the config, from each start; the lowest J
    is kept. `residual` replaces the binding residual (tests, the closed-form
    start)."""
    lo, hi = cfg.bounds()
    fun = residual if residual is not None else (
        lambda u: _whitened_residual(u, q_hat, C, mask, layout, cfg, seed))
    best = None
    tried: List[Tuple[np.ndarray, float]] = []
    for i, start in enumerate(starts):
        u0 = _clip_to_bounds(law_to_u(start), lo, hi)
        res = optimize.least_squares(fun, u0, bounds=(lo, hi), method="trf", diff_step=list(cfg.fd_rel_step),
                                     ftol=cfg.ftol, xtol=cfg.xtol, gtol=cfg.gtol, max_nfev=cfg.max_nfev)
        J = 2.0 * float(res.cost)
        tried.append((u0, J))
        if best is None or J < best[1]:
            best = (res, J, i)
    res, J, i = best
    names = ("log_sigma_f", "rho_f", "log_sigma_s", "log_tau_s")
    at_bound = tuple(names[j] for j in range(4)
                     if res.x[j] <= lo[j] + 1e-3 * (hi[j] - lo[j]) or res.x[j] >= hi[j] - 1e-3 * (hi[j] - lo[j]))
    return InnerResult(law=u_to_law(res.x), u=np.array(res.x, dtype=float), J=J, nfev=int(res.nfev),
                       start_index=i, status=int(res.status), message=str(res.message), starts=tried,
                       at_bound=at_bound)


def closed_form_start(q_hat: np.ndarray, SE: np.ndarray, mask: np.ndarray, layout: CellLayout,
                      cfg: NoiseFitConfig, u0: Optional[np.ndarray] = None) -> NoiseLaw:
    """The closed-form start: argmin of (F1) with W_0 = diag(1 / SE^2) on the
    closed-form statistics q_cf (instant; no simulation)."""
    lo, hi = cfg.bounds()
    if u0 is None:
        u0 = law_to_u(NoiseLaw(0.04, 0.2, 0.05, 0.04))
    u0 = _clip_to_bounds(u0, lo, hi)

    def fun(u):
        q = closed_form_vector(u_to_law(u), layout, cfg.window_cfg)
        return (q[mask] - q_hat[mask]) / SE[mask]

    res = optimize.least_squares(fun, u0, bounds=(lo, hi), method="trf")
    return u_to_law(res.x)


# ===========================================================================
#  The iteration (D-050) and the result
# ===========================================================================
@dataclass(eq=False)
class CellFitResult:
    specimen_id: int
    law: NoiseLaw
    u: np.ndarray
    J: float
    dof: int
    p_value: float
    kappa: float
    Sigma: np.ndarray                      # (d_eff, d_eff) final, kept statistics
    labels_kept: Tuple[str, ...]
    stats_dropped: Tuple[str, ...]
    q_hat: np.ndarray                      # (15,)
    q_sim: np.ndarray                      # (15,) at the solution
    SE: np.ndarray                         # (15,) W_0's bootstrap SE (NaN where dropped)
    ss_tilt: Dict[int, float]              # sweep -> mV/s, the SS tilts the simulator carried (D-107.2)
    ls_tilt: Dict[int, float]              # sweep -> mV/s, the LS tilts the simulator carried
    n_solves: int
    converged: bool
    flags: Tuple[str, ...]
    history: List[Dict[str, object]]
    per_sweep_cov: Dict[int, np.ndarray]   # sweep -> covariance of its vector at the solution (final Sigma run)
    config: NoiseFitConfig
    seconds: float

    def to_dict(self) -> Dict[str, object]:
        """JSON-ready: arrays as lists, the law as a dict, NaN kept as None."""
        def arr(a):
            return [None if (isinstance(x, float) and not math.isfinite(x)) else x
                    for x in np.asarray(a, dtype=float).ravel().tolist()]

        def num(x):
            x = float(x)
            return None if not math.isfinite(x) else x

        return {
            "specimen_id": int(self.specimen_id),
            "law": {"sigma_f_mV": self.law.sigma_f, "rho_f": self.law.rho_f, "sigma_s_mV": self.law.sigma_s,
                    "tau_s_s": self.law.tau_s},
            "u": arr(self.u), "J": num(self.J), "dof": int(self.dof),
            "p_value": num(self.p_value), "kappa": num(self.kappa),
            "Sigma": [arr(row) for row in np.asarray(self.Sigma)], "labels_kept": list(self.labels_kept),
            "stats_dropped": list(self.stats_dropped), "q_hat": arr(self.q_hat), "q_sim": arr(self.q_sim),
            "SE": arr(self.SE),
            "ss_tilt_mV_per_s": {str(k): num(v) for k, v in self.ss_tilt.items()},
            "ls_tilt_mV_per_s": {str(k): num(v) for k, v in self.ls_tilt.items()},
            "n_solves": int(self.n_solves), "converged": bool(self.converged),
            "flags": list(self.flags),
            "history": [{k: (arr(v) if isinstance(v, np.ndarray) else v) for k, v in h.items()} for h in self.history],
            "per_sweep_cov": {str(s): [arr(row) for row in np.asarray(c)] for s, c in self.per_sweep_cov.items()},
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.config.__dict__.items()
                       if k != "window_cfg"},
            "window_cfg": dict(self.config.window_cfg.__dict__),
            "seconds": float(self.seconds),
        }


def _starts_around(law: NoiseLaw, n: int) -> List[NoiseLaw]:
    if n <= 1:
        return [law]
    return [law, replace(law, tau_s=law.tau_s / 2.0), replace(law, tau_s=law.tau_s * 2.0)]


def fit_cell(target: CellTarget, layout: CellLayout, cfg: NoiseFitConfig = NoiseFitConfig(),
             log: Optional[Callable[[str], None]] = None) -> CellFitResult:
    """The per-cell fit of section 2.3: W_0 from the window bootstrap, the
    closed-form start and its two tau_s variants, the W_0 solve, then
    Sigma_hat-weighted solves (R_iter) until max |Delta u| <= tol_u or k_max
    solves ('no_plateau'), then the final step at R_final (Sigma_hat, one
    solve, J, kappa, the chi-square tail). The simulator carries the layout's
    real tilts throughout; no tilt is estimated (D-107.2)."""
    t_start = time.time()
    say = log if log is not None else (lambda s: None)
    seed_binding, seed_boot, seed_sigma_root = _child_seeds(cfg.seed, layout.specimen_id, 3)
    sigma_seqs = np.random.SeedSequence(seed_sigma_root).spawn(cfg.k_max + 2)
    flags: List[str] = []
    q_hat = np.asarray(target.q_hat, dtype=float)
    if q_hat.shape != (N_STATS,):
        raise ValueError("q_hat must have %d entries" % N_STATS)

    # --- W_0: the window-level bootstrap SE of the medians (D-049)
    rng_boot = np.random.default_rng(seed_boot)
    SE = np.full(N_STATS, np.nan)
    if target.per_window_ss.shape[0] >= 2:
        SE[:N_SS] = ncs.bootstrap_median_se(target.per_window_ss, cfg.B_boot, rng_boot)
    if target.per_window_ls.shape[0] >= 2:
        SE[N_SS:] = ncs.bootstrap_median_se(target.per_window_ls, cfg.B_boot, rng_boot)
    mask = np.isfinite(q_hat)
    # a finite statistic without a bootstrap SE (a single window): SE from the
    # simulated single-window spread at the closed-form start, / sqrt(K) x 1.2533
    need_sim_se = mask & ~(np.isfinite(SE) & (SE > 0))
    if need_sim_se.any():
        prelim = np.isfinite(SE) & (SE > 0) & mask
        if prelim.sum() >= 5:
            law0 = closed_form_start(q_hat, SE, prelim, layout, cfg)
        else:
            law0 = NoiseLaw(0.04, 0.2, 0.05, 0.04)
        _q, pool_ss, pool_ls = binding_function(law0, layout, cfg, seed_binding, return_pool=True)
        sd = np.full(N_STATS, np.nan)
        if pool_ss.shape[0] >= 2:
            sd[:N_SS] = np.nanstd(pool_ss, axis=0, ddof=1) * 1.2533 / math.sqrt(max(layout.K, 1))
        if pool_ls.shape[0] >= 2:
            sd[N_SS:] = np.nanstd(pool_ls, axis=0, ddof=1) * 1.2533 / math.sqrt(max(layout.K_LS, 1))
        SE = np.where(need_sim_se, sd, SE)
        flags.append("se_from_simulation:" + ",".join(ncs.STAT_LABELS[j] for j in np.flatnonzero(need_sim_se)))
    mask = mask & np.isfinite(SE) & (SE > 0)
    d_eff = int(mask.sum())
    dropped = tuple(ncs.STAT_LABELS[j] for j in np.flatnonzero(~mask))
    if dropped:
        flags.append("stats_dropped:" + ",".join(dropped))
    if d_eff < 5:
        raise ValueError("only %d usable statistics; the fit needs at least 5" % d_eff)
    SE_masked = np.where(mask, SE, np.nan)

    # --- the closed-form start
    law_cf = closed_form_start(q_hat, SE, mask, layout, cfg)
    say("start (closed form): %s" % (law_cf,))

    history: List[Dict[str, object]] = []

    def record(kind, inner, R, t0):
        history.append({"solve": len(history), "weight": kind, "R": int(R), "u": np.array(inner.u),
                        "J": float(inner.J), "nfev": int(inner.nfev),
                        "start_index": int(inner.start_index), "status": int(inner.status),
                        "at_bound": list(inner.at_bound), "seconds": float(time.time() - t0)})
        say("solve %d (%s, R=%d): u=%s J=%.4g nfev=%d start=%d %s %.0f s"
            % (len(history) - 1, kind, R, np.array2string(inner.u, precision=4), inner.J,
               inner.nfev, inner.start_index, inner.at_bound, time.time() - t0))

    # --- solve 0: W_0 = diag(1 / SE^2), three starts
    t0 = time.time()
    C0 = np.diag(SE[mask])
    inner = solve_inner(q_hat, C0, _starts_around(law_cf, 3), layout, cfg, seed_binding, mask)
    record("W0", inner, 0, t0)
    u_prev = inner.u.copy()
    law = inner.law

    # --- Sigma_hat-weighted solves
    converged = False
    chol_jitter = False
    for k in range(1, cfg.k_max + 1):
        t0 = time.time()
        rng_k = np.random.default_rng(sigma_seqs[k - 1])
        sig = simulated_covariance(law, layout, cfg.R_iter, rng_k, window_cfg=cfg.window_cfg)
        C, jit = _cholesky_lower(sig.Sigma[np.ix_(mask, mask)])
        chol_jitter = chol_jitter or jit
        inner = solve_inner(q_hat, C, _starts_around(law, cfg.starts_after_first), layout, cfg,
                            seed_binding, mask)
        record("Sigma_%d" % k, inner, cfg.R_iter, t0)
        du = float(np.max(np.abs(inner.u - u_prev)))
        u_prev = inner.u.copy()
        law = inner.law
        if du <= cfg.tol_u:
            converged = True
            break
    if not converged:
        flags.append("no_plateau")

    # --- final step at R_final
    t0 = time.time()
    rng_f = np.random.default_rng(sigma_seqs[cfg.k_max + 1])
    sig = simulated_covariance(law, layout, cfg.R_final, rng_f, per_sweep=True, window_cfg=cfg.window_cfg)
    Sigma_m = sig.Sigma[np.ix_(mask, mask)]
    C, jit = _cholesky_lower(Sigma_m)
    chol_jitter = chol_jitter or jit
    inner = solve_inner(q_hat, C, [law], layout, cfg, seed_binding, mask)
    record("final", inner, cfg.R_final, t0)
    law = inner.law
    if chol_jitter:
        flags.append("sigma_jitter")
    if inner.at_bound:
        flags.append("at_bound:" + ",".join(inner.at_bound))
    dof = d_eff - 4
    kappa = correlation_condition_number(Sigma_m)
    # q_sim and J at the reported theta_hat with the final Sigma_hat
    q_sim = binding_function(law, layout, cfg, seed_binding)
    r_w = linalg.solve_triangular(C, (q_sim - q_hat)[mask], lower=True)
    J = float(r_w @ r_w)
    p_value = float(chi2.sf(J, dof)) if dof > 0 else float("nan")
    per_sweep_cov = {s: np.cov(v, rowvar=False, ddof=1) for s, v in sig.per_sweep.items()}
    return CellFitResult(specimen_id=layout.specimen_id, law=law, u=inner.u, J=J, dof=dof,
                         p_value=p_value, kappa=kappa, Sigma=Sigma_m,
                         labels_kept=tuple(ncs.STAT_LABELS[j] for j in np.flatnonzero(mask)),
                         stats_dropped=dropped, q_hat=q_hat, q_sim=q_sim, SE=SE_masked,
                         ss_tilt={int(s): float(b) for s, b in zip(layout.sweep_numbers, layout.ss_tilt)},
                         ls_tilt={int(s): float(b) for s, b in zip(layout.ls_sweep_number, layout.ls_tilt)},
                         n_solves=len(history),
                         converged=converged, flags=tuple(flags), history=history, per_sweep_cov=per_sweep_cov,
                         config=cfg, seconds=float(time.time() - t_start))
