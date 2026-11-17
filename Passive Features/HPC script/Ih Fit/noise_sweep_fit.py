"""noise_sweep_fit -- Block 9 ("Block 4b") of specs/SPEC.md: the per-sweep
stage of the noise fit -- each sweep's law, its bootstrap variances and the
poolings per protocol (D-073, D-076 (i), D-115, D-116, D-122, D-153.1,
D-154.2, D-154.3; sections 2.5 and 2.6 (e), (g), (h) of the spec: (P0'),
(P1)-(P4), (D-076.1)).

In plain words: Block 4 fitted one law theta_hat(c) = (sigma_f, sigma_m,
tau_m, sigma_s, tau_s) to the whole cell (the three-component law of 2.6
(a)). Each twin sweep of Stage 7 must carry the noise of its own real sweep
(D-073), so here every sweep s is fitted on its own -- its SS (or LS)
statistics against the same simulator restricted to that sweep, with the two
time constants tau_m and tau_s held at the cell's values -- giving the three
sizes sigma_f(c, s), sigma_m(c, s) and sigma_s(c, s). One sweep says little
about sigma_m and sigma_s, so the sweeps of a protocol share strength: one
parametric bootstrap (B_v refits of replicates simulated at the sweep's
estimate) gives the sampling variances v_m(c, s) of log sigma_m(c, s) and
v_s(c, s) of log sigma_s(c, s), and, separately for each of the two sizes,
the DerSimonian-Laird estimate of the between-sweep variance shrinks each
sweep's log size towards its protocol's mean by lambda = tau2 / (tau2 + v)
(D-076.1, D-154.2). sigma_f is not pooled. A sweep whose fit fails -- no
convergence, a coordinate at a bound, or a fast part collapsed below half the
cell's (D-154.3) -- is flagged, left out of the pooling, and its twin carries
the cell's whole law theta_hat(c) (D-122.3).

Layers: model and analysis only -- no file I/O (an optional `log` callable is
the only output channel besides the results), no plotting. Units: theta in
mV, mV, s, mV, s; tilts in mV/s; u_s = (log sigma_f, log sigma_m, log
sigma_s) with the sigmas in mV; v and tau2 in (log mV)^2. The statistic
counts come from noise_fit / noise_cell_stats (none is written here). Pure
ASCII, LF.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg, optimize

import noise_cell_stats as ncs
import noise_fit as nf
from noise_fit import CellFitResult, CellLayout, CellTarget, NoiseFitConfig
from noise_law import NoiseLaw
from noise_windows import WindowConfig

PROTOCOLS: Tuple[str, ...] = ("ss", "ls")
U_NAMES: Tuple[str, ...] = ("log_sigma_f", "log_sigma_m", "log_sigma_s")
_U_IN_CELL: Tuple[int, ...] = tuple(nf.U_NAMES.index(n) for n in U_NAMES)   # (0, 1, 3): u_s inside Block 4's u
COORDS: Tuple[str, ...] = ("sigma_m", "sigma_s")      # the pooled sizes (2.6 (e)); sigma_f is not pooled
_SLICE = {"ss": slice(0, nf.N_SS), "ls": slice(nf.N_SS, nf.N_STATS)}
_PROTOCOL_CODE = {"ss": 0, "ls": 1}
_NAN = float("nan")


# ===========================================================================
#  Configuration
# ===========================================================================
@dataclass(frozen=True)
class SweepStageConfig:
    """Sections 2.5 and 2.6 (e) and the Block 9 entry. B_v, M_refit and
    max_failed_refit_frac are D-122.1-D-122.2 (user); min_sigma_f_ratio is
    D-154.3 (user); min_stats (d_s >= 4) and the seed are the assistant's
    choices, flagged in the spec. The main per-sweep solve takes Block 4's
    NoiseFitConfig (M_binding = 3000, bounds, steps, tolerances, max_nfev)."""
    B_v: int = 50                         # bootstrap replicates per sweep (D-122.1)
    M_refit: Optional[int] = 1000         # the refits' binding size; None: the main fit's M_binding (D-122.1)
    max_failed_refit_frac: float = 0.1    # flag 'v_refits_failed' above this share of B_v (a report; D-122.2)
    min_stats: int = 4                    # d_s >= 4: one degree of freedom over the three coordinates (2.5 (a))
    seed: int = 20261030                  # root of the per-sweep streams (sweep_seeds)
    min_sigma_f_ratio: float = 0.5        # 'degenerate:sigma_f' below this share of sigma_f_hat(c) (D-154.3)

    def __post_init__(self):
        if int(self.B_v) < 2:
            raise ValueError("SweepStageConfig: B_v >= 2")
        if self.M_refit is not None and int(self.M_refit) < 1:
            raise ValueError("SweepStageConfig: M_refit is None or >= 1")
        if not 0.0 <= float(self.max_failed_refit_frac) <= 1.0:
            raise ValueError("SweepStageConfig: max_failed_refit_frac in [0, 1]")
        if int(self.min_stats) < 1:
            raise ValueError("SweepStageConfig: min_stats >= 1")
        try:
            r = float(self.min_sigma_f_ratio)
        except (TypeError, ValueError):
            raise ValueError("SweepStageConfig: min_sigma_f_ratio must be a number in [0, 1]") from None
        if isinstance(self.min_sigma_f_ratio, bool) or not 0.0 <= r <= 1.0:
            raise ValueError("SweepStageConfig: min_sigma_f_ratio must be a number in [0, 1]")


def _proto(protocol: str) -> str:
    pr = str(protocol).lower()
    if pr not in PROTOCOLS:
        raise ValueError("protocol must be 'ss' or 'ls', got %r" % (protocol,))
    return pr


def _law_of(u3: np.ndarray, tau_m: float, tau_s: float) -> NoiseLaw:
    """theta(u_s) = (exp u_1, exp u_2, tau_m, exp u_3, tau_s): tau_m and tau_s
    held at the cell's (P0', 2.6 (e))."""
    return NoiseLaw(math.exp(float(u3[0])), math.exp(float(u3[1])), float(tau_m), math.exp(float(u3[2])),
                    float(tau_s))


def _u3_of(law: NoiseLaw) -> np.ndarray:
    """u_s(law) = (log sigma_f, log sigma_m, log sigma_s): Block 4's u at U_NAMES."""
    return nf.law_to_u(law)[list(_U_IN_CELL)]


def _bounds3(cfg: NoiseFitConfig) -> Tuple[np.ndarray, np.ndarray]:
    """The three SD bounds of Block 4's config, in the order of u_s."""
    lo, hi = cfg.bounds()
    idx = list(_U_IN_CELL)
    return lo[idx].copy(), hi[idx].copy()


def _diff_step3(cfg: NoiseFitConfig) -> List[float]:
    """Block 4's relative steps for the coordinates of u_s: entries 0, 1, 3 of
    fd_rel_step, (1e-3, 3e-3, 1e-3) by default (2.6 (e))."""
    return [float(cfg.fd_rel_step[j]) for j in _U_IN_CELL]


# ===========================================================================
#  The sweep's layout, target and streams (2.5 (a), (b))
# ===========================================================================
def sweep_layout(layout: CellLayout, protocol: str, sweep: int) -> CellLayout:
    """The cell's layout restricted to one sweep. 'ss': the sweep's kept
    windows (positions, polarities, sweep numbers) and its tilt, no LS
    window; 'ls': its one pre-window and its tilt, no SS window. Every other
    field (fs, dt, n_w, pre_n, j0, j_floor, P, train length, n_ls) unchanged,
    so Block 4's simulator and binding function run on it as on the cell."""
    # Custom: the layout restriction (no library object describes the archive's layout).
    pr, s = _proto(protocol), int(sweep)
    if pr == "ss":
        if s not in layout.sweep_numbers:
            raise ValueError("SS sweep %d has no kept window in the cell's layout" % s)
        keep = layout.sweep_number == s
        i = layout.sweep_numbers.index(s)
        return replace(layout, sweep_numbers=(s,), sweep_index=np.zeros(int(keep.sum()), dtype=int),
                       position=layout.position[keep].copy(), polarity_is_dep=layout.polarity_is_dep[keep].copy(),
                       sweep_number=layout.sweep_number[keep].copy(), ss_tilt=layout.ss_tilt[[i]].copy(),
                       ls_sweep_number=np.zeros(0, dtype=int), ls_tilt=np.zeros(0))
    k = np.flatnonzero(layout.ls_sweep_number == s)
    if k.size != 1:
        raise ValueError("LS sweep %d is not one pre-window of the cell's layout (found %d)" % (s, k.size))
    return replace(layout, sweep_numbers=(), sweep_index=np.zeros(0, dtype=int), position=np.zeros(0, dtype=int),
                   polarity_is_dep=np.zeros(0, dtype=bool), sweep_number=np.zeros(0, dtype=int),
                   ss_tilt=np.zeros(0), ls_sweep_number=layout.ls_sweep_number[k].copy(),
                   ls_tilt=layout.ls_tilt[k].copy())


def _sweep_of(sub: CellLayout) -> Tuple[str, int]:
    """(protocol, sweep) of a layout restricted by sweep_layout."""
    if sub.K > 0 and sub.n_sweeps == 1 and sub.K_LS == 0:
        return "ss", int(sub.sweep_numbers[0])
    if sub.K == 0 and sub.K_LS == 1:
        return "ls", int(sub.ls_sweep_number[0])
    raise ValueError("a per-sweep layout holds one sweep of one protocol (see sweep_layout)")


def sweep_target(target: CellTarget, layout: CellLayout, protocol: str, sweep: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """q_hat(c, s) (d,) and its mask (d,) bool (2.5 (a), d = nf.N_STATS):
    an SS sweep's nf.N_SS statistics are the medians over its kept windows
    (Block 3's per_sweep_vectors), an LS sweep's nf.N_LS are its pre-window's
    row; NaN and False outside the sweep's protocol and where a statistic is
    not finite."""
    pr, s = _proto(protocol), int(sweep)
    pw_ss = np.asarray(target.per_window_ss, dtype=float)
    pw_ls = np.asarray(target.per_window_ls, dtype=float)
    if pw_ss.shape[0] != layout.K or pw_ls.shape[0] != layout.K_LS:
        raise ValueError("the target's windows (%d SS, %d LS) do not match the layout's (%d, %d)"
                         % (pw_ss.shape[0], pw_ls.shape[0], layout.K, layout.K_LS))
    q = np.full(nf.N_STATS, np.nan)
    if pr == "ss":
        if s not in layout.sweep_numbers:
            raise ValueError("SS sweep %d has no kept window in the cell's layout" % s)
        q[_SLICE["ss"]] = ncs.per_sweep_vectors(pw_ss, layout.sweep_number)[s]
    else:
        k = np.flatnonzero(layout.ls_sweep_number == s)
        if k.size != 1:
            raise ValueError("LS sweep %d is not one pre-window of the cell's layout" % s)
        q[_SLICE["ls"]] = pw_ls[int(k[0])]
    return q, np.isfinite(q)


def sweep_seeds(seed: int, specimen_id: int, protocol: str, sweep: int) -> Tuple[int, int]:
    """(seed_binding_s, seed_v_s): SeedSequence([seed, specimen_id, 0 for SS
    or 1 for LS, sweep]).spawn(2), each child reduced to one uint64 (as Block
    4's child seeds). seed_binding_s feeds every q_sim(. | s) of the sweep
    (the main solve and the refits: common random numbers); seed_v_s the
    bootstrap's replicates."""
    seq = np.random.SeedSequence([int(seed), int(specimen_id), _PROTOCOL_CODE[_proto(protocol)], int(sweep)])
    a, b = seq.spawn(2)
    return int(a.generate_state(1, dtype=np.uint64)[0]), int(b.generate_state(1, dtype=np.uint64)[0])


# ===========================================================================
#  (P1): the fit of one sweep
# ===========================================================================
@dataclass(eq=False)
class SweepFit:
    protocol: str
    sweep: int
    u: np.ndarray                 # (3,) (log sigma_f, log sigma_m, log sigma_s) at the solution
    law: NoiseLaw                 # theta_hat(c, s); tau_m = tau_m_hat(c), tau_s = tau_s_hat(c)
    J: float                      # (P1) at the solution
    d: int                        # statistics kept
    dof: int                      # d - 3 (a report)
    n_calls: int                  # calls of the residual, the finite-difference Jacobian's included
    status: int                   # least_squares' status
    at_bound: Tuple[str, ...]
    failed: bool
    reason: str                   # '', 'no_convergence' or 'at_bound:<coordinates>' (fit_sweep's own rule)
    flags: Tuple[str, ...] = ()   # 'sigma_jitter' when W_s's Cholesky needed a jitter


def _whitener(Sigma_s: np.ndarray, protocol: str, mask: np.ndarray) -> Tuple[np.ndarray, bool]:
    """The lower Cholesky factor of Sigma_hat_s on the kept statistics (W_s =
    Sigma_hat_s^{-1}, 2.5 (c)). Sigma_s is the protocol's (N_SS, N_SS) or
    (N_LS, N_LS) covariance (Block 4's per_sweep_cov[s]) or a (d, d) one."""
    S = np.asarray(Sigma_s, dtype=float)
    sl = _SLICE[protocol]
    d_pr = sl.stop - sl.start
    if S.shape == (d_pr, d_pr):
        S_full = np.full((nf.N_STATS, nf.N_STATS), np.nan)
        S_full[sl, sl] = S
    elif S.shape == (nf.N_STATS, nf.N_STATS):
        S_full = S
    else:
        raise ValueError("Sigma_s of shape %s for an %s sweep" % (S.shape, protocol.upper()))
    outside = np.ones(nf.N_STATS, dtype=bool)
    outside[sl] = False
    if np.any(mask & outside):
        raise ValueError("the mask keeps statistics outside the sweep's protocol")
    Sm = S_full[np.ix_(mask, mask)]
    if not np.all(np.isfinite(Sm)):
        raise ValueError("Sigma_s is not finite on the kept statistics")
    return nf._cholesky_lower(Sm)


def fit_sweep(q_hat_s: np.ndarray, mask: np.ndarray, Sigma_s: np.ndarray, sub: CellLayout,
              law_start: NoiseLaw, cfg: NoiseFitConfig, seed_binding_s: int) -> SweepFit:
    """(P1): argmin over u_s = (log sigma_f, log sigma_m, log sigma_s) of
    (q_sim(theta(u_s) | s) - q_hat(c, s))^T W_s (q_sim - q_hat) on the kept
    statistics, tau_m and tau_s held at law_start's (tau_m_hat(c),
    tau_s_hat(c); P0', 2.6 (e)); q_sim(. | s) is Block 4's binding function
    on the restricted layout `sub` at cfg.M_binding, re-seeded from
    seed_binding_s at every call; one least_squares(method='trf') call with
    Block 4's three SD bounds, diff_step (fd_rel_step entries 0, 1, 3), ftol,
    xtol, gtol and max_nfev, started at u_s(law_start). Fails (2.5 (c)) on
    status <= 0 ('no_convergence') or a coordinate within 1e-3 of its range
    from a bound ('at_bound:<coordinates>'). The rule of D-154.3
    ('degenerate:sigma_f', against the cell's sigma_f) is the stage's, checked
    next by run_sweep on the sweep's own fit."""
    protocol, sweep = _sweep_of(sub)
    q_hat_s = np.asarray(q_hat_s, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if q_hat_s.shape != (nf.N_STATS,) or mask.shape != (nf.N_STATS,):
        raise ValueError("q_hat_s and mask must have %d entries" % nf.N_STATS)
    d = int(mask.sum())
    if d < 1 or not np.all(np.isfinite(q_hat_s[mask])):
        raise ValueError("fit_sweep needs at least one kept, finite statistic")
    C, jitter = _whitener(Sigma_s, protocol, mask)
    lo, hi = _bounds3(cfg)
    tau_m, tau_s = float(law_start.tau_m), float(law_start.tau_s)
    calls = [0]

    def residual(u: np.ndarray) -> np.ndarray:
        calls[0] += 1
        q = nf.binding_function(_law_of(u, tau_m, tau_s), sub, cfg, seed_binding_s, which=protocol)
        return linalg.solve_triangular(C, q[mask] - q_hat_s[mask], lower=True)

    u0 = nf._clip_to_bounds(_u3_of(law_start), lo, hi)
    res = optimize.least_squares(residual, u0, bounds=(lo, hi), method="trf", diff_step=_diff_step3(cfg),
                                 ftol=cfg.ftol, xtol=cfg.xtol, gtol=cfg.gtol, max_nfev=cfg.max_nfev)
    u = np.array(res.x, dtype=float)
    span = hi - lo
    at_bound = tuple(U_NAMES[j] for j in range(len(U_NAMES))
                     if u[j] <= lo[j] + 1e-3 * span[j] or u[j] >= hi[j] - 1e-3 * span[j])
    status = int(res.status)
    reason = "no_convergence" if status <= 0 else ("at_bound:" + ",".join(at_bound) if at_bound else "")
    return SweepFit(protocol=protocol, sweep=sweep, u=u, law=_law_of(u, tau_m, tau_s), J=2.0 * float(res.cost), d=d,
                    dof=d - len(U_NAMES), n_calls=int(calls[0]), status=status, at_bound=at_bound, failed=bool(reason),
                    reason=reason, flags=("sigma_jitter",) if jitter else ())


# ===========================================================================
#  (P2): the sampling variances of log sigma_m(c, s) and log sigma_s(c, s) by one parametric bootstrap
# ===========================================================================
def _v_usable(v: float) -> bool:
    """2.6 (e) as amended (D-155): a bootstrap variance the pooling can weight
    by 1 / v -- finite and strictly positive. NaN (B_ok < 2) or 0 (every
    successful refit returned the same value) fails the sweep with 'v_none'."""
    v = float(v)
    return math.isfinite(v) and v > 0.0


@dataclass(eq=False)
class SweepBootstrap:
    v_m: float                    # (P2) on y_m: var(ddof 1) over the refits that did not fail; NaN when B_ok < 2
    v_s: float                    # (P2) on y_s, the same refits
    y_m: np.ndarray               # (B,) log sigma_m of each refit, NaN where it failed
    y_s: np.ndarray               # (B,) log sigma_s of each refit, NaN where it failed
    B: int                        # replicates drawn (B_v)
    B_ok: int                     # refits that did not fail
    flags: Tuple[str, ...]        # 'v_refits_failed:<n>/<B>' above max_failed_refit_frac; 'v_none' if v_m or v_s
                                  # is NaN or not > 0 (_v_usable)


def _replicate_vector(law: NoiseLaw, sub: CellLayout, rng: np.random.Generator,
                      window_cfg: WindowConfig) -> np.ndarray:
    """One replicate of the sweep at `law` (its layout and tilt), reduced as
    in 2.5 (a): (d,) with the protocol's statistics, NaN elsewhere."""
    protocol, sweep = _sweep_of(sub)
    ss_stats, ls_stats = nf.simulate_cell_windows(law, sub, rng, parts=(protocol,), window_cfg=window_cfg)
    q = np.full(nf.N_STATS, np.nan)
    if protocol == "ss":
        q[_SLICE["ss"]] = ncs.per_sweep_vectors(ss_stats, sub.sweep_number)[sweep]
    else:
        q[_SLICE["ls"]] = ls_stats[0]
    return q


def bootstrap_v(fit: SweepFit, mask: np.ndarray, Sigma_s: np.ndarray, sub: CellLayout, law_start: NoiseLaw,
                cfg: NoiseFitConfig, stage_cfg: SweepStageConfig, seed_binding_s: int, seed_v_s: int,
                refit: Callable[..., SweepFit] = fit_sweep) -> SweepBootstrap:
    """(P2), 2.5 (d) with D-122.1-D-122.2 and 2.6 (e): B_v replicates of the
    sweep at theta_hat(c, s) = fit.law (tau_m_hat(c), tau_s_hat(c), the
    sweep's layout and tilt), drawn in order from default_rng(seed_v_s), each
    reduced to its statistic vector and refitted by (P1) -- the same
    seed_binding_s, Sigma_s and start law_start = theta_hat(c), the binding
    function at M_refit -- each refit giving both y_m_b = log sigma_m and
    y_s_b = log sigma_s (its coordinates 2 and 3). v_m and v_s are their
    sample variances (ddof 1) over the same B_ok refits. A failed refit
    (fit_sweep's rule, or fewer than min_stats finite statistics) is left out
    of both and counted. `refit` has fit_sweep's signature (a test double in
    the smoke)."""
    if fit.failed:
        raise ValueError("a sweep whose fit failed is not bootstrapped (2.5 (d))")
    B = int(stage_cfg.B_v)
    cfg_refit = cfg if stage_cfg.M_refit is None else replace(cfg, M_binding=int(stage_cfg.M_refit))
    mask = np.asarray(mask, dtype=bool)
    rng = np.random.default_rng(int(seed_v_s))
    y_m = np.full(B, np.nan)
    y_s = np.full(B, np.nan)
    # Custom: the bootstrap loop (no library bootstraps an indirect-inference estimator).
    for b in range(B):
        q_b = _replicate_vector(fit.law, sub, rng, cfg.window_cfg)
        m_b = mask & np.isfinite(q_b)
        if int(m_b.sum()) < int(stage_cfg.min_stats):
            continue
        r = refit(q_b, m_b, Sigma_s, sub, law_start, cfg_refit, seed_binding_s)
        if not r.failed:
            y_m[b] = float(r.u[1])            # log sigma_m (U_NAMES[1])
            y_s[b] = float(r.u[2])            # log sigma_s (U_NAMES[2])
    ok = np.isfinite(y_m) & np.isfinite(y_s)
    y_m[~ok] = np.nan
    y_s[~ok] = np.nan
    B_ok = int(ok.sum())
    n_failed = B - B_ok
    v_m = float(np.var(y_m[ok], ddof=1)) if B_ok >= 2 else _NAN
    v_s = float(np.var(y_s[ok], ddof=1)) if B_ok >= 2 else _NAN
    flags: List[str] = []
    if n_failed > float(stage_cfg.max_failed_refit_frac) * B:
        flags.append("v_refits_failed:%d/%d" % (n_failed, B))
    if not (_v_usable(v_m) and _v_usable(v_s)):
        flags.append("v_none")
    return SweepBootstrap(v_m=v_m, v_s=v_s, y_m=y_m, y_s=y_s, B=B, B_ok=B_ok, flags=tuple(flags))


# ===========================================================================
#  (P3), (P4), (D-076.1): the pooling of one protocol
# ===========================================================================
@dataclass(eq=False)
class PoolResult:
    protocol: str
    sweeps: Tuple[int, ...]
    k: int
    Q: float                      # Cochran's Q at the fixed-effect mean
    tau2_raw: float               # (P3) before the clip
    tau2: float                   # max(0, tau2_raw)
    mu: float                     # (P4): the random-effects mean at the clipped tau2
    lam: np.ndarray               # (k,) lambda(c, s) = tau2 / (tau2 + v_s)
    y_shrunk: np.ndarray          # (k,) the shrunk log size (D-076.1): log sigma_m_tilde or log sigma_s_tilde


def pool_protocol(y: np.ndarray, v: np.ndarray, *, protocol: str = "", sweeps: Sequence[int] = ()) -> PoolResult:
    """(P3), (P4), (D-076.1) on the k sweeps of one protocol, for one pooled
    size (2.6 (e): called once with y = log sigma_m(c, s), v = v_m(c, s) and
    once with y = log sigma_s(c, s), v = v_s(c, s)); v > 0. k >= 2: the DerSimonian-Laird moment
    estimate, clipped at 0, the random-effects mean recomputed at the clipped
    value, each y shrunk by lambda; k = 1: mu = y, tau2 and lambda NaN, y
    unshrunk; k = 0: nothing pooled (NaN)."""
    # Custom: (P3)-(P4) with the clip and the recomputed random-effects mean; statsmodels'
    # combine_effects(method_re="dl") returns tau2 unclipped and weights with it (D-076's
    # clarification of 2026-10-26), so it is the smoke's oracle of tau2_raw only.
    y = np.asarray(y, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if y.shape != v.shape:
        raise ValueError("pool_protocol: y and v differ in length")
    k = int(y.size)
    sw = tuple(int(s) for s in sweeps)
    if sw and len(sw) != k:
        raise ValueError("pool_protocol: %d sweeps for %d values" % (len(sw), k))
    if k == 0:
        return PoolResult(protocol, sw, 0, _NAN, _NAN, _NAN, _NAN, np.zeros(0), np.zeros(0))
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(v)) and np.all(v > 0)):
        raise ValueError("pool_protocol: y finite and v finite > 0 are required")
    if k == 1:
        return PoolResult(protocol, sw, 1, _NAN, _NAN, _NAN, float(y[0]), np.array([_NAN]), y.copy())
    w = 1.0 / v
    w_sum = float(w.sum())
    y_fe = float((w * y).sum() / w_sum)
    Q = float((w * (y - y_fe) ** 2).sum())
    tau2_raw = (Q - (k - 1)) / (w_sum - float((w * w).sum()) / w_sum)
    tau2 = max(0.0, tau2_raw)
    w_re = 1.0 / (v + tau2)
    mu = float((w_re * y).sum() / w_re.sum())
    lam = tau2 / (tau2 + v)
    return PoolResult(protocol, sw, k, Q, float(tau2_raw), float(tau2), mu, lam, mu + lam * (y - mu))


# ===========================================================================
#  The stage: one sweep, the cell's pooling, the whole cell
# ===========================================================================
@dataclass(eq=False)
class SweepStageResult:
    specimen_id: int
    rows: List[Dict[str, object]]                    # one per sweep of S(c): the fields of 2.5 (g), 2.6 (h)
    pools: Dict[str, Dict[str, PoolResult]]          # 'ss', 'ls' -> 'sigma_m', 'sigma_s' -> PoolResult
    twin_laws: Dict[Tuple[str, int], NoiseLaw]       # (protocol, sweep) -> theta_tilde(c, s)
    flags: Tuple[str, ...]
    config: Optional[SweepStageConfig]
    seconds: float

    def to_dict(self) -> Dict[str, object]:
        """JSON-ready: arrays as lists, laws as dicts, NaN as None."""
        return {
            "specimen_id": int(self.specimen_id),
            "rows": [_jsonable(r) for r in self.rows],
            "pools": {pr: {co: _jsonable({"protocol": p.protocol, "sweeps": list(p.sweeps), "k": p.k, "Q": p.Q,
                                          "tau2_raw": p.tau2_raw, "tau2": p.tau2, "mu": p.mu, "lam": p.lam,
                                          "y_shrunk": p.y_shrunk}) for co, p in by_co.items()}
                      for pr, by_co in self.pools.items()},
            "twin_laws": {"%s:%d" % key: _law_dict(law) for key, law in self.twin_laws.items()},
            "flags": list(self.flags),
            "config": None if self.config is None else _jsonable(asdict(self.config)),
            "seconds": float(self.seconds),
        }

    def table_rows(self) -> List[Dict[str, object]]:
        """The per-sweep columns of 2.6 (g), in its order, one dict per sweep.
        The `_mV` columns without `raw` are the twin's theta_tilde(c, s) (for a
        failed sweep theta_hat(c)'s, D-122.3); the `raw` columns the sweep's
        own fit; v the bootstrap variances; w_m and w the shrinkage weights
        lambda_m and lambda. The `raw`, `v` and `w` columns are NaN for a
        failed sweep. The sweep's tilt as Block 4's data flow names it."""
        out = []
        for r in self.rows:
            twin = self.twin_laws[(str(r["protocol"]), int(r["sweep"]))]
            failed = bool(r["failed"])

            def own(key: str) -> float:
                return _NAN if failed else float(r[key])
            out.append({
                "specimen_id": int(self.specimen_id),
                "noise_sweep_protocol": str(r["protocol"]).upper(),
                "noise_sweep_number": int(r["sweep"]),
                "noise_tilt_mV_per_s": float(r["tilt_mV_per_s"]),
                "noise_sweep_sigma_f_mV": twin.sigma_f,
                "noise_sweep_sigma_m_raw_mV": own("sigma_m_mV"),
                "noise_sweep_sigma_m_mV": twin.sigma_m,
                "noise_sweep_v_log_sigma_m": own("v_log_sigma_m"),
                "noise_sweep_w_m": own("lambda_m"),
                "noise_sweep_sigma_s_raw_mV": own("sigma_s_mV"),
                "noise_sweep_sigma_s_mV": twin.sigma_s,
                "noise_sweep_v_log_sigma_s": own("v_log_sigma_s"),
                "noise_sweep_w": own("lambda"),
                "noise_sweep_flags": ";".join(r["flags"]),
            })
        return out

    def cell_columns(self) -> Dict[str, object]:
        """Block 9's per-cell columns of 2.6 (g), in its order, led by
        specimen_id: tau2_hat and mu_hat of log sigma_m, then of log sigma_s,
        each for SS then LS (the mu columns the assistant's addition)."""
        out: Dict[str, object] = {"specimen_id": int(self.specimen_id)}
        for co in COORDS:
            for pr in PROTOCOLS:
                p = self.pools.get(pr, {}).get(co)
                out["noise_cell_tau2_log_%s_%s" % (co, pr)] = _NAN if p is None else p.tau2
                out["noise_cell_mu_log_%s_%s" % (co, pr)] = _NAN if p is None else p.mu
        return out


def _law_dict(law: NoiseLaw) -> Dict[str, float]:
    """The law dict of 2.6 (g)."""
    return {"sigma_f_mV": law.sigma_f, "sigma_m_mV": law.sigma_m, "tau_m_s": law.tau_m,
            "sigma_s_mV": law.sigma_s, "tau_s_s": law.tau_s}


def _jsonable(x):
    """NaN and inf -> None, arrays and tuples -> lists, recursively."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else None
    return x


def run_sweep(cell_fit: CellFitResult, target: CellTarget, layout: CellLayout, cfg: NoiseFitConfig,
              stage_cfg: SweepStageConfig, protocol: str, sweep: int,
              log: Optional[Callable[[str], None]] = None, *,
              refit: Callable[..., SweepFit] = fit_sweep) -> Dict[str, object]:
    """One sweep's fit (P1) and bootstrap (P2): a dict of the per-sweep
    fields of 2.5 (g) and 2.6 (h) before the pooling (one cluster job per
    (cell, sweep)). The sweep fails with 'stats' (d_s < min_stats, no solve),
    with its fit's reason ('no_convergence', 'at_bound:<coordinates>'), then,
    checked next, with 'degenerate:sigma_f' (sigma_f_hat(c, s) <
    min_sigma_f_ratio sigma_f_hat(c), D-154.3), or with 'v_none' (v_m or v_s
    NaN, B_ok < 2, or not strictly positive, D-155); a sweep whose fit
    failed is not bootstrapped; a failed
    sweep is flagged 'sweep_fit_failed:<pr>:<sweep>:<reason>'. `refit` is
    passed to bootstrap_v (a test double in the smoke)."""
    t0 = time.time()
    say = log if log is not None else (lambda s: None)
    pr, s = _proto(protocol), int(sweep)
    sub = sweep_layout(layout, pr, s)
    q_s, mask = sweep_target(target, layout, pr, s)
    d = int(mask.sum())
    seed_b, seed_v = sweep_seeds(stage_cfg.seed, layout.specimen_id, pr, s)
    tau_m, tau_s = float(cell_fit.law.tau_m), float(cell_fit.law.tau_s)
    row: Dict[str, object] = {
        "specimen_id": int(layout.specimen_id), "protocol": pr, "sweep": s,
        "K_s": int(sub.K) if pr == "ss" else None, "n_LS": int(sub.n_ls) if pr == "ls" else None,
        "tilt_mV_per_s": float(sub.ss_tilt[0] if pr == "ss" else sub.ls_tilt[0]),
        "d": d, "labels_kept": [ncs.STAT_LABELS[j] for j in np.flatnonzero(mask)],
        "u": [_NAN] * len(U_NAMES), "sigma_f_mV": _NAN, "sigma_m_mV": _NAN, "tau_m_s": tau_m, "sigma_s_mV": _NAN,
        "tau_s_s": tau_s, "log_sigma_m": _NAN, "log_sigma_s": _NAN, "J": _NAN, "dof": d - len(U_NAMES),
        "n_calls": 0, "status": None, "at_bound": [], "failed": False, "reason": "",
        "v_log_sigma_m": _NAN, "v_log_sigma_s": _NAN, "B_v": int(stage_cfg.B_v), "B_ok": 0, "n_refits_failed": 0,
        "bootstrap_y_m": [], "bootstrap_y": [], "flags": [], "seconds": 0.0,
    }
    flags: List[str] = []
    if d < int(stage_cfg.min_stats):
        row["failed"], row["reason"] = True, "stats"
    else:
        if s not in cell_fit.per_sweep_cov:
            raise ValueError("the per-cell result carries no covariance for sweep %d (per_sweep_cov)" % s)
        Sigma_s = cell_fit.per_sweep_cov[s]
        fit = fit_sweep(q_s, mask, Sigma_s, sub, cell_fit.law, cfg, seed_b)
        reason = fit.reason
        if not fit.failed and fit.law.sigma_f < float(stage_cfg.min_sigma_f_ratio) * float(cell_fit.law.sigma_f):
            # Custom: D-154.3, checked after status and at_bound: a collapsed fast part fails the sweep.
            reason = "degenerate:sigma_f"
        flags.extend(fit.flags)
        row.update({"u": fit.u.tolist(), "sigma_f_mV": fit.law.sigma_f, "sigma_m_mV": fit.law.sigma_m,
                    "sigma_s_mV": fit.law.sigma_s, "log_sigma_m": float(fit.u[1]), "log_sigma_s": float(fit.u[2]),
                    "J": fit.J, "dof": fit.dof, "n_calls": fit.n_calls, "status": fit.status,
                    "at_bound": list(fit.at_bound), "failed": bool(reason), "reason": reason})
        say("%s sweep %d: fit u=%s J=%.4g calls=%d status=%d%s (%.0f s)"
            % (pr.upper(), s, np.array2string(fit.u, precision=4), fit.J, fit.n_calls, fit.status,
               (" FAILED " + reason) if reason else "", time.time() - t0))
        if not reason:
            boot = bootstrap_v(fit, mask, Sigma_s, sub, cell_fit.law, cfg, stage_cfg, seed_b, seed_v, refit=refit)
            flags.extend(f for f in boot.flags if f != "v_none")
            row.update({"v_log_sigma_m": boot.v_m, "v_log_sigma_s": boot.v_s, "B_ok": boot.B_ok,
                        "n_refits_failed": boot.B - boot.B_ok, "bootstrap_y_m": boot.y_m.tolist(),
                        "bootstrap_y": boot.y_s.tolist()})
            if not (_v_usable(boot.v_m) and _v_usable(boot.v_s)):
                row["failed"], row["reason"] = True, "v_none"
            say("%s sweep %d: v_m=%.4g v_s=%.4g B_ok=%d/%d (%.0f s)"
                % (pr.upper(), s, boot.v_m, boot.v_s, boot.B_ok, boot.B, time.time() - t0))
    if row["failed"]:
        flags.append("sweep_fit_failed:%s:%d:%s" % (pr, s, row["reason"]))
    row["flags"] = flags
    row["seconds"] = float(time.time() - t0)
    return row


_ROW_KEYS = {"sigma_m": ("log_sigma_m", "v_log_sigma_m", "lambda_m", "sigma_m_tilde_mV"),
             "sigma_s": ("log_sigma_s", "v_log_sigma_s", "lambda", "sigma_s_tilde_mV")}


def pool_cell(rows: Sequence[Dict[str, object]], cell_law: NoiseLaw, specimen_id: int, *,
              stage_cfg: Optional[SweepStageConfig] = None, seconds: Optional[float] = None) -> SweepStageResult:
    """The poolings per protocol over the rows of run_sweep (the sweeps whose
    fit did not fail), each applied twice and independently (2.6 (e)): to
    (log sigma_m, v_m) and to (log sigma_s, v_s), by pool_protocol. The twins'
    laws by (P0') -- theta_tilde(c, s) = (sigma_f(c, s), sigma_m_tilde(c, s),
    tau_m_hat(c), sigma_s_tilde(c, s), tau_s_hat(c)) -- and, for a failed
    sweep, theta_hat(c) whole (2.5 (f), D-122.3). The rows come back with
    lambda_m, sigma_m_tilde_mV, lambda, sigma_s_tilde_mV and the twin's law
    added."""
    t0 = time.time()
    out_rows = [dict(r) for r in rows]
    for r in out_rows:
        r["flags"] = list(r["flags"])
        if not r["failed"] and not (_v_usable(r["v_log_sigma_m"]) and _v_usable(r["v_log_sigma_s"])):
            # D-155: a row made elsewhere with an unusable variance is a failed sweep, never pooled
            r["failed"], r["reason"] = True, "v_none"
            r["flags"].append("sweep_fit_failed:%s:%d:v_none" % (r["protocol"], int(r["sweep"])))
    pools: Dict[str, Dict[str, PoolResult]] = {}
    twins: Dict[Tuple[str, int], NoiseLaw] = {}
    for pr in PROTOCOLS:
        idx = [i for i, r in enumerate(out_rows) if r["protocol"] == pr and not r["failed"]]
        sweeps = [out_rows[i]["sweep"] for i in idx]
        pools[pr] = {}
        for co in COORDS:
            k_y, k_v, k_lam, k_tilde = _ROW_KEYS[co]
            pool = pool_protocol([out_rows[i][k_y] for i in idx], [out_rows[i][k_v] for i in idx],
                                 protocol=pr, sweeps=sweeps)
            pools[pr][co] = pool
            for j, i in enumerate(idx):
                out_rows[i][k_lam] = float(pool.lam[j])
                out_rows[i][k_tilde] = math.exp(float(pool.y_shrunk[j]))
        for i in idx:
            r = out_rows[i]
            twins[(pr, int(r["sweep"]))] = NoiseLaw(float(r["sigma_f_mV"]), float(r["sigma_m_tilde_mV"]),
                                                    float(cell_law.tau_m), float(r["sigma_s_tilde_mV"]),
                                                    float(cell_law.tau_s))   # (P0')
    for r in out_rows:
        key = (str(r["protocol"]), int(r["sweep"]))
        if r["failed"]:
            # Custom: the failed-sweep rule (D-122.3): the twin carries the cell's whole law.
            for co in COORDS:
                r[_ROW_KEYS[co][2]], r[_ROW_KEYS[co][3]] = _NAN, _NAN
            twins[key] = cell_law
        r["twin_law"] = _law_dict(twins[key])
    flags = tuple(dict.fromkeys(f for r in out_rows for f in r["flags"]))
    secs = float(sum(float(r.get("seconds", 0.0)) for r in out_rows) + (time.time() - t0)) if seconds is None \
        else float(seconds)
    return SweepStageResult(specimen_id=int(specimen_id), rows=out_rows, pools=pools, twin_laws=twins, flags=flags,
                            config=stage_cfg, seconds=secs)


def fit_sweeps(cell_fit: CellFitResult, target: CellTarget, layout: CellLayout,
               cfg: Optional[NoiseFitConfig] = None, stage_cfg: SweepStageConfig = SweepStageConfig(),
               log: Optional[Callable[[str], None]] = None) -> SweepStageResult:
    """The whole stage for one cell: run_sweep on every sweep of S(c) -- the
    SS sweeps with kept windows in the layout's order, then the LS sweeps --
    then pool_cell. cfg defaults to the per-cell fit's own config."""
    t0 = time.time()
    if int(cell_fit.specimen_id) != int(layout.specimen_id):
        raise ValueError("the per-cell result is of specimen %d, the layout of %d"
                         % (cell_fit.specimen_id, layout.specimen_id))
    cfg = cell_fit.config if cfg is None else cfg
    rows = [run_sweep(cell_fit, target, layout, cfg, stage_cfg, "ss", s, log) for s in layout.sweep_numbers]
    rows += [run_sweep(cell_fit, target, layout, cfg, stage_cfg, "ls", int(s), log) for s in layout.ls_sweep_number]
    res = pool_cell(rows, cell_fit.law, layout.specimen_id, stage_cfg=stage_cfg)
    res.seconds = float(time.time() - t0)
    return res
