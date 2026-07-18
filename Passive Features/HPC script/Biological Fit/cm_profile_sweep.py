# -*- coding: utf-8 -*-
"""
cm_profile_sweep.py
===================

Per-cell C_m profile-likelihood sweep over the SS time-weight constant tau_w.

Purpose
-------
tau_w (the exponential brief-pulse time-weight constant) is a MODELING knob that
moves the point estimate (theory note section 6, section 9). It must be chosen by
data, not by eye. This module selects it by the sharpness of the C_m
profile-likelihood: a flat profile means C_m is still unidentified; a sharp
minimum means tau_w has restored the log C_m column of the weighted information
matrix A (theory section 8) and C_m is identified.

The object (for each FIXED cell c, each FIXED tau_w, on a grid of FIXED log C_m)
-----------------------------------------------------------------------------
    PL_{tau_w}^{(c)}(C_m) = min_{log R_m, log R_a}
                              L_{tau_w}^{(c)}(log C_m, log R_m, log R_a),

where L_{tau_w}^{(c)} is the SAME multi-protocol 'relative' loss the fit uses,
built via passive_long_step_training.build_multi_protocol_loss with

    ss_sample_weight_fn = exp_time_weight(t0_s, tau_w * 1e-3).

Isolating tau_w as the ONLY varying quantity (the train/validation split and the
window are held fixed at whatever Cell 4 produced) is why this module calls
build_multi_protocol_loss DIRECTLY rather than re-calling integrate_long_step per
tau_w -- re-patching would also re-split, confounding the comparison.

Sharpness metrics (reported per cell, per tau_w; SELECTION is on HW_rho)
-----------------------------------------------------------------------
    HW_rho^{(c)}(tau_w) = width of { log C_m : PL <= (1 + rho) * PL_min }
                          [log-C_m units; smaller = sharper], crossings
                          linearly interpolated so the metric is grid-independent.
    kappa^{(c)}(tau_w)  = d^2 PL / d(log C_m)^2 at the profile minimiser
                          [larger = sharper], from a local quadratic fit.

FLAGGED: L_{tau_w} is the unitless 'relative' quasi-MLE objective, so HW_rho and
kappa are DESCRIPTIVE sharpness measures, NOT chi^2-calibrated CIs (theory
section 6). For CIs use the sandwich A^{-1} B A^{-1} or a bootstrap under the true
stationary noise -- not the weighted Fisher inverse.

Synthetic bias check (only when C_m^star is known)
--------------------------------------------------
    bias_log^{(c)}(tau_w) = log Cm_hat^{(c)}(tau_w) - log C_m^star.
The best tau_w for cell c has BOTH small HW_rho AND small |bias_log|; these are
reported separately and NOT collapsed into one score (the user chooses the
trade-off).

Separation of concerns (scientific-coding directive)
----------------------------------------------------
    profile_cm           : computes PL_{tau_w}^{(c)}(C_m) for one (cell, tau_w).
    profile_sharpness    : extracts (Cm_hat, kappa, HW_rho) from one profile.
    build_relative_loss_for_tau_w : the only coupling to the fitting module.
    sweep_tau_w_per_cell : orchestration across cells x tau_w (per-cell winners).
    save_profiles / load_profiles : I/O (JSON).
    plot_profiles_per_cell : visualisation (matplotlib; never tangled with logic).
None of the numeric functions plot or do I/O; none of the I/O / plot functions
re-fit. Swapping the grid, the metric, or the loss never touches the others.

This module is decoupled from NEURON exactly as passive_long_step_training is:
it consumes the duck-typed `cell` API (set_passive / set_e_pas / simulate, via
the loss closure) and SweepBundle-like train bundles only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

# The ONLY coupling to the fitting code: the loss builder and the weight factory.
from passive_long_step_training import (
    build_multi_protocol_loss,
    exp_time_weight,
    gauss_time_weight,
)


# ===========================================================================
#  Result containers (plain, serialisable)
# ===========================================================================
@dataclass
class CmProfile:
    """One C_m profile PL_{tau_w}^{(c)}(C_m) plus its sharpness summary.

    All C_m quantities are stored in BOTH linear (cm_grid) and the native
    log-grid (logcm_grid) because the metrics live in log-C_m units while the
    user reads C_m in uF/cm^2.
    """
    specimen_id: int
    tau_w_ms: float
    shape: str                       # 'exp' | 'gauss'
    cm_grid: List[float]             # uF/cm^2 (linear), len = n_grid
    logcm_grid: List[float]          # log(cm_grid)
    pl: List[float]                  # PL_{tau_w}^{(c)}(C_m) at each grid point
    rm_path: List[float]             # warm-started argmin R_m at each grid point
    ra_path: List[float]             # warm-started argmin R_a at each grid point
    cm_hat: float                    # argmin C_m (uF/cm^2)
    logcm_hat: float                 # argmin log C_m
    pl_min: float                    # PL at the minimiser
    kappa: float                     # curvature d^2 PL / d(log C_m)^2 at min
    hw_rho: float                    # relative-rise half-width (log-C_m units)
    rho: float                       # the threshold used for hw_rho
    bias_log: Optional[float] = None  # log cm_hat - log cm_true (synthetic only)


# ===========================================================================
#  1. The profile: PL_{tau_w}^{(c)}(C_m) for ONE (cell, tau_w)
# ===========================================================================
def profile_cm(
    loss: Callable[[float, float, float], float],
    *,
    cm_bounds: Tuple[float, float] = (0.3, 3.0),
    n_grid: int = 21,
    rm_bounds: Tuple[float, float] = (1e3, 1e5),
    ra_bounds: Tuple[float, float] = (50.0, 1000.0),
    rm_init: float = 15000.0,
    ra_init: float = 150.0,
    inner_method: str = "Powell",
    inner_maxiter: int = 200,
) -> Dict[str, np.ndarray]:
    """Compute PL(C_m) = min_{log R_m, log R_a} loss(log C_m, log R_m, log R_a)
    on a log-spaced C_m grid, for a SINGLE fixed loss closure (one cell, one
    tau_w). Warm-starts (log R_m, log R_a) along the grid so the inner optimiser
    tracks the iso-tau_m valley instead of re-searching from scratch.

    `loss` is exactly the closure returned by build_multi_protocol_loss; it takes
    (cm_log, rm_log, ra_log) and resets the cell's passive params internally on
    every call, so reusing a fitted cell object is safe (its current state is
    overwritten). Derivative-free Powell is used because `loss` is a black-box
    NEURON simulation (no analytic gradient; finite differences are noisy and
    expensive).

    Returns arrays (NOT a CmProfile -- that is assembled by the orchestrator):
        logcm_grid, cm_grid, pl, rm_path, ra_path.
    """
    logcm_grid = np.linspace(np.log(cm_bounds[0]), np.log(cm_bounds[1]), n_grid)
    log_rm_bounds = (np.log(rm_bounds[0]), np.log(rm_bounds[1]))
    log_ra_bounds = (np.log(ra_bounds[0]), np.log(ra_bounds[1]))

    pl = np.full(n_grid, np.nan)
    rm_path = np.full(n_grid, np.nan)
    ra_path = np.full(n_grid, np.nan)

    warm = np.array([np.log(rm_init), np.log(ra_init)], dtype=float)
    for i, lcm in enumerate(logcm_grid):
        def _inner(x):                       # x = (log R_m, log R_a)
            return float(loss(float(lcm), float(x[0]), float(x[1])))
        res = minimize(
            _inner, warm, method=inner_method,
            bounds=[log_rm_bounds, log_ra_bounds],
            options={"maxiter": inner_maxiter, "xtol": 1e-4, "ftol": 1e-4},
        )
        pl[i] = float(res.fun)
        rm_path[i] = float(res.x[0])
        ra_path[i] = float(res.x[1])
        if np.all(np.isfinite(res.x)):       # warm-start the next grid point
            warm = res.x.copy()

    return dict(
        logcm_grid=logcm_grid,
        cm_grid=np.exp(logcm_grid),
        pl=pl,
        rm_path=np.exp(rm_path),
        ra_path=np.exp(ra_path),
    )


# ===========================================================================
#  2. Sharpness metric: (Cm_hat, kappa, HW_rho) from one profile
# ===========================================================================
def profile_sharpness(
    logcm_grid: np.ndarray,
    pl: np.ndarray,
    *,
    rho: float = 0.5,
) -> Dict[str, float]:
    """Extract identifiability summaries from one PL(log C_m) curve.

    Returns dict with:
        logcm_hat : argmin log C_m (parabolic-refined when the minimum is
                    interior, so it is not pinned to a grid node)
        cm_hat    : exp(logcm_hat)  [uF/cm^2]
        pl_min    : PL at the minimiser
        kappa     : d^2 PL / d(log C_m)^2 at the minimiser, from a local
                    3-point quadratic fit [larger = sharper]
        hw_rho    : width of { log C_m : PL <= (1+rho) PL_min } in log-C_m
                    units, computed from the LOCAL QUADRATIC at the minimiser:
                        PL(c) ~= PL_min + (1/2) kappa (c - c_hat)^2
                        => threshold (1+rho) PL_min reached at
                           (1/2) kappa Delta^2 = rho PL_min
                        => HW_rho = 2 Delta = 2 sqrt(2 rho PL_min / kappa).
                    This is GRID-INDEPENDENT by construction (it depends only on
                    kappa and PL_min, both from the local fit), unlike a width
                    read off grid-node crossings, which on a coarse grid varies
                    by up to a grid step (the defect the smoke test caught in the
                    interpolated-crossing version). +inf when kappa <= 0 or
                    PL_min <= 0 (a flat / degenerate profile: 'no finite
                    half-width' -- the unidentified end the sweep must flag, not
                    mis-score).

                    CAVEAT (flagged): HW_rho is exact only where the profile is
                    locally quadratic near c_hat; for a strongly non-quadratic
                    profile it is the quadratic-approximation half-width, read
                    together with kappa and the saved PL array (plot it). Like
                    kappa it is a DESCRIPTIVE sharpness measure, NOT a
                    chi^2-calibrated CI (the loss is the unitless 'relative'
                    quasi-MLE objective; theory section 6).

    Pure function: no NEURON, no I/O, no plotting. Validated against an analytic
    quadratic profile in the smoke test (kappa = 2 beta, HW_rho exact).
    """
    x = np.asarray(logcm_grid, dtype=float)
    y = np.asarray(pl, dtype=float)
    if x.size < 3 or not np.all(np.isfinite(y)):
        return dict(logcm_hat=np.nan, cm_hat=np.nan, pl_min=np.nan,
                    kappa=np.nan, hw_rho=np.nan)

    j = int(np.argmin(y))

    # --- curvature kappa via a local quadratic fit y ~ a (x-x0)^2 + ... -------
    # Use the minimiser and its two neighbours when interior; fall back to the
    # three nodes at the relevant edge otherwise. np.polyfit gives the quadratic
    # coefficient a; kappa = 2a (second derivative of a x^2 + b x + c).
    if 1 <= j <= x.size - 2:
        sl = slice(j - 1, j + 2)
    elif j == 0:
        sl = slice(0, 3)
    else:
        sl = slice(x.size - 3, x.size)
    a2 = np.polyfit(x[sl], y[sl], 2)
    a = a2[0]
    kappa = float(2.0 * a)

    # --- refined minimiser AND minimum VALUE (vertex of the local parabola) ---
    # Both the vertex location and the vertex value are taken from the local
    # quadratic a c^2 + b c + cc so that pl_min is not the grid-node overshoot
    # y[j] (which depends on whether a node happens to sit at c_hat). This is
    # what makes HW_rho = 2 sqrt(2 rho pl_min / kappa) grid-independent: BOTH
    # kappa and pl_min come from the fit, neither from a single node.
    b, cc = a2[1], a2[2]
    if 1 <= j <= x.size - 2 and a > 0:
        logcm_hat = float(-b / (2.0 * a))
        if not (x[sl][0] <= logcm_hat <= x[sl][-1]):   # keep vertex in-window
            logcm_hat = float(x[j])
        pl_min = float(a * logcm_hat ** 2 + b * logcm_hat + cc)  # vertex value
    else:
        logcm_hat = float(x[j])
        pl_min = float(y[j])

    # --- relative-rise half-width from the LOCAL QUADRATIC (grid-independent) --
    #   PL(c) ~= PL_min + (1/2) kappa (c - c_hat)^2  reaches (1+rho) PL_min at
    #   (1/2) kappa Delta^2 = rho PL_min  =>  HW_rho = 2 Delta
    #                                              = 2 sqrt(2 rho PL_min / kappa).
    #   kappa <= 0 (not a minimum) or PL_min <= 0 -> no finite half-width.
    if kappa > 0.0 and pl_min > 0.0:
        hw_rho = float(2.0 * np.sqrt(2.0 * rho * pl_min / kappa))
    else:
        hw_rho = float("inf")

    return dict(logcm_hat=logcm_hat, cm_hat=float(np.exp(logcm_hat)),
                pl_min=pl_min, kappa=kappa, hw_rho=hw_rho)


# ===========================================================================
#  3. The only coupling to the fitting module: build L_{tau_w}^{(c)}
# ===========================================================================
def build_relative_loss_for_tau_w(
    cell,
    train_bundles: Sequence,
    v_rest_mV: float,
    *,
    tau_w_ms: float,
    ss_window_ms: Tuple[float, float] = (0.5, 100.0),
    ss_t0_ms: Optional[float] = None,
    shape: str = "exp",
    ls_window_ms_after_onset: float = 150.0,
    r_in_target: str = "peak",
    weighting: str = "relative",
) -> Callable[[float, float, float], float]:
    """Return the loss closure for ONE cell at ONE tau_w, identical to what the
    fit uses, with the SS time-weight fixed to `tau_w_ms` and `shape`.

    t0 for the weight defaults to the window start (the pulse offset), matching
    integrate_long_step's _make_ss_weight_fn. Everything else (window, LS window,
    r_in_target, cross-bundle weighting) is held fixed across the sweep so tau_w
    is the only quantity that varies -- the comparison is apples-to-apples.
    """
    t0_s = (ss_t0_ms if ss_t0_ms is not None else ss_window_ms[0]) * 1e-3
    scale_s = tau_w_ms * 1e-3
    if shape == "exp":
        wfn = exp_time_weight(t0_s, scale_s)
    elif shape == "gauss":
        wfn = gauss_time_weight(t0_s, scale_s)
    elif shape == "none":
        wfn = None
    else:
        raise ValueError(f"shape must be 'exp'|'gauss'|'none', got {shape!r}")

    return build_multi_protocol_loss(
        cell, train_bundles, v_rest_mV,
        ss_window_ms=tuple(ss_window_ms),
        ls_window_ms_after_onset=ls_window_ms_after_onset,
        r_in_target=r_in_target, weighting=weighting,
        ss_sample_weight_fn=wfn,
    )


# ===========================================================================
#  4. Orchestration: sweep tau_w for every cell, PER-CELL winners
# ===========================================================================
@dataclass
class CellSweepInput:
    """Everything the sweep needs for one cell, decoupled from where it came
    from (the runner adapter fills these from df + opt_inputs)."""
    specimen_id: int
    cell: object                     # duck-typed: set_passive/set_e_pas/simulate
    train_bundles: Sequence
    v_rest_mV: float
    ss_window_ms: Tuple[float, float] = (0.5, 100.0)
    cm_true: Optional[float] = None  # for the synthetic bias check (uF/cm^2)


def sweep_tau_w_per_cell(
    cells: Sequence[CellSweepInput],
    tau_w_grid_ms: Sequence[float] = (2.0, 5.0, 10.0),
    *,
    shape: str = "exp",
    rho: float = 0.5,
    cm_bounds: Tuple[float, float] = (0.3, 3.0),
    n_grid: int = 21,
    rm_bounds: Tuple[float, float] = (1e3, 1e5),
    ra_bounds: Tuple[float, float] = (50.0, 1000.0),
    r_in_target: str = "peak",
    ls_window_ms_after_onset: float = 150.0,
    verbose: bool = True,
) -> Dict[int, List[CmProfile]]:
    """For each cell c and each tau_w, compute PL_{tau_w}^{(c)}(C_m) and its
    sharpness summary. Returns {specimen_id: [CmProfile per tau_w]}.

    Selection is PER CELL (user choice): no cohort aggregation is performed here.
    Use `summarise_per_cell` to print each cell's argmin tau_w by HW_rho with the
    bias shown alongside (sharp-but-biased is worse than slightly-less-sharp-but-
    centred; the two are reported separately, never collapsed).
    """
    out: Dict[int, List[CmProfile]] = {}
    for ci in cells:
        profs: List[CmProfile] = []
        for tau in tau_w_grid_ms:
            loss = build_relative_loss_for_tau_w(
                ci.cell, ci.train_bundles, ci.v_rest_mV,
                tau_w_ms=float(tau), ss_window_ms=ci.ss_window_ms,
                shape=shape, ls_window_ms_after_onset=ls_window_ms_after_onset,
                r_in_target=r_in_target, weighting="relative",
            )
            prof = profile_cm(
                loss, cm_bounds=cm_bounds, n_grid=n_grid,
                rm_bounds=rm_bounds, ra_bounds=ra_bounds,
            )
            sh = profile_sharpness(prof["logcm_grid"], prof["pl"], rho=rho)
            bias = (None if ci.cm_true is None or not np.isfinite(sh["logcm_hat"])
                    else float(sh["logcm_hat"] - np.log(ci.cm_true)))
            cp = CmProfile(
                specimen_id=int(ci.specimen_id), tau_w_ms=float(tau), shape=shape,
                cm_grid=prof["cm_grid"].tolist(),
                logcm_grid=prof["logcm_grid"].tolist(),
                pl=prof["pl"].tolist(),
                rm_path=prof["rm_path"].tolist(),
                ra_path=prof["ra_path"].tolist(),
                cm_hat=sh["cm_hat"], logcm_hat=sh["logcm_hat"],
                pl_min=sh["pl_min"], kappa=sh["kappa"], hw_rho=sh["hw_rho"],
                rho=rho, bias_log=bias,
            )
            profs.append(cp)
            if verbose:
                bstr = "   n/a" if bias is None else f"{bias:+.3f}"
                print(f"[sweep] sid={ci.specimen_id}  tau_w={tau:5.1f} ms  "
                      f"Cm_hat={sh['cm_hat']:.3f}  HW_rho={sh['hw_rho']:.4f}  "
                      f"kappa={sh['kappa']:.4g}  bias_log={bstr}")
        out[int(ci.specimen_id)] = profs
    return out


def summarise_per_cell(
    results: Dict[int, List[CmProfile]],
) -> "List[dict]":
    """One row per (cell, tau_w) plus a per-cell winner flag (argmin HW_rho).
    Pure: returns a list of dicts (the runner wraps it in a DataFrame to print).
    The winner is by HW_rho only; bias_log is carried so the user can override
    when a sharp tau_w is badly biased (the trade-off is theirs, not collapsed)."""
    rows: List[dict] = []
    for sid, profs in results.items():
        finite = [p for p in profs if np.isfinite(p.hw_rho)]
        best_tau = (min(finite, key=lambda p: p.hw_rho).tau_w_ms
                    if finite else None)
        for p in profs:
            rows.append(dict(
                specimen_id=sid, tau_w_ms=p.tau_w_ms, shape=p.shape,
                cm_hat=p.cm_hat, hw_rho=p.hw_rho, kappa=p.kappa,
                bias_log=(np.nan if p.bias_log is None else p.bias_log),
                is_winner=(best_tau is not None and p.tau_w_ms == best_tau),
            ))
    return rows


# ===========================================================================
#  5. I/O (JSON) -- separate from compute and from plotting
# ===========================================================================
def save_profiles(results: Dict[int, List[CmProfile]], path: str) -> None:
    """Serialise the full sweep (grids + profiles + summaries) to JSON so the
    profiles can be re-inspected/re-plotted without re-running NEURON."""
    blob = {str(sid): [asdict(p) for p in profs]
            for sid, profs in results.items()}
    with open(path, "w") as fh:
        json.dump(blob, fh, indent=2)


def load_profiles(path: str) -> Dict[int, List[CmProfile]]:
    """Inverse of save_profiles."""
    with open(path) as fh:
        blob = json.load(fh)
    return {int(sid): [CmProfile(**d) for d in profs]
            for sid, profs in blob.items()}


# ===========================================================================
#  6. Plotting -- separate; never tangled with compute or I/O
# ===========================================================================
def plot_profiles_per_cell(results: Dict[int, List[CmProfile]],
                           cm_true: Optional[float] = None,
                           savepath: Optional[str] = None):
    """One panel per cell, the tau_w profiles overlaid (PL vs C_m). Marks each
    profile's minimiser and, if given, the true C_m. Returns the Figure; saves a
    PNG if `savepath` is given. Import is local so the numeric path never needs
    matplotlib."""
    import matplotlib.pyplot as plt

    sids = list(results.keys())
    n = len(sids)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow),
                             squeeze=False)
    for k, sid in enumerate(sids):
        ax = axes[k // ncol][k % ncol]
        for p in results[sid]:
            cm = np.asarray(p.cm_grid)
            pl = np.asarray(p.pl)
            ax.plot(cm, pl, marker="o", ms=3, lw=1.2,
                    label=fr"$\tau_w$={p.tau_w_ms:g} ms (HW={p.hw_rho:.3f})")
            if np.isfinite(p.cm_hat):
                ax.axvline(p.cm_hat, ls=":", lw=0.8, alpha=0.5)
        if cm_true is not None:
            ax.axvline(cm_true, color="k", ls="--", lw=1.0, label=r"$C_m^\star$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$C_m$  [$\mu$F/cm$^2$]")
        ax.set_ylabel(r"$\mathrm{PL}_{\tau_w}(C_m)$  (relative loss)")
        ax.set_title(f"specimen {sid}")
        ax.legend(fontsize=7)
    for k in range(n, nrow * ncol):          # blank unused panels
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140, bbox_inches="tight")
    return fig


# ===========================================================================
#  7. Runner adapter: build CellSweepInput list from df + opt_inputs
# ===========================================================================
def cells_from_fit(df, opt_inputs, *, cm_true: Optional[float] = None,
                   cell_col: str = "neuron_cell") -> List[CellSweepInput]:
    """Assemble the sweep inputs from Cell 4's outputs WITHOUT rebuilding cells.

    df          : the monolith fit_cells() DataFrame (BEFORE the neuron_cell drop)
    opt_inputs  : the list returned by [prepare_optimiser_inputs(cd) ...], aligned
                  row-for-row with df (same cell order).
    cm_true     : injected C_m^star for the synthetic bias check (None on real
                  data -> bias_log stays None).

    Reads train_bundles / train_window_ms / v_rest_mV off each OptimiserInputs and
    the fitted cell off df[cell_col]. Fails loud if the cell column was already
    dropped (so you don't silently sweep against a rebuilt-from-nothing cell)."""
    if cell_col not in df.columns:
        raise KeyError(
            f"[cells_from_fit] df has no '{cell_col}' column -- it was dropped "
            f"before saving. Run the sweep on the IN-MEMORY df from Cell 4 "
            f"(before df_save = df.drop(...)), not the reloaded CSV.")
    if len(df) != len(opt_inputs):
        raise ValueError(
            f"[cells_from_fit] len(df)={len(df)} != len(opt_inputs)="
            f"{len(opt_inputs)}; they must be row-aligned (same cell order).")

    cells: List[CellSweepInput] = []
    for (_, row), oi in zip(df.iterrows(), opt_inputs):
        cells.append(CellSweepInput(
            specimen_id=int(row["specimen_id"]),
            cell=row[cell_col],
            train_bundles=oi.train_bundles,
            v_rest_mV=float(oi.v_rest_mV),
            ss_window_ms=tuple(oi.train_window_ms),
            cm_true=cm_true,
        ))
    return cells
