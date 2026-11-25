# -*- coding: utf-8 -*-
"""
run_ih_fit.py
=============

Stage-6 entrypoint of the I_h campaign (plan section 8, Stage 6). It is the
I_h counterpart of `run_biological_fit.py`, which it leaves untouched.

What it adds over run_biological_fit.py
---------------------------------------
1. ARMS. One flag, `--arm`, selects a complete, named configuration:

     baseline_runB     3-D passive, LEGACY loader (100 pA cap, no depolarising
                       long steps), legacy split (2 smallest hyperpolarising
                       steps, 12 mV deflection cap), RMSD window = 60 ms after
                       onset. This is run B, reproduced. It normally does NOT
                       need re-running: `--baseline-csv` folds run B's existing
                       phase2_results.csv into the comparison table instead.
     passive_fullstep  3-D passive on the D-006 protocol: every recorded
                       hyperpolarising step admitted, roles per plan section 5,
                       RMSD window = the WHOLE step. The control that separates
                       "I_h helped" from "the window and the extra steps
                       helped" (plan section 5, [reasoning]).
     ih6               the same protocol with theta =
                       (C_m, R_m, R_a, gbar_h, dv_h, kappa_tau).
     ih5               SPEC 2.8 (D-206 to D-209): theta = (C_m, R_m, R_a,
                       gbar_h, kappa_tau), dv_h = 0; long-step roles 'd207',
                       cross-bundle weighting 'ls_amplitude', per-sample
                       long-step weight 'peak_decay' (training loss only).
                       passive_fullstep runs on the same protocol since 2.8;
                       ih6 and baseline_runB keep D-006's (d006, relative,
                       none). --ls-role-rule / --weighting /
                       --ls-sample-weight override an arm's protocol.

   What a fit records (SPEC 2.8 (a), (f); phase2_results.csv, after the
   existing columns): tau_h_pk_ms = kappa_tau x tau_h_base_pk_ms (the peak of
   the mechanism's tau_h over [-140, -40] mV at kappa_tau = 1), the protocol,
   loss_at_estimate (the training loss at the returned estimate, re-evaluated)
   and loss_at_eval_theta (at --eval-theta), v_deepest_train_mV and
   gh_at_v_deepest_S_cm2 = gbar x m_inf(v*). --tau-h-pk-bounds-ms gives the
   box of kappa_tau in ms.

     ih6_d207          SPEC 2.9 (a), D-234: the six parameters of ih6
                       (dv_h fitted again, box [-10, 10] mV, linear) on the
                       protocol of ih5 (d207, ls_amplitude, peak_decay).

   THE LOCAL REFINEMENT (SPEC 2.9 (b), Block 20). On ih5, ih6_d207 and
   passive_fullstep (`refine="nelder_mead"` [corrected 2026-11-25, Block 27:
   their default is now "trf", below]; ih6 and baseline_runB: "none")
   the Phase-2 fit of each cell ends with bounded Nelder-Mead searches from
   the best points of the global search (local_refine.py). It is applied by
   replacing `mono.gp_minimize` with `local_refine.wrap_minimizer(<the
   original>)` around the call of `mono.fit_one_cell` only, restored after it
   (also on an exception), so everything fit_one_cell derives is at the
   refined estimate; the monolith is not edited. --refine / --refine-starts /
   --refine-maxfev override it. phase2_results.csv gains refine,
   loss_after_search, loss_after_refine, refine_improved, refine_n_starts,
   refine_n_evals, refine_best_start and n_workers; refine_starts.csv has
   one row per cell and start. --n-workers W computes the sweeps of one loss
   evaluation in W processes (SPEC 2.9 (c), Block 19).
   Reading a refined fit: the *_sigma columns come from the global search's
   GP around `result.fun` (at a refined fun they shrink, or are 0), the loss
   history is that of the global evaluations, and Phase 3 uses the search's
   GP (SPEC Block 20, notes for the reader).

   THE LEAST-SQUARES REFINEMENT (SPEC 2.10 (c), Block 27, D-245). Since
   Block 27 the default refinement of ih5, ih6_d207 and passive_fullstep is
   `trf` (Nelder-Mead stays selectable by --refine nelder_mead): from the
   same starts, scipy's least_squares(method="trf") on the residual vector
   r(q) of SPEC 2.10 (a) (local_refine.refine_trf). The loss fit_one_cell
   builds is captured by a shim around mono._build_loss_function (removed in
   a finally) and its make_residuals() is the residual function; the global
   search is untouched. --refine-max-nfev (30) is its budget per start.
   phase2_results.csv gains l2_after_search, l2_after_refine,
   refine_n_converged, refine_n_failed, sqrt_l2_at_estimate,
   sqrt_l2_at_eval_theta and on_bound_<name> ("lo" / "hi" when the
   estimate's unit coordinate is below 0.01 / above 0.99); refine_starts.csv
   gains l2_0, l2, converged, optimality, n_sims, x_solver_<name>,
   active_<name>, sv_1..sv_p.

   THE SENSITIVITY MODE (SPEC 2.10 (d), Block 27, D-246). With
   --sensitivity-theta "name=value,..." and --sensitivity-out DIR nothing is
   fitted: per cell, at that point, the sensitivities of every bundle's
   trace (central differences, --sensitivity-h) are stored with the
   recorded bundles, the identifiability table of the training bundles under
   the arm's protocol (W0) and with no decay after the peak (W1), and the
   check of the model's trace against the recorded one (truth_trace_check).
   layout_from_stored rebuilds the layouts from the stored files (no NEURON).

   The default box of gbar is [1e-5, 1e-3] S/cm^2 on every arm (SPEC 2.9 (a)).

   `--fit-params` overrides an arm's axis list, so the 4-D arm of Stage 7 is
   `--arm ih6 --fit-params Cm,Rm,Ra,gbar`.

2. THE I_h LAYER. `--ih-mechanism`, `--ih-distribution`, `--vshift-base`,
   `--ih-regions`, `--ehcn`: the IhSpec (everything NOT fitted, D-005/D-007).
   `attach_ih` runs once per cell, right after `build_neuron_model`; the
   per-evaluation update and the analytic rest balance (Eq. 11.3) live inside
   ParamSpec.apply, so nothing here has to know their order.

3. PER-AXIS BOXES: `--cm-bounds` ... `--kappa-bounds`, each "lo,hi".

4. THE ROLE LOG. D-006 requires that every long step's role is recorded, not
   inferred: `ls_roles.csv` gives amplitude, polarity, role, reason and
   V_trough per bundle per cell.

5. REPORT-ONLY DIAGNOSTICS. The strongest hyperpolarising step and the rebound
   window are excluded from BOTH training and the gate (PIR conductances are
   not modelled), and are written to `report_residuals.csv` with the model's
   sag fraction, t63 and rebound so that the exclusion is checked rather than
   assumed (plan section 5, last two rows).

What it deliberately DROPS from run_biological_fit.py
-----------------------------------------------------
* The two-pass auto-tau_w loop. `cm_profile_sweep.profile_cm` minimises over
  exactly two nuisance axes -- `loss(lcm, x[0], x[1])` -- so it is 3-D-only and
  would raise on a 6-D loss. Run B used a SINGLE-POINT grid ("5.0"), i.e. the
  sweep selected nothing; so a scalar `--ss-tau-w-ms 5.0` reproduces run B's
  tau_w exactly while removing a 6-D breakage. A comma-separated grid is
  REFUSED here rather than silently ignored. `tau_w_chosen_ms` is still
  written (reason "fixed_by_cli") so the CSV schema stays comparable.
* Phase 2.5. D-006 Q8: not used for this work. `--run-phase2p5` re-enables it
  (3-D arms only -- `run_phase2p5_for_group` fixes R_a and refits (C_m, R_m)
  by name).

Known deviation from the legacy gate, stated rather than hidden
---------------------------------------------------------------
run_biological_fit.py computes the absolute-mV gate with `plst.bundle_rmsd`
on the training side and `mono._rmsd_for_validation_bundle` on the validation
side; the latter windows a long step to its first 150 ms "to minimise Ih
contamination". For an I_h arm that is backwards -- the I_h regime is what the
held-out steps are for. So for `passive_fullstep` and `ih6` BOTH sides of the
gate use `plst.bundle_rmsd` with the arm's own window mode (`gate_valid_via =
"same_window"`), and `baseline_runB` keeps the legacy asymmetric gate
(`"legacy_early"`) because reproducing run B is its whole purpose.

Heavy deps (neuron, the monolith) are imported INSIDE main(), so every helper
above main() imports and is testable without NEURON.
"""

import argparse
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ===========================================================================
#  Progress logging (timestamped + flushed, so `tail -f job.o*` is live)
# ===========================================================================
_T0 = time.time()


def log(msg: str) -> None:
    """Print '[HH:MM:SS | +MM:SS] msg' and flush immediately."""
    now = datetime.now().strftime("%H:%M:%S")
    el = time.time() - _T0
    elapsed = "+{:02d}:{:02d}".format(int(el // 60), int(el % 60))
    print("[{} | {}] {}".format(now, elapsed, msg), flush=True)


# ===========================================================================
#  PURE helpers (NEURON-free)
# ===========================================================================
def parse_float_list(s: str) -> List[float]:
    """'2.0,5.0,10.0' -> [2.0, 5.0, 10.0]. Empty tokens are dropped."""
    return [float(x) for x in str(s).split(",") if x.strip() != ""]


def parse_window(s: str) -> Tuple[float, float]:
    """'0.5,100.0' -> (0.5, 100.0)."""
    a = parse_float_list(s)
    if len(a) != 2:
        raise ValueError("window must be 'lo,hi', got {!r}".format(s))
    return (a[0], a[1])


def parse_name_list(s: str) -> Tuple[str, ...]:
    """'Cm, Rm ,Ra' -> ('Cm', 'Rm', 'Ra'). Order is preserved and meaningful."""
    return tuple(tok.strip() for tok in str(s).split(",") if tok.strip() != "")


def parse_optional_float(s: "Optional[str]") -> Optional[float]:
    """'' / 'none' / None -> None; anything else -> float."""
    if s is None:
        return None
    t = str(s).strip().lower()
    if t in ("", "none", "off", "null"):
        return None
    return float(s)


def select_phase3_subset(specimen_ids: Sequence[int], spec: str) -> List[int]:
    """Resolve which specimen_ids get Phase 3 (grammar unchanged from the
    biological driver: ''/'none' | 'all' | 'first:N' | 'frac:F')."""
    sids = [int(s) for s in specimen_ids]
    spec = (spec or "").strip().lower()
    if spec in ("", "none"):
        return []
    if spec == "all":
        return sids
    if spec.startswith("first:"):
        n = int(spec.split(":", 1)[1])
        return sids[:max(0, n)]
    if spec.startswith("frac:"):
        f = float(spec.split(":", 1)[1])
        k = max(1, int(round(f * len(sids)))) if sids else 0
        return sorted(sids)[:k]
    raise ValueError("unrecognised phase3-subset spec {!r}".format(spec))


def compute_absolute_gate(
    train_bundle_rmsds_mV: Sequence[float],
    valid_bundle_rmsds_mV: Sequence[float],
    *,
    classify_fn: Callable[..., str],
    k_good: float,
    k_fail: float,
    train_fail_mV: float,
    valid_good_mV: float,
) -> Tuple[float, float, float, str]:
    """PURE core of the absolute-mV validation gate (NEURON-free, I/O-free).

    Identical to run_biological_fit.compute_absolute_gate; duplicated rather
    than imported so this entrypoint does not depend on the other one being on
    sys.path. Given per-bundle ABSOLUTE-mV RMSDs already simulated at the fit
    point, return (train_abs_mV, valid_abs_mV, ratio_abs, status).

    Why it exists: under the 'relative' loss, PassiveFitResult.train_rmsd_mV is
    the UNITLESS relative loss, so the monolith's calibrated mV thresholds
    misfire on it. This puts both sides back in mV and re-classifies with the
    SAME thresholds. The fit itself is untouched.
    """
    tr = [float(r) for r in train_bundle_rmsds_mV if np.isfinite(r)]
    va = [float(r) for r in valid_bundle_rmsds_mV if np.isfinite(r)]
    train_abs = float(np.mean(tr)) if tr else float("nan")
    valid_abs = float(np.mean(va)) if va else float("nan")
    if np.isfinite(valid_abs) and np.isfinite(train_abs):
        ratio_abs = valid_abs / max(train_abs, 1e-9)
    else:
        ratio_abs = float("inf")
    status = classify_fn(train_abs, valid_abs, k_good, k_fail, train_fail_mV,
                         valid_rmsd_good_mV=valid_good_mV)
    return train_abs, valid_abs, ratio_abs, status


# ===========================================================================
#  ARMS -- a complete configuration under one name
# ===========================================================================
PASSIVE_NAMES: Tuple[str, ...] = ("Cm", "Rm", "Ra")
IH6_NAMES: Tuple[str, ...] = ("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau")
#: SPEC 2.8 (a), D-206: dv_h is not fitted (it is 0 on the cell).
IH5_NAMES: Tuple[str, ...] = ("Cm", "Rm", "Ra", "gbar", "kappa_tau")

#: The protocol of SPEC 2.8 (b)-(d) (D-207) and the one of D-006 it replaces
#: on the arms ih5 and passive_fullstep: (role_rule, weighting, ls_sample_weight).
PROTOCOL_D207: Tuple[str, str, str] = ("d207", "ls_amplitude", "peak_decay")
PROTOCOL_D006: Tuple[str, str, str] = ("d006", "relative", "none")


@dataclass(frozen=True)
class ArmConfig:
    """Everything an arm fixes. Fields the CLI may override are overridden in
    resolve_arm_config, never mutated afterwards (hence frozen)."""
    name: str
    fit_params: Tuple[str, ...]
    attach_ih: bool                        # insert the mechanism on the cell
    ih_protocol: bool                      # D-006 role split vs legacy split
    ls_window_mode: str                    # after_onset | step | sweep
    ls_window_ms: float                    # used only by mode 'after_onset'
    ls_max_amplitude_pA: Optional[float]   # loader cap; None = admit all
    load_depolarising_ls: bool
    n_long_train: int                      # legacy split only
    ls_deflection_cap_mV: Optional[float]  # legacy split only
    gate_valid_via: str                    # legacy_early | same_window
    role_rule: str                         # d006 | d207 (SPEC 2.8 (b))
    weighting: str                         # relative | ls_amplitude | ... (2.8 (c))
    ls_sample_weight: str                  # none | peak_decay (2.8 (d))
    refine: str                            # none | nelder_mead | trf (2.9 (b), 2.10 (c))


ARMS: Dict[str, ArmConfig] = {
    # Run B, reproduced: legacy loader, legacy split, 60 ms window, 3-D.
    "baseline_runB": ArmConfig(
        name="baseline_runB", fit_params=PASSIVE_NAMES,
        attach_ih=False, ih_protocol=False,
        ls_window_mode="after_onset", ls_window_ms=60.0,
        ls_max_amplitude_pA=100.0, load_depolarising_ls=False,
        n_long_train=2, ls_deflection_cap_mV=12.0,
        gate_valid_via="legacy_early",
        role_rule=PROTOCOL_D006[0], weighting=PROTOCOL_D006[1],
        ls_sample_weight=PROTOCOL_D006[2], refine="none"),
    # Still passive: the control for "was it I_h or the window?". Since SPEC
    # 2.8 (D-207) it runs on the protocol of ih5, so that it stays ih5's control.
    "passive_fullstep": ArmConfig(
        name="passive_fullstep", fit_params=PASSIVE_NAMES,
        attach_ih=False, ih_protocol=True,
        ls_window_mode="step", ls_window_ms=150.0,
        ls_max_amplitude_pA=None, load_depolarising_ls=True,
        n_long_train=0, ls_deflection_cap_mV=None,
        gate_valid_via="same_window",
        role_rule=PROTOCOL_D207[0], weighting=PROTOCOL_D207[1],
        ls_sample_weight=PROTOCOL_D207[2], refine="trf"),
    # The D-006 deliverable, kept as a labelled comparison (SPEC 2.8).
    "ih6": ArmConfig(
        name="ih6", fit_params=IH6_NAMES,
        attach_ih=True, ih_protocol=True,
        ls_window_mode="step", ls_window_ms=150.0,
        ls_max_amplitude_pA=None, load_depolarising_ls=True,
        n_long_train=0, ls_deflection_cap_mV=None,
        gate_valid_via="same_window",
        role_rule=PROTOCOL_D006[0], weighting=PROTOCOL_D006[1],
        ls_sample_weight=PROTOCOL_D006[2], refine="none"),
}
# SPEC 2.8 (a)-(d), D-206 to D-209: ih6 without dv_h, on the D-207 protocol.
ARMS["ih5"] = replace(
    ARMS["ih6"], name="ih5", fit_params=IH5_NAMES,
    role_rule=PROTOCOL_D207[0], weighting=PROTOCOL_D207[1],
    ls_sample_weight=PROTOCOL_D207[2], refine="trf")
# SPEC 2.9 (a), (b), D-234: the six parameters on the D-207 protocol, refined.
ARMS["ih6_d207"] = replace(
    ARMS["ih6"], name="ih6_d207", fit_params=IH6_NAMES,
    role_rule=PROTOCOL_D207[0], weighting=PROTOCOL_D207[1],
    ls_sample_weight=PROTOCOL_D207[2], refine="trf")
# SPEC 2.10 (c), Block 27, D-245: the least-squares refinement `trf` is the
# default of ih5, ih6_d207 and passive_fullstep; `nelder_mead` (2.9 (b)) is a
# labelled comparison by --refine nelder_mead; ih6 and baseline_runB: none.

#: The refinements --refine names (SPEC 2.9 (b), 2.10 (c)).
REFINE_CHOICES: Tuple[str, ...] = ("none", "nelder_mead", "trf")


def resolve_arm_config(args) -> ArmConfig:
    """ArmConfig for `args.arm`, with the CLI overrides applied.

    PURE and NEURON-free, so the arm table is testable on its own. An override
    that would contradict the arm is applied anyway and shows up in the run
    header; the arm name is a starting point, not a lock.
    """
    if args.arm not in ARMS:
        raise ValueError("unknown --arm {!r}; choose from {}"
                         .format(args.arm, sorted(ARMS)))
    cfg = ARMS[args.arm]
    over: Dict[str, object] = {}
    if getattr(args, "fit_params", None):
        over["fit_params"] = parse_name_list(args.fit_params)
        over["attach_ih"] = ("gbar" in over["fit_params"])
    if getattr(args, "ls_window", None):
        over["ls_window_mode"] = args.ls_window
    if getattr(args, "ls_window_ms", None) is not None:
        over["ls_window_ms"] = float(args.ls_window_ms)
    if getattr(args, "ls_max_amplitude_pA", None) is not None:
        over["ls_max_amplitude_pA"] = parse_optional_float(args.ls_max_amplitude_pA)
    if getattr(args, "gate_valid_via", None):
        over["gate_valid_via"] = args.gate_valid_via
    # SPEC 2.8 / Block 15: None = the arm's own protocol.
    if getattr(args, "ls_role_rule", None) is not None:
        over["role_rule"] = str(args.ls_role_rule)
    if getattr(args, "weighting", None) is not None:
        over["weighting"] = str(args.weighting)
    if getattr(args, "ls_sample_weight", None) is not None:
        over["ls_sample_weight"] = str(args.ls_sample_weight)
    # SPEC 2.9 (b) / Block 20: None = the arm's refinement.
    if getattr(args, "refine", None) is not None:
        over["refine"] = str(args.refine)
    return replace(cfg, **over) if over else cfg


# ===========================================================================
#  SPEC 2.8 (a), (f): the time constant in ms, the parameter vector given to
#  the fit, and the conductance at the deepest training trough (PURE)
# ===========================================================================
#: The voltage range of SPEC 2.8 (a) over which tau_h^base peaks (mV), and the
#: grid step of the scan (mV). 0.01 mV: on Ih_human the 0.1 mV default of
#: human_ih_params.peak_mtau_ms misses the maximum by 1.8e-3 ms.
TAU_PEAK_V_RANGE_MV: Tuple[float, float] = (-140.0, -40.0)
TAU_PEAK_DV_MV: float = 0.01


def tau_h_base_peak_ms(ih_spec) -> float:
    """tau_h^{base,pk} of SPEC 2.8 (a), ms: the maximum over v in [-140, -40]
    mV of the mechanism's activation time constant at kappa_tau = 1, for the
    spec's mechanism and base shift (human_ih_params.peak_mtau_shifted_ms).
    Never hard-coded: about 343 ms for Ih_human at base shift 0."""
    import human_ih_params as HIP
    _v, tau = HIP.peak_mtau_shifted_ms(
        str(ih_spec.mechanism), vshift_mV=float(ih_spec.vshift_base_mV),
        tau_scale=1.0, v_lo=TAU_PEAK_V_RANGE_MV[0],
        v_hi=TAU_PEAK_V_RANGE_MV[1], dv=TAU_PEAK_DV_MV)
    return float(tau)


def kappa_bounds_from_args(args) -> Tuple[float, float]:
    """The box of kappa_tau: --kappa-bounds, or, when --tau-h-pk-bounds-ms
    LO,HI is given, (LO / base, HI / base) with base = tau_h_base_peak_ms of
    the run's I_h spec (SPEC 2.8 (a): the same search on the log scale)."""
    tpk = getattr(args, "tau_h_pk_bounds_ms", None)
    if tpk is None:
        return parse_window(args.kappa_bounds)
    lo_ms, hi_ms = parse_window(tpk)
    base = tau_h_base_peak_ms(build_ih_spec(args))
    return (float(lo_ms) / base, float(hi_ms) / base)


def build_spec_from_args(args, cfg):
    """The ParamSpec of the run: the arm's names, the boxes of the CLI."""
    return build_param_spec(
        cfg.fit_params,
        cm_bounds=parse_window(args.cm_bounds),
        rm_bounds=parse_window(args.rm_bounds),
        ra_bounds=parse_window(args.ra_bounds),
        gbar_bounds=parse_window(args.gbar_bounds),
        dvh_bounds=parse_window(args.dvh_bounds),
        kappa_bounds=kappa_bounds_from_args(args))


#: Names whose physical value is a voltage shift (may be zero or negative);
#: every other fitted quantity is positive (param_spec: a log axis).
_SIGNED_NAMES = ("dv_h",)


def parse_eval_theta(s: str, names: Sequence[str]) -> Dict[str, float]:
    """'Cm=0.9,Rm=3e4,...' -> {name: float}. Every name of `names` exactly
    once and nothing else; every value finite, and > 0 unless the name is a
    voltage shift. ValueError otherwise (the CLI turns it into exit 2)."""
    out: Dict[str, float] = {}
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        if tok.count("=") != 1:
            raise ValueError("--eval-theta item {!r} is not name=value".format(tok))
        k, v = (x.strip() for x in tok.split("="))
        if k in out:
            raise ValueError("--eval-theta names {!r} twice".format(k))
        try:
            val = float(v)
        except ValueError:
            raise ValueError("--eval-theta {}={!r} is not a number".format(k, v))
        if not np.isfinite(val):
            raise ValueError("--eval-theta {}={!r} is not finite".format(k, v))
        if k not in _SIGNED_NAMES and not val > 0.0:
            raise ValueError("--eval-theta {}={!r} must be > 0".format(k, v))
        out[k] = val
    want = list(names)
    if sorted(out) != sorted(want):
        raise ValueError(
            "--eval-theta must name each of {} exactly once; got {}"
            .format(",".join(want), ",".join(out) if out else "nothing"))
    return {n: out[n] for n in want}


def v_deepest_train_mV(roles: Optional[dict]) -> float:
    """v* of SPEC 2.8 (f): the minimum v_trough_mV over the long-step records
    with role 'train'; NaN when there is none (or no role record at all)."""
    recs = list((roles or {}).get("records", []) or [])
    vals = [float(r["v_trough_mV"]) for r in recs
            if r.get("role") == "train" and r.get("v_trough_mV") is not None]
    vals = [v for v in vals if np.isfinite(v)]
    return float(min(vals)) if vals else float("nan")


def gh_at_voltage_S_cm2(gbar_S_cm2: float, v_mV: float, ih_spec,
                        dv_h_mV: float = 0.0) -> float:
    """gbar * m_inf(v) (S/cm^2), m_inf from human_ih_params.minf for the
    spec's mechanism and base shift and the given dv_h; NaN for a NaN v."""
    import human_ih_params as HIP
    if not (np.isfinite(v_mV) and np.isfinite(gbar_S_cm2)):
        return float("nan")
    return float(gbar_S_cm2) * float(HIP.minf(
        float(v_mV), str(ih_spec.mechanism),
        vshift_mV=float(ih_spec.vshift_base_mV),
        vshift_minf_mV=float(dv_h_mV)))


# ===========================================================================
#  SPEC 2.9 (b), (e): the refinement's settings and what a fit records (PURE)
# ===========================================================================
#: Columns of SPEC 2.9 (e) / Block 20, written after the Block 15 ones.
REFINE_RECORD_COLUMNS: Tuple[str, ...] = (
    "refine", "loss_after_search", "loss_after_refine", "refine_improved",
    "refine_n_starts", "refine_n_evals", "refine_best_start", "n_workers")

#: Columns of SPEC Block 27, written after the Block 20 ones; then
#: on_bound_<name> for every fitted name (bound_flags).
TRF_RECORD_COLUMNS: Tuple[str, ...] = (
    "l2_after_search", "l2_after_refine", "refine_n_converged",
    "refine_n_failed", "sqrt_l2_at_estimate", "sqrt_l2_at_eval_theta")

#: The bound flag of SPEC Block 27 (the review's CR-3 b): "lo" when the
#: estimate's unit coordinate is below this, "hi" when above 1 minus this.
BOUND_FLAG_UNIT: float = 0.01

#: The fixed head of refine_starts.csv; x0_<name> then x_<name> follow, then
#: REFINE_STARTS_TAIL, x_solver_<name>, active_<name>, sv_1..sv_p (Block 27).
REFINE_STARTS_HEAD: Tuple[str, ...] = (
    "specimen_id", "start", "search_index", "f0", "fun", "nfev", "nit",
    "status")
REFINE_STARTS_TAIL: Tuple[str, ...] = (
    "l2_0", "l2", "converged", "optimality", "n_sims")


def refine_config_from_args(args, cfg):
    """The settings of the arm's refinement with the CLI's values, or None
    when the refinement is 'none':
      nelder_mead  local_refine.RefineConfig(--refine-starts, --refine-maxfev)
                   (SPEC 2.9 (b));
      trf          local_refine.TrfConfig(--refine-starts, --refine-max-nfev)
                   (SPEC 2.10 (c), Block 27).
    The other settings are the SPEC's defaults of each class."""
    if cfg.refine == "none":
        return None
    if cfg.refine not in REFINE_CHOICES:
        raise ValueError("unknown refinement {!r}; choose from {}"
                         .format(cfg.refine, REFINE_CHOICES))
    import local_refine as LR
    if cfg.refine == "nelder_mead":
        return LR.RefineConfig(n_starts=int(args.refine_starts),
                               maxfev_per_start=int(args.refine_maxfev))
    return LR.TrfConfig(n_starts=int(args.refine_starts),
                        max_nfev=int(args.refine_max_nfev))


def _is_trf_result(rr) -> bool:
    """A local_refine.TrfResult (Block 25) rather than a RefineResult."""
    return rr is not None and hasattr(rr, "l2_search")


def refine_record(fr, cfg, n_workers: int) -> Dict[str, object]:
    """The Block 20 columns of one cell's row (SPEC 2.9 (e), Block 27).

    A failed fit (no global-search result) has NaN losses; with the
    refinement off, both losses are the search's best (result.fun), not
    improved, 0 starts, 0 evaluations, best start -1. Under trf:
    loss_after_search = L at the first start (TrfResult.l_search),
    loss_after_refine = L(q_hat), refine_n_evals = n_sims; with no start
    (every global value penalised) both losses are the search's best."""
    nan = float("nan")
    rec: Dict[str, object] = dict(
        refine=str(cfg.refine), loss_after_search=nan, loss_after_refine=nan,
        refine_improved=False, refine_n_starts=0, refine_n_evals=0,
        refine_best_start=-1, n_workers=int(n_workers))
    gp = getattr(fr, "gp_result", None)
    if gp is None:
        return rec
    rr = getattr(gp, "refine", None)
    if rr is None:
        best = float(gp.fun)
        rec.update(loss_after_search=best, loss_after_refine=best)
        return rec
    if _is_trf_result(rr):
        if len(rr.starts):
            l_search, l_hat = float(rr.l_search), float(gp.fun)
        else:
            l_search = l_hat = float(np.min(np.asarray(gp.func_vals,
                                                       dtype=float)))
        rec.update(loss_after_search=l_search, loss_after_refine=l_hat,
                   refine_improved=bool(rr.improved),
                   refine_n_starts=int(len(rr.starts)),
                   refine_n_evals=int(rr.n_sims),
                   refine_best_start=int(rr.best_start))
        return rec
    rec.update(loss_after_search=float(rr.fun_search),
               loss_after_refine=float(rr.fun),
               refine_improved=bool(rr.improved),
               refine_n_starts=int(len(rr.starts)),
               refine_n_evals=int(rr.n_evals),
               refine_best_start=int(rr.best_start))
    return rec


def trf_record(fr, n_failed: int) -> Dict[str, object]:
    """l2_after_search, l2_after_refine (TrfResult.l2_search, .l2; NaN
    without trf), refine_n_converged (starts with `converged`; 0 without
    trf), refine_n_failed (the residual function's n_failed; 0 without trf).
    SPEC Block 27."""
    nan = float("nan")
    rec: Dict[str, object] = dict(l2_after_search=nan, l2_after_refine=nan,
                                  refine_n_converged=0, refine_n_failed=0)
    rr = getattr(getattr(fr, "gp_result", None), "refine", None)
    if not _is_trf_result(rr):
        return rec
    rec.update(l2_after_search=float(rr.l2_search),
               l2_after_refine=float(rr.l2),
               refine_n_converged=int(sum(1 for st in rr.starts
                                          if bool(st["converged"]))),
               refine_n_failed=int(n_failed))
    return rec


def bound_flag(u: float, threshold: float = BOUND_FLAG_UNIT) -> str:
    """'lo' when the unit coordinate u is below `threshold`, 'hi' when above
    1 - threshold, else '' (also for a non-finite u). SPEC Block 27."""
    u = float(u)
    if not np.isfinite(u):
        return ""
    if u < threshold:
        return "lo"
    if u > 1.0 - threshold:
        return "hi"
    return ""


def bound_flags(spec, theta: Optional[dict],
                threshold: float = BOUND_FLAG_UNIT) -> Dict[str, str]:
    """{on_bound_<name>: bound_flag(u_i)} for every name of the spec, u_i the
    unit coordinate (SPEC 2.9 (b)) of the PHYSICAL `theta` in the spec's box
    (q = log for a log axis). '' for a missing or non-finite value."""
    theta = dict(theta or {})
    out: Dict[str, str] = {}
    for ax in spec.axes:
        v = theta.get(ax.name, float("nan"))
        flag = ""
        try:
            v = float(v)
            if np.isfinite(v) and (v > 0.0 or not ax.log):
                lo, hi = ax.q_bounds()
                flag = bound_flag((ax.to_q(v) - lo) / (hi - lo), threshold)
        except (TypeError, ValueError):
            flag = ""
        out["on_bound_" + ax.name] = flag
    return out


def refine_starts_columns(spec) -> List[str]:
    """Header of refine_starts.csv for the spec's fitted names (Blocks 20, 27)."""
    p = len(spec.names)
    return (list(REFINE_STARTS_HEAD) + ["x0_" + n for n in spec.names]
            + ["x_" + n for n in spec.names] + list(REFINE_STARTS_TAIL)
            + ["x_solver_" + n for n in spec.names]
            + ["active_" + n for n in spec.names]
            + ["sv_%d" % (k + 1) for k in range(p)])


def refine_start_rows(fr, spec, specimen_id: int) -> List[dict]:
    """One row per start of this cell's refinement; the points in PHYSICAL
    units (spec.to_physical of the starts' q). Empty without a refinement.

    Under trf: f0 = l_0, fun = l, nit = njev; the tail l2_0, l2, converged,
    optimality, n_sims, x_solver_<name>, active_<name>, sv_k from the start's
    record. Under Nelder-Mead the tail is NaN except converged = (status ==
    0), n_sims = nfev, x_solver_<name> = x_<name>, active_<name> = 0."""
    rr = getattr(getattr(fr, "gp_result", None), "refine", None)
    if rr is None:
        return []
    trf = _is_trf_result(rr)
    p = len(spec.names)
    nan = float("nan")
    rows = []
    for k, st in enumerate(rr.starts):
        if trf:
            row: Dict[str, object] = dict(
                specimen_id=int(specimen_id), start=int(k),
                search_index=int(st["index"]), f0=float(st["l_0"]),
                fun=float(st["l"]), nfev=int(st["nfev"]),
                nit=int(st["njev"]), status=int(st["status"]))
        else:
            row = dict(
                specimen_id=int(specimen_id), start=int(k),
                search_index=int(st["index"]), f0=float(st["f0"]),
                fun=float(st["fun"]), nfev=int(st["nfev"]), nit=int(st["nit"]),
                status=int(st["status"]))
        x0 = spec.to_physical(st["x0"])
        x = spec.to_physical(st["x"])
        for n in spec.names:
            row["x0_" + n] = float(x0[n])
        for n in spec.names:
            row["x_" + n] = float(x[n])
        if trf:
            xs = spec.to_physical(st["x_solver"])
            sv = [float(v) for v in np.asarray(st["sv"], dtype=float)]
            row.update(l2_0=float(st["l2_0"]), l2=float(st["l2"]),
                       converged=bool(st["converged"]),
                       optimality=float(st["optimality"]),
                       n_sims=int(st["n_sims"]))
            for n in spec.names:
                row["x_solver_" + n] = float(xs[n])
            for n, a in zip(spec.names, st["active_mask"]):
                row["active_" + n] = int(a)
            for j in range(p):
                row["sv_%d" % (j + 1)] = sv[j] if j < len(sv) else nan
        else:
            row.update(l2_0=nan, l2=nan, converged=bool(int(st["status"]) == 0),
                       optimality=nan, n_sims=int(st["nfev"]))
            for n in spec.names:
                row["x_solver_" + n] = float(x[n])
            for n in spec.names:
                row["active_" + n] = 0
            for j in range(p):
                row["sv_%d" % (j + 1)] = nan
        rows.append(row)
    return rows


# ===========================================================================
#  The fitted parameter vector and the (not fitted) I_h configuration
# ===========================================================================
def build_param_spec(names: Sequence[str], *, cm_bounds, rm_bounds, ra_bounds,
                     gbar_bounds, dvh_bounds, kappa_bounds):
    """ParamSpec for `names`, validated against the two shapes the pipeline
    supports. Importing param_spec here (not at module scope) keeps the import
    cost off the pure helpers above.

    Accepted, in this exact order:
        ('Cm','Rm','Ra')                                   -> make_passive_spec
        ('Cm','Rm','Ra','gbar'[,'dv_h'][,'kappa_tau'])     -> make_ih_spec

    A kinetic knob is FROZEN at its base value by leaving it out (D-005: a
    parameter that fails the Stage-7 gate is frozen and the failure recorded,
    never silently carried). Any other name set is an error, loudly: a typo
    that silently produced a 3-D fit labelled 'ih6' is the failure mode this
    rejects.
    """
    import param_spec as PS
    nm = tuple(names)
    if nm == PASSIVE_NAMES:
        return PS.make_passive_spec(cm_bounds, rm_bounds, ra_bounds)
    allowed = [PASSIVE_NAMES + ("gbar",),
               PASSIVE_NAMES + ("gbar", "dv_h"),
               PASSIVE_NAMES + ("gbar", "kappa_tau"),
               IH6_NAMES]
    if nm not in [tuple(a) for a in allowed]:
        raise ValueError(
            "--fit-params must be {} or {} (order matters); got {}"
            .format(",".join(PASSIVE_NAMES),
                    " / ".join(",".join(a) for a in allowed), ",".join(nm)))
    return PS.make_ih_spec(
        cm_bounds, rm_bounds, ra_bounds, gbar_bounds,
        dvh_bounds=(dvh_bounds if "dv_h" in nm else None),
        kappa_bounds=(kappa_bounds if "kappa_tau" in nm else None))


def build_ih_spec(args):
    """IhSpec (mechanism, spatial law, regions, base shift, E_h) from the CLI.
    Everything here is CONFIGURED, never fitted (D-005)."""
    import ih_mechanism as IM
    return IM.IhSpec(
        mechanism=args.ih_mechanism,
        distribution=args.ih_distribution,
        regions=parse_name_list(args.ih_regions),
        vshift_base_mV=float(args.vshift_base),
        ehcn_mV=parse_optional_float(args.ehcn),
        mtau_min_ms=float(args.mtau_min_ms))


# ===========================================================================
#  Serialisation
# ===========================================================================
#: Scalar fields carried over from run_biological_fit's writer, so that this
#: CSV and run B's are column-compatible where the quantities mean the same.
_RESULT_FIELDS = [
    "specimen_id", "layer", "dendrite_type", "F", "fit_target",
    "cm_uF_per_cm2", "rm_Ohm_cm2", "ra_Ohm_cm",
    "cm_sigma", "rm_sigma", "ra_sigma",
    "train_rmsd_mV", "valid_rmsd_mV", "valid_to_train_ratio",
    "rin_MOhm_allen", "tau_ms_allen", "v_rest_mV",
    "validation_status", "n_calls", "n_initial", "wall_time_s",
    "noise_sigma_mV", "noise_rho_lag1", "error_message",
    "tau_w_chosen_ms", "tau_w_hw_rho", "tau_w_reason",
    "validation_status_relative", "train_rel_loss",
    "train_rmsd_abs_mV", "valid_rmsd_abs_mV", "valid_to_train_ratio_abs",
    "cm_phase2", "rm_phase2", "ra_phase2", "train_rmsd_phase2",
]

_STR_FIELDS = {
    "layer", "dendrite_type", "fit_target", "validation_status",
    "validation_status_relative", "tau_w_reason", "error_message",
    "arm", "param_names", "ih_mechanism", "ih_distribution",
}

def results_to_dataframe(results: Sequence, *, arm: str = "",
                         ih_label: str = "",
                         morphology_source: str = "") -> pd.DataFrame:
    """One row per fit result: the legacy scalar block, then the arm's OWN
    parameter columns, then the derived I_h quantities.

    `results_to_dataframe` in the biological driver writes a FIXED column
    list, so a six-parameter theta would be dropped on the way to the CSV.
    `param_spec.flatten_result_params` supplies exactly the axes that were
    fitted, named as the spec names them, plus `rail_<name>` -- which for
    gbar is the difference between "the data wanted no I_h" and "the box was
    too narrow" and must never be read off the value alone -- and, last, every
    key of `result.ih_summary` (the derived quantities of plan section 6.5).
    Those are NOT re-emitted here: one writer, one column per quantity.

    A 3-D arm therefore has no I_h columns at all rather than a column of
    NaN. Merging arms across CSVs is pandas' problem, not this writer's.

    `morphology_source` is "archive" (the raw reconstruction.swc) or the
    --swc-dir the arbours were read from (D-013). It is a column, not only a
    log line, because a fit on the diameter-corrected arbours and one on the
    raw ones are otherwise the same CSV with different numbers in it.
    """
    import param_spec as PS
    rows = []
    for r in results:
        row: Dict[str, object] = {}
        for f in _RESULT_FIELDS:
            val = getattr(r, f, None)
            row[f] = ("" if f in _STR_FIELDS else np.nan) if val is None else val
        row["arm"] = arm
        row["ih_label"] = ih_label
        row["morphology_source"] = morphology_source
        row.update(PS.flatten_result_params(r))
        row.update(getattr(r, "fit_record", None) or {})   # Block 15, last
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=_RESULT_FIELDS
                            + ["arm", "ih_label", "morphology_source"])
    cols = list(_RESULT_FIELDS) + ["arm", "ih_label", "morphology_source"]
    for row in rows:                      # stable order, no duplicates
        for k in row:
            if k not in cols:
                cols.append(k)
    return pd.DataFrame(rows, columns=cols)


def roles_to_dataframe(roles: Optional[dict], specimen_id: int) -> pd.DataFrame:
    """D-006's audit trail: which long step played which role, why, and how
    deep it went. Empty frame when the legacy split was used (no roles)."""
    if not roles:
        return pd.DataFrame(columns=["specimen_id", "amplitude_pA", "polarity",
                                     "role", "reason", "v_trough_mV"])
    recs = list(roles.get("records", []))
    for rec in recs:
        rec["specimen_id"] = int(specimen_id)
    return pd.DataFrame(recs, columns=["specimen_id", "amplitude_pA",
                                       "polarity", "role", "reason",
                                       "v_trough_mV"])


# ===========================================================================
#  Orchestration (NEURON-side; lazy heavy imports)
# ===========================================================================
def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    sys.path.insert(0, args.code_dir)

    cfg = resolve_arm_config(args)
    tau_w_ms = _resolve_scalar_tau_w(args.ss_tau_w_ms)

    # --- lazy heavy imports (only when actually running on the cluster) -----
    import passive_long_step_training as plst
    import passive_fitting_hpc_fixed as mono
    import ih_mechanism as IM
    import local_refine as LR
    from neuron import h
    h.load_file("stdrun.hoc")

    def _clear():
        for s in list(h.allsec()):
            h.delete_section(sec=s)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ss_window = parse_window(args.ss_window_ms)
    group_label = Path(args.archive_dir).name

    spec = build_spec_from_args(args, cfg)
    ih_spec = build_ih_spec(args) if cfg.attach_ih else None
    ih_label = ih_spec.label() if ih_spec is not None else ""
    tau_base_ms = (tau_h_base_peak_ms(ih_spec) if ih_spec is not None
                   else float("nan"))
    eval_theta = (parse_eval_theta(args.eval_theta, spec.names)
                  if args.eval_theta is not None else None)
    refine_cfg = refine_config_from_args(args, cfg)

    if cfg.attach_ih and "gbar" not in spec.names:
        raise SystemExit("[FATAL] arm attaches I_h but gbar is not fitted; "
                         "pass --fit-params including gbar, or an arm that "
                         "does not attach the mechanism.")
    if not cfg.attach_ih and "gbar" in spec.names:
        raise SystemExit("[FATAL] gbar is fitted but the mechanism is not "
                         "attached; the loss would be flat in gbar.")
    if args.run_phase2p5 and spec.n != 3:
        raise SystemExit(
            "[FATAL] --run-phase2p5 fixes R_a and refits (C_m, R_m) by name; "
            "it is 3-D-only. Got a {}-D spec ({}).".format(
                spec.n, ",".join(spec.names)))

    log("ARM {} | GROUP {} | theta = ({}) | I_h: {}"
        .format(cfg.name, group_label, ", ".join(spec.names),
                ih_label if ih_label else "not inserted"))
    log("archive={}  swc_dir={}  output={}"
        .format(args.archive_dir, args.swc_dir or "(archive's own)", out))
    log("protocol: ls_window={} ({} ms if after_onset) | loader cap={} pA | "
        "depolarising LS={} | roles: drop {} weakest / {} strongest, "
        "dep_n_validation={}, v_trough_min={}"
        .format(cfg.ls_window_mode, cfg.ls_window_ms,
                cfg.ls_max_amplitude_pA, cfg.load_depolarising_ls,
                args.n_drop_weakest, args.n_drop_strongest,
                args.dep_n_validation, args.v_trough_min))
    log("loss (SPEC 2.8): role rule={} | weighting={} | LS sample weight={} "
        "(tau_d={} ms) | eval-theta: {}"
        .format(cfg.role_rule, cfg.weighting, cfg.ls_sample_weight,
                args.ls_tau_d_ms,
                (", ".join("%s=%.6g" % kv for kv in eval_theta.items())
                 if eval_theta is not None else "none")))
    log("box: {}".format(_box_text(spec, tau_base_ms)))
    log("refinement (SPEC 2.9 (b), 2.10 (c)): {} | workers per loss "
        "evaluation: {}".format(_refine_text(cfg, refine_cfg),
                                int(args.n_workers)))
    sens_theta = (parse_eval_theta(args.sensitivity_theta, spec.names)
                  if args.sensitivity_theta is not None else None)
    if sens_theta is not None:
        log("SENSITIVITY MODE (SPEC Block 27, 2.10 (d)): no fit; theta = {} | "
            "h={} check_h={} -> {}".format(
                ", ".join("%s=%.6g" % kv for kv in sens_theta.items()),
                args.sensitivity_h, args.sensitivity_check_h,
                args.sensitivity_out))
    log("dt (ENFORCED): brief={} ms  long={} ms | tau_w={} ms (fixed) | "
        "budget {}/{}"
        .format(args.dt_brief_ms, args.dt_long_ms, tau_w_ms,
                args.n_calls, args.n_initial))

    # ---- 1. PATCH the split + the loss (once; tau_w is fixed) --------------
    log("[1/5] PATCH -- integrate_long_step (spec={}, ih_protocol={}) ..."
        .format(spec.label, cfg.ih_protocol))
    plst.integrate_long_step(
        mono, spec=spec, ih_protocol=cfg.ih_protocol,
        ls_window_mode=cfg.ls_window_mode,
        n_drop_weakest=args.n_drop_weakest,
        n_drop_strongest=args.n_drop_strongest,
        dep_n_validation=(None if args.dep_n_validation < 0
                          else int(args.dep_n_validation)),
        v_trough_min_mV=parse_optional_float(args.v_trough_min),
        train_all_hyp=bool(args.train_all_hyp),
        n_long_train=cfg.n_long_train,
        max_ls_train_deflection_mV=cfg.ls_deflection_cap_mV,
        r_in_target=args.r_in_target, weighting=cfg.weighting,
        role_rule=cfg.role_rule, ls_sample_weight=cfg.ls_sample_weight,
        ls_tau_d_ms=float(args.ls_tau_d_ms),
        ss_window_ms=ss_window, ss_time_weight=args.ss_time_weight,
        ss_tau_w_ms=tau_w_ms,
        ss_t0_ms=parse_optional_float(args.ss_t0_ms),
        ls_window_ms_after_onset=float(cfg.ls_window_ms),
        dt_brief_ms=float(args.dt_brief_ms),
        dt_long_ms=float(args.dt_long_ms),
        n_workers=int(args.n_workers),
        verbose=True)
    log("[1/5] PATCH -- done.")

    # ---- 2. LOAD this group's archives ------------------------------------
    log("[2/5] LOAD -- reading specimen_* archives + optimiser inputs ...")
    cells_data = load_group_cells(mono, args, cfg, verbose=True)
    if not cells_data:
        log("[ABORT] No cells loaded from archive: {}".format(args.archive_dir))
        sys.exit(1)
    opt_inputs = [mono.prepare_optimiser_inputs(cd, fit_target=args.fit_target)
                  for cd in cells_data]
    log("[2/5] LOAD -- {} cell(s) loaded.".format(len(cells_data)))

    role_frames = [roles_to_dataframe(getattr(oi, "ls_roles", None),
                                      int(cd.specimen_id))
                   for cd, oi in zip(cells_data, opt_inputs)]
    role_frames = [f for f in role_frames if len(f)]
    if role_frames:
        pd.concat(role_frames, ignore_index=True).to_csv(
            out / "ls_roles.csv", index=False)
        log("[2/5] LOAD -- long-step roles -> ls_roles.csv")

    if sens_theta is not None:
        n_ok = _run_sensitivity_mode(mono, plst, IM, cells_data, opt_inputs,
                                     spec, cfg, args, ih_spec, sens_theta,
                                     tau_w_ms, _clear)
        if n_ok == 0:
            log("[ABORT] Sensitivity mode: no cell done.")
            sys.exit(1)
        log("DONE (sensitivity mode) arm {} group {} -- {} cell(s), total "
            "{:.0f}s".format(cfg.name, group_label, n_ok, time.time() - _T0))
        return

    # ---- 3. FIT, one cell at a time ---------------------------------------
    log("[3/5] FIT -- {} cell(s), sequential ...".format(len(cells_data)))
    results: List[object] = []
    diag_rows: List[dict] = []
    start_rows: List[dict] = []
    failed_ids: List[int] = []
    for i, (cd, oi) in enumerate(zip(cells_data, opt_inputs)):
        sid = int(cd.specimen_id)
        t_cell = time.time()
        log("  cell {}/{} (specimen {}): START"
            .format(i + 1, len(cells_data), sid))
        _clear()
        cell = None
        try:
            cell = mono.build_neuron_model(cd.swc_path, F=args.F)
            if ih_spec is not None:
                factors = IM.attach_ih(cell, ih_spec)
                log("    I_h attached: {} segment(s), law {!r}, E_h={:.2f} mV"
                    .format(len(factors), ih_spec.distribution,
                            float(ih_spec.ehcn_mV)))
            # Hard gate: both loss builders wrap their body in
            # "except Exception -> 1e6", so a DtEnforcementError raised inside
            # a loss evaluation would become a finite penalty and the fit would
            # proceed on wrongly-integrated simulations. Verify dt here, where
            # nothing can swallow the failure.
            mono.assert_dt_enforced(
                cell, dt_values_ms=(args.dt_brief_ms, args.dt_long_ms),
                v_init_mV=float(oi.v_rest_mV), verbose=(i == 0))
            _assert_loss_live(mono, cell, oi, spec)

            log("  cell {}/{} ({}): FIT ({}-D, n_calls={}) ..."
                .format(i + 1, len(cells_data), sid, spec.n, args.n_calls))
            t_fit = time.time()
            # SPEC 2.9 (b), 2.10 (c): the refinement wraps the global search
            # of the Phase-2 fit only; the originals are back before anything
            # else (Phase 2.5, Phase 3) can call them. Under trf the loss
            # fit_one_cell builds is captured by a shim around
            # mono._build_loss_function, and its make_residuals() (built
            # once, after the search) is the refinement's residual function.
            orig_gp_minimize = mono.gp_minimize
            orig_build_loss = mono._build_loss_function
            captured: Dict[str, object] = {"loss": None, "resid": None}
            if isinstance(refine_cfg, LR.TrfConfig):
                mono._build_loss_function = _capturing_builder(
                    orig_build_loss, captured)
                mono.gp_minimize = LR.wrap_minimizer_trf(
                    orig_gp_minimize, _residuals_getter(captured), refine_cfg)
            elif refine_cfg is not None:
                mono.gp_minimize = LR.wrap_minimizer(orig_gp_minimize,
                                                     refine_cfg)
            try:
                fr = mono.fit_one_cell(cell, cd, oi, spec=spec, F=args.F,
                                       n_calls=args.n_calls,
                                       n_initial=args.n_initial, seed=i)
            finally:
                mono.gp_minimize = orig_gp_minimize
                mono._build_loss_function = orig_build_loss
                if captured["resid"] is not None:
                    captured["resid"].close()
            fr.tau_w_chosen_ms = float(tau_w_ms)
            fr.tau_w_hw_rho = float("nan")
            fr.tau_w_reason = "fixed_by_cli"
            # SPEC 2.8 (f): the training loss at the estimate and at the
            # given theta, through the builder the fit minimised. Evaluating
            # it moves the cell, so the estimate is re-applied just below.
            fr.fit_record = _fit_record(mono, cell, oi, fr, spec, cfg, args,
                                        ih_spec, tau_base_ms, eval_theta)
            fr.fit_record.update(refine_record(fr, cfg, int(args.n_workers)))
            # SPEC Block 27: L_2, convergence and the bound flags, after the
            # Block 20 columns; also moves the cell (re-applied below).
            fr.fit_record.update(trf_record(
                fr, (int(captured["resid"].n_failed)
                     if captured["resid"] is not None else 0)))
            fr.fit_record.update(_l2_record(mono, cell, oi, fr, spec, cfg,
                                            eval_theta))
            fr.fit_record.update(bound_flags(spec, getattr(fr, "params", None)))
            start_rows.extend(refine_start_rows(fr, spec, sid))

            # theta is ON the cell already (fit_one_cell's last apply), but be
            # explicit: every post-fit diagnostic below assumes the fit point.
            _apply_theta(spec, cell, fr, float(fr.v_rest_mV))
            if ih_spec is not None:
                fr.ih_summary = IM.ih_rest_summary(
                    cell, float(fr.v_rest_mV), float(fr.params["Rm"]))
                drift = IM.rest_drift(cell, float(fr.v_rest_mV),
                                      t_ms=args.rest_drift_ms,
                                      dt_ms=float(args.dt_long_ms))
                fr.ih_summary["rest_drift_mV"] = float(drift)
                log("    I_h at rest: m_inf={:.4f}  tau_h={:.0f} ms  "
                    "gh/g_pas={:.3f}  e_pas(soma)={:.2f} mV  drift={:.2e} mV"
                    .format(fr.ih_summary["m_inf_at_rest"],
                            fr.ih_summary["tau_h_at_rest_ms"],
                            fr.ih_summary["gh_rest_over_gpas"],
                            fr.ih_summary["e_pas_soma_mV"], drift))

            _apply_absolute_gate(mono, plst, cell, oi, fr, cfg, spec, args)
            diag_rows.extend(_long_step_diagnostics(
                mono, plst, IM, cell, oi, fr, cfg, args))

            results.append(fr)
            log("    search best {:.6g} -> refined {:.6g} (improved {}; {} "
                "start(s), {} local evaluation(s), best start {})"
                .format(fr.fit_record["loss_after_search"],
                        fr.fit_record["loss_after_refine"],
                        fr.fit_record["refine_improved"],
                        fr.fit_record["refine_n_starts"],
                        fr.fit_record["refine_n_evals"],
                        fr.fit_record["refine_best_start"]))
            log("    L_2: search {:.6g} -> refined {:.6g}; sqrt(L_2) at "
                "estimate {:.6g}, at eval-theta {:.6g}; {} converged start(s), "
                "{} failed point(s); bound flags: {}"
                .format(fr.fit_record["l2_after_search"],
                        fr.fit_record["l2_after_refine"],
                        fr.fit_record["sqrt_l2_at_estimate"],
                        fr.fit_record["sqrt_l2_at_eval_theta"],
                        fr.fit_record["refine_n_converged"],
                        fr.fit_record["refine_n_failed"],
                        ", ".join("%s=%s" % (n, fr.fit_record["on_bound_" + n]
                                             or "-") for n in spec.names)))
            log("    tau_h_pk={:.4g} ms (base {:.4g} ms) | loss at estimate="
                "{:.6g}, at eval-theta={:.6g} | v*={:.3f} mV, g_h(v*)={:.4g} "
                "S/cm^2".format(fr.fit_record["tau_h_pk_ms"], tau_base_ms,
                                fr.fit_record["loss_at_estimate"],
                                fr.fit_record["loss_at_eval_theta"],
                                fr.fit_record["v_deepest_train_mV"],
                                fr.fit_record["gh_at_v_deepest_S_cm2"]))
            log("  cell {}/{} ({}): FIT done in {:.0f}s -> {} | status={} "
                "(valid_abs={:.3f} mV) | cell total {:.0f}s"
                .format(i + 1, len(cells_data), sid, time.time() - t_fit,
                        ", ".join("%s=%.4g" % (n, fr.params[n])
                                  for n in spec.names),
                        fr.validation_status,
                        float(getattr(fr, "valid_rmsd_abs_mV", np.nan)),
                        time.time() - t_cell))
        except Exception as exc:  # noqa: BLE001
            log("  cell {}/{} ({}): FAILED after {:.0f}s -> {}: {}"
                .format(i + 1, len(cells_data), sid, time.time() - t_cell,
                        type(exc).__name__, exc))
            failed_ids.append(sid)
            if args.fail_fast:
                raise
        finally:
            if cell is not None:
                try:
                    cell.destroy()
                except Exception:
                    pass
    _clear()

    if failed_ids:
        (out / "failed_cells.txt").write_text(
            "\n".join(str(s) for s in failed_ids) + "\n")
        log("[3/5] FIT -- {} cell(s) FAILED (see failed_cells.txt): {}"
            .format(len(failed_ids), failed_ids))

    results_to_dataframe(
        results, arm=cfg.name, ih_label=ih_label,
        morphology_source=(str(args.swc_dir) if args.swc_dir else "archive"),
    ).to_csv(out / "phase2_results.csv", index=False)
    pd.DataFrame(start_rows, columns=refine_starts_columns(spec)).to_csv(
        out / "refine_starts.csv", index=False)
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(out / "ls_diagnostics.csv", index=False)
        log("[3/5] FIT -- per-step residuals + sag -> ls_diagnostics.csv")
    log("[3/5] FIT -- done: {} fit(s) -> phase2_results.csv".format(len(results)))

    if not results:
        log("[ABORT] No successful fits; skipping Phase 2.5 / Phase 3.")
        sys.exit(1)

    # ---- 4. PHASE 2.5 -- OFF by default (D-006 Q8) ------------------------
    phase2p5_ran = bool(args.run_phase2p5)
    if phase2p5_ran:
        log("[4/5] PHASE 2.5 -- profile Ra + fix at cohort median + refit ...")
        t_p25 = time.time()
        mono.run_phase2p5_for_group(
            results=results, cells_data=cells_data, opt_inputs=opt_inputs,
            F=args.F, group_label=group_label, output_dir=out,
            n_floor=args.n_floor, n_ra_profile=args.n_ra_profile,
            n_calls=args.n_calls, n_initial=args.n_initial,
            acq_func=mono.DEFAULT_ACQ_FUNC, make_plots=True, seed=0,
            verbose=True)
        results_to_dataframe(results, arm=cfg.name, ih_label=ih_label).to_csv(
            out / "phase2p5_combined_results.csv", index=False)
        log("[4/5] PHASE 2.5 -- done in {:.0f}s".format(time.time() - t_p25))
    else:
        log("[4/5] PHASE 2.5 -- OFF (D-006 Q8). R_a stays free; Phase 3 is "
            "{}-D.".format(spec.n))

    # ---- 5. PHASE 3 (bootstrap CIs on a subset) ---------------------------
    fittable = [int(r.specimen_id) for r in results
                if r.validation_status in ("good", "to_refine")
                and getattr(r, "gp_result", None) is not None]
    subset = select_phase3_subset(fittable, args.phase3_subset)
    if subset:
        log("[5/5] PHASE 3 -- bootstrap subset ({}/{} fittable): {} ..."
            .format(len(subset), len(fittable), subset))
        t_p3 = time.time()
        _run_phase3_subset(mono, IM, results, cells_data, subset,
                           phase2p5_ran, ih_spec, args, out)
        log("[5/5] PHASE 3 -- done in {:.0f}s".format(time.time() - t_p3))
    else:
        log("[5/5] PHASE 3 -- empty subset (spec={!r}; {} fittable) -> skipped"
            .format(args.phase3_subset, len(fittable)))

    log("DONE arm {} group {} -- total {:.0f}s"
        .format(cfg.name, group_label, time.time() - _T0))


# ---------------------------------------------------------------------------
#  Small NEURON-side helpers
# ---------------------------------------------------------------------------
def _resolve_scalar_tau_w(s: str) -> float:
    """One tau_w, or a loud refusal.

    The two-pass auto-tau_w loop is not run here (see the module docstring):
    its profiler minimises over exactly two nuisance axes and is 3-D-only. A
    comma-separated grid is therefore rejected rather than quietly collapsed
    to its first entry, which would report a tau_w that was never selected.
    """
    vals = parse_float_list(s)
    if len(vals) != 1:
        raise SystemExit(
            "[FATAL] --ss-tau-w-ms takes ONE value here (got {!r}). The "
            "per-cell tau_w sweep needs cm_profile_sweep.profile_cm, which is "
            "3-D-only; run B used a single-point grid, so a scalar reproduces "
            "it exactly. Use run_biological_fit.py for a real sweep."
            .format(s))
    return float(vals[0])


def _apply_theta(spec, cell, fit_result, v_rest_mV: float) -> None:
    """Put the fitted theta back on the cell, through the SPEC -- so the I_h
    knobs and the analytic rest balance (Eq. 11.3) are applied in the one
    correct order, rather than re-implemented here."""
    theta = dict(getattr(fit_result, "params", {}) or {})
    if not theta:                                   # 3-D legacy result
        theta = {"Cm": float(fit_result.cm_uF_per_cm2),
                 "Rm": float(fit_result.rm_Ohm_cm2),
                 "Ra": float(fit_result.ra_Ohm_cm)}
    spec.apply(cell, theta, float(v_rest_mV))


def _box_text(spec, tau_base_ms: float) -> str:
    """Each axis' box; kappa_tau also as tau_h_pk in ms (SPEC 2.8 (a))."""
    parts = []
    for ax in spec.axes:
        txt = "{}[{:g}, {:g}]{}".format(ax.name, ax.lo, ax.hi,
                                        "" if ax.log else " lin")
        if ax.name == "kappa_tau" and np.isfinite(tau_base_ms):
            txt += " = tau_h_pk [{:.4g}, {:.4g}] ms (base {:.4g} ms)".format(
                ax.lo * tau_base_ms, ax.hi * tau_base_ms, tau_base_ms)
        parts.append(txt)
    return "; ".join(parts)


def _loss_at(mono, cell, oi, spec, theta) -> float:
    """The training loss L at a PHYSICAL theta, through the (patched) builder
    fit_one_cell minimises, on the same bundles, rest and window."""
    L = mono._build_loss_function(
        cell, oi.train_bundles, float(oi.v_rest_mV),
        tuple(oi.train_window_ms), spec=spec)
    try:
        return float(L(*spec.to_q(theta)))
    finally:
        L.close()                     # Block 19: its worker pool, if any


#: Columns of SPEC 2.8 (f) / Block 15, written after the existing ones.
FIT_RECORD_COLUMNS: Tuple[str, ...] = (
    "tau_h_pk_ms", "tau_h_base_pk_ms", "ls_role_rule", "weighting",
    "ls_sample_weight", "ls_tau_d_ms", "loss_at_estimate",
    "loss_at_eval_theta", "v_deepest_train_mV", "gh_at_v_deepest_S_cm2")


def _fit_record(mono, cell, oi, fr, spec, cfg, args, ih_spec, tau_base_ms,
                eval_theta) -> Dict[str, object]:
    """The Block 15 columns of one cell's row (SPEC 2.8 (a), (f)).

    Leaves the cell at whatever theta was evaluated last: the caller
    re-applies the estimate."""
    theta = {n: float(v) for n, v in
             (getattr(fr, "params", None) or {}).items()}
    est_ok = (all(n in theta for n in spec.names)
              and all(np.isfinite(theta[n]) for n in spec.names))
    nan = float("nan")
    rec: Dict[str, object] = dict(
        tau_h_pk_ms=nan, tau_h_base_pk_ms=float(tau_base_ms),
        ls_role_rule=cfg.role_rule, weighting=cfg.weighting,
        ls_sample_weight=cfg.ls_sample_weight,
        ls_tau_d_ms=float(args.ls_tau_d_ms),
        loss_at_estimate=nan, loss_at_eval_theta=nan,
        v_deepest_train_mV=v_deepest_train_mV(getattr(oi, "ls_roles", None)),
        gh_at_v_deepest_S_cm2=nan)
    if ih_spec is not None and est_ok:
        # kappa_tau frozen (not fitted) is 1, as param_spec._apply_ih applies it
        rec["tau_h_pk_ms"] = float(theta.get("kappa_tau", 1.0)) * float(tau_base_ms)
        rec["gh_at_v_deepest_S_cm2"] = gh_at_voltage_S_cm2(
            theta["gbar"], float(rec["v_deepest_train_mV"]), ih_spec,
            dv_h_mV=float(theta.get("dv_h", 0.0)))
    if est_ok:
        rec["loss_at_estimate"] = _loss_at(mono, cell, oi, spec, theta)
    if eval_theta is not None:
        rec["loss_at_eval_theta"] = _loss_at(mono, cell, oi, spec, eval_theta)
    return rec


# ---------------------------------------------------------------------------
#  SPEC Block 27: the trf refinement's plumbing and the L_2 record
# ---------------------------------------------------------------------------
def _refine_text(cfg, refine_cfg) -> str:
    """The run header's description of the refinement."""
    if refine_cfg is None:
        return "none"
    if cfg.refine == "trf":
        return ("trf (scipy least_squares): {} start(s), max_nfev {} each, "
                "d_min {} (unit), forward FD step {} (unit), xtol {}, ftol "
                "{}, gtol {}".format(
                    refine_cfg.n_starts, refine_cfg.max_nfev,
                    refine_cfg.min_start_dist_unit, refine_cfg.fd_step_unit,
                    refine_cfg.xtol, refine_cfg.ftol, refine_cfg.gtol))
    return ("nelder_mead: {} start(s), maxfev {} each, d_min {} (unit), step "
            "{}, xatol {}, fatol {} x L(start)".format(
                refine_cfg.n_starts, refine_cfg.maxfev_per_start,
                refine_cfg.min_start_dist_unit, refine_cfg.simplex_step_unit,
                refine_cfg.xatol_unit, refine_cfg.fatol_rel))


def _capturing_builder(orig_build, captured: dict):
    """A shim around mono._build_loss_function that records the loss it
    returns in captured['loss'] (the last one built) and returns it
    unchanged. SPEC Block 27."""
    def _build(*a, **k):
        loss = orig_build(*a, **k)
        captured["loss"] = loss
        return loss
    _build.wrapped_builder = orig_build
    return _build


def _residuals_getter(captured: dict):
    """get_residuals of local_refine.wrap_minimizer_trf: the captured loss's
    make_residuals(), built once and kept in captured['resid']."""
    def _get():
        if captured["resid"] is None:
            loss = captured["loss"]
            if loss is None or not hasattr(loss, "make_residuals"):
                raise RuntimeError(
                    "trf refinement: no loss with make_residuals() was built "
                    "by mono._build_loss_function during fit_one_cell")
            captured["resid"] = loss.make_residuals()
        return captured["resid"]
    return _get


#: Weightings for which the residual vector of SPEC 2.10 (a) is defined.
_L2_WEIGHTINGS = ("relative", "ls_amplitude")


def _sqrt_l2_at(mono, cell, oi, spec, theta) -> float:
    """sqrt(L_2) of SPEC 2.10 (a) at a PHYSICAL theta, through
    loss.make_residuals() of a loss built as _loss_at builds it."""
    L = mono._build_loss_function(
        cell, oi.train_bundles, float(oi.v_rest_mV),
        tuple(oi.train_window_ms), spec=spec)
    R = None
    try:
        R = L.make_residuals()
        row = R(np.asarray([spec.to_q(theta)], dtype=np.float64))[0]
        return float(np.sqrt(R.l_and_l2(row)[1]))
    finally:
        if R is not None:
            R.close()
        L.close()


def _l2_record(mono, cell, oi, fr, spec, cfg, eval_theta) -> Dict[str, float]:
    """sqrt_l2_at_estimate and sqrt_l2_at_eval_theta (SPEC Block 27): NaN
    when the arm's weighting has no residual vector, when the estimate is
    not finite, or (the second) without --eval-theta. Moves the cell."""
    nan = float("nan")
    rec = dict(sqrt_l2_at_estimate=nan, sqrt_l2_at_eval_theta=nan)
    if cfg.weighting not in _L2_WEIGHTINGS:
        return rec
    theta = {n: float(v) for n, v in
             (getattr(fr, "params", None) or {}).items()}
    try:
        if (all(n in theta for n in spec.names)
                and all(np.isfinite(theta[n]) for n in spec.names)):
            rec["sqrt_l2_at_estimate"] = _sqrt_l2_at(mono, cell, oi, spec,
                                                     theta)
        if eval_theta is not None:
            rec["sqrt_l2_at_eval_theta"] = _sqrt_l2_at(mono, cell, oi, spec,
                                                       eval_theta)
    except ValueError as exc:
        # residual_layout refuses a training bundle the scalar loss still
        # evaluates (no pre-window sample, < 2 window samples, D_b not > 0):
        # the diagnostic columns stay NaN rather than failing the cell's fit.
        log("    WARNING: sqrt(L_2) not computed (the residual layout "
            "refused this cell's training bundles): {}".format(exc))
    return rec


def load_group_cells(mono, args, cfg, *, verbose: bool = True):
    """mono.load_cells_from_archive of the run's group with the arm's loader
    settings (the call main makes)."""
    return mono.load_cells_from_archive(
        args.archive_dir, n_avg_groups=args.n_avg_groups,
        ls_max_amplitude_pA=cfg.ls_max_amplitude_pA,
        ls_fallback_amplitude_pA=args.ls_fallback_amplitude_pA,
        load_depolarising_ls=cfg.load_depolarising_ls,
        ls_dep_max_amplitude_pA=parse_optional_float(args.ls_dep_max_amplitude_pA),
        swc_dir=args.swc_dir, specimen_ids=None,
        max_cells=args.max_cells, verbose=verbose)


def _q_box_centre(spec) -> List[float]:
    """Midpoint of every axis IN q-SPACE (log for a log axis, linear for a
    linear one). Used only as a generic off-optimum probe point."""
    out = []
    for ax in spec.axes:
        lo, hi = ax.q_bounds()
        out.append(0.5 * (float(lo) + float(hi)))
    return out


#: What both loss builders return when a simulation raises. It is FINITE, so
#: `isfinite` never proves a loss is live -- the value has to be BELOW this.
_LOSS_CRASH_PENALTY = 1e6


def _assert_loss_live(mono, cell, oi, spec) -> None:
    """The loss must VARY with C_m and stay BELOW the crash penalty.

    Both loss builders wrap their body in `except Exception -> 1e6`, so a
    dead cell, a failed simulate() or an orphaned IClamp does not raise: it
    returns a large FINITE number. The optimiser would then wander a flat
    plateau and report a fitted theta that no simulation ever produced. A
    finiteness test passes on 1e6 and would miss exactly this, so both the
    penalty bound and the variation are asserted.

    This probes the PATCHED loss -- the one fit_one_cell will actually
    minimise -- rather than rebuilding a reference, so it also catches a spec
    that never reached the builder.
    """
    L = mono._build_loss_function(
        cell, oi.train_bundles, float(oi.v_rest_mV),
        tuple(oi.train_window_ms), spec=spec)
    q0 = _q_box_centre(spec)
    icm = spec.index("Cm")
    vals = []
    try:
        for cm in (0.5, 1.0, 2.0):
            q = list(q0)
            q[icm] = float(np.log(cm))
            vals.append(float(L(*q)))
    finally:
        L.close()                     # Block 19: its worker pool, if any
    if not np.all(np.isfinite(vals)):
        raise RuntimeError(
            "loss NOT FINITE across C_m {} at the q-box centre."
            .format([round(v, 4) for v in vals]))
    if max(vals) >= _LOSS_CRASH_PENALTY:
        raise RuntimeError(
            "loss hit the {:.0e} crash penalty at C_m in (0.5, 1, 2) -> a "
            "simulation raised and was swallowed. Values {}. Most often: the "
            "mechanism is not compiled (no x86_64/ in the process working "
            "directory), or this cell's sections were deleted by a later "
            "build_neuron_model()."
            .format(_LOSS_CRASH_PENALTY, [round(v, 4) for v in vals]))
    if max(vals) - min(vals) < 1e-9:
        raise RuntimeError(
            "loss constant across C_m {} at the q-box centre -> dead cell or "
            "a mis-wired spec; the fit would be garbage."
            .format([round(v, 4) for v in vals]))


def _pre_baseline_mV(t_s, v_mV, onset_s: float, guard_s: float = 5e-3) -> float:
    """Mean voltage of a trace's own pre-step window [0, onset - guard].
    Each trace is referenced to ITS OWN baseline so that a DC offset between
    the recording and the model does not enter the sag fraction."""
    t = np.asarray(t_s, dtype=float)
    v = np.asarray(v_mV, dtype=float)
    m = (t >= 0.0) & (t <= max(float(onset_s) - guard_s, 0.0))
    return float(np.mean(v[m])) if m.any() else float(v[0])


# ---------------------------------------------------------------------------
#  Absolute-mV validation gate
# ---------------------------------------------------------------------------
def _apply_absolute_gate(mono, plst, cell, oi, fr, cfg, spec, args) -> None:
    """Recompute train/validation RMSD in ABSOLUTE mV at the fit point and
    re-classify with the monolith's calibrated mV thresholds, then MUTATE fr.

    `cfg.gate_valid_via` selects the validation side:
      'legacy_early'  mono._rmsd_for_validation_bundle -- a long step windowed
                      to its first DEFAULT_VALID_WINDOW_MS_AFTER_ONSET ms "to
                      minimise Ih contamination". Correct for run B; wrong for
                      an I_h arm, where that contamination is the signal.
      'same_window'   plst.bundle_rmsd with the arm's own window mode, i.e.
                      exactly what the training side uses. Both sides are then
                      unweighted absolute mV over the same convention.

    The caller has already put theta on the cell through the spec; this does
    NOT re-apply it (re-applying with set_passive alone would silently drop
    the I_h knobs and the rest balance).
    """
    fr.validation_status_relative = getattr(fr, "validation_status", "")
    fr.train_rel_loss = float(getattr(fr, "train_rmsd_mV", np.nan))

    if not np.isfinite(getattr(fr, "cm_uF_per_cm2", np.nan)):
        fr.train_rmsd_abs_mV = np.nan
        fr.valid_rmsd_abs_mV = np.nan
        fr.valid_to_train_ratio_abs = float("inf")
        return

    rmsd_kw = dict(ss_window_ms=tuple(oi.train_window_ms),
                   ls_window_ms_after_onset=float(cfg.ls_window_ms),
                   ls_window_mode=cfg.ls_window_mode,
                   r_in_target=args.r_in_target, ss_sample_weight_fn=None)

    train_rmsds = []
    for b in oi.train_bundles:
        rmsd, _defl = plst.bundle_rmsd(cell, b, float(fr.v_rest_mV), **rmsd_kw)
        train_rmsds.append(float(rmsd))

    valid_rmsds = []
    for vb in oi.validation_bundles:
        try:
            if cfg.gate_valid_via == "legacy_early":
                r = mono._rmsd_for_validation_bundle(
                    cell, vb, float(fr.v_rest_mV),
                    valid_window_ms_after_onset=(
                        mono.DEFAULT_VALID_WINDOW_MS_AFTER_ONSET))
            else:
                r, _d = plst.bundle_rmsd(cell, vb, float(fr.v_rest_mV),
                                         **rmsd_kw)
            valid_rmsds.append(float(r))
        except Exception:
            pass

    train_abs, valid_abs, ratio_abs, status_abs = compute_absolute_gate(
        train_rmsds, valid_rmsds,
        classify_fn=mono._classify_fit,
        k_good=mono.DEFAULT_K_GOOD, k_fail=mono.DEFAULT_K_FAIL,
        train_fail_mV=mono.DEFAULT_TRAIN_RMSD_FAIL_MV,
        valid_good_mV=mono.DEFAULT_VALID_RMSD_GOOD_MV)

    fr.train_rmsd_abs_mV = train_abs
    fr.valid_rmsd_abs_mV = valid_abs
    fr.valid_to_train_ratio_abs = ratio_abs
    fr.validation_status = status_abs


# ---------------------------------------------------------------------------
#  Per-long-step diagnostics (reported, never minimised -- plan sections 5, 7)
# ---------------------------------------------------------------------------
def _long_step_diagnostics(mono, plst, IM, cell, oi, fr, cfg, args
                           ) -> List[dict]:
    """One row per LONG bundle of this cell, whatever its role.

    Carries (a) the RMSD over the arm's own window, (b) the three sub-window
    RMSDs of plan section 7 -- charging, sag, rebound -- which are written and
    never minimised, and (c) the sag fraction / t63 / rebound of the DATA and
    of the MODEL side by side. This is what makes the two exclusions of plan
    section 5 checkable: the strongest step and the post-offset rebound are
    kept out of training and out of the gate, so their residual has to be
    looked at, not assumed small.

    The cell must already be at the fit point (the caller applies theta
    through the spec).
    """
    by_role = [("train", list(getattr(oi, "train_bundles", []) or [])),
               ("validate", list(getattr(oi, "validation_bundles", []) or [])),
               ("report", list(getattr(oi, "report_bundles", []) or []))]

    rows: List[dict] = []
    for role, bundles in by_role:
        for b in bundles:
            if mono._is_brief_pulse(b):
                continue
            rows.append(_one_long_step_row(mono, plst, IM, cell, oi, fr, cfg,
                                           args, b, role))
    return rows


def _one_long_step_row(mono, plst, IM, cell, oi, fr, cfg, args, b, role) -> dict:
    """One row of `ls_diagnostics.csv` for ONE long step.

    Carries (a) the RMSD over the arm's own window, (b) the three sub-window
    RMSDs of plan section 7 -- charging, sag, rebound -- and (c) the sag
    fraction / t63 / rebound of the DATA and of the MODEL side by side. Each
    trace is referenced to ITS OWN pre-step baseline, so a DC offset between
    recording and model does not enter the sag fraction.

    The cell must already be at the fit point; the caller applies theta
    through the spec.
    """
    v_rest = float(fr.v_rest_mV)
    base_row = dict(specimen_id=int(fr.specimen_id), role=role,
                    amplitude_pA=float(b.amplitude_pA))
    try:
        t_sim_s, v_sim = mono._simulate_long_square(cell, b, v_rest)
    except Exception as exc:  # noqa: BLE001
        base_row["error"] = "{}: {}".format(type(exc).__name__, exc)
        return base_row

    rmsd_win, defl = plst.bundle_rmsd(
        cell, b, v_rest, ss_window_ms=tuple(oi.train_window_ms),
        ls_window_ms_after_onset=float(cfg.ls_window_ms),
        ls_window_mode=cfg.ls_window_mode,
        r_in_target=args.r_in_target, ss_sample_weight_fn=None)

    pre_s = (0.0, float(b.stim_onset_s))
    subs = plst.ls_subwindows_s(b, charge_ms=float(args.charge_ms))
    sub_rmsd = {}
    for key, win in subs.items():
        try:
            sub_rmsd[key] = float(mono._baseline_subtracted_rmsd(
                b.t, b.v_mV, t_sim_s, v_sim,
                pre_window_s=pre_s, rmsd_window_s=win))
        except Exception:                      # empty or degenerate window
            sub_rmsd[key] = float("nan")

    onset_ms = float(b.stim_onset_s) * 1e3
    offset_ms = (float(b.stim_onset_s) + float(b.stim_duration_s)) * 1e3
    base_exp = _pre_baseline_mV(b.t, b.v_mV, float(b.stim_onset_s))
    base_sim = _pre_baseline_mV(t_sim_s, v_sim, float(b.stim_onset_s))
    sag_exp = IM.sag_metrics(np.asarray(b.t, dtype=float) * 1e3,
                             np.asarray(b.v_mV, dtype=float),
                             v_rest_mV=base_exp, onset_ms=onset_ms,
                             offset_ms=offset_ms,
                             ss_window_ms=float(args.sag_ss_window_ms))
    sag_sim = IM.sag_metrics(np.asarray(t_sim_s, dtype=float) * 1e3,
                             np.asarray(v_sim, dtype=float),
                             v_rest_mV=base_sim, onset_ms=onset_ms,
                             offset_ms=offset_ms,
                             ss_window_ms=float(args.sag_ss_window_ms))

    base_row.update(dict(
        deflection_mV=float(defl), rmsd_window_mV=float(rmsd_win),
        rmsd_charge_mV=sub_rmsd.get("charge", float("nan")),
        rmsd_sag_mV=sub_rmsd.get("sag", float("nan")),
        rmsd_rebound_mV=sub_rmsd.get("rebound", float("nan")),
        baseline_exp_mV=base_exp, baseline_sim_mV=base_sim, error=""))
    for tag, d in (("exp", sag_exp), ("sim", sag_sim)):
        for k, val in d.items():
            base_row["{}_{}".format(tag, k)] = float(val)
    base_row["sag_fraction_error"] = (float(sag_sim["sag_fraction"])
                                      - float(sag_exp["sag_fraction"]))
    return base_row



# ---------------------------------------------------------------------------
#  Phase 3 (bootstrap CIs) over the arm's own parameter vector
# ---------------------------------------------------------------------------
def _run_phase3_subset(mono, IM, results, cells_data, subset_ids,
                       phase2p5_ran, ih_spec, args, out) -> None:
    """Bootstrap CIs for the selected subset. Differs from the biological
    driver in two places only:

      * the rebuilt cell gets `attach_ih` and the theta is applied through the
        SPEC, so the bootstrap starts from the same model the fit ended at
        (applying (C_m,R_m,R_a) alone would drop the I_h knobs AND the rest
        balance, and every replicate would be fit from a drifting cell);
      * `fix_ra` follows Phase 2.5, which is OFF here by default, so the
        bootstrap is over the FULL parameter vector.
    """
    print("\n{0}\n  PHASE 3 -- bootstrap (subset: {1})\n{0}"
          .format("=" * 60, subset_ids))
    common = dict(B=args.bootstrap_B, alpha=0.95, fit_mode="fast",
                  n_calls=args.bootstrap_n_calls,
                  n_initial=args.bootstrap_n_initial,
                  ball_radius_log=args.ball_radius_log,
                  rmsd_reject_mult=5.0, n_workers=1,
                  fix_ra=phase2p5_ran)
    gp_kwargs = dict(n_grid=args.gp_n_grid,
                     inner_grid_per_axis=args.gp_inner_grid,
                     envelope_k=2.0, n_validation_per_bound=5,
                     validation_ball_logradius=0.05,
                     trust_abs_mv=0.10, trust_zscore=3.0)
    by_sid = {int(cd.specimen_id): cd for cd in cells_data}
    rows = []
    for i, fr in enumerate(results):
        sid = int(fr.specimen_id)
        if sid not in subset_ids:
            continue
        if (fr.validation_status not in ("good", "to_refine")
                or getattr(fr, "gp_result", None) is None):
            print("[Phase 3] {}: not fittable (status={}) -> skip"
                  .format(sid, fr.validation_status))
            continue
        cd = by_sid[sid]
        spec = getattr(fr, "param_spec", None)
        pc = None
        try:
            pc = mono.build_neuron_model(cd.swc_path, F=float(fr.F))
            if ih_spec is not None:
                IM.attach_ih(pc, ih_spec)
            _apply_theta(spec, pc, fr, float(fr.v_rest_mV))
            fr.neuron_cell = pc
            if args.bootstrap_mode == "parametric":
                bkw = {**common, "bootstrap_mode": "parametric",
                       "swc_path": str(cd.swc_path),
                       "noise_mode": args.noise_mode, "seed": i}
            else:
                bkw = {**common, "bootstrap_mode": "nonparametric",
                       "pulse_pool": cd.ss_individual_pulses,
                       "swc_path": str(cd.swc_path),
                       "n_pulses_per_replicate":
                           max(1, len(cd.ss_individual_pulses) // 3),
                       "n_avg_groups_bootstrap": args.n_avg_groups, "seed": i}
            p3 = mono.phase3_full_for_cell(
                fit_result=fr, root_dir=str(out), bootstrap_kwargs=bkw,
                gp_kwargs={**gp_kwargs, "seed": i}, verbose=True)
            mono.save_replot_bundle(phase3_result=p3, fit_result=fr,
                                    cell_data=cd, F_used=float(fr.F),
                                    root_dir=str(out), verbose=True)
            b = p3.bootstrap
            for ip, p in enumerate(getattr(b, "param_names",
                                           ("Cm", "Rm", "Ra"))):
                rows.append(dict(specimen_id=sid, parameter=p,
                                 mle=b.mle_physical[ip],
                                 ci_bca_lo=b.ci_bca[p][0],
                                 ci_bca_hi=b.ci_bca[p][1],
                                 ci_perc_lo=b.ci_percentile[p][0],
                                 ci_perc_hi=b.ci_percentile[p][1],
                                 ci_norm_lo=b.ci_normal[p][0],
                                 ci_norm_hi=b.ci_normal[p][1],
                                 n_kept=b.n_kept, mode=b.bootstrap_mode))
        except Exception as exc:  # noqa: BLE001
            print("[Phase 3] {} FAILED: {}: {}"
                  .format(sid, type(exc).__name__, exc))
        finally:
            if pc is not None:
                try:
                    pc.destroy()
                except Exception:
                    pass
            fr.neuron_cell = None
    if rows:
        pd.DataFrame(rows).to_csv(out / "phase3_full_summary.csv", index=False)
        print("[Phase 3] CIs -> {}".format(out / "phase3_full_summary.csv"))


# ---------------------------------------------------------------------------
#  SPEC Block 27: the sensitivity mode (2.10 (d)) -- no fit
# ---------------------------------------------------------------------------
#: tau_d (ms) of the weighting W1 of the identifiability table: no decay
#: after the peak (D-246).
SENS_W1_TAU_D_MS: float = 1e12
#: The first segment of a long bundle in truth_trace_check (ms after onset).
TRUTH_CHECK_FIRST_MS: float = 5.0


def _read_bundles_npz(bundles_npz) -> Dict[str, np.ndarray]:
    """The arrays of a bundles_<specimen>.npz (a path, an open NpzFile or a
    mapping), read without pickle."""
    if isinstance(bundles_npz, (str, Path)):
        with np.load(str(bundles_npz), allow_pickle=False) as z:
            return {k: np.array(z[k]) for k in z.files}
    keys = (list(bundles_npz.files) if hasattr(bundles_npz, "files")
            else list(bundles_npz.keys()))
    return {k: np.array(bundles_npz[k]) for k in keys}


def _read_meta(meta) -> dict:
    """`meta` as a dict: a dict, a JSON string, or the path of the
    sensitivity_<specimen>.json."""
    import json
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, Path) or (isinstance(meta, str)
                                  and not meta.lstrip().startswith("{")):
        return json.loads(Path(meta).read_text())
    return json.loads(str(meta))


def stored_bundles(bundles_npz) -> List[object]:
    """The bundles of a bundles_<specimen>.npz as duck-typed objects (t, s;
    v_mV, mV; amplitude_pA; stim_onset_s; stim_duration_s;
    n_repeats_averaged; sweep_numbers), in the stored order. Pure."""
    import json
    from types import SimpleNamespace
    z = _read_bundles_npz(bundles_npz)
    n = int(np.asarray(z["is_train"]).shape[0])
    sweeps = json.loads(str(z["sweep_numbers"]))
    out = []
    for b in range(n):
        out.append(SimpleNamespace(
            t=np.asarray(z["t_%d" % b], dtype=np.float64),
            v_mV=np.asarray(z["v_%d" % b], dtype=np.float64),
            amplitude_pA=float(z["amplitude_pA"][b]),
            stim_onset_s=float(z["stim_onset_s"][b]),
            stim_duration_s=float(z["stim_duration_s"][b]),
            n_repeats_averaged=int(z["n_repeats_averaged"][b]),
            sweep_numbers=[int(s) for s in sweeps[b]]))
    return out


def write_bundles_npz(path, bundles: Sequence, is_train: Sequence[bool]) -> None:
    """bundles_<specimen>.npz of SPEC Block 27 (np.savez, no pickle): per
    bundle b t_<b> (s) and v_<b> (mV, recorded); over bundles is_train,
    is_brief (bool), amplitude_pA, stim_onset_s, stim_duration_s (float64),
    n_repeats_averaged (int64), sweep_numbers (a JSON string: a list of
    lists)."""
    import json
    import passive_long_step_training as plst
    bundles = list(bundles)
    if len(is_train) != len(bundles):
        raise ValueError("is_train holds %d entries for %d bundles"
                         % (len(is_train), len(bundles)))
    arrays: Dict[str, np.ndarray] = {}
    for b, bd in enumerate(bundles):
        arrays["t_%d" % b] = np.asarray(bd.t, dtype=np.float64)
        arrays["v_%d" % b] = np.asarray(bd.v_mV, dtype=np.float64)
    arrays["is_train"] = np.asarray([bool(x) for x in is_train], dtype=bool)
    arrays["is_brief"] = np.asarray([bool(plst._is_brief(bd)) for bd in bundles],
                                    dtype=bool)
    arrays["amplitude_pA"] = np.asarray([float(bd.amplitude_pA) for bd in bundles],
                                        dtype=np.float64)
    arrays["stim_onset_s"] = np.asarray([float(bd.stim_onset_s) for bd in bundles],
                                        dtype=np.float64)
    arrays["stim_duration_s"] = np.asarray(
        [float(bd.stim_duration_s) for bd in bundles], dtype=np.float64)
    arrays["n_repeats_averaged"] = np.asarray(
        [int(getattr(bd, "n_repeats_averaged", 1)) for bd in bundles],
        dtype=np.int64)
    arrays["sweep_numbers"] = np.array(json.dumps(
        [[int(s) for s in (getattr(bd, "sweep_numbers", None) or [])]
         for bd in bundles]))
    np.savez(str(path), **arrays)


def ss_weight_fn_from_meta(meta):
    """The SS time weight of the fit's loss from `meta` (ss_time_weight,
    ss_tau_w_ms, ss_t0_ms, ss_window_ms), built as integrate_long_step builds
    it: t0 = ss_t0_ms, or the window's start when null; None for 'none'."""
    import passive_long_step_training as plst
    m = _read_meta(meta)
    kind = str(m["ss_time_weight"])
    if kind == "none":
        return None
    win = tuple(float(x) for x in m["ss_window_ms"])
    t0_ms = m.get("ss_t0_ms")
    t0_s = (float(t0_ms) if t0_ms is not None else win[0]) * 1e-3
    scale_s = float(m["ss_tau_w_ms"]) * 1e-3
    if kind == "exp":
        return plst.exp_time_weight(t0_s, scale_s)
    if kind == "gauss":
        return plst.gauss_time_weight(t0_s, scale_s)
    raise ValueError("meta ss_time_weight must be exp, gauss or none, got %r"
                     % (kind,))


def sensitivity_meta(*, specimen_id: int, spec, theta: Dict[str, float],
                     v_rest_mV: float, cfg, args, ss_window_ms, tau_w_ms: float
                     ) -> dict:
    """The `meta` of SPEC Block 27 (JSON-serialisable): the point, the box
    (q), the arm, the protocol of the fit's loss, dt, the steps, n_workers."""
    return dict(
        specimen_id=int(specimen_id), names=[str(n) for n in spec.names],
        bounds=[[float(lo), float(hi)] for lo, hi in
                (ax.q_bounds() for ax in spec.axes)],
        theta={str(n): float(theta[n]) for n in spec.names},
        v_rest_mV=float(v_rest_mV), arm=str(cfg.name),
        ss_window_ms=[float(x) for x in ss_window_ms],
        ls_window_mode=str(cfg.ls_window_mode),
        ls_window_ms_after_onset=float(cfg.ls_window_ms),
        r_in_target=str(args.r_in_target), weighting=str(cfg.weighting),
        ls_sample_weight=str(cfg.ls_sample_weight),
        ls_tau_d_ms=float(args.ls_tau_d_ms),
        ss_time_weight=str(args.ss_time_weight), ss_tau_w_ms=float(tau_w_ms),
        ss_t0_ms=parse_optional_float(args.ss_t0_ms),
        dt_brief_ms=float(args.dt_brief_ms), dt_long_ms=float(args.dt_long_ms),
        h=float(args.sensitivity_h), check_h=float(args.sensitivity_check_h),
        n_workers=int(args.n_workers))


def identifiability_table(bundles_npz, meta, sens, sens_check, names,
                          specimen_id: int, *, v_deepest_mV: float = float("nan"),
                          m_inf_at_v_deepest: float = float("nan")
                          ) -> List[dict]:
    """The rows of identifiability_<specimen>.csv (SPEC Block 27): W0 = the
    meta's protocol, W1 = the same with ls_tau_d_ms = SENS_W1_TAU_D_MS; J =
    jacobian_from_sensitivities of the TRAINING bundles under that layout;
    on W0 only fd_step_check (relative_difference(J(check_h), J(h)), NaN
    without sens_check), loss_at_theta and sqrt_l2_at_theta (from traces0),
    v_deepest_train_mV, m_inf_at_v_deepest. Pure."""
    import identifiability as IDF
    import passive_long_step_training as plst
    m = _read_meta(meta)
    rows: List[dict] = []
    for tag, tau in (("W0", None), ("W1", SENS_W1_TAU_D_MS)):
        layout, tix = layout_from_stored(bundles_npz, m, ls_tau_d_ms=tau)
        J = IDF.jacobian_from_sensitivities([sens["S"][i] for i in tix], layout)
        row: Dict[str, object] = dict(
            specimen_id=int(specimen_id), weighting=tag,
            ls_tau_d_ms=float(m["ls_tau_d_ms"] if tau is None else tau))
        row.update(IDF.identifiability_row(J, list(names)))
        if tag == "W0":
            fd = float("nan")
            if sens_check is not None:
                Jc = IDF.jacobian_from_sensitivities(
                    [sens_check["S"][i] for i in tix], layout)
                fd = IDF.relative_difference(Jc, J)
            r = plst.assemble_residuals(layout,
                                        [sens["traces0"][i] for i in tix])
            L, L2 = plst.l_and_l2(layout, r)
            row.update(fd_step_check=float(fd), loss_at_theta=float(L),
                       sqrt_l2_at_theta=float(np.sqrt(L2)),
                       v_deepest_train_mV=float(v_deepest_mV),
                       m_inf_at_v_deepest=float(m_inf_at_v_deepest))
        rows.append(row)
    return rows


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def truth_trace_rows(traces0: Sequence[np.ndarray], bundles: Sequence,
                     is_train: Sequence[bool], ss_window_ms, specimen_id: int
                     ) -> List[dict]:
    """truth_trace_check_<specimen>.csv (SPEC Block 27): per bundle the
    difference d = traces0[b] - (v_b - its pre-window mean) over the whole
    bundle (pre-window (-10 ms, 0) brief, (0, onset) long, ends included):
    max_abs_mV and t_max_s (the first sample of the maximum); for a long
    bundle the RMS of d over [onset, onset + 5 ms) (rms_first5ms),
    [onset + 5 ms, offset] (rms_rest_of_step), (offset, end]
    (rms_after_offset; NaN without a sample); for a brief bundle rms_window
    over its RMSD window [ss_window_ms] (ends included), the three NaN.
    Pure."""
    import passive_long_step_training as plst
    ssw = tuple(float(x) for x in ss_window_ms)
    nan = float("nan")
    rows = []
    for b, (tr, bd) in enumerate(zip(traces0, bundles)):
        t = np.asarray(bd.t, dtype=np.float64)
        v = np.asarray(bd.v_mV, dtype=np.float64)
        brief = bool(plst._is_brief(bd))
        pre_w, win = plst._bundle_windows_s(bd, ssw, "step", 0.0)
        pm = (t >= pre_w[0]) & (t <= pre_w[1])
        d = np.asarray(tr, dtype=np.float64) - (v - float(np.mean(v[pm])))
        k = int(np.argmax(np.abs(d)))
        row: Dict[str, object] = dict(
            specimen_id=int(specimen_id), bundle=int(b),
            is_train=bool(is_train[b]), is_brief=brief,
            amplitude_pA=float(bd.amplitude_pA),
            max_abs_mV=float(np.abs(d[k])), t_max_s=float(t[k]),
            rms_first5ms=nan, rms_rest_of_step=nan, rms_after_offset=nan,
            rms_window=nan)
        if brief:
            row["rms_window"] = _rms(d[(t >= win[0]) & (t <= win[1])])
        else:
            on = float(bd.stim_onset_s)
            off = on + float(bd.stim_duration_s)
            t5 = on + TRUTH_CHECK_FIRST_MS * 1e-3
            row["rms_first5ms"] = _rms(d[(t >= on) & (t < t5)])
            row["rms_rest_of_step"] = _rms(d[(t >= t5) & (t <= off)])
            row["rms_after_offset"] = _rms(d[t > off])
        rows.append(row)
    return rows


def _m_inf_at(v_mV: float, ih_spec, dv_h_mV: float) -> float:
    """m_inf(v) of the spec's mechanism, base shift and dv_h; NaN without
    I_h or for a NaN v."""
    import human_ih_params as HIP
    if ih_spec is None or not np.isfinite(v_mV):
        return float("nan")
    return float(HIP.minf(float(v_mV), str(ih_spec.mechanism),
                          vshift_mV=float(ih_spec.vshift_base_mV),
                          vshift_minf_mV=float(dv_h_mV)))


def _run_sensitivity_mode(mono, plst, IM, cells_data, opt_inputs, spec, cfg,
                          args, ih_spec, theta, tau_w_ms, clear) -> int:
    """SPEC Block 27, 2.10 (d): for each loaded cell, built as for a fit,
    the sensitivities at `theta` (physical) and the four files in
    --sensitivity-out. Returns the number of cells done."""
    import json
    import trace_sensitivity as TS
    outd = Path(args.sensitivity_out)
    outd.mkdir(parents=True, exist_ok=True)
    q = np.asarray(spec.to_q(theta), dtype=np.float64)
    bounds = [tuple(float(x) for x in ax.q_bounds()) for ax in spec.axes]
    n_ok = 0
    for i, (cd, oi) in enumerate(zip(cells_data, opt_inputs)):
        sid = int(cd.specimen_id)
        t_cell = time.time()
        log("  cell {}/{} (specimen {}): SENSITIVITY START"
            .format(i + 1, len(cells_data), sid))
        clear()
        cell = None
        ev = None
        try:
            cell = mono.build_neuron_model(cd.swc_path, F=args.F)
            if ih_spec is not None:
                IM.attach_ih(cell, ih_spec)
            mono.assert_dt_enforced(
                cell, dt_values_ms=(args.dt_brief_ms, args.dt_long_ms),
                v_init_mV=float(oi.v_rest_mV), verbose=(i == 0))
            _assert_loss_live(mono, cell, oi, spec)
            train = list(oi.train_bundles)
            valid = list(oi.validation_bundles)
            bundles = train + valid
            is_train = [True] * len(train) + [False] * len(valid)
            v_rest = float(oi.v_rest_mV)
            ev = plst.build_trace_evaluator(
                cell, bundles, v_rest, spec=spec,
                dt_brief_ms=float(args.dt_brief_ms),
                dt_long_ms=float(args.dt_long_ms),
                n_workers=int(args.n_workers))
            sens = TS.compute_sensitivities(ev, q, bounds,
                                            h=float(args.sensitivity_h),
                                            scheme="central")
            sens_c = None
            if float(args.sensitivity_check_h) > 0:
                sens_c = TS.compute_sensitivities(
                    ev, q, bounds, h=float(args.sensitivity_check_h),
                    scheme="central")
            meta = sensitivity_meta(specimen_id=sid, spec=spec, theta=theta,
                                    v_rest_mV=v_rest, cfg=cfg, args=args,
                                    ss_window_ms=tuple(oi.train_window_ms),
                                    tau_w_ms=tau_w_ms)
            bpath = outd / ("bundles_%d.npz" % sid)
            write_bundles_npz(bpath, bundles, is_train)
            TS.save_sensitivities(outd / ("sensitivity_%d.npz" % sid), sens,
                                  t_list=[np.asarray(b.t, dtype=np.float64)
                                          for b in bundles],
                                  names=spec.names, meta=meta)
            (outd / ("sensitivity_%d.json" % sid)).write_text(
                json.dumps(meta, indent=1) + "\n")
            v_star = v_deepest_train_mV(getattr(oi, "ls_roles", None))
            rows = identifiability_table(
                bpath, meta, sens, sens_c, spec.names, sid,
                v_deepest_mV=v_star,
                m_inf_at_v_deepest=_m_inf_at(v_star, ih_spec,
                                             float(theta.get("dv_h", 0.0))))
            pd.DataFrame(rows).to_csv(outd / ("identifiability_%d.csv" % sid),
                                      index=False)
            tt = truth_trace_rows(sens["traces0"], bundles, is_train,
                                  tuple(oi.train_window_ms), sid)
            pd.DataFrame(tt).to_csv(outd / ("truth_trace_check_%d.csv" % sid),
                                    index=False)
            w0 = rows[0]
            log("    {} point(s) simulated; sigma = [{}], ratio {:.3g}; "
                "fd_step_check {:.3g}; L {:.6g}, sqrt(L_2) {:.6g}; truth "
                "trace max |d| {:.3g} mV | {:.0f}s".format(
                    int(sens["n_points"]) + (int(sens_c["n_points"])
                                             if sens_c is not None else 0),
                    ", ".join("%.4g" % w0["sigma_%d" % (k + 1)]
                              for k in range(len(spec.names))),
                    w0["sigma_ratio"], w0["fd_step_check"],
                    w0["loss_at_theta"], w0["sqrt_l2_at_theta"],
                    max(r["max_abs_mV"] for r in tt), time.time() - t_cell))
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            log("  cell {}/{} ({}): SENSITIVITY FAILED after {:.0f}s -> {}: {}"
                .format(i + 1, len(cells_data), sid, time.time() - t_cell,
                        type(exc).__name__, exc))
            if args.fail_fast:
                raise
        finally:
            if ev is not None:
                ev.close()
            if cell is not None:
                try:
                    cell.destroy()
                except Exception:
                    pass
    clear()
    return n_ok


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def _parse_args(argv):
    ap = argparse.ArgumentParser(
        description="I_h campaign: one arm, one group, one PBS job.")
    ap.add_argument("--archive-dir", required=True,
                    help="Group archive dir: <ROOT>/<GROUP> with specimen_*/.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--code-dir", required=True,
                    help="Dir with passive_fitting_hpc_fixed.py, "
                         "passive_long_step_training.py, param_spec.py, "
                         "ih_mechanism.py, human_ih_params.py and x86_64/.")
    ap.add_argument("--swc-dir", default=None,
                    help="Override morphologies: <id>.swc / specimen_<id>.swc "
                         "/ specimen_<id>/reconstruction.swc. The loader "
                         "REFUSES a cell whose SWC is absent here rather than "
                         "falling back to the archive's (C4).")
    ap.add_argument("--n-avg-groups", type=int, default=1)
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--fail-fast", action="store_true")

    # --- arm and parameter vector ------------------------------------------
    ap.add_argument("--arm", default="ih6", choices=sorted(ARMS),
                    help="Named configuration (default ih6).")
    ap.add_argument("--fit-params", default=None,
                    help="Override the arm's axes, e.g. 'Cm,Rm,Ra,gbar' for "
                         "the 4-D arm. Omitted kinetic knobs are FROZEN at "
                         "their base value (D-005).")
    ap.add_argument("--cm-bounds", default="0.3,3.0")
    ap.add_argument("--rm-bounds", default="1000.0,100000.0")
    ap.add_argument("--ra-bounds", default="50.0,1000.0")
    ap.add_argument("--gbar-bounds", default="1e-5,1e-3",
                    help="S/cm^2 (SPEC 2.9 (a), D-234; was 1e-6,1e-2 under "
                         "D-208).")
    ap.add_argument("--dvh-bounds", default="-10.0,10.0")
    ap.add_argument("--kappa-bounds", default="0.5,2.0")
    ap.add_argument("--tau-h-pk-bounds-ms", default=None,
                    help="'LO,HI' (ms): the box of tau_h_pk; REPLACES "
                         "--kappa-bounds by (LO/base, HI/base), base = the "
                         "peak of the mechanism's tau_h at kappa_tau = 1 "
                         "(SPEC 2.8 (a)).")
    ap.add_argument("--eval-theta", default=None,
                    help="'Cm=..,Rm=..,Ra=..,gbar=..[,dv_h=..],kappa_tau=..': the "
                         "training loss is also evaluated here and written as "
                         "loss_at_eval_theta. Every name of the arm's spec "
                         "exactly once, else exit 2.")

    # --- the I_h configuration (never fitted; D-005/D-007) -----------------
    ap.add_argument("--ih-mechanism", default="Ih_human",
                    choices=["Ih_human", "Ih"],
                    help="Base kinetics: Rich 2021 (default) or Kole 2006.")
    ap.add_argument("--ih-distribution", default="uniform",
                    choices=["uniform", "eyal_exp_323", "hay_exp_dmax"])
    ap.add_argument("--ih-regions", default="soma,dend,apic",
                    help="Where the mechanism is inserted. The axon stub is "
                         "excluded by default (a Hay artefact).")
    ap.add_argument("--vshift-base", type=float, default=0.0,
                    help="Configuration shift of BOTH curves, mV. +20 on "
                         "--ih-mechanism Ih reproduces Kalmbach 2018 (their "
                         "'-20 mV' is the term inside the rate argument).")
    ap.add_argument("--ehcn", default=None,
                    help="Override E_h (mV). Default: the mechanism's own.")
    ap.add_argument("--mtau-min-ms", type=float, default=0.0,
                    help="Ih_human only; 0 = the published model.")

    # --- protocol (plan section 5 / D-006) ---------------------------------
    ap.add_argument("--ls-window", default=None,
                    choices=["after_onset", "step", "sweep"],
                    help="Override the arm's RMSD window on long steps.")
    ap.add_argument("--ls-window-ms", type=float, default=None,
                    help="Window length for --ls-window after_onset (ms).")
    ap.add_argument("--ls-max-amplitude-pA", default=None,
                    help="Loader cap on |I| for hyperpolarising long steps; "
                         "'none' admits every recorded sweep.")
    ap.add_argument("--ls-fallback-amplitude-pA", type=float, default=300.0)
    ap.add_argument("--ls-dep-max-amplitude-pA", default="100.0",
                    help="Cap on the depolarising validation steps; 'none' "
                         "admits every spike-free one.")
    ap.add_argument("--n-drop-weakest", type=int, default=1,
                    help="Hyperpolarising steps withheld from training at the "
                         "weak end (D-006 default 1: h_1 validates).")
    ap.add_argument("--n-drop-strongest", type=int, default=1,
                    help="Withheld at the strong end (D-006 default 1: h_n is "
                         "report-only; PIR conductances are unmodelled).")
    ap.add_argument("--dep-n-validation", type=int, default=3,
                    help="At most this many smallest spike-free depolarising "
                         "steps validate. Negative = all of them.")
    ap.add_argument("--v-trough-min", default=None,
                    help="Demote a training step whose V_trough is below this "
                         "voltage (mV) to report-only. Off by default.")
    ap.add_argument("--train-all-hyp", action="store_true",
                    help="D-007's labelled opt-in: train on h_1..h_n.")
    ap.add_argument("--ls-role-rule", default=None, choices=["d006", "d207"],
                    help="Long-step roles; default None = the arm's "
                         "(SPEC 2.8 (b)).")
    ap.add_argument("--ls-sample-weight", default=None,
                    choices=["none", "peak_decay"],
                    help="Per-sample weight of a long step in the TRAINING "
                         "loss; default None = the arm's (SPEC 2.8 (d)).")
    ap.add_argument("--ls-tau-d-ms", type=float, default=500.0,
                    help="tau_d of the peak_decay weight (ms).")
    ap.add_argument("--gate-valid-via", default=None,
                    choices=["legacy_early", "same_window"],
                    help="Override how the absolute-mV gate windows the "
                         "validation bundles (see the module docstring).")

    # --- loss / fit --------------------------------------------------------
    ap.add_argument("--fit-target", default="hyp", choices=["dep", "hyp", "both"])
    ap.add_argument("--F", type=float, default=1.9)
    ap.add_argument("--n-calls", type=int, default=200)
    ap.add_argument("--n-initial", type=int, default=100)
    ap.add_argument("--r-in-target", default="peak", choices=["peak", "steady"])
    ap.add_argument("--weighting", default=None,
                    help="Cross-bundle weighting; default None = the arm's "
                         "(relative, or ls_amplitude on ih5 and "
                         "passive_fullstep: SPEC 2.8 (c)).")
    ap.add_argument("--ss-window-ms", default="0.5,100.0")
    ap.add_argument("--ss-t0-ms", default=None)
    ap.add_argument("--ss-time-weight", default="exp",
                    choices=["exp", "gauss", "none"])
    ap.add_argument("--ss-tau-w-ms", default="5.0",
                    help="ONE value (no sweep here; see the module docstring).")
    ap.add_argument("--dt-brief-ms", type=float, default=0.1)
    ap.add_argument("--dt-long-ms", type=float, default=0.1)
    ap.add_argument("--refine", default=None, choices=list(REFINE_CHOICES),
                    help="Local refinement after the global search; default "
                         "None = the arm's (trf on ih5, ih6_d207 and "
                         "passive_fullstep: SPEC 2.10 (c); nelder_mead: 2.9 "
                         "(b), a labelled comparison).")
    ap.add_argument("--refine-starts", type=int, default=3,
                    help="At most this many starts (both refinements).")
    ap.add_argument("--refine-maxfev", type=int, default=100,
                    help="Nelder-Mead only: scipy's maxfev of one start.")
    ap.add_argument("--refine-max-nfev", type=int, default=30,
                    help="trf only: scipy least_squares' max_nfev of one "
                         "start (TrfConfig.max_nfev).")
    ap.add_argument("--n-workers", type=int, default=1,
                    help="Processes computing the sweeps of one loss "
                         "evaluation (SPEC 2.9 (c)); 1 = serial.")

    # --- sensitivity mode (SPEC Block 27, 2.10 (d)): no fit ----------------
    ap.add_argument("--sensitivity-theta", default=None,
                    help="'name=value,...' (every fitted name, physical "
                         "units, inside the box): compute and store the "
                         "sensitivities there instead of fitting.")
    ap.add_argument("--sensitivity-out", default=None,
                    help="Directory of the sensitivity mode's files "
                         "(required with --sensitivity-theta).")
    ap.add_argument("--sensitivity-h", type=float, default=1e-3,
                    help="Central-difference step, unit coordinates.")
    ap.add_argument("--sensitivity-check-h", type=float, default=1e-2,
                    help="The step of fd_step_check; 0 = no check.")

    # --- diagnostics -------------------------------------------------------
    ap.add_argument("--charge-ms", type=float, default=150.0,
                    help="Charging/sag sub-window boundary after onset (ms).")
    ap.add_argument("--sag-ss-window-ms", type=float, default=100.0,
                    help="Window before the step offset used as V_ss.")
    ap.add_argument("--rest-drift-ms", type=float, default=2000.0,
                    help="Unstimulated settle used to report the rest drift.")

    # --- Phase 2.5 (OFF by default; D-006 Q8) ------------------------------
    ap.add_argument("--run-phase2p5", action="store_true",
                    help="3-D arms only: fix R_a at the cohort median and "
                         "refit (C_m, R_m).")
    ap.add_argument("--n-floor", type=int, default=4)
    ap.add_argument("--n-ra-profile", type=int, default=50)

    # --- Phase 3 -----------------------------------------------------------
    ap.add_argument("--phase3-subset", default="frac:0.5",
                    help="none | all | first:N | frac:F (default frac:0.5).")
    ap.add_argument("--bootstrap-B", type=int, default=200)
    ap.add_argument("--bootstrap-mode", default="nonparametric",
                    choices=["parametric", "nonparametric"])
    ap.add_argument("--noise-mode", default="block",
                    choices=["iid", "ar1", "block"])
    ap.add_argument("--bootstrap-n-calls", type=int, default=100)
    ap.add_argument("--bootstrap-n-initial", type=int, default=30)
    ap.add_argument("--ball-radius-log", type=float, default=0.2)
    ap.add_argument("--gp-n-grid", type=int, default=80)
    ap.add_argument("--gp-inner-grid", type=int, default=30)
    args = ap.parse_args(argv)
    # skopt refuses n_calls < n_initial_points. Inside the per-cell loop that
    # ValueError is logged and the loop moves on, so a whole group would end
    # as failed_cells.txt; refuse the configuration before any cell is read.
    for calls, init, name in ((args.n_calls, args.n_initial, ""),
                              (args.bootstrap_n_calls, args.bootstrap_n_initial,
                               "bootstrap-")):
        if init > calls:
            ap.error("--%sn-initial (%d) exceeds --%sn-calls (%d): gp_minimize "
                     "refuses this." % (name, init, name, calls))
    for name in ("refine_starts", "refine_maxfev", "refine_max_nfev",
                 "n_workers"):
        if getattr(args, name) < 1:
            ap.error("--%s must be >= 1, got %d"
                     % (name.replace("_", "-"), getattr(args, name)))
    # --eval-theta names the arm's spec exactly: refuse it here (exit 2)
    # rather than after the cells are loaded.
    if args.eval_theta is not None:
        try:
            parse_eval_theta(args.eval_theta, resolve_arm_config(args).fit_params)
        except ValueError as exc:
            ap.error(str(exc))
    # SPEC Block 27: the sensitivity mode's point is inside the box, its
    # directory given and its steps in fd_jacobian's range (exit 2).
    if args.sensitivity_theta is not None:
        try:
            cfg = resolve_arm_config(args)
            theta = parse_eval_theta(args.sensitivity_theta, cfg.fit_params)
            spec = build_spec_from_args(args, cfg)
        except (ValueError, KeyError) as exc:
            ap.error("--sensitivity-theta: %s" % (str(exc).replace(
                "--eval-theta", "--sensitivity-theta"),))
        out_of_box = [n for n in spec.names
                      if not (spec.axis(n).lo <= theta[n] <= spec.axis(n).hi)]
        if out_of_box:
            ap.error("--sensitivity-theta outside the box on %s: %s"
                     % (",".join(out_of_box), "; ".join(
                         "%s=%g not in [%g, %g]" % (n, theta[n], spec.axis(n).lo,
                                                    spec.axis(n).hi)
                         for n in out_of_box)))
        if not args.sensitivity_out:
            ap.error("--sensitivity-theta needs --sensitivity-out DIR")
    if not (0.0 < args.sensitivity_h <= 0.5):
        ap.error("--sensitivity-h must lie in (0, 0.5], got %r"
                 % (args.sensitivity_h,))
    if not (args.sensitivity_check_h == 0.0
            or 0.0 < args.sensitivity_check_h <= 0.5):
        ap.error("--sensitivity-check-h must be 0 or lie in (0, 0.5], got %r"
                 % (args.sensitivity_check_h,))
    return args


# ---------------------------------------------------------------------------
#  SPEC Block 27: the layout of stored bundles, pure
# ---------------------------------------------------------------------------
def layout_from_stored(bundles_npz, meta, *, ls_tau_d_ms=None,
                       ls_window_mode=None, ls_sample_weight=None):
    """(layout, train_index): passive_long_step_training.residual_layout of
    the stored TRAINING bundles of `bundles_npz` (Block 27's
    bundles_<specimen>.npz, a path or a loaded mapping) under `meta`'s
    protocol (a dict, its JSON string or the sensitivity_<specimen>.json),
    with the given overrides (None = meta's); train_index: the positions of
    the training bundles among the stored ones. No NEURON. SPEC Block 27."""
    import passive_long_step_training as plst
    m = _read_meta(meta)
    z = _read_bundles_npz(bundles_npz)
    bundles = stored_bundles(z)
    train_index = [int(i) for i in
                   np.flatnonzero(np.asarray(z["is_train"], dtype=bool))]
    layout = plst.residual_layout(
        [bundles[i] for i in train_index],
        ss_window_ms=tuple(float(x) for x in m["ss_window_ms"]),
        ls_window_ms_after_onset=float(m["ls_window_ms_after_onset"]),
        ls_window_mode=str(m["ls_window_mode"] if ls_window_mode is None
                           else ls_window_mode),
        r_in_target=str(m["r_in_target"]),
        weighting=str(m["weighting"]),
        ss_sample_weight_fn=ss_weight_fn_from_meta(m),
        ls_sample_weight=str(m["ls_sample_weight"] if ls_sample_weight is None
                             else ls_sample_weight),
        ls_tau_d_ms=float(m["ls_tau_d_ms"] if ls_tau_d_ms is None
                          else ls_tau_d_ms))
    return layout, train_index

if __name__ == "__main__":
    main()
