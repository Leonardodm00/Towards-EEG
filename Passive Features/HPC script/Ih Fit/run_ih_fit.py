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


ARMS: Dict[str, ArmConfig] = {
    # Run B, reproduced: legacy loader, legacy split, 60 ms window, 3-D.
    "baseline_runB": ArmConfig(
        name="baseline_runB", fit_params=PASSIVE_NAMES,
        attach_ih=False, ih_protocol=False,
        ls_window_mode="after_onset", ls_window_ms=60.0,
        ls_max_amplitude_pA=100.0, load_depolarising_ls=False,
        n_long_train=2, ls_deflection_cap_mV=12.0,
        gate_valid_via="legacy_early"),
    # D-006 protocol, still passive: the control for "was it I_h or the window?"
    "passive_fullstep": ArmConfig(
        name="passive_fullstep", fit_params=PASSIVE_NAMES,
        attach_ih=False, ih_protocol=True,
        ls_window_mode="step", ls_window_ms=150.0,
        ls_max_amplitude_pA=None, load_depolarising_ls=True,
        n_long_train=0, ls_deflection_cap_mV=None,
        gate_valid_via="same_window"),
    # The deliverable.
    "ih6": ArmConfig(
        name="ih6", fit_params=IH6_NAMES,
        attach_ih=True, ih_protocol=True,
        ls_window_mode="step", ls_window_ms=150.0,
        ls_max_amplitude_pA=None, load_depolarising_ls=True,
        n_long_train=0, ls_deflection_cap_mV=None,
        gate_valid_via="same_window"),
}


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
    return replace(cfg, **over) if over else cfg


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
    from neuron import h
    h.load_file("stdrun.hoc")

    def _clear():
        for s in list(h.allsec()):
            h.delete_section(sec=s)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ss_window = parse_window(args.ss_window_ms)
    group_label = Path(args.archive_dir).name

    spec = build_param_spec(
        cfg.fit_params,
        cm_bounds=parse_window(args.cm_bounds),
        rm_bounds=parse_window(args.rm_bounds),
        ra_bounds=parse_window(args.ra_bounds),
        gbar_bounds=parse_window(args.gbar_bounds),
        dvh_bounds=parse_window(args.dvh_bounds),
        kappa_bounds=parse_window(args.kappa_bounds))
    ih_spec = build_ih_spec(args) if cfg.attach_ih else None
    ih_label = ih_spec.label() if ih_spec is not None else ""

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
        r_in_target=args.r_in_target, weighting=args.weighting,
        ss_window_ms=ss_window, ss_time_weight=args.ss_time_weight,
        ss_tau_w_ms=tau_w_ms,
        ss_t0_ms=parse_optional_float(args.ss_t0_ms),
        ls_window_ms_after_onset=float(cfg.ls_window_ms),
        dt_brief_ms=float(args.dt_brief_ms),
        dt_long_ms=float(args.dt_long_ms),
        verbose=True)
    log("[1/5] PATCH -- done.")

    # ---- 2. LOAD this group's archives ------------------------------------
    log("[2/5] LOAD -- reading specimen_* archives + optimiser inputs ...")
    cells_data = mono.load_cells_from_archive(
        args.archive_dir, n_avg_groups=args.n_avg_groups,
        ls_max_amplitude_pA=cfg.ls_max_amplitude_pA,
        ls_fallback_amplitude_pA=args.ls_fallback_amplitude_pA,
        load_depolarising_ls=cfg.load_depolarising_ls,
        ls_dep_max_amplitude_pA=parse_optional_float(args.ls_dep_max_amplitude_pA),
        swc_dir=args.swc_dir, specimen_ids=None,
        max_cells=args.max_cells, verbose=True)
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

    # ---- 3. FIT, one cell at a time ---------------------------------------
    log("[3/5] FIT -- {} cell(s), sequential ...".format(len(cells_data)))
    results: List[object] = []
    diag_rows: List[dict] = []
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
            fr = mono.fit_one_cell(cell, cd, oi, spec=spec, F=args.F,
                                   n_calls=args.n_calls,
                                   n_initial=args.n_initial, seed=i)
            fr.tau_w_chosen_ms = float(tau_w_ms)
            fr.tau_w_hw_rho = float("nan")
            fr.tau_w_reason = "fixed_by_cli"

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
    for cm in (0.5, 1.0, 2.0):
        q = list(q0)
        q[icm] = float(np.log(cm))
        vals.append(float(L(*q)))
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
    ap.add_argument("--gbar-bounds", default="1e-6,1e-3")
    ap.add_argument("--dvh-bounds", default="-10.0,10.0")
    ap.add_argument("--kappa-bounds", default="0.5,2.0")

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
    ap.add_argument("--weighting", default="relative")
    ap.add_argument("--ss-window-ms", default="0.5,100.0")
    ap.add_argument("--ss-t0-ms", default=None)
    ap.add_argument("--ss-time-weight", default="exp",
                    choices=["exp", "gauss", "none"])
    ap.add_argument("--ss-tau-w-ms", default="5.0",
                    help="ONE value (no sweep here; see the module docstring).")
    ap.add_argument("--dt-brief-ms", type=float, default=0.1)
    ap.add_argument("--dt-long-ms", type=float, default=0.1)

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
    return args


if __name__ == "__main__":
    main()
