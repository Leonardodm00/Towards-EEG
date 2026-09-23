# -*- coding: utf-8 -*-
"""
ih_recovery_report.py
=====================

Stage 7's analysis half: given the manifest (ground truth) and one
`phase2_results.csv` per arm (estimates), produce the recovery table, the
C_m-inflation read-out, the false-positive-control table, and the gate verdict
that decides whether Stage 8 may start.

PURE: pandas and numpy only. No NEURON, no fitting, no plotting -- plotting
lives in `plot_recovery` at the bottom and is never called by the statistics.
That separation is what lets `smoke_ih_recovery.py` exercise the arithmetic on
fixtures with a KNOWN answer, which is the only way to find out whether a
recovery number means what it says.

The two error measures, and why they are not the same measure
-------------------------------------------------------------
For a cell c and a parameter axis i, with true value theta_i(c) and estimate
theta_hat_i(c):

    LOG axes   (C_m, R_m, R_a, gbar, kappa_tau), all of them positive and
    searched in log coordinates:

        eps_i(c) = log( theta_hat_i(c) / theta_i(c) )                  (R.1)

    dimensionless; eps = log(1.25) is "25 % too large". This is the measure
    the search itself uses, so a tolerance on it is a tolerance in the
    optimiser's own metric.

    LINEAR axis (dv_h only -- the one linear dimension of the spec, D-005):

        eps_dvh(c) = dvh_hat(c) - dvh(c)                     [mV]      (R.2)

    A log ratio is undefined here (dv_h takes both signs and passes through
    zero), and a relative error on a quantity whose zero is a modelling
    convention rather than an origin would be meaningless. The tolerance is
    therefore an absolute voltage, in mV.

Which cells enter which statistic -- this is load-bearing
----------------------------------------------------------
* gbar, dv_h and kappa_tau recovery is computed on the I_h cells ONLY
  (`is_fp_control == False`). On a false-positive-control cell the true gbar
  sits at the box floor, so (R.1) is dominated by the floor rather than by the
  fit, and dv_h and kappa_tau are not identifiable at all when gbar ~ 0: they
  multiply a current that is not there. Including those cells would make the
  recovery look worse (or, if the fit also rails, spuriously perfect) for
  reasons that have nothing to do with recovery.
* The C_m inflation of a passive arm is computed on the I_h cells ONLY, for
  the same reason in reverse: a passive cell has no I_h to absorb into C_m.
* The false-positive control is its own table, and it asks a different
  question: does the 6-D fit rail gbar to the floor on a cell with no I_h, or
  does it invent one and absorb C_m into it?

A railed axis is not an estimate
--------------------------------
`rail_<name>` marks an axis sitting on a bound. For gbar the two sides mean
opposite things -- the low rail is "the data wanted no I_h", the high rail is
"the box was too narrow" -- and neither is a measurement of the truth. Railed
cells are counted and reported per arm and per axis, and `--drop-railed`
excludes them from the recovery medians (default: keep them, and say how many
there were, because silently dropping the failures is how a gate passes).
"""

from dataclasses import dataclass, asdict
from math import log
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ===========================================================================
#  Vocabulary
# ===========================================================================
#: manifest column holding the ground truth of each fitted axis
TRUTH_COLUMN: Dict[str, str] = {
    "Cm": "cm_true",
    "Rm": "rm_true",
    "Ra": "ra_true",
    "gbar": "ih_gihbar_S_cm2",
    "dv_h": "ih_dvh_mV",
    "kappa_tau": "ih_kappa_tau",
}

#: axes whose error is the log ratio (R.1)
LOG_AXES: Tuple[str, ...] = ("Cm", "Rm", "Ra", "gbar", "kappa_tau")
#: axes whose error is the plain difference (R.2)
LINEAR_AXES: Tuple[str, ...] = ("dv_h",)

#: The search box, needed ONLY to say which SIDE a railed axis railed on --
#: the CSV records that an axis is on a bound, not which one. Keep in step
#: with param_spec's defaults; `recovery_long` cross-checks it against the
#: `rail_<name>` column and refuses on a disagreement, because a box here that
#: differs from the box the fit used would silently mislabel every rail.
DEFAULT_BOX: Dict[str, Tuple[float, float]] = {
    "Cm": (0.3, 3.0),
    "Rm": (1e3, 1e5),
    "Ra": (50.0, 1000.0),
    "gbar": (1e-6, 1e-3),
    "dv_h": (-10.0, 10.0),
    "kappa_tau": (0.5, 2.0),
}


@dataclass(frozen=True)
class RecoveryTolerances:
    """The Stage-7 gate (plan section 8, Stage 7).

    PROPOSED by the assistant, not yet a user decision -- the plan marks the
    numbers [open]. They are applied to the MEDIAN over cells of the absolute
    error, per (arm, axis), and every raw per-cell error is written out, so a
    different tolerance can be applied afterwards without re-running anything.
    """
    log_ratio_cm: float = log(1.25)        # |log(Cm_hat/Cm)| <= log 1.25
    log_ratio_gbar: float = log(1.25)      # |log(gbar_hat/gbar)| <= log 1.25
    abs_dvh_mV: float = 3.0                # |dvh_hat - dvh| <= 3 mV
    log_ratio_kappa: float = log(1.5)      # |log(kappa_hat/kappa)| <= log 1.5
    #: fraction of false-positive-control cells whose 6-D fit must rail gbar
    #: at the LOW bound (i.e. correctly report "no I_h here")
    fp_min_low_rail_frac: float = 0.8
    #: and whose C_m must still be recovered, since there is no I_h to absorb
    fp_log_ratio_cm: float = log(1.25)

    def for_axis(self, axis: str) -> Optional[float]:
        """The tolerance applied to `axis`, or None if the axis is reported
        but not gated (R_m and R_a are: they are nuisance here, and the plan
        sets no tolerance for them)."""
        return {"Cm": self.log_ratio_cm, "gbar": self.log_ratio_gbar,
                "dv_h": self.abs_dvh_mV,
                "kappa_tau": self.log_ratio_kappa}.get(axis)

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


# ===========================================================================
#  Loading
# ===========================================================================
def load_arm_results(path: Union[Path, str], arm: str) -> pd.DataFrame:
    """One arm's `phase2_results.csv`, tagged with the arm name.

    Only the columns this module reads are required; everything else is
    carried through untouched so the caller can keep diagnostics.
    """
    df = pd.read_csv(path)
    if "specimen_id" not in df.columns:
        raise KeyError("{} has no specimen_id column".format(path))
    df = df.copy()
    df["arm"] = str(arm)
    return df


def _rail_side(value: float, lo: float, hi: float, rtol: float = 1e-3) -> str:
    """'lo' | 'hi' | '' -- which bound `value` is sitting on, if any.

    The comparison is the one `ParamSpec.rail_flags` makes: relative in the
    axis's own coordinate, so a log axis is compared in log.
    """
    if not np.isfinite(value):
        return ""
    if lo > 0.0 and hi > 0.0:                    # positive axis -> compare in log
        v, l, h = np.log(value), np.log(lo), np.log(hi)
    else:                                        # linear axis
        v, l, h = float(value), float(lo), float(hi)
    span = max(abs(h - l), 1e-12)
    if abs(v - l) <= rtol * span:
        return "lo"
    if abs(v - h) <= rtol * span:
        return "hi"
    return ""


# ===========================================================================
#  The long table: one row per (arm, cell, axis)
# ===========================================================================
def recovery_long(manifest: pd.DataFrame,
                  arm_results: Sequence[pd.DataFrame],
                  *,
                  box: Optional[Dict[str, Tuple[float, float]]] = None,
                  strict_rails: bool = True) -> pd.DataFrame:
    """Join truth to estimates and compute the per-cell error of every axis.

    Returns one row per (arm, specimen_id, parameter) with columns
        arm, specimen_id, is_fp_control, parameter, truth, estimate,
        err          -- (R.1) for a log axis, (R.2) for a linear one
        abs_err      -- |err|
        err_kind     -- "log_ratio" | "difference"
        rail, rail_side, sigma, validation_status

    A cell present in the manifest but absent from an arm's results (it
    failed, or was not in that group) simply has no row for that arm; the
    per-arm `n` in the summary is what reveals it.
    """
    box = dict(DEFAULT_BOX if box is None else box)
    if manifest.empty:
        raise ValueError("empty manifest")
    man = manifest.set_index("specimen_id", drop=False)

    rows: List[dict] = []
    mismatches: List[str] = []
    for res in arm_results:
        arm = str(res["arm"].iloc[0]) if len(res) else "?"
        # the axes THIS arm actually fitted, as the fit itself named them
        names = _fitted_axes(res)
        for _, r in res.iterrows():
            sid = int(r["specimen_id"])
            if sid not in man.index:
                continue
            truth_row = man.loc[sid]
            fp = bool(truth_row.get("is_fp_control", False))
            for axis in names:
                if axis not in TRUTH_COLUMN or axis not in res.columns:
                    continue
                est = _f(r.get(axis))
                tru = _f(truth_row.get(TRUTH_COLUMN[axis]))
                if axis in LOG_AXES:
                    kind = "log_ratio"
                    err = (np.log(est / tru)
                           if (np.isfinite(est) and np.isfinite(tru)
                               and est > 0.0 and tru > 0.0) else np.nan)
                else:
                    kind = "difference"
                    err = (est - tru if (np.isfinite(est) and np.isfinite(tru))
                           else np.nan)
                lo, hi = box.get(axis, (np.nan, np.nan))
                side = (_rail_side(est, lo, hi)
                        if np.isfinite(lo) and np.isfinite(hi) else "")
                railed_csv = r.get("rail_" + axis, None)
                railed = (bool(railed_csv) if railed_csv is not None
                          and not (isinstance(railed_csv, float)
                                   and not np.isfinite(railed_csv))
                          else bool(side))
                if (railed_csv is not None and bool(railed_csv) != bool(side)
                        and np.isfinite(est)):
                    mismatches.append(
                        "{} specimen {} axis {}: CSV rail={} but the box {} "
                        "in this report says {!r}"
                        .format(arm, sid, axis, bool(railed_csv), (lo, hi),
                                side or "not on a bound"))
                rows.append(dict(
                    arm=arm, specimen_id=sid, is_fp_control=fp,
                    parameter=axis, truth=tru, estimate=est,
                    err=err, abs_err=abs(err) if np.isfinite(err) else np.nan,
                    err_kind=kind, rail=railed, rail_side=side,
                    sigma=_f(r.get(axis + "_sigma")),
                    validation_status=str(r.get("validation_status", "")),
                ))
    if mismatches and strict_rails:
        raise ValueError(
            "the search box in this report disagrees with the one the fit "
            "used, so every rail would be mislabelled. Pass the fit's own box "
            "via --box, or strict_rails=False to continue anyway. First "
            "disagreements:\n  " + "\n  ".join(mismatches[:5]))
    return pd.DataFrame(rows)


def _fitted_axes(res: pd.DataFrame) -> Tuple[str, ...]:
    """The axes an arm fitted, read from the `param_names` column the fit
    wrote ('Cm|Rm|Ra|gbar|...'), falling back to whichever known axis columns
    are present. Reading them off the result is what keeps a 3-D arm and a
    6-D arm in one table without either being assumed."""
    if "param_names" in res.columns and len(res):
        v = res["param_names"].dropna()
        if len(v):
            return tuple(str(v.iloc[0]).split("|"))
    return tuple(a for a in TRUTH_COLUMN if a in res.columns)


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ===========================================================================
#  Summaries
# ===========================================================================
def recovery_summary(long_df: pd.DataFrame, *,
                     drop_railed: bool = False,
                     include_fp: bool = False) -> pd.DataFrame:
    """Per (arm, parameter): the median absolute error and its spread.

    `include_fp=False` (the default) excludes the false-positive-control
    cells, for the reason in the module docstring: on those cells the I_h
    axes have no identifiable truth to recover.
    """
    if long_df.empty:
        return pd.DataFrame(columns=["arm", "parameter", "n", "n_railed",
                                     "median_abs_err", "q25_abs_err",
                                     "q75_abs_err", "median_err", "err_kind"])
    d = long_df if include_fp else long_df[~long_df["is_fp_control"]]
    n_railed = (d.groupby(["arm", "parameter"])["rail"].sum()
                .rename("n_railed").reset_index())
    if drop_railed:
        d = d[~d["rail"]]
    out = []
    for (arm, par), g in d.groupby(["arm", "parameter"], sort=True):
        e = g["abs_err"].to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        s = g["err"].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        out.append(dict(
            arm=arm, parameter=par, n=int(e.size),
            median_abs_err=float(np.median(e)) if e.size else np.nan,
            q25_abs_err=float(np.percentile(e, 25)) if e.size else np.nan,
            q75_abs_err=float(np.percentile(e, 75)) if e.size else np.nan,
            # the SIGNED median: a bias shows here and not in the absolute one
            median_err=float(np.median(s)) if s.size else np.nan,
            err_kind=str(g["err_kind"].iloc[0]),
        ))
    res = pd.DataFrame(out)
    return res.merge(n_railed, on=["arm", "parameter"], how="left")


def cm_inflation(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per arm, the C_m bias on the cells that DO carry I_h.

    This is the quantity the whole project turns on: if the passive arms
    inflate C_m on I_h-bearing cells and the I_h arm does not, the railing at
    the C_m bound has an explanation. Reported as the median signed log ratio
    (R.1) and as the equivalent multiplicative factor exp(median), which is
    what a reader wants to see ("1.4x too large").
    """
    d = long_df[(long_df["parameter"] == "Cm") & (~long_df["is_fp_control"])]
    out = []
    for arm, g in d.groupby("arm", sort=True):
        s = g["err"].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        out.append(dict(
            arm=arm, n=int(s.size),
            median_log_ratio=float(np.median(s)) if s.size else np.nan,
            median_factor=float(np.exp(np.median(s))) if s.size else np.nan,
            q25_factor=float(np.exp(np.percentile(s, 25))) if s.size else np.nan,
            q75_factor=float(np.exp(np.percentile(s, 75))) if s.size else np.nan,
            n_railed_cm=int(g["rail"].sum()),
        ))
    return pd.DataFrame(out)


def fp_control_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Per arm, what happened on the cells with NO I_h.

    Two numbers, and they are not the same question:
      low_rail_frac  the fraction of FP cells whose gbar estimate sits on the
                     LOW bound, i.e. the fit correctly reported "no I_h here".
      median_cm_abs_log_ratio
                     |log(Cm_hat/Cm)| on those same cells. A fit that invents
                     an I_h on a passive cell pays for it in C_m, so this is
                     where the invention becomes visible.
    """
    d = long_df[long_df["is_fp_control"]]
    out = []
    for arm, g in d.groupby("arm", sort=True):
        gb = g[g["parameter"] == "gbar"]
        cm = g[g["parameter"] == "Cm"]
        n_fp = int(gb["specimen_id"].nunique()) or int(cm["specimen_id"].nunique())
        low = int((gb["rail_side"] == "lo").sum())
        hi = int((gb["rail_side"] == "hi").sum())
        e = cm["abs_err"].to_numpy(dtype=float)
        e = e[np.isfinite(e)]
        out.append(dict(
            arm=arm, n_fp_cells=n_fp,
            n_gbar_low_rail=low, n_gbar_high_rail=hi,
            low_rail_frac=(low / n_fp) if n_fp else np.nan,
            median_cm_abs_log_ratio=float(np.median(e)) if e.size else np.nan,
            median_cm_factor=(float(np.exp(np.median(
                cm["err"].dropna().to_numpy(dtype=float))))
                if len(cm["err"].dropna()) else np.nan),
        ))
    return pd.DataFrame(out)


# ===========================================================================
#  The gate
# ===========================================================================
def evaluate_gate(summary: pd.DataFrame,
                  fp: pd.DataFrame,
                  *,
                  arm: str = "ih6",
                  tol: RecoveryTolerances = RecoveryTolerances(),
                  ) -> Tuple[pd.DataFrame, bool]:
    """Apply the Stage-7 tolerances to ONE arm and return (verdict table, pass).

    The gate is on `arm` (the six-parameter fit) because that is what Stage 8
    would run; the other arms are reported and not gated. An axis with no
    tolerance (R_m, R_a) appears in the table with verdict "not gated". An
    axis the arm did not fit does not appear at all, and a REQUIRED axis that
    is missing fails the gate -- silence is not a pass.
    """
    rows: List[dict] = []
    ok_all = True
    sub = summary[summary["arm"] == arm]
    for axis in ("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau"):
        t = tol.for_axis(axis)
        g = sub[sub["parameter"] == axis]
        if g.empty:
            if t is not None:
                rows.append(dict(arm=arm, parameter=axis, n=0,
                                 median_abs_err=np.nan, tolerance=t,
                                 verdict="FAIL (axis absent)"))
                ok_all = False
            continue
        m = float(g["median_abs_err"].iloc[0])
        n = int(g["n"].iloc[0])
        if t is None:
            rows.append(dict(arm=arm, parameter=axis, n=n, median_abs_err=m,
                             tolerance=np.nan, verdict="not gated"))
            continue
        passed = bool(np.isfinite(m) and m <= t)
        ok_all = ok_all and passed
        rows.append(dict(arm=arm, parameter=axis, n=n, median_abs_err=m,
                         tolerance=float(t),
                         verdict="pass" if passed else "FAIL"))

    # the false-positive control, when the cohort carries one
    f = fp[fp["arm"] == arm] if len(fp) else fp
    if len(f):
        frac = float(f["low_rail_frac"].iloc[0])
        passed = bool(np.isfinite(frac) and frac >= tol.fp_min_low_rail_frac)
        ok_all = ok_all and passed
        rows.append(dict(arm=arm, parameter="gbar (FP control, low rail)",
                         n=int(f["n_fp_cells"].iloc[0]), median_abs_err=frac,
                         tolerance=float(tol.fp_min_low_rail_frac),
                         verdict="pass" if passed else "FAIL"))
        cmerr = float(f["median_cm_abs_log_ratio"].iloc[0])
        passed = bool(np.isfinite(cmerr) and cmerr <= tol.fp_log_ratio_cm)
        ok_all = ok_all and passed
        rows.append(dict(arm=arm, parameter="Cm (FP control)",
                         n=int(f["n_fp_cells"].iloc[0]), median_abs_err=cmerr,
                         tolerance=float(tol.fp_log_ratio_cm),
                         verdict="pass" if passed else "FAIL"))
    return pd.DataFrame(rows), bool(ok_all)


def frozen_axes_on_failure(verdict: pd.DataFrame) -> List[str]:
    """The axes D-005 says to FREEZE at their base value for the campaign:
    the kinetic knobs that failed their tolerance. C_m and gbar are not
    freezable -- if either fails, the six-parameter fit is not usable and the
    answer is not a smaller spec but a different protocol."""
    bad = verdict[verdict["verdict"].astype(str).str.startswith("FAIL")]
    return [a for a in ("dv_h", "kappa_tau")
            if a in set(bad["parameter"].astype(str))]


# ===========================================================================
#  Report I/O
# ===========================================================================
def write_report(out_dir: Union[Path, str], *,
                 long_df: pd.DataFrame, summary: pd.DataFrame,
                 inflation: pd.DataFrame, fp: pd.DataFrame,
                 verdict: pd.DataFrame, passed: bool,
                 tol: RecoveryTolerances) -> Dict[str, Path]:
    """Write every table, plus a one-line verdict file a job script can read."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_cell": out / "recovery_per_cell.csv",
        "summary": out / "recovery_summary.csv",
        "inflation": out / "cm_inflation.csv",
        "fp": out / "fp_control.csv",
        "verdict": out / "gate_verdict.csv",
    }
    long_df.to_csv(paths["per_cell"], index=False)
    summary.to_csv(paths["summary"], index=False)
    inflation.to_csv(paths["inflation"], index=False)
    fp.to_csv(paths["fp"], index=False)
    verdict.to_csv(paths["verdict"], index=False)
    (out / "gate_verdict.txt").write_text(
        "STAGE 7 GATE: {}\ntolerances: {}\n".format(
            "PASS" if passed else "FAIL",
            ", ".join("{}={:.4g}".format(k, v)
                      for k, v in tol.as_dict().items())))
    paths["verdict_txt"] = out / "gate_verdict.txt"
    return paths


def format_report(summary: pd.DataFrame, inflation: pd.DataFrame,
                  fp: pd.DataFrame, verdict: pd.DataFrame,
                  passed: bool) -> str:
    """The human-readable block printed at the end of a run (and into the PBS
    log, which is where anyone will actually read it)."""
    L = ["", "=" * 72, "  STAGE 7 -- SYNTHETIC RECOVERY", "=" * 72, ""]
    L.append("Recovery (median |error| over cells; log ratio, except dv_h in mV)")
    L.append(summary.to_string(index=False, float_format=lambda v: "%.4f" % v))
    L.append("")
    L.append("C_m bias on the I_h-bearing cells (factor = exp(median log ratio))")
    L.append(inflation.to_string(index=False, float_format=lambda v: "%.4f" % v))
    if len(fp):
        L.append("")
        L.append("False-positive control (cells with NO I_h)")
        L.append(fp.to_string(index=False, float_format=lambda v: "%.4f" % v))
    L.append("")
    L.append("GATE")
    L.append(verdict.to_string(index=False, float_format=lambda v: "%.4f" % v))
    L.append("")
    L.append("  VERDICT: {}".format("PASS" if passed else "FAIL"))
    if not passed:
        frozen = frozen_axes_on_failure(verdict)
        if frozen:
            L.append("  D-005: freeze {} at the base value for the campaign "
                     "and record the failure (--fit-params without it)."
                     .format(", ".join(frozen)))
        else:
            L.append("  No freezable axis failed. A C_m or gbar failure is not "
                     "fixed by a smaller spec -- the protocol is what has to "
                     "change (plan section 7's sub-window weighting is the "
                     "first thing to try).")
    L.append("=" * 72)
    return "\n".join(L)


# ===========================================================================
#  Plotting (separate from every statistic above; never called by them)
# ===========================================================================
def plot_recovery(long_df: pd.DataFrame, out_path: Union[Path, str], *,
                  tol: RecoveryTolerances = RecoveryTolerances()) -> Path:
    """One panel per fitted axis: estimate vs truth, on the axis's own scale,
    with the tolerance band. Cells of the false-positive control are drawn
    with an open marker, because they are a different question."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axes = [a for a in ("Cm", "Rm", "Ra", "gbar", "dv_h", "kappa_tau")
            if a in set(long_df["parameter"])]
    arms = sorted(long_df["arm"].unique())
    fig, axs = plt.subplots(len(arms), len(axes),
                            figsize=(3.0 * len(axes), 2.8 * len(arms)),
                            squeeze=False)
    for i, arm in enumerate(arms):
        for j, axis in enumerate(axes):
            ax = axs[i][j]
            g = long_df[(long_df["arm"] == arm) & (long_df["parameter"] == axis)]
            for fp_flag, marker, face in ((False, "o", None), (True, "o", "none")):
                gg = g[g["is_fp_control"] == fp_flag]
                if not len(gg):
                    continue
                ax.scatter(gg["truth"], gg["estimate"], s=18, marker=marker,
                           facecolors=face, edgecolors="C0" if not fp_flag else "C3",
                           linewidths=1.0, alpha=0.85)
            fin = g[["truth", "estimate"]].to_numpy(dtype=float)
            fin = fin[np.all(np.isfinite(fin), axis=1)]
            if fin.size:
                lo, hi = float(np.min(fin)), float(np.max(fin))
                pad = 0.05 * (hi - lo) if hi > lo else 1.0
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                        "k--", lw=0.8, zorder=0)
            if axis in LOG_AXES:
                ax.set_xscale("log"); ax.set_yscale("log")
            if i == 0:
                ax.set_title(axis, fontsize=10)
            if j == 0:
                ax.set_ylabel("{}\nestimate".format(arm), fontsize=9)
            if i == len(arms) - 1:
                ax.set_xlabel("truth", fontsize=9)
            ax.tick_params(labelsize=7)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
