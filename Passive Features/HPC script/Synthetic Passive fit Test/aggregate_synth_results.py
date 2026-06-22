# -*- coding: utf-8 -*-
"""
aggregate_synth_results.py
==========================

Phase 3 (reporting) of the da Vinci synthetic passive-fit benchmark.

Ingest every per-cohort output under OUTPUT_ROOT, join recovered parameters back
onto the seeded manifest (the injected ground truth), write a single
``benchmark_summary.csv``, and render TWO figure sets from the same numbers:

    figures/paper/   vector PDF + 300 dpi PNG, compact, small fonts  (publication)
    figures/talk/    large fonts, high contrast, 1 idea / figure     (slides)

Pure pandas / matplotlib (headless Agg). The join + coverage maths
(``build_summary`` / ``build_coverage``) are unit-tested without NEURON or
matplotlib in smoke_aggregate_synth_results.py.

Notation (units carried; the 1e-3 in tau_m is explicit)
-------------------------------------------------------
    tau_m = Rm * Cm * 1e-3   [ms]
    <param>_ratio = recovered / injected           (1.0 == perfect)
    <param>_dex   = log10(<param>_ratio)            (0.0 == perfect; +-bias in dex)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless: cluster + CI safe
import matplotlib.pyplot as plt  # noqa: E402


# ===========================================================================
#  Palette + styles (Okabe-Ito; colourblind-safe)
# ===========================================================================
PARAM_COLOR: Dict[str, str] = {
    "Cm": "#0072B2", "Rm": "#D55E00", "Ra": "#009E73",
    "tau_m": "#CC79A7", "Rin": "#56B4E9",
}
STATUS_COLOR = {"good": "#009E73", "to_refine": "#E69F00", "failed": "#D55E00"}

_PAPER_RC = {
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight", "lines.linewidth": 1.2,
}
_TALK_RC = {
    "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.bbox": "tight", "lines.linewidth": 2.4,
}
STYLES = {
    "paper": dict(rc=_PAPER_RC, formats=("pdf", "png"), dpi=300, scale=1.0),
    "talk":  dict(rc=_TALK_RC,  formats=("png",),       dpi=200, scale=1.35),
}


def _save(fig, name: str, style: str, out_dir: Path) -> None:
    cfg = STYLES[style]
    d = out_dir / "figures" / style
    d.mkdir(parents=True, exist_ok=True)
    for ext in cfg["formats"]:
        fig.savefig(d / f"{name}.{ext}", dpi=cfg["dpi"])
    plt.close(fig)


def _fs(style: str, w: float, h: float) -> Tuple[float, float]:
    s = STYLES[style]["scale"]
    return (w * s, h * s)


def _safe_bins(v, target: int = 15):
    """Histogram bins that never raise on constant / tiny-range data."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:                       # all equal -> a valid tiny range
        return np.linspace(lo - 0.5, hi + 0.5, 4)
    return int(min(target, max(3, v.size // 2 or 3)))


# ===========================================================================
#  1. Loading per-cohort outputs
# ===========================================================================
_RECOVERED_PREFERENCE = ("phase2p5_combined_results.csv", "phase2_results.csv")


def load_recovered(output_root: Path | str) -> pd.DataFrame:
    """Concatenate the best-available per-cohort recovered table (prefers the
    post-Phase-2.5 combined file), tagging each row with its cohort dir name and
    merging tau_w_choice.csv when present."""
    output_root = Path(output_root)
    frames: List[pd.DataFrame] = []
    for cohort_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        chosen = next((cohort_dir / f for f in _RECOVERED_PREFERENCE
                       if (cohort_dir / f).exists()), None)
        if chosen is None:
            print(f"[agg][WARN] no recovered CSV in {cohort_dir.name}; skipped.")
            continue
        df = pd.read_csv(chosen)
        df["cohort_dir"] = cohort_dir.name
        df["from_phase2p5"] = (chosen.name == "phase2p5_combined_results.csv")
        tw = cohort_dir / "tau_w_choice.csv"
        if tw.exists() and "tau_w_chosen_ms" not in df.columns:
            df = df.merge(pd.read_csv(tw)[["specimen_id", "tau_w_chosen_ms"]],
                          on="specimen_id", how="left")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no per-cohort recovered CSVs under {output_root}")
    return pd.concat(frames, ignore_index=True)


def load_phase3(output_root: Path | str) -> pd.DataFrame:
    """Concatenate phase3_full_summary.csv (long: one row per cell x parameter).
    Empty frame if Phase 3 was not run anywhere."""
    output_root = Path(output_root)
    frames = [pd.read_csv(p) for p in sorted(output_root.glob("*/phase3_full_summary.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_generated_meta(archive_root: Optional[Path | str]) -> pd.DataFrame:
    """Optional: per-cell measured rin/sag from generation (archive side)."""
    if archive_root is None:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in sorted(Path(archive_root).glob("*/generated_meta.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ===========================================================================
#  2. Join + derived quantities  (PURE; unit-tested)
# ===========================================================================
_GT_COLS = ["specimen_id", "cohort", "morph_name", "cm_true", "rm_true",
            "ra_true", "tau_m_true_ms", "ih_gihbar_S_cm2", "noise_sigma_mV",
            "use_ih", "ra_mode"]


def build_summary(manifest: pd.DataFrame, recovered: pd.DataFrame,
                  gen_meta: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Inner-join injected truth (manifest) with recovered params, compute
    recovery ratios and dex errors (post-2.5; and pre-2.5 if *_phase2 present)."""
    gt = manifest[[c for c in _GT_COLS if c in manifest.columns]].copy()
    # manifest is authoritative for GT columns: drop any same-named columns from
    # the recovered side so the merge never produces _x/_y suffixes.
    overlap = [c for c in recovered.columns
               if c in (set(_GT_COLS) - {"specimen_id"})]
    rec = recovered.drop(columns=overlap)
    s = gt.merge(rec, on="specimen_id", how="inner")
    if s.empty:
        raise ValueError("summary join matched 0 cells (manifest vs recovered "
                         "specimen_ids disjoint).")
    if gen_meta is not None and not gen_meta.empty:
        s = s.merge(gen_meta[[c for c in ("specimen_id", "rin_MOhm_true",
                                          "sag_ratio_true") if c in gen_meta.columns]],
                    on="specimen_id", how="left")

    s["tau_fit_ms"] = s["rm_Ohm_cm2"] * s["cm_uF_per_cm2"] * 1e-3   # Ohm*uF -> ms
    for p, rec, tru in (("cm", "cm_uF_per_cm2", "cm_true"),
                        ("rm", "rm_Ohm_cm2", "rm_true"),
                        ("ra", "ra_Ohm_cm", "ra_true")):
        s[f"{p}_ratio"] = s[rec] / s[tru]
        s[f"{p}_dex"] = np.log10(s[f"{p}_ratio"])
    s["tau_ratio"] = s["tau_fit_ms"] / s["tau_m_true_ms"]
    s["tau_dex"] = np.log10(s["tau_ratio"])

    # pre-Phase-2.5 recovery (stashed by the monolith as *_phase2), if present
    if {"cm_phase2", "rm_phase2"} <= set(s.columns):
        s["tau_fit_phase2_ms"] = s["rm_phase2"] * s["cm_phase2"] * 1e-3
        s["cm_ratio_phase2"] = s["cm_phase2"] / s["cm_true"]
        s["rm_ratio_phase2"] = s["rm_phase2"] / s["rm_true"]
        if "ra_phase2" in s.columns:
            s["ra_ratio_phase2"] = s["ra_phase2"] / s["ra_true"]
        s["tau_ratio_phase2"] = s["tau_fit_phase2_ms"] / s["tau_m_true_ms"]
    return s


def build_coverage(phase3: pd.DataFrame, manifest: pd.DataFrame
                   ) -> pd.DataFrame:
    """Per (cell, parameter): does the bootstrap CI contain the injected value?

    Returns long df with columns specimen_id, parameter, true, mle,
    covered_bca, covered_perc. Ra rows are kept but flagged degenerate when the
    CI has zero width (fix_ra bootstrap pins Ra)."""
    if phase3.empty:
        return pd.DataFrame(columns=["specimen_id", "parameter", "true", "mle",
                                     "covered_bca", "covered_perc", "degenerate"])
    truth = manifest[["specimen_id", "cm_true", "rm_true", "ra_true"]].copy()
    long = phase3.merge(truth, on="specimen_id", how="left")
    true_map = {"Cm": "cm_true", "Rm": "rm_true", "Ra": "ra_true"}
    long["true"] = long.apply(lambda r: r.get(true_map.get(r["parameter"], ""), np.nan),
                              axis=1)

    def _cov(lo, hi, t):
        return bool(np.isfinite(lo) and np.isfinite(hi) and lo <= t <= hi)

    rows = []
    for _, r in long.iterrows():
        bca = (r.get("ci_bca_lo", np.nan), r.get("ci_bca_hi", np.nan))
        perc = (r.get("ci_perc_lo", np.nan), r.get("ci_perc_hi", np.nan))
        rows.append(dict(
            specimen_id=int(r["specimen_id"]), parameter=str(r["parameter"]),
            true=float(r["true"]), mle=float(r.get("mle", np.nan)),
            covered_bca=_cov(bca[0], bca[1], r["true"]),
            covered_perc=_cov(perc[0], perc[1], r["true"]),
            degenerate=bool(np.isfinite(bca[0]) and np.isfinite(bca[1])
                            and abs(bca[1] - bca[0]) < 1e-9),
        ))
    return pd.DataFrame(rows)


# ===========================================================================
#  3. Figures (each guarded; skips gracefully on missing data)
# ===========================================================================
def _annot_median(ax, vals, x, color):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size:
        ax.text(x, 1.02, f"med {np.median(vals):.2f}", ha="center",
                va="bottom", fontsize="x-small", color=color,
                transform=ax.get_xaxis_transform())


def fig_recovery_distributions(s: pd.DataFrame, style: str, out_dir: Path) -> None:
    params = [("cm_ratio", "Cm"), ("rm_ratio", "Rm"), ("ra_ratio", "Ra"),
              ("tau_ratio", "tau_m")]
    with plt.rc_context(STYLES[style]["rc"]):
        fig, ax = plt.subplots(figsize=_fs(style, 6.4, 3.0))
        data, labels, colors = [], [], []
        for i, (col, name) in enumerate(params):
            v = s[col].replace([np.inf, -np.inf], np.nan).dropna().values
            if v.size == 0:
                continue
            data.append(v); labels.append(name); colors.append(PARAM_COLOR[name])
            _annot_median(ax, v, i + 1, PARAM_COLOR[name])
        if not data:
            plt.close(fig); return
        parts = ax.violinplot(data, showmedians=True, widths=0.8)
        for b, c in zip(parts["bodies"], colors):
            b.set_facecolor(c); b.set_alpha(0.45); b.set_edgecolor(c)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in parts:
                parts[key].set_color("0.3")
        ax.axhline(1.0, color="k", lw=0.8, ls="--", zorder=0)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels([r"$C_m$", r"$R_m$", r"$R_a$", r"$\tau_m$"][:len(labels)])
        ax.set_ylabel("recovered / injected")
        ax.set_title(f"Parameter recovery (n = {len(s)} cells)")
        _save(fig, "recovery_distributions", style, out_dir)


def fig_bias_vs_injected(s: pd.DataFrame, style: str, out_dir: Path) -> None:
    specs = [("cm_true", "cm_dex", "Cm", r"injected $C_m$ [$\mu$F/cm$^2$]"),
             ("rm_true", "rm_dex", "Rm", r"injected $R_m$ [$\Omega\,$cm$^2$]"),
             ("ra_true", "ra_dex", "Ra", r"injected $R_a$ [$\Omega\,$cm]")]
    with plt.rc_context(STYLES[style]["rc"]):
        fig, axes = plt.subplots(1, 3, figsize=_fs(style, 9.5, 3.0))
        for ax, (xc, yc, name, xlab) in zip(axes, specs):
            d = s[[xc, yc]].replace([np.inf, -np.inf], np.nan).dropna()
            if d.empty:
                ax.set_visible(False); continue
            ax.scatter(d[xc], d[yc], s=14, alpha=0.5, color=PARAM_COLOR[name],
                       edgecolor="none")
            # binned-median trend (no statsmodels dependency)
            x = np.log10(d[xc].values); y = d[yc].values
            bins = np.linspace(x.min(), x.max(), 7)
            idx = np.digitize(x, bins)
            bx, by = [], []
            for b in range(1, len(bins)):
                m = idx == b
                if m.sum() >= 3:
                    bx.append(10 ** np.mean(x[m])); by.append(np.median(y[m]))
            if bx:
                ax.plot(bx, by, "-o", color="0.15", ms=3, lw=1.4)
            ax.axhline(0.0, color="k", lw=0.8, ls="--")
            ax.set_xscale("log"); ax.set_xlabel(xlab)
            ax.set_title(name)
        axes[0].set_ylabel(r"recovery bias  $\log_{10}$(rec / inj) [dex]")
        fig.suptitle("Estimator bias across the search box", y=1.02)
        _save(fig, "bias_vs_injected", style, out_dir)


def fig_iso_tau_scatter(s: pd.DataFrame, style: str, out_dir: Path) -> None:
    d = s[["cm_uF_per_cm2", "rm_Ohm_cm2", "tau_ratio"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3:
        return
    lx, ly = np.log10(d["cm_uF_per_cm2"]), np.log10(d["rm_Ohm_cm2"])
    r = float(np.corrcoef(lx, ly)[0, 1])
    with plt.rc_context(STYLES[style]["rc"]):
        fig, ax = plt.subplots(figsize=_fs(style, 4.6, 4.0))
        sc = ax.scatter(d["cm_uF_per_cm2"], d["rm_Ohm_cm2"], s=22,
                        c=d["tau_ratio"], cmap="coolwarm", vmin=0.6, vmax=1.4,
                        edgecolor="0.3", linewidth=0.3)
        # iso-tau_m guide lines (log Rm = log tau - log Cm => slope -1)
        cm_line = np.array([d["cm_uF_per_cm2"].min(), d["cm_uF_per_cm2"].max()])
        for tau in (5.0, 15.0, 45.0):  # ms
            ax.plot(cm_line, tau / (cm_line * 1e-3), color="0.6", lw=0.7, ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"recovered $C_m$ [$\mu$F/cm$^2$]")
        ax.set_ylabel(r"recovered $R_m$ [$\Omega\,$cm$^2$]")
        ax.set_title(rf"Iso-$\tau_m$ valley   corr$(\log\hat C_m,\log\hat R_m)$ = {r:.3f}")
        fig.colorbar(sc, ax=ax, label=r"$\tau_m$ ratio")
        _save(fig, "iso_tau_identifiability", style, out_dir)


def fig_recovered_vs_injected(s: pd.DataFrame, style: str, out_dir: Path) -> None:
    specs = [("cm_true", "cm_uF_per_cm2", "Cm"),
             ("rm_true", "rm_Ohm_cm2", "Rm"),
             ("tau_m_true_ms", "tau_fit_ms", "tau_m")]
    with plt.rc_context(STYLES[style]["rc"]):
        fig, axes = plt.subplots(1, 3, figsize=_fs(style, 9.5, 3.2))
        for ax, (xc, yc, name) in zip(axes, specs):
            d = s[[xc, yc]].replace([np.inf, -np.inf], np.nan).dropna()
            if d.empty:
                ax.set_visible(False); continue
            lo = min(d[xc].min(), d[yc].min()); hi = max(d[xc].max(), d[yc].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=0)
            ax.scatter(d[xc], d[yc], s=16, alpha=0.6, color=PARAM_COLOR[name],
                       edgecolor="none")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(f"injected {name}"); ax.set_ylabel(f"recovered {name}")
            ax.set_title(name)
        fig.suptitle("Recovered vs injected (identity = perfect)", y=1.02)
        _save(fig, "recovered_vs_injected", style, out_dir)


def fig_tau_w(s: pd.DataFrame, style: str, out_dir: Path) -> None:
    if "tau_w_chosen_ms" not in s.columns or s["tau_w_chosen_ms"].dropna().empty:
        return
    with plt.rc_context(STYLES[style]["rc"]):
        has_sag = "sag_ratio_true" in s.columns and s["sag_ratio_true"].notna().any()
        fig, axes = plt.subplots(1, 2 if has_sag else 1,
                                 figsize=_fs(style, 7.5 if has_sag else 4.0, 3.0),
                                 squeeze=False)
        ax = axes[0][0]
        tw = s["tau_w_chosen_ms"].dropna()
        ax.hist(tw, bins=_safe_bins(tw, target=max(3, tw.nunique())), color=PARAM_COLOR["Cm"], alpha=0.8,
                edgecolor="white")
        ax.set_xlabel(r"chosen $\tau_w$ [ms]"); ax.set_ylabel("cells")
        ax.set_title(r"Per-cell $\tau_w$ selection")
        if has_sag:
            ax2 = axes[0][1]
            d = s[["sag_ratio_true", "tau_w_chosen_ms"]].dropna()
            ax2.scatter(d["sag_ratio_true"], d["tau_w_chosen_ms"], s=18,
                        alpha=0.6, color=PARAM_COLOR["Ra"], edgecolor="none")
            ax2.set_xlabel("injected sag ratio")
            ax2.set_ylabel(r"chosen $\tau_w$ [ms]")
            ax2.set_title(r"$\tau_w$ vs sag")
        _save(fig, "tau_w_selection", style, out_dir)


def fig_phase2p5_effect(s: pd.DataFrame, style: str, out_dir: Path) -> None:
    if "ra_ratio_phase2" not in s.columns:
        return
    with plt.rc_context(STYLES[style]["rc"]):
        fig, axes = plt.subplots(1, 2, figsize=_fs(style, 7.5, 3.2))
        # (a) Ra: free per-cell (pre-2.5) vs cohort-fixed (post-2.5)
        ax = axes[0]
        coh = s.groupby("cohort").agg(
            ra_true=("ra_true", "first"), ra_fixed=("ra_Ohm_cm", "first")).dropna()
        if not coh.empty:
            lo, hi = coh.min().min(), coh.max().max()
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.scatter(coh["ra_true"], coh["ra_fixed"], s=30, color=PARAM_COLOR["Ra"])
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"injected cohort $R_a$ [$\Omega\,$cm]")
            ax.set_ylabel(r"Phase 2.5 fixed $R_a$ [$\Omega\,$cm]")
            ax.set_title("Cohort $R_a$ recovery")
        # (b) tau_m bias before vs after 2.5
        ax2 = axes[1]
        for col, lab, c in (("tau_ratio_phase2", "pre-2.5", "0.6"),
                            ("tau_ratio", "post-2.5", PARAM_COLOR["tau_m"])):
            if col in s.columns:
                v = s[col].replace([np.inf, -np.inf], np.nan).dropna()
                ax2.hist(v, bins=_safe_bins(v), histtype="step", lw=1.8, label=lab,
                         color=c)
        ax2.axvline(1.0, color="k", lw=0.8, ls="--")
        ax2.set_xlabel(r"$\tau_m$ ratio"); ax2.set_ylabel("cells")
        ax2.set_title(r"$\tau_m$ recovery: Phase 2.5 effect"); ax2.legend()
        _save(fig, "phase2p5_effect", style, out_dir)


def fig_validation_health(s: pd.DataFrame, style: str, out_dir: Path) -> None:
    if "validation_status" not in s.columns:
        return
    with plt.rc_context(STYLES[style]["rc"]):
        fig, axes = plt.subplots(1, 2, figsize=_fs(style, 7.5, 3.0))
        order = ["good", "to_refine", "failed"]
        counts = s["validation_status"].value_counts().reindex(order).fillna(0)
        axes[0].bar(order, counts.values,
                    color=[STATUS_COLOR[o] for o in order], alpha=0.85)
        axes[0].set_ylabel("cells"); axes[0].set_title("Validation status")
        # valid/train ratio vs injected gIhbar (contamination axis)
        ax = axes[1]
        if {"valid_to_train_ratio", "ih_gihbar_S_cm2"} <= set(s.columns):
            d = s[["ih_gihbar_S_cm2", "valid_to_train_ratio", "validation_status"]].dropna()
            for st in order:
                dd = d[d["validation_status"] == st]
                if not dd.empty:
                    ax.scatter(dd["ih_gihbar_S_cm2"], dd["valid_to_train_ratio"],
                               s=16, alpha=0.6, label=st, color=STATUS_COLOR[st],
                               edgecolor="none")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"injected $\bar g_{Ih}$ [S/cm$^2$]")
            ax.set_ylabel("valid / train RMSD ratio")
            ax.set_title("Contamination vs fit health"); ax.legend()
        _save(fig, "validation_health", style, out_dir)


def fig_ci_calibration(coverage: pd.DataFrame, style: str, out_dir: Path,
                       nominal: float = 0.95) -> None:
    if coverage.empty:
        return
    cov = coverage[~coverage["degenerate"]]      # drop fix_ra-pinned Ra
    if cov.empty:
        return
    with plt.rc_context(STYLES[style]["rc"]):
        fig, ax = plt.subplots(figsize=_fs(style, 4.6, 3.2))
        params = [p for p in ("Cm", "Rm", "Ra") if p in cov["parameter"].unique()]
        x = np.arange(len(params)); w = 0.38
        for off, col, lab in ((-w / 2, "covered_bca", "BCa"),
                              (+w / 2, "covered_perc", "percentile")):
            vals = [cov[cov["parameter"] == p][col].mean() for p in params]
            ns = [int((cov["parameter"] == p).sum()) for p in params]
            bars = ax.bar(x + off, vals, w, label=lab, alpha=0.85)
            for b, n in zip(bars, ns):
                ax.text(b.get_x() + b.get_width() / 2, 0.02, f"n={n}",
                        ha="center", va="bottom", fontsize="x-small")
        ax.axhline(nominal, color="k", lw=1.0, ls="--",
                   label=f"nominal {nominal:.0%}")
        ax.set_xticks(x); ax.set_xticklabels(params)
        ax.set_ylim(0, 1.05); ax.set_ylabel("empirical CI coverage")
        ax.set_title("Bootstrap CI calibration (subset)"); ax.legend()
        _save(fig, "ci_calibration", style, out_dir)


def render_all(summary: pd.DataFrame, coverage: pd.DataFrame, out_dir: Path,
               styles: Sequence[str] = ("paper", "talk")) -> None:
    figs = (fig_recovery_distributions, fig_bias_vs_injected,
            fig_iso_tau_scatter, fig_recovered_vs_injected, fig_tau_w,
            fig_phase2p5_effect, fig_validation_health)
    for style in styles:
        for fn in figs:
            try:
                fn(summary, style, out_dir)
            except Exception as exc:  # noqa: BLE001 — one bad fig never kills the rest
                print(f"[agg][WARN] {fn.__name__} ({style}) failed: "
                      f"{type(exc).__name__}: {exc}")
        try:
            fig_ci_calibration(coverage, style, out_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[agg][WARN] fig_ci_calibration ({style}) failed: {exc}")
    print(f"[agg] figures -> {out_dir / 'figures'}/{{paper,talk}}/")


# ===========================================================================
#  4. CLI
# ===========================================================================
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Aggregate synthetic-benchmark results + figures.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-root", required=True, help="root of per-cohort fit outputs")
    ap.add_argument("--archive-root", default=None,
                    help="optional: per-cohort archives (for generated sag/rin)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--styles", default="paper,talk")
    ap.add_argument("--code-dir", default=None,
                    help="dir holding synth_gt_grid.py (for load_manifest dtypes)")
    args = ap.parse_args(argv)

    if args.code_dir:
        import sys
        sys.path.insert(0, args.code_dir)
    try:
        from synth_gt_grid import load_manifest
        manifest = load_manifest(args.manifest)
    except Exception:
        manifest = pd.read_csv(args.manifest)   # fallback: plain read

    recovered = load_recovered(args.output_root)
    phase3 = load_phase3(args.output_root)
    gen_meta = load_generated_meta(args.archive_root)

    summary = build_summary(manifest, recovered, gen_meta)
    coverage = build_coverage(phase3, manifest)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "benchmark_summary.csv", index=False)
    if not coverage.empty:
        coverage.to_csv(out / "phase3_coverage.csv", index=False)
    print(f"[agg] benchmark_summary.csv: {len(summary)} cells -> {out}")
    # headline numbers
    for col, name in (("cm_ratio", "Cm"), ("rm_ratio", "Rm"),
                      ("ra_ratio", "Ra"), ("tau_ratio", "tau_m")):
        v = summary[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(v):
            print(f"      {name:5s} ratio: median={v.median():.3f}  "
                  f"IQR=[{v.quantile(.25):.3f}, {v.quantile(.75):.3f}]")

    render_all(summary, coverage, out,
               styles=tuple(x for x in args.styles.split(",") if x in STYLES))


if __name__ == "__main__":
    main()
