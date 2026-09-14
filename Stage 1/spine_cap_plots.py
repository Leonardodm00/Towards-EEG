"""spine_cap_plots -- figures for the distal tip cap (spine_cap).

Towards-EEG stage S1.3 (data-generation side; runs in Colab, not on the HPC).

Follows the same contract as s1_plots and alignment_plots: every function here
takes frames or profile dicts and RETURNS a matplotlib Figure. Nothing writes a
file, reads a file, or computes a scientific quantity. Saving is
alignment_plots.save_figure's job; the geometry, the leaf gate and the meridian
profiles all come from spine_cap, so there is exactly one implementation of
each and no chance of a plot silently diverging from the numbers.

WHICH FIGURES EARN THEIR PLACE, AND WHY
---------------------------------------
  spine_gallery         The one that shows what the model actually IS. Each
                        panel is the meridian outline of one reconstructed
                        spine: the frustum chain drawn from the real node
                        radii, with the cap shaded. If the reconstruction is
                        wrong -- necks that are wider than heads, two-node
                        spines, radii pinned at the 50 nm fallback -- it is
                        visible here and nowhere else. Read this before
                        trusting any aggregate on this page.

  tip_radius_distribution
                        r_t is the ONLY free input to the cap: A_cap depends on
                        (r_t, h) and nothing else. This is its distribution,
                        spine tips against shaft tips, with the fallback radius
                        and h marked. A spike at the fallback means those caps
                        are an artefact of a missing radius, not a measurement.

  cap_area_curve        A_cap(r_t) for several h, with the empirical r_t
                        distribution underneath. Shows how much of the answer
                        is the choice of h and how much is the data, and where
                        the flat-disc lower bound sits.

  cap_contribution      Per-spine A_cap / A_spine. The distribution of the
                        correction itself. A long right tail means a minority
                        of short, blunt spines are absorbing the change.

  f_bracket             The decision figure. F per cell under the three modes
                        (no cap / flat disc / spherical cap) against the
                        published human bands. This is what the whole exercise
                        was for, and it is reported as a BRACKET because the
                        cap model is an assumption, not a measurement.

  tip_audit             Gate 1 as a picture: true skeleton leaves (capped)
                        against label-boundary ends (not capped) per cell. A
                        large boundary count means the labeller is truncating
                        spines mid-structure, which is a different problem from
                        the cap and would invalidate the correction.

LITERATURE REFERENCE VALUES
---------------------------
Drawn as CONTEXT, not ground truth. The F values are confocal Marching-Cubes
measurements on a different preparation from H01 EM skeletons, so the two are
not the same estimand; see LITERATURE below for sources.

DEPENDENCIES: numpy, pandas, matplotlib. No scipy.

Pure ASCII source (HPC-safe).
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                      # headless-safe; Colab still displays
import matplotlib.pyplot as plt            # noqa: E402

import spine_cap as sc                     # noqa: E402

MODULE_VERSION = "spine_cap_plots-1.0.0"


# --------------------------------------------------------------------------- #
# Published values used as comparison context. Every entry names its source.  #
# --------------------------------------------------------------------------- #
LITERATURE = {
    # F = (shaft area + spine area) / shaft area. Eyal et al. 2016 eLife,
    # Methods, computed from Benavides-Piccione 2013 confocal reconstructions.
    # Two donors, hence two bands; the 85-year-old is the age-matched one for
    # H01's 45-year-old donor only loosely, and neither is EM.
    "F_human": [
        ("Eyal 2016, temporal basal, 85 y", 1.48, 2.30),   # 1.89 +/- 0.41
        ("Eyal 2016, temporal basal, 40 y", 1.76, 3.02),   # 2.39 +/- 0.63
        ("Eyal 2016, cingulate basal, 85 y", 1.47, 2.15),  # 1.81 +/- 0.34
    ],
    "F_used_in_models": 1.9,               # Eyal 2016, value adopted for L2/3
    # Spine head surface area, human L2/3 temporal, Eyal et al. 2018 (n = 150).
    "head_area_um2": (1.51, 4.25),         # 2.88 +/- 1.37
    # H01 skeletonisation constants, Shapson-Coe et al. 2024 supplement.
    "h01_erosion_um": 0.100,
    "h01_node_spacing_um": 0.300,
}

_C = {"main": "#1f77b4", "alt": "#ff7f0e", "ok": "#2ca02c",
      "bad": "#d62728", "grey": "#7f7f7f", "purple": "#9467bd",
      "cap": "#D85A30"}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _pool(frames, require_columns=()):
    """Concatenate a {nid: DataFrame} mapping into one frame."""
    if isinstance(frames, pd.DataFrame):
        df = frames
    else:
        parts = [f for f in frames.values() if len(f)]
        df = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=list(require_columns)))
    for c in require_columns:
        if c not in df.columns:
            raise ValueError("pooled frame lacks required column %r; got %r"
                             % (c, list(df.columns)))
    return df


def _empty(fig, message):
    fig.text(0.5, 0.5, message, ha="center", va="center", fontsize=11,
             color=_C["grey"])
    return fig


def _ecdf(values):
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([]), np.array([])
    return v, np.arange(1, len(v) + 1) / float(len(v))


# --------------------------------------------------------------------------- #
# 1. The gallery: what the reconstruction actually looks like                  #
# --------------------------------------------------------------------------- #
def spine_gallery(profiles, n_cols=5, n_max=20, share_scale=True,
                  show_nodes=True, title=None):
    """Meridian outlines of n reconstructed spines, cap shaded.

    profiles : list of dicts from spine_cap.spine_profiles(). Each panel draws
        the primary (longest) root-to-tip branch as a surface of revolution:
        the outline is +/- r(u) against path distance u from the shaft surface.
        This is not an artist's impression -- it is exactly the surface whose
        lateral area spine_density integrates.

    share_scale : if True every panel uses the same axes limits, so relative
        size is readable across panels. Set False to see shape detail on the
        small spines at the cost of comparability.
    """
    profiles = list(profiles)[:n_max]
    n = len(profiles)
    if n == 0:
        return _empty(plt.figure(figsize=(8, 3)), "no spines to draw")

    n_cols = int(max(1, min(n_cols, n)))
    n_rows = int(np.ceil(n / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5 * n_cols, 2.5 * n_rows),
                             squeeze=False)

    if share_scale:
        u_max = max(max(p["cap_u_um"] or p["u_um"]) for p in profiles)
        r_max = max(max(p["r_um"]) for p in profiles)
    else:
        u_max = r_max = None

    for k, ax in enumerate(np.ravel(axes)):
        if k >= n:
            ax.axis("off")
            continue
        p = profiles[k]
        u = np.asarray(p["u_um"], dtype=float)
        r = np.asarray(p["r_um"], dtype=float)

        # the base segment (shaft node -> spine root) is drawn separately in
        # grey: spine_density attributes all of it to the spine, and on a
        # thick shaft it can dominate A_spine. That is an attribution
        # convention, not spine membrane, and it should be visible.
        has_base = len(u) > 1 and p["A_base_um2"] > 0.0
        i0 = 1 if has_base else 0
        if has_base:
            ax.fill_between(u[:2], -r[:2], r[:2], color=_C["grey"],
                            alpha=0.22, lw=0)
            ax.plot(u[:2], r[:2], color=_C["grey"], lw=1.2)
            ax.plot(u[:2], -r[:2], color=_C["grey"], lw=1.2)
        ax.fill_between(u[i0:], -r[i0:], r[i0:], color=_C["main"],
                        alpha=0.28, lw=0)
        ax.plot(u[i0:], r[i0:], color=_C["main"], lw=1.2)
        ax.plot(u[i0:], -r[i0:], color=_C["main"], lw=1.2)

        if p["capped"] and len(p["cap_u_um"]) > 1:
            cu = np.asarray(p["cap_u_um"], dtype=float)
            cr = np.asarray(p["cap_r_um"], dtype=float)
            ax.fill_between(cu, -cr, cr, color=_C["cap"], alpha=0.42, lw=0)
            ax.plot(cu, cr, color=_C["cap"], lw=1.4)
            ax.plot(cu, -cr, color=_C["cap"], lw=1.4)

        # the shaft surface, i.e. where the spine attaches
        ax.axvline(0.0, color=_C["grey"], lw=0.8, ls=":")
        if show_nodes:
            ax.plot(u, r, ".", color=_C["main"], ms=3.5)
            ax.plot(u, -r, ".", color=_C["main"], ms=3.5)

        tot = p["A_spine_um2"]
        frac = 100.0 * p["A_cap_um2"] / tot if tot > 0 else 0.0
        fbase = 100.0 * p["A_base_um2"] / tot if tot > 0 else 0.0
        ax.set_title("A=%.2f um2  cap %+.0f%%  base %.0f%%\n"
                     "%d nodes, %d tip%s"
                     % (tot, frac, fbase, p["n_nodes"], p["n_tips"],
                        "" if p["n_tips"] == 1 else "s"),
                     fontsize=7.5)
        ax.set_aspect("equal", adjustable="box")
        if share_scale:
            ax.set_xlim(-0.05 * u_max, 1.08 * u_max)
            ax.set_ylim(-1.15 * r_max, 1.15 * r_max)
        ax.tick_params(labelsize=6)
        if k % n_cols == 0:
            ax.set_ylabel("r (um)", fontsize=7)
        if k >= n - n_cols:
            ax.set_xlabel("u from shaft (um)", fontsize=7)

    fig.suptitle(title or ("reconstructed spine meridians, n = %d\n"
                           "grey: base segment (shaft radius -> neck, "
                           "attributed to the spine)   "
                           "blue: frustum chain from H01 radii   "
                           "orange: added cap" % n), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------------------- #
# 2. r_t, the only free input to the cap                                       #
# --------------------------------------------------------------------------- #
def tip_radius_distribution(audits, h_um=None, default_radius_um=0.05,
                            bins=40):
    """Histogram of terminal radii at capped tips: spine tips vs shaft tips.

    audits : one dict from spine_cap.audit_tips(), or a {nid: dict} mapping.
    """
    if isinstance(audits, dict) and "spine_r_tip_um" in audits:
        audits = {"cell": audits}
    h_um = LITERATURE["h01_erosion_um"] if h_um is None else h_um

    sp = np.concatenate([np.asarray(a["spine_r_tip_um"], dtype=float)
                         for a in audits.values()]) if audits else np.array([])
    ot = np.concatenate([np.asarray(a["other_r_tip_um"], dtype=float)
                         for a in audits.values()]) if audits else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if len(sp) == 0 and len(ot) == 0:
        return _empty(fig, "no capped tips found")

    ax = axes[0]
    hi = float(np.nanmax(np.concatenate([sp, ot]))) if len(sp) + len(ot) else 1.0
    edges = np.linspace(0.0, hi * 1.02, bins + 1)
    if len(sp):
        ax.hist(sp, bins=edges, color=_C["main"], alpha=0.65,
                label="spine tips (n=%d)" % len(sp))
    if len(ot):
        ax.hist(ot, bins=edges, color=_C["ok"], alpha=0.55,
                label="shaft/axon tips (n=%d)" % len(ot))
    ax.axvline(default_radius_um, color=_C["bad"], ls="--", lw=1.2,
               label="fallback radius %.0f nm" % (default_radius_um * 1000))
    ax.axvline(h_um, color=_C["cap"], ls="-.", lw=1.2,
               label="h = %.0f nm (hemisphere when r_t = h)" % (h_um * 1000))
    ax.set_xlabel("terminal radius r_t (um)")
    ax.set_ylabel("tips")
    ax.set_title("r_t at true skeleton leaves")
    ax.legend(fontsize=7)

    ax = axes[1]
    for v, lab, c in ((sp, "spine tips", _C["main"]),
                      (ot, "shaft/axon tips", _C["ok"])):
        x, y = _ecdf(v)
        if len(x):
            ax.plot(x, y, color=c, lw=1.6, label=lab)
    ax.axvline(default_radius_um, color=_C["bad"], ls="--", lw=1.2)
    ax.axvline(h_um, color=_C["cap"], ls="-.", lw=1.2)
    ax.set_xlabel("terminal radius r_t (um)")
    ax.set_ylabel("ECDF")
    ax.set_ylim(0, 1)
    ax.set_title("cumulative")
    ax.legend(fontsize=7)

    fig.suptitle("r_t is the only free input to the cap: "
                 "A_cap = pi (r_t^2 + h^2)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# --------------------------------------------------------------------------- #
# 3. How much of the answer is h?                                              #
# --------------------------------------------------------------------------- #
def cap_area_curve(audits=None, h_list=(0.0, 0.05, 0.10, 0.15),
                   r_max_um=0.8, default_radius_um=0.05):
    """A_cap(r_t) for several h, with the empirical r_t distribution below.

    audits may be None, in which case only the curves are drawn.
    """
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.5, 1]})
    r = np.linspace(1e-4, r_max_um, 400)

    ax = axes[0]
    for h in h_list:
        y = np.pi * (r ** 2 + h ** 2)
        lab = ("flat disc (h = 0), lower bound" if h == 0.0
               else "h = %.0f nm" % (h * 1000))
        ax.plot(r, y, lw=1.8 if h == LITERATURE["h01_erosion_um"] else 1.1,
                ls="--" if h == 0.0 else "-",
                label=lab,
                color=_C["cap"] if h == LITERATURE["h01_erosion_um"] else None)
    lo, hi = LITERATURE["head_area_um2"]
    ax.axhspan(lo, hi, color=_C["grey"], alpha=0.12, lw=0,
               label="published whole head area (Eyal 2018)")
    ax.set_ylabel("A_cap (um2)")
    ax.set_title("cap area vs terminal radius; the H01 erosion fixes h")
    ax.legend(fontsize=7.5)

    # The absolute curves collapse onto each other because A_cap / A_disc
    # = 1 + (h/r_t)^2, which is within a few percent of 1 whenever r_t >> h.
    # That is the point of this panel: r_t dominates, h barely matters, so the
    # cap SHAPE is not where the uncertainty lives. Only for very thin tips
    # (r_t approaching h) does the choice of h change the answer.
    ax = axes[1]
    for h in h_list:
        if h == 0.0:
            continue
        ax.plot(r, 1.0 + (h / r) ** 2, lw=1.8 if h ==
                LITERATURE["h01_erosion_um"] else 1.1,
                label="h = %.0f nm" % (h * 1000),
                color=_C["cap"] if h == LITERATURE["h01_erosion_um"] else None)
    ax.axhline(1.0, color=_C["grey"], ls="--", lw=1.0)
    ax.axhline(1.1, color=_C["grey"], ls=":", lw=0.8)
    ax.set_ylim(0.95, 2.0)
    ax.set_ylabel("A_cap / A_disc")
    ax.set_title("sensitivity to h:  A_cap / A_disc = 1 + (h / r_t)^2",
                 fontsize=9)
    ax.legend(fontsize=7.5)

    ax = axes[2]
    if audits:
        if isinstance(audits, dict) and "spine_r_tip_um" in audits:
            audits = {"cell": audits}
        sp = np.concatenate([np.asarray(a["spine_r_tip_um"], dtype=float)
                             for a in audits.values()])
        if len(sp):
            ax.hist(sp, bins=np.linspace(0, r_max_um, 45),
                    color=_C["main"], alpha=0.7)
            ax.axvline(float(np.median(sp)), color=_C["bad"], lw=1.3,
                       label="median r_t = %.3f um" % float(np.median(sp)))
            ax.legend(fontsize=7.5)
    else:
        ax.text(0.5, 0.5, "no audit supplied", transform=ax.transAxes,
                ha="center", va="center", color=_C["grey"], fontsize=9)
    ax.axvline(default_radius_um, color=_C["bad"], ls="--", lw=1.0)
    ax.set_xlabel("terminal radius r_t (um)")
    ax.set_ylabel("spine tips")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4. The size of the correction, per spine                                     #
# --------------------------------------------------------------------------- #
def cap_contribution(geom_frames, bins=40):
    """Per-spine A_cap / A_spine, as histogram and ECDF.

    geom_frames : frames from spine_geometry.build_spine_geometry(cap_tips=True)
    """
    df = _pool(geom_frames, require_columns=("A_spine_um2", "A_cap_um2"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if len(df) == 0:
        return _empty(fig, "no spines")

    a = df["A_spine_um2"].to_numpy(dtype=float)
    c = df["A_cap_um2"].to_numpy(dtype=float)
    ok = np.isfinite(a) & np.isfinite(c) & (a > 0)
    if not ok.any():
        return _empty(fig, "no spines with positive area")
    frac = 100.0 * c[ok] / a[ok]

    ax = axes[0]
    ax.hist(frac, bins=bins, color=_C["cap"], alpha=0.75)
    ax.axvline(float(np.median(frac)), color=_C["bad"], lw=1.4,
               label="median %.1f%%" % float(np.median(frac)))
    ax.axvline(float(np.mean(frac)), color=_C["grey"], lw=1.2, ls="--",
               label="mean %.1f%%" % float(np.mean(frac)))
    ax.set_xlabel("A_cap / A_spine  (%)")
    ax.set_ylabel("spines")
    ax.set_title("size of the correction, per spine (n = %d)" % int(ok.sum()))
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(a[ok], c[ok], s=6, alpha=0.35, color=_C["main"], lw=0)
    ax.set_xlabel("A_spine, capped (um2)")
    ax.set_ylabel("A_cap (um2)")
    ax.set_title("absolute cap vs spine size")
    lo, hi = LITERATURE["head_area_um2"]
    ax.axvspan(lo, hi, color=_C["grey"], alpha=0.12, lw=0,
               label="published head area (Eyal 2018)")
    ax.legend(fontsize=7.5)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 5. The decision figure: F as a bracket                                       #
# --------------------------------------------------------------------------- #
def f_bracket(f_by_mode, show_literature=True):
    """F per cell under the three cap modes, against the published bands.

    f_by_mode : {"no cap": {nid: F}, "flat disc": {nid: F},
                 "cap h=100nm": {nid: F}}
        Any number of modes; order of the dict is the order plotted. The
        caller computes F with spine_density.cell_f_implied_from_phi, so this
        function never recomputes it.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    modes = list(f_by_mode.keys())
    if not modes:
        return _empty(fig, "no modes supplied")
    nids = sorted({n for m in modes for n in f_by_mode[m]}, key=str)
    if not nids:
        return _empty(fig, "no cells supplied")

    if show_literature:
        # bands go in the legend, not as inline text: with several overlapping
        # bands the inline labels collide and become unreadable
        for k, (lab, lo, hi) in enumerate(LITERATURE["F_human"]):
            ax.axhspan(lo, hi, color=_C["grey"], alpha=0.09 + 0.03 * k, lw=0,
                       label="%s: %.2f-%.2f" % (lab, lo, hi))
        ax.axhline(LITERATURE["F_used_in_models"], color=_C["grey"],
                   ls=":", lw=1.2,
                   label="F = %.1f adopted for L2/3 (Eyal 2016)"
                         % LITERATURE["F_used_in_models"])

    x = np.arange(len(nids), dtype=float)
    width = 0.8 / max(1, len(modes))
    palette = [_C["grey"], _C["main"], _C["cap"], _C["purple"], _C["ok"]]
    for j, m in enumerate(modes):
        vals = [f_by_mode[m].get(n, np.nan) for n in nids]
        ax.bar(x + j * width - 0.4 + width / 2.0, vals, width * 0.92,
               color=palette[j % len(palette)], alpha=0.85, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nids], rotation=20, fontsize=8)
    ax.set_ylabel("F  =  (shaft + spine area) / shaft area")
    ax.set_ylim(bottom=1.0)
    ax.set_title("F bracket: the cap model is an assumption, so report a range")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 6. Gate 1 as a picture                                                       #
# --------------------------------------------------------------------------- #
def tip_audit(audits):
    """Per cell: true skeleton leaves (capped) vs label-boundary ends (not).

    audits : {nid: dict} from spine_cap.audit_tips().
    """
    fig, ax = plt.subplots(figsize=(9, 4.2))
    if isinstance(audits, dict) and "spine_r_tip_um" in audits:
        audits = {"cell": audits}
    nids = sorted(audits.keys(), key=str)
    if not nids:
        return _empty(fig, "no audits supplied")

    leaf = [audits[n]["spine_n_true_leaf"] for n in nids]
    bnd = [audits[n]["spine_n_label_boundary_end"] for n in nids]
    bad = [audits[n]["spine_n_true_leaf_bad_radius"] for n in nids]
    x = np.arange(len(nids), dtype=float)

    ax.bar(x - 0.25, leaf, 0.24, color=_C["ok"], label="true leaf -> CAPPED")
    ax.bar(x, bnd, 0.24, color=_C["bad"],
           label="label-boundary end -> not capped")
    ax.bar(x + 0.25, bad, 0.24, color=_C["grey"],
           label="true leaf, r <= 0 -> not capped")
    for xi, (l_, b_) in enumerate(zip(leaf, bnd)):
        tot = l_ + b_
        if tot:
            ax.text(xi, max(l_, b_) * 1.02, "%.1f%% boundary"
                    % (100.0 * b_ / tot), ha="center", fontsize=7,
                    color=_C["bad"] if b_ else _C["grey"])

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in nids], rotation=20, fontsize=8)
    ax.set_ylabel("spine tips")
    ax.set_title("Gate 1: only true skeleton leaves saw the 100 nm erosion.\n"
                 "A large boundary count means the labeller truncates spines "
                 "mid-structure -- a different problem from the cap.",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
