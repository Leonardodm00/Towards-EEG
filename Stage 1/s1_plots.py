"""s1_plots -- figures for the S1 S1 quantities.

Towards-EEG stage S1 (data-generation side; runs in Colab, not on the HPC).

Companion to alignment_plots, and follows the same contract: every function
here takes frames and RETURNS a matplotlib Figure. Nothing in this module
writes a file, reads a file, or computes a scientific quantity. Saving is
alignment_plots.save_figure's job; computing is spine_geometry's and
truncation_flag's. A plot function that recomputed its own inputs would be a
second, silently divergent implementation of the thing being plotted.

WHICH FIGURES EARN THEIR PLACE, AND WHY
---------------------------------------
Two of these are results, not diagnostics:

  neck_resistance_sweep   The S1.8 analytic-sweep deliverable. It asks whether
                          H01 neck geometry yields resistances consistent with
                          the published range, and how much of the answer is
                          the assumed rho_a rather than the measurement. This
                          is the external-validity check on the reconstruction
                          that nobody has run.

  attenuation_factor      The decision figure for assumption A2. kappa(sigma)
                          = 1 / (1 + g_syn R_neck(sigma)) is the factor by
                          which relocating a synapse from a spine head to the
                          shaft misstates the charge delivered. If kappa is
                          near 1 across the whole (rho_a, g_syn) grid, A2 is
                          close to exact and the spine-resolved export is not
                          worth its cost. If it is not, the plot says by how
                          much and under which assumptions.

The other three are diagnostics that gate whether the first two mean anything:

  neck_geometry           Are the measured spine dimensions plausible at all,
                          against published human values? Includes the raw
                          neck-radius histogram with the fallback value
                          marked, because a cell whose radii are all at the
                          50 nm fallback produces resistances that look
                          entirely credible and mean nothing.

  spine_distance_profile  Does spine density peak where the literature says it
                          does? A profile peaking at the wrong distance points
                          at the labeller, not at the biology.

  truncation_diagnostics  Is the taper threshold defensible? The threshold is
                          NOT calibrated against labelled data. If the
                          taper_ratio distribution is bimodal, the trough
                          calibrates it empirically and the default should
                          move there. If it is unimodal, no threshold is
                          defensible and the flag should be treated as a
                          ranking rather than a verdict. The histogram is the
                          evidence either way; read it before trusting
                          frac_truncated.

LITERATURE REFERENCE VALUES
---------------------------
The comparison bands drawn on these figures are listed in LITERATURE below,
each with its source. They are drawn as CONTEXT, not as ground truth: three of
the four spine-neck resistance sources are rodent, and the spread between them
is roughly an order of magnitude, which is itself the point being illustrated.

DEPENDENCIES: numpy, pandas, matplotlib. No scipy.

Pure ASCII source (HPC-safe).
"""

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                      # headless-safe; Colab still displays
import matplotlib.pyplot as plt            # noqa: E402

MODULE_VERSION = "s1_plots-1.0.0"


# --------------------------------------------------------------------------- #
# Published values used as comparison context. Every entry names its source.  #
# --------------------------------------------------------------------------- #
LITERATURE = {
    # Spine-neck resistance, MOhm. Note the range across sources: this is a
    # contested quantity, and part of the spread is the assumed rho_a rather
    # than a measurement disagreement.
    "R_neck_MOhm": [
        ("Eyal 2018 (human, model)", 50.0, 80.0),
        ("Eyal 2018 envelope", 19.0, 128.0),
        ("Gulledge 2012 (EM, rho=100)", 1.0, 400.0),
        ("Acker 2016 (mouse, VSD+FRAP)", 179.0, 204.0),
        ("Harnett 2012 (rat CA1)", 470.0, 558.0),
    ],
    # Spine dimensions, human L2/3, Eyal et al. 2018 (n = 150).
    "neck_diam_um": (0.20, 0.30),          # prototypical 0.25
    "neck_length_um": (0.84, 1.84),        # 1.34 +/- 0.50
    "head_area_um2": (1.51, 4.25),         # 2.88 +/- 1.37
    # Distance of peak spine density from the soma, human temporal basal
    # dendrites, Benavides-Piccione et al. 2021.
    "peak_density_um": 90.0,
}

_C = {"main": "#1f77b4", "alt": "#ff7f0e", "ok": "#2ca02c",
      "bad": "#d62728", "grey": "#7f7f7f", "purple": "#9467bd"}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _pool(frames, require_columns=()):
    """Concatenate a {nid: DataFrame} mapping into one frame.

    Accepts either a mapping or an already-pooled DataFrame, so a caller with
    a single cell does not have to wrap it.
    """
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


def _ecdf(values):
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.array([]), np.array([])
    return v, np.arange(1, len(v) + 1) / float(len(v))


def _empty(fig, message):
    fig.text(0.5, 0.5, message, ha="center", va="center", fontsize=11,
             color=_C["grey"])
    return fig


def _band(ax, lo, hi, label, color, alpha=0.12):
    ax.axvspan(lo, hi, color=color, alpha=alpha, label=label, lw=0)


def neck_resistance_mohm(g_per_cm, rho_a_ohm_cm):
    """R [MOhm] from the stored geometric factor. Mirrors spine_geometry."""
    return np.asarray(g_per_cm, dtype=float) * float(rho_a_ohm_cm) / 1.0e6


# --------------------------------------------------------------------------- #
# 1. Neck geometry: is the reconstruction plausible at all?                    #
# --------------------------------------------------------------------------- #
def neck_geometry(spine_frames, default_radius_um=0.05):
    """Four panels of measured spine dimensions against published values.

    Panel 4 is the one to read first: if the neck-radius histogram is a single
    spike at the fallback value, every resistance derived from this bank is a
    function of neck LENGTH alone, and panels 1-3 and every other figure in
    this module are describing an artefact.
    """
    df = _pool(spine_frames, ("L_neck_um", "d_neck_equiv_um", "A_head_um2",
                              "has_neck"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    if len(df) == 0:
        return _empty(fig, "no spines")

    wn = df[df["has_neck"].astype(bool)]

    ax = axes[0, 0]
    if len(wn):
        ax.hist(wn["L_neck_um"].dropna(), bins=40, color=_C["main"])
    lo, hi = LITERATURE["neck_length_um"]
    _band(ax, lo, hi, "Eyal 2018, 1.34 +/- 0.50", _C["ok"])
    ax.set_xlabel("neck length (um)"); ax.set_ylabel("spines")
    ax.set_title("neck length, n = %d with a labelled neck" % len(wn))
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    if len(wn):
        d = wn["d_neck_equiv_um"].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(d, bins=40, color=_C["alt"])
    lo, hi = LITERATURE["neck_diam_um"]
    _band(ax, lo, hi, "Eyal 2018, 0.20-0.30", _C["ok"])
    ax.set_xlabel("equivalent uniform neck diameter (um)")
    ax.set_ylabel("spines")
    ax.set_title("the diameter Eq. (4) would need to give this R")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.hist(df["A_head_um2"].dropna(), bins=40, color=_C["purple"])
    lo, hi = LITERATURE["head_area_um2"]
    _band(ax, lo, hi, "Eyal 2018, 2.88 +/- 1.37", _C["ok"])
    ax.set_xlabel("head membrane area (um^2)"); ax.set_ylabel("spines")
    ax.set_title("head area, n = %d spines" % len(df))
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    if "neck_r_mean_um" in df.columns and len(wn):
        r = wn["neck_r_mean_um"].dropna()
        ax.hist(r, bins=40, color=_C["grey"])
        n_at_default = int(np.sum(np.isclose(r, default_radius_um)))
        frac = n_at_default / float(len(r)) if len(r) else float("nan")
        ax.axvline(default_radius_um, color=_C["bad"], ls="--", lw=1.4,
                   label="fallback %.3f um (%.0f%% of spines)"
                   % (default_radius_um, 100.0 * frac))
        ax.legend(fontsize=7)
        if frac > 0.5:
            ax.set_title("MEASURED? NO -- radii are mostly the fallback")
        else:
            ax.set_title("mean neck radius, fallback marked")
    ax.set_xlabel("mean neck radius (um)"); ax.set_ylabel("spines")

    fig.suptitle("spine geometry against published human values")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 2. Neck resistance across the rho_a sweep                                    #
# --------------------------------------------------------------------------- #
def neck_resistance_sweep(spine_frames,
                          rho_a_ohm_cm=(100.0, 200.0, 300.0, 400.0)):
    """R_neck ECDF at each rho_a, with the published range for context.

    The left panel is the result; the right panel places this bank's median
    among the published estimates. The published estimates disagree with each
    other by roughly an order of magnitude, and three of the four are rodent,
    so agreement with any single one of them is weak evidence. What the figure
    is really for is the WIDTH: how much of the answer is rho_a rather than
    geometry.
    """
    df = _pool(spine_frames, ("g_per_cm", "has_neck"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    wn = df[df["has_neck"].astype(bool)] if len(df) else df
    if len(wn) == 0:
        return _empty(fig, "no spines with a labelled neck")

    ax = axes[0]
    for rho in rho_a_ohm_cm:
        r = neck_resistance_mohm(wn["g_per_cm"].values, rho)
        x, y = _ecdf(r)
        if len(x):
            ax.step(x, y, where="post", lw=1.6,
                    label="rho_a = %.0f ohm cm  (median %.0f MOhm)"
                    % (rho, np.median(r)))
    ax.set_xscale("log")
    ax.set_xlabel("R_neck (MOhm, log scale)")
    ax.set_ylabel("cumulative fraction of spines")
    ax.set_title("how much of R_neck is the rho_a assumption?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    labels, lows, highs = [], [], []
    for name, lo, hi in LITERATURE["R_neck_MOhm"]:
        labels.append(name); lows.append(lo); highs.append(hi)
    ypos = np.arange(len(labels))
    ax.barh(ypos, np.array(highs) - np.array(lows), left=lows, height=0.55,
            color=_C["grey"], alpha=0.55)
    for rho, col in zip(rho_a_ohm_cm,
                        [_C["main"], _C["alt"], _C["ok"], _C["purple"],
                         _C["bad"]] * 3):
        med = float(np.median(neck_resistance_mohm(wn["g_per_cm"].values, rho)))
        ax.axvline(med, color=col, lw=1.6,
                   label="this bank, rho_a = %.0f (%.0f MOhm)" % (rho, med))
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xscale("log")
    ax.set_xlabel("R_neck (MOhm, log scale)")
    ax.set_title("published estimates disagree by ~an order of magnitude")
    # NOT 'lower right': that corner sits on top of the lowest literature
    # band, which is the human one and the most relevant of the five.
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("spine-neck resistance, n = %d spines with a labelled neck"
                 % len(wn))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 3. The A2 decision figure                                                    #
# --------------------------------------------------------------------------- #
def attenuation_factor(spine_frames,
                       rho_a_ohm_cm=(100.0, 200.0, 300.0, 400.0),
                       g_syn_nS=(0.2, 0.5, 1.0, 2.0),
                       kappa_threshold=0.9):
    """kappa(sigma) = 1 / (1 + g_syn R_neck(sigma)): does A2 matter?

    kappa is the factor by which moving a synapse from its spine head to the
    shaft misstates the charge delivered to the dendrite, in the quasi-steady
    approximation. kappa = 1 means the relocation is exact.

    Left  : median kappa over the (rho_a, g_syn) grid, annotated.
    Right : the full distribution at each rho_a, at the largest g_syn given --
            the worst case of those supplied, since kappa falls as g_syn rises.

    CAVEAT, and it is the whole caveat of this figure. kappa as written is a
    CHARGE ratio under a quasi-steady assumption. It is not the spine-to-soma
    VOLTAGE attenuation ratio that voltage-dye studies report, and the two do
    not have the same dependence on g_syn -- published work finds the voltage
    ratio close to independent of synaptic conductance while this expression
    is explicitly not. That is not a contradiction (both numerator and
    denominator of a voltage ratio saturate in g_syn, so the ratio need not),
    but it does mean this figure must not be compared directly against a
    published attenuation ratio.
    """
    df = _pool(spine_frames, ("g_per_cm", "has_neck"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    wn = df[df["has_neck"].astype(bool)] if len(df) else df
    if len(wn) == 0:
        return _empty(fig, "no spines with a labelled neck")

    rho_a_ohm_cm = tuple(rho_a_ohm_cm)
    g_syn_nS = tuple(g_syn_nS)

    # kappa = 1 / (1 + g R). g in nS = 1e-9 S, R in Ohm -> g*R dimensionless.
    grid = np.zeros((len(g_syn_nS), len(rho_a_ohm_cm)), dtype=float)
    for i, g in enumerate(g_syn_nS):
        for j, rho in enumerate(rho_a_ohm_cm):
            r_ohm = wn["g_per_cm"].values * rho
            grid[i, j] = float(np.median(1.0 / (1.0 + g * 1e-9 * r_ohm)))

    ax = axes[0]
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                   vmin=min(0.5, float(grid.min())), vmax=1.0)
    ax.set_xticks(range(len(rho_a_ohm_cm)))
    ax.set_xticklabels(["%.0f" % r for r in rho_a_ohm_cm])
    ax.set_yticks(range(len(g_syn_nS)))
    ax.set_yticklabels(["%.2f" % g for g in g_syn_nS])
    ax.set_xlabel("rho_a (ohm cm)"); ax.set_ylabel("g_syn (nS)")
    for i in range(len(g_syn_nS)):
        for j in range(len(rho_a_ohm_cm)):
            ax.text(j, i, "%.3f" % grid[i, j], ha="center", va="center",
                    fontsize=8,
                    color="white" if grid[i, j] < 0.85 else "black")
    ax.set_title("median kappa: 1.000 means relocation is exact")
    fig.colorbar(im, ax=ax, label="median kappa")

    ax = axes[1]
    g_worst = max(g_syn_nS)
    for rho in rho_a_ohm_cm:
        r_ohm = wn["g_per_cm"].values * rho
        kappa = 1.0 / (1.0 + g_worst * 1e-9 * r_ohm)
        x, y = _ecdf(kappa)
        if len(x):
            frac_below = float(np.mean(kappa < kappa_threshold))
            ax.step(x, y, where="post", lw=1.6,
                    label="rho_a = %.0f  (%.0f%% below %.2f)"
                    % (rho, 100.0 * frac_below, kappa_threshold))
    ax.axvline(kappa_threshold, color=_C["bad"], ls="--", lw=1.2)
    ax.set_xlabel("kappa at g_syn = %.2f nS" % g_worst)
    ax.set_ylabel("cumulative fraction of spines")
    ax.set_title("worst case of the g_syn values supplied")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("assumption A2: charge error from relocating a spine synapse "
                 "to the shaft (quasi-steady)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4. Spine distribution along the dendrite                                     #
# --------------------------------------------------------------------------- #
def spine_distance_profile(spine_frames, bin_width_um=10.0, max_d_um=None):
    """Spine count and spine area against path distance from the soma.

    The comparison is to the ~90 um density peak reported for human temporal
    basal dendrites. Two mismatches to watch for: a peak at the wrong distance
    points at the labeller rather than the biology, and a profile that does
    not fall away proximally undermines the 60 um convention that F_lit uses.

    NOTE the profile here is per 10 um SHELL, not per micron of cable, so it
    is not the same quantity as psi(nu, b, d) and confounds spine density with
    how much cable sits at each distance. It is a fast look, not the S1.7
    comparison; use spine_density.psi_vs_distance for that.
    """
    df = _pool(spine_frames, ("d_base_um", "A_spine_um2"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    if len(df) == 0:
        return _empty(fig, "no spines")

    d = pd.to_numeric(df["d_base_um"], errors="coerce")
    ok = np.isfinite(d)
    d = d[ok]
    a = pd.to_numeric(df["A_spine_um2"], errors="coerce")[ok]
    if len(d) == 0:
        return _empty(fig, "no finite path distances")

    top = float(max_d_um if max_d_um else np.nanmax(d))
    edges = np.arange(0.0, top + bin_width_um, bin_width_um)
    idx = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    areas = np.bincount(idx, weights=a.values, minlength=len(edges) - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    peak_lit = LITERATURE["peak_density_um"]
    for ax, y, lab, col in ((axes[0], counts, "spines per %.0f um shell"
                             % bin_width_um, _C["main"]),
                            (axes[1], areas, "spine area per shell (um^2)",
                             _C["purple"])):
        ax.bar(centres, y, width=bin_width_um * 0.9, color=col)
        ax.axvline(peak_lit, color=_C["ok"], ls="--", lw=1.4,
                   label="literature peak ~%.0f um" % peak_lit)
        ax.axvline(60.0, color=_C["grey"], ls=":", lw=1.2,
                   label="F_lit cutoff 60 um")
        if y.sum() > 0:
            ax.axvline(centres[int(np.argmax(y))], color=_C["bad"], lw=1.4,
                       label="this bank's peak %.0f um"
                       % centres[int(np.argmax(y))])
        ax.set_xlabel("path distance from soma to spine base (um)")
        ax.set_ylabel(lab)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")

    axes[0].set_title("where are the spines?")
    axes[1].set_title("where is the spine membrane?")
    fig.suptitle("spine distribution along the dendrite, n = %d spines"
                 % len(d))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 5. Truncation diagnostics                                                    #
# --------------------------------------------------------------------------- #
def truncation_diagnostics(trunc_frames, ratio_threshold=0.7,
                           margin_um=10.0):
    """Four panels on whether the truncation flag is trustworthy.

    Panel 1 is the important one. The taper threshold is NOT calibrated
    against labelled truncated/real tips; it is a legible default. If this
    histogram is bimodal, the trough is where the threshold belongs and the
    default should be moved there and recorded. If it is unimodal, no
    threshold separates the two populations and frac_truncated should be
    reported as a ranking, not a count.
    """
    df = _pool(trunc_frames, ("taper_ratio", "taper_determinate"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    if len(df) == 0:
        return _empty(fig, "no dendrite tips")

    det = df[df["taper_determinate"].astype(bool)]

    ax = axes[0, 0]
    if len(det):
        r = pd.to_numeric(det["taper_ratio"], errors="coerce").dropna()
        r = r[np.isfinite(r)]
        ax.hist(r, bins=50, color=_C["main"])
        ax.axvline(ratio_threshold, color=_C["bad"], ls="--", lw=1.6,
                   label="threshold %.2f (UNCALIBRATED)" % ratio_threshold)
        ax.legend(fontsize=7)
    ax.set_xlabel("taper ratio  r_tip / r_ref   (1.0 = no narrowing)")
    ax.set_ylabel("tips")
    ax.set_title("bimodal -> threshold is calibratable; unimodal -> it is not")

    ax = axes[0, 1]
    if "truncation_basis" in df.columns:
        vc = df["truncation_basis"].value_counts()
        order = [b for b in ("none", "taper", "z_boundary", "both",
                             "unresolved") if b in vc.index]
        vals = [int(vc[b]) for b in order]
        cols = {"none": _C["ok"], "taper": _C["bad"],
                "z_boundary": _C["alt"], "both": _C["purple"],
                "unresolved": _C["grey"]}
        ax.bar(range(len(order)), vals,
               color=[cols.get(b, _C["grey"]) for b in order])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20, fontsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("tips")
    ax.set_title("what evidence flagged each tip")

    ax = axes[1, 0]
    if "dist_to_z_bound_um" in df.columns:
        z = pd.to_numeric(df["dist_to_z_bound_um"], errors="coerce").dropna()
        ax.hist(z[np.isfinite(z)], bins=50, color=_C["alt"])
        ax.axvline(margin_um, color=_C["bad"], ls="--", lw=1.4,
                   label="margin %.0f um" % margin_um)
        ax.legend(fontsize=7)
    ax.set_xlabel("distance from tip to nearest pooled z bound (um)")
    ax.set_ylabel("tips")
    ax.set_title("z is the only boundary axis trusted (see module docstring)")

    ax = axes[1, 1]
    if "is_truncated" in df.columns and "d_from_soma_um" in df.columns:
        for flag, col, lab in ((False, _C["ok"], "not truncated"),
                               (True, _C["bad"], "truncated")):
            sub = pd.to_numeric(
                df.loc[df["is_truncated"].astype(bool) == flag,
                       "d_from_soma_um"], errors="coerce").dropna()
            x, y = _ecdf(sub)
            if len(x):
                ax.step(x, y, where="post", lw=1.6,
                        label="%s (n = %d)" % (lab, len(x)))
        ax.legend(fontsize=8, loc="lower right")
    ax.set_xlabel("path distance from soma to tip (um)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title("are flagged tips the distal ones, as expected?")
    ax.grid(alpha=0.3)

    fig.suptitle("truncation evidence, n = %d tips (%d determinate)"
                 % (len(df), len(det)))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 6. Batch rollup                                                              #
# --------------------------------------------------------------------------- #
def s1_batch(s1_df):
    """Per-cell rollup of the S1 quantities across the bank.

    Cells whose radii are suspect are drawn in the warning colour throughout,
    because none of their resistance-derived quantities mean anything and a
    bank plot that hides that invites reading them as data.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    if s1_df is None or len(s1_df) == 0:
        return _empty(fig, "no S1 rows")

    df = s1_df.reset_index(drop=True)
    n = len(df)
    x = np.arange(n)
    trust = (df["resistance_trustworthy"].fillna(False).astype(bool).values
             if "resistance_trustworthy" in df.columns
             else np.ones(n, dtype=bool))
    cols = np.where(trust, _C["main"], _C["bad"])

    ax = axes[0, 0]
    if "n_spines" in df.columns:
        ax.bar(x, df["n_spines"].fillna(0).values, color=cols)
    ax.set_ylabel("spines"); ax.set_xlabel("cell")
    ax.set_title("spines per cell (red = radii suspect)")

    ax = axes[0, 1]
    if "frac_with_neck" in df.columns:
        ax.bar(x, df["frac_with_neck"].fillna(0).values, color=cols)
        ax.axhline(0.5, color=_C["grey"], ls=":", lw=1.2)
    ax.set_ylim(0, 1); ax.set_ylabel("fraction"); ax.set_xlabel("cell")
    ax.set_title("spines with a labelled neck")

    ax = axes[1, 0]
    med = [c for c in df.columns
           if c.startswith("R_neck_MOhm_rho") and c.endswith("_median")]
    if med:
        for c in sorted(med):
            ax.plot(x, df[c].values, "o-", ms=4, lw=1.0,
                    label=c.replace("R_neck_MOhm_", "").replace("_median", ""))
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    ax.set_ylabel("median R_neck (MOhm)"); ax.set_xlabel("cell")
    ax.set_title("median neck resistance per cell, by rho_a")

    ax = axes[1, 1]
    if "frac_truncated" in df.columns:
        ax.bar(x, df["frac_truncated"].fillna(0).values, color=_C["alt"])
    ax.set_ylim(0, 1); ax.set_ylabel("fraction of tips")
    ax.set_xlabel("cell")
    ax.set_title("tips flagged as truncated")

    n_bad = int(np.sum(~trust))
    fig.suptitle("S1 across the bank, n = %d cells%s"
                 % (n, ("  --  %d with SUSPECT RADII" % n_bad) if n_bad
                    else ""))
    fig.tight_layout()
    return fig
