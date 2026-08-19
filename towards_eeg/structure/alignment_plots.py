"""Diagnostic plots for stage S1 alignment.

Separation of concerns: this module contains NO pipeline logic, NO file
discovery and NO Drive paths. Every function takes DataFrames or plain dicts
and RETURNS a figure object; writing to disk is the separate concern of
`save_figure`. That keeps the plots swappable without touching alignment.py,
and testable without running the pipeline.

Two backends, deliberately different jobs:
  plotly      3-D arbours you rotate by hand. The frame is the thing being
              tested, so being able to spin it is the point, not decoration.
  matplotlib  everything that is a distribution, a profile or a batch summary,
              and static PNGs of the arbours for a report.

Plot repertoire
---------------
Per cell
  arbour_comparison        raw-centred vs aligned, side by side, by class   [plotly]
  arbour_static            the same pair as a PNG                            [mpl]
  depth_profile            aligned-z histogram of dendrite vs axon           [mpl]
  neighbourhood            the k chosen references around the target soma    [mpl]
Batch
  batch_quality            neighbour distance, angular spread, det error     [mpl]
  orientation_consistency  post-alignment arbour direction vs +z             [mpl]
  rigidity                 per-cell |delta| on every section 7 quantity      [mpl]

The rigidity panel is the one to read first on any batch: alignment is a rigid
transform, so every bar must be exactly zero. A non-zero bar means the
transform is not rigid or was applied at the wrong step.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")                      # headless-safe; Colab still displays
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.collections import LineCollection   # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402


MODULE_VERSION = "alignment_plots-1.0.0"

# One colour per compartment class, stable across every figure so a reader can
# compare panels without re-reading a legend.
CLASS_COLOUR = {
    "soma": "#111111",
    "dend": "#1f77b4",
    "apic_dend": "#17becf",
    "basal_dend": "#1f77b4",
    "axon": "#d62728",
    "ais": "#ff7f0e",
    "spine": "#9467bd",
    "glia": "#8c564b",
    "unknown": "#7f7f7f",
}
DEFAULT_COLOUR = "#7f7f7f"


# --------------------------------------------------------------------------- #
#  Geometry helpers                                                            #
# --------------------------------------------------------------------------- #
def _edges(df, id_col="id", parent_col="p"):
    """(N, 2, 3) array of parent -> child segments, in the frame's own units."""
    pos = {int(r[0]): (r[1], r[2], r[3]) for r in
           df[[id_col, "x", "y", "z"]].to_numpy()}
    segs, cls = [], []
    col = "compartment_class" if "compartment_class" in df.columns else None
    for r in df.itertuples(index=False):
        p = int(getattr(r, parent_col))
        if p == -1 or p not in pos:
            continue
        segs.append([pos[p], (r.x, r.y, r.z)])
        cls.append(str(getattr(r, col)) if col else "dend")
    return np.asarray(segs, dtype=float), np.asarray(cls, dtype=object)


def _subsample(segs, cls, max_segments):
    """Deterministic thinning. Returns everything if already under the cap."""
    if max_segments is None or len(segs) <= max_segments:
        return segs, cls
    step = int(np.ceil(len(segs) / float(max_segments)))
    return segs[::step], cls[::step]


def _scale(df, units):
    """Return a copy with x, y, z in um. `units` is 'nm' or 'um'."""
    out = df.copy()
    if units == "nm":
        for c in ("x", "y", "z"):
            out[c] = out[c] / 1000.0
    elif units != "um":
        raise ValueError("units must be 'nm' or 'um', got %r" % units)
    return out


def arbour_direction(df, class_column="compartment_class",
                     classes=("dend", "apic_dend", "basal_dend"),
                     percentile=90.0):
    """Unit vector from the soma to the centre of mass of the DISTAL dendrite.

    Mirrors the `calculate_z_alignment_math` construction described in the
    Alignment Metadata README: isolate the most distal dendritic tips (top
    100-percentile by distance from the soma) and take the direction to their
    centre of mass. Recomputed here from the frame itself so the post-alignment
    check does not simply read back the v_com the bank was built from.

    Returns None when there is no dendrite to measure.
    """
    if class_column in df.columns:
        d = df[df[class_column].astype(str).isin(classes)]
    else:
        d = df
    if not len(d):
        return None
    P = d[["x", "y", "z"]].to_numpy(float)
    r = np.linalg.norm(P, axis=1)                 # soma is at the origin
    if not np.isfinite(r).any() or r.max() <= 0:
        return None
    tips = P[r >= np.percentile(r, percentile)]
    if not len(tips):
        return None
    com = tips.mean(axis=0)
    n = np.linalg.norm(com)
    return None if n == 0 else com / n


def angle_from_z_deg(v):
    """Angle between a unit vector and +z, in degrees. None passes through."""
    if v is None:
        return float("nan")
    c = float(np.clip(np.dot(v, [0.0, 0.0, 1.0]), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


# --------------------------------------------------------------------------- #
#  Per-cell: interactive                                                       #
# --------------------------------------------------------------------------- #
def arbour_comparison(df_raw, df_aligned, nid, raw_units="nm",
                      aligned_units="nm", max_segments=30000):
    """Two linked 3-D panels: raw-centred vs aligned. Returns a plotly Figure.

    `df_raw` is centred on its own soma here so the two panels differ ONLY by
    the rotation -- otherwise the raw panel sits at MICrONS coordinates around
    2.7e6 nm and no visual comparison is possible.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    raw = _scale(df_raw, raw_units)
    root = raw[raw["p"] == -1]
    if len(root):
        origin = root.iloc[0][["x", "y", "z"]].to_numpy(float)
        for i, c in enumerate("xyz"):
            raw[c] = raw[c] - origin[i]
    ali = _scale(df_aligned, aligned_units)

    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("raw, centred on soma", "aligned (apical -> +z)"))

    for col, frame in ((1, raw), (2, ali)):
        segs, cls = _subsample(*_edges(frame), max_segments)
        if not len(segs):
            continue
        for klass in pd.unique(cls):
            m = cls == klass
            s = segs[m]
            xs = np.empty(len(s) * 3); ys = np.empty(len(s) * 3)
            zs = np.empty(len(s) * 3)
            xs[0::3], xs[1::3], xs[2::3] = s[:, 0, 0], s[:, 1, 0], np.nan
            ys[0::3], ys[1::3], ys[2::3] = s[:, 0, 1], s[:, 1, 1], np.nan
            zs[0::3], zs[1::3], zs[2::3] = s[:, 0, 2], s[:, 1, 2], np.nan
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", name=str(klass),
                legendgroup=str(klass), showlegend=(col == 1),
                line=dict(color=CLASS_COLOUR.get(str(klass), DEFAULT_COLOUR),
                          width=2),
                hoverinfo="name"), row=1, col=col)
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode="markers", showlegend=False,
            marker=dict(size=4, color="#111111"), name="soma",
            hovertext="soma"), row=1, col=col)

    axes = dict(xaxis_title="x (um)", yaxis_title="y (um)", zaxis_title="z (um)",
                aspectmode="data")
    fig.update_layout(title="neuron %s -- alignment check" % nid,
                      scene=axes, scene2=axes, height=650,
                      legend=dict(itemsizing="constant"))
    return fig


# --------------------------------------------------------------------------- #
#  Per-cell: static                                                            #
# --------------------------------------------------------------------------- #
def arbour_static(df_raw, df_aligned, nid, raw_units="nm", aligned_units="nm",
                  max_segments=20000):
    """The same comparison as a PNG-able matplotlib figure, in 3-D."""
    raw = _scale(df_raw, raw_units)
    root = raw[raw["p"] == -1]
    if len(root):
        origin = root.iloc[0][["x", "y", "z"]].to_numpy(float)
        for i, c in enumerate("xyz"):
            raw[c] = raw[c] - origin[i]
    ali = _scale(df_aligned, aligned_units)

    fig = plt.figure(figsize=(13, 6))
    for k, (frame, title) in enumerate(((raw, "raw, centred"),
                                        (ali, "aligned (apical -> +z)")), 1):
        ax = fig.add_subplot(1, 2, k, projection="3d")
        segs, cls = _subsample(*_edges(frame), max_segments)
        if len(segs):
            colours = [CLASS_COLOUR.get(str(c), DEFAULT_COLOUR) for c in cls]
            ax.add_collection3d(Line3DCollection(segs, colors=colours,
                                                 linewidths=0.4))
            P = segs.reshape(-1, 3)
            for setlim, lo, hi in ((ax.set_xlim, P[:, 0].min(), P[:, 0].max()),
                                   (ax.set_ylim, P[:, 1].min(), P[:, 1].max()),
                                   (ax.set_zlim, P[:, 2].min(), P[:, 2].max())):
                if hi - lo < 1e-9:                     # planar or linear arbour
                    lo, hi = lo - 0.5, hi + 0.5
                setlim(lo, hi)
            try:
                span = [max(float(np.ptp(P[:, i])), 1e-9) for i in range(3)]
                ax.set_box_aspect(span)
            except Exception:                          # noqa: BLE001
                pass
        ax.scatter([0], [0], [0], c="#111111", s=18)
        ax.set_title(title)
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)"); ax.set_zlabel("z (um)")
    fig.suptitle("neuron %s -- alignment check" % nid)
    fig.tight_layout()
    return fig


def depth_profile(df_aligned, nid, units="nm", class_column="compartment_class"):
    """Aligned-z distribution by class, plus the cumulative dendritic profile.

    Two reasons this matters beyond looking nice. D-4 assigns apical as the
    subtree maximising z-extent, so the dendritic mass should sit predominantly
    at positive z after alignment. And `fetch_mapped_synapse_indices` masks on
    aligned z for cortical-layer slicing, so this IS the axis downstream cuts on.
    """
    d = _scale(df_aligned, units)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    if class_column in d.columns:
        for klass, g in d.groupby(d[class_column].astype(str)):
            if len(g) < 2:
                continue
            ax.hist(g["z"], bins=60, histtype="step", linewidth=1.4,
                    label="%s (n=%d)" % (klass, len(g)),
                    color=CLASS_COLOUR.get(klass, DEFAULT_COLOUR))
    else:
        ax.hist(d["z"], bins=60, histtype="step", linewidth=1.4)
    ax.axvline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("aligned z (um)"); ax.set_ylabel("nodes")
    ax.set_title("depth distribution by class")
    ax.legend(fontsize=7)

    ax = axes[1]
    dend = (d[d[class_column].astype(str).str.contains("dend")]
            if class_column in d.columns else d)
    if len(dend):
        z = np.sort(dend["z"].to_numpy(float))
        ax.plot(z, np.linspace(0, 1, len(z)), lw=1.6, color="#1f77b4")
        frac = float((z > 0).mean())
        ax.axvline(0.0, color="k", lw=0.8, ls="--")
        ax.set_title("cumulative dendrite depth\n%.1f%% of dendrite at z > 0"
                     % (100 * frac))
    ax.set_xlabel("aligned z (um)"); ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0, 1)
    fig.suptitle("neuron %s -- aligned depth profile" % nid)
    fig.tight_layout()
    return fig


def neighbourhood(soma_pos_nm, metadata_df, diag, nid):
    """Where the k chosen references sit relative to the target soma.

    Left: the bank in the xy plane, target marked, chosen references ringed.
    Right: the distance-ordered bank with the k=chosen cut shown, so it is
    visible whether the choice was a tight local cluster or an arbitrary slice
    of a flat distribution.
    """
    soma_pos_nm = np.asarray(soma_pos_nm, float)
    ref = metadata_df[["soma_x", "soma_y", "soma_z"]].to_numpy(float)
    dist_um = np.linalg.norm(ref - soma_pos_nm, axis=1) / 1000.0
    chosen = np.asarray(diag.get("neighbour_row_indices", []), dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.scatter(ref[:, 0] / 1000.0, ref[:, 1] / 1000.0, s=14, c="#bbbbbb",
               label="bank (n=%d)" % len(ref))
    if len(chosen):
        ax.scatter(ref[chosen, 0] / 1000.0, ref[chosen, 1] / 1000.0, s=90,
                   facecolors="none", edgecolors="#d62728", linewidths=1.6,
                   label="chosen k=%d" % len(chosen))
    ax.scatter([soma_pos_nm[0] / 1000.0], [soma_pos_nm[1] / 1000.0], s=70,
               marker="*", c="#111111", label="target %s" % nid)
    ax.set_xlabel("soma x (um)"); ax.set_ylabel("soma y (um)")
    ax.set_title("reference bank, xy")
    ax.legend(fontsize=8); ax.set_aspect("equal", adjustable="datalim")

    ax = axes[1]
    order = np.argsort(dist_um)
    ax.plot(np.arange(len(order)), dist_um[order], lw=1.4, color="#1f77b4")
    k = len(chosen)
    if k:
        ax.axvline(k - 0.5, color="#d62728", ls="--", lw=1.2,
                   label="k = %d cut" % k)
        ax.scatter(np.arange(k), np.sort(dist_um)[:k], c="#d62728", s=28, zorder=3)
    ax.set_xlabel("bank entry, ordered by distance")
    ax.set_ylabel("distance to target soma (um)")
    ax.set_title("locality of the chosen references")
    ax.legend(fontsize=8)

    spread = diag.get("pairwise_angle_deg_max")
    if spread is not None:
        fig.suptitle("neuron %s -- neighbourhood (max pairwise rotation spread "
                     "%.1f deg)" % (nid, spread))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Batch                                                                       #
# --------------------------------------------------------------------------- #
def _records_frame(records):
    rows = []
    for r in records:
        d = r.get("alignment", {}) or {}
        nd = d.get("neighbour_distance_um") or [float("nan")]
        rows.append({
            "nid": r.get("nid"),
            "qc_status": r.get("qc_status"),
            "n_sections": r.get("n_sections", np.nan),
            "nearest_um": float(np.min(nd)),
            "farthest_chosen_um": float(np.max(nd)),
            "pairwise_spread_deg": d.get("pairwise_angle_deg_max", np.nan),
            "det_error": abs(float(d.get("det", 1.0)) - 1.0),
            "orthonormality_error": d.get("orthonormality_error", np.nan),
            "angle_from_z_deg": r.get("angle_from_z_deg", np.nan),
            "f_implied": r.get("f_implied", np.nan),
            "F_lit": r.get("F_lit", np.nan),
        })
    return pd.DataFrame(rows)


def batch_quality(records):
    """Four panels of alignment quality across the batch."""
    df = _records_frame(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    ax = axes[0, 0]
    ax.hist(df["nearest_um"].dropna(), bins=20, color="#1f77b4")
    ax.set_xlabel("distance to nearest reference (um)"); ax.set_ylabel("cells")
    ax.set_title("how local was the frame")

    ax = axes[0, 1]
    ax.hist(df["pairwise_spread_deg"].dropna(), bins=20, color="#ff7f0e")
    ax.set_xlabel("max pairwise rotation spread among the k (deg)")
    ax.set_ylabel("cells")
    ax.set_title("coherence of the chosen references")

    ax = axes[1, 0]
    ax.scatter(df["nearest_um"], df["pairwise_spread_deg"], s=26,
               c="#2ca02c", alpha=0.8)
    ax.set_xlabel("nearest reference (um)")
    ax.set_ylabel("pairwise spread (deg)")
    ax.set_title("is a closer reference a more coherent one?")

    ax = axes[1, 1]
    err = np.maximum(df[["det_error", "orthonormality_error"]].to_numpy(float),
                     1e-18)
    ax.semilogy(np.arange(len(df)), err[:, 0], "o", ms=5, label="|det - 1|")
    ax.semilogy(np.arange(len(df)), err[:, 1], "s", ms=4,
                label="orthonormality")
    ax.axhline(1e-9, color="#d62728", ls="--", lw=1.0, label="tolerance 1e-9")
    ax.set_xlabel("cell"); ax.set_ylabel("error")
    ax.set_title("the mean really is a rotation")
    ax.legend(fontsize=8)

    fig.suptitle("alignment quality, n = %d cells" % len(df))
    fig.tight_layout()
    return fig


def orientation_consistency(records):
    """Does the batch actually end up pointing the same way?

    The acid test of alignment across cells: each cell's OWN distal dendritic
    direction, recomputed from its aligned frame, plotted as an angle from +z.
    A working alignment concentrates this near 0 deg. A broad or bimodal
    distribution means the reference bank is not delivering a common frame,
    and D-4's apical assignment is standing on sand.
    """
    df = _records_frame(records)
    ang = df["angle_from_z_deg"].to_numpy(float)
    ang = ang[np.isfinite(ang)]

    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 2, 1)
    if len(ang):
        ax.hist(ang, bins=np.linspace(0, 180, 37), color="#1f77b4")
        ax.axvline(float(np.median(ang)), color="#d62728", ls="--", lw=1.4,
                   label="median %.1f deg" % np.median(ang))
        ax.axvline(90.0, color="k", lw=0.8, ls=":", label="90 deg")
        ax.legend(fontsize=8)
    ax.set_xlabel("angle of the aligned arbour from +z (deg)")
    ax.set_ylabel("cells"); ax.set_xlim(0, 180)
    ax.set_title("post-alignment orientation")

    ax = fig.add_subplot(1, 2, 2, projection="polar")
    if len(ang):
        theta = np.radians(ang)
        ax.scatter(theta, np.ones_like(theta), s=40, alpha=0.75, c="#1f77b4")
        ax.set_thetamin(0); ax.set_thetamax(180)
    ax.set_yticklabels([]); ax.set_title("same, polar (0 deg = +z)")

    n_bad = int((ang > 90).sum()) if len(ang) else 0
    fig.suptitle("orientation consistency, n = %d | %d cell(s) beyond 90 deg "
                 "(upside down)" % (len(ang), n_bad))
    fig.tight_layout()
    return fig


REGRESSION_LABELS = ("n_sections", "n_branches", "f_implied", "F_lit",
                     "A_shaft_um2", "A_spine_um2")


def rigidity(regression_records):
    """Per-cell |aligned - unaligned| on every section 7 quantity.

    READ THIS PANEL FIRST. Alignment is rigid, so every bar must be exactly
    zero. Anything above the floor means the transform is not rigid or was
    applied at the wrong step -- not that the plot needs a tolerance.

    `regression_records` is a list of dicts with keys 'nid', 'unaligned',
    'aligned', each of the latter a result dict from the exporter.
    """
    fig, ax = plt.subplots(figsize=(11, 4.6))
    nids = [str(r["nid"]) for r in regression_records]
    width = 0.8 / max(len(REGRESSION_LABELS), 1)
    any_nonzero = False

    for i, key in enumerate(REGRESSION_LABELS):
        vals = []
        for r in regression_records:
            a = r["unaligned"].get(key)
            b = r["aligned"].get(key)
            try:
                v = abs(float(a) - float(b))
            except (TypeError, ValueError):
                v = float("nan")
            vals.append(v)
        vals = np.asarray(vals, float)
        any_nonzero |= bool(np.nansum(vals) > 0)
        ax.bar(np.arange(len(nids)) + i * width, np.maximum(vals, 1e-18),
               width=width, label=key)

    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e2)
    ax.axhline(1e-18, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(nids)) + 0.4 - width / 2)
    ax.set_xticklabels(nids, rotation=45, ha="right")
    ax.set_ylabel("|aligned - unaligned|")
    ax.set_title("rigidity check -- every bar must sit on the floor%s"
                 % ("" if not any_nonzero else "   *** NON-ZERO DELTA ***"))
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  Saving                                                                      #
# --------------------------------------------------------------------------- #
def save_figure(fig, path, dpi=150, close=True):
    """Write a figure. Dispatches on type: plotly -> .html, matplotlib -> image.

    Returns the path actually written, which may differ from `path` if the
    extension did not match the backend.
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    if hasattr(fig, "write_html"):                     # plotly
        if not path.lower().endswith(".html"):
            path = os.path.splitext(path)[0] + ".html"
        fig.write_html(path, include_plotlyjs="cdn")
        return path

    if not os.path.splitext(path)[1]:
        path = path + ".png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path
