"""Figures of the per-node inference -- Block 34 in specs/SPEC.md (plan steps F5, E; D-129.4).

Plotting only: every array arrives prepared (Blocks 30, 32 by contract);
nothing scientific is computed here and no input array is written to. The one
transform drawn, the residual over sigma (I_k(v_n) - B_k tau_k(v_n)) / sigma,
is a display of the inputs.

  figure_fit        one node's fit: per plane shown the cleaned profile and
                    the fitted model B_k tau_k (gl) on one axis, under it the
                    residual over sigma with lines at +-2; planes the fit did
                    not use (fit.planes False) drawn greyed
  figure_diameters  d_fit with its +-se bar against Allen's diameter and
                    against the first-stage diameter (two panels sharing the
                    y axis), the identity line, hollow markers for the rows
                    not converged, classes by marker (and colour) in a legend
  figure_chi2       chi2_k / N per plane and node, a line at 1
  save              the PNG of a figure

Conventions read from SPEC.md: profiles (n_planes, n_v) in grey levels on the
offsets v (um) of Block 5; ks the plane indices k; a NodeFit of Block 30
(theta = (d um, mu 1/um, v0 um, z_ax um), se (4,), planes (n_planes,) bool,
chi2_per_plane (n_planes,), status, at_bound).

Colours: the palette of the project's figures (cleaning_figures). Identity
never rests on colour alone: profile = markers, model = line; classes and
nodes differ by marker. Pure ASCII, for the cluster (hpc-python-compat).
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping

import numpy as np

INK, INK2, BLUE, ORANGE = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
SURFACE, GRID = "#fcfcfb", "#e4e3df"
GREYED = "#b9b8b3"                       # planes the fit did not use
CLASS_COLOURS = ("#2a78d6", "#eb6834", "#1f9e6e", "#8a4fc7", "#c23b5a", "#7a6a2a", "#0b0b0b")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p")
FIT_COLS = 4                             # planes per row of figure_fit before it wraps
RESID_LINE = 2.0                         # the residual lines at +-2 sigma (SPEC Block 34)
DIAM_KEYS = ("node_id", "d_allen_um", "d_fit_um", "se_um", "status")
CHI2_KEYS = ("node_id", "chi2_per_plane", "ks")
PARAM_UNITS = (("d", "um"), ("mu", "/um"), ("v0", "um"), ("z", "um"))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=7)


def _fmt(value):
    if isinstance(value, (float, np.floating)):
        return "%.4g" % float(value)
    return str(value)


def _plane_cols(ks, planes):
    """Positions along the plane axis of the planes to show (plane indices k taken from ks)."""
    ks = [int(k) for k in ks]
    if planes is None:
        return list(range(len(ks)))
    wanted = [int(k) for k in np.asarray(planes).reshape(-1)]
    if not wanted:
        raise ValueError("planes is empty")
    out = []
    for k in wanted:
        if k not in ks:
            raise ValueError("plane %d is not in ks %r" % (k, ks))
        out.append(ks.index(k))
    return out


def _fit_title(meta, fit):
    """Title from meta (node_id, class; each optional) and the fit (d +- se, mu, status)."""
    parts = []
    if "node_id" in meta:
        parts.append("node %s" % _fmt(meta["node_id"]))
    if "class" in meta:
        parts.append("class %s" % _fmt(meta["class"]))
    head = "  |  ".join(parts) if parts else "node"
    theta = np.asarray(fit.theta, dtype=float)
    se = np.asarray(fit.se, dtype=float)
    status = str(fit.status)
    if fit.at_bound:
        status += " (%s)" % ", ".join(str(b) for b in fit.at_bound)
    line = "d = %.4g +- %.2g um,  mu = %.4g +- %.2g /um,  status %s" % (theta[0], se[0], theta[1], se[1], status)
    return head + "\n" + line


def figure_fit(profiles_gl, model_gl, v, ks, sigma_gl, fit, meta, planes=None):
    """One node's fit. Returns the matplotlib Figure.

    profiles_gl, model_gl: (n_planes, n_v) grey levels -- the cleaned profiles
      I_k(v_n) and the fitted model B_k tau_k(v_n; theta_hat).
    v: (n_v,) offsets along the line, um; ks: (n_planes,) plane indices k.
    sigma_gl: the noise SD (gl, finite > 0) the residual is divided by.
    fit: a Block 30 NodeFit (theta, se (4,), planes (n_planes,) bool,
      chi2_per_plane (n_planes,), status, at_bound).
    meta: mapping for the title -- node_id, class; each optional.
    planes: plane indices k (values of ks) to show, in that order; None = all.

    Layout: per shown plane a column of two axes, the profile panel (profile:
    ink markers, model: blue line) over the residual panel ((I - B tau) /
    sigma, lines at +-2); at most FIT_COLS planes per row, rows balanced; profile
    panels share y, residual panels share y. Planes with fit.planes False are
    drawn in GREYED and titled "not used".
    ValueError for shapes that disagree, an empty `planes`, a plane not in ks,
    sigma_gl not finite and > 0, meta not a mapping.
    """
    prof = np.asarray(profiles_gl, dtype=float)
    model = np.asarray(model_gl, dtype=float)
    v = np.asarray(v, dtype=float)
    ks = np.asarray(ks)
    if prof.ndim != 2:
        raise ValueError("profiles_gl must be (n_planes, n_v), got shape %r" % (prof.shape,))
    if model.shape != prof.shape:
        raise ValueError("model_gl %r and profiles_gl %r must have one shape" % (model.shape, prof.shape))
    n_planes, n_v = prof.shape
    if v.ndim != 1 or v.size != n_v:
        raise ValueError("v must be (n_v,) = (%d,), got shape %r" % (n_v, v.shape))
    if ks.ndim != 1 or ks.size != n_planes:
        raise ValueError("ks must be (n_planes,) = (%d,), got shape %r" % (n_planes, ks.shape))
    sigma = float(sigma_gl)
    if not (math.isfinite(sigma) and sigma > 0):
        raise ValueError("sigma_gl must be finite and > 0, got %r" % (sigma_gl,))
    used = np.asarray(fit.planes)
    chi2 = np.asarray(fit.chi2_per_plane, dtype=float)
    if used.shape != (n_planes,) or chi2.shape != (n_planes,):
        raise ValueError("fit.planes %r and fit.chi2_per_plane %r must be (n_planes,) = (%d,)"
                         % (used.shape, chi2.shape, n_planes))
    if np.shape(fit.theta) != (4,) or np.shape(fit.se) != (4,):
        raise ValueError("fit.theta and fit.se must be (4,)")
    if not isinstance(meta, Mapping):
        raise ValueError("meta must be a mapping (node_id, class)")
    cols = _plane_cols(ks, planes)
    resid = (prof - model) / sigma        # display transform of the inputs (SPEC Block 34)

    plt = _plt()
    n = len(cols)
    n_r = int(math.ceil(n / float(FIT_COLS)))
    n_c = int(math.ceil(n / float(n_r)))     # rows balanced: 5 planes as 3 + 2, not 4 + 1
    fig = plt.figure(figsize=(2.9 * n_c + 0.6, 3.5 * n_r + 0.8), facecolor=SURFACE, layout="constrained")
    gs = fig.add_gridspec(2 * n_r, n_c, height_ratios=[2.2, 1.0] * n_r)
    top0 = bot0 = None
    for i, j in enumerate(cols):
        r, c = divmod(i, n_c)
        ax_p = fig.add_subplot(gs[2 * r, c], sharey=top0)
        ax_r = fig.add_subplot(gs[2 * r + 1, c], sharex=ax_p, sharey=bot0)
        top0 = top0 or ax_p
        bot0 = bot0 or ax_r
        _style(ax_p)
        _style(ax_r)
        on = bool(used[j])
        c_obs, c_mod, c_res, c_lim = (INK, BLUE, INK2, ORANGE) if on else (GREYED,) * 4
        ax_p.plot(v, prof[j], ls="none", marker="o", ms=2.8, color=c_obs, label="profile (cleaned)", gid="profile")
        ax_p.plot(v, model[j], ls="-", lw=1.4, color=c_mod, label="model B tau", gid="model")
        ax_r.axhline(0.0, color=GRID, lw=1.0, zorder=1)
        for s in (-RESID_LINE, RESID_LINE):
            ax_r.axhline(s, color=c_lim, lw=1.0, ls="--", gid="resid_limit")
        ax_r.plot(v, resid[j], ls="-", lw=0.8, marker=".", ms=3, color=c_res, gid="residual")
        ax_r.set_xlim(v[0], v[-1])
        if on:
            ax_p.set_title("k = %d,  chi2/N = %.3g" % (int(ks[j]), chi2[j]), fontsize=8.5, color=INK)
        else:
            ax_p.set_title("k = %d  (not used)" % int(ks[j]), fontsize=8.5, color=GREYED)
        ax_r.set_xlabel("v (um)", fontsize=7.5, color=INK2)
        if c == 0:
            ax_p.set_ylabel("grey level", fontsize=7.5, color=INK2)
            ax_r.set_ylabel("residual / sigma", fontsize=7.5, color=INK2)
        if i == 0:
            ax_p.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle(_fit_title(meta, fit) + "\nsigma = %.3g gl; dashed: +-%g" % (sigma, RESID_LINE),
                 fontsize=9.5, color=INK)
    return fig


def _check_rows(rows, keys, what):
    if isinstance(rows, Mapping) or rows is None or len(rows) == 0:
        raise ValueError("%s: rows must be a non-empty list of mappings" % what)
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("%s: row %d is not a mapping" % (what, i))
        missing = [k for k in keys if k not in row]
        if missing:
            raise ValueError("%s: row %d lacks the key(s) %r" % (what, i, missing))


def _given(value):
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _class_styles(rows):
    """Class name -> (marker, colour), classes sorted by name (deterministic)."""
    names = sorted({str(r.get("class", "unclassified")) for r in rows})
    return {name: (MARKERS[i % len(MARKERS)], CLASS_COLOURS[i % len(CLASS_COLOURS)]) for i, name in enumerate(names)}


def figure_diameters(rows, cfg=None):
    """The fitted diameters against Allen's and the first stage's. Returns the matplotlib Figure.

    rows: non-empty list of mappings, each with node_id, d_allen_um, d_fit_um,
      se_um (um) and status, and optionally d_pilot_um (the first-stage
      diameter, um; None or NaN = not given) and class.
    cfg: a DiameterConfig or None. Display only: when given, the fit's
      diameter bounds cfg.inference.d_bounds_um that fall within the drawn
      range are drawn as dotted lines (and listed in the legend).

    Two panels sharing the y axis (d_fit): against d_allen_um (all rows) and
    against d_pilot_um (the rows that give it). In each, one marker per row
    with its +-se bar (rows whose se is not finite: no bar), the identity line
    y = x, classes by marker and colour (rows without class: "unclassified"),
    hollow markers for status != "converged"; a legend lists the classes and
    the hollow convention.
    ValueError for empty rows or a row without a required key.
    """
    _check_rows(rows, DIAM_KEYS, "figure_diameters")
    styles = _class_styles(rows)
    plt = _plt()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.9), sharey=True, facecolor=SURFACE, layout="constrained")
    xkeys = (("d_allen_um", "Allen's diameter (um)"), ("d_pilot_um", "first-stage diameter (um)"))
    bounds_drawn = False
    y_all = np.array([float(r["d_fit_um"]) for r in rows], dtype=float)
    for ax, (xkey, xlabel) in zip(axes, xkeys):
        _style(ax)
        sel = [r for r in rows if _given(r.get(xkey))]
        vals = list(y_all[np.isfinite(y_all)])
        for name in sorted(styles):
            marker, colour = styles[name]
            for conv in (True, False):
                grp = [r for r in sel if str(r.get("class", "unclassified")) == name
                       and (str(r["status"]) == "converged") == conv]
                if not grp:
                    continue
                x = np.array([float(r[xkey]) for r in grp], dtype=float)
                y = np.array([float(r["d_fit_um"]) for r in grp], dtype=float)
                se = np.array([float(r["se_um"]) if r["se_um"] is not None else np.nan for r in grp], dtype=float)
                bar = np.isfinite(se) & np.isfinite(x) & np.isfinite(y)
                if np.any(bar):
                    ax.errorbar(x[bar], y[bar], yerr=se[bar], fmt="none", ecolor=colour, elinewidth=0.9,
                                capsize=2, alpha=0.8, zorder=2)
                ax.plot(x, y, ls="none", marker=marker, ms=6, mec=colour, mew=1.3,
                        mfc=colour if conv else "none", zorder=3,
                        gid="data|%s|%s" % (name, "converged" if conv else "not_converged"))
                vals.extend(list(x[np.isfinite(x)]))
        if vals:
            lo, hi = float(min(vals)), float(max(vals))
            pad = 0.05 * (hi - lo) if hi > lo else 0.1
            ends = [max(lo - pad, 0.0), hi + pad]
        else:
            ends = [0.0, 1.0]
        ax.plot(ends, ends, color=INK2, lw=1.0, ls="--", zorder=1, gid="identity")
        if cfg is not None:
            for b in cfg.inference.d_bounds_um:
                if ends[0] <= b <= ends[1]:
                    ax.axhline(b, color=ORANGE, lw=0.9, ls=":", gid="d_bound")
                    bounds_drawn = True
        ax.set_xlim(ends)
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
        if not sel:
            ax.text(0.5, 0.5, "no row gives %s" % xkey, transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.5, color=INK2)
        ax.set_title("%d node%s" % (len(sel), "" if len(sel) == 1 else "s"), fontsize=8.5, color=INK)
    axes[0].set_ylabel("fitted diameter d_fit +- se (um)", fontsize=8.5, color=INK2)
    handles = [Line2D([], [], ls="none", marker=styles[n][0], ms=6, mec=styles[n][1], mfc=styles[n][1], label=n)
               for n in sorted(styles)]
    handles.append(Line2D([], [], ls="none", marker="o", ms=6, mec=INK2, mfc="none", label="not converged (hollow)"))
    handles.append(Line2D([], [], color=INK2, lw=1.0, ls="--", label="identity y = x"))
    if bounds_drawn:
        handles.append(Line2D([], [], color=ORANGE, lw=0.9, ls=":", label="fit bounds on d"))
    fig.legend(handles=handles, loc="outside right upper", fontsize=7.5, frameon=False)
    fig.suptitle("per-node fit: fitted diameter against Allen's and the first stage's", fontsize=9.5, color=INK)
    return fig


def figure_chi2(rows):
    """chi2_k / N per plane and node. Returns the matplotlib Figure.

    rows: non-empty list of mappings, each with node_id, chi2_per_plane
      (n_planes,) (NaN for planes not used, Block 30) and ks (n_planes,) the
      plane indices.
    One line per node, x = k, y = chi2_k / N, nodes by marker and colour and
    listed in a legend; a dashed line at 1. The y axis is logarithmic when the
    finite values are all > 0 and span more than a factor 30.
    ValueError for empty rows, a row without a required key, shapes that
    disagree.
    """
    _check_rows(rows, CHI2_KEYS, "figure_chi2")
    series = []
    for i, row in enumerate(rows):
        k = np.asarray(row["ks"])
        c = np.asarray(row["chi2_per_plane"], dtype=float)
        if k.ndim != 1 or c.shape != k.shape or k.size == 0:
            raise ValueError("figure_chi2: row %d: ks %r and chi2_per_plane %r must be one non-empty (n_planes,)"
                             % (i, k.shape, c.shape))
        series.append((row["node_id"], k, c))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=SURFACE, layout="constrained")
    _style(ax)
    finite = []
    for i, (nid, k, c) in enumerate(series):
        ax.plot(k, c, ls="-", lw=0.9, marker=MARKERS[i % len(MARKERS)], ms=5,
                color=CLASS_COLOURS[i % len(CLASS_COLOURS)], label="node %s" % _fmt(nid), gid="chi2|%d" % i)
        finite.extend(list(c[np.isfinite(c)]))
    ax.axhline(1.0, color=INK2, lw=1.0, ls="--", gid="unity", label="chi2/N = 1")
    if finite and min(finite) > 0 and max(finite) / min(finite) > 30.0:
        ax.set_yscale("log")
    ks_all = sorted({int(x) for _, k, _ in series for x in k})
    ax.set_xticks(ks_all)
    ax.set_xlabel("plane k", fontsize=8.5, color=INK2)
    ax.set_ylabel("chi2_k / N (mean squared residual / sigma^2)", fontsize=8.5, color=INK2)
    ncol = 1 + (len(series) - 1) // 14
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=ncol)
    ax.set_title("per-plane chi2 / N of the per-node fits (%d node%s)" % (len(series), "" if len(series) == 1
                                                                          else "s"), fontsize=9.5, color=INK)
    return fig


def save(fig, path, dpi):
    """Write fig as PNG at dpi (> 0) to path, creating its folder. Returns the path.
    The figure stays open: the caller closes it."""
    if not (float(dpi) > 0):
        raise ValueError("dpi must be > 0, got %r" % (dpi,))
    path = os.fspath(path)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    return path
