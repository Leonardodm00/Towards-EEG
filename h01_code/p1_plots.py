#!/usr/bin/env python3
"""p1_plots v1.0 -- figures from the artefacts of one P1 export tree.

Three families of figure, from the files `run_p1_export.py` writes under
<out_dir>/<cell_id>/ (see README section 8) and the raw skeleton it read:

  pruning       the ALIGNED skeleton before and after spine pruning
                (raw nodes rotated with the cell's own alignment record;
                pruned-spine nodes taken from neuron_<id>_spine_nodes.csv)
  downsampling  the raw aligned skeleton next to the LFPy compartments the
                d_lambda rule produced from it, with every incoming synapse
                drawn at its aligned position, coloured exc / inh, and joined
                to the compartment it was snapped to (lfpy_idx). Needs NEURON +
                LFPy: the compartments are rebuilt from the exported .hoc with
                the (cm, Ra, lambda_f, d_lambda) of the record, exactly as a
                downstream simulation must
  stats         spine statistics (neck radius, head radius, neck length,
                R_neck, density along the path distance, synapses per spine,
                partition source) and arbour profiles (cable, diameter, spine
                area per path-distance bin) for one cell; plus a population
                figure from p1_summary.csv, optionally paired against a second
                tree's summary (the D-003 SST / PV-VIP comparison)

WHICH phi THE ARBOUR FIGURE DRAWS. Two different tables carry the name:

  <out_dir>/<id>/neuron_<id>_phi.csv        P1, spine area from SKELETON
                                            frustums -- the FALLBACK model;
                                            its F is F_skel
  <root>/out/neuron_<id>_phi_mesh.csv       P3, the DELIVERABLE: same rows,
                                            same shaft areas (always skeleton
                                            frustums), spine_area_um2 REPLACED
                                            by the mesh measurement
                                            (h01_spine_area_F.phi_with_spine_areas,
                                            variant mesh_beyond); its F is the
                                            deliverable F

The standing decision is spine area from the MESH, skeleton only as fallback,
shaft area ALWAYS from skeleton frustums. `--phi auto` (the default) therefore
uses the P3 mesh table when it exists and falls back to P1's skeleton table
when it does not; every panel and the title say WHICH, and the fallback is
labelled as a fallback rather than as the spine area. `--phi skel` / `--phi
mesh` force the choice (mesh is refused if the file is absent).

Layout of this file: loaders (I/O only) -> transforms (pure functions on
frames) -> plotting (matplotlib only) -> CLI. Nothing in the plotting layer
reads a file; nothing in the loaders knows about axes.

Usage (from h01_code, env spine_env):

  python3 p1_plots.py --root ../h01 --out-dir ../h01/p1_inh_SST \\
          --cell 489469961 --fig-dir ../h01/figures
  python3 p1_plots.py --root ../h01 --out-dir ../h01/p1 --population \\
          --compare ../h01/p1_inh_PVVIP --fig-dir ../h01/figures

Coordinates: every skeleton figure is in the ALIGNED frame, micrometres,
soma at the origin, +z the cortical axis the alignment bank defines
(alignment.aligned_um). `--proj xz` (default) shows the depth axis vertical;
`--proj 3d` draws the arbour in three dimensions on a cubic box, viewed from
(`--elev`, `--azim`). Mark sizes come from STYLE and are tuned with
`--lw-scale` and `--syn-size`; the default output is 220 dpi.
"""
from __future__ import print_function

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

PLOTS_VERSION = "p1_plots v1.2"

# --------------------------------------------------------------------------- #
# palette -- categorical hues in fixed order (validated: all-pairs, light)     #
# --------------------------------------------------------------------------- #
COL_INH = "#2a78d6"     # blue
COL_EXC = "#eb6834"     # orange
COL_SPINE = "#4a3aa7"   # violet: nodes removed by pruning
COL_SHAFT = "#52514e"   # dark grey: kept skeleton
COL_AXON = "#b0afa8"    # light grey
COL_COMP = ("#52514e", "#a09f99")   # alternating compartments
COL_SOMA = "#0b0b0b"

PROJECTIONS = {"xz": ("x", "z"), "yz": ("y", "z"), "xy": ("x", "y"), "3d": None}
_AXIS = {"x": 0, "y": 1, "z": 2}

# Mark sizes, in one place. `--lw-scale` multiplies every line width and
# `--syn-size` sets the shaft-synapse marker (the pruned-spine marker keeps
# its ratio to it), so the balance between skeleton and synapses is one knob
# each rather than a hunt through the plotting code.
STYLE = {"lw_shaft": 1.1, "lw_spine": 1.9, "lw_axon": 0.7, "lw_conn": 0.4,
         # syn_size is a marker DIAMETER in points; matplotlib's `s` is an
         # AREA in points^2, so the plotting code squares it. v1.0 passed
         # s = 9 / 16 directly, i.e. diameters 3.0 / 4.0 -- these are smaller.
         "syn_size": 2.2, "syn_spine_ratio": 1.6, "soma_size": 5.0,
         "comp_lw_scale": 2.0, "comp_lw_min": 0.8, "comp_lw_max": 7.0,
         "elev": 18.0, "azim": -70.0}


def resolve_style(**over):
    """STYLE with overrides; `lw_scale` multiplies every line width."""
    st = dict(STYLE)
    scale = float(over.pop("lw_scale", 1.0) or 1.0)
    for k, v in over.items():
        if v is not None:
            st[k] = v
    for k in ("lw_shaft", "lw_spine", "lw_axon", "lw_conn", "comp_lw_scale",
              "comp_lw_min", "comp_lw_max"):
        st[k] = float(st[k]) * scale
    return st


# --------------------------------------------------------------------------- #
# 1. loaders -- I/O only                                                      #
# --------------------------------------------------------------------------- #
def cell_dir(out_dir, cid):
    return os.path.join(out_dir, str(int(cid)))


def _artefact(out_dir, cid, suffix):
    p = os.path.join(cell_dir(out_dir, cid), "neuron_%d_%s" % (int(cid), suffix))
    if not os.path.isfile(p):
        raise SystemExit("missing artefact: %s" % p)
    return p


def load_raw_skeleton(neurons_dir, cid):
    """The H01 skeleton P1 read: id, p, x, y, z (nm), r (nm), annotated_type."""
    p = os.path.join(neurons_dir, "neuron_%d.csv" % int(cid))
    if not os.path.isfile(p):
        raise SystemExit("missing skeleton: %s" % p)
    df = pd.read_csv(p, usecols=["id", "p", "x", "y", "z", "r", "annotated_type"])
    return df


def load_record(out_dir, cid):
    return json.load(open(_artefact(out_dir, cid, "p1.json")))


def load_alignment(out_dir, cid):
    return json.load(open(_artefact(out_dir, cid, "alignment.json")))


def load_spine_nodes(out_dir, cid):
    return pd.read_csv(_artefact(out_dir, cid, "spine_nodes.csv"))


def load_spine_stats(out_dir, cid):
    return pd.read_csv(_artefact(out_dir, cid, "spine_stats.csv"))


def load_mapped_synapses(out_dir, cid):
    p = os.path.join(cell_dir(out_dir, cid), "neuron_%d_mapped_synapses.csv" % int(cid))
    if not os.path.isfile(p):        # exported without a synapse file
        return pd.DataFrame(columns=["x", "y", "z", "synapse_type", "lfpy_idx",
                                     "on_pruned_spine", "anchor_x", "anchor_y", "anchor_z"])
    return pd.read_csv(p)


def load_phi(out_dir, cid):
    """P1's phi: spine area from SKELETON frustums (the fallback model)."""
    return pd.read_csv(_artefact(out_dir, cid, "phi.csv"))


def load_phi_mesh(root, cid, out_subdir="out"):
    """P3's deliverable phi (`<root>/out/neuron_<id>_phi_mesh.csv`), or None if
    P2/P3 have not run for this cell. Same rows and the same shaft areas as
    P1's table; `spine_area_um2` is the MESH measurement and the skeleton
    value it replaced is kept as `spine_area_skel_um2`
    (h01_spine_area_F.phi_with_spine_areas)."""
    p = os.path.join(root, out_subdir, "neuron_%d_phi_mesh.csv" % int(cid))
    return pd.read_csv(p) if os.path.isfile(p) else None


def resolve_phi(root, out_dir, cid, mode="mesh"):
    """(phi, label, is_mesh) for the arbour figure. `mode` is auto | mesh | skel.
    A mesh table missing its marker column is refused rather than drawn as
    though it were one."""
    if mode not in ("auto", "mesh", "skel"):
        raise SystemExit("--phi must be auto, mesh or skel")
    if mode != "skel":
        m = load_phi_mesh(root, cid)
        if m is not None:
            if "spine_area_skel_um2" not in m.columns:
                raise SystemExit(
                    "phi_mesh for cell %d has no spine_area_skel_um2 column -- "
                    "it was not produced by phi_with_spine_areas" % int(cid))
            return m, "mesh (P3 deliverable, mesh_beyond)", True
        if mode == "mesh":
            raise SystemExit(
                "no P3 phi_mesh for cell %d under %s/out. F is reported from the "
                "mesh_beyond variant by decision; the skeleton table is a comparison "
                "bracket, not the deliverable. Run P2/P3, or pass --phi skel to see "
                "the bracket, labelled as one." % (int(cid), root))
    return load_phi(out_dir, cid), "SKELETON FALLBACK (P1; mesh not measured yet)", False


def load_summary(out_dir):
    p = os.path.join(out_dir, "p1_summary.csv")
    if not os.path.isfile(p):
        raise SystemExit("missing %s -- run run_p1_export.py --summarise --out-dir %s first"
                         % (p, out_dir))
    return pd.read_csv(p)


_PT3D = re.compile(r"^\s*pt3dadd\(\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)")
_SEC = re.compile(r"^\s*(\w+)\[(\d+)\]\s*\{")


def parse_hoc_pt3d(path):
    """Every pt3dadd of the exported .hoc: section array, index, x, y, z, d (um).
    The exported tree, exactly as NEURON will build it."""
    rows, sec = [], (None, -1)
    for line in open(path):
        m = _SEC.match(line)
        if m:
            sec = (m.group(1), int(m.group(2)))
            continue
        m = _PT3D.match(line)
        if m and sec[0] is not None:
            rows.append((sec[0], sec[1]) + tuple(float(m.group(k)) for k in range(1, 5)))
    return pd.DataFrame(rows, columns=["array", "type_idx", "x", "y", "z", "d"])


def build_lfpy_cell(hoc_path, segmentation, stage1_dir=None):
    """Rebuild the LFPy cell the snapper indexed against, from the record's
    `result.segmentation` (cm, Ra, lambda_f, d_lambda, nsegs_method).
    Needs NEURON + LFPy; raises ImportError otherwise."""
    if stage1_dir and stage1_dir not in sys.path:
        sys.path.insert(0, stage1_dir)
    import alignment                                   # stage1 module
    return alignment.default_cell_factory(
        hoc_path, cm=float(segmentation["cm"]), Ra=float(segmentation["Ra"]),
        lambda_f=float(segmentation.get("lambda_f", 100.0)),
        d_lambda=float(segmentation.get("d_lambda", 0.1)),
        nsegs_method=segmentation.get("nsegs_method", "lambda_f"))


# --------------------------------------------------------------------------- #
# 2. transforms -- pure functions                                             #
# --------------------------------------------------------------------------- #
def align_raw(raw, alignment_json):
    """Apply the cell's own alignment (soma_pos_nm, mean_matrix) to the raw
    skeleton: centre on the soma, rotate, convert to um. Same transform as
    alignment.aligned_um, written out here so this file does not import the
    exporter for a matrix product."""
    al = alignment_json["alignment"]
    soma = np.asarray(al["soma_pos_nm"], float)
    Rm = np.asarray(al["mean_matrix"], float)
    if abs(np.linalg.det(Rm) - 1.0) > 1e-6:
        raise ValueError("mean_matrix det %.6f != 1" % np.linalg.det(Rm))
    xyz = (raw[["x", "y", "z"]].to_numpy(float) - soma) @ Rm.T / 1000.0
    out = raw.copy()
    out["x_al_um"], out["y_al_um"], out["z_al_um"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["r_um"] = out["r"].astype(float) / 1000.0
    return out


def split_pruned(raw_al, spine_nodes):
    """(kept, pruned): the raw aligned frame split by membership of the pruned
    spine node set. Demoted continuations are NOT in spine_nodes, so they
    stay in `kept`, which is what the exporter did."""
    ids = set(int(v) for v in spine_nodes["node_id"].tolist())
    mask = raw_al["id"].isin(ids)
    return raw_al[~mask].copy(), raw_al[mask].copy()


def frame_segments(df, xcol="x_al_um", ycol="y_al_um", zcol="z_al_um"):
    """Parent->child line segments of a skeleton frame, (N, 2, 3) in the
    frame's own coordinates; rows whose parent is not in the frame are
    skipped (the root, and the base of a pruned spine when the frame is the
    pruned set alone)."""
    pos = df.set_index("id")[[xcol, ycol, zcol]]
    child = df[df["p"].isin(pos.index)]
    a = pos.loc[child["p"].to_numpy()].to_numpy(float)
    b = pos.loc[child["id"].to_numpy()].to_numpy(float)
    return np.stack([a, b], axis=1), child


def compartments_from_cell(cell):
    """One row per LFPy compartment: x0..z1 (um), d (um), section name.
    Handles LFPy >= 2.2 (cell.x of shape (totnsegs, 2)) and the older
    xstart/xend attributes."""
    if hasattr(cell, "x") and np.ndim(cell.x) == 2:
        x0, x1 = cell.x[:, 0], cell.x[:, -1]
        y0, y1 = cell.y[:, 0], cell.y[:, -1]
        z0, z1 = cell.z[:, 0], cell.z[:, -1]
    else:
        x0, x1, y0, y1, z0, z1 = (cell.xstart, cell.xend, cell.ystart, cell.yend,
                                  cell.zstart, cell.zend)
    n = len(x0)
    names = []
    for i in range(n):
        try:
            names.append(str(cell.get_idx_name(i)[1]))
        except Exception:
            names.append("")
    return pd.DataFrame({"idx": np.arange(n), "x0": x0, "y0": y0, "z0": z0,
                         "x1": x1, "y1": y1, "z1": z1,
                         "d": np.asarray(cell.d, float), "sec": names})


def compartment_midpoints(comps):
    return pd.DataFrame({"x": (comps["x0"] + comps["x1"]) / 2.0,
                         "y": (comps["y0"] + comps["y1"]) / 2.0,
                         "z": (comps["z0"] + comps["z1"]) / 2.0}, index=comps["idx"].to_numpy())


def synapse_connectors(syn, midpoints):
    """(N, 2, 3) segments from each synapse to the midpoint of the compartment
    it was snapped to (lfpy_idx). A synapse on a pruned spine is drawn from
    its ANCHOR, which is where the snap started (alignment.snap_synapses)."""
    if len(syn) == 0:
        return np.zeros((0, 2, 3)), syn
    s = syn[syn["lfpy_idx"].isin(midpoints.index)].copy()
    on = s["on_pruned_spine"].astype(bool).to_numpy() if "on_pruned_spine" in s else np.zeros(len(s), bool)
    start = np.array(s[["x", "y", "z"]].to_numpy(float), copy=True)
    if on.any():
        start[on] = s.loc[on, ["anchor_x", "anchor_y", "anchor_z"]].to_numpy(float)
    end = midpoints.loc[s["lfpy_idx"].to_numpy()][["x", "y", "z"]].to_numpy(float)
    return np.stack([start, end], axis=1), s


def spine_density_profile(spine_stats, phi, bin_um=20.0):
    """Spines per um of shaft cable in bins of path distance from the soma.
    Numerator: spine bases by d_from_um (spine_stats). Denominator: shaft
    cable per bin (phi.seg_len_um). Returns one row per bin."""
    edges = np.arange(0.0, float(max(phi["d_from_um"].max(), spine_stats["d_from_um"].max()
                                     if len(spine_stats) else 0.0)) + bin_um, bin_um)
    cable, _ = np.histogram(phi["d_from_um"].to_numpy(float), bins=edges,
                            weights=phi["seg_len_um"].to_numpy(float))
    n_sp, _ = np.histogram(spine_stats["d_from_um"].to_numpy(float), bins=edges) \
        if len(spine_stats) else (np.zeros(len(edges) - 1), edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        dens = np.where(cable > 0, n_sp / cable, np.nan)
    return pd.DataFrame({"d_lo_um": edges[:-1], "d_hi_um": edges[1:], "n_spines": n_sp,
                         "cable_um": cable, "spines_per_um": dens})


def arbour_profile(phi, bin_um=20.0):
    """Per path-distance bin: shaft cable (um), length-weighted mean shaft
    diameter (um), shaft area, spine area, and their ratio."""
    d = phi["d_from_um"].to_numpy(float)
    edges = np.arange(0.0, float(d.max()) + bin_um, bin_um)
    L = phi["seg_len_um"].to_numpy(float)
    cable, _ = np.histogram(d, bins=edges, weights=L)
    diam_w, _ = np.histogram(d, bins=edges, weights=L * phi["shaft_diam_um"].to_numpy(float))
    a_sh, _ = np.histogram(d, bins=edges, weights=phi["shaft_area_um2"].to_numpy(float))
    a_sp, _ = np.histogram(d, bins=edges, weights=phi["spine_area_um2"].to_numpy(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        diam = np.where(cable > 0, diam_w / cable, np.nan)
        ratio = np.where(a_sh > 0, a_sp / a_sh, np.nan)
    return pd.DataFrame({"d_lo_um": edges[:-1], "d_hi_um": edges[1:], "cable_um": cable,
                         "shaft_diam_um": diam, "A_shaft_um2": a_sh, "A_spine_um2": a_sp,
                         "A_spine_over_A_shaft": ratio})


def pair_summaries(a, b, key="cell_id"):
    """Inner join of two trees' summaries on cell id, suffixed _a / _b."""
    return a.merge(b, on=key, suffixes=("_a", "_b"))


# --------------------------------------------------------------------------- #
# 3. plotting -- matplotlib only                                              #
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": False})
    return plt, LineCollection


def is3d(proj):
    return proj == "3d"


def _panels(fig, ncols, proj):
    """One row of `ncols` panels, 3-D when asked. 3-D axes cannot come from
    fig.subplots, and they do not share limits, so `_equalise` sets both."""
    if is3d(proj):
        return [fig.add_subplot(1, ncols, i + 1, projection="3d")
                for i in range(ncols)]
    return list(np.atleast_1d(fig.subplots(1, ncols, sharex=True, sharey=True)))


def _lc(ax, segs, proj, color, lw, alpha=1.0, zorder=1, **kw):
    """Draw (N, 2, 3) segments: all three coordinates in 3-D, two of them
    otherwise."""
    if len(segs) == 0:
        return
    if is3d(proj):
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        lc = Line3DCollection(np.asarray(segs, float), colors=color,
                              linewidths=lw, alpha=alpha, zorder=zorder, **kw)
        ax.add_collection3d(lc)
        return
    _, LineCollection = _mpl()
    ia, ib = (_AXIS[c] for c in PROJECTIONS[proj])
    lc = LineCollection(np.asarray(segs, float)[:, :, [ia, ib]], colors=color,
                        linewidths=lw, alpha=alpha, zorder=zorder, **kw)
    ax.add_collection(lc)


def _points(ax, xyz, proj, **kw):
    """Scatter (N, 3) points in whichever projection is in force."""
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    if len(xyz) == 0:
        return None
    if is3d(proj):
        kw.setdefault("depthshade", False)
        return ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], **kw)
    ia, ib = (_AXIS[c] for c in PROJECTIONS[proj])
    return ax.scatter(xyz[:, ia], xyz[:, ib], **kw)


def _label_axes(ax, proj, ylabel=True):
    if is3d(proj):
        ax.set_xlabel("x (um)", labelpad=-4)
        ax.set_ylabel("y (um)", labelpad=-4)
        ax.set_zlabel("z (um, cortical axis)", labelpad=-4)
        ax.tick_params(labelsize=7, pad=-2)
        return
    a, b = PROJECTIONS[proj]
    ax.set_xlabel("%s (um, aligned)" % a)
    if ylabel:
        ax.set_ylabel("%s (um, aligned)" % b)


def _equalise(axes, pts, proj, style, pad_frac=0.03):
    """Isotropic limits over every panel, from the (N, 3) point cloud `pts`.
    In 3-D a cube of the largest range, so no axis is silently stretched."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    if is3d(proj):
        ctr = (lo + hi) / 2.0
        half = max(float((hi - lo).max()), 1.0) * (1.0 + pad_frac) / 2.0
        for ax in axes:
            ax.set_xlim(ctr[0] - half, ctr[0] + half)
            ax.set_ylim(ctr[1] - half, ctr[1] + half)
            ax.set_zlim(ctr[2] - half, ctr[2] + half)
            ax.set_box_aspect((1.0, 1.0, 1.0))
            ax.view_init(elev=float(style["elev"]), azim=float(style["azim"]))
        return
    ia, ib = (_AXIS[c] for c in PROJECTIONS[proj])
    pad = pad_frac * max(hi[ia] - lo[ia], hi[ib] - lo[ib], 1.0)
    for ax in axes:
        ax.set_aspect("equal")
    axes[0].set_xlim(lo[ia] - pad, hi[ia] + pad)
    axes[0].set_ylim(lo[ib] - pad, hi[ib] + pad)


def _soma(ax, proj, style):
    _points(ax, [[0.0, 0.0, 0.0]], proj, s=float(style["soma_size"]) ** 2,
            c=COL_SOMA, edgecolors="none", zorder=7)


def _xyz(df, cols=("x_al_um", "y_al_um", "z_al_um")):
    return df[list(cols)].to_numpy(float)


def plot_pruning(kept, pruned, proj, cid, record, style=None, fig=None):
    """Two panels on identical limits: the aligned skeleton with the pruned
    spine nodes highlighted, and the pruned skeleton alone. `proj` is one of
    xz / yz / xy / 3d."""
    plt, _ = _mpl()
    style = style or resolve_style()
    fig = fig or plt.figure(figsize=(12, 5.8) if is3d(proj) else (11, 5.5))
    axes = _panels(fig, 2, proj)
    full = pd.concat([kept, pruned])
    segs_all, child = frame_segments(full)
    is_axon = child["annotated_type"].astype(str).str.lower().str.startswith("axon").to_numpy()
    segs_kept, child_k = frame_segments(kept)
    axon_k = child_k["annotated_type"].astype(str).str.lower().str.startswith("axon").to_numpy()
    # spine segments: the parent of a spine root is a shaft node, so they are
    # taken from `full` rather than from `pruned` alone
    spine_ids = set(pruned["id"].tolist())
    is_spine = child["id"].isin(spine_ids).to_numpy()
    _lc(axes[0], segs_all[is_axon], proj, COL_AXON, style["lw_axon"], zorder=1)
    _lc(axes[0], segs_all[~is_axon & ~is_spine], proj, COL_SHAFT, style["lw_shaft"], zorder=2)
    _lc(axes[0], segs_all[is_spine], proj, COL_SPINE, style["lw_spine"], zorder=3)
    _lc(axes[1], segs_kept[axon_k], proj, COL_AXON, style["lw_axon"], zorder=1)
    _lc(axes[1], segs_kept[~axon_k], proj, COL_SHAFT, style["lw_shaft"], zorder=2)
    for ax in axes:
        _soma(ax, proj, style)
        _label_axes(ax, proj, ylabel=(ax is axes[0]) or is3d(proj))
    n_sp = int((record.get("spine_tables") or {}).get("n_spines", len(pruned)))
    axes[0].set_title("before pruning: %d nodes, %d spine nodes in %d spines"
                      % (len(full), len(pruned), n_sp), fontsize=9)
    axes[1].set_title("after pruning: %d nodes kept (demoted continuations stay)"
                      % len(kept), fontsize=9)
    _equalise(axes, _xyz(full), proj, style)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], color=COL_SHAFT, lw=2, label="shaft (kept)"),
                            Line2D([], [], color=COL_SPINE, lw=2, label="spine nodes (pruned)"),
                            Line2D([], [], color=COL_AXON, lw=2, label="axon")],
                   loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("cell %d -- P1 pruning, aligned frame (%s)" % (cid, proj), fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return fig


def _draw_synapses(ax, syn, proj, style, connectors=None):
    """Incoming synapses at their aligned positions: filled disc on the shaft,
    hollow triangle when the synapse sat on a spine that pruning removed. With
    `connectors`, the thin lines to the compartment each was snapped to."""
    if connectors is not None and len(connectors):
        _lc(ax, connectors, proj, "#7a7975", style["lw_conn"], alpha=0.8, zorder=5)
    if not len(syn):
        return
    s_shaft = float(style["syn_size"]) ** 2
    s_spine = (float(style["syn_size"]) * float(style["syn_spine_ratio"])) ** 2
    for label, col in (("exc", COL_EXC), ("inh", COL_INH)):
        sel = syn[syn["synapse_type"].astype(str) == label]
        if not len(sel):
            continue
        on = (sel["on_pruned_spine"].astype(bool).to_numpy()
              if "on_pruned_spine" in sel else np.zeros(len(sel), bool))
        xyz = sel[["x", "y", "z"]].to_numpy(float)
        _points(ax, xyz[~on], proj, s=s_shaft, c=col, edgecolors="none", zorder=6,
                label="%s on shaft (%d)" % (label, int((~on).sum())))
        if on.any():
            _points(ax, xyz[on], proj, s=s_spine, facecolors="none", edgecolors=col,
                    linewidths=0.7, marker="^", zorder=6,
                    label="%s on pruned spine (%d)" % (label, int(on.sum())))


def plot_downsampling(raw_al, comps, syn, proj, cid, segmentation, style=None, fig=None):
    """Left: the raw aligned skeleton, every node, with the incoming synapses.
    Right: the LFPy compartments the d_lambda rule produced from the exported
    .hoc under the record's (cm, Ra, lambda_f, d_lambda) -- line width scaled
    by compartment diameter, alternating shades so each is visible -- with
    every synapse joined to the compartment its lfpy_idx names."""
    plt, _ = _mpl()
    style = style or resolve_style()
    fig = fig or plt.figure(figsize=(13, 6.0) if is3d(proj) else (12, 6))
    axes = _panels(fig, 2, proj)
    segs, child = frame_segments(raw_al)
    is_axon = child["annotated_type"].astype(str).str.lower().str.startswith("axon").to_numpy()
    _lc(axes[0], segs[is_axon], proj, COL_AXON, style["lw_axon"], zorder=1)
    _lc(axes[0], segs[~is_axon], proj, COL_SHAFT, style["lw_shaft"], zorder=2)
    _draw_synapses(axes[0], syn, proj, style)
    axes[0].set_title("raw skeleton, %d nodes; %d incoming synapses"
                      % (len(raw_al), len(syn)), fontsize=9)
    csegs = np.stack([comps[["x0", "y0", "z0"]].to_numpy(float),
                      comps[["x1", "y1", "z1"]].to_numpy(float)], axis=1)
    is_ax_c = comps["sec"].astype(str).str.startswith("axon").to_numpy()
    lw = np.clip(comps["d"].to_numpy(float) * style["comp_lw_scale"],
                 style["comp_lw_min"], style["comp_lw_max"])
    lw = np.where(is_ax_c, style["lw_axon"], lw)
    colors = np.where(np.arange(len(comps)) % 2 == 0, COL_COMP[0], COL_COMP[1])
    colors = np.where(is_ax_c, COL_AXON, colors)
    _lc(axes[1], csegs, proj, list(colors), lw, zorder=2, capstyle="butt")
    mids = compartment_midpoints(comps)
    conn, s_used = synapse_connectors(syn, mids)
    _draw_synapses(axes[1], s_used, proj, style, connectors=conn)
    axes[1].set_title("d_lambda compartments: %d segs (cm %.3g, Ra %.4g, "
                      "lambda_f %.0f Hz, d_lambda %.2g)"
                      % (len(comps), float(segmentation["cm"]), float(segmentation["Ra"]),
                         float(segmentation.get("lambda_f", 100.0)),
                         float(segmentation.get("d_lambda", 0.1))), fontsize=9)
    for ax in axes:
        _soma(ax, proj, style)
        _label_axes(ax, proj, ylabel=(ax is axes[0]) or is3d(proj))
    _equalise(axes, _xyz(raw_al), proj, style)
    h, lb = axes[1].get_legend_handles_labels()
    if h:
        axes[1].legend(h, lb, loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("cell %d -- raw skeleton vs exported compartments (%s); a connector "
                 "joins each synapse (its anchor if it sat on a pruned spine) to its lfpy_idx"
                 % (cid, proj), fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return fig


def _hist(ax, v, bins, xlabel, color=COL_SHAFT, log=False):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    else:
        if log:
            v = v[v > 0]
            if len(v) > 1 and v.max() / v.min() > 10.0:   # a log axis only when it buys something
                bins = np.logspace(np.log10(v.min()), np.log10(v.max()), bins)
                ax.set_xscale("log")
        ax.hist(v, bins=bins, color=color, edgecolor="white", linewidth=0.4)
        ax.axvline(np.median(v), color=COL_EXC, lw=1.2)
        ax.text(0.98, 0.95, "median %.3g\nn = %d" % (np.median(v), len(v)), ha="right",
                va="top", transform=ax.transAxes, fontsize=8)
    ax.set_xlabel(xlabel)


def plot_spine_stats(spine_stats, density, cid, fig=None):
    """Six panels: neck r_min, head r_max, neck length, R_neck, density
    along the path distance, synapses per spine by partition source."""
    plt, _ = _mpl()
    fig = fig or plt.figure(figsize=(12, 7))
    axes = fig.subplots(2, 3).ravel()
    _hist(axes[0], spine_stats["neck_r_min_nm"], 30, "neck radius, min over neck nodes (nm)")
    _hist(axes[1], spine_stats["head_r_max_nm"], 30, "head radius, max over head nodes (nm)")
    _hist(axes[2], spine_stats["neck_len_nm"], 30, "neck length (nm)")
    _hist(axes[3], spine_stats["R_neck_skel_Ra100_MOhm"], 30,
          "R_neck from skeleton frustums, Ra = 100 ohm cm (MOhm)", log=True)
    ax = axes[4]
    x = (density["d_lo_um"] + density["d_hi_um"]) / 2.0
    ax.bar(x, density["spines_per_um"], width=(density["d_hi_um"] - density["d_lo_um"]) * 0.9,
           color=COL_SHAFT, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("path distance from soma (um)")
    ax.set_ylabel("spines per um of shaft")
    ax.axvline(60.0, color=COL_EXC, lw=1.0, ls="--")
    ax.text(60.0, ax.get_ylim()[1] * 0.95, " 60 um (F gate)", color=COL_EXC, fontsize=8, va="top")
    ax = axes[5]
    src = spine_stats["partition_source"].astype(str)
    order = ["labeller", "rescued", "kept_at_tie", "undecidable", "no_vote"]
    cats = [c for c in order if (src == c).any()] + sorted(set(src) - set(order))
    col_of = dict(zip(order, (COL_SHAFT, COL_SPINE, "#1baf7a", COL_AXON, "#e34948")))
    nsyn = spine_stats["n_syn"].fillna(0).astype(int)
    width = 0.8 / max(len(cats), 1)
    for k, c in enumerate(cats):
        counts = nsyn[src == c].value_counts().sort_index()
        ax.bar(counts.index + (k - (len(cats) - 1) / 2.0) * width, counts.values, width=width,
               color=col_of.get(c, "#eda100"), label="%s (%d)" % (c, int((src == c).sum())),
               edgecolor="white", linewidth=0.4)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("incoming synapses on the spine")
    ax.set_ylabel("spines")
    ax.legend(frameon=False, fontsize=8, title="partition source")
    for a_ in axes[:4]:
        a_.set_ylabel("spines")
    fig.suptitle("cell %d: %d pruned spines, %d with synapses"
                 % (cid, len(spine_stats), int((nsyn > 0).sum())), fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


def plot_arbour(profile, phi, cid, record, phi_label, is_mesh, F_value=None, fig=None):
    """Four panels along the path distance: shaft cable per bin, mean shaft
    diameter, spine and shaft area per bin, their ratio (the local
    A_spine / A_shaft that F integrates beyond 60 um).

    `phi_label` names WHICH spine-area model the two right-hand panels show
    (mesh deliverable or skeleton fallback) and is written into their axis
    labels and the title; `F_value` is the F computed from THIS phi. The two
    left-hand panels are identical either way -- shaft cable and shaft
    diameter never come from the mesh."""
    plt, _ = _mpl()
    fig = fig or plt.figure(figsize=(12, 3.6))
    axes = fig.subplots(1, 4)
    x = (profile["d_lo_um"] + profile["d_hi_um"]) / 2.0
    w = (profile["d_hi_um"] - profile["d_lo_um"]) * 0.9
    axes[0].bar(x, profile["cable_um"], width=w, color=COL_SHAFT, edgecolor="white", linewidth=0.4)
    axes[0].set_ylabel("shaft cable per bin (um)")
    axes[1].plot(x, profile["shaft_diam_um"], color=COL_SHAFT, lw=1.5)
    axes[1].set_ylabel("length-weighted mean shaft diameter (um)")
    axes[2].bar(x, profile["A_shaft_um2"], width=w, color=COL_SHAFT, edgecolor="white",
                linewidth=0.4, label="shaft")
    axes[2].bar(x, profile["A_spine_um2"], width=w, bottom=profile["A_shaft_um2"], color=COL_SPINE,
                edgecolor="white", linewidth=0.4,
                label="spine, %s" % ("mesh" if is_mesh else "skeleton fallback"))
    axes[2].set_ylabel("membrane area per bin (um2)")
    axes[2].legend(frameon=False, fontsize=8)
    axes[3].plot(x, profile["A_spine_over_A_shaft"], color=COL_SPINE, lw=1.5)
    axes[3].set_ylabel("A_spine / A_shaft per bin\n(spine area: %s)"
                       % ("mesh" if is_mesh else "SKELETON FALLBACK"))
    res = record.get("result") or {}
    for ax in axes:
        ax.set_xlabel("path distance from soma (um)")
        ax.axvline(60.0, color=COL_EXC, lw=1.0, ls="--")
    fig.suptitle("cell %d arbour: %d branches, %d sections, %.0f um cable | spine area from %s: "
                 "F(d > 60 um) %s, A_shaft %.0f um2, A_spine %.0f um2"
                 % (cid, int(res.get("n_branches", 0)), int(res.get("n_sections", 0)),
                    float(phi["seg_len_um"].sum()), phi_label,
                    ("%.3f" % F_value) if F_value is not None else "n/a",
                    float(phi["shaft_area_um2"].sum()),
                    float(phi["spine_area_um2"].sum())), fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


def plot_population(summary, label, compare=None, compare_label=None, fig=None):
    """From p1_summary.csv: F_lit, spines vs cable, soma diameter, arbour
    angle, qc verdicts, compartments. With `compare` (a second tree's
    summary of the SAME cells), the last two panels become paired: totnsegs
    tree A vs tree B, and the qc_status cross-tab."""
    plt, _ = _mpl()
    ok = summary[summary["status"] == "ok"]
    fig = fig or plt.figure(figsize=(12, 7))
    axes = fig.subplots(2, 3).ravel()
    _hist(axes[0], ok["F_lit"], 30,
          "F_lit from P1: SKELETON FALLBACK\n(not the deliverable; that is P3 mesh_beyond)")
    ax = axes[1]
    ax.scatter(ok["total_length_um"], ok["n_spines"], s=8, c=COL_SHAFT, edgecolors="none")
    ax.set_xlabel("total cable (um)"); ax.set_ylabel("pruned spines")
    _hist(axes[2], ok["soma_diameter_um"], 30, "soma diameter (um)")
    _hist(axes[3], ok["angle_from_z_deg"], 30, "aligned arbour angle from +z (deg)")
    if compare is None:
        ax = axes[4]
        vc = summary["qc_status"].fillna("none").value_counts()
        ax.bar(range(len(vc)), vc.values, color=COL_SHAFT, edgecolor="white")
        ax.set_xticks(range(len(vc))); ax.set_xticklabels(vc.index, rotation=20, ha="right")
        ax.set_ylabel("cells"); ax.set_xlabel("qc_status")
        _hist(axes[5], ok["totnsegs"], 30, "compartments per cell (totnsegs)")
        fig.suptitle("%s: %d cells, %d ok" % (label, len(summary), len(ok)), fontsize=10)
    else:
        p = pair_summaries(summary, compare)
        p = p[(p["status_a"] == "ok") & (p["status_b"] == "ok")]
        ax = axes[4]
        ax.scatter(p["totnsegs_a"], p["totnsegs_b"], s=8, c=COL_SHAFT, edgecolors="none")
        lim = [0, max(float(p["totnsegs_a"].max()), float(p["totnsegs_b"].max()), 1.0) * 1.05]
        ax.plot(lim, lim, color=COL_AXON, lw=1)
        ax.plot(lim, [v * np.sqrt(2.0) for v in lim], color=COL_EXC, lw=1, ls="--", label="ratio sqrt(2)")
        ax.set_xlabel("totnsegs, %s" % label); ax.set_ylabel("totnsegs, %s" % compare_label)
        ratio = (p["totnsegs_b"] / p["totnsegs_a"]).to_numpy(float)
        ax.text(0.02, 0.95, "median ratio %.3f, n = %d" % (np.median(ratio), len(p)),
                transform=ax.transAxes, va="top", fontsize=8)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax = axes[5]
        ct = pd.crosstab(p["qc_status_a"], p["qc_status_b"])
        ax.imshow(ct.values, cmap="Greys", aspect="auto")
        ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, rotation=20, ha="right")
        ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels(ct.index)
        ax.set_xlabel("qc_status, %s" % compare_label); ax.set_ylabel("qc_status, %s" % label)
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                ax.text(j, i, str(int(ct.values[i, j])), ha="center", va="center", fontsize=8,
                        color="white" if ct.values[i, j] > ct.values.max() / 2.0 else "black")
        fig.suptitle("%s (%d cells, %d ok) paired with %s (%d ok); %d cells ok in both"
                     % (label, len(summary), len(ok), compare_label,
                        int((compare["status"] == "ok").sum()), len(p)), fontsize=10)
    for a_ in (axes[0], axes[2], axes[3]):
        a_.set_ylabel("cells")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


# --------------------------------------------------------------------------- #
# 4. CLI                                                                       #
# --------------------------------------------------------------------------- #
def F_beyond(phi, stage1_dir=None, cutoff_um=60.0):
    """F = 1 + sum A_spine / sum A_shaft beyond the cutoff, from THIS phi,
    through spine_density.cell_f_beyond_cutoff -- the same function the
    exporter uses, never a reimplementation. None if that module is absent."""
    if stage1_dir and stage1_dir not in sys.path:
        sys.path.insert(0, stage1_dir)
    try:
        import spine_density as sd
    except ImportError:
        return None
    return float(sd.cell_f_beyond_cutoff(phi, cutoff_um=cutoff_um, by="d_from_um")["F"])


def cell_figures(root, out_dir, cid, fig_dir, proj="xz", bin_um=20.0, stage1_dir=None,
                 skip_lfpy=False, dpi=220, phi_mode="mesh", style=None):
    """Produce the per-cell figures; returns {name: path}. The downsampling
    figure is skipped (with a message) when LFPy cannot be imported."""
    os.makedirs(fig_dir, exist_ok=True)
    tree = os.path.basename(os.path.normpath(out_dir))
    tag = "%s_%d" % (tree, int(cid))
    rec = load_record(out_dir, cid)
    alj = load_alignment(out_dir, cid)
    raw = load_raw_skeleton(os.path.join(root, "neurons"), cid)
    raw_al = align_raw(raw, alj)
    sn = load_spine_nodes(out_dir, cid)
    kept, pruned = split_pruned(raw_al, sn)
    stats = load_spine_stats(out_dir, cid)
    syn = load_mapped_synapses(out_dir, cid)
    style = style or resolve_style()
    out = {}
    fig = plot_pruning(kept, pruned, proj, int(cid), rec, style)
    out["pruning"] = os.path.join(fig_dir, "%s_pruning_%s.png" % (tag, proj))
    fig.savefig(out["pruning"], dpi=dpi); _close(fig)
    # The spine density profile needs phi only for the SHAFT cable denominator,
    # which is identical in the two tables, so it is drawn from P1's either way.
    phi_skel = load_phi(out_dir, cid)
    dens = spine_density_profile(stats, phi_skel, bin_um)
    fig = plot_spine_stats(stats, dens, int(cid))
    out["spines"] = os.path.join(fig_dir, "%s_spines.png" % tag)
    fig.savefig(out["spines"], dpi=dpi); _close(fig)
    # The arbour figure reports F, so it is drawn ONLY from the spine-area
    # model asked for: with the default (mesh_beyond, the deliverable) and no
    # P3 table, the figure is SKIPPED with the reason rather than quietly
    # falling back to the skeleton bracket.
    try:
        phi, phi_label, is_mesh = resolve_phi(root, out_dir, cid, phi_mode)
    except SystemExit as e:
        print("arbour figure skipped: %s" % e)
    else:
        prof = arbour_profile(phi, bin_um)
        fig = plot_arbour(prof, phi, int(cid), rec, phi_label, is_mesh,
                          F_beyond(phi, stage1_dir))
        out["arbour"] = os.path.join(fig_dir, "%s_arbour_%s.png"
                                     % (tag, "mesh" if is_mesh else "skel"))
        fig.savefig(out["arbour"], dpi=dpi); _close(fig)
    seg = (rec.get("result") or {}).get("segmentation") or {}
    if skip_lfpy:
        print("downsampling figure skipped (--skip-lfpy)")
    else:
        hoc = _artefact(out_dir, cid, "aligned.hoc")
        try:
            cell = build_lfpy_cell(hoc, seg, stage1_dir)
        except ImportError as e:
            print("downsampling figure skipped: LFPy/NEURON not importable (%s); "
                  "run in spine_env" % e)
            cell = None
        if cell is not None:
            comps = compartments_from_cell(cell)
            want = int((rec.get("result") or {}).get("totnsegs") or 0)
            if want and len(comps) != want:
                print("WARNING: rebuilt %d compartments but the record says totnsegs %d -- "
                      "the (cm, Ra, lambda_f, d_lambda) do not reproduce the export"
                      % (len(comps), want))
            fig = plot_downsampling(raw_al, comps, syn, proj, int(cid), seg, style)
            out["downsampling"] = os.path.join(fig_dir, "%s_downsampling_%s.png" % (tag, proj))
            fig.savefig(out["downsampling"], dpi=dpi); _close(fig)
    return out


def population_figure(out_dir, fig_dir, compare=None, dpi=220):
    os.makedirs(fig_dir, exist_ok=True)
    label = os.path.basename(os.path.normpath(out_dir))
    s = load_summary(out_dir)
    c = load_summary(compare) if compare else None
    clabel = os.path.basename(os.path.normpath(compare)) if compare else None
    fig = plot_population(s, label, c, clabel)
    path = os.path.join(fig_dir, "%s_population%s.png" % (label, ("_vs_" + clabel) if compare else ""))
    fig.savefig(path, dpi=dpi); _close(fig)
    return path


def _close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="campaign root (has neurons/)")
    ap.add_argument("--out-dir", required=True, help="the P1 tree: <root>/p1, <root>/p1_inh_SST, ...")
    ap.add_argument("--fig-dir", required=True)
    ap.add_argument("--cell", type=int, action="append", default=[], help="cell id (repeatable)")
    ap.add_argument("--population", action="store_true", help="population figure from p1_summary.csv")
    ap.add_argument("--compare", default=None, help="second tree whose p1_summary.csv is paired by cell")
    ap.add_argument("--proj", default="xz", choices=sorted(PROJECTIONS),
                    help="xz / yz / xy project the aligned frame; 3d draws it in 3-D "
                         "(--elev, --azim set the view)")
    ap.add_argument("--bin-um", type=float, default=20.0)
    ap.add_argument("--phi", default="mesh", choices=("mesh", "auto", "skel"),
                    help="spine-area model for the arbour figure. DEFAULT mesh: the "
                         "P3 mesh_beyond deliverable, and the figure is skipped with "
                         "a reason if P2/P3 have not run. auto falls back to P1's "
                         "skeleton bracket; skel forces it. Both label it a fallback")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--lw-scale", type=float, default=1.0,
                    help="multiply every skeleton / compartment line width")
    ap.add_argument("--syn-size", type=float, default=None,
                    help="synapse marker size (default %.3g); the pruned-spine "
                         "marker keeps its ratio to it" % STYLE["syn_size"])
    ap.add_argument("--elev", type=float, default=None, help="3d elevation, deg")
    ap.add_argument("--azim", type=float, default=None, help="3d azimuth, deg")
    ap.add_argument("--stage1-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage1"))
    ap.add_argument("--skip-lfpy", action="store_true")
    a = ap.parse_args(argv)
    if not a.cell and not a.population:
        ap.error("give --cell <id> and/or --population")
    print(PLOTS_VERSION)
    style = resolve_style(lw_scale=a.lw_scale, syn_size=a.syn_size,
                          elev=a.elev, azim=a.azim)
    for cid in a.cell:
        out = cell_figures(a.root, a.out_dir, cid, a.fig_dir, a.proj, a.bin_um, a.stage1_dir,
                           a.skip_lfpy, a.dpi, a.phi, style)
        for k, v in out.items():
            print("  %-13s %s" % (k, v))
    if a.population:
        print("  population    %s" % population_figure(a.out_dir, a.fig_dir, a.compare, a.dpi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
