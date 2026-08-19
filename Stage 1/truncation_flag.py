"""truncation_flag -- flag dendrite tips that are EM-cut rather than real endings.

Towards-EEG stage S1 (data-generation side; runs in Colab, not on the HPC).

WHY THIS EXISTS
---------------
TEEG-18 section 3.4.1 identifies a confound: a dendrite that ends because the
EM reconstruction ran out of tissue, rather than because the dendrite actually
ends there, contributes shaft area to spine_density's f_implied denominator
with none of the spine area that a real, fully-imaged stretch of dendrite
would carry -- which biases f_implied down. It also means total_length_um,
psi_vs_distance's normalisation, and any radial-extent statistic are all
computed over an arbour that is smaller than the true one, by an unknown,
cell-specific amount. This module identifies which dendrite tips are the
suspect ones, so they can be excluded or weighted rather than silently trusted.

TWO INDEPENDENT SIGNALS, NOT ONE
---------------------------------
1. TERMINAL TAPER (build_taper_table / flag_by_taper). A real dendrite ending
   narrows toward the tip; an EM cut ends abruptly, near the local shaft
   calibre, because the reconstruction simply stopped. This signal needs no
   assumption about where the dataset's boundary is -- it is a property of the
   tip alone. It is therefore the PRIMARY signal.

2. BOUNDARY PROXIMITY (pool_axis_bounds / flag_by_boundary). A tip that sits
   near the edge of where reconstructed tissue has been observed at all is
   corroborating evidence for a cut, and is the only signal available for tips
   too short to assess taper on. It needs a boundary estimate, and that
   estimate is honest about its own limits -- see below.

WHY THE TWO AXES ARE TREATED DIFFERENTLY, AND WHY THAT MATTERS
----------------------------------------------------------------
According to the H01 paper (Shapson-Coe et al. 2024, Science 384:eadk4858;
full text in the project knowledge base), the imaged block is approximately
3 mm x 2 mm in the imaging plane but only ~170 um deep along the sectioning
axis (5019 sections, mean thickness 33.9 nm). That asymmetry matters here:

  * Z (sectioning axis). 170 um total is the same order of magnitude as a
    single dendritic arbour's radius. A cell need not be unusually deep or
    shallow for a real tip to sit within a few tens of microns of the top or
    bottom face. Pooling z-extent across many cells (even within one
    subpopulation) is a reasonable proxy for the physical slab boundary,
    because the slab thickness is a tissue-wide constant, not a property of
    which cells happen to be nearby.

  * X, Y (in-plane axes). At millimetre scale, a single subpopulation's cells
    occupy a small, spatially local patch of that plane. The union of ONE
    subpopulation's own cells is NOT a trustworthy proxy for the lateral
    reconstruction boundary: a cell whose dendrite reaches slightly past its
    neighbours' extent would be flagged even though there is, in all
    likelihood, untouched tissue for hundreds of microns beyond it. This
    module still reports x/y proximity (pooled across as many cells as are
    supplied, ideally more than one subpopulation), but labels it explicitly
    as the WEAKER signal, and the combination rule below does not let x/y
    proximity alone drive the final flag.

  This is inferred from the paper's tissue geometry, not from any per-cell
  coordinate origin -- the actual position of a given reconstruction's
  bounding box within the full H01 release is not known to this module. If
  that origin becomes available (e.g. from the H01 release metadata), the
  pooled-bounds proxy in this module can be replaced with an authoritative
  one; the taper signal is unaffected either way.

COMBINATION RULE (combine_flags)
---------------------------------
  is_truncated = taper_says_cut victim's determinate)
                 OR (taper indeterminate AND z_boundary_near)

x/y proximity is reported as a diagnostic column, never as a standalone
trigger. A tip flagged on taper alone, on z-boundary alone (when taper is
indeterminate), or on both, is truncated; a tip near an x/y bound with a
clearly tapering ending and no z-proximity is NOT flagged, on the reasoning
above.

WHAT THIS MODULE DOES NOT DO
------------------------------
It does not modify the skeleton, prune anything, or change any existing
export. It reads a labelled frame (the same one spine_density.build_phi and
spine_geometry.build_spine_geometry consume) and returns a table of dendrite
TIPS with two independent pieces of evidence. Consuming that table -- to
exclude tips from a length statistic, downweight a branch, or annotate a
manifest column -- is a decision for the caller.

FREE PARAMETERS, STATED AS SUCH
---------------------------------
`taper_ratio_threshold` (default 0.7) and `reference_path_um` (default 5.0)
are not calibrated against a labelled truncated/real-ending dataset; they are
defaults chosen to be a legible starting point. Reporting `taper_ratio` and
`path_length_available_um` as continuous columns (not just the boolean flag)
means the threshold can be revisited without re-running the geometry pass.

DEPENDENCIES: numpy, pandas, spine_density (shared shaft/spine convention,
same reasoning as spine_geometry -- see that module's docstring for why the
underscore-private reuse is deliberate here too).

Pure ASCII source (HPC-safe).
"""

import math

import numpy as np
import pandas as pd

import spine_density as sd

MODULE_VERSION = "truncation_flag-1.0.0"

DEFAULT_REFERENCE_PATH_UM = 5.0
DEFAULT_TAPER_RATIO_THRESHOLD = 0.7
DEFAULT_MIN_PATH_FOR_TAPER_UM = 2.0
DEFAULT_MARGIN_UM = 10.0          # matches the S1 default agreed for S1.8/0.6
H01_SLAB_THICKNESS_UM = 170.0     # Shapson-Coe et al. 2024; sanity check only

_REQUIRED_COLUMNS = ("id", "p", "x", "y", "z", "annotated_type")


# --------------------------------------------------------------------------- #
# Shaft topology helpers (shared convention with spine_density/spine_geometry) #
# --------------------------------------------------------------------------- #
def _shaft_children_map(node, children):
    return {n: [c for c in children.get(n, ()) if node[c]["is_shaft"]]
            for n in node}


def _terminal_shaft_nodes(node, children, root):
    """Shaft nodes with zero shaft children -- true dendrite tips.

    Excludes the root itself even if it happens to satisfy the condition
    (a single-node cell with no dendrite is not a truncation candidate).
    """
    shaft_children = _shaft_children_map(node, children)
    return sorted(n for n, a in node.items()
                 if a["is_shaft"] and n != root and len(shaft_children[n]) == 0)


def _walk_back_profile(node, tip_id, max_path_um):
    """Cumulative path distance and radius from tip_id back along parents,
    following ONLY shaft nodes, up to max_path_um (or until the chain runs
    out of shaft ancestors).

    Returns (cum_dist, radii): two arrays, both starting at 0.0 / r(tip), in
    order of increasing distance from the tip. Stops the instant a
    non-shaft or absent parent is reached; the caller reads the last cum_dist
    entry to see how far back it actually got.
    """
    cum = [0.0]
    rad = [node[tip_id]["r"]]
    cur = tip_id
    dist = 0.0
    while dist < max_path_um:
        p = node[cur]["p"]
        if p not in node or not node[p]["is_shaft"]:
            break
        seg = sd._segment_length_um(node, cur, p)
        dist += seg
        cum.append(dist)
        rad.append(node[p]["r"])
        cur = p
        if p == -1:
            break
    return np.array(cum), np.array(rad)


def _radius_at_distance(cum, rad, target_um):
    """Linear interpolation of radius at path distance target_um back from the
    tip. If the profile does not reach target_um, returns the radius at the
    farthest point actually reached (i.e. no extrapolation).
    """
    if cum[-1] <= target_um:
        return float(rad[-1])
    return float(np.interp(target_um, cum, rad))


# --------------------------------------------------------------------------- #
# Signal 1: terminal taper                                                     #
# --------------------------------------------------------------------------- #
def build_taper_table(df,
                      nid=None,
                      shaft_regex=None,
                      spine_labels=None,
                      default_radius_nm=None,
                      input_units="nm",
                      reference_path_um=DEFAULT_REFERENCE_PATH_UM):
    """One row per dendrite tip, with the radius taper evidence.

    Parameters
    ----------
    df : pandas.DataFrame
        The labelled frame (id, p, x, y, z, annotated_type, and preferably r),
        same convention as spine_density.build_phi. Spine-classified nodes are
        excluded from the shaft walk automatically (is_shaft only matches
        SHAFT_REGEX, which spine labels do not).
    reference_path_um : float
        How far back along the shaft to look for the taper reference radius.
        See DEFAULT_REFERENCE_PATH_UM discussion in the module docstring.

    Returns
    -------
    pandas.DataFrame, one row per terminal shaft node:
        nid, tip_node_id, x_um, y_um, z_um, d_from_soma_um,
        r_tip_um, r_ref_um, taper_ratio, path_length_available_um,
        taper_determinate (bool: path_length_available_um >=
        DEFAULT_MIN_PATH_FOR_TAPER_UM)
    """
    shaft_regex = sd.SHAFT_REGEX if shaft_regex is None else shaft_regex
    spine_labels = sd.SPINE_LABELS if spine_labels is None else spine_labels
    default_radius_nm = (sd.DEFAULT_RADIUS_NM if default_radius_nm is None
                         else default_radius_nm)

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("input DataFrame is missing required columns: %r"
                         % missing)

    node, children, root = sd._prepare_nodes(
        df, shaft_regex, spine_labels, default_radius_nm, input_units)
    path_dist = sd._path_distances_um(node, children, root)
    tips = _terminal_shaft_nodes(node, children, root)

    rows = []
    for t in tips:
        cum, rad = _walk_back_profile(node, t, reference_path_um)
        r_tip = float(rad[0])
        r_ref = _radius_at_distance(cum, rad, reference_path_um)
        # cum[-1] can OVERSHOOT reference_path_um by design (the walk grabs
        # one extra node past the target so _radius_at_distance has something
        # to interpolate against). Cap the reported availability at the
        # target: "at least reference_path_um of cable existed" is the
        # correct statement when cum[-1] >= reference_path_um, not "cum[-1]
        # um existed". Only report the true (shorter) figure when the chain
        # genuinely ran out before reaching the target.
        path_avail = min(float(cum[-1]), reference_path_um)
        rows.append({
            "nid": nid,
            "tip_node_id": t,
            "x_um": node[t]["x"],
            "y_um": node[t]["y"],
            "z_um": node[t]["z"],
            "d_from_soma_um": path_dist.get(t, float("nan")),
            "r_tip_um": r_tip,
            "r_ref_um": r_ref,
            "taper_ratio": (r_tip / r_ref) if r_ref > 0 else float("nan"),
            "path_length_available_um": path_avail,
            "taper_determinate": path_avail >= DEFAULT_MIN_PATH_FOR_TAPER_UM,
        })

    columns = ["nid", "tip_node_id", "x_um", "y_um", "z_um",
              "d_from_soma_um", "r_tip_um", "r_ref_um", "taper_ratio",
              "path_length_available_um", "taper_determinate"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def flag_by_taper(taper_df, ratio_threshold=DEFAULT_TAPER_RATIO_THRESHOLD):
    """Add `taper_flag_truncated` (bool or None) to a taper table.

    True  : taper_determinate and taper_ratio >= ratio_threshold (no
            meaningful narrowing -> looks cut)
    False : taper_determinate and taper_ratio <  ratio_threshold (tapers ->
            looks like a real ending)
    None  : not taper_determinate -- the stub was too short to judge; the
            caller must fall back to boundary evidence or leave it unresolved.
    """
    out = taper_df.copy()
    flag = np.where(
        out["taper_determinate"],
        out["taper_ratio"].values >= ratio_threshold,
        None)
    out["taper_flag_truncated"] = pd.array(flag, dtype="boolean")
    return out


# --------------------------------------------------------------------------- #
# Signal 2: boundary proximity                                                 #
# --------------------------------------------------------------------------- #
def pool_axis_bounds(frames,
                     shaft_regex=None,
                     spine_labels=None,
                     default_radius_nm=None,
                     input_units="nm",
                     include_soma=True,
                     include_axon=False):
    """Per-axis [min, max] pooled over dendrite (+ optionally soma) node
    coordinates across MANY cells.

    Parameters
    ----------
    frames : iterable of (nid, DataFrame)
        Pass as many cells as you reasonably can -- ideally spanning more than
        one subpopulation for x/y, since a single subpopulation's spatial
        footprint is local (see module docstring). Streamed one at a time, so
        this does not require holding every frame in memory simultaneously.
    include_axon : bool
        Off by default. Axons can run much farther than dendrites and would
        pull the pooled bounds out to a scale irrelevant to a DENDRITE
        truncation question.

    Returns
    -------
    dict with keys x, y, z (each a (min, max) tuple, in um), n_cells, n_nodes,
    z_range_um, and z_range_fraction_of_slab (z_range_um / H01_SLAB_THICKNESS_UM
    -- close to 1.0 corroborates using pooled z as a boundary proxy; well
    below 1.0 is a sign the pooled sample has not yet sampled the full depth
    and the z proximity flag below should be treated cautiously).
    """
    shaft_regex = sd.SHAFT_REGEX if shaft_regex is None else shaft_regex
    spine_labels = sd.SPINE_LABELS if spine_labels is None else spine_labels
    default_radius_nm = (sd.DEFAULT_RADIUS_NM if default_radius_nm is None
                         else default_radius_nm)

    mins = {"x": math.inf, "y": math.inf, "z": math.inf}
    maxs = {"x": -math.inf, "y": -math.inf, "z": -math.inf}
    n_cells = 0
    n_nodes = 0

    for nid, df in frames:
        node, children, root = sd._prepare_nodes(
            df, shaft_regex, spine_labels, default_radius_nm, input_units)
        n_cells += 1
        axon_ids = set()
        if include_axon:
            axon_mask = df["annotated_type"].astype(str).str.contains(
                "axon", case=False, na=False)
            axon_ids = set(df.loc[axon_mask, "id"].tolist())
        for i, a in node.items():
            is_dend = a["is_shaft"]
            is_soma = (i == root) and include_soma
            is_axon_node = include_axon and (i in axon_ids)
            if not (is_dend or is_soma or is_axon_node):
                continue
            n_nodes += 1
            for ax in ("x", "y", "z"):
                v = a[ax]
                if v < mins[ax]:
                    mins[ax] = v
                if v > maxs[ax]:
                    maxs[ax] = v

    if n_cells == 0 or n_nodes == 0:
        return {"x": (float("nan"), float("nan")),
               "y": (float("nan"), float("nan")),
               "z": (float("nan"), float("nan")),
               "n_cells": n_cells, "n_nodes": n_nodes,
               "z_range_um": float("nan"),
               "z_range_fraction_of_slab": float("nan")}

    z_range = maxs["z"] - mins["z"]
    return {
        "x": (mins["x"], maxs["x"]),
        "y": (mins["y"], maxs["y"]),
        "z": (mins["z"], maxs["z"]),
        "n_cells": n_cells,
        "n_nodes": n_nodes,
        "z_range_um": z_range,
        "z_range_fraction_of_slab": z_range / H01_SLAB_THICKNESS_UM,
    }


def flag_by_boundary(tip_df,
                     bounds,
                     margin_um=DEFAULT_MARGIN_UM,
                     xy_columns=("x_um", "y_um"),
                     z_column="z_um"):
    """Add per-axis proximity columns and `z_boundary_flag_truncated`.

    `margin_um` may be a single float (applied to all three axes) or a dict
    with keys 'x', 'y', 'z' for per-axis margins. The default is uniform,
    honouring the previously agreed 10 um default; widen z specifically via
    the dict form if the slab-thickness reasoning in the module docstring
    warrants it for a given run.

    Adds: dist_to_x_bound_um, dist_to_y_bound_um, dist_to_z_bound_um,
    x_boundary_near, y_boundary_near, z_boundary_flag_truncated.
    x_boundary_near / y_boundary_near are DIAGNOSTIC ONLY -- see module
    docstring for why they must not drive the final flag alone.
    """
    if isinstance(margin_um, dict):
        m = {"x": margin_um.get("x", DEFAULT_MARGIN_UM),
            "y": margin_um.get("y", DEFAULT_MARGIN_UM),
            "z": margin_um.get("z", DEFAULT_MARGIN_UM)}
    else:
        m = {"x": margin_um, "y": margin_um, "z": margin_um}

    out = tip_df.copy()
    xcol, ycol = xy_columns
    for ax, col in (("x", xcol), ("y", ycol), ("z", z_column)):
        lo, hi = bounds[ax]
        if math.isnan(lo):
            out["dist_to_%s_bound_um" % ax] = float("nan")
            continue
        d_lo = out[col].values - lo
        d_hi = hi - out[col].values
        d = np.minimum(d_lo, d_hi)
        out["dist_to_%s_bound_um" % ax] = d

    out["x_boundary_near"] = out["dist_to_x_bound_um"] <= m["x"]
    out["y_boundary_near"] = out["dist_to_y_bound_um"] <= m["y"]
    out["z_boundary_flag_truncated"] = out["dist_to_z_bound_um"] <= m["z"]
    return out


# --------------------------------------------------------------------------- #
# Combination                                                                  #
# --------------------------------------------------------------------------- #
def combine_flags(df):
    """Apply the combination rule documented at the top of this module.

    Requires `taper_flag_truncated` (from flag_by_taper) and
    `z_boundary_flag_truncated` (from flag_by_boundary) to already be present.
    Adds `is_truncated` (bool) and `truncation_basis` (one of 'taper',
    'z_boundary', 'both', 'none', 'unresolved' -- the last meaning taper was
    indeterminate AND no boundary bounds were available for this axis).
    """
    for c in ("taper_flag_truncated", "z_boundary_flag_truncated"):
        if c not in df.columns:
            raise ValueError("combine_flags requires column %r; run "
                             "flag_by_taper / flag_by_boundary first" % c)

    out = df.copy()
    taper = out["taper_flag_truncated"]
    zbound = out["z_boundary_flag_truncated"].astype(bool)

    taper_true = taper.fillna(False).astype(bool)
    taper_indeterminate = taper.isna()

    is_trunc = taper_true | (taper_indeterminate & zbound)

    basis = np.full(len(out), "none", dtype=object)
    basis[(taper_true & zbound).values] = "both"
    basis[(taper_true & ~zbound).values] = "taper"
    basis[(taper_indeterminate & zbound).values] = "z_boundary"
    basis[(taper_indeterminate & ~zbound).values] = "unresolved"

    out["is_truncated"] = is_trunc.astype(bool)
    out["truncation_basis"] = basis
    return out


# --------------------------------------------------------------------------- #
# Convenience pipeline + aggregation                                           #
# --------------------------------------------------------------------------- #
def build_truncation_table(df, bounds, nid=None,
                           reference_path_um=DEFAULT_REFERENCE_PATH_UM,
                           ratio_threshold=DEFAULT_TAPER_RATIO_THRESHOLD,
                           margin_um=DEFAULT_MARGIN_UM,
                           input_units="nm"):
    """build_taper_table -> flag_by_taper -> flag_by_boundary -> combine_flags,
    in one call. `bounds` is the dict returned by pool_axis_bounds (or hand-
    built with the same keys).
    """
    taper = build_taper_table(df, nid=nid, input_units=input_units,
                              reference_path_um=reference_path_um)
    taper = flag_by_taper(taper, ratio_threshold=ratio_threshold)
    taper = flag_by_boundary(taper, bounds, margin_um=margin_um)
    return combine_flags(taper)


def cell_truncation_summary(trunc_df):
    """Cell-level rollup: counts and the length lost to flagged tips.

    `d_from_soma_um` on a truncated tip is a lower bound on how much cable
    was cut off, not an estimate of the missing length -- it says nothing
    about how much farther the true dendrite would have run. Reported as
    `d_from_soma_flagged_median_um` for context only.
    """
    n = len(trunc_df)
    if n == 0:
        return {"n_tips": 0}
    flagged = trunc_df[trunc_df["is_truncated"]]
    basis_counts = trunc_df["truncation_basis"].value_counts().to_dict()
    return {
        "n_tips": int(n),
        "n_truncated": int(len(flagged)),
        "frac_truncated": float(len(flagged)) / n,
        "n_taper_indeterminate": int(trunc_df["taper_flag_truncated"]
                                    .isna().sum()),
        "basis_counts": basis_counts,
        "d_from_soma_flagged_median_um": (
            float(flagged["d_from_soma_um"].median()) if len(flagged)
            else float("nan")),
        "d_from_soma_all_median_um": float(trunc_df["d_from_soma_um"].median()),
    }
