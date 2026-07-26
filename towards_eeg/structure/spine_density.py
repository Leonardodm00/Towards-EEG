"""spine_density -- compute the spine-area density field phi(nu, b, d).

Towards-EEG stage S1.3 (data-generation side; runs in Colab, not on the HPC).

WHAT THIS MODULE DOES
---------------------
Given ONE reconstructed, spine-labelled neuron (an SWC-like DataFrame whose
'annotated_type' column already carries 'head'/'neck'/'spine' labels produced
upstream by label_dendritic_spines_robust), it prunes the spine geometry
conceptually and returns the spine membrane area retained as a distance-resolved
areal density along the surviving dendritic shaft:

    phi(nu, b, d) = dA_spine / dx        [ um^2 per um of shaft length ]

sampled per unbranched shaft run (branch b), stored against arclength x from the
branch's proximal end together with d0(nu, b) = path distance from soma to that
proximal end, so the path distance d = d0(nu, b) + x is always recoverable.

At model-build time (NOT here) the per-segment correction factor is derived, not
stored:

    F(nu, s) = 1 + integral_{x1(s)}^{x2(s)} phi dx / A_shaft(nu, s)

This module never stores F. It stores phi. See integrate_phi_over_segment and
cell_f_implied_from_phi for the QC helpers.

DESIGN DECISIONS (confirmed with the user; deviations from the notebook flagged)
-------------------------------------------------------------------------------
1. NO hard proximal cutoff. _compute_F_for_neuron in the notebook excludes every
   segment within soma_cutoff_nm = 60 um. Here phi is built across ALL distances
   with no exclusion, so the empirical ~60 um proximal cutoff comes out EMERGENT
   (phi ~ 0 proximally because few spines are there) rather than imposed. Verify
   with the emergent-cutoff check in the smoke test.
2. Spine area is attributed to the shaft segment at the spine's BASE (where it
   re-attaches after pruning), NOT to the spine head's own path distance. The
   unambiguous, always-defined target is the shaft segment whose DISTAL node is
   the base node (every non-root node has exactly one incoming segment).
3. Units. Input coordinates and radii are assumed nm (the H01 convention). All
   geometry is converted to um at the boundary; phi is returned in um^2/um.

REPRESENTATION CHOICE (flagged deviation from contract C8.2 wording)
--------------------------------------------------------------------
C8.2 says phi is stored "piecewise-linear". Here phi is stored PIECEWISE-CONSTANT
per shaft segment: each spine's (lumped) membrane area is divided by the length of
one shaft segment. This is chosen so that the segment-wise integral is EXACT and
spine area is conserved by construction (sum of phi_seg * len_seg == total spine
area). A piecewise-linear vertex representation, if a downstream consumer needs
one, is a resampling of this. The breakpoints are the shaft pt3dadd vertices, so
phi is still "sampled at vertices" in the sense C8.2 intends.

DEPENDENCIES: numpy, pandas only. No scipy, no networkx (identification lives
upstream). Pure ASCII source (HPC-safe) even though this file targets Colab.
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Defaults mirror the notebook (morpholgy_pathways__6_.py)                     #
#   shaft regex: L2009 / L1701 / L1995                                         #
#   spine labels: L1996  ('spine','head','neck')                              #
#   default radius fallback: L2032 / L1678  (50 nm)                           #
# --------------------------------------------------------------------------- #
SHAFT_REGEX = r"dendrite|apical|^1$"
SPINE_LABELS = ("spine", "head", "neck")
DEFAULT_RADIUS_NM = 50.0
NM_PER_UM = 1000.0

MODULE_VERSION = "spine_density-1.2.0"

_REQUIRED_COLUMNS = ("id", "p", "x", "y", "z", "annotated_type")


# --------------------------------------------------------------------------- #
# Geometry primitive -- identical to notebook _frustum_lateral_area (L1982)    #
# --------------------------------------------------------------------------- #
def _frustum_lateral_area(r1, r2, length):
    """Lateral surface area of a truncated cone with end radii r1, r2 and axial
    end-face separation `length`. Same formula and units as the notebook."""
    slant = math.sqrt((r1 - r2) ** 2 + length ** 2)
    return math.pi * (r1 + r2) * slant


# --------------------------------------------------------------------------- #
# Node index, classification, topology                                        #
# --------------------------------------------------------------------------- #
def _prepare_nodes(df, shaft_regex, spine_labels, default_radius_nm, input_units):
    """Return dict id -> node record in um, plus children map and root id.

    Node record keys: p, x, y, z, r (um), is_shaft (bool), is_spine (bool).
    """
    import re

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("input DataFrame is missing required columns: %r" % missing)

    work = df.copy()
    if "r" not in work.columns:
        work["r"] = default_radius_nm

    scale = NM_PER_UM if input_units == "nm" else 1.0

    shaft_re = re.compile(shaft_regex, re.IGNORECASE)
    spine_set = {s.lower() for s in spine_labels}

    node = {}
    for row in work.itertuples(index=False):
        label = str(getattr(row, "annotated_type"))
        node[getattr(row, "id")] = {
            "p": getattr(row, "p"),
            "x": float(getattr(row, "x")) / scale,
            "y": float(getattr(row, "y")) / scale,
            "z": float(getattr(row, "z")) / scale,
            "r": float(getattr(row, "r")) / scale,
            "is_shaft": bool(shaft_re.search(label)),
            "is_spine": label.lower() in spine_set,
        }

    roots = [nid for nid, a in node.items() if a["p"] == -1 or a["p"] not in node]
    if len(roots) == 0:
        raise ValueError("no root node (p == -1) found")
    root = roots[0]

    children = defaultdict(list)
    for nid, a in node.items():
        if nid == root:
            continue
        children[a["p"]].append(nid)

    return node, children, root


def _segment_length_um(node, a_id, b_id):
    a = node[a_id]
    b = node[b_id]
    return math.sqrt(
        (a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2
    )


def _path_distances_um(node, children, root):
    """Path distance from the soma to every node, in um (BFS along the tree)."""
    dist = {root: 0.0}
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in children.get(parent, ()):
            dist[child] = dist[parent] + _segment_length_um(node, parent, child)
            stack.append(child)
    return dist


# --------------------------------------------------------------------------- #
# Spine area attribution (decision 2: attribute to the base segment)          #
# --------------------------------------------------------------------------- #
def _attribute_spine_area(node, children):
    """Return dict (parent_id, child_id) -> summed spine membrane area (um^2).

    A spine is a maximal terminal subtree of spine-labelled nodes. Its total
    membrane area (sum of frustum areas over its own incoming segments) is
    attributed to the shaft segment whose DISTAL node is the spine's base node
    (the shaft node the spine hangs off). For a spine sitting directly on the
    root (no incoming shaft segment), it is attributed to the root's first shaft
    child segment instead, so no area is silently dropped.
    """
    shaft_children = {
        n: [c for c in children.get(n, ()) if node[c]["is_shaft"]] for n in node
    }

    def spine_subtree(root_id):
        """All spine-labelled nodes reachable through spine children."""
        out = [root_id]
        stack = [root_id]
        while stack:
            cur = stack.pop()
            for c in children.get(cur, ()):
                if node[c]["is_spine"]:
                    out.append(c)
                    stack.append(c)
        return out

    seg_spine_area = defaultdict(float)
    dropped = 0
    for nid, a in node.items():
        if not a["is_spine"]:
            continue
        parent = a["p"]
        if parent in node and node[parent]["is_spine"]:
            continue  # not a spine root; handled from its root
        # nid is a spine root; base = its parent (a shaft node or the soma)
        base = parent
        subtree = spine_subtree(nid)
        area = 0.0
        for s in subtree:
            sp = node[s]["p"]
            area += _frustum_lateral_area(node[sp]["r"], node[s]["r"],
                                          _segment_length_um(node, sp, s))
        if base in node and node[base]["p"] in node:
            # base has an incoming shaft segment (parent(base) -> base)
            seg_spine_area[(node[base]["p"], base)] += area
        elif base in node and shaft_children.get(base):
            # base is the root: attribute to its first outgoing shaft segment
            seg_spine_area[(base, shaft_children[base][0])] += area
        else:
            dropped += area
    return seg_spine_area, dropped


# --------------------------------------------------------------------------- #
# Branch decomposition (unbranched shaft runs)                                #
# --------------------------------------------------------------------------- #
def _decompose_branches(node, children, root):
    """Partition the shaft into maximal unbranched runs.

    Returns a list of branches, each an ordered list of node ids [j, n1, ..., nk]
    where j is a junction (root or shaft branch point) and n1..nk is the chain of
    single-shaft-child nodes. Every shaft segment appears in exactly one branch.
    """
    shaft_children = {
        n: [c for c in children.get(n, ()) if node[c]["is_shaft"]] for n in node
    }
    junctions = {root}
    for n in node:
        if n != root and len(shaft_children[n]) >= 2:
            junctions.add(n)

    branches = []
    for j in junctions:
        for c in shaft_children[j]:
            branch = [j, c]
            cur = c
            while True:
                sc = shaft_children[cur]
                if len(sc) == 1 and cur not in junctions:
                    branch.append(sc[0])
                    cur = sc[0]
                else:
                    break
            branches.append(branch)
    return branches


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def build_phi(df,
              nid=None,
              shaft_regex=SHAFT_REGEX,
              spine_labels=SPINE_LABELS,
              default_radius_nm=DEFAULT_RADIUS_NM,
              input_units="nm",
              self_check=True):
    """Compute phi(nu, b, d) for one spine-labelled neuron.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns id, p, x, y, z, annotated_type (optional r). 'annotated_type'
        MUST already contain 'head'/'neck'/'spine' for spine nodes (run the
        upstream labeler first). Coordinates/radii in nm by default.
    nid : hashable or None
        Morphology identifier nu, copied into the output for bookkeeping.
    input_units : {'nm', 'um'}
        Units of the input coordinates and radii. Output is always um.
    self_check : bool
        If True, assert internal invariants (segment coverage; d0 + x == path
        distance). Cheap; leave on.

    Returns
    -------
    pandas.DataFrame, one row per shaft segment, columns:
        nid, branch_id, seg_index,
        node_from, node_to,
        x0_um, x1_um, seg_len_um,       # arclength along the branch
        d0_um, d_from_um, d_to_um,      # path distance from soma
        shaft_diam_um,                  # r_from + r_to (mean diameter of segment)
        shaft_area_um2, spine_area_um2,
        phi_um,                         # spine_area_um2 / seg_len_um  [um^2/um]
        psi                             # phi / (pi * shaft_diam_um), dimensionless
    """
    node, children, root = _prepare_nodes(
        df, shaft_regex, spine_labels, default_radius_nm, input_units)
    path_dist = _path_distances_um(node, children, root)
    seg_spine_area, dropped = _attribute_spine_area(node, children)
    branches = _decompose_branches(node, children, root)

    rows = []
    n_segments = 0
    for b_idx, branch in enumerate(branches):
        d0 = path_dist[branch[0]]
        x = 0.0
        for k in range(1, len(branch)):
            a_id = branch[k - 1]
            b_id = branch[k]
            seg_len = _segment_length_um(node, a_id, b_id)
            x0 = x
            x1 = x + seg_len
            x = x1
            r_a = node[a_id]["r"]
            r_b = node[b_id]["r"]
            shaft_area = _frustum_lateral_area(r_a, r_b, seg_len)
            spine_area = seg_spine_area.get((a_id, b_id), 0.0)
            phi = spine_area / seg_len if seg_len > 0 else 0.0
            delta = r_a + r_b
            psi = phi / (math.pi * delta) if delta > 0 else 0.0
            rows.append({
                "nid": nid,
                "branch_id": b_idx,
                "seg_index": k - 1,
                "node_from": a_id,
                "node_to": b_id,
                "x0_um": x0,
                "x1_um": x1,
                "seg_len_um": seg_len,
                "d0_um": d0,
                "d_from_um": d0 + x0,
                "d_to_um": d0 + x1,
                "shaft_diam_um": delta,
                "shaft_area_um2": shaft_area,
                "spine_area_um2": spine_area,
                "phi_um": phi,
                "psi": psi,
            })
            n_segments += 1

            if self_check:
                # d0 + x must equal the independent BFS path distance
                assert abs((d0 + x1) - path_dist[b_id]) < 1e-6, (
                    "branch arclength inconsistent with path distance at node "
                    "%r" % b_id)

    phi_df = pd.DataFrame(rows)

    if self_check:
        # every shaft segment covered exactly once
        total_shaft_segments = sum(
            1 for nid_ in node
            if nid_ != root and node[nid_]["is_shaft"]
        )
        assert n_segments == total_shaft_segments, (
            "branch decomposition covered %d shaft segments but the tree has %d"
            % (n_segments, total_shaft_segments))
        # all attributed spine area landed on some segment
        attributed = sum(seg_spine_area.values())
        recovered = float(phi_df["spine_area_um2"].sum()) if len(phi_df) else 0.0
        assert abs(attributed - recovered) < 1e-6, (
            "attributed spine area %.6f not fully recovered in output %.6f"
            % (attributed, recovered))

    phi_df.attrs["dropped_spine_area_um2"] = dropped
    return phi_df


def integrate_phi_over_segment(phi_df, branch_id, x_start_um, x_end_um):
    """Integral of phi over [x_start, x_end] on one branch, in um^2.

    Exact for the piecewise-constant representation: sums phi_seg times the
    overlap of each shaft segment's [x0, x1] with [x_start, x_end]. This is the
    numerator of the build-time correction factor F(nu, s).
    """
    lo, hi = (x_start_um, x_end_um) if x_start_um <= x_end_um else (x_end_um, x_start_um)
    sub = phi_df[phi_df["branch_id"] == branch_id]
    total = 0.0
    for r in sub.itertuples(index=False):
        overlap = max(0.0, min(hi, r.x1_um) - max(lo, r.x0_um))
        if overlap > 0.0:
            total += r.phi_um * overlap
    return total


def cell_f_implied_from_phi(phi_df):
    """Cell-level f_implied = 1 + (total spine area) / (total shaft area).

    QC quantity only (never a multiplier). With no proximal cutoff this equals
    the notebook's global F recomputed over the whole dendrite, and is the number
    to compare against the age-matched literature F in S1.7.
    """
    if len(phi_df) == 0:
        return float("nan")
    a_spine = float(phi_df["spine_area_um2"].sum())
    a_shaft = float(phi_df["shaft_area_um2"].sum())
    return 1.0 + a_spine / a_shaft if a_shaft > 0 else float("nan")


def cell_f_beyond_cutoff(phi_df, cutoff_um=60.0, by="d_from_um"):
    """Cell-level F restricted to segments beyond a proximal cutoff.

    THIS, NOT cell_f_implied_from_phi, is the quantity comparable to the
    published literature F (Eyal et al. 2016/2018; Benavides-Piccione et al.,
    multiple regions): those studies exclude segments within `cutoff_um` of the
    soma from BOTH numerator and denominator (justified by very low proximal
    spine density), rather than including them at phi ~ 0 as
    cell_f_implied_from_phi does. The two are different quantities by
    construction: the no-cutoff whole-cell f_implied is always <= the
    cutoff-restricted F, because the proximal shaft contributes area to the
    denominator with ~no matching spine area, pulling the whole-cell ratio
    toward 1. Comparing f_implied directly to a published F is therefore not a
    like-for-like comparison; use this function for that comparison instead.

    Parameters
    ----------
    phi_df : pandas.DataFrame
        Output of build_phi.
    cutoff_um : float
        Distance threshold. Default 60.0 matches Eyal et al. (2016/2018) and
        Benavides-Piccione et al. (Htemp/Hcing/MCA1/HCA1); note the same
        sources use 30.0 for mouse.
    by : {'d_from_um', 'd_to_um', 'mid'}
        Which distance to test against cutoff_um when deciding whether a
        segment counts as "beyond" it. 'd_from_um' (the proximal end of the
        segment) is the conservative choice -- a segment straddling the
        boundary is excluded, matching "segments ... at a distance of at
        least cutoff_um from the soma" as a whole-segment criterion. 'mid'
        matches the convention used in psi_vs_distance.

    Returns
    -------
    dict with keys:
        F, A_shaft_um2, A_spine_um2, n_segments_included, n_segments_total,
        frac_shaft_area_included  (how much of the cell this F is actually
        describing -- a low fraction means the comparison rests on a small,
        possibly noisy, distal remainder)
    """
    if len(phi_df) == 0:
        return {"F": float("nan"), "A_shaft_um2": 0.0, "A_spine_um2": 0.0,
                "n_segments_included": 0, "n_segments_total": 0,
                "frac_shaft_area_included": float("nan")}

    if by == "mid":
        d = 0.5 * (phi_df["d_from_um"].values + phi_df["d_to_um"].values)
    elif by in ("d_from_um", "d_to_um"):
        d = phi_df[by].values
    else:
        raise ValueError("by must be 'd_from_um', 'd_to_um', or 'mid'")

    mask = d >= cutoff_um
    a_shaft_all = float(phi_df["shaft_area_um2"].sum())
    a_shaft = float(phi_df.loc[mask, "shaft_area_um2"].sum())
    a_spine = float(phi_df.loc[mask, "spine_area_um2"].sum())
    return {
        "F": (1.0 + a_spine / a_shaft) if a_shaft > 0 else float("nan"),
        "A_shaft_um2": a_shaft,
        "A_spine_um2": a_spine,
        "n_segments_included": int(mask.sum()),
        "n_segments_total": int(len(phi_df)),
        "frac_shaft_area_included": (a_shaft / a_shaft_all) if a_shaft_all > 0
        else float("nan"),
    }


def psi_vs_distance(phi_df, bin_width_um=10.0, max_d_um=None):
    """Aggregate phi into path-distance bins as the dimensionless density psi.

    This is the S1.7 input: the profile compared against published
    distance-resolved human spine densities.

    Aggregation is AREA-WEIGHTED, not a mean of per-segment psi:

        psi_bin = (sum of spine area in bin) / (sum of shaft area in bin)
        F_bin   = 1 + psi_bin

    which is the physically meaningful pooling (identical in form to the
    notebook's F_by_bin) and is not dominated by very short segments the way an
    unweighted mean of per-segment psi would be.

    APPROXIMATION (flagged): each shaft segment is assigned WHOLE to the bin
    containing its midpoint 0.5 * (d_from + d_to), rather than split across bin
    boundaries. Valid when segment length << bin_width_um; H01 skeleton segments
    are ~1 um, so a 10 um bin is comfortably in that regime. Check the reported
    'max_seg_len_um' against bin_width_um before trusting a narrow binning.

    Parameters
    ----------
    phi_df : pandas.DataFrame
        Output of build_phi.
    bin_width_um : float
        Width of the path-distance bins, in um.
    max_d_um : float or None
        Upper edge of the last bin. If None, taken from the data.

    Returns
    -------
    pandas.DataFrame with one row per bin:
        d_lo_um, d_hi_um, d_mid_um, n_segments,
        shaft_len_um, shaft_area_um2, spine_area_um2,
        phi_mean_um   (= spine area / shaft length in the bin)
        psi_mean      (= spine area / shaft area in the bin)
        F_bin         (= 1 + psi_mean)
    """
    if len(phi_df) == 0:
        return pd.DataFrame(columns=[
            "d_lo_um", "d_hi_um", "d_mid_um", "n_segments", "shaft_len_um",
            "shaft_area_um2", "spine_area_um2", "phi_mean_um", "psi_mean",
            "F_bin"])

    d_mid = 0.5 * (phi_df["d_from_um"].values + phi_df["d_to_um"].values)
    d_top = float(np.max(phi_df["d_to_um"].values)) if max_d_um is None \
        else float(max_d_um)
    n_bins = max(1, int(math.ceil(d_top / bin_width_um)))
    edges = np.arange(n_bins + 1, dtype=float) * bin_width_um

    idx = np.clip(np.digitize(d_mid, edges) - 1, 0, n_bins - 1)

    rows = []
    seg_len = phi_df["seg_len_um"].values
    a_shaft = phi_df["shaft_area_um2"].values
    a_spine = phi_df["spine_area_um2"].values
    for b in range(n_bins):
        m = (idx == b)
        n_seg = int(np.count_nonzero(m))
        L = float(seg_len[m].sum())
        As = float(a_shaft[m].sum())
        Asp = float(a_spine[m].sum())
        rows.append({
            "d_lo_um": float(edges[b]),
            "d_hi_um": float(edges[b + 1]),
            "d_mid_um": float(0.5 * (edges[b] + edges[b + 1])),
            "n_segments": n_seg,
            "shaft_len_um": L,
            "shaft_area_um2": As,
            "spine_area_um2": Asp,
            "phi_mean_um": (Asp / L) if L > 0 else float("nan"),
            "psi_mean": (Asp / As) if As > 0 else float("nan"),
            "F_bin": (1.0 + Asp / As) if As > 0 else float("nan"),
        })
    out = pd.DataFrame(rows)
    out.attrs["max_seg_len_um"] = float(np.max(seg_len))
    out.attrs["bin_width_um"] = float(bin_width_um)
    return out


def save_phi(phi_df, path):
    """Persist phi to CSV (long format, one row per shaft segment)."""
    phi_df.to_csv(path, index=False)
    return path


def compute_and_save_phi(df, nid, output_dir, **kwargs):
    """Convenience: build phi for one neuron and write neuron_{nid}_phi.csv."""
    import os
    phi_df = build_phi(df, nid=nid, **kwargs)
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "neuron_%s_phi.csv" % nid)
    save_phi(phi_df, out)
    return phi_df, out
