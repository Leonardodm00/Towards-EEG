"""Measure the magnitude of the free-snap synapse misplacement that spine
pruning (D-7) activates.

Scientific question
-------------------
`Alignment.py::map_and_save_synapses_to_lfpy_idx` snaps every synapse with
`cell.get_closest_idx(x, y, z)`, a free 3-D nearest-compartment search. While
spines are present in the .hoc a spine-head synapse snaps to its own head. Once
`prune_spines` removes the head, the same call snaps to whatever compartment is
now nearest. TEEG_16 section 3.3 asserts this silently relocates excitatory
input. This module measures, on real data, how much.

For every synapse it computes two indices on the SAME LFPy cell:

    idx_naive   = get_closest_idx(synapse coordinate)         current behaviour
    idx_anchor  = get_closest_idx(base-shaft coordinate)      the D-7 redirect

and classifies the disagreement topologically on the section tree, which is the
electrotonically meaningful comparison: a shift of a few segments within the
parent section is harmless, a jump to a foreign branch is not.

Scope and honest limits
-----------------------
1. An LFPy.Cell holds ONE neuron, so `get_closest_idx` can only reach this
   cell's own compartments. The real defect includes snapping onto a DIFFERENT
   neuron's dendrite in dense neuropil, which a single-cell model cannot
   express. Every number this module reports is therefore a LOWER BOUND on the
   true misplacement rate.
2. It measures placement only. It does not simulate, and says nothing about the
   downstream voltage consequence (that is R-8).

Design
------
No file paths, no Drive, no plotting, no NEURON import. The LFPy cell and the
raw -> aligned coordinate transform are both injected by the caller, so the
whole module is exercisable against a stub cell with no NEURON installed.

Units: every coordinate ENTERING this module is raw nanometres. Every
coordinate and distance LEAVING it is aligned micrometres -- with ONE
exception: map_synapses_to_nodes_raw (Section 0) both takes and returns RAW
nanometres, because its output feeds alignment.resolve_synapse_anchors, which
performs the raw -> aligned transform itself, on the anchor coordinate rather
than the synapse coordinate. See that function's own docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


MODULE_VERSION = "synapse_redirect_audit-1.2.0"

REL_SAME = "same"
REL_ANCESTOR = "ancestor"
REL_DESCENDANT = "descendant"
REL_FOREIGN = "foreign"


# --------------------------------------------------------------------------- #
#  0. Raw synapse CSV -> the node_id/syn_x_nm/syn_y_nm/syn_z_nm contract        #
#     alignment.resolve_synapse_anchors requires (O7: promoted from being      #
#     copy-pasted notebook code in TWO Colab drivers into ONE tested function) #
# --------------------------------------------------------------------------- #
def map_synapses_to_nodes_raw(syn_df, df_raw, voxel_res=(8.0, 8.0, 33.0),
                              direction=None, id_column="id",
                              xyz_columns=("x", "y", "z"),
                              type_column="synapse_type"):
    """Assign each raw synapse to its nearest skeleton node, by NODE ID.

    `syn_df` is the H01 synapse export as it arrives on disk: one row per
    synapse, voxel-space `location_x/y/z`, and (usually) a `direction` and a
    `synapse_type` column. `df_raw` is the matching skeleton, unclassified, in
    the same raw-nm space `export_neuron` takes as input.

    Returns a DataFrame with exactly the columns
    resolve_synapse_anchors/align_and_export require -- node_id, syn_x_nm,
    syn_y_nm, syn_z_nm -- plus `synapse_label` (the SIGMA_SYN vocabulary
    write_mapped_synapses consumes; see classify_synapse_label for why it is
    the label and NOT synapse_type) and snap_distance_nm for QC. This
    is NOT the final lfpy_idx assignment: it is the upstream step that turns a
    voxel coordinate into a node id, which resolve_synapse_anchors then uses to
    decide whether that node sits on a spine.

    IMPORTANT (the bug this replaces): the KD-tree query returns POSITIONAL
    indices into `df_raw`. On every real skeleton checked, `id` is NOT equal to
    row position (e.g. on 794820508, max id 63397 over 63147 rows). Using the
    positional index as if it were the node id silently mismaps every synapse.
    This function converts explicitly via `df_raw[id_column].to_numpy()[pos]`.

    Parameters
    ----------
    syn_df : DataFrame
        Raw synapse export. Must have location_x/y/z (voxel units) and,
        optionally, 'direction' and 'synapse_type'.
    df_raw : DataFrame
        The unclassified skeleton, with `xyz_columns` in the SAME raw-nm space
        as the .hoc that will be built from it.
    voxel_res : (float, float, float)
        Voxel size in nm, applied as location_xyz * voxel_res -> nm. H01's
        default (8, 8, 33) reflects its anisotropic (thin) z-sampling.
    direction : str or None
        If given and 'direction' is a column, keep only rows matching it
        (e.g. 'incoming' for postsynaptic sites on this cell).
    id_column, xyz_columns : str, (str, str, str)
        Column names in `df_raw` for the node id and its raw coordinate.
    type_column : str
        Column in `syn_df` holding the H01 excitatory/inhibitory code.

    Raises
    ------
    ValueError if, after filtering, no synapse has a usable coordinate.
    """
    for col in ("location_x", "location_y", "location_z"):
        if col not in syn_df.columns:
            raise ValueError("syn_df lacks required column %r" % col)
    for col in xyz_columns + (id_column,):
        if col not in df_raw.columns:
            raise ValueError("df_raw lacks required column %r" % col)

    out = syn_df
    if direction is not None and "direction" in out.columns:
        out = out[out["direction"] == direction]
    out = out.dropna(subset=["location_x", "location_y", "location_z"])
    if out.empty:
        raise ValueError(
            "no synapses left with a usable coordinate (direction=%r on %d "
            "input rows)" % (direction, len(syn_df)))

    cx, cy, cz = xyz_columns
    node_coords_nm = df_raw[[cx, cy, cz]].to_numpy(dtype=float)
    tree = cKDTree(node_coords_nm)

    syn_coords_nm = (out[["location_x", "location_y", "location_z"]]
                     .to_numpy(dtype=float) * np.asarray(voxel_res, dtype=float))
    dist_nm, pos_idx = tree.query(syn_coords_nm)

    ids = df_raw[id_column].to_numpy()
    node_ids = ids[pos_idx]                     # THE fix: id, not position

    return pd.DataFrame({
        "node_id": node_ids.astype(np.int64),
        "syn_x_nm": syn_coords_nm[:, 0],
        "syn_y_nm": syn_coords_nm[:, 1],
        "syn_z_nm": syn_coords_nm[:, 2],
        "synapse_label": classify_synapse_label(out, type_column),
        "snap_distance_nm": dist_nm,
    }).reset_index(drop=True)


def classify_synapse_label(syn_df, type_column="synapse_type"):
    """H01 synapse_type -> the SIGMA_SYN mapper vocabulary.

    Returns a plain list of 'exc_syn' / 'inh_syn' / 'unknown_syn', which is
    node_classify.SIGMA_SYN -- the SAME vocabulary map_synapses_to_segments.py
    produces and the one alignment.write_mapped_synapses consumes. It emits
    `synapse_label`, NOT `synapse_type`: write_mapped_synapses DERIVES
    synapse_type from synapse_label via to_downstream_type and overwrites
    whatever synapse_type it was handed, so a function upstream that emits
    synapse_type instead has its work silently discarded and every row ends up
    'unknown'. Exactly one column is authoritative, and it is this one.

    H01 codes the type as an INTEGER, 2 = excitatory (asymmetric),
    1 = inhibitory (symmetric) -- confirmed on the real export, dtype int64
    with values {1, 2} only. Integer codes are matched EXACTLY rather than by
    substring: '12' contains both '1' and '2', so substring matching is only
    safe while the code alphabet happens to be single-digit, which is not a
    property of the data worth relying on. Text exports ('asymmetric',
    'excitatory', ...) still fall through to substring matching, so both
    conventions are handled.

    An absent column is not an error -- some exports genuinely lack it -- but
    every row is then 'unknown_syn', and the caller is expected to notice.
    """
    if type_column not in syn_df.columns:
        return ["unknown_syn"] * len(syn_df)

    col = syn_df[type_column]
    labels = []
    for v in col.tolist():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            labels.append("unknown_syn")
            continue
        # exact integer code first
        code = None
        try:
            f = float(v)
            if f == int(f):
                code = int(f)
        except (TypeError, ValueError):
            code = None
        if code == 2:
            labels.append("exc_syn")
            continue
        if code == 1:
            labels.append("inh_syn")
            continue
        if code is not None:
            labels.append("unknown_syn")          # a numeric code we don't know
            continue
        s = str(v).lower()
        if any(k in s for k in ("exc", "asymmetric")):
            labels.append("exc_syn")
        elif any(k in s for k in ("inh", "symmetric")):
            labels.append("inh_syn")
        else:
            labels.append("unknown_syn")
    return labels


# --------------------------------------------------------------------------- #
#  1. Section topology, rebuilt from the exporter's own section_table          #
# --------------------------------------------------------------------------- #
def hoc_name(array, type_idx):
    """The name the exporter writes into the .hoc, e.g. 'dend[26]'."""
    return "%s[%d]" % (str(array), int(type_idx))


def build_section_tree(section_table):
    """Parent map over hoc section names.

    Parameters
    ----------
    section_table : DataFrame
        The exporter's `neuron_{nid}_section_table.csv`. Requires columns
        section_id, array, type_idx, parent_sec_id.

    Returns
    -------
    parent_of : dict, hoc name -> parent hoc name (or None at the root)
    """
    for col in ("section_id", "array", "type_idx", "parent_sec_id"):
        if col not in section_table.columns:
            raise ValueError("section_table lacks required column %r" % col)

    id_to_name = {
        int(r.section_id): hoc_name(r.array, r.type_idx)
        for r in section_table.itertuples(index=False)
    }
    parent_of = {}
    for r in section_table.itertuples(index=False):
        me = id_to_name[int(r.section_id)]
        pid = int(r.parent_sec_id)
        parent_of[me] = id_to_name.get(pid) if pid >= 0 else None

    # a parent map must be acyclic and reach a root from every node
    for name in parent_of:
        seen, cur = set(), name
        while cur is not None:
            if cur in seen:
                raise ValueError("cycle in section tree at %r" % cur)
            seen.add(cur)
            cur = parent_of.get(cur)
    return parent_of


def ancestor_chain(name, parent_of):
    """[name, parent, ..., root]. Raises if `name` is unknown."""
    if name not in parent_of:
        raise KeyError("section %r not in the section tree" % name)
    chain, cur = [], name
    while cur is not None:
        chain.append(cur)
        cur = parent_of.get(cur)
    return chain


def relate_sections(sec_a, sec_b, parent_of):
    """Topological relation of `sec_a` (the naive snap) to `sec_b` (the truth).

    Returns one of REL_SAME / REL_ANCESTOR / REL_DESCENDANT / REL_FOREIGN.
    ANCESTOR means sec_a lies between sec_b and the soma; DESCENDANT means
    sec_a lies distal to sec_b on the same path. Only FOREIGN means the synapse
    left the branch entirely.
    """
    if sec_a == sec_b:
        return REL_SAME
    if sec_a in ancestor_chain(sec_b, parent_of)[1:]:
        return REL_ANCESTOR
    if sec_b in ancestor_chain(sec_a, parent_of)[1:]:
        return REL_DESCENDANT
    return REL_FOREIGN


def section_path(sec_a, sec_b, parent_of):
    """Ordered list of sections on the tree path from sec_a to sec_b."""
    chain_a = ancestor_chain(sec_a, parent_of)
    chain_b = ancestor_chain(sec_b, parent_of)
    set_b = set(chain_b)
    lca = next(s for s in chain_a if s in set_b)
    up = chain_a[: chain_a.index(lca)]
    down = chain_b[: chain_b.index(lca)]
    return up + [lca] + list(reversed(down))


def path_length_um(sec_a, sec_b, parent_of, sec_length_um):
    """Cable distance between the MIDPOINTS of sec_a and sec_b, in um.

    Equals half of each endpoint section plus the whole of every section
    strictly between them, which is sum(L over the path) - L_a/2 - L_b/2.
    """
    if sec_a == sec_b:
        return 0.0
    path = section_path(sec_a, sec_b, parent_of)
    total = float(sum(sec_length_um[s] for s in path))
    return total - 0.5 * sec_length_um[sec_a] - 0.5 * sec_length_um[sec_b]


# --------------------------------------------------------------------------- #
#  2. Flattened LFPy index -> section                                          #
# --------------------------------------------------------------------------- #
def build_segment_index(cell):
    """Map every flattened LFPy compartment index to its owning section.

    LFPy builds cell.x/y/z by iterating `for sec in allseclist: for seg in sec`,
    so the flattened index is cumulative in allseclist order. This function
    reproduces that and CROSS-CHECKS it against cell.totnsegs, and against
    cell.get_idx_name() when that API is present.

    Returns
    -------
    sec_of_idx    : int array, length totnsegs
    sec_names     : list of hoc names in allseclist order
    sec_length_um : dict, hoc name -> section length L in um
    """
    sec_names, sec_of_idx, sec_length_um = [], [], {}
    for i, sec in enumerate(cell.allseclist):
        name = sec.name()
        if "." in name:                      # strip a NEURON cell prefix
            name = name.split(".")[-1]
        sec_names.append(name)
        sec_length_um[name] = float(sec.L)
        sec_of_idx.extend([i] * int(sec.nseg))

    sec_of_idx = np.asarray(sec_of_idx, dtype=int)

    totnsegs = int(getattr(cell, "totnsegs", len(sec_of_idx)))
    if len(sec_of_idx) != totnsegs:
        raise ValueError(
            "segment count mismatch: allseclist gives %d, cell.totnsegs is %d"
            % (len(sec_of_idx), totnsegs))

    # independent confirmation of the ORDERING, not just the count
    getname = getattr(cell, "get_idx_name", None)
    if callable(getname):
        try:
            rec = getname(idx=np.arange(totnsegs))
            got = [str(n).split(".")[-1] for n in np.asarray(rec["name"])]
            mine = [sec_names[k] for k in sec_of_idx]
            if got != mine:
                raise ValueError(
                    "flattened index ordering disagrees with cell.get_idx_name; "
                    "first mismatch at %d (%r vs %r)"
                    % (next(i for i, (a, b) in enumerate(zip(got, mine)) if a != b),
                       got[:1], mine[:1]))
        except (KeyError, TypeError, IndexError):
            pass                              # older LFPy: count check stands
    return sec_of_idx, sec_names, sec_length_um


# --------------------------------------------------------------------------- #
#  3. Which nodes are on a spine, and where is that spine's base               #
# --------------------------------------------------------------------------- #
def map_nodes_to_spine_bases(df_labelled, spine_classes, class_column="compartment_class",
                             id_column="id", parent_column="p"):
    """For every spine node, find its spine root and the shaft node it sits on.

    Walks up the parent chain from each spine node until a non-spine node is
    reached. That node is the BASE; the last spine node before it is the spine
    ROOT. This is deliberately reimplemented rather than imported so that
    `verify_against_spine_bases` below is a genuine independent check on
    morphology_exporter.prune_spines rather than a tautology.

    Parameters
    ----------
    df_labelled : DataFrame
        The frame as it stands immediately BEFORE pruning: classified, mislabels
        resolved, spines labelled. Requires id, p and `class_column`.
    spine_classes : container of str
        The class values that count as spine (e.g. {'spine'}).

    Returns
    -------
    DataFrame with node_id, spine_root_id, base_node_id, depth_in_spine
    """
    ids = df_labelled[id_column].to_numpy()
    par = df_labelled[parent_column].to_numpy()
    cls = df_labelled[class_column].astype(str).to_numpy()

    parent_of = dict(zip(ids.tolist(), par.tolist()))
    is_spine = {int(i): (c in spine_classes) for i, c in zip(ids.tolist(), cls)}

    rows = []
    for nid in ids.tolist():
        nid = int(nid)
        if not is_spine.get(nid, False):
            continue
        chain, cur = [], nid
        while cur != -1 and is_spine.get(int(cur), False):
            chain.append(int(cur))
            nxt = parent_of.get(int(cur), -1)
            if nxt == cur:
                raise ValueError("self-parent at node %d" % cur)
            cur = int(nxt)
            if len(chain) > len(ids):
                raise ValueError("cycle walking up from node %d" % nid)
        rows.append({
            "node_id": nid,
            "spine_root_id": chain[-1],
            "base_node_id": int(cur),          # -1 if the spine reaches the root
            "depth_in_spine": len(chain) - 1,
        })
    return pd.DataFrame(rows, columns=["node_id", "spine_root_id",
                                       "base_node_id", "depth_in_spine"])


def verify_against_spine_bases(node_map, spine_bases_df):
    """Cross-check `map_nodes_to_spine_bases` against the exporter's own table.

    Two independent implementations must agree on the (spine_root_id,
    base_node_id) pairs. Returns a report dict; raises nothing, so the caller
    decides whether a disagreement is fatal.
    """
    mine = set(map(tuple, node_map[["spine_root_id", "base_node_id"]]
                   .drop_duplicates().to_numpy().tolist()))
    theirs = set(map(tuple, spine_bases_df[["spine_root_id", "base_node_id"]]
                     .drop_duplicates().to_numpy().tolist()))
    return {
        "n_mine": len(mine),
        "n_theirs": len(theirs),
        "n_agree": len(mine & theirs),
        "only_mine": sorted(mine - theirs)[:20],
        "only_theirs": sorted(theirs - mine)[:20],
        "agree": mine == theirs,
    }


# --------------------------------------------------------------------------- #
#  4. The audit                                                                #
# --------------------------------------------------------------------------- #
def audit_redirect(syn_df, node_xyz_nm, node_map, cell, transform_fn,
                   parent_of, sec_of_idx, sec_names, sec_length_um,
                   syn_xyz_columns=("syn_x_nm", "syn_y_nm", "syn_z_nm"),
                   node_id_column="node_id",
                   type_column="synapse_type"):
    """Per-synapse comparison of the naive free-snap against the D-7 redirect.

    Parameters
    ----------
    syn_df : DataFrame
        One row per synapse, in RAW nm, already mapped to a skeleton node by
        `map_synapses_to_segments`. Needs the three coordinate columns, the
        node id column, and optionally a type column.
    node_xyz_nm : dict
        node_id -> (x, y, z) in raw nm, from the PRE-PRUNE frame. Must contain
        every base node referenced by `node_map`.
    node_map : DataFrame
        Output of `map_nodes_to_spine_bases`.
    cell : object
        Anything exposing get_closest_idx(x=, y=, z=), allseclist, totnsegs.
    transform_fn : callable
        transform_fn(array Nx3 raw nm) -> array Nx3 aligned um. MUST be the
        identical centre-and-rotate the skeleton went through.

    Returns
    -------
    DataFrame, one row per synapse.
    """
    cx, cy, cz = syn_xyz_columns
    for col in (cx, cy, cz, node_id_column):
        if col not in syn_df.columns:
            raise ValueError("syn_df lacks required column %r" % col)

    base_of_node = dict(zip(node_map[node_id_column].astype(int).tolist(),
                            node_map["base_node_id"].astype(int).tolist()))
    root_of_node = dict(zip(node_map[node_id_column].astype(int).tolist(),
                            node_map["spine_root_id"].astype(int).tolist()))

    syn_nm = syn_df[[cx, cy, cz]].to_numpy(dtype=float)
    syn_um = np.asarray(transform_fn(syn_nm), dtype=float)
    if syn_um.shape != syn_nm.shape:
        raise ValueError("transform_fn changed the array shape")

    # anchor: the base shaft node for a spine synapse, else the synapse itself
    node_ids = syn_df[node_id_column].to_numpy(dtype=int)
    on_spine = np.array([int(n) in base_of_node for n in node_ids])

    anchor_nm = syn_nm.copy()
    for k, n in enumerate(node_ids.tolist()):
        if not on_spine[k]:
            continue
        b = base_of_node[int(n)]
        if b < 0 or b not in node_xyz_nm:
            on_spine[k] = False               # unattachable: treat as shaft
            continue
        anchor_nm[k] = node_xyz_nm[b]
    anchor_um = np.asarray(transform_fn(anchor_nm), dtype=float)

    rows = []
    for k in range(len(syn_df)):
        i_naive = int(cell.get_closest_idx(x=syn_um[k, 0], y=syn_um[k, 1],
                                           z=syn_um[k, 2]))
        i_anchor = int(cell.get_closest_idx(x=anchor_um[k, 0], y=anchor_um[k, 1],
                                            z=anchor_um[k, 2]))
        s_naive = sec_names[sec_of_idx[i_naive]]
        s_anchor = sec_names[sec_of_idx[i_anchor]]
        rel = relate_sections(s_naive, s_anchor, parent_of)
        rows.append({
            "node_id": int(node_ids[k]),
            "synapse_type": (str(syn_df[type_column].iloc[k])
                             if type_column in syn_df.columns else "unknown"),
            "on_pruned_spine": bool(on_spine[k]),
            "spine_root_id": int(root_of_node.get(int(node_ids[k]), -1)),
            "base_node_id": int(base_of_node.get(int(node_ids[k]), -1)),
            "lfpy_idx_naive": i_naive,
            "lfpy_idx_anchor": i_anchor,
            "sec_naive": s_naive,
            "sec_anchor": s_anchor,
            "same_compartment": i_naive == i_anchor,
            "same_section": s_naive == s_anchor,
            "relation": rel,
            "path_um": path_length_um(s_naive, s_anchor, parent_of, sec_length_um),
            "euclid_um": float(np.linalg.norm(syn_um[k] - anchor_um[k])),
            "syn_x": syn_um[k, 0], "syn_y": syn_um[k, 1], "syn_z": syn_um[k, 2],
            "anchor_x": anchor_um[k, 0], "anchor_y": anchor_um[k, 1],
            "anchor_z": anchor_um[k, 2],
        })
    return pd.DataFrame(rows)


def summarise_audit(audit_df, type_column="synapse_type"):
    """Headline numbers. This is what settles TEEG_16 section 3.3."""
    out = {"n_synapses": int(len(audit_df)),
           "n_on_pruned_spine": int(audit_df["on_pruned_spine"].sum())}

    at_risk = audit_df[audit_df["on_pruned_spine"]]
    out["n_moved_compartment"] = int((~at_risk["same_compartment"]).sum())
    out["n_moved_section"] = int((~at_risk["same_section"]).sum())
    out["n_foreign_branch"] = int((at_risk["relation"] == REL_FOREIGN).sum())

    denom = max(len(at_risk), 1)
    out["pct_moved_section"] = 100.0 * out["n_moved_section"] / denom
    out["pct_foreign_branch"] = 100.0 * out["n_foreign_branch"] / denom

    foreign = at_risk[at_risk["relation"] == REL_FOREIGN]
    for label, frame, col in (("path_um", foreign, "path_um"),
                              ("euclid_um", foreign, "euclid_um")):
        if len(frame):
            v = frame[col].to_numpy(dtype=float)
            out["median_%s_foreign" % label] = float(np.median(v))
            out["p90_%s_foreign" % label] = float(np.percentile(v, 90))
            out["max_%s_foreign" % label] = float(v.max())
        else:
            out["median_%s_foreign" % label] = float("nan")
            out["p90_%s_foreign" % label] = float("nan")
            out["max_%s_foreign" % label] = float("nan")

    out["by_relation"] = {k: int(v) for k, v in
                          at_risk["relation"].value_counts().items()}

    by_type = {}
    if type_column in audit_df.columns:
        for t, g in at_risk.groupby(type_column):
            by_type[str(t)] = {
                "n_on_pruned_spine": int(len(g)),
                "n_moved_section": int((~g["same_section"]).sum()),
                "n_foreign_branch": int((g["relation"] == REL_FOREIGN).sum()),
                "pct_foreign_branch": 100.0 * float((g["relation"] == REL_FOREIGN).mean()),
            }
    out["by_type"] = by_type
    out["module_version"] = MODULE_VERSION
    return out


def format_summary(summary):
    """Plain-text report. No plotting, no file I/O."""
    L = []
    L.append("SYNAPSE REDIRECT AUDIT (%s)" % summary["module_version"])
    L.append("-" * 66)
    L.append("synapses total                 %6d" % summary["n_synapses"])
    L.append("on a PRUNED SPINE (at risk)    %6d" % summary["n_on_pruned_spine"])
    L.append("")
    L.append("of those, the free snap lands on:")
    L.append("  a different compartment      %6d" % summary["n_moved_compartment"])
    L.append("  a different SECTION          %6d  (%.1f%%)"
             % (summary["n_moved_section"], summary["pct_moved_section"]))
    L.append("  a FOREIGN BRANCH             %6d  (%.1f%%)"
             % (summary["n_foreign_branch"], summary["pct_foreign_branch"]))
    L.append("")
    L.append("relation of naive snap to the true base section:")
    for k, v in sorted(summary["by_relation"].items()):
        L.append("  %-12s %6d" % (k, v))
    L.append("")
    L.append("displacement for the foreign-branch cases:")
    L.append("  cable path  median %8.2f  p90 %8.2f  max %8.2f um"
             % (summary["median_path_um_foreign"], summary["p90_path_um_foreign"],
                summary["max_path_um_foreign"]))
    L.append("  euclidean   median %8.2f  p90 %8.2f  max %8.2f um"
             % (summary["median_euclid_um_foreign"],
                summary["p90_euclid_um_foreign"], summary["max_euclid_um_foreign"]))
    if summary["by_type"]:
        L.append("")
        L.append("by synapse type:")
        for t, d in sorted(summary["by_type"].items()):
            L.append("  %-8s at risk %5d | wrong section %5d | FOREIGN %5d (%.1f%%)"
                     % (t, d["n_on_pruned_spine"], d["n_moved_section"],
                        d["n_foreign_branch"], d["pct_foreign_branch"]))
    L.append("")
    L.append("NOTE: a single-cell LFPy model cannot snap onto another neuron's")
    L.append("dendrite, so every figure above is a LOWER BOUND on the real rate.")
    return "\n".join(L)
