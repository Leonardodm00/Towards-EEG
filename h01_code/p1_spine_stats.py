#!/usr/bin/env python3
"""Per-spine statistics written by P1 at the moment spines are pruned.

Handoff section 4.1 (2026-09-19): when P1 prunes a spine subtree from the
exported morphology and redirects its synapses to the parent shaft node, the
labelled node frame -- per-node radius, head/neck class -- is in hand and
would otherwise be thrown away. Everything needed to LOCATE every spine
afterwards and to do statistics on it is written here: the nodes that belong
to it, the neck radius, the length, the base. Consumers: idea I-001
(section-by-section R_neck for human spines) and the P2/P3 mesh areas, which
join on the same key.

Two tables per cell, long first, wide derived from it by groupby so the two
can never disagree:

    neuron_<id>_spine_nodes.csv    one row per spine node       (long_table)
    neuron_<id>_spine_stats.csv    one row per spine            (wide_table)

KEY. The join key across P1 / P2 / P3 is `root_node_id`, the skeleton id of
the first spine node (the node whose parent is not a spine node). P3's
cell<id>_spines.csv carries the same id as `root_id`. `spine_id` is the
derived string f"{cell_id}:{root_node_id}", written as a column for
convenience; it is never the primary key (decision 2026-09-20).

SPINE DEFINITION. Identical to morphology_exporter.prune_spines: a spine is
a maximal parent/child-connected component of nodes whose compartment_class
is node_classify.CLS_SPINE; its root is the component node whose parent is
outside the component; its base is the parent of the root. The frame this
runs on is the one align_and_export(return_frames=True) hands back: labelled
(annotated_type in {head, neck} on spine nodes), demoted continuations
already shaft, re-classified, soma enforced, NOT yet pruned, in raw nm.

WHAT IS PURE PANDAS HERE AND WHAT IS PROJECT CODE. Path distances, frustum
areas and segment lengths are taken from spine_density's own helpers
(_prepare_nodes, _path_distances_um, _segment_length_um,
_frustum_lateral_area), the same ones h01_spine_area_F.skeleton_spine_table
uses, so `d_from_um` here is phi's d_from_um and `A_skel_frustum_um2` is the
skel variant's per-spine area. The three partition votes are recomputed with
shaft_continuation.score_spine_roots + continuation_inspect.taper_table /
three_vote on the post-demotion frame (the exporter reports only counts).

R_neck. For each spine,

    R_neck_skel_Ra100_MOhm = sum over neck nodes i of  Ra * l_i / (pi r_i^2)

with Ra = 100 ohm cm, l_i the length of the segment node -> parent and r_i
the node's own skeleton radius (distal end of that segment), converted to
MOhm. It is linear in Ra, so the analysis rescales it. THE BASE SEGMENT: the
segment base -> root runs from the parent's CENTRELINE node to the first
spine node, so most of it lies inside the parent's own radius (all of it,
when the root sits on the shaft surface; several um of it, when the base is
the collapsed soma node). For every LENGTH-type quantity (neck_len_nm,
head_len_nm, path_len_nm, R_neck) the root's segment therefore enters with
its length BEYOND the base radius, max(0, l_root - r_base); the centreline
length is kept in base_seg_len_nm and path_len_centreline_nm, and the long
table's seg_len_nm is the raw centreline length for every node. Radius
statistics are per node and unaffected. Without this the fixture spine
whose root sits on the shaft surface carried 31 percent of its R_neck inside
the shaft (review, 2026-09-20). Even so the value is an estimate from single
per-node radii, not the section-by-section integral of Ofer et al. 2026
(preprint, cited in the decisions log).

PARTITION VOTES. Recomputed on the POST-demotion frame; every root present
survived step 4b, and `partition_source` records why:
    labeller     rho/cos said spine -- never a demotion candidate
    rescued      rho/cos said shaft, the taper said head (spine_like)
    undecidable  rho/cos said shaft, too short for the taper -- kept
    kept_at_tie  rho/cos AND taper say shaft NOW, but step 4b kept it: it
                 lost the branch-point tie-break to a sibling that was
                 demoted, and became the sole candidate only afterwards
                 (shaft_continuation resolves ties to the best candidate).
                 The exporter's decision stands; this label makes the
                 disagreement visible instead of hiding it.
    no_vote      the scorer skips roots whose base is not a shaft node
                 (soma, axon): no rho/cos, no taper.

Pure ASCII, LF only. No NEURON, no network.
"""
import math

import numpy as np
import pandas as pd

MODULE_VERSION = "p1_spine_stats v1.0"

RA_REF_OHM_CM = 100.0
CLS_SPINE = "spine"
HEAD_LABELS = ("head",)
NECK_LABELS = ("neck",)

# `spine_part` is the labeller's head/neck split (annotated_type on the
# labelled frame). The handoff table called this column compartment_class;
# renamed because on the frame compartment_class is the class 'spine' for
# every one of these nodes, and one name for two objects is how joins go
# wrong.
LONG_COLUMNS = (
    "cell_id", "spine_id", "root_node_id", "node_id", "parent_node_id", "k",
    "x_nm", "y_nm", "z_nm", "x_al_um", "y_al_um", "z_al_um",
    "radius_nm", "seg_len_nm", "spine_part", "is_leaf")

WIDE_COLUMNS = (
    "cell_id", "spine_id", "root_node_id", "base_node_id",
    "n_nodes", "n_head_nodes", "n_neck_nodes",
    "base_seg_len_nm", "base_seg_len_beyond_shaft_nm",
    "path_len_nm", "path_len_centreline_nm", "neck_len_nm", "head_len_nm",
    "neck_r_min_nm", "neck_r_median_nm", "neck_r_mean_nm", "neck_r_base_nm",
    "head_r_max_nm", "base_shaft_r_nm",
    "base_x_nm", "base_y_nm", "base_z_nm", "tip_x_nm", "tip_y_nm", "tip_z_nm",
    "base_x_al_um", "base_y_al_um", "base_z_al_um",
    "tip_x_al_um", "tip_y_al_um", "tip_z_al_um",
    "d_from_um", "base_class", "branch_order",
    "spine_base_section", "section_id", "base_lfpy_idx",
    "n_syn", "syn_ids", "n_syn_exc", "n_syn_inh",
    "partition_source", "rho_vote", "cos_vote", "taper_vote", "bulge_vote",
    "A_skel_frustum_um2", "A_skel_nobase_um2", "R_neck_skel_Ra100_MOhm")


def spine_id_of(cell_id, root_node_id):
    return "%d:%d" % (int(cell_id), int(root_node_id))


# --------------------------------------------------------------------------- #
# Components                                                                  #
# --------------------------------------------------------------------------- #
def spine_components(labelled, class_column="compartment_class"):
    """root_node_id -> ordered list of (node_id, k) for every spine, k being
    the number of edges from the root along the component. Same definition
    as morphology_exporter.prune_spines; BFS order within a component."""
    if class_column not in labelled.columns:
        raise ValueError("frame has no %r column" % class_column)
    ids = labelled["id"].to_numpy(dtype=np.int64)
    par = labelled["p"].to_numpy(dtype=np.int64)
    parent = dict(zip(ids.tolist(), par.tolist()))
    is_spine = set(ids[labelled[class_column].astype(str).to_numpy() == CLS_SPINE].tolist())
    children = {}
    for i, p in parent.items():
        if i in is_spine and p in is_spine:
            children.setdefault(p, []).append(i)
    roots = sorted(i for i in is_spine if parent.get(i) not in is_spine)
    out = {}
    for r in roots:
        order, queue = [(r, 0)], [(r, 0)]
        while queue:
            cur, k = queue.pop(0)
            for ch in sorted(children.get(cur, ())):
                order.append((ch, k + 1))
                queue.append((ch, k + 1))
        out[r] = order
    return out


def _node_arrays(labelled):
    ids = labelled["id"].to_numpy(dtype=np.int64)
    pos = {int(v): i for i, v in enumerate(ids)}
    xyz = labelled[["x", "y", "z"]].to_numpy(dtype=float)
    rad = (labelled["r"].to_numpy(dtype=float) if "r" in labelled.columns
           else np.full(len(ids), np.nan))
    par = labelled["p"].to_numpy(dtype=np.int64)
    return ids, pos, xyz, rad, par


def branch_orders(labelled, class_column="compartment_class",
                  dend_classes=("dend", "apic_dend", "basal_dend")):
    """node_id -> number of DENDRITIC branch points on the path root -> node.
    A branch point is a dendrite-class node with >= 2 dendrite-class
    children; the soma is never one (its dendrite/axon fan-out is not a
    branch order), spines never count. The soma root has order 0."""
    ids, pos, xyz, rad, par = _node_arrays(labelled)
    cls = labelled[class_column].astype(str).to_numpy()
    dend = {int(i) for i, c in zip(ids, cls) if c in dend_classes}
    n_shaft_children = {}
    children = {}
    for i, p in zip(ids.tolist(), par.tolist()):
        children.setdefault(p, []).append(i)
        if i in dend and p in dend:
            n_shaft_children[p] = n_shaft_children.get(p, 0) + 1
    roots = [int(i) for i, p in zip(ids, par) if p == -1 or int(p) not in pos]
    order = {}
    stack = [(r, 0) for r in roots]
    for r, _ in stack:
        order[r] = 0
    while stack:
        cur, o = stack.pop()
        step = 1 if n_shaft_children.get(cur, 0) >= 2 else 0
        for ch in children.get(cur, ()):
            order[ch] = o + step
            stack.append((ch, o + step))
    return order


# --------------------------------------------------------------------------- #
# Long table                                                                  #
# --------------------------------------------------------------------------- #
def long_table(labelled, cell_id, align_fn=None, class_column="compartment_class",
               label_column="annotated_type"):
    """One row per spine node. `align_fn(df, nid) -> df` is
    alignment.make_align_fn(soma_pos, mean_matrix); when given, the aligned
    coordinates (nm -> um once) are added, else those columns are NaN."""
    comps = spine_components(labelled, class_column)
    ids, pos, xyz, rad, par = _node_arrays(labelled)
    labels = labelled[label_column].astype(str).str.lower().to_numpy()
    has_child = set(par.tolist())
    al = None
    if align_fn is not None:
        a = align_fn(labelled[["id", "p", "x", "y", "z"]].copy(), cell_id)
        al = a[["x", "y", "z"]].to_numpy(dtype=float) / 1000.0
    rows = []
    for root, order in comps.items():
        sid = spine_id_of(cell_id, root)
        for node, k in order:
            i = pos[node]
            p = int(par[i])
            seg = (float(np.linalg.norm(xyz[i] - xyz[pos[p]]))
                   if p in pos else float("nan"))
            lab = labels[i]
            cls = "head" if lab in HEAD_LABELS else ("neck" if lab in NECK_LABELS
                                                    else lab)
            rows.append({
                "cell_id": int(cell_id), "spine_id": sid, "root_node_id": int(root),
                "node_id": int(node), "parent_node_id": p, "k": int(k),
                "x_nm": float(xyz[i, 0]), "y_nm": float(xyz[i, 1]),
                "z_nm": float(xyz[i, 2]),
                "x_al_um": float(al[i, 0]) if al is not None else np.nan,
                "y_al_um": float(al[i, 1]) if al is not None else np.nan,
                "z_al_um": float(al[i, 2]) if al is not None else np.nan,
                "radius_nm": float(rad[i]), "seg_len_nm": seg,
                "spine_part": cls, "is_leaf": bool(node not in has_child)})
    df = pd.DataFrame(rows, columns=LONG_COLUMNS)
    return df


# --------------------------------------------------------------------------- #
# Votes                                                                       #
# --------------------------------------------------------------------------- #
def vote_table(labelled, shc, cinsp, sd, rho_shaft_min=None, cos_shaft_min=None,
               min_len_nm=None, bulge_min=None, peak_frac_min=None):
    """The three partition votes per surviving spine root, recomputed with the
    same functions the exporter's step 4b used (shaft_continuation
    .score_spine_roots, continuation_inspect.taper_table / three_vote). On the
    POST-demotion frame every root present is one the three votes kept, so
    `demote` is False for all; what is recorded is WHY it was kept:
    partition_source is one of labeller / rescued / undecidable / kept_at_tie
    (see the module docstring; kept_at_tie is a root that lost the
    branch-point tie-break at step 4b and would be demoted if voted alone
    now). Roots the scorer skips (base not a shaft node) are absent here and
    get 'no_vote' in the wide table. Returns a frame keyed root_node_id with
    rho_vote, cos_vote, taper_vote, bulge_vote."""
    kw = {}
    if rho_shaft_min is not None:
        kw["rho_shaft_min"] = float(rho_shaft_min)
    if cos_shaft_min is not None:
        kw["cos_shaft_min"] = float(cos_shaft_min)
    table, _rep = shc.score_spine_roots(labelled, spine_density=sd, **kw)
    if not len(table):
        return pd.DataFrame(columns=["root_node_id", "partition_source",
                                     "rho_vote", "cos_vote", "taper_vote",
                                     "bulge_vote"])
    tkw = {k: v for k, v in (("min_len_nm", min_len_nm), ("bulge_min", bulge_min),
                             ("peak_frac_min", peak_frac_min)) if v is not None}
    taper = cinsp.taper_table(labelled, table["root"], **tkw)
    voted = cinsp.three_vote(table, taper)
    is_shaft = voted["is_shaft"].to_numpy(bool)
    verdict = voted["verdict"].astype(str).to_numpy()
    src = np.where(~is_shaft, "labeller",
                   np.where(verdict == "spine_like", "rescued",
                            np.where(verdict == "branch_like", "kept_at_tie",
                                     "undecidable")))
    return pd.DataFrame({
        "root_node_id": voted["root"].astype(np.int64).to_numpy(),
        "partition_source": src,
        "rho_vote": voted["rho"].astype(float).to_numpy(),
        "cos_vote": voted["cos"].astype(float).to_numpy(),
        "taper_vote": voted["verdict"].astype(str).to_numpy(),
        "bulge_vote": voted["bulge"].astype(float).to_numpy()
        if "bulge" in voted.columns else np.nan})


# --------------------------------------------------------------------------- #
# Wide table                                                                  #
# --------------------------------------------------------------------------- #
def _effective_seg_len(g, root_id, base_r_nm):
    """node_id -> segment length entering every LENGTH-type statistic: the
    root's segment shortened by the base radius (never below 0), all others
    the raw centreline length; non-finite -> 0."""
    out = {}
    for r_ in g.itertuples(index=False):
        n_ = int(r_.node_id)
        raw = float(r_.seg_len_nm) if np.isfinite(r_.seg_len_nm) else 0.0
        out[n_] = max(0.0, raw - float(base_r_nm)) if n_ == int(root_id) else raw
    return out


def _r_neck_mohm(seg_len_nm, radius_nm, ra_ohm_cm=RA_REF_OHM_CM):
    """sum Ra * l / (pi r^2), nm -> cm, ohm -> MOhm. NaN-safe: a segment with
    a non-finite length or radius contributes nothing and is counted."""
    l_cm = np.asarray(seg_len_nm, float) * 1e-7
    r_cm = np.asarray(radius_nm, float) * 1e-7
    ok = np.isfinite(l_cm) & np.isfinite(r_cm) & (r_cm > 0)
    if not ok.any():
        return float("nan")
    return float(np.sum(ra_ohm_cm * l_cm[ok] / (math.pi * r_cm[ok] ** 2)) / 1e6)


def wide_table(long_df, labelled, cell_id, sd, spine_bases=None,
               mapped_synapses=None, votes=None, class_column="compartment_class"):
    """One row per spine, derived from the long table by groupby plus the
    joins the long table cannot carry (base geometry, path distance, hoc
    section, synapses, votes)."""
    ids, pos, xyz, rad, par = _node_arrays(labelled)
    cls_all = labelled[class_column].astype(str).to_numpy()
    node, children, root = sd._prepare_nodes(labelled, sd.SHAFT_REGEX,
                                             sd.SPINE_LABELS,
                                             sd.DEFAULT_RADIUS_NM, "nm")
    dist_um = sd._path_distances_um(node, children, root)
    border = branch_orders(labelled, class_column)

    # aligned base/tip: from the long table's aligned columns (NaN if absent)
    have_al = bool(np.isfinite(long_df["x_al_um"].to_numpy(float)).any()) \
        if len(long_df) else False

    sb = None
    if spine_bases is not None and len(spine_bases):
        sb = spine_bases.set_index("spine_root_id")
    syn_by_root = {}
    if mapped_synapses is not None and len(mapped_synapses):
        m = mapped_synapses
        on = m["on_pruned_spine"].astype(bool) if "on_pruned_spine" in m.columns \
            else pd.Series(False, index=m.index)
        for r, g in m[on].groupby("spine_root_id"):
            syn_by_root[int(r)] = g
    vt = votes.set_index("root_node_id") if votes is not None and len(votes) else None

    rows = []
    for root_id, g in long_df.groupby("root_node_id", sort=True):
        root_id = int(root_id)
        g = g.sort_values("k")
        base = int(par[pos[root_id]])
        neck = g[g["spine_part"] == "neck"]
        head = g[g["spine_part"] == "head"]
        base_in = base in pos
        bi = pos[base] if base_in else None
        base_r = float(rad[bi]) if base_in and np.isfinite(rad[bi]) else 0.0
        # effective segment lengths: the root's segment enters with its length
        # beyond the base radius (module docstring), every other one as is
        eff = _effective_seg_len(g, root_id, base_r)
        root_seg = float(g.loc[g["node_id"] == root_id, "seg_len_nm"].iloc[0])
        # path length base -> farthest node along k-chains, effective and centreline
        cum, cum_c = {}, {}
        for r_ in g.itertuples(index=False):
            n_, p_ = int(r_.node_id), int(r_.parent_node_id)
            raw = float(r_.seg_len_nm) if np.isfinite(r_.seg_len_nm) else 0.0
            cum[n_] = cum.get(p_, 0.0) + eff[n_]
            cum_c[n_] = cum_c.get(p_, 0.0) + raw
        tip_node = max(cum_c, key=cum_c.get)
        ti = pos[tip_node]
        # skeleton frustum areas, spine_density's own helper (um)
        area, a_base = 0.0, float("nan")
        for r_ in g.itertuples(index=False):
            s, p = int(r_.node_id), int(r_.parent_node_id)
            if s in node and p in node:
                area += sd._frustum_lateral_area(node[p]["r"], node[s]["r"],
                                                 sd._segment_length_um(node, p, s))
        if base_in and root_id in node and base in node:
            a_base = sd._frustum_lateral_area(node[base]["r"], node[root_id]["r"],
                                              sd._segment_length_um(node, base, root_id))
        syn = syn_by_root.get(root_id)
        if syn is not None:
            syn_ids = ";".join(str(v) for v in syn["syn_id"].tolist()) \
                if "syn_id" in syn.columns else ""
            lab = syn["synapse_label"].astype(str) if "synapse_label" in syn.columns \
                else pd.Series([], dtype=str)
            n_exc = int(lab.str.startswith("exc").sum())
            n_inh = int(lab.str.startswith("inh").sum())
            lfpy = sorted(set(int(v) for v in syn["lfpy_idx"].dropna().tolist())) \
                if "lfpy_idx" in syn.columns else []
            base_lfpy = ";".join(str(v) for v in lfpy)
            n_syn = int(len(syn))
        else:
            syn_ids, n_exc, n_inh, base_lfpy, n_syn = "", 0, 0, "", 0
        sec_name, sec_id = "", -1
        if sb is not None and root_id in sb.index:
            raw_sec = sb.loc[root_id, "spine_base_section"]
            sec_name = "" if (raw_sec is None or (isinstance(raw_sec, float)
                                                  and np.isnan(raw_sec))) else str(raw_sec)
            sec_id = int(sb.loc[root_id, "section_id"])
        v = vt.loc[root_id] if (vt is not None and root_id in vt.index) else None
        rows.append({
            "cell_id": int(cell_id), "spine_id": spine_id_of(cell_id, root_id),
            "root_node_id": root_id, "base_node_id": base,
            "n_nodes": int(len(g)), "n_head_nodes": int(len(head)),
            "n_neck_nodes": int(len(neck)),
            "base_seg_len_nm": root_seg if np.isfinite(root_seg) else np.nan,
            "base_seg_len_beyond_shaft_nm": eff[root_id],
            "path_len_nm": float(cum[tip_node]),
            "path_len_centreline_nm": float(cum_c[tip_node]),
            "neck_len_nm": float(sum(eff[int(n_)] for n_ in neck["node_id"])) if len(neck) else 0.0,
            "head_len_nm": float(sum(eff[int(n_)] for n_ in head["node_id"])) if len(head) else 0.0,
            "neck_r_min_nm": float(neck["radius_nm"].min()) if len(neck) else np.nan,
            "neck_r_median_nm": float(neck["radius_nm"].median()) if len(neck) else np.nan,
            "neck_r_mean_nm": float(neck["radius_nm"].mean()) if len(neck) else np.nan,
            "neck_r_base_nm": (float(g.iloc[0]["radius_nm"])
                               if g.iloc[0]["spine_part"] == "neck" else np.nan),
            "head_r_max_nm": float(head["radius_nm"].max()) if len(head) else np.nan,
            "base_shaft_r_nm": float(rad[bi]) if base_in else np.nan,
            "base_x_nm": float(xyz[bi, 0]) if base_in else np.nan,
            "base_y_nm": float(xyz[bi, 1]) if base_in else np.nan,
            "base_z_nm": float(xyz[bi, 2]) if base_in else np.nan,
            "tip_x_nm": float(xyz[ti, 0]), "tip_y_nm": float(xyz[ti, 1]),
            "tip_z_nm": float(xyz[ti, 2]),
            "base_x_al_um": np.nan, "base_y_al_um": np.nan, "base_z_al_um": np.nan,
            "tip_x_al_um": (float(g.loc[g["node_id"] == tip_node, "x_al_um"].iloc[0])
                            if have_al else np.nan),
            "tip_y_al_um": (float(g.loc[g["node_id"] == tip_node, "y_al_um"].iloc[0])
                            if have_al else np.nan),
            "tip_z_al_um": (float(g.loc[g["node_id"] == tip_node, "z_al_um"].iloc[0])
                            if have_al else np.nan),
            "d_from_um": float(dist_um.get(base, np.nan)) if base_in else np.nan,
            "base_class": str(cls_all[bi]) if base_in else "",
            "branch_order": int(border.get(base, -1)) if base_in else -1,
            "spine_base_section": sec_name, "section_id": sec_id,
            "base_lfpy_idx": base_lfpy,
            "n_syn": n_syn, "syn_ids": syn_ids, "n_syn_exc": n_exc, "n_syn_inh": n_inh,
            "partition_source": (str(v["partition_source"]) if v is not None else "no_vote"),
            "rho_vote": float(v["rho_vote"]) if v is not None else np.nan,
            "cos_vote": float(v["cos_vote"]) if v is not None else np.nan,
            "taper_vote": str(v["taper_vote"]) if v is not None else "",
            "bulge_vote": float(v["bulge_vote"]) if v is not None else np.nan,
            "A_skel_frustum_um2": float(area),
            "A_skel_nobase_um2": float(area - a_base) if np.isfinite(a_base) else float(area),
            "R_neck_skel_Ra100_MOhm": _r_neck_mohm(
                np.array([eff[int(n_)] for n_ in neck["node_id"]], float),
                neck["radius_nm"].to_numpy(float)),
        })
    return pd.DataFrame(rows, columns=WIDE_COLUMNS)


def add_aligned_base(wide, labelled, align_fn, cell_id):
    """Fill base_*_al_um from the base node's raw coordinate through
    alignment.make_align_fn's map (nm -> um once). The base is a SHAFT node,
    absent from the long table, which is why the wide table cannot take it
    from there and the map is re-applied here -- same function, exact."""
    if align_fn is None or not len(wide):
        return wide
    ids, pos, xyz, rad, par = _node_arrays(labelled)
    a = align_fn(labelled[["id", "p", "x", "y", "z"]].copy(), cell_id)
    al = a[["x", "y", "z"]].to_numpy(dtype=float) / 1000.0
    out = wide.copy()
    for i, b in enumerate(out["base_node_id"].to_numpy(dtype=np.int64)):
        if int(b) in pos:
            j = pos[int(b)]
            out.loc[out.index[i], ["base_x_al_um", "base_y_al_um", "base_z_al_um"]] = \
                [float(al[j, 0]), float(al[j, 1]), float(al[j, 2])]
    return out


def write_tables(out_dir, cell_id, long_df, wide_df):
    """Atomic-ish, idempotent: written to .tmp then renamed."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = {"spine_nodes": os.path.join(out_dir, "neuron_%d_spine_nodes.csv" % int(cell_id)),
             "spine_stats": os.path.join(out_dir, "neuron_%d_spine_stats.csv" % int(cell_id))}
    for key, df in (("spine_nodes", long_df), ("spine_stats", wide_df)):
        tmp = paths[key] + ".tmp"
        df.to_csv(tmp, index=False, lineterminator="\n")
        os.replace(tmp, paths[key])
    return paths


def check_wide_is_groupby_of_long(long_df, wide_df):
    """The consistency invariant the smoke test asserts: every count and every
    neck/head statistic in the wide table equals the groupby of the long one
    (lengths through the same effective-segment rule)."""
    if not len(long_df) and not len(wide_df):
        return True, ""
    g = long_df.groupby("root_node_id")
    w = wide_df.set_index("root_node_id")
    if sorted(g.groups.keys()) != sorted(w.index.tolist()):
        return False, "root sets differ"
    for root, sub in g:
        neck = sub[sub["spine_part"] == "neck"]
        head = sub[sub["spine_part"] == "head"]
        base_r = float(w.loc[root, "base_shaft_r_nm"])
        eff = _effective_seg_len(sub, root, base_r if np.isfinite(base_r) else 0.0)
        checks = [
            (w.loc[root, "n_nodes"], len(sub)),
            (w.loc[root, "n_neck_nodes"], len(neck)),
            (w.loc[root, "n_head_nodes"], len(head)),
        ]
        for a, b in checks:
            if int(a) != int(b):
                return False, "root %d count mismatch %r vs %r" % (root, a, b)
        if len(neck):
            for col, val in (("neck_r_min_nm", neck["radius_nm"].min()),
                             ("neck_r_median_nm", neck["radius_nm"].median()),
                             ("neck_r_mean_nm", neck["radius_nm"].mean()),
                             ("neck_len_nm", sum(eff[int(n_)] for n_ in neck["node_id"]))):
                if abs(float(w.loc[root, col]) - float(val)) > 1e-9:
                    return False, "root %d %s %r vs %r" % (root, col, w.loc[root, col], val)
        if len(head) and abs(float(w.loc[root, "head_r_max_nm"]) - float(head["radius_nm"].max())) > 1e-9:
            return False, "root %d head_r_max" % root
    return True, ""
