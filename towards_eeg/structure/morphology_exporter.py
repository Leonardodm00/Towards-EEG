"""morphology_exporter -- stage S1.0: the one merged exporter.

WHAT THIS MODULE ESTABLISHES
----------------------------
A single definition that replaces BOTH inherited exporters and resolves the
defects each of them carries, in one traversal rather than four passes
(roadmap risk R-2's own mitigation).

The two definitions it replaces, and why neither was usable:

  morpholgy_pathways__6_.py L842   emits the synapse label as a section name
                                   (defect O8): 'exc_syn[k]' / 'inh_syn[k]'
                                   become section arrays, so 25-50 percent of
                                   sections lose their compartment class and in
                                   one sampled file the soma itself is emitted
                                   as exc_syn[0].
  morpholgy_pathways__6_.py L1331  emits spine sub-structure as sections
                                   ('head[k]', 'neck[k]', 'spine[k]') at a
                                   hard-coded 3000 nm threshold, and no domain.

Both are additionally reached through a SHADOWED aligner
(align_neurons_to_neighborhood at L973 and L1448), so the inherited bank is
heterogeneous: the L2/L4 batch was written by L973+L842 and the L5 batch by
L1448+L1331, into the same folder with the same filename pattern. That is why
regeneration is mandatory and in-place repair is not possible.

Shared with both, and also fixed here:
  * AIS annotations fell through to 'dend' ('axon' not in 'ais')
  * Astrocyte annotations fell through to 'dend' (merge-error contamination)
  * the section traversal split the cable at every class change, and since the
    synapse label WAS the class, every synapse-bearing node fractured the
    cable. That, not electrotonic length, is what set the discretisation:
    16,280 sections for 16,362 um of cable, median section length 0.541 um.

WHAT IS PRESERVED FROM THE INHERITED EXPORTER, DELIBERATELY
-----------------------------------------------------------
The .hoc emission conventions are kept verbatim because the inherited bank is
known to load in NEURON and we do not want to debug a new dialect at the same
time as new logic:
  * pt3dclear() then one pt3dadd(x, y, z, diam) per node, coordinates in um
  * connect child(0), parent(1)
  * the parent node is repeated as the first pt3d of each child section, so
    sections are geometrically continuous and no length is double counted
  * a one-node section is emitted as two points at z-r and z+r with diam = 2r.
    For the soma this is exactly area preserving: pi*diam*L = 4*pi*r^2, the
    sphere area. See soma_enforce for the full argument.

ORDER OF OPERATIONS, AND WHY
----------------------------
  1. divert the synapse label into its own frame (C-09)
  2. classify                (S1.1; synapse label never reaches this step)
  3. resolve mislabels       (S1.1; astrocyte and unknown annotations)
  4. label spines            (spine_labeller, threshold passed EXPLICITLY)
  5. re-classify             (the labeller rewrote annotated_type)
  6. enforce the soma        (S1.2; soft QC per decision Q6)
  7. build phi               (spine_density, on the UNPRUNED frame -- the spine
                             geometry must still exist to have an area)
  8. prune spines            (D-7: geometry removed, area retained in phi)
  9. decompose sections      (on the PRUNED frame)
 10. reconcile phi <-> hoc   (section_id <-> branch_id <-> node_from/node_to)
 11. write .hoc, phi, section map, synapse frame, provenance
 12. validate in NEURON      (optional, requires the neuron package)

Step 3 before step 4 is deliberate. While the frame still carries the raw H01
vocabulary no node is labelled head or neck, so the k-NN vote CANNOT resolve a
mid-cable node to 'spine' -- the class does not exist yet. This makes that
failure mode unreachable rather than merely mitigated. It also means a node
resolved to dendrite is then evaluated for spine-hood like any other dendrite,
which is the consistent behaviour.

Step 5 is mandatory: the labeller overwrites annotated_type and leaves
compartment_class stale.

Step 7 before step 8 is not an implementation detail: phi is the reason the
pruning is lossless.

Step 9 after step 8 is what makes phi's branches and the emitted dend sections
the SAME partition. spine_density._decompose_branches splits the shaft at nodes
with two or more SHAFT children; a node with one dendrite child and one spine
child is a branch point in the raw tree but not a shaft junction. Once spines
are pruned it has a single child and is not a branch point either. The two
decompositions therefore coincide by construction rather than by coincidence,
and the section map records the correspondence explicitly.

O7 HYGIENE
----------
_prepare_nodes and _decompose_branches are IMPORTED from spine_density, never
reimplemented (Doc 14 section 7 point 1). They are private names there. They
are used privately here rather than promoted, so that spine_density.py stays
byte-identical to the file whose sha256 Doc 14 records. Promoting them to
public API is a spine_density-1.3.0 + Doc 2 rev 4 action to be taken together,
not silently now.

CONTRACT DEVIATIONS, FLAGGED
----------------------------
  * 'ais[k]' is a fifth section array; C8.1 admits four pairs (see
    node_classify). Ratify (ais, none) or fold into axon[].
  * domains are all 'none' until S1.4, so dendrites emit 'dend[k]'. The hook
    assign_domain() has the final signature already.

ASCII only, LF only, no top-level side effects.
"""

import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict

import numpy as np
import pandas as pd

import spine_density as sd
import node_classify as nc
import soma_enforce as se


MODULE_VERSION = "morphology_exporter-1.0.0"

# Project setting D-S1.3-d. Chosen between the two shadowed defaults
# (5000 nm at L1653, 3000 nm at L2353). Passed explicitly at every call site
# and recorded in provenance; never left to a callee's default.
SPINE_LENGTH_THRESHOLD_NM = 4000.0

NM_PER_UM = 1000.0

# Section-array emission order in the .hoc header.
ARRAY_ORDER = ("soma", "axon", "ais", "dend", "apic_dend", "basal_dend")


# --------------------------------------------------------------------------- #
# S1.4 hook -- signature is final, behaviour is not implemented               #
# --------------------------------------------------------------------------- #
def assign_domain(df, class_column="compartment_class",
                  soma_id=None, out_column="domain"):
    """Assign (apical / basal / none) to every node. RETIRED: always 'none'.

    Decision (this handoff, superseding D-4): S3 is dropped entirely, every
    dendrite is mechanism-uniform 'dend'. D-4's geometric rule -- after
    alignment the apical trunk runs along +z by construction, so from the
    soma the subtree maximising path length and z-extent is apical and the
    rest is basal -- is recorded here for history only and will not run.

    Every node gets DOM_NONE, so dendrites emit 'dend[k]', the array name the
    inherited bank already used. assert_domain_collapsed on the OUTPUT of this
    function (not on the constant it assigns) is the guard: it is what a
    future accidental edit to the line below would actually trip, rather than
    only a docstring saying not to make that edit.
    """
    out = df.copy()
    out[out_column] = nc.DOM_NONE
    nc.assert_domain_collapsed(out[out_column].unique().tolist(),
                               "assign_domain output")
    return out


# --------------------------------------------------------------------------- #
# Spine pruning                                                               #
# --------------------------------------------------------------------------- #
def prune_spines(df, class_column="compartment_class"):
    """Remove spine-classified nodes and everything below them.

    Returns (df_pruned, info). info records, for every pruned spine, the BASE
    node -- the shaft node the spine hung off -- which is what
    'spine_base_section' in C-09 is resolved against. The base is recorded
    BEFORE pruning, per contract C-09: a post-hoc nearest-neighbour snap is
    unsafe, because a spine head sits ~2 um from its own shaft and in dense
    neuropil is frequently closer to a different neuron's dendrite.

    A spine is a maximal parent/child-connected component of spine-classified
    nodes; its base is the parent of its root.
    """
    if class_column not in df.columns:
        raise ValueError("frame has no %r column; run classify_frame first"
                         % class_column)
    cls = df[class_column].astype(str)
    spine_mask = cls == nc.CLS_SPINE
    info = {"n_spine_nodes": int(spine_mask.sum()), "n_spines": 0,
            "spine_bases": [], "n_nodes_removed": 0}
    if not spine_mask.any():
        return df.copy(), info

    spine_ids = set(df.loc[spine_mask, "id"].tolist())
    par = dict(zip(df["id"].tolist(), df["p"].tolist()))
    children = defaultdict(list)
    for i, p in par.items():
        if p != -1:
            children[p].append(i)

    roots = [i for i in spine_ids if par.get(i) not in spine_ids]
    remove = set()
    bases = []
    for r in roots:
        comp, stack = [], [r]
        while stack:
            cur = stack.pop()
            comp.append(cur)
            stack.extend(children.get(cur, ()))
        remove.update(comp)
        bases.append({"spine_root_id": _pyid(r),
                      "base_node_id": _pyid(par.get(r)),
                      "n_nodes": len(comp)})
    info["n_spines"] = len(roots)
    info["spine_bases"] = bases

    out = df.loc[~df["id"].isin(remove)].copy()
    info["n_nodes_removed"] = int(len(df) - len(out))
    return out, info


def _pyid(v):
    """numpy scalar -> plain python, so reports stay JSON-safe."""
    if v is None:
        return None
    if isinstance(v, (str, bool)):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


# --------------------------------------------------------------------------- #
# Section decomposition                                                       #
# --------------------------------------------------------------------------- #
def decompose_sections(df, class_column="compartment_class",
                       domain_column="domain"):
    """Partition the PRUNED tree into NEURON sections.

    A section is broken at a branch point, at a compartment-class change, and
    at a domain change. It is NOT broken by anything else -- in particular not
    by a synapse, which is the defect this replaces.

    Returns a list of dicts with keys:
        section_id, array, class, domain, nodes (ordered ids), parent_sec_id

    The parent node is repeated as the first entry of each child's node list,
    exactly as the inherited exporter does, so that sections join without a gap
    and no segment is counted twice.
    """
    if len(df) == 0:
        return []
    root_id, n_roots = se.topological_root(df)
    if root_id is None:
        raise ValueError("cannot decompose: no root node (p == -1)")

    cls = dict(zip(df["id"].tolist(), df[class_column].astype(str).tolist()))
    if domain_column in df.columns:
        dom = dict(zip(df["id"].tolist(), df[domain_column].astype(str).tolist()))
    else:
        dom = {i: nc.DOM_NONE for i in df["id"].tolist()}

    children = defaultdict(list)
    for i, p in zip(df["id"].tolist(), df["p"].tolist()):
        if p != -1:
            children[p].append(i)

    sections = []
    # stack entries: (node_id, accumulated node list, class, domain, parent_sec)
    stack = [(root_id, [], cls[root_id], dom[root_id], -1)]
    while stack:
        node_id, acc, c_cls, c_dom, parent_sec = stack.pop()
        acc = list(acc)
        acc.append(node_id)
        chs = children.get(node_id, [])

        if len(chs) == 0:
            sections.append(_sec(acc, c_cls, c_dom, parent_sec, len(sections)))
            continue

        if len(chs) == 1:
            ch = chs[0]
            if cls[ch] == c_cls and dom[ch] == c_dom:
                stack.append((ch, acc, c_cls, c_dom, parent_sec))
            else:
                sid = len(sections)
                sections.append(_sec(acc, c_cls, c_dom, parent_sec, sid))
                stack.append((ch, [node_id], cls[ch], dom[ch], sid))
            continue

        sid = len(sections)
        sections.append(_sec(acc, c_cls, c_dom, parent_sec, sid))
        for ch in chs:
            stack.append((ch, [node_id], cls[ch], dom[ch], sid))

    # index within each array
    counts = defaultdict(int)
    for s in sections:
        s["type_idx"] = counts[s["array"]]
        counts[s["array"]] += 1
    return sections


def _sec(nodes, cls, dom, parent_sec, sid):
    return {"section_id": sid,
            "array": nc.section_array_name(cls, dom),
            "class": cls, "domain": dom,
            "nodes": [_pyid(n) for n in nodes],
            "parent_sec_id": parent_sec}


# --------------------------------------------------------------------------- #
# phi <-> section reconciliation (decision Q1: store the map)                 #
# --------------------------------------------------------------------------- #
def build_section_map(sections, phi_df):
    """Explicit map section_id <-> branch_id, joined on (node_from, node_to).

    Doc 14 section 7 point 1 names the node-id pair as the stable join key,
    valid regardless of how either side decomposes. This function materialises
    the correspondence instead of leaving it implicit.

    Returns (seg_map, sec_table):
        seg_map   one row per emitted SEGMENT: section_id, array, type_idx,
                  seg_index, node_from, node_to, branch_id, phi_um,
                  spine_area_um2, shaft_area_um2, seg_len_um, d_from_um
        sec_table one row per SECTION: section_id, array, type_idx, class,
                  domain, parent_sec_id, n_pt3d, node_first, node_last,
                  branch_id (or -1), n_segments
    """
    if phi_df is not None and len(phi_df):
        key = {}
        for r in phi_df.itertuples(index=False):
            key[(_pyid(r.node_from), _pyid(r.node_to))] = r
    else:
        key = {}

    seg_rows, sec_rows = [], []
    for s in sections:
        nodes = s["nodes"]
        branch_ids = []
        n_seg = 0
        for i in range(1, len(nodes)):
            a, b = nodes[i - 1], nodes[i]
            r = key.get((a, b))
            branch_ids.append(int(r.branch_id) if r is not None else -1)
            seg_rows.append({
                "section_id": s["section_id"],
                "array": s["array"],
                "type_idx": s["type_idx"],
                "seg_index": i - 1,
                "node_from": a,
                "node_to": b,
                "branch_id": int(r.branch_id) if r is not None else -1,
                "phi_um": float(r.phi_um) if r is not None else 0.0,
                "spine_area_um2": float(r.spine_area_um2) if r is not None else 0.0,
                "shaft_area_um2": float(r.shaft_area_um2) if r is not None else 0.0,
                "seg_len_um": float(r.seg_len_um) if r is not None else float("nan"),
                "d_from_um": float(r.d_from_um) if r is not None else float("nan"),
            })
            n_seg += 1
        uniq = sorted({b for b in branch_ids if b >= 0})
        sec_rows.append({
            "section_id": s["section_id"],
            "array": s["array"],
            "type_idx": s["type_idx"],
            "class": s["class"],
            "domain": s["domain"],
            "parent_sec_id": s["parent_sec_id"],
            "n_pt3d": len(nodes),
            "n_segments": n_seg,
            "node_first": nodes[0],
            "node_last": nodes[-1],
            "branch_id": (uniq[0] if len(uniq) == 1 else (-1 if not uniq else -2)),
            "n_branch_ids": len(uniq),
        })
    return pd.DataFrame(seg_rows), pd.DataFrame(sec_rows)


def resolve_spine_base_sections(spine_bases, sections, df_pruned):
    """Attach each pruned spine to the section carrying its base segment.

    Convention D-S1.3-b, matched exactly to spine_density's attribution: the
    spine belongs to the segment whose DISTAL node is the base node, i.e. the
    edge (parent(base) -> base). If the base is the root (no incoming edge) the
    spine goes to the root's first outgoing section, so no spine is orphaned.
    """
    par = dict(zip(df_pruned["id"].tolist(), df_pruned["p"].tolist()))
    edge_to_sec = {}
    first_child_sec = {}
    for s in sections:
        nodes = s["nodes"]
        for i in range(1, len(nodes)):
            edge_to_sec[(nodes[i - 1], nodes[i])] = s
        if len(nodes) >= 2 and nodes[0] not in first_child_sec:
            first_child_sec[nodes[0]] = s

    out = []
    for sb in spine_bases:
        base = sb["base_node_id"]
        sec = None
        if base in par and par[base] != -1:
            sec = edge_to_sec.get((_pyid(par[base]), base))
        if sec is None:
            sec = first_child_sec.get(base)
        out.append({
            "spine_root_id": sb["spine_root_id"],
            "base_node_id": base,
            "n_nodes": sb["n_nodes"],
            "spine_base_section": ("%s[%d]" % (sec["array"], sec["type_idx"])
                                   if sec is not None else None),
            "section_id": (sec["section_id"] if sec is not None else -1),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# .hoc emission                                                               #
# --------------------------------------------------------------------------- #
def write_hoc(sections, df, path, input_units="nm", header_comment=None):
    """Write a NEURON-compliant .hoc. Conventions preserved from the inherited
    exporter (see module docstring). Coordinates converted to um exactly once.

    Guards two invariants on the array names about to be emitted: I-16 (no
    synapse label leaks in as a class) and the retirement of the apical/basal
    split (no section may be routed into apic_dend/basal_dend). The second
    guard is the SINK: it fires regardless of whether a section got there via
    assign_domain, a hand-built `sections` list, or anything else, which is
    what makes it the actual enforcement rather than assign_domain's
    self-check alone.
    """
    scale = NM_PER_UM if input_units == "nm" else 1.0
    nd = df.set_index("id")[["x", "y", "z", "r"]].to_dict("index")

    counts = defaultdict(int)
    for s in sections:
        counts[s["array"]] = max(counts[s["array"]], s["type_idx"] + 1)
    nc.assert_domain_collapsed(counts.keys(), "emitted section arrays")

    arrays = [a for a in ARRAY_ORDER if counts.get(a, 0) > 0]
    extra = sorted(a for a in counts if a not in ARRAY_ORDER and counts[a] > 0)
    arrays.extend(extra)
    nc.assert_no_synapse_leakage(arrays, "emitted section arrays")

    lines = ["// NEURON HOC morphology generated by %s" % MODULE_VERSION]
    if header_comment:
        for ln in str(header_comment).splitlines():
            lines.append("// %s" % ln)
    lines.append("")
    for a in arrays:
        lines.append("create %s[%d]" % (a, counts[a]))
    lines.append("")

    for s in sections:
        if s["parent_sec_id"] != -1:
            p = sections[s["parent_sec_id"]]
            lines.append("connect %s[%d](0), %s[%d](1)"
                         % (s["array"], s["type_idx"], p["array"], p["type_idx"]))
    lines.append("")

    for s in sections:
        lines.append("%s[%d] {" % (s["array"], s["type_idx"]))
        lines.append("  pt3dclear()")
        ns = s["nodes"]
        if len(ns) == 1:
            d = nd[ns[0]]
            x, y, z = d["x"] / scale, d["y"] / scale, d["z"] / scale
            r = d["r"] / scale
            lines.append("  pt3dadd(%r, %r, %r, %r)" % (x, y, z - r, 2.0 * r))
            lines.append("  pt3dadd(%r, %r, %r, %r)" % (x, y, z + r, 2.0 * r))
        else:
            for n in ns:
                d = nd[n]
                lines.append("  pt3dadd(%r, %r, %r, %r)"
                             % (d["x"] / scale, d["y"] / scale, d["z"] / scale,
                                2.0 * d["r"] / scale))
        lines.append("}")
        lines.append("")

    text = "\n".join(lines)
    text.encode("ascii")                      # raises if anything non-ASCII crept in
    with open(path, "w", newline="\n") as fh:
        fh.write(text)
    return {"path": str(path), "n_sections": len(sections),
            "arrays": {a: int(counts[a]) for a in arrays},
            "sha256": hashlib.sha256(text.encode("ascii")).hexdigest()}


# --------------------------------------------------------------------------- #
# Provenance                                                                  #
# --------------------------------------------------------------------------- #
def _git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL)
        return out.decode("ascii").strip()
    except Exception:                          # noqa: BLE001
        return None


def exporter_id(label_fn=None, threshold_nm=SPINE_LENGTH_THRESHOLD_NM):
    """git commit + qualnames + threshold, per S1.0's own exit requirement."""
    parts = [MODULE_VERSION, nc.MODULE_VERSION, se.MODULE_VERSION,
             sd.MODULE_VERSION]
    commit = _git_commit()
    if commit:
        parts.append("git:%s" % commit[:12])
    name = "?"
    if label_fn is not None:
        name = (getattr(label_fn, "__qualname__", None)
                or getattr(label_fn, "__name__", None)
                or type(label_fn).__name__)
    parts.append("labeller:%s" % name)
    parts.append("thr:%.0fnm" % float(threshold_nm))
    return "|".join(parts)


# --------------------------------------------------------------------------- #
# The S1.0 entry point                                                        #
# --------------------------------------------------------------------------- #
def export_neuron(df_raw,
                  nid,
                  output_dir,
                  label_fn=None,
                  spine_length_threshold_nm=SPINE_LENGTH_THRESHOLD_NM,
                  input_units="nm",
                  align_fn=None,
                  mislabel_policy="knn_relabel",
                  mislabel_k=5,
                  mislabel_max_distance_nm=None,
                  origin_tolerance_nm=None,
                  plot_fn=None,
                  write_files=True,
                  return_frames=False,
                  verbose=True):
    """Export one morphology: .hoc plus phi plus the C-09-bound tables.

    Parameters
    ----------
    df_raw : DataFrame
        Skeleton with columns id, p, x, y, z, r, annotated_type, and optionally
        synapse_label. Already spine-labelled if label_fn is None.
    label_fn : callable or None
        In-memory spine labeller, called as label_fn(df, threshold_nm) -> df.
        If None the frame is assumed already labelled (head / neck / spine).
    align_fn : callable or None
        align_fn(df, nid) -> df, applied AFTER phi is built. phi depends only on
        path distances and radii, both preserved by a rigid transform, so the
        order is immaterial for phi and required for the .hoc.
    return_frames : bool
        When True, res['frames']['labelled'] carries the PRE-PRUNE labelled
        frame in RAW nm -- classified, mislabels resolved, spine-labelled,
        re-classified, soma-enforced. This is the exact frame
        alignment.resolve_synapse_anchors needs. It exists because the
        alternative is for the caller to replicate steps 2-6 by hand, which is
        what colab_run_synapse_redirect_audit CELL 6 used to do: two copies of
        the same six steps, silently diverging the moment the mislabel policy
        or the spine threshold changes. Off by default because the frame is
        large and most callers do not need it.

    Returns
    -------
    dict with the per-cell record, every report, and the written paths.
    """
    res = {"nid": _pyid(nid), "module_version": MODULE_VERSION,
           "spine_length_threshold_nm": float(spine_length_threshold_nm),
           "qc_status": se.QC_PASS, "reasons": [], "files": {}}

    df = df_raw.copy()

    # 1. synapse label out of the geometry path, before anything touches it ---
    syn_frame = nc.extract_synapse_frame(df, nid=nid, input_units=input_units)

    # 2. classify (synapse label is NOT passed) -------------------------------
    df = nc.classify_frame(df)
    res["classes_raw"] = nc.class_counts(df)

    # 3. resolve mislabels, BEFORE the spine labeller runs --------------------
    #    Order is load-bearing. At this point 'annotated_type' still carries the
    #    raw H01 vocabulary (Soma / Dendrite / Axon / AIS / Astrocyte) and no
    #    node is yet labelled head or neck, so the k-NN vote cannot resolve a
    #    mid-cable node to 'spine' -- the category does not exist yet. Running
    #    the resolution here makes that failure mode unreachable rather than
    #    merely mitigated by VOTE_CLASS_ALIAS, which is retained as a safety net
    #    for callers who resolve an already-labelled frame.
    #
    #    The vote tally is unchanged by this move: the labeller RENAMES nodes,
    #    it does not move them, so the points voting 'Dendrite' here are exactly
    #    the points that would vote 'spine'-aliased-to-'dend' afterwards.
    df, mis = nc.resolve_mislabelled_nodes(
        df, policy=mislabel_policy, k=mislabel_k,
        max_distance_nm=mislabel_max_distance_nm, rewrite_annotation=True)
    res["mislabel_report"] = mis

    # 4. spines ---------------------------------------------------------------
    if label_fn is not None:
        df = label_fn(df, spine_length_threshold_nm)

    # 5. re-classify ----------------------------------------------------------
    #    MANDATORY, not tidiness. The labeller overwrites 'annotated_type' with
    #    'head' / 'neck' and does NOT touch 'compartment_class', so without this
    #    the class column is stale for every spine node -- the same two-columns-
    #    drifting defect that made phi and the sections disagree. The resolution
    #    from step 3 survives because rewrite_annotation encoded it in
    #    'annotated_type', which is why that flag is required here.
    df = nc.classify_frame(df)
    res["classes_labelled"] = nc.class_counts(df)

    # 6. soma ----------------------------------------------------------------
    df, soma_rep = se.enforce_soma(
        df, nid=nid, origin_tolerance_nm=origin_tolerance_nm,
        plot_fn=plot_fn,
        plot_path=(os.path.join(output_dir, "neuron_%s_soma_qc.html" % nid)
                   if (plot_fn is not None and write_files) else None),
        verbose=verbose)
    res["soma_report"] = soma_rep
    res["qc_status"] = soma_rep["qc_status"]
    res["reasons"] = list(soma_rep["reasons"])
    if soma_rep["qc_status"] == se.QC_FAIL:
        return res

    # 7. phi, on the UNPRUNED frame ------------------------------------------
    phi = sd.build_phi(df, nid=nid, input_units=input_units, self_check=True)
    res["f_implied"] = float(sd.cell_f_implied_from_phi(phi))
    flit = sd.cell_f_beyond_cutoff(phi, cutoff_um=60.0, by="d_from_um")
    res["F_lit"] = float(flit["F"])
    res["A_shaft_um2"] = float(phi["shaft_area_um2"].sum())
    res["A_spine_um2"] = float(phi["spine_area_um2"].sum())
    res["n_branches"] = int(phi["branch_id"].nunique())

    # 7b. hand back the pre-prune labelled frame, if asked --------------------
    #     Taken HERE and not later: after this line prune_spines removes the
    #     spine geometry, and the anchors resolved downstream are defined
    #     against nodes that would no longer exist. Still raw nm; the caller's
    #     transform is applied by resolve_synapse_anchors, not here.
    if return_frames:
        res["frames"] = {"labelled": df.copy()}

    # 8. prune ---------------------------------------------------------------
    df_pruned, spine_info = prune_spines(df)
    res["spine_report"] = {k: v for k, v in spine_info.items()
                           if k != "spine_bases"}
    res["spine_report"]["n_spines"] = spine_info["n_spines"]

    # 9. domains (S1.4 stub) + sections --------------------------------------
    df_pruned = assign_domain(df_pruned, soma_id=soma_rep["soma_id"])
    if align_fn is not None:
        df_pruned = align_fn(df_pruned, nid)
        res["aligned"] = True
    else:
        res["aligned"] = False
    sections = decompose_sections(df_pruned)
    res["n_sections"] = len(sections)

    # 10. reconcile ----------------------------------------------------------
    seg_map, sec_table = build_section_map(sections, phi)
    base_tab = resolve_spine_base_sections(spine_info["spine_bases"], sections,
                                           df_pruned)
    res["n_sections_multi_branch"] = int((sec_table["n_branch_ids"] > 1).sum())
    res["n_spines_unattached"] = int((base_tab["section_id"] < 0).sum()) \
        if len(base_tab) else 0

    voc = nc.section_vocabulary(df_pruned, domain_column="domain")
    res["section_vocabulary"] = sorted(["%s:%s" % (c, d) for c, d in voc])

    # 11. write ---------------------------------------------------------------
    if write_files:
        os.makedirs(output_dir, exist_ok=True)
        hoc_path = os.path.join(output_dir, "neuron_%s_aligned.hoc" % nid)
        res["files"]["hoc"] = write_hoc(
            sections, df_pruned, hoc_path, input_units=input_units,
            header_comment="exporter_id: %s"
                           % exporter_id(label_fn, spine_length_threshold_nm))
        for name, frame in (("phi", phi), ("segment_map", seg_map),
                            ("section_table", sec_table),
                            ("spine_bases", base_tab), ("synapses", syn_frame)):
            p = os.path.join(output_dir, "neuron_%s_%s.csv" % (nid, name))
            frame.to_csv(p, index=False)
            res["files"][name] = p
        prov = {
            "exporter_id": exporter_id(label_fn, spine_length_threshold_nm),
            "record": {k: v for k, v in res.items()
                       if k not in ("files", "frames")},
            "files": {k: (v if isinstance(v, str) else v.get("path"))
                      for k, v in res["files"].items()},
        }
        pp = os.path.join(output_dir, "neuron_%s_provenance.json" % nid)
        with open(pp, "w", newline="\n") as fh:
            fh.write(json.dumps(prov, indent=2, sort_keys=True, default=str))
        res["files"]["provenance"] = pp

    res["exporter_id"] = exporter_id(label_fn, spine_length_threshold_nm)
    if verbose:
        print("[OK] neuron %s: %d sections, %d spines pruned, "
              "f_implied %.3f, F_lit %.3f, qc=%s"
              % (nid, res["n_sections"], spine_info["n_spines"],
                 res["f_implied"], res["F_lit"], res["qc_status"]))
    return res


# --------------------------------------------------------------------------- #
# NEURON validation (I-13a, I-15, I-16, I-18 on the EMITTED file)             #
# --------------------------------------------------------------------------- #
def validate_hoc(path, min_soma_diam_um=4.0, expect_soma_root=True):
    """Load the emitted .hoc in NEURON and assert the S1 exit invariants.

    Requires the 'neuron' package. Returns a JSON-safe report; 'ok' is the
    conjunction of the invariants that could be evaluated.
    """
    rep = {"path": str(path), "neuron_available": False, "ok": False,
           "violations": []}
    try:
        from neuron import h
    except Exception as e:                     # noqa: BLE001
        rep["import_error"] = repr(e)
        return rep
    rep["neuron_available"] = True

    h.load_file("stdlib.hoc")
    h.load_file(str(path))
    secs = list(h.allsec())
    rep["n_sections"] = len(secs)

    arrays = defaultdict(int)
    for s in secs:
        arrays[s.name().split("[")[0]] += 1
    rep["arrays"] = dict(arrays)

    # I-16: no synapse label anywhere in the vocabulary
    try:
        nc.assert_no_synapse_leakage(list(arrays), "emitted .hoc vocabulary")
        rep["I16_no_synapse_sections"] = True
    except ValueError as e:
        rep["I16_no_synapse_sections"] = False
        rep["violations"].append(str(e))

    # I-13a: vocabulary is non-empty and within the admissible arrays
    admissible = set(ARRAY_ORDER)
    unknown = sorted(set(arrays) - admissible)
    rep["I13a_vocabulary_admissible"] = (len(arrays) > 0 and not unknown)
    if unknown:
        rep["violations"].append("unexpected section arrays: %r" % unknown)

    # I-15: exactly one somatic section, and it is the topology root
    somatic = [s for s in secs if s.name().startswith("soma")]
    roots = [s for s in secs if s.parentseg() is None]
    rep["n_somatic_sections"] = len(somatic)
    rep["root_sections"] = [s.name() for s in roots]
    rep["I15_single_soma_is_root"] = (
        len(somatic) == 1 and len(roots) == 1
        and (roots[0].name() == somatic[0].name() if expect_soma_root else True))
    if not rep["I15_single_soma_is_root"]:
        rep["violations"].append(
            "I-15: %d somatic sections, roots=%r"
            % (len(somatic), [s.name() for s in roots]))

    # I-18: the name-based and geometry-based somata agree in the EMITTED file
    diams = [(s.name(), max(seg.diam for seg in s)) for s in secs]
    biggest = max(diams, key=lambda t: t[1])
    rep["largest_diameter_section"] = [biggest[0], float(biggest[1])]
    name_ok = bool(somatic) and biggest[0] == somatic[0].name()
    geom_ok = biggest[1] >= float(min_soma_diam_um)
    rep["I18_name_geometry_agree"] = bool(name_ok and geom_ok)
    if not name_ok:
        rep["violations"].append(
            "I-18: largest-calibre section is %s, not the somatic section"
            % biggest[0])
    if not geom_ok:
        rep["violations"].append(
            "I-18: largest diameter %.2f um is below the %.2f um soma floor"
            % (biggest[1], float(min_soma_diam_um)))

    L = [float(s.L) for s in secs]
    rep["total_length_um"] = float(np.sum(L)) if L else 0.0
    rep["median_section_length_um"] = float(np.median(L)) if L else float("nan")
    rep["n_seg_total"] = int(sum(s.nseg for s in secs))
    rep["ok"] = (rep.get("I16_no_synapse_sections", False)
                 and rep.get("I13a_vocabulary_admissible", False)
                 and rep.get("I15_single_soma_is_root", False)
                 and rep.get("I18_name_geometry_agree", False))
    return rep
