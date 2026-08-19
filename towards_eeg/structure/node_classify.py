"""node_classify -- stage S1.1: compartment class, and only compartment class.

WHAT THIS MODULE ESTABLISHES
----------------------------
The compartment class of a skeleton node is a function of its structural
annotation ALONE. The synapse label is a per-synapse attribute; it belongs to
contract C-09 and it never touches a section name. This is the fix for defect
O8, in which get_hoc_type (morpholgy_pathways__6_.py L842, L864-874) tested
'synapse_label' FIRST and returned, destroying the compartment class of every
synapse-bearing node -- 25-50 percent of sections in the sampled bank, and in
one file the soma itself.

The fix is structural, not a reordering: classify_frame() is not given the
synapse label at all. There is no code path by which it can leak.

MEASURED SOURCE VOCABULARY
--------------------------
The H01 skeleton CSVs carry 'annotated_type' in Capitalised form, with five
values observed over the two files inspected on 23 July 2026:

    Soma, Dendrite, Axon, AIS, Astrocyte

Two of these were silently wrong under the old exporter, because it tested
'axon' in s / 'soma' in s on the lowercased string and fell through to 'dend':

    AIS       -> exported as dendrite  (74 nodes in neuron_794820508)
    Astrocyte -> exported as dendrite  (347 nodes in neuron_606394351, 5.4%)

AIS is now its own class (kept, so it can be removed downstream by name).
Astrocyte is a segmentation merge error and is resolved by
resolve_mislabelled_nodes().

SINGLE SOURCE OF TRUTH FOR 'DENDRITE'
-------------------------------------
The dendrite rule is spine_density.SHAFT_REGEX, imported, NOT re-typed here.
phi's shaft set and the exporter's dend set must be the same set of nodes or
phi's branch indices and the emitted sections describe different objects. If
the source vocabulary ever grows, SHAFT_REGEX is the one place to change it.
Likewise the spine labels are spine_density.SPINE_LABELS. This is deliberate
O7 hygiene: no constant defined twice.

CONTRACT DEVIATION -- FLAGGED, NOT SILENT
-----------------------------------------
C8.1 admits exactly four (class, domain) pairs and does not include AIS.
Emitting 'ais[k]' is a deviation, made because the AIS must be identifiable to
be removable. Either ratify (ais, none) in Doc 2 rev 4, or fold AIS into
axon[] and carry a separate index. Do not leave the contract and the artefact
disagreeing. See also D-S1.3-e, the same situation for C8.2.

ASCII only, LF only, no top-level side effects, no I/O.
"""

import re
from collections import defaultdict

import numpy as np
import pandas as pd

import spine_density as sd


MODULE_VERSION = "node_classify-1.0.0"

# --- compartment classes ---------------------------------------------------
CLS_SOMA = "soma"
CLS_AXON = "axon"
CLS_AIS = "ais"
CLS_DEND = "dend"
CLS_SPINE = "spine"
CLS_GLIA = "glia"
CLS_UNKNOWN = "unknown"

COMPARTMENT_CLASSES = (CLS_SOMA, CLS_AXON, CLS_AIS, CLS_DEND,
                       CLS_SPINE, CLS_GLIA, CLS_UNKNOWN)

# --- domains: apical/basal split is RETIRED (decision: S3 dropped entirely,
#     every dendrite is 'dend'; D-4's geometric rule below will not run).
#     DOM_APICAL / DOM_BASAL are kept as vocabulary, not deleted, because
#     SECTION_ARRAY and section_array_name still need them to be complete
#     lookup tables (see test N5). assert_domain_collapsed, below the class
#     tables, is the runtime guard that they are never actually ASSIGNED. ---
DOM_NONE = "none"
DOM_APICAL = "apical"
DOM_BASAL = "basal"

# --- the synapse label set. NEVER a compartment class, NEVER a section name.
SIGMA_SYN = ("exc_syn", "inh_syn", "unknown_syn")

# --- classification rules, FIRST MATCH WINS. Order is load-bearing:
#     AIS must be tested before AXON, or 'axon initial segment' would be
#     classified as axon. Astrocyte before dendrite is not required but is
#     kept explicit. The dendrite pattern is imported, not re-typed.
CLASS_RULES = (
    (r"\bais\b|axon[ _-]?initial", CLS_AIS),
    (r"soma|cell[ _-]?body", CLS_SOMA),
    (r"axon", CLS_AXON),
    (r"astrocyte|astro|glia", CLS_GLIA),
    (sd.SHAFT_REGEX, CLS_DEND),
)

_COMPILED_RULES = tuple((re.compile(p, re.IGNORECASE), c) for p, c in CLASS_RULES)
_SPINE_SET = frozenset(s.lower() for s in sd.SPINE_LABELS)

# --- canonical annotation token per class.
#     When a node is RELABELLED, both 'compartment_class' AND 'annotated_type'
#     must move together. spine_density.build_phi decides shaft membership by
#     matching SHAFT_REGEX against 'annotated_type', NOT against the class
#     column, so rewriting only the class produces a node that the exporter
#     emits as a dendrite section and that phi has never seen -- a silent
#     divergence between the two partitions, which is exactly the failure mode
#     Doc 14 section 7 point 1 exists to prevent. Observed on neuron_794820508:
#     one Astrocyte node relabelled to dend produced 1760 dend sections against
#     1759 phi branches. Every token below is chosen to match the rule that
#     classifies it, so classify_node(CANONICAL_ANNOTATION[c]) == c.
CANONICAL_ANNOTATION = {
    CLS_SOMA: "Soma",
    CLS_AXON: "Axon",
    CLS_AIS: "AIS",
    CLS_DEND: "Dendrite",
    CLS_GLIA: "Astrocyte",
    CLS_SPINE: sd.SPINE_LABELS[0],
}

# --- how a reference node's class is COUNTED when it votes.
#     A spine is not a separate structure from the dendrite; it is dendritic
#     membrane. So a spine node next to a query point is perfectly good evidence
#     that the query point sits in dendritic territory, and excluding spine
#     nodes from the reference set would throw away most of the spatial
#     information in a spiny cell (37,586 spine nodes against 25,439 shaft nodes
#     on neuron_794820508). Their votes are therefore COUNTED AS 'dend'.
#
#     What must never happen is 'spine' WINNING, for two reasons. Spine
#     membership is decided topologically by the labeller -- a terminal subtree
#     below the length threshold -- not spatially, so a spatial vote is not
#     evidence about it. And a mid-cable node voted into 'spine' would then be
#     removed by prune_spines together with the entire subtree beneath it,
#     silently deleting dendrite. Folding the vote into 'dend' keeps the
#     evidence and removes the failure mode in one step.
VOTE_CLASS_ALIAS = {CLS_SPINE: CLS_DEND}

# --- (class, domain) -> NEURON section-array name, per C8.1 plus the AIS
#     deviation and the provisional pre-S1.4 (dend, none).
SECTION_ARRAY = {
    (CLS_SOMA, DOM_NONE): "soma",
    (CLS_AXON, DOM_NONE): "axon",
    (CLS_AIS, DOM_NONE): "ais",
    (CLS_DEND, DOM_APICAL): "apic_dend",
    (CLS_DEND, DOM_BASAL): "basal_dend",
    (CLS_DEND, DOM_NONE): "dend",
}


# --------------------------------------------------------------------------- #
# Classification                                                              #
# --------------------------------------------------------------------------- #
def classify_node(annotated_type):
    """Compartment class of one node, from its structural annotation only.

    Parameters
    ----------
    annotated_type : str
        The 'annotated_type' field of the H01 skeleton CSV, or the value the
        spine labeller rewrote it to ('head' / 'neck' / 'spine').

    Returns
    -------
    str, one of COMPARTMENT_CLASSES.

    Notes
    -----
    Total function: an unrecognised annotation returns CLS_UNKNOWN rather than
    falling through to CLS_DEND. Silent fall-through to dendrite is exactly how
    AIS and Astrocyte nodes became dendrite in the inherited bank.
    """
    s = str(annotated_type)
    if s.lower() in _SPINE_SET:
        return CLS_SPINE
    for rx, cls in _COMPILED_RULES:
        if rx.search(s):
            return cls
    return CLS_UNKNOWN


def classify_frame(df, annotation_column="annotated_type",
                   out_column="compartment_class"):
    """Add a compartment_class column to a skeleton frame.

    The synapse label is NOT an argument and is NOT read. That is the O8 fix:
    the leak is closed by the call signature, not by rule ordering.

    Returns a COPY; the input frame is never mutated.
    """
    if annotation_column not in df.columns:
        raise ValueError("frame has no %r column" % annotation_column)
    out = df.copy()
    # unique-value mapping: the frames are 6k-63k rows and the vocabulary is
    # ~5 values, so classify each distinct annotation once.
    uniq = out[annotation_column].astype(str).unique()
    lut = {u: classify_node(u) for u in uniq}
    out[out_column] = out[annotation_column].astype(str).map(lut)
    return out


def class_counts(df, out_column="compartment_class"):
    """Counts per compartment class, as a plain dict (JSON-safe)."""
    if out_column not in df.columns:
        raise ValueError("frame has no %r column; run classify_frame first"
                         % out_column)
    vc = df[out_column].value_counts()
    return {str(k): int(v) for k, v in vc.items()}


# --------------------------------------------------------------------------- #
# Invariants                                                                  #
# --------------------------------------------------------------------------- #
def assert_no_synapse_leakage(values, what="compartment_class"):
    """I-16 at the node level: no synapse label may appear as a class.

    Parameters
    ----------
    values : iterable of str
        Compartment classes, or section-array names, or a section vocabulary.

    Raises
    ------
    ValueError if any value is in SIGMA_SYN or contains '_syn'.
    """
    bad = sorted({str(v) for v in values
                  if str(v) in SIGMA_SYN or "_syn" in str(v)})
    if bad:
        raise ValueError(
            "I-16 violation: synapse label(s) %r appear in %s" % (bad, what))
    return True


# --- The retirement guard for the apical/basal split ------------------------
#     RETIRED_DOMAIN_VALUES covers BOTH value spaces the split could leak
#     into: a node's domain label (DOM_APICAL / DOM_BASAL) and a section's
#     array name (apic_dend / basal_dend). One frozenset, one function, called
#     from both places, so there is exactly one definition of "the split is
#     off" rather than two that could drift apart.
RETIRED_DOMAIN_ARRAYS = frozenset({"apic_dend", "basal_dend"})
RETIRED_DOMAIN_VALUES = frozenset({DOM_APICAL, DOM_BASAL}) | RETIRED_DOMAIN_ARRAYS


def assert_domain_collapsed(values, what="domain"):
    """Raise if apical/basal domain identification ever actually ran.

    Parameters
    ----------
    values : iterable of str
        Either domain labels (checked against DOM_APICAL / DOM_BASAL) or
        section-array names (checked against apic_dend / basal_dend).
    what : str
        Named in the error message, so a failure identifies WHERE the split
        leaked back in, not only that it did.

    Raises
    ------
    ValueError if any value in RETIRED_DOMAIN_VALUES is present.

    Notes
    -----
    This does NOT touch SECTION_ARRAY or section_array_name: those remain a
    complete, working table on request (test N5 requires
    section_array_name(CLS_DEND, DOM_APICAL) == 'apic_dend' to keep working).
    This function guards USE, not DEFINITION. It is called from two places
    only: assign_domain's own output in morphology_exporter.py (the source --
    catches a domain column that stops being all-DOM_NONE) and write_hoc's
    emitted array names (the sink -- catches ANY route to an apic_dend or
    basal_dend section reaching the .hoc file, independent of how the domain
    column upstream was produced).
    """
    bad = sorted({str(v) for v in values if str(v) in RETIRED_DOMAIN_VALUES})
    if bad:
        raise ValueError(
            "domain identification is retired (S3 dropped; decision D-4 will "
            "not run): %r appeared in %s. apic_dend/basal_dend are reserved, "
            "never-populated vocabulary -- see node_classify.SECTION_ARRAY."
            % (bad, what))
    return True


def section_vocabulary(df, class_column="compartment_class",
                       domain_column=None):
    """varsigma(nu): the set of (class, domain) pairs present in one cell.

    domain_column may be None before S1.4, in which case every domain is
    DOM_NONE. Spine-classified nodes are excluded: spines are pruned, so they
    never contribute a section (D-7).
    """
    if class_column not in df.columns:
        raise ValueError("frame has no %r column" % class_column)
    cls = df[class_column].astype(str)
    keep = cls != CLS_SPINE
    if domain_column is None:
        dom = pd.Series([DOM_NONE] * len(df), index=df.index)
    else:
        dom = df[domain_column].astype(str)
    pairs = frozenset(zip(cls[keep].tolist(), dom[keep].tolist()))
    assert_no_synapse_leakage([c for c, _ in pairs], "section vocabulary")
    return pairs


def section_array_name(cls, domain=DOM_NONE):
    """(class, domain) -> NEURON section-array name. Raises on an unmapped pair."""
    key = (str(cls), str(domain))
    if key not in SECTION_ARRAY:
        raise ValueError("no section array defined for %r; admissible: %r"
                         % (key, sorted(SECTION_ARRAY)))
    return SECTION_ARRAY[key]


# --------------------------------------------------------------------------- #
# C-09 side: the synapse label lives HERE, in its own frame                   #
# --------------------------------------------------------------------------- #
def extract_synapse_frame(df, nid=None, synapse_column="synapse_label",
                          input_units="nm"):
    """Pull the synapse annotation out of the skeleton into a C-09-bound frame.

    One row per synapse-bearing NODE (not per synapse -- C-09 is per synapse and
    is completed at S1.6, when spine bases and lfpy_idx are resolved). This
    function exists so that the exporter can strip 'synapse_label' from the
    geometry path entirely.

    Returns a frame with columns: nid, node_id, x, y, z (um), synapse_label.
    Returns an empty frame with those columns if the input has no synapse
    column or no labelled rows.
    """
    cols = ["nid", "node_id", "x", "y", "z", "synapse_label"]
    if synapse_column not in df.columns:
        return pd.DataFrame(columns=cols)
    m = df[synapse_column].notna() & (df[synapse_column].astype(str) != "")
    m &= df[synapse_column].astype(str).str.lower() != "none"
    sub = df[m]
    if len(sub) == 0:
        return pd.DataFrame(columns=cols)
    scale = 1000.0 if input_units == "nm" else 1.0
    out = pd.DataFrame({
        "nid": nid,
        "node_id": sub["id"].values,
        "x": sub["x"].values / scale,
        "y": sub["y"].values / scale,
        "z": sub["z"].values / scale,
        "synapse_label": sub[synapse_column].astype(str).values,
    })
    return out[cols]


def strip_synapse_column(df, synapse_column="synapse_label"):
    """Return a copy of df with the synapse column removed.

    Called by the exporter immediately after extract_synapse_frame, so that the
    geometry path physically cannot see the label. Belt and braces alongside
    classify_frame's signature.
    """
    if synapse_column in df.columns:
        return df.drop(columns=[synapse_column])
    return df.copy()


# --------------------------------------------------------------------------- #
# Mislabelled-node resolution (Astrocyte and friends)                         #
# --------------------------------------------------------------------------- #
def _offending_subtrees(df, offending_ids):
    """Group offending node ids into maximal parent/child-connected components.

    Returns a list of lists of node ids. Two offending nodes are in the same
    component iff one is the parent of the other.
    """
    off = set(offending_ids)
    par = dict(zip(df["id"].tolist(), df["p"].tolist()))
    children = defaultdict(list)
    for nid_, p in par.items():
        if p in off and nid_ in off:
            children[p].append(nid_)
    roots = [n for n in off if par.get(n) not in off]
    comps = []
    for r in roots:
        comp, stack = [], [r]
        while stack:
            cur = stack.pop()
            comp.append(cur)
            stack.extend(children.get(cur, ()))
        comps.append(comp)
    return comps


def resolve_mislabelled_nodes(df,
                              offending_classes=(CLS_GLIA, CLS_UNKNOWN),
                              policy="knn_relabel",
                              k=5,
                              max_distance_nm=None,
                              unresolved_action="relabel",
                              class_column="compartment_class",
                              annotation_column="annotated_type",
                              rewrite_annotation=True,
                              vote_class_alias=None):
    """Reassign nodes whose annotation is a known mislabel.

    Astrocyte nodes in an H01 neuron skeleton are segmentation merge errors or
    nearest-annotation lookup errors. Measured structure (two cells, 23 July
    2026): they form MAXIMAL PURE SUBTREES rooted directly on the soma -- they
    never have a dendrite parent and never have a dendrite child. A purely
    topological vote is therefore degenerate (the nearest non-offending ancestor
    is always the soma), which is why the vote is spatial.

    The vote is taken at SUBTREE granularity, not node granularity: a 347-node
    astrocyte process is one object and should get one decision, not 347
    independent ones.

    Parameters
    ----------
    offending_classes : tuple of str
        Classes to reassign. Default resolves glia and unrecognised annotations.
    policy : {'knn_relabel', 'drop', 'keep'}
        'knn_relabel'  reassign to the majority class of the k nearest
                       non-offending nodes of the SAME cell (inverse-distance
                       weighted).
        'drop'         remove the offending subtrees and everything below them.
        'keep'         leave them; classify only, do not act.
    k : int
        Neighbours per offending node in the vote.
    max_distance_nm : float or None
        Guard. If set, a subtree whose MEDIAN nearest-neighbour distance exceeds
        this is not confidently adjacent to any part of the cell, and is handled
        by unresolved_action instead of being relabelled. None disables the
        guard (every subtree is relabelled regardless of distance).
    unresolved_action : {'relabel', 'drop', 'keep'}
        What to do with subtrees failing the guard.
    rewrite_annotation : bool
        If True (default) a relabelled node also has its annotation rewritten to
        CANONICAL_ANNOTATION[new_class], so that consumers keyed on
        'annotated_type' -- spine_density.build_phi in particular -- agree with
        consumers keyed on the class column. Setting this False reintroduces the
        divergence described next to CANONICAL_ANNOTATION and is provided only
        for diagnosing it.
    vote_class_alias : dict or None
        Applied when TALLYING a reference node's vote, so that a class can
        contribute evidence under a different name. None uses VOTE_CLASS_ALIAS,
        which counts spine nodes as dendrite for the reasons given there. Pass
        an empty dict to tally every class under its own name.

    Returns
    -------
    (df_out, report)
        df_out : copy with class_column updated and, under 'drop', rows removed
        report : JSON-safe dict. Per subtree it records n_nodes, the median and
                 maximum nearest-neighbour distance in nm, the weighted vote,
                 the decision and the assigned class. READ THE DISTANCES: a
                 subtree voting 'axon' from 6 um away is not a measurement.
    """
    if policy not in ("knn_relabel", "drop", "keep"):
        raise ValueError("policy must be knn_relabel, drop or keep")
    if unresolved_action not in ("relabel", "drop", "keep"):
        raise ValueError("unresolved_action must be relabel, drop or keep")
    if class_column not in df.columns:
        raise ValueError("frame has no %r column; run classify_frame first"
                         % class_column)

    out = df.copy()
    cls = out[class_column].astype(str)
    off_mask = cls.isin(list(offending_classes))
    report = {
        "module_version": MODULE_VERSION,
        "policy": policy,
        "k": int(k),
        "max_distance_nm": (None if max_distance_nm is None
                            else float(max_distance_nm)),
        "unresolved_action": unresolved_action,
        "offending_classes": list(offending_classes),
        "n_offending_nodes": int(off_mask.sum()),
        "n_subtrees": 0,
        "subtrees": [],
        "n_relabelled": 0,
        "n_dropped": 0,
    }
    if not off_mask.any() or policy == "keep":
        return out, report

    off_ids = out.loc[off_mask, "id"].tolist()
    comps = _offending_subtrees(out, off_ids)
    report["n_subtrees"] = len(comps)

    alias = VOTE_CLASS_ALIAS if vote_class_alias is None else dict(vote_class_alias)
    ref = out.loc[~off_mask]
    report["vote_class_alias"] = {str(a): str(b) for a, b in alias.items()}
    have_ref = len(ref) > 0
    if have_ref:
        from scipy.spatial import cKDTree
        ref_xyz = ref[["x", "y", "z"]].values.astype(float)
        ref_cls = ref[class_column].astype(str).values
        tree = cKDTree(ref_xyz)
        k_eff = int(min(k, len(ref)))

    id_to_pos = {v: i for i, v in enumerate(out["id"].tolist())}
    drop_ids = set()

    for comp in comps:
        rec = {"n_nodes": len(comp),
               "original_class": str(cls.iloc[id_to_pos[comp[0]]])}
        if not have_ref:
            rec.update({"decision": "no_reference_nodes", "assigned_class": None})
            report["subtrees"].append(rec)
            continue

        q = out.iloc[[id_to_pos[c] for c in comp]][["x", "y", "z"]].values
        q = q.astype(float)
        dd, ii = tree.query(q, k=k_eff)
        dd = np.atleast_2d(dd.reshape(len(comp), k_eff))
        ii = np.atleast_2d(ii.reshape(len(comp), k_eff))

        weights = defaultdict(float)
        for row_d, row_i in zip(dd, ii):
            for dist, idx in zip(row_d, row_i):
                c = ref_cls[idx]
                weights[alias.get(c, c)] += 1.0 / (float(dist) + 1.0)
        vote = max(weights.items(), key=lambda kv: kv[1])[0]
        nn0 = dd[:, 0]
        rec["median_nn_distance_nm"] = float(np.median(nn0))
        rec["max_nn_distance_nm"] = float(np.max(nn0))
        rec["vote"] = {str(a): round(float(b), 6) for a, b in weights.items()}

        guarded = (max_distance_nm is not None
                   and rec["median_nn_distance_nm"] > float(max_distance_nm))
        action = unresolved_action if guarded else (
            "relabel" if policy == "knn_relabel" else "drop")
        rec["guard_failed"] = bool(guarded)

        if action == "relabel":
            cls_pos = out.columns.get_loc(class_column)
            ann_pos = (out.columns.get_loc(annotation_column)
                       if (rewrite_annotation
                           and annotation_column in out.columns) else None)
            token = CANONICAL_ANNOTATION.get(vote)
            for c in comp:
                out.iat[id_to_pos[c], cls_pos] = vote
                if ann_pos is not None and token is not None:
                    out.iat[id_to_pos[c], ann_pos] = token
            report["n_relabelled"] += len(comp)
            rec["decision"] = "relabelled"
            rec["assigned_class"] = str(vote)
            rec["assigned_annotation"] = (token if ann_pos is not None else None)
        elif action == "drop":
            drop_ids.update(comp)
            rec["decision"] = "dropped"
            rec["assigned_class"] = None
        else:
            rec["decision"] = "kept"
            rec["assigned_class"] = None
        report["subtrees"].append(rec)

    if drop_ids:
        # drop the offending subtrees and any descendants left orphaned
        keep = ~out["id"].isin(drop_ids)
        out = out.loc[keep].copy()
        alive = set(out["id"].tolist())
        changed = True
        while changed:
            orphan = out["id"].isin(alive) & (~out["p"].isin(alive)) & (out["p"] != -1)
            if not orphan.any():
                changed = False
            else:
                out = out.loc[~orphan].copy()
                alive = set(out["id"].tolist())
        report["n_dropped"] = int(len(df) - len(out))

    return out, report
