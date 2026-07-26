"""Stage S1 alignment: coordinate frame and synapse index.

This is the merged module specified by TEEG_16 section 3. It joins three things
that were previously separate or wrong:

  1. `Alignment.py`'s alignment mathematics, adopted VERBATIM (section 3.1)
  2. our S1.0-S1.3 exporter, replacing `Alignment.py`'s L104 exporter entirely
     (section 3.2), and
  3. the base-anchor synapse redirect that spine pruning makes mandatory
     (section 3.3), measured at 14.1% foreign-branch misplacement on
     neuron 794820508.

Pipeline order, settled 25 July 2026 and load-bearing: PRUNE IN RAW nm SPACE,
ALIGN LAST. Alignment is a rigid transform, so every area, count and phi value
must come back bit-identical afterwards -- `regression_check` below asserts it.

Coordinate discipline
---------------------
Everything ENTERING this module is raw nanometres. There is EXACTLY ONE
nm -> um conversion in the whole pipeline, performed by
`morphology_exporter.write_hoc(input_units='nm')`. `Alignment.py` divided by
1000 in the aligner (L257-259) AND again in its exporter (L116); the adapter
here deliberately stays in nm so that cannot recur (section 3.1).

`aligned_um()` is the single exception: it produces aligned MICROMETRES, and is
used only for quantities that must live in the same frame as the .hoc -- the
synapse coordinates and their anchors. It applies the identical `soma_pos` and
`mean_matrix` the skeleton went through, which is the coupling section 3.1
calls out as the thing that guarantees one frame.

What is NOT here
----------------
No Drive paths, no plotting, no notebook globals. The LFPy cell is injected via
`cell_factory` so the redirect and the emitter are testable with no NEURON
installed. Plotting belongs behind the `plot_fn` hook `soma_enforce` already
defines.
"""

from __future__ import annotations

import ast
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

import node_classify as nc
import morphology_exporter as mx


MODULE_VERSION = "alignment-1.0.0"

# Eyal et al. 2016 (Table 1, n=6) for human L2/3 pyramidal cells. cm is the
# INTRINSIC, shaft-referenced value: contract C-14 `cm_reference` requires that
# F be applied at model-build time (invariant I-17) and never folded in here.
# Eyal state that Ra has negligible impact on their RMSD and "should be viewed
# with some caution"; it is nonetheless one of the two parameters that fix the
# lfpy_idx index space, which is why it is a required argument below and is
# stamped into provenance rather than defaulted silently.
CM_EYAL_L23 = 0.50            # uF/cm2, shaft-referenced
RA_EYAL_L23 = 268.5           # ohm cm, mean of the six fitted cells

# Yao et al. 2021 / Guet-McCreight et al. 2023, all interneuron subtypes.
# Interneurons are largely aspiny, so F = 1.0 and no spine correction applies.
RA_YAO_INTERNEURON = 100.0
CM_YAO_SST = 1.00
CM_YAO_PV = 2.00
CM_YAO_VIP = 2.00


# --------------------------------------------------------------------------- #
#  1. The reference bank                                                       #
# --------------------------------------------------------------------------- #
def load_alignment_metadata(path):
    """Load a per-(layer, subpopulation) reference bank.

    Schema, from Alignment.py L219-222: soma_x, soma_y, soma_z,
    rotation_matrix (a stringified 3x3 list), v_com_x/y/z. The delivered banks
    also carry FA_2D and angle_from_mean.

    The FA gate that produced the bank belongs to metadata EXTRACTION, not to
    alignment; nothing here re-filters on it.
    """
    df = pd.read_csv(path)
    for col in ("soma_x", "soma_y", "soma_z", "rotation_matrix"):
        if col not in df.columns:
            raise ValueError("metadata %r lacks required column %r" % (path, col))
    if isinstance(df["rotation_matrix"].iloc[0], str):
        df["rotation_matrix"] = df["rotation_matrix"].apply(ast.literal_eval)

    M = np.array(df["rotation_matrix"].tolist(), dtype=float)
    if M.ndim != 3 or M.shape[1:] != (3, 3):
        raise ValueError("rotation_matrix column is not Nx3x3, got %r" % (M.shape,))
    bad = np.abs(np.linalg.det(M) - 1.0) > 1e-6
    if bad.any():
        raise ValueError("%d reference rotations are not proper rotations "
                         "(det != 1), first at row %d"
                         % (int(bad.sum()), int(np.argmax(bad))))
    return df


def soma_position_nm(df_raw):
    """Alignment.py L235-241. The soma is the topological root, in nm."""
    root_rows = df_raw[df_raw["p"] == -1]
    if root_rows.empty:
        raise ValueError("no root node (p == -1) in the raw skeleton")
    return root_rows.iloc[0][["x", "y", "z"]].to_numpy(dtype=float)


def neighbourhood_rotation(soma_pos, metadata_df, k_neighbors=3):
    """Alignment.py L244-251, verbatim, with diagnostics added.

    LOCAL alignment: the k spatially nearest references inside the cell's own
    (layer, subpopulation) bank, averaged on the rotation group. Settled 25 July
    2026 -- k=3 local, not a whole-bank mean.

    `R.from_matrix(...).mean()` is a true rotation-group mean via scipy, not an
    elementwise average (which would not be a rotation). Do not "simplify" it.

    Returns (mean_matrix, diagnostics). The diagnostics are recorded, never
    used as a gate: the angular spread among the chosen references is a
    property of the bank and belongs in provenance so a downstream reader can
    judge how local the frame really was.
    """
    soma_pos = np.asarray(soma_pos, dtype=float)
    reference_somas = metadata_df[["soma_x", "soma_y", "soma_z"]].to_numpy(float)

    distances = np.linalg.norm(reference_somas - soma_pos, axis=1)
    k_actual = int(min(k_neighbors, len(metadata_df)))
    if k_actual < 1:
        raise ValueError("empty metadata bank")
    nearest = np.argsort(distances)[:k_actual]

    matrices = np.array(metadata_df.iloc[nearest]["rotation_matrix"].tolist(),
                        dtype=float)
    rot = R.from_matrix(matrices)
    mean_matrix = rot.mean().as_matrix()

    det = float(np.linalg.det(mean_matrix))
    orth = float(np.abs(mean_matrix @ mean_matrix.T - np.eye(3)).max())
    if abs(det - 1.0) > 1e-9 or orth > 1e-9:
        raise ValueError("rotation-group mean is not a rotation: "
                         "det=%.12f, orthonormality error=%.3e" % (det, orth))

    pair = np.degrees([(R.from_matrix(matrices[i]).inv()
                        * R.from_matrix(matrices[j])).magnitude()
                       for i in range(k_actual) for j in range(i + 1, k_actual)])
    dev = np.degrees((rot.mean().inv() * rot).magnitude())

    diag = {
        "k_requested": int(k_neighbors),
        "k_actual": k_actual,
        "neighbour_row_indices": [int(i) for i in nearest],
        "neighbour_distance_um": [float(d) for d in distances[nearest] / 1000.0],
        "pairwise_angle_deg_max": float(pair.max()) if len(pair) else 0.0,
        "angle_from_mean_deg_max": float(np.max(dev)),
        "det": det,
        "orthonormality_error": orth,
    }
    if "neuron_id" in metadata_df.columns:
        diag["neighbour_nids"] = [int(v) for v in
                                  metadata_df.iloc[nearest]["neuron_id"].tolist()]
    return mean_matrix, diag


# --------------------------------------------------------------------------- #
#  2. The transform, in both the forms the pipeline needs                      #
# --------------------------------------------------------------------------- #
def aligned_nm(coords_nm, soma_pos, mean_matrix):
    """Centre on the soma and rotate. STAYS IN NANOMETRES.

    Alignment.py L242 + L254: centered = coords - soma_pos;
    rotated = centered @ mean_matrix.T.
    """
    coords_nm = np.asarray(coords_nm, dtype=float)
    if coords_nm.ndim == 1:
        coords_nm = coords_nm[None, :]
    return (coords_nm - np.asarray(soma_pos, dtype=float)) @ np.asarray(
        mean_matrix, dtype=float).T


def aligned_um(coords_nm, soma_pos, mean_matrix):
    """Same transform, then the single nm -> um conversion, for quantities that
    must land in the .hoc frame (synapses and their anchors). Alignment.py
    L50-59 applies exactly this to synapses, reusing the identical soma_pos and
    mean_matrix passed down from the aligner -- that coupling is what puts
    skeleton and synapses in one frame. It is preserved here by construction:
    both callers take the same two arguments.
    """
    return aligned_nm(coords_nm, soma_pos, mean_matrix) / 1000.0


def make_align_fn(soma_pos, mean_matrix):
    """Build the `align_fn(df, nid)` hook that morphology_exporter.export_neuron
    applies at step 9, AFTER pruning and AFTER phi. Returns a frame still in nm:
    write_hoc(input_units='nm') performs the one conversion.
    """
    def align_fn(df, nid=None):
        out = df.copy()
        rotated = aligned_nm(out[["x", "y", "z"]].to_numpy(float),
                             soma_pos, mean_matrix)
        out["x"], out["y"], out["z"] = rotated[:, 0], rotated[:, 1], rotated[:, 2]
        return out
    return align_fn


# --------------------------------------------------------------------------- #
#  3. The redirect (section 3.3)                                               #
# --------------------------------------------------------------------------- #
def spine_node_to_base(df_labelled, class_column="compartment_class"):
    """node_id -> (spine_root_id, base_node_id) for every spine node.

    A spine is a maximal parent/child-connected component of spine-class nodes
    and its base is the parent of its root -- the same definition
    morphology_exporter.prune_spines uses, computed on the PRE-PRUNE frame so
    the mapping exists before the nodes are removed.
    """
    if class_column not in df_labelled.columns:
        raise ValueError("frame has no %r column; run classify_frame first"
                         % class_column)
    ids = df_labelled["id"].to_numpy()
    par = dict(zip(ids.tolist(), df_labelled["p"].to_numpy().tolist()))
    spine = set(df_labelled.loc[
        df_labelled[class_column].astype(str) == nc.CLS_SPINE, "id"].tolist())

    out = {}
    for nid in spine:
        chain, cur = [], int(nid)
        while cur in spine:
            chain.append(cur)
            nxt = int(par.get(cur, -1))
            if nxt == cur:
                raise ValueError("self-parent at node %d" % cur)
            cur = nxt
            if len(chain) > len(ids):
                raise ValueError("cycle walking up from node %d" % nid)
        out[int(nid)] = (chain[-1], cur)
    return out


def resolve_synapse_anchors(syn_df, df_labelled, soma_pos, mean_matrix,
                            node_id_column="node_id",
                            xyz_columns=("syn_x_nm", "syn_y_nm", "syn_z_nm"),
                            class_column="compartment_class"):
    """Attach a base-shaft anchor to every synapse that sits on a spine.

    Section 3.3: for every synapse whose node is a spine node or a descendant of
    one, record the base shaft node, carry that node's RAW coordinate through
    the identical alignment transform, and snap from the anchor rather than
    from the head coordinate.

    The anchor is a COORDINATE, not a `dend[k]` section string, for the three
    reasons in section 3.3: get_closest_idx takes coordinates; a section is
    subdivided by lambda_f so a section index alone does not identify a
    compartment; and section indices are not stable across S1.4 re-decomposition
    whereas a physical location is. `spine_base_section` is still emitted, as a
    non-authoritative audit field, because contract C-09 rev 3 declares it.

    Returns a copy of `syn_df` with on_pruned_spine, base_node_id,
    spine_root_id, x/y/z (aligned um) and anchor_x/y/z (aligned um).
    """
    cx, cy, cz = xyz_columns
    for col in (cx, cy, cz, node_id_column):
        if col not in syn_df.columns:
            raise ValueError("syn_df lacks required column %r" % col)

    base_map = spine_node_to_base(df_labelled, class_column=class_column)
    node_xyz = {int(r.id): (float(r.x), float(r.y), float(r.z))
                for r in df_labelled.itertuples(index=False)}

    out = syn_df.copy().reset_index(drop=True)
    syn_nm = out[[cx, cy, cz]].to_numpy(dtype=float)
    anchor_nm = syn_nm.copy()

    on_spine = np.zeros(len(out), dtype=bool)
    base_ids = np.full(len(out), -1, dtype=np.int64)
    root_ids = np.full(len(out), -1, dtype=np.int64)
    n_unresolved = 0

    for k, nid in enumerate(out[node_id_column].to_numpy(dtype=np.int64).tolist()):
        hit = base_map.get(int(nid))
        if hit is None:
            continue                                  # shaft synapse, untouched
        root, base = hit
        if base < 0 or base not in node_xyz:
            n_unresolved += 1                         # spine reaching the root
            continue
        on_spine[k] = True
        root_ids[k] = root
        base_ids[k] = base
        anchor_nm[k] = node_xyz[base]

    syn_um = aligned_um(syn_nm, soma_pos, mean_matrix)
    anc_um = aligned_um(anchor_nm, soma_pos, mean_matrix)

    out["on_pruned_spine"] = on_spine
    out["spine_root_id"] = root_ids
    out["base_node_id"] = base_ids
    out["x"], out["y"], out["z"] = syn_um[:, 0], syn_um[:, 1], syn_um[:, 2]
    out["anchor_x"] = anc_um[:, 0]
    out["anchor_y"] = anc_um[:, 1]
    out["anchor_z"] = anc_um[:, 2]
    out.attrs["n_unresolved_spine_bases"] = int(n_unresolved)
    return out


DOWNSTREAM_TYPE = {"exc_syn": "exc", "inh_syn": "inh", "unknown_syn": "unknown"}


def to_downstream_type(label):
    """Mapper vocabulary -> the vocabulary fetch_mapped_synapse_indices filters
    on. That function raises ValueError on anything but 'exc' / 'inh', so
    emitting only the mapper's form silently breaks the population code
    (section 3.3). Both are emitted; this produces the downstream one.
    """
    s = str(label).lower()
    if s in DOWNSTREAM_TYPE:
        return DOWNSTREAM_TYPE[s]
    if any(k in s for k in ("2", "exc", "asymmetric")):
        return "exc"
    if any(k in s for k in ("1", "inh", "symmetric")):
        return "inh"
    return "unknown"


# --------------------------------------------------------------------------- #
#  4. The snapper                                                              #
# --------------------------------------------------------------------------- #
def default_cell_factory(hoc_path, cm, Ra, lambda_f=100.0, d_lambda=0.1,
                         nsegs_method="lambda_f", custom_fun=None,
                         custom_fun_args=None):
    """Instantiate the LFPy cell the snapper indexes against.

    `cm` and `Ra` are REQUIRED, not defaulted. Section 5, measured: the
    flattened lfpy_idx space is fixed by (cm, Ra, lambda_f, d_lambda,
    nsegs_method), and one standard deviation on Ra alone moves 81-99.9% of
    indices on a real cell. A mapped_synapses table without that stamp is not
    interpretable.

    In LFPy 2.3.7 `passive_parameters` carries only g_pas and e_pas; Ra and cm
    are separate TOP-LEVEL kwargs. Passing them inside passive_parameters is
    silently ignored (section 5.4). They are passed top-level here.
    """
    import LFPy                                        # deferred: no NEURON in tests
    kwargs = dict(morphology=hoc_path, passive=False, nsegs_method=nsegs_method,
                  lambda_f=lambda_f, d_lambda=d_lambda, Ra=float(Ra),
                  cm=float(cm), delete_sections=True)
    if custom_fun is not None:
        kwargs["custom_fun"] = custom_fun
        kwargs["custom_fun_args"] = custom_fun_args or [{}]
    return LFPy.Cell(**kwargs)


def snap_synapses(anchored_df, cell):
    """Snap each synapse to a flattened LFPy compartment index.

    A synapse flagged `on_pruned_spine` is snapped from its ANCHOR; every other
    synapse from its own coordinate. This is the only behavioural difference
    from Alignment.py L78, and it is the whole of D-7's fix.

    Also records `lfpy_idx_naive`, the index the unfixed free snap would have
    produced, so the defect magnitude stays auditable in the emitted artefact
    rather than only in a one-off study.
    """
    out = anchored_df.copy().reset_index(drop=True)
    idx, naive = [], []
    for r in out.itertuples(index=False):
        naive.append(int(cell.get_closest_idx(x=float(r.x), y=float(r.y),
                                              z=float(r.z))))
        if bool(r.on_pruned_spine):
            idx.append(int(cell.get_closest_idx(x=float(r.anchor_x),
                                                y=float(r.anchor_y),
                                                z=float(r.anchor_z))))
        else:
            idx.append(naive[-1])
    out["lfpy_idx"] = idx
    out["lfpy_idx_naive"] = naive
    out["redirected"] = out["lfpy_idx"] != out["lfpy_idx_naive"]
    return out


# --------------------------------------------------------------------------- #
#  5. The C-09 emitter                                                         #
# --------------------------------------------------------------------------- #
C09_COLUMNS = ["x", "y", "z", "synapse_label", "synapse_type", "lfpy_idx",
               "spine_base_section", "lambda_f", "nsegs_method",
               "cm", "Ra", "d_lambda",
               "on_pruned_spine", "anchor_x", "anchor_y", "anchor_z",
               "base_node_id", "lfpy_idx_naive", "redirected"]


def write_mapped_synapses(snapped_df, path, cm, Ra, lambda_f=100.0,
                          d_lambda=0.1, nsegs_method="lambda_f",
                          spine_base_section=None):
    """Emit `neuron_{nid}_mapped_synapses.csv`.

    Beyond contract C-09 rev 3's fields this stamps cm, Ra and d_lambda into
    every row. C-09 as written requires only lambda_f and nsegs_method
    (invariant I-13b), which section 5 shows is insufficient: two tables can
    satisfy I-13b exactly and still disagree on 99.9% of lfpy_idx. Flagged for
    Doc 2 rev 4; emitted here so the artefact is self-describing meanwhile.
    """
    out = snapped_df.copy()
    if "synapse_label" not in out.columns:
        out["synapse_label"] = "unknown_syn"
    out["synapse_type"] = out["synapse_label"].map(to_downstream_type)
    out["lambda_f"] = float(lambda_f)
    out["nsegs_method"] = str(nsegs_method)
    out["cm"] = float(cm)
    out["Ra"] = float(Ra)
    out["d_lambda"] = float(d_lambda)

    if spine_base_section is not None:
        out["spine_base_section"] = out["base_node_id"].map(
            lambda b: spine_base_section.get(int(b), "")).fillna("")
    else:
        out["spine_base_section"] = ""

    cols = [c for c in C09_COLUMNS if c in out.columns]
    extra = [c for c in out.columns if c not in cols]
    out = out[cols + extra]
    out.to_csv(path, index=False)
    return {"path": str(path), "n_rows": int(len(out)),
            "n_redirected": int(out["redirected"].sum())
            if "redirected" in out.columns else 0}


def spine_base_section_map(spine_bases_df):
    """base_node_id -> spine_base_section, from the exporter's own table."""
    if not len(spine_bases_df):
        return {}
    return {int(r.base_node_id): str(r.spine_base_section)
            for r in spine_bases_df.itertuples(index=False)}


# --------------------------------------------------------------------------- #
#  6. Regression guard                                                         #
# --------------------------------------------------------------------------- #
REGRESSION_KEYS = ("qc_status", "n_sections", "n_branches",
                   "n_sections_multi_branch", "n_spines_unattached",
                   "f_implied", "F_lit", "A_shaft_um2", "A_spine_um2")


def regression_check(res_unaligned, res_aligned, rtol=0.0, atol=0.0):
    """Alignment is rigid, so section 7's quantities must be BIT-IDENTICAL.

    Defaults are exact equality on purpose. Any movement means the transform is
    not rigid or was applied at the wrong step -- not that the tolerance needs
    loosening.
    """
    diffs = {}
    for k in REGRESSION_KEYS:
        a, b = res_unaligned.get(k), res_aligned.get(k)
        if isinstance(a, float) and isinstance(b, float):
            if not (a == b or abs(a - b) <= atol + rtol * abs(a)):
                diffs[k] = (a, b)
        elif a != b:
            diffs[k] = (a, b)
    return {"identical": not diffs, "diffs": diffs}


# --------------------------------------------------------------------------- #
#  7. Orchestrator                                                             #
# --------------------------------------------------------------------------- #
def align_and_export(df_raw, nid, output_dir, metadata_df, cm, Ra,
                     df_labelled=None, syn_df=None, label_fn=None,
                     k_neighbors=3, lambda_f=100.0, d_lambda=0.1,
                     nsegs_method="lambda_f", cell_factory=None,
                     spine_length_threshold_nm=None, mislabel_k=5,
                     verbose=True, **export_kwargs):
    """Run the section 3 pipeline for one neuron.

    Steps 1-8 and 10 are morphology_exporter.export_neuron, unchanged. Step 9
    is injected as its `align_fn` hook. Steps 11-12 are the snapper and the
    C-09 emitter, and run only when `syn_df` and `df_labelled` are supplied.

    `df_labelled` is the PRE-PRUNE labelled frame. export_neuron does not
    return it, so a caller wanting the redirect must build it and pass it in.
    """
    if spine_length_threshold_nm is None:
        spine_length_threshold_nm = mx.SPINE_LENGTH_THRESHOLD_NM

    soma_pos = soma_position_nm(df_raw)
    mean_matrix, diag = neighbourhood_rotation(soma_pos, metadata_df, k_neighbors)

    res = mx.export_neuron(
        df_raw, nid, output_dir, label_fn=label_fn,
        spine_length_threshold_nm=spine_length_threshold_nm,
        align_fn=make_align_fn(soma_pos, mean_matrix),
        mislabel_k=mislabel_k, verbose=verbose, **export_kwargs)

    res["alignment"] = dict(diag)
    res["alignment"]["soma_pos_nm"] = [float(v) for v in soma_pos]
    res["alignment"]["mean_matrix"] = [[float(v) for v in row]
                                       for row in mean_matrix]
    res["segmentation"] = {"cm": float(cm), "Ra": float(Ra),
                           "lambda_f": float(lambda_f),
                           "d_lambda": float(d_lambda),
                           "nsegs_method": str(nsegs_method)}
    res["module_versions"] = {"alignment": MODULE_VERSION,
                              "morphology_exporter": mx.MODULE_VERSION}

    if res["qc_status"] == "fail" or syn_df is None or df_labelled is None:
        return res

    anchored = resolve_synapse_anchors(syn_df, df_labelled, soma_pos, mean_matrix)
    res["n_synapses"] = int(len(anchored))
    res["n_on_pruned_spine"] = int(anchored["on_pruned_spine"].sum())
    res["n_unresolved_spine_bases"] = int(
        anchored.attrs.get("n_unresolved_spine_bases", 0))

    factory = cell_factory or default_cell_factory
    hoc_entry = res["files"]["hoc"]
    hoc_path = hoc_entry if isinstance(hoc_entry, str) else hoc_entry["path"]
    cell = factory(hoc_path, cm=cm, Ra=Ra, lambda_f=lambda_f,
                   d_lambda=d_lambda, nsegs_method=nsegs_method)
    try:
        snapped = snap_synapses(anchored, cell)
        res["totnsegs"] = int(getattr(cell, "totnsegs", -1))
    finally:
        closer = getattr(cell, "__del__", None)
        if callable(closer):
            try:
                closer()
            except Exception:                          # noqa: BLE001
                pass

    base_map = {}
    sb = res["files"].get("spine_bases")
    if isinstance(sb, str) and os.path.isfile(sb):
        base_map = spine_base_section_map(pd.read_csv(sb))

    out_path = os.path.join(output_dir,
                            "neuron_%s_mapped_synapses.csv" % nid)
    res["files"]["mapped_synapses"] = write_mapped_synapses(
        snapped, out_path, cm=cm, Ra=Ra, lambda_f=lambda_f,
        d_lambda=d_lambda, nsegs_method=nsegs_method,
        spine_base_section=base_map)
    res["n_redirected"] = int(snapped["redirected"].sum())

    prov_path = os.path.join(output_dir, "neuron_%s_alignment.json" % nid)
    with open(prov_path, "w", newline="\n") as fh:
        fh.write(json.dumps({k: v for k, v in res.items() if k != "files"},
                            indent=2, sort_keys=True, default=str))
    res["files"]["alignment_provenance"] = prov_path

    if verbose:
        print("[OK] neuron %s aligned: %d synapses, %d on pruned spines, "
              "%d redirected, totnsegs %s"
              % (nid, res["n_synapses"], res["n_on_pruned_spine"],
                 res["n_redirected"], res.get("totnsegs")))
    return res
