# Colab driver for the D-7 synapse redirect audit (TEEG_16 section 3.3).
#
# This is a plain script meant to be pasted into Colab CELL BY CELL, following
# this project's existing convention (Alignment.py, Alignment Metadata/Usage.py
# are the same style). Cell boundaries are marked with '# %% CELL N'.
#
# What it does, cell by cell:
#   1  upload synapse_redirect_audit.py + smoke_synapse_redirect_audit.py
#   2  run the smoke test IN Colab (8/8 must pass before anything below runs)
#   3  mount Drive, pip install LFPy/neuron
#   4  point at your existing modules and pipeline folders  <- EDIT THIS CELL
#   5  soma_pos / mean_matrix, k=3 local, verbatim Alignment.py L235-251
#   6  replicate export_neuron steps 2-6 to recover the PRE-PRUNE labelled
#      frame the audit needs (export_neuron does not return it)
#   7  run export_neuron itself (aligned .hoc, section_table, spine_bases)
#      and cross-check qc_status against cell 6, then verify the two
#      independent spine-base walks agree
#   8  per-synapse node correspondence, fixing the id-vs-row-position issue
#      that map_synapses_to_segments.py's own return value does not expose
#   9  build the aligned LFPy.Cell with cm/Ra as top-level kwargs (never
#      inside passive_parameters -- section 5.4)
#  10  run the audit, print the report, save the CSV
#
# Every coordinate that ENTERS this pipeline is raw nanometres. The only
# um conversion happens where write_hoc and the transform_fn do it,
# consistent with TEEG_16 section 3.1's "one deviation, deliberate".


# %% CELL 1 -- upload the two files delivered with this driver
from google.colab import files
uploaded = files.upload()
for needed in ("synapse_redirect_audit.py", "smoke_synapse_redirect_audit.py"):
    if needed not in uploaded:
        raise FileNotFoundError(
            "expected %r in the upload dialog -- select both files together"
            % needed)
print("uploaded:", list(uploaded))


# %% CELL 2 -- smoke test BEFORE anything else runs
import subprocess, sys

r = subprocess.run([sys.executable, "smoke_synapse_redirect_audit.py"],
                   capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr)
    raise RuntimeError("smoke test did not pass 8/8 -- stop, do not proceed")


# %% CELL 3 -- environment
from google.colab import drive
drive.mount('/content/drive')
get_ipython().system('pip install -q neuron LFPy')  # noqa: F821  (Colab magic)


# %% CELL 4 -- EDIT THIS CELL: paths and the target neuron -------------------
import os
import sys

# Folder that holds the S1 modules, IF they are on Drive. Per TEEG_16 section 8
# the S1 working tree is "still loose files" and was never committed, so on a
# fresh session they are usually NOT on Drive at all. Cell 4b below finds them
# wherever they are, or prompts you to upload them.
MODULES_DIR = "/content/drive/MyDrive/Colab Notebooks"
if os.path.isdir(MODULES_DIR):
    sys.path.insert(0, MODULES_DIR)

SKELETONS_DIR = "/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons"
SYNAPSES_DIR = "/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons"  # EDIT if different
METADATA_CSV = SKELETONS_DIR + "/alignment_metadata_L2.csv"   # this cell's subpopulation bank
OUTPUT_DIR = "/content/drive/MyDrive/Colab Notebooks/Aligned Neurons HOC"

NID = 794820508                 # must have BOTH a skeleton csv and a synapses csv
K_NEIGHBORS = 3                 # confirmed: local, not the whole-bank mean

CM_BASE = 0.50                  # Eyal 2016/2018, shaft-referenced (uF/cm2)
RA = 268.5                      # Eyal 2016 Table 1 mean; replace with your
                                # fitted value once the passive fit lands
LAMBDA_F = 100.0
D_LAMBDA = 0.1


# %% CELL 4b -- locate or upload the S1 modules ------------------------------
# The full dependency chain, in dependency order. morphology_exporter imports
# spine_density, node_classify AND soma_enforce; soma_enforce imports
# node_classify. Uploading only the files you edited is not enough.
# Alignment.py is deliberately NOT here: importing it would execute its
# top-level drive.mount() and its L378 call. Its L235-251 mathematics is
# copied verbatim into cell 5 instead.
import importlib

NEEDED = ["spine_density.py", "node_classify.py", "soma_enforce.py",
          "morphology_exporter.py", "spine_labeller.py"]


def _have(fname):
    return any(os.path.isfile(os.path.join(d, fname)) for d in sys.path)


missing = [f for f in NEEDED if not _have(f)]
if missing:
    print("searching Drive for:", missing)
    hits = {}
    for root, dirs, fnames in os.walk("/content/drive/MyDrive"):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in list(missing):
            if f in fnames:
                hits.setdefault(f, os.path.join(root, f))
        if len(hits) == len(missing):
            break
    for f, p in hits.items():
        d = os.path.dirname(p)
        if d not in sys.path:
            sys.path.insert(0, d)
        print("  found %-26s %s" % (f, d))
    missing = [f for f in NEEDED if not _have(f)]

if missing:
    print("\nNOT on Drive -- select these in the dialog:", missing)
    from google.colab import files as _files
    up = _files.upload()
    if "" not in sys.path:
        sys.path.insert(0, "")
    still = [f for f in NEEDED if f not in up and not _have(f)]
    if still:
        raise FileNotFoundError("still missing: %s" % still)

for f in NEEDED:
    m = importlib.import_module(f[:-3])
    print("OK  %-26s %s" % (f, getattr(m, "MODULE_VERSION", "")))
print("\nall S1 modules importable")


# %% CELL 5 -- imports + soma_pos / mean_matrix, verbatim Alignment.py L235-251
import ast
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import node_classify as nc
import soma_enforce as se
import morphology_exporter as mx
import spine_labeller as sl
import synapse_redirect_audit as sra

df_raw = pd.read_csv("%s/neuron_%s.csv" % (SKELETONS_DIR, NID))

metadata_df = pd.read_csv(METADATA_CSV)
if isinstance(metadata_df["rotation_matrix"].iloc[0], str):
    metadata_df["rotation_matrix"] = metadata_df["rotation_matrix"].apply(
        ast.literal_eval)
reference_somas = metadata_df[["soma_x", "soma_y", "soma_z"]].to_numpy(float)


def soma_pos_and_mean_matrix(df_raw, reference_somas, metadata_df, k_neighbors):
    """Verbatim Alignment.py L235-251 (soma extraction + neighbourhood mean
    rotation). Not imported from Alignment.py because that file also does
    drive.mount() and disk I/O at import time; the mathematics is copied
    exactly, unchanged.
    """
    root_rows = df_raw[df_raw["p"] == -1]
    if root_rows.empty:
        raise ValueError("no root node (p == -1) in the raw skeleton")
    soma_pos = root_rows.iloc[0][["x", "y", "z"]].to_numpy(float)

    distances = np.linalg.norm(reference_somas - soma_pos, axis=1)
    k_actual = min(k_neighbors, len(metadata_df))
    nearest = np.argsort(distances)[:k_actual]
    matrices = np.array(metadata_df.iloc[nearest]["rotation_matrix"].tolist())
    mean_matrix = R.from_matrix(matrices).mean().as_matrix()
    return soma_pos, mean_matrix, nearest, distances[nearest]


soma_pos, mean_matrix, nearest_idx, nearest_dist_nm = \
    soma_pos_and_mean_matrix(df_raw, reference_somas, metadata_df, K_NEIGHBORS)
print("soma_pos (nm):", soma_pos)
print("k=%d nearest reference distance (um): %s"
     % (K_NEIGHBORS, np.round(nearest_dist_nm / 1000.0, 1)))
print("mean_matrix det = %.6f (must be 1.0)" % np.linalg.det(mean_matrix))


def to_aligned_um(raw_nm):
    """raw nm -> aligned um. Same soma_pos, same mean_matrix, every call."""
    raw_nm = np.asarray(raw_nm, dtype=float)
    return ((raw_nm - soma_pos) @ mean_matrix.T) / 1000.0


def align_fn(df_pruned, nid):
    """Passed to export_neuron. Applied AFTER pruning (step 9). Stays in NM:
    write_hoc does the single nm -> um conversion (section 3.1 deviation)."""
    out = df_pruned.copy()
    centered = out[["x", "y", "z"]].to_numpy(float) - soma_pos
    rotated = centered @ mean_matrix.T
    out["x"], out["y"], out["z"] = rotated[:, 0], rotated[:, 1], rotated[:, 2]
    return out


# %% CELL 6 -- the pre-prune labelled frame (export_neuron steps 2-6, replicated)
def label_fn(df, threshold_nm):
    """In-memory adapter around the verbatim L1653 labeller, which is
    disk-based (reads neuron_{nid}.csv from a directory)."""
    import tempfile, shutil, os
    tmp = tempfile.mkdtemp()
    try:
        df.to_csv(os.path.join(tmp, "neuron_%s.csv" % NID), index=False)
        out = sl.label_dendritic_spines_robust(
            [NID], input_dir=tmp, output_dir=None,
            spine_length_threshold_nm=threshold_nm)
        return out[NID] if isinstance(out, dict) else out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_prepruned_frame(df_raw, nid, label_fn, threshold_nm,
                          mislabel_policy="knn_relabel", mislabel_k=5,
                          mislabel_max_distance_nm=None, verbose=False):
    """Reproduces morphology_exporter.export_neuron steps 2-6 EXACTLY (same
    calls, same defaults) up to but excluding pruning, so the audit can walk
    spine -> base BEFORE prune_spines removes the spine nodes. export_neuron
    does not return this intermediate frame, hence the duplication here.
    """
    df = nc.classify_frame(df_raw)
    df, mis = nc.resolve_mislabelled_nodes(
        df, policy=mislabel_policy, k=mislabel_k,
        max_distance_nm=mislabel_max_distance_nm, rewrite_annotation=True)
    df = label_fn(df, threshold_nm)
    df = nc.classify_frame(df)
    df, soma_rep = se.enforce_soma(df, nid=nid, origin_tolerance_nm=None,
                                   plot_fn=None, verbose=verbose)
    return df, soma_rep, mis


df_lab, soma_rep, mislabel_rep = build_prepruned_frame(
    df_raw, NID, label_fn, mx.SPINE_LENGTH_THRESHOLD_NM)
print("pre-prune frame: qc_status=%s reasons=%s"
     % (soma_rep["qc_status"], soma_rep["reasons"]))


# %% CELL 7 -- run the real exporter, then cross-check the two spine-base walks
res = mx.export_neuron(
    df_raw, NID, OUTPUT_DIR, label_fn=label_fn,
    spine_length_threshold_nm=mx.SPINE_LENGTH_THRESHOLD_NM,
    align_fn=align_fn, mislabel_policy="knn_relabel", mislabel_k=5,
    mislabel_max_distance_nm=None, verbose=True)

assert res["qc_status"] == soma_rep["qc_status"], (
    "cell 6 and export_neuron disagree on qc_status: %r vs %r -- "
    "the two pipelines have drifted apart, stop and diff them"
    % (soma_rep["qc_status"], res["qc_status"]))

node_map = sra.map_nodes_to_spine_bases(df_lab, spine_classes={"spine"})
spine_bases_df = pd.read_csv(res["files"]["spine_bases"])
crosscheck = sra.verify_against_spine_bases(node_map, spine_bases_df)
print("independent spine-base walk vs exporter's own table:", crosscheck)
assert crosscheck["agree"], (
    "the audit's spine-base walk disagrees with morphology_exporter's -- "
    "stop, do not trust the audit until this is resolved: %r" % crosscheck)


# %% CELL 8 -- per-synapse node correspondence (id-vs-position fix)
def map_synapses_to_nodes_raw(nid, df_raw, synapses_dir,
                              voxel_res=(8.0, 8.0, 33.0)):
    """Same computation as map_synapses_to_segments.py (identical KD-tree,
    identical voxel_res), but returns a PER-SYNAPSE table with each synapse's
    own raw coordinate and its assigned node id -- which the original function
    does not return (it only writes a collapsed, many-to-one label per node).

    IMPORTANT: the KD-tree query returns POSITIONAL indices into df_raw. On
    both real skeletons checked, the 'id' column is NOT equal to row position
    (id runs past the row count, e.g. max id 63397 over 63147 rows on
    794820508). Using the positional index as if it were the node id would
    silently mismap every synapse. This function converts explicitly via
    df_raw['id'].to_numpy()[pos_idx].
    """
    syn_path = "%s/neuron_%s_synapses.csv" % (synapses_dir, nid)
    syn_df = pd.read_csv(syn_path)
    if "direction" in syn_df.columns:
        syn_df = syn_df[syn_df["direction"] == "incoming"]
    syn_df = syn_df.dropna(subset=["location_x", "location_y", "location_z"])
    if syn_df.empty:
        raise ValueError("no incoming synapses with coordinates for %r" % nid)

    node_coords_nm = df_raw[["x", "y", "z"]].to_numpy(float)
    tree = cKDTree(node_coords_nm)

    syn_coords_nm = syn_df[["location_x", "location_y", "location_z"]] \
        .to_numpy(float) * np.array(voxel_res)
    dist_nm, pos_idx = tree.query(syn_coords_nm)

    ids = df_raw["id"].to_numpy()
    node_ids = ids[pos_idx]                     # the fix: id, not position

    raw_type = syn_df.get("synapse_type", pd.Series(["unknown"] * len(syn_df),
                                                     index=syn_df.index))
    raw_type = raw_type.astype(str).str.lower()
    syn_type = np.where(
        raw_type.str.contains("2|exc|asymmetric", regex=True), "exc",
        np.where(raw_type.str.contains("1|inh|symmetric", regex=True),
                "inh", "unknown"))

    out = pd.DataFrame({
        "node_id": node_ids.astype(int),
        "syn_x_nm": syn_coords_nm[:, 0],
        "syn_y_nm": syn_coords_nm[:, 1],
        "syn_z_nm": syn_coords_nm[:, 2],
        "synapse_type": syn_type,
        "snap_distance_nm": dist_nm,
    })
    return out


syn = map_synapses_to_nodes_raw(NID, df_raw, SYNAPSES_DIR)
print("synapses mapped to nodes:", len(syn), "| by type:",
     syn["synapse_type"].value_counts().to_dict())
print("synapse-to-node snap distance (nm): median %.1f  max %.1f"
     % (syn["snap_distance_nm"].median(), syn["snap_distance_nm"].max()))


# %% CELL 9 -- build the aligned LFPy.Cell -- cm/Ra as TOP-LEVEL kwargs only
import LFPy


def _fpath(entry):
    """res['files'] is NOT homogeneous. The CSV entries are assigned as plain
    path strings, but write_hoc RETURNS A DICT
    {'path', 'n_sections', 'arrays', 'sha256'}, so res['files']['hoc'] is a
    dict. export_neuron's provenance writer flattens it via
    (v if isinstance(v, str) else v.get('path')), which is why the JSON on disk
    shows a clean string while the in-memory value does not. Passing the dict
    straight to LFPy.Cell raises
    'Could not recognize Cell keyword argument morphology'.
    """
    return entry if isinstance(entry, str) else entry["path"]


hoc_info = res["files"]["hoc"]
if isinstance(hoc_info, dict):
    print("hoc  sections=%d  arrays=%s" % (hoc_info["n_sections"],
                                           hoc_info["arrays"]))
    print("     sha256=%s" % hoc_info["sha256"][:16])
    assert hoc_info["n_sections"] == res["n_sections"], (
        "write_hoc and export_neuron disagree on the section count: %d vs %d"
        % (hoc_info["n_sections"], res["n_sections"]))

cell = LFPy.Cell(morphology=_fpath(res["files"]["hoc"]), passive=False,
                 nsegs_method="lambda_f", lambda_f=LAMBDA_F, d_lambda=D_LAMBDA,
                 Ra=RA, cm=CM_BASE, delete_sections=True)
print("totnsegs =", cell.totnsegs, "| Ra =", RA, "| cm =", CM_BASE)


# %% CELL 10 -- run the audit
sec_of_idx, sec_names, sec_length_um = sra.build_segment_index(cell)
section_table = pd.read_csv(_fpath(res["files"]["section_table"]))
parent_of = sra.build_section_tree(section_table)

node_xyz_nm = {int(r.id): (r.x, r.y, r.z) for r in df_lab.itertuples(index=False)}

audit = sra.audit_redirect(
    syn, node_xyz_nm, node_map, cell, to_aligned_um,
    parent_of, sec_of_idx, sec_names, sec_length_um)

summary = sra.summarise_audit(audit)
print(sra.format_summary(summary))

out_csv = "%s/neuron_%s_redirect_audit.csv" % (OUTPUT_DIR, NID)
audit.to_csv(out_csv, index=False)
print("\nwritten:", out_csv)

cell.__del__()
