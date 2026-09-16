"""s0_ingest -- bring a `Reconstructed neurons/neuron_{id}.csv` into the S0 contract.

Pure computation, no network, no plotting. ASCII only, LF only.

Why this exists
---------------
CELL 6 of the Spine Mesh Analysis notebook reads `data/{id}.csv` with columns
`id, p, x, y, z, annotated_type` in nanometres (TEEG_27 sec. 7 item 1). The
file that actually holds that information on Drive is
`Colab Notebooks/Reconstructed neurons/neuron_{id}.csv`, written by
`automated_reconstruction.py` (repo Leonardodm00/Towards-EEG). Reading that
pipeline (Rec_Utility.py) establishes the frame facts this module relies on:

  * x, y, z are the cloud-volume `skeleton.get(id).to_swc()` coordinates,
    i.e. the raw c3 release frame in nanometres. The bounding-box affine used
    to label nodes (`annotate_stitched_neuron_nopr`) is applied to TEMPORARY
    coordinates only; the stored coordinates are untouched.
  * `r` is the SWC radius = Kimimaro DBF inscribed-sphere radius (H01 ref
    eq. 4), NOT a cross-sectional radius.
  * `annotated_type` is Title-case H01 6-class text ('Axon', 'Dendrite',
    'Soma', ... 'Myelinated Axon', 'Unknown'); the spine labeller later
    rewrites some nodes to 'spine' / 'head' / 'neck'.
  * `p` may contain -99999: `treat_distant_orphans` uses it as a dummy parent
    for far orphan roots that are already topologically attached. It is a
    root marker, not a node id.
  * The soma is collapsed to a single 'virtual soma' root; that node's
    position and radius do not describe a membrane and must be excluded
    from any registration metric.

Contract written to `data/{id}.csv`
-----------------------------------
columns: id (int), p (int, -1 at roots), x, y, z (float, nm), r (float, nm),
annotated_type (lower-case str), is_root (bool), n_roots_in_file (int).
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REQUIRED = ("id", "p", "x", "y", "z", "annotated_type")
ROOT_MARKERS = (-1, -99999)
NM_MIN_EXTENT = 1.0e4      # a real cell spans >> 10 um in at least one axis
NM_MAX_COORD = 5.0e6       # H01 volume is ~ 3.6 x 2.7 x 0.17 mm; 5 mm is a hard ceiling


def load_reconstructed(path: str) -> pd.DataFrame:
    """Read a Reconstructed-neurons CSV. Raises if the required columns are absent."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise KeyError("missing columns %s in %s; have %s" % (missing, path, list(df.columns)))
    return df


def normalise(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Return (table in the S0 contract, report). Never mutates the input.

    Steps: cast ids to int; map every root marker to -1; lower-case the label;
    drop columns the contract does not carry; add is_root and n_roots_in_file.
    """
    out = pd.DataFrame({
        "id": df["id"].astype(np.int64).to_numpy(),
        "p": df["p"].astype(np.int64).to_numpy(),
        "x": df["x"].astype(np.float64).to_numpy(),
        "y": df["y"].astype(np.float64).to_numpy(),
        "z": df["z"].astype(np.float64).to_numpy(),
        "r": (df["r"].astype(np.float64).to_numpy() if "r" in df.columns
              else np.full(len(df), np.nan)),
        "annotated_type": df["annotated_type"].astype(str).str.strip().str.lower().to_numpy(),
    })
    n_dummy = int(np.isin(out["p"].to_numpy(), [m for m in ROOT_MARKERS if m != -1]).sum())
    out.loc[out["p"].isin(ROOT_MARKERS), "p"] = -1
    out["is_root"] = out["p"].eq(-1)
    out["n_roots_in_file"] = int(out["is_root"].sum())
    report = {
        "n_nodes": int(len(out)),
        "n_roots": int(out["is_root"].sum()),
        "n_dummy_parents_mapped_to_root": n_dummy,
        "labels": out["annotated_type"].value_counts().to_dict(),
        "extent_nm": (out[["x", "y", "z"]].max() - out[["x", "y", "z"]].min()).round(1).to_dict(),
        "r_nan_fraction": float(np.isnan(out["r"].to_numpy()).mean()),
    }
    return out, report


def validate(out: pd.DataFrame) -> None:
    """Structural gate on the S0 contract. Raises AssertionError with the reason."""
    ids = out["id"].to_numpy()
    par = out["p"].to_numpy()
    assert len(np.unique(ids)) == len(ids), "duplicate node ids"
    known = set(ids.tolist())
    bad = [q for q in np.unique(par).tolist() if q != -1 and q not in known]
    assert not bad, "parent ids absent from table: %s" % bad[:10]
    assert not (par == ids).any(), "self-parent"
    assert int((par == -1).sum()) >= 1, "no root"
    xyz = out[["x", "y", "z"]].to_numpy()
    assert np.isfinite(xyz).all(), "non-finite coordinates"
    ext = xyz.max(axis=0) - xyz.min(axis=0)
    assert ext.max() >= NM_MIN_EXTENT, "extent %.1f nm too small: not nanometres?" % ext.max()
    assert xyz.max() <= NM_MAX_COORD, "coordinate %.3g exceeds H01 volume: not nanometres?" % xyz.max()
    r = out["r"].to_numpy()
    assert (np.isnan(r) | (r > 0)).all(), "non-positive radius"


def write_s0_table(src_csv: str, data_dir: str, cell: int) -> Tuple[str, Dict[str, object]]:
    """Ingest src_csv and write `data_dir/{cell}.csv`. Returns (path, report)."""
    df = load_reconstructed(src_csv)
    out, report = normalise(df)
    validate(out)
    os.makedirs(data_dir, exist_ok=True)
    dst = os.path.join(data_dir, "%d.csv" % cell)
    out.to_csv(dst, index=False, lineterminator="\n")
    report["written"] = dst
    return dst, report


def registration_mask(out: pd.DataFrame) -> np.ndarray:
    """Boolean mask of nodes that may enter the S2 d/r statistic.

    Excludes roots (the virtual soma is a centroid, not a membrane point),
    soma-labelled nodes, and nodes with unknown or NaN radius.
    """
    lab = out["annotated_type"].to_numpy()
    r = out["r"].to_numpy()
    keep = (~out["is_root"].to_numpy()) & (lab != "soma") & np.isfinite(r) & (r > 0)
    return keep
