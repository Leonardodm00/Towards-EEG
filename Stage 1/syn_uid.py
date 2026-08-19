"""syn_uid -- stable, content-addressed identifiers for H01 synapses.

Towards-EEG stage S1 (data-generation side).

WHY THIS EXISTS
---------------
Every downstream index that identifies a synapse today is derived, not intrinsic:

  * `lfpy_idx` is a function of the segmentation Lambda = (cm, Ra, lambda_f,
    d_lambda, nsegs_method). Change any element of Lambda and essentially every
    index moves. It is a cache, not an identity.
  * `node_id` is a function of the skeleton, and it does not survive pruning:
    a synapse on a pruned spine is redirected to its base node, so two different
    synapses can end up sharing a node_id.
  * row position survives nothing at all.

Any paired comparison -- spine-resolved against spine-pruned exports, or one
segmentation against another -- needs a key that is a function of the SYNAPSE,
not of the model built around it. That is what syn_uid is: a digest of the
synapse's own raw coordinates and type, assigned once, at load time, before any
filtering, mapping or pruning.

    syn_uid(s) = first `digest_bytes` of
                 sha256("v1|nid|x_nm|y_nm|z_nm|type|direction|k")

with coordinates rounded to integer nanometres and `k` a within-key occurrence
counter (see below). The "v1" prefix is a format tag: if the key composition
ever changes, bump it rather than silently reissuing different digests for the
same synapses.

THE OCCURRENCE COUNTER, AND WHY IT IS NOT OPTIONAL
--------------------------------------------------
Two rows in an H01 synapse export can carry identical coordinates and type. A
pure content digest would then collide, and a collision is worse than a bad key
because it silently merges two synapses in a join. `k` disambiguates: rows
sharing a key are sorted deterministically and numbered 0, 1, 2, ... The sort is
by the key itself and then by original row position, so the assignment is stable
for a given input file but NOT stable against a reordering of the input file.
That is an honest limitation and is reported: `n_keys_with_multiplicity` in the
report tells you how many identifiers depend on row order. If it is zero -- the
expected case -- syn_uid is a pure function of content and is stable against
anything.

THE NODE-COLLAPSE PROBLEM THIS MODULE MEASURES BUT DOES NOT FIX
---------------------------------------------------------------
The legacy mapping step assigns each synapse to its nearest skeleton node by
writing a label INTO the skeleton frame:

    neuron_df.at[df_idx, 'synapse_label'] = label

That is last-write-wins per node. Two synapses whose nearest node is the same
node produce ONE labelled node, carrying whichever label was written last. The
downstream extract_synapse_frame then emits one row per synapse-bearing NODE.
So a per-node count is not a per-synapse count, and mixed excitatory/inhibitory
collisions resolve arbitrarily.

`collapse_report` below quantifies this for a given cell: how many raw synapses,
how many distinct nearest nodes, how many synapses lost, and how many of the
lost ones were label-discordant (a collision between an excitatory and an
inhibitory synapse, which is the case that actually changes the model). It does
NOT change the mapping. Whether the collapse matters is an empirical question
this function answers; fixing it is a separate decision.

DEPENDENCIES: numpy, pandas. scipy.spatial only for collapse_report, and only
when a skeleton is supplied.

Pure ASCII source (HPC-safe).
"""

import hashlib

import numpy as np
import pandas as pd

MODULE_VERSION = "syn_uid-1.0.0"

KEY_FORMAT_VERSION = "v1"

# H01 voxel -> nm. Matches the scaling in the legacy mapping step.
H01_VOXEL_NM = (8.0, 8.0, 33.0)

DEFAULT_DIGEST_BYTES = 8


def _digest(parts, digest_bytes):
    raw = "|".join(str(p) for p in parts).encode("ascii", "strict")
    return hashlib.sha256(raw).hexdigest()[: 2 * digest_bytes]


def assign_syn_uid(syn_df,
                   nid,
                   coord_columns=("location_x", "location_y", "location_z"),
                   voxel_scale=H01_VOXEL_NM,
                   type_column="synapse_type",
                   direction_column="direction",
                   digest_bytes=DEFAULT_DIGEST_BYTES,
                   uid_column="syn_uid"):
    """Add a stable `syn_uid` column to a RAW H01 synapse export.

    Call this as early as possible -- on the file as downloaded, before any
    direction filter, before the KD-tree mapping, before anything drops rows.
    Filtering afterwards preserves the identifiers of the rows that survive,
    which is the entire point.

    Parameters
    ----------
    syn_df : pandas.DataFrame
        The raw export. Needs `coord_columns`; `type_column` and
        `direction_column` are used if present and skipped (with a note in the
        report) if not.
    nid : hashable
        Neuron id. Enters the key, so the same physical synapse seen from its
        two partner cells gets two different identifiers. That is deliberate:
        C-09 is per (cell, synapse), and a shared identifier would make the
        per-cell tables collide on join.
    voxel_scale : tuple of three floats
        Voxel -> nm scaling applied to the coordinates before rounding. Pass
        (1.0, 1.0, 1.0) if the input is already nm.
    digest_bytes : int
        Bytes of sha256 kept. 8 bytes = 16 hex chars gives a collision
        probability below 1e-9 for fewer than about 6 million synapses, which
        is far above any per-cell count. Raise it, do not lower it.

    Returns
    -------
    (out, report) : (DataFrame, dict)
        `out` is a copy of syn_df with `uid_column` inserted as the first
        column. `report` records the key composition and the multiplicity
        statistics; store it in the provenance JSON.
    """
    for c in coord_columns:
        if c not in syn_df.columns:
            raise ValueError("synapse frame is missing coordinate column %r; "
                             "columns present: %r"
                             % (c, list(syn_df.columns)))
    if uid_column in syn_df.columns:
        raise ValueError("frame already has a %r column; refusing to reissue "
                         "identifiers (that would silently break any existing "
                         "join)" % uid_column)

    out = syn_df.copy().reset_index(drop=True)
    cx, cy, cz = coord_columns
    sx, sy, sz = voxel_scale

    # integer nm: rounding at the key boundary makes the digest insensitive to
    # float formatting, which differs between pandas versions and CSV round-trips
    xs = np.rint(pd.to_numeric(out[cx], errors="coerce").values * sx)
    ys = np.rint(pd.to_numeric(out[cy], errors="coerce").values * sy)
    zs = np.rint(pd.to_numeric(out[cz], errors="coerce").values * sz)

    n_nan_coord = int(np.isnan(xs).sum() + np.isnan(ys).sum()
                      + np.isnan(zs).sum())

    has_type = type_column in out.columns
    has_dir = direction_column in out.columns
    types = (out[type_column].astype(str).values if has_type
             else np.array([""] * len(out)))
    dirs = (out[direction_column].astype(str).values if has_dir
            else np.array([""] * len(out)))

    base_keys = [
        "%s|%s|%s|%s|%s|%s|%s" % (
            KEY_FORMAT_VERSION, nid,
            "nan" if np.isnan(x) else "%d" % int(x),
            "nan" if np.isnan(y) else "%d" % int(y),
            "nan" if np.isnan(z) else "%d" % int(z),
            t, d)
        for x, y, z, t, d in zip(xs, ys, zs, types, dirs)
    ]

    # deterministic within-key occurrence counter
    order = sorted(range(len(base_keys)), key=lambda i: (base_keys[i], i))
    counter = {}
    occ = [0] * len(base_keys)
    for i in order:
        k = base_keys[i]
        occ[i] = counter.get(k, 0)
        counter[k] = occ[i] + 1

    uids = [_digest([base_keys[i], occ[i]], digest_bytes)
            for i in range(len(base_keys))]

    out.insert(0, uid_column, uids)

    n_multi = sum(1 for k, v in counter.items() if v > 1)
    report = {
        "module_version": MODULE_VERSION,
        "key_format_version": KEY_FORMAT_VERSION,
        "nid": nid,
        "n_rows": int(len(out)),
        "n_distinct_keys": int(len(counter)),
        "n_keys_with_multiplicity": int(n_multi),
        "max_multiplicity": int(max(counter.values())) if counter else 0,
        "n_rows_order_dependent": int(sum(v for v in counter.values() if v > 1)),
        "n_nan_coordinates": n_nan_coord,
        "used_type_column": bool(has_type),
        "used_direction_column": bool(has_dir),
        "voxel_scale_nm": list(voxel_scale),
        "digest_bytes": int(digest_bytes),
        "uid_is_unique": bool(len(set(uids)) == len(uids)),
    }
    if not report["uid_is_unique"]:
        raise AssertionError(
            "syn_uid collision after occurrence counting -- this should be "
            "impossible; %d rows, %d distinct uids"
            % (len(uids), len(set(uids))))
    return out, report


def assign_syn_uid_mapped(mapped_df,
                          nid=None,
                          coord_columns=("x", "y", "z"),
                          voxel_scale=(1000.0, 1000.0, 1000.0),
                          type_column="synapse_label",
                          direction_column=None,
                          digest_bytes=DEFAULT_DIGEST_BYTES,
                          uid_column="syn_uid"):
    """Assign identifiers to the ALREADY-MAPPED frame (nid, node_id, x, y, z,
    synapse_label), whose coordinates are in um.

    Use this only when the raw export is not available. The resulting
    identifiers are stable across exports and across segmentations, but they
    identify a synapse-bearing NODE, not a synapse: any synapses lost to the
    node collapse described in the module docstring were already gone before
    this function saw the frame, and no identifier can recover them. The
    returned report carries `identifies` = 'node' to make that explicit in the
    provenance record.
    """
    if nid is None and "nid" in mapped_df.columns and len(mapped_df):
        nid = mapped_df["nid"].iloc[0]
    out, report = assign_syn_uid(
        mapped_df, nid, coord_columns=coord_columns, voxel_scale=voxel_scale,
        type_column=type_column,
        direction_column=(direction_column or "__absent__"),
        digest_bytes=digest_bytes, uid_column=uid_column)
    report["identifies"] = "node"
    report["warning"] = ("assigned on the mapped frame; one row per "
                         "synapse-bearing node, not per synapse")
    return out, report


def collapse_report(raw_syn_df,
                    skeleton_df,
                    coord_columns=("location_x", "location_y", "location_z"),
                    voxel_scale=H01_VOXEL_NM,
                    skeleton_xyz=("x", "y", "z"),
                    type_column="synapse_type",
                    direction_column="direction",
                    direction_value="incoming"):
    """Quantify how many synapses the nearest-node mapping collapses.

    Reproduces the legacy nearest-node assignment (raw nm coordinates, KD-tree
    over skeleton nodes) and counts what the last-write-wins label assignment
    would discard. Changes nothing.

    Returns a dict with:
      n_synapses                 rows after the direction filter
      n_distinct_nodes           distinct nearest nodes
      n_synapses_lost            n_synapses - n_distinct_nodes
      frac_lost
      n_nodes_multi              nodes receiving more than one synapse
      max_per_node
      n_nodes_discordant         multi-synapse nodes whose synapses do NOT all
                                 share a label -- the collisions that change
                                 the excitatory/inhibitory balance
      n_synapses_discordant
      nn_dist_nm_p50/p95/max     nearest-node distances, as a sanity check that
                                 the mapping is finding real neighbours
    """
    from scipy.spatial import cKDTree

    syn = raw_syn_df
    if direction_column in syn.columns and direction_value is not None:
        syn = syn[syn[direction_column] == direction_value]
    syn = syn.dropna(subset=list(coord_columns))
    if len(syn) == 0:
        return {"n_synapses": 0, "n_distinct_nodes": 0, "n_synapses_lost": 0,
                "frac_lost": float("nan")}

    sx, sy, sz = voxel_scale
    coords = np.column_stack([
        pd.to_numeric(syn[coord_columns[0]]).values * sx,
        pd.to_numeric(syn[coord_columns[1]]).values * sy,
        pd.to_numeric(syn[coord_columns[2]]).values * sz,
    ])
    nodes = skeleton_df[list(skeleton_xyz)].values.astype(float)
    tree = cKDTree(nodes)
    dist, idx = tree.query(coords)

    labels = (syn[type_column].astype(str).values if type_column in syn.columns
              else np.array([""] * len(syn)))
    per_node = {}
    for j, lab in zip(idx, labels):
        per_node.setdefault(int(j), []).append(lab)

    multi = {j: v for j, v in per_node.items() if len(v) > 1}
    discordant = {j: v for j, v in multi.items() if len(set(v)) > 1}
    return {
        "n_synapses": int(len(syn)),
        "n_distinct_nodes": int(len(per_node)),
        "n_synapses_lost": int(len(syn) - len(per_node)),
        "frac_lost": float(len(syn) - len(per_node)) / len(syn),
        "n_nodes_multi": int(len(multi)),
        "max_per_node": int(max(len(v) for v in per_node.values())),
        "n_nodes_discordant": int(len(discordant)),
        "n_synapses_discordant": int(sum(len(v) for v in discordant.values())),
        "nn_dist_nm_p50": float(np.percentile(dist, 50)),
        "nn_dist_nm_p95": float(np.percentile(dist, 95)),
        "nn_dist_nm_max": float(np.max(dist)),
    }
