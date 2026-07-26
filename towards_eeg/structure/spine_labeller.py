"""spine_labeller -- the H01 spine identification step, extracted verbatim.

WHY THIS FILE EXISTS
--------------------
label_dendritic_spines_robust lives in morpholgy_pathways__6_.py, which is a
Colab NOTEBOOK EXPORT, not an importable module: it carries top-level code that
mounts Drive twice (L13, L22), runs alignment over ~450 neurons (L762), runs two
further alignment passes (L1170, L1636) and calls compute_F_factors (L2553).
Importing that file would execute the entire pipeline. This module therefore
carries the single function needed by stage S1.3, extracted VERBATIM.

O7 BOOKKEEPING -- WHICH DEFINITION THIS IS
------------------------------------------
morpholgy_pathways__6_.py contains TWO shadowed definitions of the labeller.
This file carries the FIRST one:

    source file   : morpholgy_pathways__6_.py
    line range    : 1653-1825 (inclusive), copied without modification
    variant       : head/neck aware (emits 'head' and 'neck' labels)
    default thr.  : 5000.0 nm  -- LEFT AS FOUND, deliberately
    sha256 (body) : 14b2e701cc9bd2dcacec27e581e36abee3963c19eec9888d7227075e515713a4

    the OTHER definition, NOT used here, is at L2353: threshold default 3000 nm,
    no head/neck split.

The default threshold is left at the original 5000 nm so this file stays a
byte-faithful copy whose hash can be compared against the source. The project
setting of 4000 nm is passed EXPLICITLY at the call site by
phi_pipeline_colab.compute_phi_factors, and is recorded in the provenance JSON.
Do not "fix" the default here: that would break the hash correspondence and
hide which value was actually used.

This is a copy, and copies are how O7 happened in the first place. It is
justified here only because the export cannot be imported; when S1.0 builds the
merged exporter in the repository, the labeller must become ONE definition and
this file should be replaced by an import of it.
"""

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from scipy.interpolate import splprep, splev, interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

SOURCE_FILE = "morpholgy_pathways__6_.py"
SOURCE_LINES = (1653, 1825)
SOURCE_SHA256 = "14b2e701cc9bd2dcacec27e581e36abee3963c19eec9888d7227075e515713a4"


# --------------------------------------------------------------------------- #
# VERBATIM EXTRACTION BEGINS -- do not edit below this line                    #
# --------------------------------------------------------------------------- #
def label_dendritic_spines_robust(neuron_ids, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons', output_dir=None, spine_length_threshold_nm=5000.0, num_interp_points=100, smoothing_sigma=2.0):
    """
    Identifies dendritic spines by evaluating entire terminal subtrees,
    then evaluates their morphological radius profiles to explicitly
    label discrete nodes as 'head' or 'neck'.
    """
    updated_neurons = {}

    print(f" Searching and quantifying dendritic spines across {len(neuron_ids)} neurons (Threshold: {spine_length_threshold_nm} nm)...")

    for nid in neuron_ids:
        filepath = os.path.join(input_dir, f"neuron_{nid}.csv")

        if not os.path.exists(filepath):
            print(f"[WARN] File for neuron {nid} not found. Skipping.")
            continue

        df = pd.read_csv(filepath)

        if not {'id', 'p', 'x', 'y', 'z', 'annotated_type'}.issubset(df.columns):
            print(f"[WARN] Missing required columns in neuron {nid}. Skipping.")
            continue

        # Ensure radius column exists for morphology
        if 'r' not in df.columns:
            df['r'] = 50.0

        # 1. Build a Directed Graph and Helper Dicts
        G = nx.DiGraph()
        G.add_nodes_from(df['id'])
        valid_edges = df[df['p'] != -1][['p', 'id']].values
        G.add_edges_from(valid_edges)

        node_dict = df.set_index('id').to_dict('index')
        node_types = df.set_index('id')['annotated_type'].to_dict()

        children_map = defaultdict(list)
        for _, row in df[df['p'] != -1].iterrows():
            children_map[row['p']].append(row['id'])

        spine_nodes = set()

        # 2. Identify all branch points in the neuron
        branch_points = [n for n in G.nodes() if G.out_degree(n) > 1]

        # 3. Evaluate the subtrees attached to each branch point (Find Spines)
        for bp in branch_points:
            parent_type = str(node_types.get(bp, ''))
            if not re.search(r'dendrite|apical|^1$', parent_type, re.IGNORECASE):
                continue

            for child in G.successors(bp):
                try:
                    subtree_nodes = nx.descendants(G, child)
                    subtree_nodes.add(child)
                except nx.NetworkXError:
                    continue

                total_subtree_length = 0.0
                for node in subtree_nodes:
                    parent = list(G.predecessors(node))[0]
                    p1 = np.array([node_dict[node]['x'], node_dict[node]['y'], node_dict[node]['z']])
                    p2 = np.array([node_dict[parent]['x'], node_dict[parent]['y'], node_dict[parent]['z']])
                    total_subtree_length += np.linalg.norm(p1 - p2)

                if 0 < total_subtree_length <= spine_length_threshold_nm:
                    spine_nodes.update(subtree_nodes)

        # 4. Morphological Head/Neck Separation
        if spine_nodes:
            spine_roots = [
                nid for nid in spine_nodes
                if node_dict[nid]['p'] not in spine_nodes
            ]

            head_nodes = set()
            neck_nodes = set()

            for root_id in spine_roots:
                # Extract paths from root to tips
                paths = []
                def dfs_paths(current_node, current_path):
                    current_path.append(current_node)
                    spine_children = [c for c in children_map[current_node] if c in spine_nodes]
                    if not spine_children:
                        paths.append(list(current_path))
                    else:
                        for child in spine_children:
                            dfs_paths(child, list(current_path))
                dfs_paths(root_id, [])

                for path in paths:
                    clean_path, path_coords, path_radii = [], [], []
                    for n in path:
                        coord = np.array([node_dict[n]['x'], node_dict[n]['y'], node_dict[n]['z']])
                        if not path_coords or np.linalg.norm(coord - path_coords[-1]) > 1e-4:
                            path_coords.append(coord)
                            path_radii.append(node_dict[n]['r'])
                            clean_path.append(n)

                    if len(path_coords) < 3:
                        head_nodes.update(clean_path)
                        continue

                    path_coords = np.array(path_coords)
                    path_radii = np.array(path_radii)

                    # Interpolation
                    k = min(3, len(path_coords) - 1)
                    tck, u = splprep(path_coords.T, s=0, k=k)
                    u_new = np.linspace(0, 1, num_interp_points)
                    smooth_coords = np.array(splev(u_new, tck)).T
                    interp_kind = 'cubic' if len(path_coords) > 3 else 'linear'
                    smooth_radii = interp1d(u, path_radii, kind=interp_kind)(u_new)
                    smooth_radii = np.clip(smooth_radii, a_min=1.0, a_max=None)

                    # Distance calculations on interpolated curve
                    diffs = np.diff(smooth_coords, axis=0)
                    ds = np.linalg.norm(diffs, axis=1)
                    ds = np.insert(ds, 0, 0)

                    radii_tip_to_base = smooth_radii[::-1]
                    ds_tip_to_base = ds[::-1]
                    dist_from_tip = np.cumsum(ds_tip_to_base)

                    filtered_radii = gaussian_filter1d(radii_tip_to_base, sigma=smoothing_sigma)
                    prominence_threshold = np.max(filtered_radii) * 0.05
                    minima, _ = find_peaks(-filtered_radii, prominence=prominence_threshold)

                    # Identify the boundary threshold
                    total_length = dist_from_tip[-1]
                    if len(minima) > 0:
                        neck_start_idx = minima[0]
                    else:
                        target_dist = total_length / 3.0
                        neck_start_idx = np.searchsorted(dist_from_tip, target_dist)
                        neck_start_idx = min(neck_start_idx, len(dist_from_tip) - 1)

                    cutoff_dist_from_tip = dist_from_tip[neck_start_idx]

                    # Map labels back to discrete nodes
                    orig_ds = [0.0]
                    for i in range(1, len(path_coords)):
                        orig_ds.append(np.linalg.norm(path_coords[i] - path_coords[i-1]))
                    orig_cum_dist = np.cumsum(orig_ds)
                    orig_total_len = orig_cum_dist[-1]

                    for i, node in enumerate(clean_path):
                        node_dist_from_tip = orig_total_len - orig_cum_dist[i]
                        if node_dist_from_tip <= cutoff_dist_from_tip:
                            head_nodes.add(node)
                        else:
                            neck_nodes.add(node)

            # Resolve branch conflicts (if a node is shared, err on the side of 'head')
            final_neck = neck_nodes - head_nodes
            final_head = head_nodes

            # 5. Apply the new labels and save
            df.loc[df['id'].isin(final_neck), 'annotated_type'] = 'neck'
            df.loc[df['id'].isin(final_head), 'annotated_type'] = 'head'

        updated_neurons[nid] = df
        print(f"[OK] Neuron {nid}: Identified and relabeled {len(spine_nodes)} spine nodes into Heads/Necks.")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_filepath = os.path.join(output_dir, f"neuron_{nid}_spines.csv")
            df.to_csv(out_filepath, index=False)

    return updated_neurons


