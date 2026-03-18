import os
import re
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from scipy.interpolate import splprep, splev, interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def apply_spine_labels_to_df(df, spine_length_threshold_nm=5000.0, num_interp_points=100, smoothing_sigma=1.0):
    """
    Identifies spines, evaluates their morphological radius profiles, 
    and explicitly labels nodes as 'head' or 'neck' in the DataFrame.
    """
    if 'r' not in df.columns:
        df['r'] = 50.0  # Fallback radius if none exists
        
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
    branch_points = [n for n in G.nodes() if G.out_degree(n) > 1]

    # --- 1. Identify all Spine Nodes ---
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

    # --- 2. Morphological Head/Neck Separation ---
    if spine_nodes:
        # Find the base/root of every isolated spine
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
                # Filter out duplicate coordinates to prevent interpolation crashes
                clean_path, path_coords, path_radii = [], [], []
                for n in path:
                    coord = np.array([node_dict[n]['x'], node_dict[n]['y'], node_dict[n]['z']])
                    if not path_coords or np.linalg.norm(coord - path_coords[-1]) > 1e-4:
                        path_coords.append(coord)
                        path_radii.append(node_dict[n]['r'])
                        clean_path.append(n)
                
                # If the branch is too short, label it entirely as head
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
                
                # Calculate real distances of original nodes to map labels back
                orig_ds = [0.0]
                for i in range(1, len(path_coords)):
                    orig_ds.append(np.linalg.norm(path_coords[i] - path_coords[i-1]))
                orig_cum_dist = np.cumsum(orig_ds)
                orig_total_len = orig_cum_dist[-1]
                
                # Assign labels
                for i, node in enumerate(clean_path):
                    node_dist_from_tip = orig_total_len - orig_cum_dist[i]
                    if node_dist_from_tip <= cutoff_dist_from_tip:
                        head_nodes.add(node)
                    else:
                        neck_nodes.add(node)
                        
        # Resolve branch conflicts: If any tip claims a node is part of a head, it's a head
        final_neck = neck_nodes - head_nodes
        final_head = head_nodes
        
        # 3. Apply the new labels to the dataframe
        df.loc[df['id'].isin(final_neck), 'annotated_type'] = 'neck'
        df.loc[df['id'].isin(final_head), 'annotated_type'] = 'head'

    return df


def export_neuron_to_hoc(aligned_neuron_df, output_filepath):
    """
    Takes an aligned neuron DataFrame and writes a NEURON-compliant .hoc file.
    Extracts unbranched sections iteratively to avoid RecursionError.
    Converts coordinates from nm to um before building the output.
    """
    df = aligned_neuron_df.copy()

    # --- NEW: Call the spine labeler directly on the dataframe ---
    df = apply_spine_labels_to_df(df, spine_length_threshold_nm=3000.0)

    # --- Convert coordinates from nm to um ---
    df['x'] = df['x'] / 1000.0
    df['y'] = df['y'] / 1000.0
    df['z'] = df['z'] / 1000.0

    if 'r' not in df.columns:
        df['r'] = 0.5
    else:
        df['r'] = df['r'] / 1000.0

    # 1. Map labels to standard NEURON section arrays
    def get_hoc_type(annot):
        annot_str = str(annot).lower()
        if 'axon' in annot_str: return 'axon'
        elif 'soma' in annot_str: return 'soma'
        elif 'head' in annot_str: return 'head' # Explicitly separate heads
        elif 'neck' in annot_str: return 'neck' # Explicitly separate necks
        elif 'spine' in annot_str: return 'spine' # Fallback
        else: return 'dend'

    df['hoc_type'] = df['annotated_type'].apply(get_hoc_type)

    # 2. Build graph relationships
    children = df.groupby('p')['id'].apply(list).to_dict()
    node_data = df.set_index('id').to_dict('index')

    root_rows = df[df['p'] == -1]
    if root_rows.empty:
        print("⚠️ Cannot export to HOC: No root node found.")
        return
    root_id = root_rows.iloc[0]['id']

    section_records = []

    # 3. Iterative traversal to extract unbranched sections using a Stack
    stack = [(root_id, [], node_data[root_id]['hoc_type'], -1)]

    while stack:
        node_id, current_sec_nodes, current_type, parent_sec_id = stack.pop()
        current_sec_nodes.append(node_id)
        chs = children.get(node_id, [])

        if len(chs) == 0:
            section_records.append({'type': current_type, 'nodes': current_sec_nodes, 'parent_sec_id': parent_sec_id})

        elif len(chs) == 1:
            child = chs[0]
            child_type = node_data[child]['hoc_type']
            if child_type == current_type:
                stack.append((child, current_sec_nodes, current_type, parent_sec_id))
            else:
                sec_id = len(section_records)
                section_records.append({'type': current_type, 'nodes': current_sec_nodes, 'parent_sec_id': parent_sec_id})
                stack.append((child, [node_id], child_type, sec_id))

        else:
            sec_id = len(section_records)
            section_records.append({'type': current_type, 'nodes': current_sec_nodes, 'parent_sec_id': parent_sec_id})
            for child in chs:
                child_type = node_data[child]['hoc_type']
                stack.append((child, [node_id], child_type, sec_id))

    # 4. Count and index the section arrays
    counts = {'soma': 0, 'axon': 0, 'dend': 0, 'head': 0, 'neck': 0, 'spine': 0}
    for sec in section_records:
        sec['type_idx'] = counts[sec['type']]
        counts[sec['type']] += 1

    # 5. Write the output
    with open(output_filepath, 'w') as f:
        f.write("// NEURON HOC morphology generated from aligned data\n\n")

        for t in ['soma', 'axon', 'dend', 'head', 'neck', 'spine']:
            if counts[t] > 0:
                f.write(f"create {t}[{counts[t]}]\n")
        f.write("\n")

        for sec in section_records:
            if sec['parent_sec_id'] != -1:
                parent_sec = section_records[sec['parent_sec_id']]
                f.write(f"connect {sec['type']}[{sec['type_idx']}](0), {parent_sec['type']}[{parent_sec['type_idx']}](1)\n")
        f.write("\n")

        for sec in section_records:
            f.write(f"{sec['type']}[{sec['type_idx']}] {{\n")
            f.write("  pt3dclear()\n")

            if len(sec['nodes']) == 1:
                node = sec['nodes'][0]
                d = node_data[node]
                r = d.get('r', 0.5)
                diam = r * 2
                f.write(f"  pt3dadd({d['x']}, {d['y']}, {d['z'] - r}, {diam})\n")
                f.write(f"  pt3dadd({d['x']}, {d['y']}, {d['z'] + r}, {diam})\n")
            else:
                for node in sec['nodes']:
                    d = node_data[node]
                    diam = d.get('r', 0.5) * 2
                    f.write(f"  pt3dadd({d['x']}, {d['y']}, {d['z']}, {diam})\n")
            f.write("}\n\n")


            

def align_neurons_to_neighborhood(neuron_ids, metadata_filepath, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons',outpath = '/content/drive/MyDrive/Colab Notebooks/Aligned Neurons HOC', k_neighbors=3, show_plot=True):
    """
    Centers each neuron's soma, finds the nearest 'k' reference neurons,
    applies their average rotation, and optionally plots the transformation.
    """
    aligned_neurons = {}

    # --- 1. Load and Parse Metadata ---
    if not os.path.exists(metadata_filepath):
        print(f"⚠️ Metadata file not found: {metadata_filepath}")
        return aligned_neurons

    metadata_df = pd.read_csv(metadata_filepath)

    if isinstance(metadata_df['rotation_matrix'].iloc[0], str):
        metadata_df['rotation_matrix'] = metadata_df['rotation_matrix'].apply(ast.literal_eval)

    reference_somas = metadata_df[['soma_x', 'soma_y', 'soma_z']].values

    print(f"🔄 Aligning {len(neuron_ids)} neurons using top {k_neighbors} neighbors from {os.path.basename(metadata_filepath)}...")

    for nid in neuron_ids:
        filepath = os.path.join(input_dir, f"neuron_{nid}.csv")
        if not os.path.exists(filepath):
            print(f"⚠️ File for neuron {nid} not found. Skipping.")
            continue

        df = pd.read_csv(filepath)

        # --- 2. Find Soma and Center the Neuron ---
        root_rows = df[df['p'] == -1]
        if root_rows.empty:
            print(f"⚠️ No root node found for {nid}. Skipping.")
            continue

        soma_pos = root_rows.iloc[0][['x', 'y', 'z']].values.astype(float)

        coords = df[['x', 'y', 'z']].values
        centered_coords = coords - soma_pos

        # --- 3. Find k-Nearest Neighbors ---
        distances = np.linalg.norm(reference_somas - soma_pos, axis=1)
        k_actual = min(k_neighbors, len(metadata_df))
        nearest_indices = np.argsort(distances)[:k_actual]
        nearest_metadata = metadata_df.iloc[nearest_indices]

        # --- 4. Average the Rotations ---
        matrices = np.array(nearest_metadata['rotation_matrix'].tolist())
        rotations = R.from_matrix(matrices)
        mean_rotation = rotations.mean()
        mean_matrix = mean_rotation.as_matrix()

        # --- 5. Apply the Rotation ---
        rotated_coords = np.dot(centered_coords, mean_matrix.T)

        # Update dataframe
        df['x'] = rotated_coords[:, 0]
        df['y'] = rotated_coords[:, 1]
        df['z'] = rotated_coords[:, 2]
        aligned_neurons[nid] = df

        # Save .hoc file
        hoc_filename = os.path.join(outpath, f"neuron_{nid}_aligned.hoc")
        export_neuron_to_hoc(df, hoc_filename)



        mean_dist = np.mean(distances[nearest_indices])
        print(f"✅ Neuron {nid} aligned and saved to HOC (Mean neighbor distance: {mean_dist:.1f} nm).")

        # --- 6. Plotting ---
        if show_plot:
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                subplot_titles=(f"Pre-Rotation (Raw space)", f"Post-Rotation (Aligned space)")
            )

            # --- Left Panel: Centered Raw Skeleton ---
            df_temp = df[['id', 'p']].copy()
            df_temp['x'], df_temp['y'], df_temp['z'] = centered_coords[:, 0], centered_coords[:, 1], centered_coords[:, 2]

            node_map = df_temp.set_index('id')[['x', 'y', 'z']].to_dict('index')
            cx, cy, cz = [], [], []
            for _, row in df_temp[df_temp['p'] != -1].iterrows():
                pid = row['p']
                if pid in node_map:
                    parent = node_map[pid]
                    cx.extend([row['x'], parent['x'], None])
                    cy.extend([row['y'], parent['y'], None])
                    cz.extend([row['z'], parent['z'], None])

            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz, mode='lines',
                line=dict(color='lightgrey', width=2), opacity=0.5, name='Raw Target', hoverinfo='skip'
            ), row=1, col=1)

            # Determine a scale for the sticks based on the target neuron's size
            max_dist = np.max(np.linalg.norm(centered_coords, axis=1)) if len(centered_coords) > 0 else 50000
            stick_scale = max_dist * 0.4

            # Plot Neighbors in Left and Right Panels
            for i, (_, neighbor) in enumerate(nearest_metadata.iterrows()):
                # Raw Space Math
                n_soma_orig = np.array([neighbor['soma_x'], neighbor['soma_y'], neighbor['soma_z']])
                n_soma_rel = n_soma_orig - soma_pos # Position relative to target soma

                v_com_orig = np.array([neighbor['v_com_x'], neighbor['v_com_y'], neighbor['v_com_z']])
                v_com_orig = v_com_orig / np.linalg.norm(v_com_orig)
                stick_end_raw = n_soma_rel + v_com_orig * stick_scale

                # Aligned Space Math (Rotate neighbor positions and vectors)
                n_soma_rot = np.dot(n_soma_rel, mean_matrix.T)
                v_com_rot = np.dot(v_com_orig, mean_matrix.T)
                stick_end_rot = n_soma_rot + v_com_rot * stick_scale

                # Draw Left Panel (Raw Neighbors)
                fig.add_trace(go.Scatter3d(
                    x=[n_soma_rel[0]], y=[n_soma_rel[1]], z=[n_soma_rel[2]],
                    mode='markers', marker=dict(size=5, color='crimson'),
                    name=f'N{i+1} Soma (Raw)'
                ), row=1, col=1)
                fig.add_trace(go.Scatter3d(
                    x=[n_soma_rel[0], stick_end_raw[0]], y=[n_soma_rel[1], stick_end_raw[1]], z=[n_soma_rel[2], stick_end_raw[2]],
                    mode='lines', line=dict(color='crimson', width=4), showlegend=False
                ), row=1, col=1)

                # Draw Right Panel (Rotated Neighbors)
                fig.add_trace(go.Scatter3d(
                    x=[n_soma_rot[0]], y=[n_soma_rot[1]], z=[n_soma_rot[2]],
                    mode='markers', marker=dict(size=5, color='darkorange'),
                    name=f'N{i+1} Soma (Rotated)'
                ), row=1, col=2)
                fig.add_trace(go.Scatter3d(
                    x=[n_soma_rot[0], stick_end_rot[0]], y=[n_soma_rot[1], stick_end_rot[1]], z=[n_soma_rot[2], stick_end_rot[2]],
                    mode='lines', line=dict(color='darkorange', width=4), showlegend=False
                ), row=1, col=2)

            # --- Right Panel: Aligned Skeleton ---
            df_temp['x'], df_temp['y'], df_temp['z'] = rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2]

            node_map_rot = df_temp.set_index('id')[['x', 'y', 'z']].to_dict('index')
            rx, ry, rz = [], [], []
            for _, row in df_temp[df_temp['p'] != -1].iterrows():
                pid = row['p']
                if pid in node_map_rot:
                    parent = node_map_rot[pid]
                    rx.extend([row['x'], parent['x'], None])
                    ry.extend([row['y'], parent['y'], None])
                    rz.extend([row['z'], parent['z'], None])

            fig.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz, mode='lines',
                line=dict(color='royalblue', width=2), opacity=0.6, name='Aligned Target', hoverinfo='skip'
            ), row=1, col=2)

            # Layout Syncing
            fig.update_layout(
                title=f"Neuron {nid} Alignment (Mean rotation of {k_actual} nearest cells)",
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data', bgcolor='white'),
                scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data', bgcolor='white'),
                width=1400, height=700, margin=dict(l=0, r=0, b=0, t=50)
            )
            fig.show()

    return aligned_neurons
