import os
import re
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from collections import defaultdict
from scipy.interpolate import splprep, splev, interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def label_dendritic_spines_robust(neuron_ids, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons', output_dir=None, spine_length_threshold_nm=5000.0, num_interp_points=100, smoothing_sigma=2.0):
    """
    Identifies dendritic spines by evaluating entire terminal subtrees,
    then evaluates their morphological radius profiles to explicitly 
    label discrete nodes as 'head' or 'neck'.
    """
    updated_neurons = {}

    print(f"🔍 Searching and quantifying dendritic spines across {len(neuron_ids)} neurons (Threshold: {spine_length_threshold_nm} nm)...")

    for nid in neuron_ids:
        filepath = os.path.join(input_dir, f"neuron_{nid}.csv")

        if not os.path.exists(filepath):
            print(f"⚠️ File for neuron {nid} not found. Skipping.")
            continue

        df = pd.read_csv(filepath)

        if not {'id', 'p', 'x', 'y', 'z', 'annotated_type'}.issubset(df.columns):
            print(f"⚠️ Missing required columns in neuron {nid}. Skipping.")
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
        print(f"✅ Neuron {nid}: Identified and relabeled {len(spine_nodes)} spine nodes into Heads/Necks.")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_filepath = os.path.join(output_dir, f"neuron_{nid}_spines.csv")
            df.to_csv(out_filepath, index=False)

    return updated_neurons


def plot_color_coded_neurons(neuron_dict, neuron_ids):
    """
    Plots the 3D skeleton of neurons, color-coding sections by their 'annotated_type'.
    """
    # Updated color palette to include Head and Neck
    color_palette = {
        'soma': 'black',
        'axon': 'royalblue',
        'dendrite': 'crimson',
        'apical': 'mediumorchid',
        'head': 'gold',
        'neck': 'indigo',
        'spine': 'springgreen', # Fallback
        'unclassified': 'lightgrey'
    }

    print(f"🎨 Plotting {len(neuron_ids)} color-coded neurons...")

    for nid in neuron_ids:
        if nid not in neuron_dict:
            print(f"⚠️ Neuron {nid} not found in the provided dictionary. Skipping.")
            continue

        df = neuron_dict[nid]

        if not {'id', 'p', 'x', 'y', 'z', 'annotated_type'}.issubset(df.columns):
            print(f"⚠️ Missing required columns in neuron {nid}. Skipping.")
            continue

        fig = go.Figure()
        node_map = df.set_index('id').to_dict('index')

        traces_data = {key: {'x': [], 'y': [], 'z': []} for key in color_palette.keys()}

        # 1. Build line segments
        for _, row in df[df['p'] != -1].iterrows():
            pid = row['p']
            if pid in node_map:
                parent = node_map[pid]

                raw_type = str(row['annotated_type']).lower()
                assigned_type = 'unclassified'

                for key in color_palette.keys():
                    if key in raw_type:
                        assigned_type = key
                        break

                traces_data[assigned_type]['x'].extend([row['x'], parent['x'], None])
                traces_data[assigned_type]['y'].extend([row['y'], parent['y'], None])
                traces_data[assigned_type]['z'].extend([row['z'], parent['z'], None])

        # 2. Add traces to the figure
        for struct_type, coords in traces_data.items():
            if len(coords['x']) > 0:
                # Highlight spine components by making them thicker
                line_width = 4 if struct_type in ['spine', 'head', 'neck'] else 2

                fig.add_trace(go.Scatter3d(
                    x=coords['x'], y=coords['y'], z=coords['z'],
                    mode='lines',
                    line=dict(color=color_palette[struct_type], width=line_width),
                    name=struct_type.capitalize(),
                    hoverinfo='skip'
                ))

        # 3. Highlight the Soma specifically as a marker
        soma_nodes = df[df['annotated_type'].astype(str).str.contains('soma', case=False, na=False)]
        if not soma_nodes.empty:
            fig.add_trace(go.Scatter3d(
                x=soma_nodes['x'], y=soma_nodes['y'], z=soma_nodes['z'],
                mode='markers',
                marker=dict(size=6, color=color_palette['soma'], symbol='circle'),
                name='Soma Nodes'
            ))

        # 4. Layout configuration
        fig.update_layout(
            title=f"Neuron {nid} - Morphological Compartments",
            scene=dict(
                xaxis_title='X (nm)',
                yaxis_title='Y (nm)',
                zaxis_title='Z (nm)',
                aspectmode='data',
                bgcolor='white'
            ),
            height=800, width=1000,
            margin=dict(l=0, r=0, b=0, t=50),
            legend=dict(itemsizing='constant')
        )

        fig.show()

# Example Usage:
# Assuming `updated_neurons_dict` is the output from the `label_dendritic_spines` function
target_ids = [5629504348]
spine_df_dict = label_dendritic_spines_robust(target_ids, spine_length_threshold_nm=5000.0)
plot_color_coded_neurons(spine_df_dict, target_ids)
