import os
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R # Added missing import



def calculate_dendrite_xy_anisotropy(neuron_ids, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons', show_plot=True):
    """
    Reads neuron skeletons, filters for dendritic nodes, calculates the 2D
    Fractional Anisotropy (FA) in the xy-plane using PCA, and plots the results.
    """
    results = {}

    print(f"📊 Calculating Dendritic XY-Plane Anisotropy for {len(neuron_ids)} neurons...")

    for nid in neuron_ids:
        # --- 1. Import Data ---
        filepath = os.path.join(input_dir, f"neuron_{nid}.csv")

        if not os.path.exists(filepath):
            print(f"⚠️ File for neuron {nid} not found. Skipping.")
            continue

        df = pd.read_csv(filepath)

        if not {'x', 'y'}.issubset(df.columns):
            print(f"⚠️ Missing 'x' or 'y' columns in neuron {nid}. Skipping.")
            continue

        # --- 2. Filter for Dendrites ---
        if 'annotated_type' not in df.columns:
            print(f"⚠️ 'annotated_type' missing for {nid}. Skipping.")
            continue

        # Use the regex from your alignment function to capture dendrites/apical nodes
        dendrite_mask = df['annotated_type'].astype(str).str.contains('dendrite|apical|^1$', case=False, regex=True)
        dendrite_df = df[dendrite_mask]

        if dendrite_df.empty:
            print(f"⚠️ No dendritic points found for {nid}. Skipping.")
            continue

        # --- 3. Extract and Mean-Center XY Coordinates (Dendrites Only) ---
        coords_xy = dendrite_df[['x', 'y']].values
        center = np.mean(coords_xy, axis=0)
        centered_coords = coords_xy - center

        # --- 4. Calculate the Covariance Matrix ---
        cov_matrix = np.cov(centered_coords, rowvar=False)

        # --- 5. Calculate Eigenvalues and Eigenvectors ---
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        l1, l2 = eigenvalues[0], eigenvalues[1]

        # --- 6. Calculate 2D Fractional Anisotropy (FA) ---
        if (l1**2 + l2**2) == 0:
            fa_2d = 0.0
        else:
            fa_2d = np.abs(l1 - l2) / np.sqrt(l1**2 + l2**2)

        results[nid] = {
            'FA_2D': fa_2d,
            'lambda_1': l1,
            'lambda_2': l2,
            'center_x': center[0],
            'center_y': center[1]
        }

        print(f"🔬 Neuron {nid} | Dendritic 2D FA: {fa_2d:.4f} (λ1: {l1:.1f}, λ2: {l2:.1f})")

        # --- 7. Plotting ---
        if show_plot:
            fig = go.Figure()

            # Plot the FULL neuron in the background (light grey)
            fig.add_trace(go.Scatter(
                x=df['x'], y=df['y'], mode='markers',
                marker=dict(size=2, color='lightgrey', opacity=0.3),
                name='All Nodes (Axon/Soma/etc.)',
                hoverinfo='skip'
            ))

            # Plot the DENDRITES used for PCA (teal)
            fig.add_trace(go.Scatter(
                x=dendrite_df['x'], y=dendrite_df['y'], mode='markers',
                marker=dict(size=3, color='teal', opacity=0.7),
                name='Dendritic Nodes (PCA Input)'
            ))

            # Scale vectors by 2 standard deviations for clear visualization
            scale_l1 = np.sqrt(l1) * 2
            scale_l2 = np.sqrt(l2) * 2

            v1 = eigenvectors[:, 0] * scale_l1
            v2 = eigenvectors[:, 1] * scale_l2

            # Draw Primary Axis (Red)
            fig.add_trace(go.Scatter(
                x=[center[0], center[0] + v1[0], None, center[0], center[0] - v1[0]],
                y=[center[1], center[1] + v1[1], None, center[1], center[1] - v1[1]],
                mode='lines+markers', line=dict(color='crimson', width=4),
                name=f'Principal Axis 1 (λ1)'
            ))

            # Draw Secondary Axis (Blue)
            fig.add_trace(go.Scatter(
                x=[center[0], center[0] + v2[0], None, center[0], center[0] - v2[0]],
                y=[center[1], center[1] + v2[1], None, center[1], center[1] - v2[1]],
                mode='lines+markers', line=dict(color='royalblue', width=4),
                name='Principal Axis 2 (λ2)'
            ))

            fig.update_layout(
                title=f"Neuron {nid} Dendritic XY-Plane PCA (Anisotropy = {fa_2d:.4f})",
                xaxis_title="X", yaxis_title="Y",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                template="plotly_white",
                height=700, width=700
            )
            fig.show()

    return pd.DataFrame.from_dict(results, orient='index')






import os
import gc
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_z_alignment_math(neuron_ids, input_dir, percentile=95, show_plot=False):
    """
    Locates the soma, isolates the distal dendritic tail, calculates the CoM vector,
    and computes the necessary rotation matrix to align that vector to the Z-axis.
    Does NOT modify or save the original neuron data.
    If show_plot is True, displays the original and rotated skeleton side-by-side.
    """
    metadata_records = []

    print(f"📐 Calculating rotation matrices for {len(neuron_ids)} neurons...")

    for nid in neuron_ids:
        filepath = os.path.join(input_dir, f"neuron_{nid}.csv")
        if not os.path.exists(filepath):
            continue

        df = pd.read_csv(filepath)

        # --- 1. Capture Original Soma ---
        root_rows = df[df['p'] == -1]
        if root_rows.empty:
            del df
            continue

        soma_x, soma_y, soma_z = root_rows.iloc[0][['x', 'y', 'z']]
        soma_pos = np.array([soma_x, soma_y, soma_z])

        # --- 2. Calculate Distances (Relative to Soma) ---
        coords = df[['x', 'y', 'z']].values
        relative_coords = coords - soma_pos
        distances = np.linalg.norm(relative_coords, axis=1)

        distal_threshold = np.percentile(distances, percentile)
        distal_mask = distances >= distal_threshold

        # --- 3. Filter for Dendrites ---
        if 'annotated_type' not in df.columns:
            del df
            continue

        dendrite_mask = df['annotated_type'].astype(str).str.contains('dendrite|apical|^1$', case=False, regex=True)
        candidate_nodes = set(df[distal_mask & dendrite_mask]['id'].values)

        if not candidate_nodes:
            del df
            continue

        # --- 4. Reconstruct Hierarchy (NetworkX) ---
        G = nx.DiGraph()
        G.add_nodes_from(df['id'])
        valid_edges = df[df['p'] != -1][['p', 'id']].values
        G.add_edges_from(valid_edges)

        soma_id = root_rows.iloc[0]['id']
        primary_branches = list(G.successors(soma_id))
        largest_pool = []

        for pb in primary_branches:
            try:
                arbor_nodes = set(nx.descendants(G, pb))
                arbor_nodes.add(pb)
                pool = candidate_nodes.intersection(arbor_nodes)
                if len(pool) > len(largest_pool):
                    largest_pool = list(pool)
            except nx.NetworkXError:
                continue

        if len(largest_pool) < 5:
            del df, G
            continue

        # --- 5. Vector Calculation ---
        pool_df = df[df['id'].isin(largest_pool)]
        distal_coords = pool_df[['x', 'y', 'z']].values

        distal_center = np.mean(distal_coords, axis=0)
        v_com = distal_center - soma_pos
        v_com = v_com / np.linalg.norm(v_com) # Normalize to unit vector

        # --- 6. Calculate the Rotation Matrix ---
        z_axis = np.array([0.0, 0.0, 1.0])
        cross_prod = np.cross(v_com, z_axis)
        dot_prod = np.dot(v_com, z_axis)

        if np.linalg.norm(cross_prod) < 1e-6:
            rot_matrix = np.eye(3)
        else:
            angle = np.arccos(np.clip(dot_prod, -1.0, 1.0))
            rot_vec = (cross_prod / np.linalg.norm(cross_prod)) * angle
            rotation = R.from_rotvec(rot_vec)
            rot_matrix = rotation.as_matrix()

        # --- 7. Plotting the Skeleton and Vectors ---
        if show_plot:
            rotated_v_com = rot_matrix.dot(v_com)

            # Scale vectors so they reach the edge of the neuron for visibility
            scale_factor = np.max(distances) * 0.95
            scaled_v_com = v_com * scale_factor
            scaled_rot_v_com = rotated_v_com * scale_factor

            # Calculate rotated skeleton coordinates for the plot
            rotated_coords = np.dot(relative_coords, rot_matrix.T)

            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                subplot_titles=("Original Orientation (Centered)", "Rotated to Z-Axis")
            )

            # --- Left Panel: Original Skeleton ---
            df_temp = df[['id', 'p']].copy()
            df_temp['x'] = relative_coords[:, 0]
            df_temp['y'] = relative_coords[:, 1]
            df_temp['z'] = relative_coords[:, 2]

            node_map = df_temp.set_index('id')[['x', 'y', 'z']].to_dict('index')
            ox, oy, oz = [], [], []
            for _, row in df_temp[df_temp['p'] != -1].iterrows():
                pid = row['p']
                if pid in node_map:
                    parent = node_map[pid]
                    ox.extend([row['x'], parent['x'], None])
                    oy.extend([row['y'], parent['y'], None])
                    oz.extend([row['z'], parent['z'], None])

            fig.add_trace(go.Scatter3d(
                x=ox, y=oy, z=oz, mode='lines',
                line=dict(color='lightgrey', width=2), opacity=0.5, name='Original Skeleton', hoverinfo='skip'
            ), row=1, col=1)

            fig.add_trace(go.Scatter3d(
                x=[0, scaled_v_com[0]], y=[0, scaled_v_com[1]], z=[0, scaled_v_com[2]],
                mode='lines+markers', line=dict(color='green', width=6),
                marker=dict(size=6, symbol='diamond'), name='Original CoM Vector'
            ), row=1, col=1)

            # --- Right Panel: Rotated Skeleton ---
            df_temp['x'] = rotated_coords[:, 0]
            df_temp['y'] = rotated_coords[:, 1]
            df_temp['z'] = rotated_coords[:, 2]

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
                line=dict(color='royalblue', width=2), opacity=0.6, name='Rotated Skeleton', hoverinfo='skip'
            ), row=1, col=2)

            fig.add_trace(go.Scatter3d(
                x=[0, scaled_rot_v_com[0]], y=[0, scaled_rot_v_com[1]], z=[0, scaled_rot_v_com[2]],
                mode='lines+markers', line=dict(color='blue', width=6),
                marker=dict(size=6, symbol='circle'), name='Aligned Vector'
            ), row=1, col=2)

            # Sync layout and aspect ratio
            fig.update_layout(
                title=f"Neuron {nid} Alignment Verification",
                scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data', bgcolor='white'),
                scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data', bgcolor='white'),
                width=1400, height=700, margin=dict(l=0, r=0, b=0, t=50)
            )
            fig.show()

            # Clean up plotting variables
            del df_temp, rotated_coords, node_map, node_map_rot, ox, oy, oz, rx, ry, rz
            if 'fig' in locals():
                del fig

        # --- 8. Store Metadata ---
        metadata_records.append({
            'neuron_id': nid,
            'soma_x': soma_x,
            'soma_y': soma_y,
            'soma_z': soma_z,
            'v_com_x': v_com[0],
            'v_com_y': v_com[1],
            'v_com_z': v_com[2],
            'rotation_matrix': rot_matrix.tolist()
        })

        print(f"✅ Neuron {nid} math calculated.")

        # --- 9. Aggressive Garbage Collection ---
        del df, G, pool_df, coords, relative_coords
        gc.collect()

    return pd.DataFrame(metadata_records)




import pandas as pd

import os
import gc
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.transform import Rotation as R

def extract_alignment_metadata(neuron_ids, fa_threshold, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons', percentile=90, show_plot=False):
    """
    1. Calculates Fractional Anisotropy (FA).
    2. Filters neurons > fa_threshold.
    3. Calculates soma positions and Z-axis rotation matrices.
    4. Returns a metadata DataFrame containing these values.
    """
    print(f"🚀 Starting metadata extraction for {len(neuron_ids)} neurons (FA > {fa_threshold})...")

    # Step 1: Calculate FA
    fa_df = calculate_dendrite_xy_anisotropy(neuron_ids, input_dir=input_dir, show_plot=False)
    if fa_df.empty:
        print("⚠️ No FA data could be calculated. Aborting.")
        return pd.DataFrame()

    # Step 2: Filter neurons
    passed_neurons = fa_df[fa_df['FA_2D'] > fa_threshold].index.tolist()
    print(f"\n✅ {len(passed_neurons)} neurons passed the FA > {fa_threshold} threshold.")

    if not passed_neurons:
        print("🛑 No neurons passed. Stopping pipeline.")
        return pd.DataFrame()

    # Step 3: Extract Spatial Metadata (RAM Safe)
    metadata_df = calculate_z_alignment_math(
        passed_neurons,
        input_dir=input_dir,
        percentile=percentile,
        show_plot=show_plot
    )

    # Optional: Merge the FA data into the final metadata output for a complete summary
    final_df = metadata_df.merge(fa_df[['FA_2D']], left_on='neuron_id', right_index=True, how='left')

    return final_df
