import os
import ast
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def calculate_spanning_probabilities_jitter(neuron_ids, metadata_filepath, input_dir, voxel_size=10.0, grid_extent=1000.0, k_neighbors=3):
    """
    Calculates the spatial probability (0.0 to 1.0) that a given voxel contains 
    axonal or dendritic arbors, utilizing a geometric 'jitter' to account for segment radius.
    Includes dynamic radial interpolation if the 25th percentile of radii exceeds voxel size.
    """
    if not os.path.exists(metadata_filepath):
        print(f"⚠️ Metadata file not found: {metadata_filepath}")
        return None, None

    # 1. Load Reference Metadata for Rototranslation
    metadata_df = pd.read_csv(metadata_filepath)
    if isinstance(metadata_df['rotation_matrix'].iloc[0], str):
        metadata_df['rotation_matrix'] = metadata_df['rotation_matrix'].apply(ast.literal_eval)
    reference_somas = metadata_df[['soma_x', 'soma_y', 'soma_z']].values

    # 2. Define the 3D Voxel Grid
    bins_x = np.arange(-grid_extent, grid_extent + voxel_size, voxel_size)
    bins_y = np.arange(-grid_extent, grid_extent + voxel_size, voxel_size)
    bins_z = np.arange(-grid_extent, grid_extent + voxel_size, voxel_size)
    
    num_bins = (len(bins_x)-1, len(bins_y)-1, len(bins_z)-1)
    
    axon_presence_total = np.zeros(num_bins, dtype=np.float32)
    dend_presence_total = np.zeros(num_bins, dtype=np.float32)

    valid_neurons_processed = 0
    print(f"🔄 Calculating spatial probabilities (Jitter Method) for {len(neuron_ids)} neurons...")

    # 3. Process sequentially
    for nid in neuron_ids:
        print(f'Processing nid: {nid}')
        filepath = os.path.join(input_dir, f"neuron_{nid}.csv")
        if not os.path.exists(filepath):
            continue

        df = pd.read_csv(filepath)
        root_rows = df[df['p'] == -1]
        if root_rows.empty:
            continue
            
        # --- ALIGNMENT ---
        soma_pos = root_rows.iloc[0][['x', 'y', 'z']].values.astype(float)
        centered_coords = df[['x', 'y', 'z']].values - soma_pos

        distances = np.linalg.norm(reference_somas - soma_pos, axis=1)
        k_actual = min(k_neighbors, len(metadata_df))
        nearest_indices = np.argsort(distances)[:k_actual]
        nearest_metadata = metadata_df.iloc[nearest_indices]

        matrices = np.array(nearest_metadata['rotation_matrix'].tolist())
        mean_matrix = R.from_matrix(matrices).mean().as_matrix()

        rotated_coords = np.dot(centered_coords, mean_matrix.T)

        df['x'] = rotated_coords[:, 0] / 1000.0
        df['y'] = rotated_coords[:, 1] / 1000.0
        df['z'] = rotated_coords[:, 2] / 1000.0
        
        if 'r' not in df.columns:
            df['r'] = 0.5 / 1000.0 
        else:
            df['r'] = df['r'] / 1000.0
        
        df['annotated_type'] = df['annotated_type'].astype(str).str.lower()
        df['is_axon'] = df['annotated_type'].str.contains('axon')
        df['is_soma'] = df['annotated_type'].str.contains('soma')
        df['is_dend'] = ~(df['is_axon'] | df['is_soma'])

        # --- VECTORIZED CONTINUOUS SEGMENT RASTERIZATION ---
        parents = df[['id', 'x', 'y', 'z']].rename(columns={'id': 'p', 'x': 'px', 'y': 'py', 'z': 'pz'})
        segments = df.merge(parents, on='p', how='inner')

        starts = segments[['px', 'py', 'pz']].values
        ends = segments[['x', 'y', 'z']].values
        radii = segments['r'].values
        is_axon = segments['is_axon'].values
        is_dend = segments['is_dend'].values
        
        lengths = np.linalg.norm(ends - starts, axis=1)
        step_size = voxel_size / 3.0 
        num_points = np.maximum(1, np.ceil(lengths / step_size)).astype(int)

        seg_idx = np.repeat(np.arange(len(num_points)), num_points)
        t_list = [np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5]) for n in num_points]
        t_vals = np.concatenate(t_list)[:, None]

        expanded_starts = starts[seg_idx]
        expanded_ends = ends[seg_idx]
        all_pts = expanded_starts + (expanded_ends - expanded_starts) * t_vals

        expanded_radii = np.repeat(radii, num_points)
        expanded_is_axon = np.repeat(is_axon, num_points)
        expanded_is_dend = np.repeat(is_dend, num_points)

        # --- NEW: CONDITIONAL JITTER EXPANSION ---
        rad_q25 = np.percentile(radii, 25)
        
        if rad_q25 > voxel_size:
            # Interpolate intermediate points along the radius
            max_r = np.max(radii)
            num_rad_steps = np.maximum(1, np.ceil(max_r / step_size)).astype(int)
            
            # Create fractional multipliers from >0 up to 1.0
            rad_fractions = np.linspace(0.0, 1.0, num_rad_steps + 1)[1:] 
            
            cardinals = np.array([
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1]
            ], dtype=np.float32)
            
            # Broadcast to create intermediate points for all 6 directions
            extended_stencil = (cardinals[None, :, :] * rad_fractions[:, None, None]).reshape(-1, 3)
            
            # Add the central point (0,0,0)
            dynamic_stencil = np.vstack([np.array([[0, 0, 0]], dtype=np.float32), extended_stencil])
        else:
            # Standard 7-point stencil
            dynamic_stencil = np.array([
                [0, 0, 0],
                [1, 0, 0], [-1, 0, 0],
                [0, 1, 0], [0, -1, 0],
                [0, 0, 1], [0, 0, -1]
            ], dtype=np.float32)

        num_stencil_pts = len(dynamic_stencil)

        # Broadcast the dynamic stencil to all interpolated points
        jittered_pts = all_pts[:, None, :] + (dynamic_stencil[None, :, :] * expanded_radii[:, None, None])
        jittered_pts = jittered_pts.reshape(-1, 3)
        
        # Expand our classification masks to match the dynamic point count
        jittered_is_axon = np.repeat(expanded_is_axon, num_stencil_pts)
        jittered_is_dend = np.repeat(expanded_is_dend, num_stencil_pts)

        # --- BINARY HISTOGRAM RASTERIZATION ---
        # 1. Evaluate Axons for THIS neuron
        axon_pts = jittered_pts[jittered_is_axon]
        if len(axon_pts) > 0:
            neuron_axon_hist, _ = np.histogramdd(axon_pts, bins=(bins_x, bins_y, bins_z))
            neuron_axon_mask = (neuron_axon_hist > 0).astype(np.float32)
            axon_presence_total += neuron_axon_mask
            
        # 2. Evaluate Dendrites for THIS neuron
        dend_pts = jittered_pts[jittered_is_dend]
        if len(dend_pts) > 0:
            neuron_dend_hist, _ = np.histogramdd(dend_pts, bins=(bins_x, bins_y, bins_z))
            neuron_dend_mask = (neuron_dend_hist > 0).astype(np.float32)
            dend_presence_total += neuron_dend_mask
        
        valid_neurons_processed += 1
        
        del df, parents, segments, all_pts, expanded_starts, expanded_ends
        del jittered_pts, axon_pts, dend_pts
        
    print(f"✅ Finished processing {valid_neurons_processed} neurons.")

    if valid_neurons_processed == 0:
        return None, None

    # 4. Final Probability Calculation
    axon_probability_field = axon_presence_total / valid_neurons_processed
    dend_probability_field = dend_presence_total / valid_neurons_processed

    return axon_probability_field, dend_probability_field

