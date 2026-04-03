import os
import numpy as np

def calculate_synaptic_overlap_locations(
    pre_mtype,
    post_mtype,
    pre_soma_pos,
    post_soma_pos,
    density_maps_dir,
    num_synapses, # Included for signature completeness, used in the next step
    voxel_size=10.0,
    grid_extent=1000.0
):
    """
    Loads pre-computed spatial probability maps for a pre- and post-synaptic neuron pair,
    shifts them based on their relative soma distances, and calculates the joint probability
    of synaptic connection per overlapping voxel.
    """
    # 1. Construct file paths (assuming the naming convention from previous steps)
    pre_axon_path = os.path.join(density_maps_dir, f"{pre_mtype}_axon.npy")
    post_dend_path = os.path.join(density_maps_dir, f"{post_mtype}_dend.npy")

    if not (os.path.exists(pre_axon_path) and os.path.exists(post_dend_path)):
        print("⚠️ Density maps not found for the specified mtypes.")
        return None, None

    # 2. Load the 3D probability arrays
    pre_grid = np.load(pre_axon_path)   # Axonal field of pre-synaptic neuron
    post_grid = np.load(post_dend_path) # Dendritic field of post-synaptic neuron

    # 3. Calculate the relative shift in voxel indices
    pre_soma_pos = np.array(pre_soma_pos, dtype=float)
    post_soma_pos = np.array(post_soma_pos, dtype=float)

    # How many voxels the pre-synaptic grid needs to shift to align with the post-synaptic grid
    physical_shift = pre_soma_pos - post_soma_pos
    voxel_shift = np.round(physical_shift / voxel_size).astype(int)

    N = pre_grid.shape[0] # Assuming cubic grids of shape (N, N, N)

    # 4. Calculate bounding box overlaps using NumPy slicing
    slices_post = []
    slices_pre = []

    for dim_shift in voxel_shift:
        # If pre is shifted right by +5, post overlap starts at 5, pre overlap starts at 0
        start_post = max(0, dim_shift)
        end_post = min(N, N + dim_shift)

        start_pre = max(0, -dim_shift)
        end_pre = min(N, N - dim_shift)

        # If the shift is larger than the grid itself, there is zero overlap
        if start_post >= N or start_pre >= N or end_post <= 0 or end_pre <= 0:
            print("ℹ️ Neurons are too far apart; bounding boxes do not overlap.")
            return np.array([]), np.array([])

        slices_post.append(slice(start_post, end_post))
        slices_pre.append(slice(start_pre, end_pre))

    # 5. Extract the overlapping sub-grids and multiply (Joint Probability)
    # P_joint = P_axon * P_dend
    overlap_post = post_grid[tuple(slices_post)]
    overlap_pre = pre_grid[tuple(slices_pre)]

    joint_probability_grid = overlap_post * overlap_pre

    # 6. Find all voxels where the joint probability > 0
    valid_indices = np.where(joint_probability_grid > 0)

    if len(valid_indices[0]) == 0:
        return np.array([]), np.array([])

    probabilities = joint_probability_grid[valid_indices]

    # 7. Convert array indices back to Global Physical Coordinates
    # First, map the sub-grid indices back to the post-synaptic grid's local indices
    local_idx_x = valid_indices[0] + slices_post[0].start
    local_idx_y = valid_indices[1] + slices_post[1].start
    local_idx_z = valid_indices[2] + slices_post[2].start

    # The center of the grid (index N//2) corresponds to the soma position
    center_idx = N // 2

    # Convert local indices to local physical distance from the post-soma
    local_x = (local_idx_x - center_idx) * voxel_size
    local_y = (local_idx_y - center_idx) * voxel_size
    local_z = (local_idx_z - center_idx) * voxel_size

    # Translate local physical distances to global space by adding the post-soma coordinates
    global_x = post_soma_pos[0] + local_x
    global_y = post_soma_pos[1] + local_y
    global_z = post_soma_pos[2] + local_z

    centroids = np.vstack((global_x, global_y, global_z)).T

    # Normalize probabilities so they sum to 1.0 (creating a proper PDF for distributing synapses)
    normalized_probabilities = probabilities / np.sum(probabilities)

    return centroids, normalized_probabilities

def distribute_synapses_probabilistically(overlap_centroids, normalized_probabilities, num_synapses):
    """
    Distributes a specified number of synapses across the overlapping voxels
    based on the joint probability mass function.

    Returns an array of 3D coordinates representing the 'target' voxels for each synapse.
    """
    print(f"🎲 Distributing {num_synapses} synapses probabilistically...")

    # 1. Edge case handling
    if overlap_centroids is None or len(overlap_centroids) == 0:
        print("⚠️ No overlapping voxels available to place synapses.")
        return np.array([])

    if num_synapses <= 0:
        print("⚠️ Number of synapses to place must be greater than zero.")
        return np.array([])

    # 2. Setup indices for sampling
    # We sample the indices rather than the 3D coordinates directly to keep the math 1D
    num_valid_voxels = len(overlap_centroids)
    voxel_indices = np.arange(num_valid_voxels)

    # 3. Probabilistic Sampling
    # replace=True is biologically crucial here: a highly probable (dense) voxel
    # might receive multiple synapses from the same connection.
    chosen_indices = np.random.choice(
        voxel_indices,
        size=num_synapses,
        p=normalized_probabilities,
        replace=False
    )

    # 4. Map the sampled indices back to their 3D physical coordinates
    chosen_centroids = overlap_centroids[chosen_indices]

    # Optional: Log the distribution spread for debugging
    unique_indices, counts = np.unique(chosen_indices, return_counts=True)
    print(f"✅ Placed {num_synapses} synapses across {len(unique_indices)} unique voxels.")

    # Optional: print the highest multi-synapse count in a single voxel
    if len(counts) > 0:
        max_in_one_voxel = np.max(counts)
        if max_in_one_voxel > 1:
            print(f"   -> Maximum synapses dropped into a single voxel: {max_in_one_voxel}")

    return chosen_centroids

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_joint_spanning_pmf(overlap_centroids, normalized_probabilities, voxel_size):
    """
    Plots 2D Maximum Intensity Projections (XY, XZ, YZ) of the abstract
    Joint Probability Mass Function (PMF) defined by the overlapping voxels.
    """
    print("🎨 Generating abstract PMF visualization...")
    if overlap_centroids is None or len(overlap_centroids) == 0:
        print("⚠️ No overlap data to visualize.")
        return

    # 1. Coordinate Setup: Create local coords from the centroid list
    # Find min/max bounds of the overlap zone only
    c_min = np.min(overlap_centroids, axis=0)
    c_max = np.max(overlap_centroids, axis=0)

    # Define local 2D extents [xmin, xmax, ymin, ymax] for imshow
    extent_xy = [c_min[0], c_max[0], c_min[1], c_max[1]]
    extent_xz = [c_min[0], c_max[0], c_min[2], c_max[2]]
    extent_yz = [c_min[1], c_max[1], c_min[2], c_max[2]]

    # 2. Setup Figure (3 panels: standard orthogonal views)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Joint Probability Mass Function (Abstract Spanning Tree Interaction)", fontsize=16)

    # Use a scatter plot with square markers to simulate a voxel heatmap.
    # We use global coordinates, but tight limits, to visualize the field shape.
    # Color represents the normalized probability (PMF weight).
    marker_size = (voxel_size * 2) # Arbitrary scaling for visualization density

    # Panel 1: XY (Top-Down)
    sc1 = ax1.scatter(overlap_centroids[:, 0], overlap_centroids[:, 1],
                      c=normalized_probabilities, cmap='plasma', s=marker_size, marker='s')
    ax1.set_title("XY Plane (Abstract PMF Shape)")
    ax1.set_xlabel("X (μm)")
    ax1.set_ylabel("Y (μm)")

    # Panel 2: XZ (Side View)
    sc2 = ax2.scatter(overlap_centroids[:, 0], overlap_centroids[:, 2],
                      c=normalized_probabilities, cmap='plasma', s=marker_size, marker='s')
    ax2.set_title("XZ Plane (Abstract PMF Shape)")
    ax2.set_xlabel("X (μm)")
    ax2.set_ylabel("Z (μm)")

    # Panel 3: YZ (End View)
    sc3 = ax3.scatter(overlap_centroids[:, 1], overlap_centroids[:, 2],
                      c=normalized_probabilities, cmap='plasma', s=marker_size, marker='s')
    ax3.set_title("YZ Plane (Abstract PMF Shape)")
    ax3.set_xlabel("Y (μm)")
    ax3.set_ylabel("Z (μm)")

    # Add one common colorbar
    cbar = fig.colorbar(sc1, ax=[ax1, ax2, ax3], orientation='vertical', fraction=0.015, pad=0.04)
    cbar.set_label('Normalized Probability Weight (PMF)')

    plt.show()


import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree



def snap_to_closest_actual_synapses(post_neuron_id, synapses_dir, chosen_centroids, post_soma_pos, synapse_type='exc'):
    """
    Takes probabilistically chosen spatial centroids (in global space), translates them
    to local space, and maps each one to the nearest actual synapse on the post-synaptic tree.
    """
    print(f"🎯 Snapping {len(chosen_centroids)} target centroids to closest '{synapse_type}' synapses on Neuron {post_neuron_id}...")

    # 1. Load the pre-mapped synapse database for the post-synaptic neuron
    syn_filepath = os.path.join(synapses_dir, f"neuron_{post_neuron_id}_mapped_synapses.csv")
    if not os.path.exists(syn_filepath):
        print(f"⚠️ Synapse mapping file not found: {syn_filepath}")
        return None

    mapped_df = pd.read_csv(syn_filepath)

    # 2. Filter for the correct biological type
    valid_synapses = mapped_df[mapped_df['synapse_type'] == synapse_type].copy()

    if valid_synapses.empty:
        print(f"⚠️ No '{synapse_type}' synapses found on post-synaptic neuron {post_neuron_id}.")
        return None

    valid_synapses.reset_index(drop=True, inplace=True)

    # 3. Extract the 3D coordinates of candidate synapses (These are in LOCAL space)
    candidate_coords_local = valid_synapses[['x', 'y', 'z']].values

    # --- FIX: TRANSLATE TARGETS TO LOCAL SPACE ---
    # Convert global target voxels back to the post-synaptic neuron's local reference frame
    post_soma_pos = np.array(post_soma_pos, dtype=float)
    local_chosen_centroids = chosen_centroids - post_soma_pos

    # 4. Build a KDTree for the local candidate synapses
    tree = cKDTree(candidate_coords_local)

    # 5. Query the tree using the LOCAL target centroids
    distances, indices = tree.query(local_chosen_centroids, k=1)

    # 6. Extract the matched rows
    matched_synapses_df = valid_synapses.iloc[indices].copy()

    # Add debugging info: The local target they snapped to, and the physical error
    matched_synapses_df['local_target_x'] = local_chosen_centroids[:, 0]
    matched_synapses_df['local_target_y'] = local_chosen_centroids[:, 1]
    matched_synapses_df['local_target_z'] = local_chosen_centroids[:, 2]
    matched_synapses_df['snapping_error_um'] = distances

    print(f"✅ Successfully mapped {len(matched_synapses_df)} synapses.")
    print(f"   -> Average snapping error: {np.mean(distances):.2f} μm")

    return matched_synapses_df





def cell_MorphSelect(
    morph_paths,
    layer_boundaries,
    layer_names,          
    synapses_per_layer,
    target_z_pos,
    synapse_base_path,
    plot_result=False     
):
    """Standalone version of the empirical morphology selector."""
    valid_layers = [i for i, syn in enumerate(synapses_per_layer) if syn > 0]
    shuffled_paths = list(enumerate(morph_paths))
    random.shuffle(shuffled_paths)
    
    for original_idx, path in shuffled_paths:
        match = re.search(r'neuron_(\d+)', os.path.basename(path))
        nid = int(match.group(1)) if match else -1
        
        csv_path = os.path.join(synapse_base_path, f"neuron_{nid}_synapses.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: Synapse CSV not found for {nid} at {csv_path}")
            continue
            
        syn_df = pd.read_csv(csv_path)
        shifted_z = syn_df['z'] + target_z_pos
        is_viable = True
        
        for layer_idx in valid_layers:
            upper_bound, lower_bound = layer_boundaries[layer_idx]
            required_synapses = synapses_per_layer[layer_idx]
            
            available_synapses = ((shifted_z >= lower_bound) & (shifted_z <= upper_bound)).sum()
            
            if available_synapses < required_synapses:
                is_viable = False
                break 
                
        if is_viable:
            return original_idx, nid
            
    raise RuntimeError(
        f"No morphology found with enough synapse capacity to satisfy "
        f"the requirements: {synapses_per_layer} at Z={target_z_pos:.1f}"
    )

def evaluate_and_select_morphology(
    post_cell_index,
    post_mtype,
    post_soma_pos,
    pre_partners_matrix,
    cell_mtypes,
    cell_coords,
    layer_bounds,
    density_maps_dir,
    morph_paths,
    synapses_dir,
    voxel_size=10.0,
    grid_extent=1000.0
):
    """Standalone version of the morphology evaluator and abstract placer."""
    print(f"\n--- Evaluating Morphology for Post-Neuron {post_cell_index} ({post_mtype}) ---")
    
    # Standardize Post-Synaptic Coordinates
    if isinstance(post_soma_pos, dict):
        post_soma_pos_arr = np.array([post_soma_pos['x'], post_soma_pos['y'], post_soma_pos['z']])
    else:
        post_soma_pos_arr = np.array(post_soma_pos, dtype=float)

    all_synapse_locations = []
    total_synapses_requested = 0

    # Iterate across all presynaptic partners
    for row in pre_partners_matrix:
        pre_idx = int(row[0])
        n_syn = int(row[1])
        
        pre_mtype = cell_mtypes[pre_idx]
        total_synapses_requested += n_syn
        
        # Standardize Pre-Synaptic Coordinates
        pre_soma_pos_raw = cell_coords[pre_idx]
        if isinstance(pre_soma_pos_raw, dict):
            pre_soma_pos_arr = np.array([pre_soma_pos_raw['x'], pre_soma_pos_raw['y'], pre_soma_pos_raw['z']])
        else:
            pre_soma_pos_arr = np.array(pre_soma_pos_raw, dtype=float)
        
        # 1. Calculate Overlap
        centroids, probs = calculate_synaptic_overlap_locations(
            pre_mtype=pre_mtype,
            post_mtype=post_mtype,
            pre_soma_pos=pre_soma_pos_arr,      
            post_soma_pos=post_soma_pos_arr,    
            density_maps_dir=density_maps_dir,
            num_synapses=n_syn,
            voxel_size=voxel_size,
            grid_extent=grid_extent
        )
        
        # 2. Distribute Synapses
        if centroids is not None and len(centroids) > 0:
            locations = distribute_synapses_probabilistically(centroids, probs, n_syn)
            if locations is not None and len(locations) > 0:
                all_synapse_locations.append(locations)

    if not all_synapse_locations:
        print(f"⚠️ Warning: No valid synaptic overlaps found for Neuron {post_cell_index}.")
        return None, np.array([])
        
    all_synapse_locations = np.vstack(all_synapse_locations)
    print(f"✅ Successfully placed {len(all_synapse_locations)}/{total_synapses_requested} synapses in abstract space.")

    # 3. Layer Formatting for cell_MorphSelect
    layer_names = list(layer_bounds.keys())
    layer_boundaries_array = np.array(list(layer_bounds.values()))
    synapses_per_layer_list = np.zeros(len(layer_names), dtype=int)
    
    z_coords = all_synapse_locations[:, 2] 

    for idx, bounds in enumerate(layer_boundaries_array):
        z_min, z_max = min(bounds), max(bounds)
        layer_mask = (z_coords >= z_min) & (z_coords <= z_max)
        synapses_per_layer_list[idx] = np.sum(layer_mask)

    print("   -> Synapse Distribution Profile:")
    for name, count in zip(layer_names, synapses_per_layer_list):
        if count > 0:
            print(f"      • {name}: {count} synapses")

    # 4. Select the best morphology
    target_z_pos = post_soma_pos_arr[2] 
    
    best_original_idx, best_nid = cell_MorphSelect(
        morph_paths=morph_paths,
        layer_boundaries=layer_boundaries_array,
        layer_names=layer_names,
        synapses_per_layer=synapses_per_layer_list,
        target_z_pos=target_z_pos,
        synapse_base_path=synapses_dir,
        plot_result=False
    )
    
    print(f"🎯 Selected Morphology ID: {best_nid}")
    return best_nid, all_synapse_locations

