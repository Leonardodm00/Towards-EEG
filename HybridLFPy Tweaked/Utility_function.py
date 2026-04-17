





'''

Utility functions

'''


'''
------------------------------------------------------
SYNAPTIC PLACEMENT
------------------------------------------------------
'''

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import re
import random

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
    # --- FIX: TRANSLATE TARGETS TO LOCAL SPACE (Dict Aware) ---
    if isinstance(post_soma_pos, dict):
        post_soma_pos = np.array([post_soma_pos['x'], post_soma_pos['y'], post_soma_pos['z']])
    else:
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

def add_virtual_synapses_to_morphology(
    final_mapped_synapses_dict,
    pre_partners_matrix,
    cell_mtypes,
    post_neuron_id,
    synapses_dir
):
    """
    Called after a morphology has been validated and selected.
    Randomly assigns segment locations for virtual sources (Thalamic 'TC' and Background 'BG_exc'/'BG_inh')
    to the correct synaptic locations (excitatory or inhibitory) on the chosen morphology.
    """
    import os
    import pandas as pd
    import random

    print(f"\n--- Appending Virtual Synapses (TC, BG) to Morphology {post_neuron_id} ---")

    # 1. Load the physical synapse database for this specific chosen morphology
    syn_filepath = os.path.join(synapses_dir, f"neuron_{post_neuron_id}_mapped_synapses.csv")
    if not os.path.exists(syn_filepath):
        print(f"⚠️ Error: Synapse mapping file not found for virtual assignment: {syn_filepath}")
        return final_mapped_synapses_dict

    mapped_df = pd.read_csv(syn_filepath)

    # 2. Extract all available excitatory and inhibitory segment indices
    exc_indices = mapped_df[mapped_df['synapse_type'] == 'exc']['lfpy_idx'].astype(int).tolist()
    inh_indices = mapped_df[mapped_df['synapse_type'] == 'inh']['lfpy_idx'].astype(int).tolist()

    if not exc_indices: print(f"⚠️ Warning: No excitatory synapses available on Morphology {post_neuron_id}.")
    if not inh_indices: print(f"⚠️ Warning: No inhibitory synapses available on Morphology {post_neuron_id}.")

    # 3. Iterate over the pre_partners_matrix to find ONLY the virtual cells
    added_tc, added_bg_exc, added_bg_inh = 0, 0, 0

    for row in pre_partners_matrix:
        pre_idx = int(row[0])
        n_syn = int(row[1])
        specific_mtype = cell_mtypes[pre_idx]

        # Filter: ONLY process the virtual/background cells in this function
        if specific_mtype not in ['TC', 'BG_exc', 'BG_inh']:
            continue
            
        # Determine the valid target pool based on the mtype
        if specific_mtype in ['TC', 'BG_exc']:
            pool = exc_indices
            target_type = 'exc'
        elif specific_mtype == 'BG_inh':
            pool = inh_indices
            target_type = 'inh'
            
        if not pool:
            # Failsafe if a tree somehow lacks necessary synapses
            final_mapped_synapses_dict[pre_idx] = []
            continue

        # Randomly sample 'n_syn' indices from the available pool (with replacement)
        # Using random.choices because a single long segment can safely host multiple virtual synapses
        chosen_segments = random.choices(pool, k=n_syn)
        final_mapped_synapses_dict[pre_idx] = chosen_segments

        # Tracking for the final print statement
        if specific_mtype == 'TC': added_tc += n_syn
        elif specific_mtype == 'BG_exc': added_bg_exc += n_syn
        elif specific_mtype == 'BG_inh': added_bg_inh += n_syn

    print(f"✅ Virtual synapses successfully attached: {added_tc} TC, {added_bg_exc} BG_exc, {added_bg_inh} BG_inh.")
    return final_mapped_synapses_dict









def evaluate_select_place(
    post_cell_index,
    post_mtype,
    post_soma_pos,
    pre_partners_matrix,
    cell_coords,
    layer_bounds,
    density_maps_dir,
    morph_paths,
    synapses_dir,
    cell_mtypes,
    mtype_fast_lookup,
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

    # CHANGED: Now a dictionary to hold locations keyed by pre_idx
    all_synapse_locations = {}
    total_synapses_requested = 0

    # Iterate across all presynaptic partners
    for row in pre_partners_matrix:
        pre_idx = int(row[0])
        n_syn = int(row[1])

        # Look up the specific biological m-type for this pre-synaptic cell
        specific_mtype = cell_mtypes[pre_idx]

        # ---> NEW ADDITION: Filter out virtual cells from spatial evaluation <---
        if specific_mtype in ['TC', 'BG_exc', 'BG_inh']:
            continue

        # Retrieve the layer and broad biological type
        layer, bio_type = mtype_fast_lookup[specific_mtype]

        # Map the broad type to the suffix used in your macro population names
        if "SS" in specific_mtype.upper():
            type_suffix = "ss"
        else:
            type_suffix = "exc" if bio_type == "Excitatory" else "inh"

        # Construct the final macro-population density string
        pre_mtype = f"population_probability_{layer}_{type_suffix}"

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
                # CHANGED: Assign the 3D locations to the dictionary using pre_idx as the key
                # (If there are duplicate pre_idx rows, we stack them to be safe)
                if pre_idx in all_synapse_locations:
                    all_synapse_locations[pre_idx] = np.vstack((all_synapse_locations[pre_idx], locations))
                else:
                    all_synapse_locations[pre_idx] = locations

    if not all_synapse_locations:
        print(f"⚠️ Warning: No valid synaptic overlaps found for Neuron {post_cell_index}.")
        return None, {} # CHANGED: Return an empty dictionary here

    # CHANGED: Stack the dictionary values into a flat array just for the layer counting below
    stacked_locations = np.vstack(list(all_synapse_locations.values()))

    print(f"✅ Successfully placed {len(stacked_locations)}/{total_synapses_requested} synapses in abstract space.")

    # 3. Layer Formatting for cell_MorphSelect
    layer_names = list(layer_bounds.keys())
    layer_boundaries_array = np.array(list(layer_bounds.values()))
    synapses_per_layer_list = np.zeros(len(layer_names), dtype=int)

    # CHANGED: Use the stacked array to extract z_coords so the math below stays exactly the same
    z_coords = stacked_locations[:, 2]

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


    # Make a working copy of the paths so we can remove bad ones
    available_morph_paths = morph_paths.copy()
    clustering_detected = True
    best_nid = None
    final_mapped_synapses_dict = {}

    while clustering_detected and len(available_morph_paths) > 0:

        # We assume cell_MorphSelect returns (original_idx, best_nid)
        best_original_idx, best_nid = cell_MorphSelect(
            morph_paths=available_morph_paths,
            layer_boundaries=layer_boundaries_array,
            layer_names=layer_names,
            synapses_per_layer=synapses_per_layer_list,
            target_z_pos=target_z_pos,
            synapse_base_path=synapses_dir,
            plot_result=False
        )

        if best_nid is None:
            print("❌ No remaining morphologies satisfy capacity requirements.")
            break # Exit the while loop

        print(f"🎯 Evaluating Selected Morphology ID: {best_nid} for synaptic spread...")

        # 1. Snap the abstract 3D coordinates to the chosen morphology
        final_mapped_synapses_dict = map_abstract_synapses_to_segments(
            abstract_synapses_dict=all_synapse_locations,
            post_neuron_id=best_nid,
            synapses_dir=synapses_dir,
            post_soma_pos=post_soma_pos_arr, # FIXED scope
            cell_mtypes=cell_mtypes,
            mtype_fast_lookup=mtype_fast_lookup
        )

        # 2. Evaluate clustering across all pre-synaptic partners
        clustering_detected = False # Reset for this specific morphology
        for pre_idx, lfpy_indices in final_mapped_synapses_dict.items():
            if len(lfpy_indices) > 1 and len(set(lfpy_indices)) == 1:
                print(f"❌ Partner {pre_idx} has {len(lfpy_indices)} synapses clumped entirely on segment {lfpy_indices[0]}.")
                clustering_detected = True
                break # FIXED logic: Immediately fail this morphology

        if clustering_detected:
            # Remove the failed morphology from the available paths before trying again
            print(f"⏭️ Morphology {best_nid} REJECTED due to clumping. Trying next candidate...")
            available_morph_paths = [p for p in available_morph_paths if f"neuron_{best_nid}" not in p]
        else:
             print(f"✅ Morphology {best_nid} ACCEPTED (Capacity & Spread validated).")

    # Final Failsafe
    if clustering_detected or best_nid is None:
         print("❌ Critical Failure: Could not find a morphology satisfying both capacity and spread.")
         return None, {}, {}


    # ---> NEW ADDITION: Attach Thalamic and Background synapses <---
    final_mapped_synapses_dict = add_virtual_synapses_to_morphology(
        final_mapped_synapses_dict=final_mapped_synapses_dict,
        pre_partners_matrix=pre_partners_matrix,
        cell_mtypes=cell_mtypes,
        post_neuron_id=best_nid,
        synapses_dir=synapses_dir
    )


    return best_original_idx, best_nid, final_mapped_synapses_dict

def map_abstract_synapses_to_segments(
    abstract_synapses_dict,
    post_neuron_id,
    synapses_dir,
    post_soma_pos,
    cell_mtypes,
    mtype_fast_lookup
):
    """
    Iterates through a dictionary of abstract 3D synapse locations, snaps them to the
    actual dendritic arbor of the chosen morphology, and returns a dictionary
    mapping pre-synaptic indices directly to their flattened 1D segment indices (lfpy_idx).
    
    The synapse type ('exc' or 'inh') is dynamically determined for each pre-synaptic neuron.

    Args:
        abstract_synapses_dict (dict): {pre_idx: np.ndarray of 3D coordinates}
        post_neuron_id (int): The selected morphology ID (e.g., 4852204352).
        synapses_dir (str): Path to the mapped synapse CSV databases.
        post_soma_pos (list/array): The [x, y, z] global position of the post-synaptic soma.
        cell_mtypes (list/array): Array mapping raw indices to morphological type strings.
        mtype_fast_lookup (dict): Dictionary mapping mtypes to a (layer, bio_type) tuple.

    Returns:
        dict: {pre_idx: [list of integer lfpy_idx segment indices]}
    """
    final_mapped_synapses_dict = {}

    if not abstract_synapses_dict:
        print("⚠️ No abstract synapses provided to map.")
        return final_mapped_synapses_dict

    print(f"\n--- Snapping Abstract Synapses to Morphology {post_neuron_id} ---")

    for pre_idx, abstract_centroids in abstract_synapses_dict.items():

        # --- DYNAMIC SYNAPSE TYPE EVALUATION ---
        try:
            pre_mtype = cell_mtypes[pre_idx]
            _, bio_type = mtype_fast_lookup[pre_mtype]
            
            # Map 'Excitatory'/'Inhibitory' to 'exc'/'inh'
            current_synapse_type = 'exc' if bio_type == 'Excitatory' else 'inh'
            print(f'Current syn type:{current_synapse_type}')
            
        except (IndexError, KeyError) as e:
            print(f"⚠️ Could not determine synapse type for pre_idx {pre_idx}. Defaulting to 'exc'. Error: {e}")
            current_synapse_type = 'exc'
        # ---------------------------------------

        # 1. Snap this specific group of coordinates to the physical tree
        snapped_df = snap_to_closest_actual_synapses(
            post_neuron_id=post_neuron_id,
            synapses_dir=synapses_dir,
            chosen_centroids=abstract_centroids,
            post_soma_pos=post_soma_pos,
            synapse_type=current_synapse_type
        )

        # 2. Extract the segment indices and assign them to the pre_idx key
        if snapped_df is not None and not snapped_df.empty:
            # We enforce .astype(int) just in case pandas loaded them as floats (e.g., 45.0)
            lfpy_indices = snapped_df['lfpy_idx'].astype(int).tolist()
            final_mapped_synapses_dict[pre_idx] = lfpy_indices
        else:
            # If for some reason snapping failed for this specific connection, return an empty list
            print(f"   -> ⚠️ Snapping failed or returned empty for pre_idx {pre_idx}")
            final_mapped_synapses_dict[pre_idx] = []

    # Calculate totals for terminal debugging
    total_mapped = sum(len(indices) for indices in final_mapped_synapses_dict.values())
    print(f"✅ Successfully mapped {total_mapped} synapses across {len(final_mapped_synapses_dict)} pre-synaptic connections to LFPy indices.")

    return final_mapped_synapses_dict

def cell_MorphSelect(
    morph_paths,
    layer_boundaries,
    layer_names,
    synapses_per_layer,
    target_z_pos,
    synapse_base_path,
    plot_result=False
):
    """Standalone version of the empirical morphology selector with safe CSV loading and exc/inh breakdown printing."""
    valid_layers = [i for i, syn in enumerate(synapses_per_layer) if syn > 0]
    shuffled_paths = list(enumerate(morph_paths))
    random.shuffle(shuffled_paths)

    for original_idx, path in shuffled_paths:
        match = re.search(r'neuron_(\d+)', os.path.basename(path))
        nid = int(match.group(1)) if match else -1

        csv_path = os.path.join(synapse_base_path, f"neuron_{nid}_mapped_synapses.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: Synapse CSV not found for {nid} at {csv_path}")
            continue

        # ==========================================
        # BULLETPROOF CSV LOADING
        # ==========================================
        syn_df = pd.read_csv(csv_path)

        # 1. Strip hidden whitespace and force lowercase (e.g., " Z " becomes "z")
        syn_df.columns = syn_df.columns.str.strip().str.lower()

        # 2. Diagnostic Fallback: Did it load a raw file instead of an aligned one?
        if 'z' not in syn_df.columns:
            if 'location_z' in syn_df.columns:
                # Auto-correct the raw columns to prevent the crash
                syn_df = syn_df.rename(columns={
                    'location_x': 'x',
                    'location_y': 'y',
                    'location_z': 'z'
                })
            else:
                # If it completely fails, print exactly what is inside the file to debug
                raise KeyError(f"Failed to find 'z' or 'location_z' in {csv_path}. The columns found were: {syn_df.columns.tolist()}")
        # ==========================================

        shifted_z = syn_df['z'] + target_z_pos
        is_viable = True

        print(f"🔍 Evaluating Morphology ID: {nid} (Target Z: {target_z_pos:.1f})")

        # Evaluate and print every required layer
        for layer_idx in valid_layers:
            lower_bound, upper_bound = layer_boundaries[layer_idx]
            required_synapses = synapses_per_layer[layer_idx]
            layer_name = layer_names[layer_idx]

            # Create a boolean mask for all points inside this layer
            layer_mask = (shifted_z >= lower_bound) & (shifted_z <= upper_bound)
            available_synapses = layer_mask.sum()

            # Count the specific types within this layer (if the column exists)
            if 'synapse_type' in syn_df.columns:
                exc_count = (layer_mask & (syn_df['synapse_type'] == 'exc')).sum()
                inh_count = (layer_mask & (syn_df['synapse_type'] == 'inh')).sum()
                breakdown_str = f"[Exc: {exc_count} | Inh: {inh_count}]"
            else:
                breakdown_str = "[Type info missing]"

            # Check if the TOTAL available meets the requirement
            if available_synapses >= required_synapses:
                status = "✅ Pass"
            else:
                status = "❌ Fail"
                is_viable = False

            # Print the total stats alongside the breakdown
            print(f"   -> {layer_name}: Available = {available_synapses} {breakdown_str} | Required (Total) = {required_synapses} {status}")

        if is_viable:
            print(f"   🎯 Morphology {nid} ACCEPTED!\n")
            return original_idx, nid
        else:
            print(f"   ⏭️ Morphology {nid} REJECTED. Moving to next candidate...\n")

    raise RuntimeError(
        f"No morphology found with enough synapse capacity to satisfy "
        f"the requirements: {synapses_per_layer} at Z={target_z_pos:.1f}"
    )



