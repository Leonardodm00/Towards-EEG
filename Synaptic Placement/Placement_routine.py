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




import os
import numpy as np
import matplotlib.pyplot as plt

def debug_plot_synaptic_overlap(pre_mtype, post_mtype, density_maps_dir, pre_soma_pos, post_soma_pos, overlap_centroids, overlap_probabilities, voxel_size=10.0):
    """
    Generates 2D Maximum Intensity Projections (XY and XZ planes) of the pre- and post-synaptic
    probability fields in global physical space, highlighting the overlapping voxels as a heatmap.
    """
    print("🎨 Generating 2D spatial overlap projections...")

    # 1. Construct file paths (assuming the naming convention from previous steps)
    pre_axon_path = os.path.join(density_maps_dir, f"{pre_mtype}_axon.npy")
    post_dend_path = os.path.join(density_maps_dir, f"{post_mtype}_dend.npy")

    if not (os.path.exists(pre_axon_path) and os.path.exists(post_dend_path)):
        print("⚠️ Density maps not found for the specified mtypes.")
        return None, None

    # 2. Load the 3D probability arrays
    pre_grid = np.load(pre_axon_path)   # Axonal field of pre-synaptic neuron
    post_grid = np.load(post_dend_path) # Dendritic field of post-synaptic neuron

    # 1. Calculate Maximum Intensity Projections (Squash 3D to 2D)
    # Assuming histogramdd output shape is (X, Y, Z)
    pre_xy = np.max(pre_grid, axis=2)
    pre_xz = np.max(pre_grid, axis=1)

    post_xy = np.max(post_grid, axis=2)
    post_xz = np.max(post_grid, axis=1)

    # 2. Mask zero values so the backgrounds are transparent when overlaid
    pre_xy_m = np.ma.masked_where(pre_xy <= 0, pre_xy)
    pre_xz_m = np.ma.masked_where(pre_xz <= 0, pre_xz)
    post_xy_m = np.ma.masked_where(post_xy <= 0, post_xy)
    post_xz_m = np.ma.masked_where(post_xz <= 0, post_xz)

    # 3. Calculate physical extents [xmin, xmax, ymin, ymax] for Matplotlib
    # We define the physical boundaries of the grids relative to their soma centers
    def get_extent(grid_shape, soma_pos, voxel_size):
        Nx, Ny, Nz = grid_shape
        hx, hy, hz = (Nx / 2) * voxel_size, (Ny / 2) * voxel_size, (Nz / 2) * voxel_size

        extent_xy = [soma_pos[0] - hx, soma_pos[0] + hx, soma_pos[1] - hy, soma_pos[1] + hy]
        extent_xz = [soma_pos[0] - hx, soma_pos[0] + hx, soma_pos[2] - hz, soma_pos[2] + hz]
        return extent_xy, extent_xz

    pre_extent_xy, pre_extent_xz = get_extent(pre_grid.shape, pre_soma_pos, voxel_size)
    post_extent_xy, post_extent_xz = get_extent(post_grid.shape, post_soma_pos, voxel_size)

    # 4. Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Spanning Tree Overlap Debugger", fontsize=16)

    # --- Plot XY Plane (Top-Down View) ---
    # .T is used to transpose from (X, Y) to (Rows, Cols) for imshow
    ax1.imshow(post_xy_m.T, extent=post_extent_xy, origin='lower', cmap='Blues', alpha=0.6, label='Post-Dendrite')
    ax1.imshow(pre_xy_m.T, extent=pre_extent_xy, origin='lower', cmap='Oranges', alpha=0.6, label='Pre-Axon')

    if overlap_centroids is not None and len(overlap_centroids) > 0:
        # Changed to a heatmap scatter using overlap_probabilities
        sc1 = ax1.scatter(overlap_centroids[:, 0], overlap_centroids[:, 1], c=overlap_probabilities, cmap='viridis', s=25, marker='s', label='Overlap Heatmap')
        cbar1 = fig.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Joint Probability')

    ax1.scatter(*post_soma_pos[:2], c='blue', s=80, marker='^', edgecolors='white', label='Post Soma')
    ax1.scatter(*pre_soma_pos[:2], c='red', s=80, marker='o', edgecolors='white', label='Pre Soma')

    ax1.set_title("XY Plane (Top-Down)")
    ax1.set_xlabel("X (μm)")
    ax1.set_ylabel("Y (μm)")
    ax1.grid(color='gray', linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right')

    # --- Plot XZ Plane (Side View) ---
    ax2.imshow(post_xz_m.T, extent=post_extent_xz, origin='lower', cmap='Blues', alpha=0.6)
    ax2.imshow(pre_xz_m.T, extent=pre_extent_xz, origin='lower', cmap='Oranges', alpha=0.6)

    if overlap_centroids is not None and len(overlap_centroids) > 0:
        # Changed to a heatmap scatter using overlap_probabilities
        sc2 = ax2.scatter(overlap_centroids[:, 0], overlap_centroids[:, 2], c=overlap_probabilities, cmap='viridis', s=25, marker='s')
        cbar2 = fig.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Joint Probability')

    ax2.scatter(post_soma_pos[0], post_soma_pos[2], c='blue', s=80, marker='^', edgecolors='white')
    ax2.scatter(pre_soma_pos[0], pre_soma_pos[2], c='red', s=80, marker='o', edgecolors='white')

    ax2.set_title("XZ Plane (Side View)")
    ax2.set_xlabel("X (μm)")
    ax2.set_ylabel("Z (μm)")
    ax2.grid(color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

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
import numpy as np
import matplotlib.pyplot as plt

def visualize_synapse_distribution_verification(
    pre_mtype, post_mtype, density_maps_dir,
    pre_soma_pos, post_soma_pos, chosen_centroids, voxel_size=10.0
):
    """
    Verifies synaptic placement by overlaying chosen centroids (as tallies)
    on the original spanning tree MIPs in global physical space.
    """
    print("🎨 Generating verification plots with spanning tree blueprints...")
    if chosen_centroids is None or len(chosen_centroids) == 0:
        print("⚠️ No synapses placed to visualize.")
        return

    # 1. File Path Construction & Validation
    pre_axon_path = os.path.join(density_maps_dir, f"{pre_mtype}_axon.npy")
    post_dend_path = os.path.join(density_maps_dir, f"{post_mtype}_dend.npy")

    if not (os.path.exists(pre_axon_path) and os.path.exists(post_dend_path)):
        print("⚠️ Density maps not found; cannot generate biological blueprint context.")
        return

    # 2. Load and Prepare Spanning Tree Blueprints (Unchanged logic)
    pre_grid = np.load(pre_axon_path)
    post_grid = np.load(post_dend_path)

    # Maximum Intensity Projections (Squash 3D to 2D)
    # Assumes (X, Y, Z) order from histogramdd
    pre_xy = np.max(pre_grid, axis=2); pre_xz = np.max(pre_grid, axis=1)
    post_xy = np.max(post_grid, axis=2); post_xz = np.max(post_grid, axis=1)

    # Mask zero values for transparency in overlay
    pre_xy_m = np.ma.masked_where(pre_xy <= 0, pre_xy)
    pre_xz_m = np.ma.masked_where(pre_xz <= 0, pre_xz)
    post_xy_m = np.ma.masked_where(post_xy <= 0, post_xy)
    post_xz_m = np.ma.masked_where(post_xz <= 0, post_xz)

    # Helper function for physical extents (Unchanged logic)
    def get_extent(grid_shape, soma_pos, voxel_size):
        Nx, Ny, Nz = grid_shape
        hx, hy, hz = (Nx / 2) * voxel_size, (Ny / 2) * voxel_size, (Nz / 2) * voxel_size
        extent_xy = [soma_pos[0] - hx, soma_pos[0] + hx, soma_pos[1] - hy, soma_pos[1] + hy]
        extent_xz = [soma_pos[0] - hx, soma_pos[0] + hx, soma_pos[2] - hz, soma_pos[2] + hz]
        return extent_xy, extent_xz

    pre_extent_xy, pre_extent_xz = get_extent(pre_grid.shape, pre_soma_pos, voxel_size)
    post_extent_xy, post_extent_xz = get_extent(post_grid.shape, post_soma_pos, voxel_size)

    # --- 3. SYNAPSE TALLY CALCULATION ---
    # Since replace=True allows multiple synapses per voxel,
    # find unique chosen coordinates and count how many times each was chosen.
    unique_centroids, counts = np.unique(chosen_centroids, axis=0, return_counts=True)

    # Create marker size scaled by synapse count (e.g., base_size + multiplier * count)
    marker_sizes = 30 + (40 * counts)

    # 4. Figure Setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Synaptic Distribution Verification ({len(chosen_centroids)} total synapses placed)", fontsize=16)

    # --- Plot XY Plane (Top-Down View) ---
    # .T transposes (X,Y) to image format (Rows,Cols)
    ax1.imshow(post_xy_m.T, extent=post_extent_xy, origin='lower', cmap='Blues', alpha=0.5, label='Post-Dendrite Blueprint')
    ax1.imshow(pre_xy_m.T, extent=pre_extent_xy, origin='lower', cmap='Oranges', alpha=0.5, label='Pre-Axon Blueprint')

    # Plot Somas (Triangles/Circles)
    ax1.scatter(*post_soma_pos[:2], c='blue', s=100, marker='^', edgecolors='white', label='Post Soma', zorder=5)
    ax1.scatter(*pre_soma_pos[:2], c='red', s=100, marker='o', edgecolors='white', label='Pre Soma', zorder=5)

    # Plot Placed Synapses (Squares scaled by Tally)
    ax1.scatter(unique_centroids[:, 0], unique_centroids[:, 1],
                c='lime', s=marker_sizes, marker='s', edgecolors='black', linewidths=0.5, label='Placed Synapse Tally', zorder=10)

    ax1.set_title("XY Plane (Verification)")
    ax1.set_xlabel("X (μm)"); ax1.set_ylabel("Y (μm)")
    ax1.grid(color='gray', linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right')

    # --- Plot XZ Plane (Side View) ---
    ax2.imshow(post_xz_m.T, extent=post_extent_xz, origin='lower', cmap='Blues', alpha=0.5)
    ax2.imshow(pre_xz_m.T, extent=pre_extent_xz, origin='lower', cmap='Oranges', alpha=0.5)

    ax2.scatter(post_soma_pos[0], post_soma_pos[2], c='blue', s=100, marker='^', edgecolors='white', zorder=5)
    ax2.scatter(pre_soma_pos[0], pre_soma_pos[2], c='red', s=100, marker='o', edgecolors='white', zorder=5)

    ax2.scatter(unique_centroids[:, 0], unique_centroids[:, 2],
                c='lime', s=marker_sizes, marker='s', edgecolors='black', linewidths=0.5, zorder=10)

    ax2.set_title("XZ Plane (Verification)")
    ax2.set_xlabel("X (μm)"); ax2.set_ylabel("Z (μm)")
    ax2.grid(color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def visualize_joint_spanning_histo(normalized_probabilities):
    """
    Plots the statistical distribution of the joint probability values
    across all overlapping voxels, rather than their spatial locations.
    """
    print("📊 Generating probability value distribution...")
    if normalized_probabilities is None or len(normalized_probabilities) == 0:
        print("⚠️ No probability data to visualize.")
        return

    # Setup Figure (2 panels: Histogram and CDF)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Statistical Distribution of Joint Probabilities (PMF)", fontsize=16)

    # --- Panel 1: Histogram ---
    # Shows how many voxels share the same probability weight
    ax1.hist(normalized_probabilities, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax1.set_title("Histogram of Probability Values")
    ax1.set_xlabel("Normalized Joint Probability")
    ax1.set_ylabel("Frequency (Number of Voxels)")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Panel 2: Cumulative Distribution Function (CDF) ---
    # Shows the accumulation of probability weight
    ax2.hist(normalized_probabilities, bins=50, color='teal', alpha=0.7, edgecolor='black', cumulative=True, density=True)
    ax2.set_title("Cumulative Distribution Function (CDF)")
    ax2.set_xlabel("Normalized Joint Probability")
    ax2.set_ylabel("Cumulative Proportion of Voxels")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
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

import os
import ast
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go

import os
import ast
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go

def visualize_final_snapped_synapses_3d(
    post_neuron_id,
    matched_synapses_df,
    input_dir,
    metadata_filepath,
    post_soma_pos,
    pre_mtype=None,          # NEW: Pre-synaptic identifier
    density_maps_dir=None,   # NEW: Path to the .npy density maps
    pre_soma_pos=None,       # NEW: Global position of the pre-soma
    voxel_size=10.0,         # NEW: Required to translate grid indices to um
    k_neighbors=3
):
    """
    Plots the snapped synapses on the 3D post-synaptic arbor, and optionally
    overlays the pre-synaptic axonal probability field as a 3D volumetric cloud.
    """
    print(f"🎨 Rendering 3D spatial verification for Neuron {post_neuron_id}...")
    fig = go.Figure()

    # ==========================================
    # 1. LOAD & PLOT PRE-SYNAPTIC AXONAL CLOUD
    # ==========================================
    if all(v is not None for v in [pre_mtype, density_maps_dir, pre_soma_pos]):
        pre_axon_path = os.path.join(density_maps_dir, f"{pre_mtype}_axon.npy")
        if os.path.exists(pre_axon_path):
            print("   -> Adding Pre-Synaptic Axonal Probability Cloud...")
            pre_grid = np.load(pre_axon_path)
            Nx, Ny, Nz = pre_grid.shape

            # Find all voxels where the probability is > 0
            valid_x, valid_y, valid_z = np.where(pre_grid > 0)

            # Convert array indices to local physical distance, then shift to global space
            center_idx = Nx // 2
            global_pre_x = (valid_x - center_idx) * voxel_size + pre_soma_pos[0]
            global_pre_y = (valid_y - center_idx) * voxel_size + pre_soma_pos[1]
            global_pre_z = (valid_z - center_idx) * voxel_size + pre_soma_pos[2]

            # Downsample the cloud if it's too dense (keeps the browser from freezing)
            max_points = 15000
            if len(global_pre_x) > max_points:
                idx = np.random.choice(len(global_pre_x), max_points, replace=False)
                global_pre_x, global_pre_y, global_pre_z = global_pre_x[idx], global_pre_y[idx], global_pre_z[idx]

            # Add the Axonal Cloud
            fig.add_trace(go.Scatter3d(
                x=global_pre_x, y=global_pre_y, z=global_pre_z,
                mode='markers',
                marker=dict(size=3, color='darkorange', opacity=0.05),
                name='Pre-Synaptic Axon Field',
                hoverinfo='none'
            ))

            # Add the Pre-Soma Marker
            fig.add_trace(go.Scatter3d(
                x=[pre_soma_pos[0]], y=[pre_soma_pos[1]], z=[pre_soma_pos[2]],
                mode='markers',
                marker=dict(size=8, color='red', symbol='circle', line=dict(color='white', width=1)),
                name='Pre-Soma Position'
            ))

    # ==========================================
    # 2. LOAD & ALIGN POST-SYNAPTIC SKELETON
    # ==========================================
    metadata_df = pd.read_csv(metadata_filepath)
    if isinstance(metadata_df['rotation_matrix'].iloc[0], str):
        metadata_df['rotation_matrix'] = metadata_df['rotation_matrix'].apply(ast.literal_eval)
    reference_somas = metadata_df[['soma_x', 'soma_y', 'soma_z']].values

    filepath = os.path.join(input_dir, f"neuron_{post_neuron_id}.csv")
    if not os.path.exists(filepath):
        print(f"⚠️ Morphology file not found: {filepath}")
        return

    df = pd.read_csv(filepath)
    root_rows = df[df['p'] == -1]
    if root_rows.empty:
        return

    soma_pos_raw = root_rows.iloc[0][['x', 'y', 'z']].values.astype(float)

    # Align to Local Space
    centered_coords = df[['x', 'y', 'z']].values - soma_pos_raw
    distances = np.linalg.norm(reference_somas - soma_pos_raw, axis=1)
    nearest_metadata = metadata_df.iloc[np.argsort(distances)[:min(k_neighbors, len(metadata_df))]]
    mean_matrix = R.from_matrix(np.array(nearest_metadata['rotation_matrix'].tolist())).mean().as_matrix()
    rotated_coords = np.dot(centered_coords, mean_matrix.T)

    # Scale and Shift to Global Space
    post_soma_pos = np.array(post_soma_pos, dtype=float)
    global_coords = (rotated_coords / 1000.0) + post_soma_pos

    df['x_glob'], df['y_glob'], df['z_glob'] = global_coords[:, 0], global_coords[:, 1], global_coords[:, 2]

    # Build Skeleton Trace
    node_map = df.set_index('id')[['x_glob', 'y_glob', 'z_glob']].to_dict('index')
    skel_x, skel_y, skel_z = [], [], []
    for _, row in df[df['p'] != -1].iterrows():
        pid = row['p']
        if pid in node_map:
            parent = node_map[pid]
            skel_x.extend([row['x_glob'], parent['x_glob'], None])
            skel_y.extend([row['y_glob'], parent['y_glob'], None])
            skel_z.extend([row['z_glob'], parent['z_glob'], None])

    fig.add_trace(go.Scatter3d(
        x=skel_x, y=skel_y, z=skel_z,
        mode='lines',
        line=dict(color='darkgrey', width=2),
        opacity=0.6,
        name='Post-Synaptic Arbor',
        hoverinfo='none'
    ))

    # Add Post-Soma Marker
    fig.add_trace(go.Scatter3d(
        x=[post_soma_pos[0]], y=[post_soma_pos[1]], z=[post_soma_pos[2]],
        mode='markers',
        marker=dict(size=8, color='blue', symbol='square', line=dict(color='white', width=1)),
        name='Post-Soma Position'
    ))

    # ==========================================
    # 3. PLOT SNAPPED SYNAPSES
    # ==========================================
    if matched_synapses_df is not None and not matched_synapses_df.empty:
        # Shift synapses from local to global space
        syn_glob_x = matched_synapses_df['x'].values + post_soma_pos[0]
        syn_glob_y = matched_synapses_df['y'].values + post_soma_pos[1]
        syn_glob_z = matched_synapses_df['z'].values + post_soma_pos[2]

        syn_type = matched_synapses_df['synapse_type'].iloc[0]
        marker_color = 'lime' if syn_type == 'exc' else 'cyan'

        hover_texts = [
            f"LFPy idx: {int(row['lfpy_idx'])}<br>Error: {row['snapping_error_um']:.2f} µm"
            for _, row in matched_synapses_df.iterrows()
        ]

        fig.add_trace(go.Scatter3d(
            x=syn_glob_x, y=syn_glob_y, z=syn_glob_z,
            mode='markers',
            marker=dict(size=6, color=marker_color, symbol='diamond', line=dict(color='black', width=1)),
            name=f'Placed {syn_type.upper()} Synapses',
            text=hover_texts,
            hoverinfo='text'
        ))

    # ==========================================
    # 4. FINAL LAYOUT
    # ==========================================
    fig.update_layout(
        title=f"Global Synaptic Placement & Presynaptic Context: Post-Neuron {post_neuron_id}",
        scene=dict(
            xaxis_title='X (µm)', yaxis_title='Y (µm)', zaxis_title='Z (µm)',
            aspectmode='data', bgcolor='white'
        ),
        width=1400, height=900,
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(x=0.02, y=0.98, itemsizing='constant')
    )

    fig.show()


def execute_synaptic_placement_pipeline(
    pre_mtype,
    post_mtype,
    pre_soma_pos,
    post_soma_pos,
    density_maps_dir,
    num_synapses,
    post_neuron_id,
    synapses_dir,
    synapse_type='exc',
    voxel_size=10.0,
    grid_extent=1000.0
):
    """
    Executes the full pipeline for mapping synapses between two neurons:
    1. Calculates 3D voxel overlaps based on pre/post density maps.
    2. Distributes the target number of synapses probabilistically across the overlap.
    3. Snaps the abstract target voxels to the closest physically realized synapses on the post-synaptic morphology.
    """
    print(f"\n--- Starting Synapse Placement Pipeline for Neuron {post_neuron_id} ---")
    
    # STEP 1: Calculate Overlap and Joint Probabilities
    print("Step 1: Calculating spatial overlap...")
    centroids, normalized_probabilities = calculate_synaptic_overlap_locations(
        pre_mtype=pre_mtype,
        post_mtype=post_mtype,
        pre_soma_pos=pre_soma_pos,
        post_soma_pos=post_soma_pos,
        density_maps_dir=density_maps_dir,
        num_synapses=num_synapses,
        voxel_size=voxel_size,
        grid_extent=grid_extent
    )
    
    # Abort if there is no physical overlap between the pre and post grids
    if centroids is None or len(centroids) == 0:
        print("❌ Pipeline aborted: No synaptic overlap found between these two positions.")
        return None

    # STEP 2: Probabilistically Sample Target Voxels
    print("Step 2: Distributing target voxels...")
    chosen_synapse_locations = distribute_synapses_probabilistically(
        overlap_centroids=centroids, 
        normalized_probabilities=normalized_probabilities, 
        num_synapses=num_synapses
    )
    
    if chosen_synapse_locations is None or len(chosen_synapse_locations) == 0:
        print("❌ Pipeline aborted: Failed to distribute synapses.")
        return None

    # STEP 3: Snap to Actual Morphological Synapses
    print("Step 3: Snapping to physical morphology...")
    final_synapse_locations_df = snap_to_closest_actual_synapses(
        post_neuron_id=post_neuron_id,
        synapses_dir=synapses_dir,
        chosen_centroids=chosen_synapse_locations,
        post_soma_pos=post_soma_pos,
        synapse_type=synapse_type
    )
    
    print("--- Pipeline Complete ---\n")
    return final_synapse_locations_df

