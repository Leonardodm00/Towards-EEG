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
