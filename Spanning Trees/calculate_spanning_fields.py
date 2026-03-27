import os
import ast
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
def calculate_spanning_fields(neuron_ids, metadata_filepath, input_dir, voxel_size=10.0, grid_extent=1000.0, k_neighbors=3):
    """
    Calculates the average fractional volume occupied by axonal and dendritic arbors 
    per voxel for a subpopulation of neurons.
    
    Includes continuous spatial interpolation to prevent undersampling of long segments.
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
    # Creating bin edges for np.histogramdd
    bins_x = np.arange(-grid_extent, grid_extent + voxel_size, voxel_size)
    bins_y = np.arange(-grid_extent, grid_extent + voxel_size, voxel_size)
    bins_z = np.arange(-grid_extent, grid_extent + voxel_size, voxel_size)
    
    voxel_volume = voxel_size ** 3
    num_bins = (len(bins_x)-1, len(bins_y)-1, len(bins_z)-1)
    
    # Initialize empty grids to accumulate volumes
    axon_grid_total = np.zeros(num_bins, dtype=np.float32)
    dend_grid_total = np.zeros(num_bins, dtype=np.float32)

    valid_neurons_processed = 0
    print(f"🔄 Calculating continuous spanning fields for {len(neuron_ids)} neurons...")

    # 3. Process sequentially to save RAM
    for nid in neuron_ids:
        filepath = os.path.join(input_dir, f"neuron_{nid}.csv")
        if not os.path.exists(filepath):
            continue

        # Load raw neuron
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

        # Overwrite coordinates (Assuming data is ALREADY in micrometers, no /1000.0 division)
        df['x'] = rotated_coords[:, 0]
        df['y'] = rotated_coords[:, 1]
        df['z'] = rotated_coords[:, 2]
        
        # Ensure radius is handled correctly
        if 'r' not in df.columns:
            df['r'] = 0.5
        
        # Categorize compartments
        df['annotated_type'] = df['annotated_type'].astype(str).str.lower()
        df['is_axon'] = df['annotated_type'].str.contains('axon')
        df['is_soma'] = df['annotated_type'].str.contains('soma')
        df['is_dend'] = ~(df['is_axon'] | df['is_soma'])

        # --- CONTINUOUS SEGMENT RASTERIZATION ---
        # Map parents to children to define line segments
        parents = df[['id', 'x', 'y', 'z']].rename(columns={'id': 'p', 'x': 'px', 'y': 'py', 'z': 'pz'})
        segments = df.merge(parents, on='p', how='inner')

        # Extract start points, end points, and radii
        starts = segments[['px', 'py', 'pz']].values
        ends = segments[['x', 'y', 'z']].values
        radii = segments['r'].values
        
        # Calculate Segment Length and Total Volume (Cylinder: V = pi * r^2 * L)
        lengths = np.linalg.norm(ends - starts, axis=1)
        volumes = np.pi * (radii ** 2) * lengths

        # Determine how many sub-points to sample along each segment 
        # (Step size smaller than voxel size guarantees continuous strokes without skipping voxels)
        step_size = voxel_size / 3.0 
        num_points = np.maximum(1, np.ceil(lengths / step_size)).astype(int)

        all_points_axon, all_vols_axon = [], []
        all_points_dend, all_vols_dend = [], []

        # Interpolate points along the lines and distribute the volume proportionally
        for i in range(len(lengths)):
            n = num_points[i]
            if n == 1:
                pts = np.array([(starts[i] + ends[i]) / 2.0])
            else:
                pts = np.linspace(starts[i], ends[i], n)

            # Split the total volume of this segment evenly across its sub-points
            vol_per_pt = volumes[i] / n

            if segments['is_axon'].iloc[i]:
                all_points_axon.append(pts)
                all_vols_axon.extend([vol_per_pt] * n)
            elif segments['is_dend'].iloc[i]:
                all_points_dend.append(pts)
                all_vols_dend.extend([vol_per_pt] * n)

        # --- HISTOGRAM RASTERIZATION ---
        # Rapidly sum distributed volumes into the 3D grid
        if all_points_axon:
            axon_pts = np.vstack(all_points_axon)
            axon_vols = np.array(all_vols_axon)
            axon_hist, _ = np.histogramdd(
                axon_pts, bins=(bins_x, bins_y, bins_z), weights=axon_vols
            )
            axon_grid_total += axon_hist
            
        if all_points_dend:
            dend_pts = np.vstack(all_points_dend)
            dend_vols = np.array(all_vols_dend)
            dend_hist, _ = np.histogramdd(
                dend_pts, bins=(bins_x, bins_y, bins_z), weights=dend_vols
            )
            dend_grid_total += dend_hist
        
        valid_neurons_processed += 1
        
        # Free memory immediately
        del df, parents, segments, all_points_axon, all_points_dend
        
    print(f"✅ Finished processing {valid_neurons_processed} neurons.")

    if valid_neurons_processed == 0:
        return None, None

    # 4. Final Normalization
    # Metric: (Sum of branch volumes) / (Number of cells * Voxel Volume)
    normalization_factor = valid_neurons_processed * voxel_volume
    
    axon_density_field = axon_grid_total / normalization_factor
    dend_density_field = dend_grid_total / normalization_factor

    return axon_density_field, dend_density_field

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_spanning_field(density_field, plot_mode='2d', voxel_size=10.0, grid_extent=1000.0, 
                        title="Neural Spanning Field", cmap='Reds', projection_type='sum', 
                        opacity_scale=0.2, min_density_threshold=1e-7):
    """
    Renders the neural spanning field in either 2D projections or a 3D interactive volume.
    
    Parameters:
    - density_field: 3D numpy array generated by calculate_spanning_fields.
    - plot_mode: '2d' (RAM-safe Matplotlib projections) or '3d' (Interactive Plotly Volume).
    - voxel_size: Edge length of the voxel in um (required for 3D alignment).
    - grid_extent: The bounding box limit in um.
    - title: Title of the plot.
    - cmap: Colormap string (e.g., 'Reds', 'Blues', 'viridis'). Works for both modes.
    - projection_type: 'sum' or 'max' (Only applies if plot_mode='2d').
    - opacity_scale: Global transparency (Only applies if plot_mode='3d').
    - min_density_threshold: Hides empty voxels (Only applies if plot_mode='3d').
    """
    
    if density_field is None or np.max(density_field) == 0:
        print("⚠️ The density field is empty. Nothing to plot.")
        return

    # ==========================================
    # MODE: 2D MATPLOTLIB PROJECTIONS
    # ==========================================
    if plot_mode.lower() == '2d':
        print(f"Preparing {projection_type} 2D projections...")
        
        if projection_type == 'sum':
            xy_proj = np.sum(density_field, axis=2)
            xz_proj = np.sum(density_field, axis=1)
            yz_proj = np.sum(density_field, axis=0)
        elif projection_type == 'max':
            xy_proj = np.max(density_field, axis=2)
            xz_proj = np.max(density_field, axis=1)
            yz_proj = np.max(density_field, axis=0)
        else:
            raise ValueError("projection_type must be 'sum' or 'max'")

        bounds = [-grid_extent, grid_extent, -grid_extent, grid_extent]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{title} ({projection_type.capitalize()} Projection)", fontsize=16, fontweight='bold')

        # Panel 1: X-Y Plane (Top-Down)
        im1 = axes[0].imshow(xy_proj.T, origin='lower', extent=bounds, cmap=cmap)
        axes[0].set_title("X-Y Plane (Top-Down)")
        axes[0].set_xlabel("X (um)")
        axes[0].set_ylabel("Y (um)")
        axes[0].grid(False)

        # Panel 2: X-Z Plane (Depth)
        im2 = axes[1].imshow(xz_proj.T, origin='lower', extent=bounds, cmap=cmap)
        axes[1].set_title("X-Z Plane (Depth)")
        axes[1].set_xlabel("X (um)")
        axes[1].set_ylabel("Z (um)")
        axes[1].grid(False)

        # Panel 3: Y-Z Plane (Depth)
        im3 = axes[2].imshow(yz_proj.T, origin='lower', extent=bounds, cmap=cmap)
        axes[2].set_title("Y-Z Plane (Depth)")
        axes[2].set_xlabel("Y (um)")
        axes[2].set_ylabel("Z (um)")
        axes[2].grid(False)

        cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), pad=0.02, shrink=0.8)
        if projection_type == 'sum':
            cbar.set_label('Total Fractional Volume per Column', rotation=270, labelpad=20)
        else:
            cbar.set_label('Max Fractional Volume Voxel', rotation=270, labelpad=20)

        plt.show()

    # ==========================================
    # MODE: 3D INTERACTIVE PLOTLY
    # ==========================================
    elif plot_mode.lower() == '3d':
        print("Preparing data for 3D rendering...")
        
        axis_vals = np.arange(-grid_extent, grid_extent, voxel_size) + (voxel_size / 2.0)
        X, Y, Z = np.meshgrid(axis_vals, axis_vals, axis_vals, indexing='ij')

        # Flatten strictly to maintain grid architecture for go.Volume
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        val_flat = density_field.flatten()
        
        max_val = val_flat.max()

        if max_val < min_density_threshold:
            print(f"⚠️ Max density ({max_val:.7f}) is below the threshold ({min_density_threshold}).")
            print("Try lowering min_density_threshold.")
            return
            
        print(f"Rendering volume... (Max density: {max_val:.5f})")

        fig = go.Figure(data=go.Volume(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            value=val_flat,
            isomin=min_density_threshold, 
            isomax=max_val,               
            opacity=opacity_scale,        
            surface_count=15,             
            colorscale=cmap,        
            caps=dict(x_show=False, y_show=False, z_show=False), 
            colorbar=dict(title='Fractional<br>Volume')
        ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (um)',
                yaxis_title='Y (um)',
                zaxis_title='Z (um)',
                aspectmode='data',       
                bgcolor='white'          
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )

        fig.show()
        
    else:
        raise ValueError("plot_mode must be either '2d' or '3d'")
