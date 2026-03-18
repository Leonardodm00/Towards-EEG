import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from scipy.interpolate import splprep, splev, interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def generate_smooth_tube(points, radii, num_interp_points=100, num_radial_points=20):
    """Helper function to generate the 3D surface coordinates of the spine."""
    points, radii = np.array(points), np.array(radii)
    num_pts = len(points)
    if num_pts < 2: return None, None, None
        
    k = min(3, num_pts - 1) 
    tck, u = splprep(points.T, s=0, k=k)
    u_new = np.linspace(0, 1, num_interp_points)
    smooth_path = np.array(splev(u_new, tck)).T
    
    interp_kind = 'cubic' if num_pts > 3 else 'linear'
    r_interp = interp1d(u, radii, kind=interp_kind)(u_new)
    r_interp = np.clip(r_interp, a_min=1.0, a_max=None) 
    
    tangents = np.gradient(smooth_path, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms == 0] = 1 
    tangents = tangents / norms[:, np.newaxis]
    
    normals, binormals = np.zeros_like(smooth_path), np.zeros_like(smooth_path)
    t0 = tangents[0]
    v = np.array([1.0, 0.0, 0.0]) if not np.allclose(np.abs(t0), [1.0, 0.0, 0.0], atol=1e-2) else np.array([0.0, 1.0, 0.0])
    n0 = np.cross(t0, v)
    n0 /= np.linalg.norm(n0)
    normals[0], binormals[0] = n0, np.cross(t0, n0)
    
    for i in range(1, num_interp_points):
        t_prev, t_curr, n_prev = tangents[i-1], tangents[i], normals[i-1]
        axis = np.cross(t_prev, t_curr)
        sin_angle = np.linalg.norm(axis)
        cos_angle = np.dot(t_prev, t_curr)
        
        if sin_angle > 1e-6:
            axis /= sin_angle
            angle = np.arctan2(sin_angle, cos_angle)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            n_curr = R @ n_prev
        else:
            n_curr = n_prev
            
        normals[i], binormals[i] = n_curr, np.cross(t_curr, n_curr)
        
    theta = np.linspace(0, 2 * np.pi, num_radial_points)
    Theta, _ = np.meshgrid(theta, np.arange(num_interp_points))
    X, Y, Z = np.zeros_like(Theta, dtype=float), np.zeros_like(Theta, dtype=float), np.zeros_like(Theta, dtype=float)
    
    for i in range(num_interp_points):
        circ_x, circ_y = r_interp[i] * np.cos(theta), r_interp[i] * np.sin(theta)
        X[i, :] = smooth_path[i, 0] + circ_x * normals[i, 0] + circ_y * binormals[i, 0]
        Y[i, :] = smooth_path[i, 1] + circ_x * normals[i, 1] + circ_y * binormals[i, 1]
        Z[i, :] = smooth_path[i, 2] + circ_x * normals[i, 2] + circ_y * binormals[i, 2]
        
    return X, Y, Z


def debug_spine_morphology(neuron_dict, neuron_id, n_samples=3, num_interp_points=100, smoothing_sigma=2.0, show_plots=True):
    """
    Quantifies spine morphology and visually plots the 3D surface alongside 
    its 1D radius profile. Includes a 1/3 length fallback for plain spines.
    """
    if neuron_id not in neuron_dict:
        print(f"⚠️ Neuron {neuron_id} not found.")
        return pd.DataFrame()
        
    df = neuron_dict[neuron_id].copy()
    if 'r' not in df.columns: df['r'] = 50.0

    node_dict = df.set_index('id').to_dict('index')
    children_map = defaultdict(list)
    for _, row in df[df['p'] != -1].iterrows():
        children_map[row['p']].append(row['id'])

    spine_roots = [
        nid for nid, data in node_dict.items() 
        if data.get('annotated_type') == 'spine' and 
        node_dict.get(data.get('p'), {}).get('annotated_type') != 'spine'
    ]

    if not spine_roots:
        print(f"🛑 No spines found in Neuron {neuron_id}.")
        return pd.DataFrame()

    if n_samples is not None:
        sampled_roots = random.sample(spine_roots, min(n_samples, len(spine_roots)))
        print(f"🔬 Debug Mode: Analyzing {len(sampled_roots)} randomly sampled spines from {len(spine_roots)} total.")
    else:
        sampled_roots = spine_roots

    metrics_records = []

    for root_id in sampled_roots:
        paths = []
        def dfs_paths(current_node, current_path):
            current_path.append(current_node)
            spine_children = [c for c in children_map[current_node] if node_dict[c]['annotated_type'] == 'spine']
            if not spine_children:
                paths.append(list(current_path))
            else:
                for child in spine_children:
                    dfs_paths(child, list(current_path))
        
        dfs_paths(root_id, [])

        for path_idx, path in enumerate(paths):
            path_coords, path_radii = [], []
            
            for node in path:
                data = node_dict[node]
                coord = np.array([data['x'], data['y'], data['z']])
                if not path_coords or np.linalg.norm(coord - path_coords[-1]) > 1e-4:
                    path_coords.append(coord)
                    path_radii.append(data['r'])
                    
            if len(path_coords) < 3: continue

            path_coords, path_radii = np.array(path_coords), np.array(path_radii)
            
            k = min(3, len(path_coords) - 1)
            tck, u = splprep(path_coords.T, s=0, k=k)
            u_new = np.linspace(0, 1, num_interp_points)
            smooth_coords = np.array(splev(u_new, tck)).T
            interp_kind = 'cubic' if len(path_coords) > 3 else 'linear'
            smooth_radii = interp1d(u, path_radii, kind=interp_kind)(u_new)
            smooth_radii = np.clip(smooth_radii, a_min=1.0, a_max=None)
            
            diffs = np.diff(smooth_coords, axis=0)
            ds = np.linalg.norm(diffs, axis=1)
            ds = np.insert(ds, 0, 0) 
            
            radii_tip_to_base = smooth_radii[::-1]
            ds_tip_to_base = ds[::-1]
            dist_from_tip = np.cumsum(ds_tip_to_base) 
            
            filtered_radii = gaussian_filter1d(radii_tip_to_base, sigma=smoothing_sigma)
            
            prominence_threshold = np.max(filtered_radii) * 0.05 
            minima, _ = find_peaks(-filtered_radii, prominence=prominence_threshold)
            
            # --- Boundary Identification Logic ---
            total_length = dist_from_tip[-1]
            if len(minima) > 0:
                neck_start_idx = minima[0]
                boundary_label = 'True Minimum (Neck)'
                marker_color = 'crimson'
                spine_class = 'Mushroom/Thin'
            else:
                # Fallback: 1/3 of the total length from the tip
                target_dist = total_length / 3.0
                # Find the array index closest to the 1/3 distance
                neck_start_idx = np.searchsorted(dist_from_tip, target_dist)
                # Ensure we don't go out of bounds
                neck_start_idx = min(neck_start_idx, len(dist_from_tip) - 1)
                
                boundary_label = 'Estimated Neck (1/3 Length Fallback)'
                marker_color = 'darkorange'
                spine_class = 'Stubby/Plain'

            # --- Generate the Debug Plot ---
            if show_plots:
                fig = make_subplots(
                    rows=1, cols=2, 
                    specs=[[{'is_3d': True}, {'type': 'xy'}]],
                    subplot_titles=(f'3D Reconstruction (Class: {spine_class})', 'Radius Profile (Tip to Base)'),
                    column_widths=[0.4, 0.6]
                )
                
                # 1. 3D Plot
                X, Y, Z = generate_smooth_tube(path_coords, path_radii, num_interp_points)
                if X is not None:
                    color_grid = np.zeros_like(X)
                    
                    # Apply coloring based on our determined neck_start_idx
                    split_idx = num_interp_points - 1 - neck_start_idx
                    color_grid[split_idx:, :] = 1 # Head is Gold (1)
                    color_grid[:split_idx, :] = 0 # Neck is Indigo (0)
                        
                    custom_colorscale = [[0, 'indigo'], [1, 'gold']]
                    
                    fig.add_trace(go.Surface(
                        x=X, y=Y, z=Z, 
                        surfacecolor=color_grid,
                        colorscale=custom_colorscale, 
                        showscale=False,
                        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5, specular=0.3),
                        scene='scene1' 
                    ))
                    
                    fig.update_layout(
                        scene=dict(aspectmode='data', xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)')
                    )
                
                # 2. 2D Profile Plot
                fig.add_trace(go.Scatter(
                    x=dist_from_tip, y=radii_tip_to_base, 
                    mode='lines', name='Raw Interpolated Radius',
                    line=dict(color='lightgrey', dash='dash')
                ), row=1, col=2)
                
                fig.add_trace(go.Scatter(
                    x=dist_from_tip, y=filtered_radii, 
                    mode='lines', name='Gaussian Smoothed (Algorithmic Input)',
                    line=dict(color='royalblue', width=3)
                ), row=1, col=2)
                
                # 3. Mark the boundary (True Minimum or 1/3 Fallback)
                min_x = dist_from_tip[neck_start_idx]
                min_y = filtered_radii[neck_start_idx]
                
                fig.add_trace(go.Scatter(
                    x=[min_x], y=[min_y], 
                    mode='markers', name=boundary_label,
                    marker=dict(color=marker_color, size=12, symbol='x')
                ), row=1, col=2)
                
                # Add background shading with the bypass for the Plotly bug
                fig.add_vrect(
                    x0=0, x1=min_x, fillcolor="gold", opacity=0.1, layer="below", 
                    line_width=0, annotation_text="Head", row=1, col=2, 
                    exclude_empty_subplots=False
                )
                fig.add_vrect(
                    x0=min_x, x1=dist_from_tip[-1], fillcolor="indigo", opacity=0.1, layer="below", 
                    line_width=0, annotation_text="Neck", row=1, col=2, 
                    exclude_empty_subplots=False
                )

                fig.update_layout(height=500, width=1200, template='plotly_white')
                fig.update_xaxes(title_text='Distance from Spine Tip (nm)', row=1, col=2)
                fig.update_yaxes(title_text='Radius (nm)', row=1, col=2)

                fig.show()

            # Record basic stats for debug return
            metrics_records.append({
                'spine_root_id': root_id, 
                'branch_idx': path_idx, 
                'class': spine_class,
                'head_len': dist_from_tip[neck_start_idx],
                'neck_len': total_length - dist_from_tip[neck_start_idx]
            })

    return pd.DataFrame(metrics_records)
# Example Usage:
# debug_spine_morphology(spine_df_dict, target_ids[0], n_samples=3)

# Example Usage:
df_metrics = debug_spine_morphology(spine_df_dict, target_ids[0], n_samples=10,smoothing_sigma=1)
