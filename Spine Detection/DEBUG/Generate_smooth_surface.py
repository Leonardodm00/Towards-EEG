import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from scipy.interpolate import splprep, splev, interp1d

def generate_smooth_tube(points, radii, num_interp_points=50, num_radial_points=20):
    """
    Generates a continuous, smooth tubular mesh along a set of 3D points.
    Uses B-splines for path smoothing and interpolates radii along the curve.
    """
    points = np.array(points)
    radii = np.array(radii)
    
    # 1. Spline Interpolation for the Centerline
    # Determine spline degree (k) based on number of points
    num_pts = len(points)
    if num_pts < 2:
        return None, None, None
        
    k = min(3, num_pts - 1) 
    
    # Generate the smooth path
    tck, u = splprep(points.T, s=0, k=k)
    u_new = np.linspace(0, 1, num_interp_points)
    smooth_path = np.array(splev(u_new, tck)).T
    
    # 2. Interpolate the Radii
    # Smoothly transition the thickness of the spine along the new path
    r_interp = interp1d(u, radii, kind='linear' if k < 2 else 'cubic')(u_new)
    # Prevent negative radii artifacts from cubic interpolation
    r_interp = np.clip(r_interp, a_min=1.0, a_max=None) 
    
    # 3. Compute the Rotation Minimizing Frame (Parallel Transport)
    # This prevents the 3D tube from twisting unnaturally at sharp corners
    tangents = np.gradient(smooth_path, axis=0)
    norms = np.linalg.norm(tangents, axis=1)
    norms[norms == 0] = 1 # Prevent division by zero
    tangents = tangents / norms[:, np.newaxis]
    
    normals = np.zeros_like(smooth_path)
    binormals = np.zeros_like(smooth_path)
    
    # Define initial normal
    t0 = tangents[0]
    v = np.array([1.0, 0.0, 0.0]) if not np.allclose(np.abs(t0), [1.0, 0.0, 0.0], atol=1e-2) else np.array([0.0, 1.0, 0.0])
    n0 = np.cross(t0, v)
    n0 /= np.linalg.norm(n0)
    normals[0] = n0
    binormals[0] = np.cross(t0, n0)
    
    # Propagate the frame smoothly along the curve
    for i in range(1, num_interp_points):
        t_prev, t_curr = tangents[i-1], tangents[i]
        n_prev = normals[i-1]
        
        axis = np.cross(t_prev, t_curr)
        sin_angle = np.linalg.norm(axis)
        cos_angle = np.dot(t_prev, t_curr)
        
        if sin_angle > 1e-6:
            axis /= sin_angle
            angle = np.arctan2(sin_angle, cos_angle)
            # Rodrigues rotation matrix
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            n_curr = R @ n_prev
        else:
            n_curr = n_prev
            
        normals[i] = n_curr
        binormals[i] = np.cross(t_curr, n_curr)
        
    # 4. Generate the 3D Surface Grid
    theta = np.linspace(0, 2 * np.pi, num_radial_points)
    Theta, _ = np.meshgrid(theta, np.arange(num_interp_points))
    
    X = np.zeros_like(Theta, dtype=float)
    Y = np.zeros_like(Theta, dtype=float)
    Z = np.zeros_like(Theta, dtype=float)
    
    for i in range(num_interp_points):
        circ_x = r_interp[i] * np.cos(theta)
        circ_y = r_interp[i] * np.sin(theta)
        
        X[i, :] = smooth_path[i, 0] + circ_x * normals[i, 0] + circ_y * binormals[i, 0]
        Y[i, :] = smooth_path[i, 1] + circ_x * normals[i, 1] + circ_y * binormals[i, 1]
        Z[i, :] = smooth_path[i, 2] + circ_x * normals[i, 2] + circ_y * binormals[i, 2]
        
    return X, Y, Z

def sample_and_reconstruct_spines_interpolated(neuron_dict, neuron_id, n_samples=3):
    """
    Extracts spines and reconstructs them using continuous spline interpolation.
    """
    if neuron_id not in neuron_dict:
        print(f"⚠️ Neuron {neuron_id} not found.")
        return
        
    df = neuron_dict[neuron_id].copy()
    if 'r' not in df.columns:
        df['r'] = 50.0

    node_dict = df.set_index('id').to_dict('index')
    children_map = defaultdict(list)
    for _, row in df[df['p'] != -1].iterrows():
        children_map[row['p']].append(row['id'])

    # 1. Identify Spine Roots
    spine_roots = [
        nid for nid, data in node_dict.items() 
        if data.get('annotated_type') == 'spine' and 
        node_dict.get(data.get('p'), {}).get('annotated_type') != 'spine'
    ]

    if not spine_roots:
        print(f"🛑 No spines found in Neuron {neuron_id}.")
        return

    sampled_roots = random.sample(spine_roots, min(n_samples, len(spine_roots)))
    
    fig = make_subplots(
        rows=1, cols=len(sampled_roots), 
        specs=[[{'type': 'surface'}] * len(sampled_roots)],
        subplot_titles=[f"Interpolated Spine {i+1}" for i in range(len(sampled_roots))]
    )

    smooth_lighting = dict(ambient=0.5, diffuse=0.8, roughness=0.4, specular=0.3, fresnel=0.2)

    for i, root_id in enumerate(sampled_roots):
        
        # 2. Extract ordered paths from root to tips (handles branching spines)
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
        
        # Origin for centering
        parent_id = node_dict[root_id]['p']
        origin = np.array([node_dict[parent_id]['x'], node_dict[parent_id]['y'], node_dict[parent_id]['z']])

        # 3. Smooth and draw each continuous branch
        for path in paths:
            path_coords = []
            path_radii = []
            
            # Anchor the spine to the parent dendrite
            p_anchor = np.array([node_dict[parent_id]['x'], node_dict[parent_id]['y'], node_dict[parent_id]['z']]) - origin
            path_coords.append(p_anchor)
            path_radii.append(node_dict[root_id]['r']) # Use root radius to prevent swelling at base
            
            # Add spine nodes
            for node in path:
                data = node_dict[node]
                coord = np.array([data['x'], data['y'], data['z']]) - origin
                
                # Filter out consecutive duplicate coordinates (which crash the interpolator)
                if np.linalg.norm(coord - path_coords[-1]) > 1e-4:
                    path_coords.append(coord)
                    path_radii.append(data['r'])

            # Generate the interpolated mesh
            if len(path_coords) >= 2:
                X, Y, Z = generate_smooth_tube(path_coords, path_radii)
                
                if X is not None:
                    fig.add_trace(go.Surface(
                        x=X, y=Y, z=Z, 
                        colorscale='Viridis', 
                        showscale=False, 
                        lighting=smooth_lighting
                    ), row=1, col=i+1)
                    
                    # Optional: Add a smooth sphere cap to the very tip of the interpolated tube
                    u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:15j]
                    tip_center = path_coords[-1]
                    tip_r = path_radii[-1]
                    cap_x = tip_center[0] + tip_r * np.cos(u) * np.sin(v)
                    cap_y = tip_center[1] + tip_r * np.sin(u) * np.sin(v)
                    cap_z = tip_center[2] + tip_r * np.cos(v)
                    fig.add_trace(go.Surface(x=cap_x, y=cap_y, z=cap_z, colorscale='Viridis', showscale=False, lighting=smooth_lighting), row=1, col=i+1)

        fig.layout[f'scene{i+1 if i > 0 else ""}'].update(
            aspectmode='data',
            xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)'
        )

    fig.update_layout(
        title=f"Neuron {neuron_id}: Interpolated 3D Reconstructed Spines",
        height=600, width=500 * len(sampled_roots),
        margin=dict(l=0, r=0, b=50, t=80)
    )
    
    fig.show()
# Example Usage:
sample_and_reconstruct_spines_interpolated(spine_df_dict, neuron_id=target_ids[0], n_samples=5)
