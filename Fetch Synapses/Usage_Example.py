import os
import LFPy
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def fetch_mapped_synapse_indices(syn_df, syn_type, z_min=-np.inf, z_max=np.inf):
    """
    Retrieves pre-calculated LFPy segment indices from a mapped synapse database 
    based on biological classification and spatial constraints.

    This function bypasses the need for stochastic or area-based synapse placement 
    by leveraging 'pre-baked' spatial snapping data. It allows for the deterministic 
    reconstruction of a connectome within specific cortical layer boundaries.

    Parameters:
    -----------
    syn_df : pandas.DataFrame
        The mapped synapse DataFrame containing at least 'synapse_type', 
        'z' (aligned depth in um), and 'lfpy_idx' (1D segment index).
    syn_type : str
        The biological class of synapses to fetch. Must be 'exc' (excitatory) 
        or 'inh' (inhibitory).
    z_min : float, optional
        The lower (more negative) vertical boundary in micrometers. 
        Defaults to -infinity.
    z_max : float, optional
        The upper (closer to 0) vertical boundary in micrometers. 
        Defaults to +infinity.

    Returns:
    --------
    numpy.ndarray
        An array of integers representing the 1D LFPy segment indices 
        where synapses should be instantiated.
    """
    if syn_type not in ['exc', 'inh']:
        raise ValueError(f"Invalid syn_type '{syn_type}'. Must be 'exc' or 'inh'.")

    # 1. Classification Filtering
    # We isolate only the synapses that match the requested biological type.
    type_mask = syn_df['synapse_type'] == syn_type
    
    # 2. Spatial Filtering (Cortical Layer Slicing)
    # We define a vertical window. Only synapses whose aligned 'z' coordinate 
    # falls within [z_min, z_max] are selected.
    depth_mask = (syn_df['z'] >= z_min) & (syn_df['z'] <= z_max)
    
    # 3. Vectorized Index Extraction
    # We apply both masks and extract the 'lfpy_idx' column as an integer array.
    filtered_df = syn_df[type_mask & depth_mask]
    syn_indices = filtered_df['lfpy_idx'].values.astype(int)
    
    return syn_indices

def interactive_debug_morphology():
    # 1. Define paths 
    base_dir = "/content/drive/MyDrive/Colab Notebooks"
    neuron_id = "3661346815"
    
    hoc_filepath = os.path.join(base_dir, "Aligned Neurons HOC", f"neuron_{neuron_id}_aligned.hoc")
    csv_file_path = os.path.join(base_dir, "Synapse database", f"neuron_{neuron_id}_mapped_synapses.csv")

    if not os.path.exists(hoc_filepath):
        print(f"⚠️ Error: File '{hoc_filepath}' not found.")
        return None

    print(f"Loading {os.path.basename(hoc_filepath)} into NEURON...")

    # Load with pt3d=True to get smooth, biological curves
    cell_parameters = {
        'morphology': hoc_filepath,
        'passive': True,
        'nsegs_method': 'lambda_f',
        'pt3d': True, 
        'dt': 0.125,
        'tstart': 0.,
        'tstop': 50.,
        'v_init': -65.,
        'verbose': False
    }

    try:
        cell = LFPy.Cell(**cell_parameters)
        print(f"✅ Success! Cell loaded with {cell.totnsegs} active compartments.")
    except Exception as e:
        print(f"❌ Failed to load cell into LFPy. Error: {e}")
        return None

    if not os.path.exists(csv_file_path):
        print(f"⚠️ Failed to find synapse CSV: {csv_file_path}")
        return None
    syn_df = pd.read_csv(csv_file_path)

    # 3. Define layer boundaries
    layer_boundaries = np.array([
        [0.0, -81.6],
        [-81.6, -587.1],
        [-587.1, -922.2],
        [-922.2, -1170.0],
        [-1170.0, -1491.7]
    ])
    layer_names = ['L1', 'L2/3', 'L4', 'L5', 'L6']

    # ---------------------------------------------------------
    # 4. SOMA PLACEMENT & DEPTH SHIFT
    # ---------------------------------------------------------
    soma_layer_idx = 2 # Placing soma in L4
    soma_z_placement = (layer_boundaries[soma_layer_idx].max() + layer_boundaries[soma_layer_idx].min()) / 2.0
    
    print(f"\nPlacing Soma in {layer_names[soma_layer_idx]} (Depth: {soma_z_placement:.1f} um)")

    # Shift the entire LFPy cell's Z coordinates down to this depth
    cell.set_pos(x=0, y=0, z=soma_z_placement)
    
    # Shift the synapse CSV Z coordinates to match
    syn_df['z'] += soma_z_placement

    # Calculate midpoints for plotting the synapses
    x_mids = cell.x.mean(axis=-1)
    y_mids = cell.y.mean(axis=-1)
    z_mids = cell.z.mean(axis=-1)

    # ---------------------------------------------------------
    # 5. FETCH SYNAPSES FOR A TARGET LAYER
    # ---------------------------------------------------------
    target_synapse_layer_idx = 1 # Highlighting synapses in L2/3
    z_max = layer_boundaries[target_synapse_layer_idx].max() 
    z_min = layer_boundaries[target_synapse_layer_idx].min() 

    exc_indices = fetch_mapped_synapse_indices(syn_df, 'exc', z_min=z_min, z_max=z_max)
    inh_indices = fetch_mapped_synapse_indices(syn_df, 'inh', z_min=z_min, z_max=z_max)

    # ---------------------------------------------------------
    # 6. GENERATE INTERACTIVE 3D PLOT
    # ---------------------------------------------------------
    print("\n--- Generating Interactive 3D Plot ---")
    fig = go.Figure()

    # --- Trace 1: Neuron Morphology (pt3d) ---
    skel_x, skel_y, skel_z = [], [], []
    for x, y, z in zip(cell.x3d, cell.y3d, cell.z3d):
        skel_x.extend(x.tolist() + [None])
        skel_y.extend(y.tolist() + [None])
        skel_z.extend(z.tolist() + [None])

    fig.add_trace(go.Scatter3d(
        x=skel_x, y=skel_y, z=skel_z,
        mode='lines',
        line=dict(color='darkslategrey', width=4),
        opacity=0.6,
        name='Morphology',
        hoverinfo='none'
    ))

    # --- Trace 2: Soma ---
    fig.add_trace(go.Scatter3d(
        x=[cell.somapos[0]], y=[cell.somapos[1]], z=[cell.somapos[2]],
        mode='markers',
        marker=dict(size=8, color='crimson', symbol='circle'),
        name=f'Soma ({layer_names[soma_layer_idx]})',
        hovertext='Soma',
        hoverinfo='text'
    ))

    # --- Trace 3: Excitatory Synapses ---
    if len(exc_indices) > 0:
        fig.add_trace(go.Scatter3d(
            x=x_mids[exc_indices], y=y_mids[exc_indices], z=z_mids[exc_indices],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Exc Synapses',
            hovertext=[f"LFPy Idx: {idx}" for idx in exc_indices],
            hoverinfo='text'
        ))

    # --- Trace 4: Inhibitory Synapses ---
    if len(inh_indices) > 0:
        fig.add_trace(go.Scatter3d(
            x=x_mids[inh_indices], y=y_mids[inh_indices], z=z_mids[inh_indices],
            mode='markers',
            marker=dict(size=5, color='blue', symbol='diamond'),
            name='Inh Synapses',
            hovertext=[f"LFPy Idx: {idx}" for idx in inh_indices],
            hoverinfo='text'
        ))

    # --- Trace 5: Layer Boundaries (Semi-transparent 3D planes) ---
    # FILTER OUT THE 'None' VALUES BEFORE CALCULATING MIN/MAX
    valid_x = [val for val in skel_x if val is not None]
    valid_y = [val for val in skel_y if val is not None]
    
    x_min, x_max = np.min(valid_x), np.max(valid_x)
    y_min, y_max = np.min(valid_y), np.max(valid_y)
    
    # Create the grid for the planes
    xx = np.array([[x_min, x_max], [x_min, x_max]])
    yy = np.array([[y_min, y_min], [y_max, y_max]])

    # Plot unique Z boundaries
    unique_z_bounds = np.unique(layer_boundaries)
    
    for z_bound in unique_z_bounds:
        zz = np.full((2, 2), z_bound)
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=zz,
            colorscale=[[0, 'lightgrey'], [1, 'lightgrey']],
            opacity=0.2,
            showscale=False,
            hoverinfo='none',
            name='Layer Boundary'
        ))

    # Add text labels for the layers on the edge of the planes
    layer_label_z = [(bounds[0] + bounds[1])/2 for bounds in layer_boundaries]
    fig.add_trace(go.Scatter3d(
        x=[x_min] * len(layer_names),
        y=[y_min] * len(layer_names),
        z=layer_label_z,
        mode='text',
        text=layer_names,
        textposition='middle left',
        textfont=dict(color='black', size=14, family='Arial Black'),
        showlegend=False,
        hoverinfo='none'
    ))

    # ---------------------------------------------------------
    # 7. FINAL LAYOUT & CAMERA
    # ---------------------------------------------------------
    fig.update_layout(
        title=f"Interactive Morphology: {neuron_id}<br><sup>Soma in {layer_names[soma_layer_idx]} | Synapses in {layer_names[target_synapse_layer_idx]}</sup>",
        scene=dict(
            xaxis_title='X (um)',
            yaxis_title='Y (um)',
            zaxis_title='Depth Z (um)',
            aspectmode='data', 
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=60)
    )

    fig.show()

    return cell

if __name__ == '__main__':
    cell = interactive_debug_morphology()


