import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def map_synapses_to_segments(neuron_id, skeletons_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons', synapses_dir='./h01_extracted_synapses'):
    """
    Loads a neuron skeleton and its associated synapses based on the neuron ID,
    finds the closest neuron segment for each incoming synapse, and 
    labels it as 'exc_syn' (Type 2) or 'inh_syn' (Type 1).
    """
    skel_path = os.path.join(skeletons_dir, f"neuron_{neuron_id}.csv")
    syn_path = os.path.join(synapses_dir, f"neuron_{neuron_id}_synapses.csv")
    
    # 1. Check if files exist
    if not os.path.exists(skel_path):
        print(f"⚠️ Skeleton file not found for neuron {neuron_id}: {skel_path}")
        return None
        
    neuron_df = pd.read_csv(skel_path)
    
    if not os.path.exists(syn_path):
        print(f"⚠️ Synapses file not found for neuron {neuron_id}: {syn_path}")
        return neuron_df # Return the unmodified skeleton

    # 2. Load and filter synapse data
    syn_df = pd.read_csv(syn_path)
    syn_df = syn_df[syn_df['direction'] == 'incoming'].dropna(subset=['location_x', 'location_y', 'location_z'])
    
    if syn_df.empty:
        print(f"⚠️ No valid incoming synapses found to map for neuron {neuron_id}.")
        return neuron_df

    # 3. Build a KD-Tree using the RAW neuron coordinates
    neuron_coords = neuron_df[['x', 'y', 'z']].values
    tree = cKDTree(neuron_coords)
    
    # 4. Get the synapse coordinates AND convert from voxels to nanometers
    syn_coords = syn_df[['location_x', 'location_y', 'location_z']].values
    syn_coords[:, 0] = syn_coords[:, 0]* 8.0   # Scale X
    syn_coords[:, 1] = syn_coords[:, 1]* 8.0   # Scale Y
    syn_coords[:, 2] = syn_coords[:, 2]* 33.0  # Scale Z
    
    # 5. Query the KD-Tree
    distances, closest_node_indices = tree.query(syn_coords)
    
    # 6. Initialize label column
    if 'synapse_label' not in neuron_df.columns:
        neuron_df['synapse_label'] = None
        
    # 7. Map labels
    for i, node_idx in enumerate(closest_node_indices):
        df_idx = neuron_df.index[node_idx]
        raw_type = str(syn_df.iloc[i]['synapse_type']).lower()
        
        # Type 2 = Excitatory, Type 1 = Inhibitory
        if any(keyword in raw_type for keyword in ['2', 'exc', 'asymmetric']):
            label = 'exc_syn'
        elif any(keyword in raw_type for keyword in ['1', 'inh', 'symmetric']):
            label = 'inh_syn'
        else:
            label = 'unknown_syn'
            
        neuron_df.at[df_idx, 'synapse_label'] = label
        
    print(f"✅ Successfully mapped {len(syn_coords)} synapses to neuron {neuron_id}.")
    
    return neuron_df
import plotly.graph_objects as go
import pandas as pd

def plot_labeled_arbor(neuron_df, title="Neuron Arbor & Synapse Labels"):
    """
    Plots a visually striking 3D arbor of a neuron, color-coded by structural 
    and synaptic labels. Uses a dark theme to make the synapses "glow".
    Point markers have been removed for a cleaner, segment-only look.
    """
    df = neuron_df.copy()
    
    if 'synapse_label' in df.columns:
        df['plot_label'] = df['synapse_label'].fillna(df['annotated_type'])
    else:
        df['plot_label'] = df['annotated_type']

    # 1. High-contrast "Neon" Color Palette
    color_map = {
        'soma': '#FFFFFF',          # Pure White
        'axon': '#B388FF',          # Electric Purple
        'dendrite': '#5C6BC0',      # Muted Indigo/Blue
        'apical': '#7986CB',        # Lighter Indigo
        'basal': '#3F51B5',         # Darker Indigo
        'exc_syn': '#FF1744',       # Neon Red/Pink
        'inh_syn': '#00E5FF',       # Neon Cyan
        'unknown_syn': '#FFEA00'    # Neon Yellow
    }

    fig = go.Figure()
    node_map = df.set_index('id')[['x', 'y', 'z']].to_dict('index')
    unique_labels = df['plot_label'].dropna().unique()

    # 2. Build Traces (Lines only)
    for label in unique_labels:
        subset = df[df['plot_label'] == label]
        label_str = str(label).lower()
        
        # Determine color
        color = color_map.get(label_str, '#888888')
        if 'exc' in label_str: color = color_map['exc_syn']
        elif 'inh' in label_str: color = color_map['inh_syn']

        # Gather line segments
        cx, cy, cz = [], [], []
        for _, row in subset.iterrows():
            pid = row['p']
            if pid in node_map:
                parent = node_map[pid]
                cx.extend([row['x'], parent['x'], None])
                cy.extend([row['y'], parent['y'], None])
                cz.extend([row['z'], parent['z'], None])

        is_synapse = 'syn' in label_str

        # Plot the skeleton branches (Segments only)
        if len(cx) > 0:
            # Thicker, brighter lines for synapses; thin, slightly transparent for structure
            line_width = 5 if is_synapse else 2.5
            opacity = 1.0 if is_synapse else 0.45
            
            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz, mode='lines',
                line=dict(color=color, width=line_width),
                opacity=opacity,
                name=str(label).replace('_', ' ').title(),
                hoverinfo='skip'  # Kept simple without hover text since points are gone
            ))

    # 3. Clean, Dark Layout (Hide axes entirely for a "floating" look)
    no_axis = dict(
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        title=''
    )

    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=20), x=0.5),
        paper_bgcolor='#111111', 
        plot_bgcolor='#111111',
        scene=dict(
            xaxis=no_axis, 
            yaxis=no_axis, 
            zaxis=no_axis, 
            aspectmode='data', 
            bgcolor='#111111'
        ),
        width=1200, height=850,
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(
            itemsizing='constant', 
            font=dict(color='white', size=14),
            bgcolor='rgba(0,0,0,0.5)', 
            bordercolor='#444444',
            borderwidth=1
        )
    )
    
    fig.show()




neurondf = map_synapses_to_segments(5390777283, skeletons_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons', synapses_dir=output_dir)

plot_labeled_arbor(neurondf, title=f"Neuron {5390777283}: Arbor & Synapse Labels")
