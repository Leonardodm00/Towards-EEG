import os
import re
import numpy as np
import pandas as pd
import networkx as nx
import os
import re
import numpy as np
import pandas as pd


def label_dendritic_spines_robust(neuron_ids, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons', output_dir=None, spine_length_threshold_nm=3000.0):
    """
    Identifies dendritic spines by evaluating entire terminal subtrees.
    This safely handles bifurcating spines and reconstruction artifacts
    by summing the total length of all branches stemming from a main arbor.
    """
    updated_neurons = {}

    print(f"🔍 Searching for complex dendritic spines across {len(neuron_ids)} neurons (Threshold: {spine_length_threshold_nm} nm)...")

    for nid in neuron_ids:
        filepath = os.path.join(input_dir, f"neuron_{nid}.csv")

        if not os.path.exists(filepath):
            print(f"⚠️ File for neuron {nid} not found. Skipping.")
            continue

        df = pd.read_csv(filepath)

        if not {'id', 'p', 'x', 'y', 'z', 'annotated_type'}.issubset(df.columns):
            print(f"⚠️ Missing required columns in neuron {nid}. Skipping.")
            continue

        # 1. Build a Directed Graph
        G = nx.DiGraph()
        G.add_nodes_from(df['id'])
        # Only add valid edges (exclude the root node's parent placeholder)
        valid_edges = df[df['p'] != -1][['p', 'id']].values
        G.add_edges_from(valid_edges)

        node_coords = df.set_index('id')[['x', 'y', 'z']].to_dict('index')
        node_types = df.set_index('id')['annotated_type'].to_dict()

        spine_nodes = set()

        # 2. Identify all branch points in the neuron
        branch_points = [n for n in G.nodes() if G.out_degree(n) > 1]

        # 3. Evaluate the subtrees attached to each branch point
        for bp in branch_points:
            # Ensure the branch point is on a dendrite/apical section
            parent_type = str(node_types.get(bp, ''))
            if not re.search(r'dendrite|apical|^1$', parent_type, re.IGNORECASE):
                continue

            # Look at every branch sprouting from this branch point
            for child in G.successors(bp):
                # Isolate the entire subtree (the child and all its descendants)
                try:
                    subtree_nodes = nx.descendants(G, child)
                    subtree_nodes.add(child)
                except nx.NetworkXError:
                    continue

                # 4. Calculate the total cable length of this entire subtree
                total_subtree_length = 0.0
                for node in subtree_nodes:
                    # In a tree, every node (except root) has exactly 1 parent
                    parent = list(G.predecessors(node))[0]

                    p1 = np.array([node_coords[node]['x'], node_coords[node]['y'], node_coords[node]['z']])
                    p2 = np.array([node_coords[parent]['x'], node_coords[parent]['y'], node_coords[parent]['z']])

                    total_subtree_length += np.linalg.norm(p1 - p2)

                # 5. If the total sum of all branches is below threshold, it's a spine complex
                if 0 < total_subtree_length <= spine_length_threshold_nm:
                    spine_nodes.update(subtree_nodes)

        # 6. Apply labels and save
        if spine_nodes:
            df.loc[df['id'].isin(spine_nodes), 'annotated_type'] = 'spine'

        updated_neurons[nid] = df
        print(f"✅ Neuron {nid}: Identified and relabeled {len(spine_nodes)} spine nodes (including bifurcations).")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_filepath = os.path.join(output_dir, f"neuron_{nid}_spines.csv")
            df.to_csv(out_filepath, index=False)

    return updated_neurons

# Example Usage:
target_ids = [5629504348]
spine_df_dict = label_dendritic_spines_robust(target_ids, spine_length_threshold_nm=5000.0)

import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_color_coded_neurons(neuron_dict, neuron_ids):
    """
    Plots the 3D skeleton of neurons, color-coding sections by their 'annotated_type'.

    Parameters:
    - neuron_dict: Dictionary mapping neuron IDs to their respective pandas DataFrames.
    - neuron_ids: List of neuron IDs to plot.
    """

    # Define a color palette for different neural structures
    color_palette = {
        'soma': 'black',
        'axon': 'royalblue',
        'dendrite': 'crimson',
        'apical': 'mediumorchid',
        'spine': 'springgreen',
        'unclassified': 'lightgrey'
    }

    print(f"🎨 Plotting {len(neuron_ids)} color-coded neurons...")

    for nid in neuron_ids:
        if nid not in neuron_dict:
            print(f"⚠️ Neuron {nid} not found in the provided dictionary. Skipping.")
            continue

        df = neuron_dict[nid]

        if not {'id', 'p', 'x', 'y', 'z', 'annotated_type'}.issubset(df.columns):
            print(f"⚠️ Missing required columns in neuron {nid}. Skipping.")
            continue

        fig = go.Figure()
        node_map = df.set_index('id').to_dict('index')

        # Group coordinates by type to minimize the number of Plotly traces (improves performance)
        traces_data = {key: {'x': [], 'y': [], 'z': []} for key in color_palette.keys()}

        # 1. Build line segments
        for _, row in df[df['p'] != -1].iterrows():
            pid = row['p']
            if pid in node_map:
                parent = node_map[pid]

                # Determine the category for coloring
                raw_type = str(row['annotated_type']).lower()
                assigned_type = 'unclassified'

                for key in color_palette.keys():
                    if key in raw_type:
                        assigned_type = key
                        break

                # Append coordinates separated by None to draw discrete line segments in one trace
                traces_data[assigned_type]['x'].extend([row['x'], parent['x'], None])
                traces_data[assigned_type]['y'].extend([row['y'], parent['y'], None])
                traces_data[assigned_type]['z'].extend([row['z'], parent['z'], None])

        # 2. Add traces to the figure
        for struct_type, coords in traces_data.items():
            if len(coords['x']) > 0:
                # Make spines slightly thicker for better visibility against the dendrites
                line_width = 4 if struct_type == 'spine' else 2

                fig.add_trace(go.Scatter3d(
                    x=coords['x'], y=coords['y'], z=coords['z'],
                    mode='lines',
                    line=dict(color=color_palette[struct_type], width=line_width),
                    name=struct_type.capitalize(),
                    hoverinfo='skip'
                ))

        # 3. Highlight the Soma specifically as a marker
        soma_nodes = df[df['annotated_type'].astype(str).str.contains('soma', case=False, na=False)]
        if not soma_nodes.empty:
            fig.add_trace(go.Scatter3d(
                x=soma_nodes['x'], y=soma_nodes['y'], z=soma_nodes['z'],
                mode='markers',
                marker=dict(size=6, color=color_palette['soma'], symbol='circle'),
                name='Soma Nodes'
            ))

        # 4. Layout configuration
        fig.update_layout(
            title=f"Neuron {nid} - Morphological Compartments",
            scene=dict(
                xaxis_title='X (nm)',
                yaxis_title='Y (nm)',
                zaxis_title='Z (nm)',
                aspectmode='data',
                bgcolor='white'
            ),
            height=800, width=1000,
            margin=dict(l=0, r=0, b=0, t=50),
            legend=dict(itemsizing='constant')
        )

        fig.show()

# Example Usage:
# Assuming `updated_neurons_dict` is the output from the `label_dendritic_spines` function
plot_color_coded_neurons(spine_df_dict, target_ids)
