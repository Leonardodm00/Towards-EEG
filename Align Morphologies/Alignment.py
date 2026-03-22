!pip install -q neuron LFPy

from google.colab import drive

# This will prompt you to authorize Colab to access your Drive
drive.mount('/content/drive')




import os
import ast
import re
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree  # <--- Added this!
import plotly.graph_objects as go
from plotly.subplots import make_subplots


import os
import pandas as pd
import numpy as np
import LFPy # Ensure LFPy is imported

def map_and_save_synapses_to_lfpy_idx(hoc_filepath, synapses_dir, neuron_id, soma_pos, rotation_matrix, voxel_res=(8.0, 8.0, 33.0)):
    """
    Transforms raw synapses into the aligned coordinate space, loads the aligned .hoc 
    into LFPy, and maps them to their closest valid 1D segment index.
    """
    syn_path = os.path.join(synapses_dir, f"neuron_{neuron_id}_synapses.csv")
    out_path = os.path.join(synapses_dir, f"neuron_{neuron_id}_mapped_synapses.csv")
    
    if not os.path.exists(syn_path):
        return

    # Load and filter raw synapses
    syn_df = pd.read_csv(syn_path)
    if 'direction' in syn_df.columns:
        syn_df = syn_df[syn_df['direction'] == 'incoming']
    syn_df = syn_df.dropna(subset=['location_x', 'location_y', 'location_z'])

    if syn_df.empty:
        return

    # --- THE TRANSFORMATION PIPELINE ---
    # 1. Extract raw coordinates and scale to nanometers
    syn_coords_nm = syn_df[['location_x', 'location_y', 'location_z']].values * np.array(voxel_res)
    
    # 2. Center using the exact same soma position as the skeleton
    centered_syns = syn_coords_nm - soma_pos
    
    # 3. Rotate using the exact same average matrix
    rotated_syns = np.dot(centered_syns, rotation_matrix.T)
    
    # 4. Scale to micrometers for NEURON/LFPy compatibility
    aligned_syns_um = rotated_syns / 1000.0
    # -----------------------------------

    # Instantiate the aligned cell in LFPy
    try:
        cell = LFPy.Cell(morphology=hoc_filepath,
                         passive=False, 
                         nsegs_method='lambda_f', # Ensure this matches simulation params!
                         delete_sections=True)
    except Exception as e:
        print(f"⚠️ Failed to load HOC into LFPy for {neuron_id}: {e}")
        return

    mapped_records = []

    # Map the ALIGNED synapses to the ALIGNED LFPy cell
    for i, (_, row) in enumerate(syn_df.iterrows()):
        syn_x, syn_y, syn_z = aligned_syns_um[i]

        flattened_idx = cell.get_closest_idx(x=syn_x, y=syn_y, z=syn_z)

        # Classify synapse type
        raw_type = str(row.get('synapse_type', '')).lower()
        if any(kw in raw_type for kw in ['2', 'exc', 'asymmetric']):
            syn_type = 'exc'
        elif any(kw in raw_type for kw in ['1', 'inh', 'symmetric']):
            syn_type = 'inh'
        else:
            syn_type = 'unknown'

        mapped_records.append({
            'lfpy_idx': int(flattened_idx),
            'x': syn_x,  # Saving the aligned coordinates for sanity checking later
            'y': syn_y,
            'z': syn_z,
            'synapse_type': syn_type
        })

    # Save the mapped dataset
    mapped_df = pd.DataFrame(mapped_records)
    mapped_df.to_csv(out_path, index=False)
    
    cell.__del__() 
    print(f"✅ Snapped {len(mapped_records)} aligned synapses to LFPy indices for {neuron_id}.")

def export_neuron_to_hoc(aligned_neuron_df, output_filepath):
    """
    Takes an aligned neuron DataFrame and writes a clean, NEURON-compliant .hoc file.
    Only uses standard biological sections (soma, axon, dend) to prevent cable fracturing.
    """
    df = aligned_neuron_df.copy()



    if 'r' not in df.columns:
        df['r'] = 0.5
    else:
        df['r'] = df['r'] / 1000.0

    # 1. Map labels to standard NEURON section arrays ONLY
    def get_hoc_type(row):
        annot_str = str(row.get('annotated_type', '')).lower()
        if 'axon' in annot_str: return 'axon'
        elif 'soma' in annot_str: return 'soma'
        else: return 'dend'

    df['hoc_type'] = df.apply(get_hoc_type, axis=1)

    # 2. Build graph relationships
    children = df.groupby('p')['id'].apply(list).to_dict()
    node_data = df.set_index('id').to_dict('index')

    root_rows = df[df['p'] == -1]
    if root_rows.empty:
        print("⚠️ Cannot export to HOC: No root node found.")
        return
    root_id = root_rows.iloc[0]['id']

    section_records = []

    # 3. Iterative traversal to extract unbranched sections
    stack = [(root_id, [], node_data[root_id]['hoc_type'], -1)]

    while stack:
        node_id, current_sec_nodes, current_type, parent_sec_id = stack.pop()
        current_sec_nodes.append(node_id)
        chs = children.get(node_id, [])

        if len(chs) == 0:
            section_records.append({'type': current_type, 'nodes': current_sec_nodes, 'parent_sec_id': parent_sec_id})
        elif len(chs) == 1:
            child = chs[0]
            child_type = node_data[child]['hoc_type']
            if child_type == current_type:
                stack.append((child, current_sec_nodes, current_type, parent_sec_id))
            else:
                sec_id = len(section_records)
                section_records.append({'type': current_type, 'nodes': current_sec_nodes, 'parent_sec_id': parent_sec_id})
                stack.append((child, [node_id], child_type, sec_id))
        else:
            sec_id = len(section_records)
            section_records.append({'type': current_type, 'nodes': current_sec_nodes, 'parent_sec_id': parent_sec_id})
            for child in chs:
                child_type = node_data[child]['hoc_type']
                stack.append((child, [node_id], child_type, sec_id))

    # 4. Count and index the section arrays
    counts = {'soma': 0, 'axon': 0, 'dend': 0}
    for sec in section_records:
        sec['type_idx'] = counts[sec['type']]
        counts[sec['type']] += 1

    # 5. Write the output
    with open(output_filepath, 'w') as f:
        f.write("// NEURON HOC morphology generated from aligned data\n\n")

        for t in ['soma', 'axon', 'dend']:
            if counts[t] > 0:
                f.write(f"create {t}[{counts[t]}]\n")
        f.write("\n")

        for sec in section_records:
            if sec['parent_sec_id'] != -1:
                parent_sec = section_records[sec['parent_sec_id']]
                f.write(f"connect {sec['type']}[{sec['type_idx']}](0), {parent_sec['type']}[{parent_sec['type_idx']}](1)\n")
        f.write("\n")

        for sec in section_records:
            f.write(f"{sec['type']}[{sec['type_idx']}] {{\n")
            f.write("  pt3dclear()\n")

            if len(sec['nodes']) == 1:
                node = sec['nodes'][0]
                d = node_data[node]
                r = d.get('r', 0.5)
                diam = r * 2
                f.write(f"  pt3dadd({d['x']}, {d['y']}, {d['z'] - r}, {diam})\n")
                f.write(f"  pt3dadd({d['x']}, {d['y']}, {d['z'] + r}, {diam})\n")
            else:
                for node in sec['nodes']:
                    d = node_data[node]
                    diam = d.get('r', 0.5) * 2
                    f.write(f"  pt3dadd({d['x']}, {d['y']}, {d['z']}, {diam})\n")
            f.write("}\n\n")




def align_neurons_to_neighborhood(neuron_ids, metadata_filepath, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons', outpath='/content/drive/MyDrive/Colab Notebooks/Aligned Neurons HOC', synapses_dir='', k_neighbors=3, show_plot=True):
    """
    Main loop: Centers the neuron, applies local rotation, scales to um, 
    exports clean HOC, and maps synapses directly to LFPy segment indices.
    """
    aligned_neurons = {}

    if not os.path.exists(metadata_filepath):
        print(f"⚠️ Metadata file not found: {metadata_filepath}")
        return aligned_neurons

    metadata_df = pd.read_csv(metadata_filepath)
    if isinstance(metadata_df['rotation_matrix'].iloc[0], str):
        metadata_df['rotation_matrix'] = metadata_df['rotation_matrix'].apply(ast.literal_eval)

    reference_somas = metadata_df[['soma_x', 'soma_y', 'soma_z']].values

    print(f"🔄 Processing {len(neuron_ids)} neurons...")
    os.makedirs(outpath, exist_ok=True)

    for nid in neuron_ids:
        filepath = os.path.join(input_dir, f"neuron_{nid}.csv")
        if not os.path.exists(filepath):
            print(f"⚠️ File for {nid} not found. Skipping.")
            continue

        df = pd.read_csv(filepath)
        
        root_rows = df[df['p'] == -1]
        if root_rows.empty:
            continue
        
        # 1. CENTER TO SOMA
        # The soma position is extracted in nanometers to match the raw skeleton
        soma_pos = root_rows.iloc[0][['x', 'y', 'z']].values.astype(float)
        centered_coords = df[['x', 'y', 'z']].values - soma_pos

        # 2. FIND NEIGHBORHOOD AND AVERAGE ROTATION
        distances = np.linalg.norm(reference_somas - soma_pos, axis=1)
        k_actual = min(k_neighbors, len(metadata_df))
        nearest_indices = np.argsort(distances)[:k_actual]
        nearest_metadata = metadata_df.iloc[nearest_indices]

        matrices = np.array(nearest_metadata['rotation_matrix'].tolist())
        mean_matrix = R.from_matrix(matrices).mean().as_matrix()

        # Apply local rotation
        rotated_coords = np.dot(centered_coords, mean_matrix.T)

        # 3. OVERWRITE DATAFRAME WITH ALIGNED COORDINATES & SCALE TO MICROMETERS
        df['x'] = rotated_coords[:, 0] / 1000.0
        df['y'] = rotated_coords[:, 1] / 1000.0
        df['z'] = rotated_coords[:, 2] / 1000.0
        aligned_neurons[nid] = df

        # 4. EXPORT CLEAN .HOC FILE
        hoc_filename = os.path.join(outpath, f"neuron_{nid}_aligned.hoc")
        export_neuron_to_hoc(df, hoc_filename)

        # 5. SNAP AND SAVE SYNAPSES DIRECTLY TO LFPY INDICES
        # We pass the raw soma_pos (nm) and mean_matrix so the mapping function 
        # can apply the exact same transformation pipeline to the synapses.
        map_and_save_synapses_to_lfpy_idx(
            hoc_filepath=hoc_filename, 
            synapses_dir=synapses_dir, 
            neuron_id=nid, 
            soma_pos=soma_pos,          
            rotation_matrix=mean_matrix 
        )

        print(f"✅ Neuron {nid} fully processed, aligned, and synapses snapped.")

        # 6. PLOTTING (Optional visual confirmation)
        if show_plot:
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scene'}, {'type': 'scene'}]],
                subplot_titles=(f"Pre-Rotation (Raw space)", f"Post-Rotation (Aligned space)")
            )

            # Left Panel: Centered Raw Skeleton
            df_temp = df[['id', 'p']].copy()
            df_temp['x'], df_temp['y'], df_temp['z'] = centered_coords[:, 0], centered_coords[:, 1], centered_coords[:, 2]

            node_map = df_temp.set_index('id')[['x', 'y', 'z']].to_dict('index')
            cx, cy, cz = [], [], []
            for _, row in df_temp[df_temp['p'] != -1].iterrows():
                pid = row['p']
                if pid in node_map:
                    parent = node_map[pid]
                    cx.extend([row['x'], parent['x'], None])
                    cy.extend([row['y'], parent['y'], None])
                    cz.extend([row['z'], parent['z'], None])

            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz, mode='lines', line=dict(color='lightgrey', width=2), 
                opacity=0.5, name='Raw Target', hoverinfo='skip'
            ), row=1, col=1)

            max_dist = np.max(np.linalg.norm(centered_coords, axis=1)) if len(centered_coords) > 0 else 50000
            stick_scale = max_dist * 0.4

            for i, (_, neighbor) in enumerate(nearest_metadata.iterrows()):
                n_soma_orig = np.array([neighbor['soma_x'], neighbor['soma_y'], neighbor['soma_z']])
                n_soma_rel = n_soma_orig - soma_pos
                v_com_orig = np.array([neighbor['v_com_x'], neighbor['v_com_y'], neighbor['v_com_z']])
                v_com_orig = v_com_orig / np.linalg.norm(v_com_orig)
                stick_end_raw = n_soma_rel + v_com_orig * stick_scale

                n_soma_rot = np.dot(n_soma_rel, mean_matrix.T)
                v_com_rot = np.dot(v_com_orig, mean_matrix.T)
                stick_end_rot = n_soma_rot + v_com_rot * stick_scale

                # Raw reference vectors
                fig.add_trace(go.Scatter3d(x=[n_soma_rel[0]], y=[n_soma_rel[1]], z=[n_soma_rel[2]], mode='markers', marker=dict(size=5, color='crimson'), name=f'N{i+1} Soma (Raw)'), row=1, col=1)
                fig.add_trace(go.Scatter3d(x=[n_soma_rel[0], stick_end_raw[0]], y=[n_soma_rel[1], stick_end_raw[1]], z=[n_soma_rel[2], stick_end_raw[2]], mode='lines', line=dict(color='crimson', width=4), showlegend=False), row=1, col=1)

                # Rotated reference vectors
                fig.add_trace(go.Scatter3d(x=[n_soma_rot[0]], y=[n_soma_rot[1]], z=[n_soma_rot[2]], mode='markers', marker=dict(size=5, color='darkorange'), name=f'N{i+1} Soma (Rotated)'), row=1, col=2)
                fig.add_trace(go.Scatter3d(x=[n_soma_rot[0], stick_end_rot[0]], y=[n_soma_rot[1], stick_end_rot[1]], z=[n_soma_rot[2], stick_end_rot[2]], mode='lines', line=dict(color='darkorange', width=4), showlegend=False), row=1, col=2)

            # Right Panel: Aligned Skeleton
            df_temp['x'], df_temp['y'], df_temp['z'] = rotated_coords[:, 0], rotated_coords[:, 1], rotated_coords[:, 2]
            node_map_rot = df_temp.set_index('id')[['x', 'y', 'z']].to_dict('index')
            rx, ry, rz = [], [], []
            for _, row in df_temp[df_temp['p'] != -1].iterrows():
                pid = row['p']
                if pid in node_map_rot:
                    parent = node_map_rot[pid]
                    rx.extend([row['x'], parent['x'], None])
                    ry.extend([row['y'], parent['y'], None])
                    rz.extend([row['z'], parent['z'], None])

            fig.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz, mode='lines', line=dict(color='royalblue', width=2), 
                opacity=0.6, name='Aligned Target', hoverinfo='skip'
            ), row=1, col=2)

            fig.update_layout(
                title=f"Neuron {nid} Alignment",
                scene=dict(aspectmode='data', bgcolor='white'),
                scene2=dict(aspectmode='data', bgcolor='white'),
                width=1400, height=700, margin=dict(l=0, r=0, b=0, t=50)
            )
            fig.show()

    return aligned_neurons






synapses_dir = '/content/drive/MyDrive/Colab Notebooks/Synapse database'
output_directory = '/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons'
output_filename = 'alignment_metadata_L4.csv'

metadata_filepath = os.path.join(output_directory, output_filename)


neuron_ids = [



           3661346815,





              ]
algn_neuron =align_neurons_to_neighborhood(neuron_ids, metadata_filepath, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons',synapses_dir = synapses_dir, k_neighbors=3, show_plot=True)

