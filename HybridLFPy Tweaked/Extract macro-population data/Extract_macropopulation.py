import numpy as np
import plotly.graph_objects as go
from collections import Counter

# 1. The Updated Extraction Function (Now includes 'cell_mtypes')
def extract_macro_populations(connectomics_data, name_list):
    """
    Extracts specific macro-populations using the fast lookup table, 
    returning only spatial coordinates, mapping indices, and connectivity dictionaries.
    """
    # Unpack only the necessary data for fast processing and requested outputs
    mtype_fast_lookup = connectomics_data['mtype_fast_lookup']
    cell_mtypes = connectomics_data['cell_mtypes']
    cell_coords = connectomics_data['cell_coords']
    post_to_pre = connectomics_data['post_to_pre']
    pre_to_post = connectomics_data['pre_to_post']
    
    extracted_data = {}
    
    for name in name_list:
        # 1. Parse the macro-population string (e.g., "L23_exc" -> "L23", "exc")
        parts = name.split('_')
        target_layer = parts[0].upper()
        target_group = parts[1].lower()
        
        global_indices = []
        
        # 2. Iterate through the array using the FAST LOOKUP
        for raw_idx, mtype in enumerate(cell_mtypes):
            if mtype not in mtype_fast_lookup:
                continue
                
            layer, bio_type = mtype_fast_lookup[mtype]
            
            # 3. Matching Logic
            is_match = False
            if layer == target_layer:
                if target_group == 'exc' and bio_type == 'Excitatory':
                    # Ensure L4_exc doesn't swallow L4_ss if both are requested
                    if target_layer == 'L4' and 'L4_ss' in name_list and 'SS' in mtype:
                        is_match = False 
                    else:
                        is_match = True
                elif target_group == 'inh' and bio_type == 'Inhibitory':
                    is_match = True
                elif target_group == 'ss' and 'SS' in mtype:
                    is_match = True
                    
            if is_match:
                global_indices.append(raw_idx)
                
        # 4. Process the matched sub-population
        global_indices = np.array(global_indices)
        
        if len(global_indices) == 0:
            print(f"Warning: No cells found for macro-population '{name}'")
            continue
            
        # Extract Coordinates
        sub_coords = cell_coords[global_indices]
        
        # Create Mapping: Local (0 to N) -> Raw Global Index
        local_to_raw_map = {local_idx: raw_idx for local_idx, raw_idx in enumerate(global_indices)}
        
       # Filter Connectivity Dictionaries (Keys are LOCAL, Values are RAW)
        sub_post_to_pre = {local_idx: post_to_pre[raw_idx] 
                           for local_idx, raw_idx in enumerate(global_indices) if raw_idx in post_to_pre}
        sub_pre_to_post = {local_idx: pre_to_post[raw_idx] 
                           for local_idx, raw_idx in enumerate(global_indices) if raw_idx in pre_to_post}
                           
        # 5. Pack strictly the requested keys into the return dictionary
        extracted_data[name] = {
            'cell_coords': sub_coords,
            'local_to_raw_map': local_to_raw_map,
            'post_to_pre': sub_post_to_pre,
            'pre_to_post': sub_pre_to_post
        }
        
    return extracted_data
# 2. The 3D Debug Plotting Function
def debug_plot_macro_populations(extracted_data, column_input):
    """
    Generates a 3D debug plot of extracted cells overlaid with theoretical layer bounds,
    and prints a detailed summary of the morphological types inside each macro-population.
    """
    fig = go.Figure()
    
    colors = ['#00CCFF', '#FF3366', '#33FF66', '#FFCC00', '#B84DFF', '#FFA500', '#FFD700', '#ADFF2F']
    radius = column_input['Geometry']['radius']
    
    print("==========================================")
    print("MACRO-POPULATION EXTRACTION DEBUG REPORT")
    print("==========================================\n")

    # A. Plot the extracted cells & print console report
    for i, (pop_name, data) in enumerate(extracted_data.items()):
        coords = data['cell_coords']
        mtypes = data['cell_mtypes']
        
        # Count the unique m-types to print to console
        mtype_counts = Counter(mtypes)
        print(f"--- {pop_name} (Total Cells: {len(coords)}) ---")
        for mt, count in mtype_counts.most_common():
            print(f"    • {mt:<12}: {count} cells")
        print("")
        
        # Subsample for Plotly performance if massive
        max_plot = 5000
        if len(coords) > max_plot:
            idx = np.random.choice(len(coords), max_plot, replace=False)
            plot_coords = coords[idx]
            plot_mtypes = mtypes[idx]
        else:
            plot_coords = coords
            plot_mtypes = mtypes

        # Add 3D scatter trace
        fig.add_trace(go.Scatter3d(
            x=plot_coords[:, 0], y=plot_coords[:, 1], z=plot_coords[:, 2],
            mode='markers', 
            name=f"{pop_name} (n={len(coords)})",
            text=plot_mtypes, 
            hoverinfo='text+name',
            marker=dict(size=3, color=colors[i % len(colors)], opacity=0.8)
        ))

    # B. Plot the Layer Boundaries from column_input
    theta = np.linspace(0, 2 * np.pi, 50)
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    
    for layer, bounds in column_input['Layers'].items():
        z_min, z_max = min(bounds), max(bounds)
        
        # Draw top and bottom planes for each layer
        for z_val in [z_min, z_max]:
            fig.add_trace(go.Scatter3d(
                x=x_circle, y=y_circle, z=np.full_like(x_circle, z_val),
                mode='lines',
                line=dict(color='rgba(255, 255, 255, 0.4)', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
            
        # Add a label in 3D space indicating the layer
        fig.add_trace(go.Scatter3d(
            x=[radius * 1.1], y=[0], z=[(z_min + z_max) / 2],
            mode='text',
            text=[f"<b>{layer}</b>"],
            textfont=dict(color='white', size=14),
            showlegend=False,
            hoverinfo='skip'
        ))

    # C. Layout Configurations
    fig.update_layout(
        title=dict(text="Extraction Debug: Spatial Bounding Check", font=dict(color='white', size=20)),
        scene=dict(
            xaxis=dict(title="X (µm)", backgroundcolor='black', gridcolor='#333', color='white'),
            yaxis=dict(title="Y (µm)", backgroundcolor='black', gridcolor='#333', color='white'),
            zaxis=dict(title="Depth Z (µm)", backgroundcolor='black', gridcolor='#333', color='white'),
            aspectmode='data'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show(renderer="colab")
