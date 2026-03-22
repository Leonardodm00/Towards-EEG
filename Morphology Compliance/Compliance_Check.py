from google.colab import drive
drive.mount('/content/drive')
!pip install neuron
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from neuron import h
import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from neuron import h

# Load standard NEURON run library
h.load_file('stdrun.hoc')

def _plot_capacity_at_z(dend_segments, syn_df, layer_boundaries, layer_names, valid_layers, target_z_pos, morph_idx):
    """Helper function to visualize the selected morphology and its synapse capacity."""
    fig, ax = plt.subplots(figsize=(10, 12))
    
    color_palette = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
    layer_colors = {layer_idx: color_palette[i % len(color_palette)] for i, layer_idx in enumerate(valid_layers)}
    
    # 1. Plot layer boundaries
    for i, bounds in enumerate(layer_boundaries):
        upper_bound, lower_bound = bounds
        ax.axhline(upper_bound, color='black', linestyle='--', alpha=0.5)
        
        bg_color = layer_colors.get(i, 'lightgrey')
        alpha_val = 0.15 if i in valid_layers else 0.05
        ax.axhspan(lower_bound, upper_bound, facecolor=bg_color, alpha=alpha_val)
        
        layer_label = layer_names[i] if i < len(layer_names) else f"Layer {i}"
        ax.text(-250, (upper_bound + lower_bound) / 2, layer_label, 
                va='center', ha='right', fontsize=12, fontweight='bold')
    
    ax.axhline(layer_boundaries[-1][1], color='black', linestyle='--', alpha=0.5)

    # 2. Plot dendritic segments (Light Grey)
    lines = []
    for p1, p2 in dend_segments:
        shifted_z1 = p1[2] + target_z_pos
        shifted_z2 = p2[2] + target_z_pos
        lines.append([(p1[0], shifted_z1), (p2[0], shifted_z2)])
        
    lc = LineCollection(lines, colors='darkgrey', linewidths=1.2, alpha=0.7)
    ax.add_collection(lc)
    
    # 3. Plot Mapped Synapses
    shifted_syn_z = syn_df['z'] + target_z_pos
    
    # Plot unused synapses in light grey
    ax.scatter(syn_df['x'], shifted_syn_z, color='lightgrey', s=8, alpha=0.5, label='Unused Synapses', zorder=3)
    
    # Plot valid synapses in their respective layer colors
    for layer_idx in valid_layers:
        upper_bound, lower_bound = layer_boundaries[layer_idx]
        mask = (shifted_syn_z >= lower_bound) & (shifted_syn_z <= upper_bound)
        
        if mask.sum() > 0:
            ax.scatter(syn_df.loc[mask, 'x'], shifted_syn_z[mask], 
                       color=layer_colors[layer_idx], s=25, 
                       label=f'Capacity in {layer_names[layer_idx]} ({mask.sum()})', zorder=4)

    # 4. Plot the soma
    ax.scatter([0], [target_z_pos], color='black', s=150, zorder=5, label='Soma')
    
    # 5. Final plot formatting
    ax.set_aspect('equal')
    ax.set_xlim(-350, 350) 
    ax.set_ylim(layer_boundaries[-1][1] - 50, layer_boundaries[0][0] + 50)
    ax.set_xlabel('X distance (um)', fontsize=12)
    ax.set_ylabel('Depth / Z distance (um)', fontsize=12)
    ax.set_title(f'Selected Morphology (ID: {morph_idx})\nCapacity Evaluated at Z = {target_z_pos:.1f} um', fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def cell_MorphSelect(
    morph_paths,
    layer_boundaries,
    layer_names,          # Added for plotting
    synapses_per_layer,
    target_z_pos,
    synapse_base_path,
    plot_result=False     # Added for debugging hook
):
    """
    Selects a valid neuronal morphology by ensuring its pre-mapped connectome 
    has enough physical synapse locations to satisfy the network's requirements 
    at a specific absolute cortical depth.

    Unlike geometric approaches that estimate viability based on dendritic length, 
    this function performs a direct empirical check. It virtually places the 
    neuron's soma at the target depth, shifts its pre-calculated synapse coordinates 
    accordingly, and counts the exact number of available docking sites falling 
    within the required cortical layers.

    Args:
        morph_paths (list of str): 
            A list of file paths to the candidate `.hoc` morphology files.
            Example: ['/path/neuron_123_aligned.hoc', '/path/neuron_456_aligned.hoc']
        layer_boundaries (numpy.ndarray): 
            A 2D array of shape (N, 2) defining the [upper_Z, lower_Z] boundaries 
            for each cortical layer in micrometers (μm).
            Example: [[0.0, -81.6], [-81.6, -587.1], ...]
        layer_names (list of str): 
            Display names for each layer, used strictly for annotating the debug plot.
            Example: ['L1', 'L2/3', 'L4', 'L5', 'L6']
        synapses_per_layer (list or numpy.ndarray of int): 
            The number of synapses the network model demands in each layer. 
            The length must exactly match the number of rows in `layer_boundaries`.
            Example: [0, 1500, 3200, 500, 0]
        target_z_pos (float): 
            The absolute Z-coordinate (in μm) where this specific cell's soma 
            will be placed in the simulated cortical column. 
        synapse_base_path (str): 
            The directory path containing the mapped synapse CSV files.
        plot_result (bool, optional): 
            If True, boots up the NEURON simulator to extract the 3D skeleton of 
            the winning morphology and generates a detailed visual report of the 
            synapse capacity across layers. Defaults to False.

    Returns:
        tuple (int, int): 
            - original_idx: The integer index of the winning morphology in the 
                            original `morph_paths` list.
            - nid: The extracted integer Neuron ID of the winning morphology.

    Raises:
        RuntimeError: 
            If the function evaluates every candidate morphology in the list and 
            none possess the required synapse capacity at the specified `target_z_pos`.
    """
    valid_layers = [i for i, syn in enumerate(synapses_per_layer) if syn > 0]
    shuffled_paths = list(enumerate(morph_paths))
    random.shuffle(shuffled_paths)
    
    for original_idx, path in shuffled_paths:
        print(f"Evaluating Original Index: {original_idx}")
        
        match = re.search(r'neuron_(\d+)', os.path.basename(path))
        nid = int(match.group(1)) if match else -1
        
        csv_path = os.path.join(synapse_base_path, f"neuron_{nid}_mapped_synapses.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: Synapse CSV not found for {nid} at {csv_path}")
            continue
            
        syn_df = pd.read_csv(csv_path)
        
        shifted_z = syn_df['z'] + target_z_pos
        is_viable = True
        
        for layer_idx in valid_layers:
            upper_bound, lower_bound = layer_boundaries[layer_idx]
            required_synapses = synapses_per_layer[layer_idx]
            
            available_synapses = ((shifted_z >= lower_bound) & (shifted_z <= upper_bound)).sum()
            
            if available_synapses < required_synapses:
                is_viable = False
                break 
                
        if is_viable:
            # ---> DEBUG PLOTTING HOOK <---
            if plot_result:
                print(f"✅ Winner found ({nid}). Extracting 3D geometry for plot...")
                h('forall delete_section()')
                success = h.load_file(str(path))
                
                if success:
                    dend_segments = []
                    for sec in h.allsec():
                        if 'dend' in sec.name().lower() or 'apic' in sec.name().lower():
                            n3d = int(h.n3d(sec=sec))
                            if n3d > 0:
                                pts = np.array([[h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)] for i in range(n3d)])
                                for i in range(n3d - 1):
                                    dend_segments.append((pts[i], pts[i+1]))
                    
                    _plot_capacity_at_z(
                        dend_segments, syn_df, layer_boundaries, layer_names, 
                        valid_layers, target_z_pos, nid
                    )
            # ------------------------------
            return original_idx, nid
            
    raise RuntimeError(
        f"No morphology found with enough synapse capacity to satisfy "
        f"the requirements: {synapses_per_layer} at Z={target_z_pos:.1f}"
    )

