from google.colab import drive
drive.mount('/content/drive')
!pip install neuron
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from neuron import h
# Load standard NEURON run library
h.load_file('stdrun.hoc')

def _plot_both_morphologies(dend_segments, layer_boundaries, layer_names, valid_layers, target_upper_z, target_lower_z, morph_idx):
    """Helper function to visualize the selected morphology at both boundaries side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
    
    color_palette = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
    layer_colors = {layer_idx: color_palette[i % len(color_palette)] for i, layer_idx in enumerate(valid_layers)}
    
    placements = [
        (axes[0], target_upper_z, "Upper Boundary Placement"),
        (axes[1], target_lower_z, "Lower Boundary Placement")
    ]
    
    for ax, soma_z, title in placements:
        # 1. Plot layer boundaries
        for i, bounds in enumerate(layer_boundaries):
            upper_bound, lower_bound = bounds
            ax.axhline(upper_bound, color='black', linestyle='--', alpha=0.5)
            
            bg_color = layer_colors.get(i, 'lightgrey')
            alpha_val = 0.1 if i in valid_layers else 0.05
            ax.axhspan(lower_bound, upper_bound, facecolor=bg_color, alpha=alpha_val)
            
            # Only draw the text labels on the left plot to avoid clutter
            if ax == axes[0]:
                layer_label = layer_names[i] if i < len(layer_names) else f"Layer {i}"
                ax.text(-200, (upper_bound + lower_bound) / 2, layer_label, 
                        va='center', ha='right', fontsize=10, fontweight='bold')
        
        ax.axhline(layer_boundaries[-1][1], color='black', linestyle='--', alpha=0.5)

        # 2. Plot dendritic segments
        lines = []
        line_colors = []
        
        for mid_z, _, p1, p2 in dend_segments:
            shifted_z1 = p1[2] + soma_z
            shifted_z2 = p2[2] + soma_z
            mid_shifted_z = mid_z + soma_z
            
            segment = [(p1[0], shifted_z1), (p2[0], shifted_z2)]
            lines.append(segment)
            
            seg_color = 'darkgrey' 
            for layer_idx in valid_layers:
                upper_bound, lower_bound = layer_boundaries[layer_idx]
                if lower_bound <= mid_shifted_z <= upper_bound:
                    seg_color = layer_colors[layer_idx]
                    break
            line_colors.append(seg_color)
            
        lc = LineCollection(lines, colors=line_colors, linewidths=1.5, alpha=0.8)
        ax.add_collection(lc)
        
        # 3. Plot the soma
        ax.scatter([0], [soma_z], color='black', s=100, zorder=5, label='Soma')
        
        # 4. Final plot formatting
        ax.set_aspect('equal')
        ax.set_xlim(-250, 250) 
        ax.set_ylim(layer_boundaries[-1][1] - 50, layer_boundaries[0][0] + 50)
        ax.set_xlabel('X distance (um)')
        if ax == axes[0]:
            ax.set_ylabel('Depth / Z distance (um)')
        ax.set_title(title)
        ax.legend()
        
    fig.suptitle(f'Selected Morphology (Index: {morph_idx}) Validated For Both Placements', fontsize=16)
    plt.tight_layout()
    plt.show()

def cell_MorphSelect(
        morph_paths,
        layer_boundaries,
        layer_names,
        synapses_per_layer,
        length_threshold,
        target_layer_idx,
        plot_result=False
    ):
        """

        Selects a random morphology meeting a length threshold within synapse-containing
        layers using the NEURON simulator library. It evaluates the morphology by placing
        its soma at both the upper and lower boundaries of its target layer.

        Args:
            morph_paths (list of str):
                A list containing the file paths to the candidate .hoc morphology files.
                Example: ['/path/to/neuron_1.hoc', '/path/to/neuron_2.hoc']

            layer_boundaries (numpy.ndarray):
                A 2D array of shape (N, 2) where N is the number of cortical layers.
                Each row must contain exactly two float values representing the
                [upper_Z, lower_Z] boundaries in micrometers (um). Values are typically
                negative, moving deeper from the pia (0.0).
                Example: np.array([[0.0, -81.6], [-81.6, -587.1], ...])

            layer_names (list of str):
                A list of strings containing the display names for each layer. The length
                must exactly match the number of rows in `layer_boundaries`. Used purely
                for labeling the output plot.
                Example: ['L1', 'L2/3', 'L4', 'L5', 'L6']

            synapses_per_layer (list of int or float):
                A 1D list storing the total number of synapses per layer. The length
                must match the rows in `layer_boundaries`. The function checks if an
                entry is > 0 to determine if a layer is "valid" for dendritic length counting.
                Example: [0, 1500, 3200, 500, 0]

            length_threshold (float):
                The minimum cumulative length of the dendritic arbor (in um) that *must* fall within the synapse-containing layers for the morphology to be accepted.
                Example: 1500.0

            target_layer_idx (int):
                A single integer representing the 0-based index of the layer to which
                the chosen neuron belongs. Used to look up the upper and lower Z-boundaries
                in `layer_boundaries` to position the soma during the stress test.
                Example: 3 (Targets the 4th row in layer_boundaries)

            plot_result (bool, optional):
                If True, generates a side-by-side matplotlib figure showing the successful
                neuron placed at both the upper and lower boundaries. Defaults to False.

        Returns:
            int: The index of the successfully selected morphology from the original
                `morph_paths` list.

        Raises:
            RuntimeError: If the function evaluates every file in `morph_paths` and none
                        meet the length threshold for both extreme soma placements.






        Neuron Morphology Selection Script

        This script acts as a strict filter for neuronal morphologies. Its goal is to
        find a single random neuron from a dataset whose dendritic tree is large enough
        to reach the right synaptic connections, regardless of where its soma sits inside
        its home layer.

        How it works:
        1. It shuffles the provided list of candidate .hoc morphologies for random selection.
        2. It loads a candidate into the NEURON simulator, isolates the dendritic tree
        (basal and apical), extracts its 3D coordinates, and scales them from nm to um.
        3. It performs a two-part stress test by virtually placing the neuron's soma at
        the absolute top of its assigned home layer, and then at the absolute bottom.
        4. For both extreme positions, it calculates exactly how much of the dendritic
        tree falls inside the specific layers designated as having synapses.
        5. If the total dendritic length in those synaptic layers drops below the minimum
        threshold during *either* the top or bottom placement, the neuron is rejected.
        6. The first neuron to successfully meet the threshold for *both* placements is
        chosen. The script then generates a side-by-side plot of both placements and
        returns the winning neuron's index. If no neurons pass, it raises an error.



        Selects a random morphology meeting a length threshold within synapse-containing
        layers using the NEURON simulator library.
        """
        valid_layers = [i for i, syn in enumerate(synapses_per_layer) if syn > 0]
        target_upper_z = layer_boundaries[target_layer_idx][0]
        target_lower_z = layer_boundaries[target_layer_idx][1]
        
        shuffled_paths = list(enumerate(morph_paths))
        random.shuffle(shuffled_paths)
        
        for original_idx, path in shuffled_paths:
            print('evaluating')
            print(original_idx)
            # Clear previous morphology from NEURON memory
            h('forall delete_section()')
            
            # Load the new .hoc file
            success = h.load_file(str(path))
            if not success:
                print(f"Warning: Could not load {path}")
                continue
                
            dend_segments = []
            
            # Iterate through all sections loaded into NEURON
            for sec in h.allsec():
                if 'dend' in sec.name().lower() or 'apic' in sec.name().lower():
                    n3d = int(h.n3d(sec=sec))
                    if n3d > 0:
                        pts = np.array([[h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)] for i in range(n3d)])
                        
                        for i in range(n3d - 1):
                            p1 = pts[i] 
                            p2 = pts[i+1] 
                            
                            length = np.linalg.norm(p1 - p2)
                            mid_z = (p1[2] + p2[2]) / 2.0
                            
                            dend_segments.append((mid_z, length, p1, p2))
                            
            valid_for_both_placements = True
            
            # Test BOTH upper and lower boundaries
            for soma_z_placement in [target_upper_z, target_lower_z]:
                
                # TRACK LENGTH PER LAYER instead of just a global sum
                length_in_valid_layers = {layer_idx: 0.0 for layer_idx in valid_layers}
                
                for mid_z, length, _, _ in dend_segments:
                    shifted_z = mid_z + soma_z_placement
                    
                    for layer_idx in valid_layers:
                        upper_bound, lower_bound = layer_boundaries[layer_idx]
                        if lower_bound <= shifted_z <= upper_bound:
                            length_in_valid_layers[layer_idx] += length
                            break
                
                total_valid_length = sum(length_in_valid_layers.values())
                
                # CONDITION 1: Does the total length inside valid layers meet the threshold?
                if total_valid_length < length_threshold:
                    valid_for_both_placements = False
                    break 
                    
                # CONDITION 2: Does the arbor actually reach EVERY valid layer?
                # If any valid layer has 0.0 length, it failed to span the necessary layers.
                spans_all_layers = all(layer_length > 0.0 for layer_length in length_in_valid_layers.values())
                
                if not spans_all_layers:
                    valid_for_both_placements = False
                    break 
                    
            # If it passed BOTH loops (and both conditions), we return it
            if valid_for_both_placements:
                if plot_result:
                    _plot_both_morphologies(
                        dend_segments, layer_boundaries, layer_names, valid_layers, 
                        target_upper_z, target_lower_z, original_idx
                    )
                return original_idx
                
        raise RuntimeError(
            f"No morphology met the length threshold of {length_threshold}um "
            f"and successfully spanned all synaptic layers for both placements."
        )

