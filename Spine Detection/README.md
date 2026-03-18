## Automated Dendritic Spine Detection & Morphological Segmentation

This repository contains a Python-based pipeline for the automated identification, extraction, and morphological segmentation of dendritic spines from 3D neuronal reconstructions.

Using graph theory and computational geometry, the algorithm isolates terminal subtrees along dendritic arbors, applies B-spline interpolation to smooth their geometric profiles, and uses signal processing to mathematically separate each spine into its Head and Neck sub-compartments.

### ✨ Key Features

Robust Topological Extraction: Uses Directed Acyclic Graphs (DAGs) to identify complete spine subtrees, effectively handling complex, bifurcating, or artifact-heavy spine reconstructions.

Continuous Spline Interpolation: Up-samples and smooths the 3D coordinates and radius profiles of spines using Scipy's B-splines.

Algorithmic Head/Neck Segmentation: Analyzes the 1D radius profile of the spine from tip to base, using Gaussian smoothing and peak-finding to locate the structural constriction (the "neck").

Interactive 3D Visualization: Renders high-fidelity, rotatable 3D models of the segmented neurons using Plotly, with vivid color-coding for different morphological compartments.

### 🧠 How It Works (The Algorithm)

The pipeline processes neuronal reconstructions (CSV files containing node id, parent_id, x, y, z, radius, and annotated_type) through four main steps:

1. **Graph Construction & Subtree Isolation**
The neuron is loaded into memory as a networkx.DiGraph. The algorithm identifies all branch points specifically located on dendrite or apical segments. For each branching child, it extracts the entire downstream topological subtree. If the total cumulative physical length (Euclidean distance of all segments) of the subtree is below the user-defined spine_length_threshold_nm, it is classified as a spine.

2. **Geometric Smoothing**
Because manual or automated traces can be jagged, the algorithm extracts the path from the spine's base to its tips. It then applies a cubic B-spline interpolation (scipy.interpolate.splprep) to smooth the 3D backbone coordinates and the corresponding radius values.

3. **Morphological Profiling**
The algorithm evaluates the interpolated radius as a 1D signal, starting from the absolute tip of the spine moving toward the dendritic shaft:

It applies a gaussian_filter1d to smooth out micro-reconstruction noise.

It uses scipy.signal.find_peaks on the inverted signal to locate the first significant local minimum (the structural constriction point).

The Head: Defined as all nodes from the tip down to this local minimum.

The Neck: Defined as all nodes from the local minimum down to the dendritic shaft.

Fallback: If no clear minimum exists (e.g., "stubby" spines), the algorithm places an estimated boundary at 1/3 of the total spine length from the tip.

4. **Discrete Mapping & Plotting**
The calculated boundary distance is mapped back to the original, discrete nodes in the DataFrame. Nodes are re-annotated as either head or neck, and the updated DataFrame is plotted in 3D.

### 🛠️ Requirements

numpy

pandas

networkx

scipy

plotly

You can install the dependencies via pip:

Bash
pip install numpy pandas networkx scipy plotly


# 🚀 Usage
 
Ensure your neuron reconstructions are saved as CSV files in your designated input directory.



### Define the neuron IDs you want to process
target_ids = [5629504348]

### 1. Run the segmentation algorithm
 *Adjust spine_length_threshold_nm depending on your dataset's scale*
spine_df_dict = label_dendritic_spines_robust(
    neuron_ids=target_ids, 
    input_dir='/path/to/your/neurons', 
    spine_length_threshold_nm=5000.0,
    num_interp_points=100,
    smoothing_sigma=2.0
)

### 2. Visualize the results
plot_color_coded_neurons(spine_df_dict, target_ids)


 
🎨 Visualization Palette
The plot_color_coded_neurons function generates an interactive WebGL plot. The morphological compartments are color-coded as follows:

🟡 Gold: Spine Head (Thick line)

🟣 Indigo: Spine Neck (Thick line)

🔴 Crimson: Dendritic Shafts

🔵 Royal Blue: Axon

🟣 Medium Orchid: Apical Dendrites

⚫ Black: Soma
