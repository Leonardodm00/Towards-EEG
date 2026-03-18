# Generate_smooth_surface

### Automated 3D Dendritic Spine Interpolation & Visualization

This repository provides a computational pipeline to extract, smooth, and visualize dendritic spines from 3D neuronal reconstructions. It converts jagged, discrete node traces into biologically realistic, continuous 3D surface meshes.

#### 🧠 What the Algorithm Does

The pipeline takes tabular neuronal morphology data (nodes, parents, 3D coordinates, and radii) and programmatically reconstructs isolated spines as high-fidelity 3D models. It handles complex, multi-branched spines and smooths out artifacts from manual or automated tracing.

#### ⚙️ How It Works

The algorithm processes the morphological data through the following steps:

**Topological Extraction (DFS)**:
It treats the neuron as a directed graph, identifies the exact nodes where spines attach to the dendritic shaft (roots), and uses a Depth-First Search (DFS) to extract every continuous path from the root to the terminal tips.

**B-Spline Smoothing (scipy.interpolate)**:
To remove jagged "elbows" between discrete traced points, it fits a cubic B-spline curve through the backbone coordinates. This generates a high-resolution, continuous 3D centerline.

**Radius Interpolation**:
The thickness (radius) of the spine is interpolated along the new curve, allowing the simulated membrane to organically swell and constrict matching the original data.

**Rotation-Minimizing Frame (Parallel Transport)**:
To draw a 3D tube without the mesh twisting or pinching at sharp corners, the algorithm calculates orthogonal normal and binormal vectors along the spline to sweep a perfect circular profile along the path.

**Interactive Rendering (plotly)**:
The resulting coordinate matrices are plotted as 3D WebGL surfaces, applying calculated lighting physics and spherical caps to create a seamless, rotatable biological model.




# Label_spines

### Graph-Based Dendritic Spine Detection & Skeleton Visualization

This repository provides a computational pipeline for the automated identification and 3D visualization of dendritic spines from neuronal morphology reconstructions. It uses graph topology to overcome common tracing errors and properly identify complex, multi-branched spines.

### 🧠 What the Algorithm Does

The algorithm processes discrete neuronal coordinate data (CSV format) to automatically locate and label dendritic spines. It mathematically isolates protrusions from the main dendritic shaft, measures their total cable length, and assigns morphological labels. Finally, it renders an interactive 3D "stick model" (skeleton) of the neuron with distinct color coding.

### ⚙️ How It Works

The pipeline operates in two main phases:

#### 1. Robust Spine Identification (label_dendritic_spines_robust)
**Graph Construction**: Converts the 3D coordinate table into a Directed Acyclic Graph (DAG) using networkx, preserving parent-child connectivity.

**Dendritic Filtering:** Scans the graph to find all branching points specifically located on dendrite or apical compartments.

**Subtree Extraction:** For every branch protruding from a dendrite, it isolates the entire downstream topological subtree (capturing all bifurcations, sub-branches, and tips).

**Length Thresholding:** It calculates the exact Euclidean distance of every segment within the subtree. If the total cumulative length of the complex is below a set threshold (e.g., 5000 nm), the entire subtree is classified and labeled as a spine.

#### 2. Interactive 3D Plotting (plot_color_coded_neurons)
**Segment Grouping:** Parses the updated dataframe and groups spatial coordinates by their anatomical labels to minimize rendering overhead.

**WebGL Rendering:** Uses plotly to render the neuron in a 3D environment. Spines are drawn with increased line thickness to make them easily distinguishable from the main arbors.




# Segment_spine

### Dendritic Spine Morphological Segmentation & Debugging

This script provides a specialized diagnostic pipeline for the quantitative segmentation of dendritic spines into Head and Neck compartments. It uses signal processing on high-resolution radius profiles to mathematically define structural boundaries.

### 🧠 What the Algorithm Does

The algorithm performs an automated "virtual dissection" of each spine. By analyzing the change in radius from the spine's tip to its base, it identifies the structural constriction that defines the neck. It provides a visual and quantitative dashboard to verify the accuracy of the segmentation logic.

### ⚙️ How It Works

The diagnostic pipeline follows these technical steps:

**High-Resolution Interpolation**:
Extracts the spine's centerline and radius data, applying Cubic B-splines to generate a continuous, high-density 3D path and a corresponding smooth radius signal.

**Radius Signal Processing:**
The 1D radius profile is evaluated from the Tip to the Base. A Gaussian filter is applied to the signal to eliminate micro-noises from the reconstruction that could cause false boundary detections.

**Boundary Detection (Neck-Finding):**

**Primary Logic:** The algorithm searches for the first significant local minimum in the radius profile. This point is classified as the "Neck" boundary.

**Fallback Logic:** For spines without a distinct constriction (stubby/plain spines), the algorithm applies a heuristic fallback, placing the boundary at 1/3 of the total spine length from the tip.

### Integrated Debug Dashboard (plotly):
Generates a side-by-side comparison:

**3D Panel:** A smooth mesh reconstruction color-coded by compartment (Gold for Head, Indigo for Neck).

**2D Panel**: A plot of the radius vs. distance from the tip, highlighting the raw data, the filtered signal, and the exact point where the algorithm triggered the "Neck" cutoff.





