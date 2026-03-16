This code is a complete pipeline for spatially aligning 3D reconstructed neurons based on their local anatomical neighborhood and exporting them into a format ready for biophysical simulation.

Here is a breakdown of exactly what the two functions do:

### 1. `export_neuron_to_hoc`

This function translates a pandas DataFrame containing a neuron's 3D coordinates and connections into a NEURON-compliant `.hoc` script.

- **Unit conversion:** Converts coordinates from nm to um before building the output.

- **Spine recognition and labelling**: done through the *apply_spine_labels_to_df()* function. For more details refer to Spine Detection folder.

- **Label Mapping:** It standardizes your anatomical labels, categorizing every point as part of an `axon`, `soma`, or `dend` (dendrite).
    
- **Iterative Topology Building:** It reconstructs the neuron's branching structure (graph topology) using a stack-based `while` loop. It walks through the points and groups contiguous segments of the same type into unbranched cables (sections). By doing this iteratively rather than recursively, it prevents Python from crashing (hitting a `RecursionError`) when processing extremely long, unbranched axons or dendrites.
    
- **HOC Generation:** It writes a text file that the NEURON simulation environment can read. It declares the section arrays (`create`), wires them together (`connect`), and populates them with their physical geometry (`pt3dadd`).
    

### 2. `align_neurons_to_neighborhood`

This is the main orchestrator function. It takes a list of raw neurons and aligns them in 3D space by mimicking the orientation of nearby reference cells.

- **Soma Centering:** It loads a raw neuron's `.csv` file, locates its soma (root node), and shifts the entire neuron's coordinates so the soma sits perfectly at the origin `(0, 0, 0)`.
    
- **Neighborhood Context:** It calculates the Euclidean distance between the target neuron's original soma position and the somas of reference neurons stored in a provided metadata file. It selects the `k` closest reference cells.
    
- **Rotational Averaging:** It takes the rotation matrices of those `k` nearest neighbors and calculates a proper mathematical average using `scipy`'s Rotation module.
    
- **Alignment & Export:** It multiplies the target neuron's coordinates by this averaged rotation matrix. This essentially rotates the target cell to match the general "up/down" axis of its immediate anatomical neighborhood. It then immediately passes this rotated data to `export_neuron_to_hoc` to save it.
    
- **Visual Validation:** If `show_plot=True`, it generates a side-by-side interactive 3D Plotly graph. The left panel displays the raw centered neuron surrounded by the somas and directional vectors of its neighbors. The right panel shows the final rotated neuron, proving that the math successfully aligned the cell along the Z-axis.
