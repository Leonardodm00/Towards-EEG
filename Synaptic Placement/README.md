# Spanning Trees: Deep Dive into the Pipeline Engines

In computational neuroscience, a major challenge is translating **statistical, population-level rules** (e.g., "Layer 4 connects to Layer 5 with X probability") into **exact physical wiring** (e.g., "Axon A connects to Dendrite segment #452 on Neuron B"). 

The following three functions act as the "Supply, Demand, and Translation" engine of your spatial simulation, solving the problem of fitting a mathematical connectome onto a physical 3D neuron.

---

## 1. `evaluate_and_select_morphology` (The Demand Generator)

**The Core Concept:**
Before you can pick a 3D morphology for your post-synaptic cell, you need to know exactly what that cell is expected to do. If the network simulation dictates that this cell must receive 50 synapses from Layer 4 and 100 synapses from Layer 5, you need to mathematically generate that blueprint first. This function aggregates all that incoming traffic.

**Step-by-Step Logic:**
1. **Iterate Through the Network:** The function takes a `pre_partners_matrix`. This is a list of *every single cell* that is trying to connect to your target post-synaptic cell, along with how many synapses they owe it. 
2. **Generate the Abstract Cloud:** For every partner in that matrix, the function calculates the 3D mathematical overlap between the two cells and probabilistically drops the required number of synapses into global 3D space. It stores all of these floating coordinates in a massive dictionary (`all_synapse_locations`).
3. **Flatten and Profile by Depth (Z-Axis):** Once it has placed every single required synapse in abstract 3D space, it extracts just the Z-coordinates of all those points. 
4. **Bin into Cortical Layers:** It compares those Z-coordinates against your predefined `layer_bounds` (e.g., Layer 4 is between -1580µm and -1200µm). It counts how many abstract synapses landed in each layer, generating a strict requirement profile (e.g., "I need a cell that can support exactly 42 synapses in L4 and 88 in L5").
5. **Call the Gatekeeper:** It passes this exact layer requirement profile down to `cell_MorphSelect` to go find a physical cell that fits the bill.

**Why it's brilliantly designed:** It prevents you from guessing. By generating the entire abstract synaptic cloud *first*, the script knows exactly what the biological demands are before it ever looks at a physical `.hoc` or `.csv` file.

---

## 2. `cell_MorphSelect` (The Biological Gatekeeper)

**The Core Concept:**
This function receives the "demand profile" generated above and acts as the supply checker. It combs through a folder of candidate cell morphologies and physically measures them to see if they can survive the biological requirements of the simulation.

**Step-by-Step Logic:**
1. **Randomize the Candidates:** It shuffles the list of candidate morphology paths. This ensures that if you are building a network of 1,000 Layer 5 cells, they don't all get assigned the exact same morphological clone (assuming multiple candidates fit the criteria).
2. **Load and Clean the Database:** It opens the `mapped_synapses.csv` for a candidate. This CSV contains every valid point on that specific cell's dendritic tree where a synapse *could* theoretically attach. It includes bulletproof data-cleaning steps (stripping whitespace, renaming columns) to prevent pandas from crashing on poorly formatted files.
3. **The Critical Spatial Shift:** The coordinates in the `.csv` are usually in *local space* (meaning the soma is at 0,0,0). The function applies `shifted_z = syn_df['z'] + target_z_pos`. This projects the cell into the global cortical volume so its branches can be compared accurately against the global layer boundaries.
4. **The Layer Capacity Test:** The function loops through the required profile. If the profile says "I need 42 synapses in Layer 4," the script masks the shifted Z-coordinates to count exactly how many physical dendritic segments this specific cell has inside the Layer 4 boundaries. 
   * If `available_synapses >= required_synapses`, it moves to check the next layer.
   * If `available_synapses < required_synapses`, the cell fails immediately. The function rejects it and loads the next candidate.
5. **Return the Winner:** The moment it finds a morphology that passes the test for *every single layer*, it halts the search and returns that morphology's ID.

**Why it's brilliantly designed:** It prevents "biological clipping." If a simulation assigns a connection from a high Layer 2/3 cell, but you randomly selected a post-synaptic morphology whose apical tuft was stunted and didn't reach Layer 2/3, the snapping function would later fail (or snap the synapse wildly out of place). `cell_MorphSelect` guarantees that the physical tree matches the mathematical blueprint.

---

## 3. `map_abstract_synapses_to_segments` (The Batch Finalizer & Translator)

**The Core Concept:**
Neural simulation environments like **NEURON** or **LFPy** do not natively simulate cells in 3D Cartesian space `(x, y, z)`. Instead, they simulate cells as a 1D topological tree made of "segments" or "compartments" (e.g., "inject current into branch #45, segment #2"). 

By the time the script reaches this function, you have a massive list of floating 3D coordinates. This function’s sole purpose is to map those physical 3D coordinates to their exact 1D topological segment IDs (`lfpy_idx`) so the electrical simulation can actually run.

**Step-by-Step Logic:**
1. **Iterate by Pre-Synaptic Source:** The function receives `abstract_synapses_dict`, which groups all the generated 3D floating coordinates by the ID of the pre-synaptic cell that created them. It loops through this dictionary one pre-synaptic ID (`pre_idx`) at a time.
2. **Call the Snapping Function (Batch Processing):** For a specific pre-synaptic cell, it takes its bundle of 3D coordinates and feeds them into `snap_to_closest_actual_synapses`. This nested function does the heavy geometric lifting: it translates the floating global coordinates back to the local space of the chosen post-synaptic cell, builds a spatial KDTree, and snaps the floating points to the nearest physical dendritic branch.
3. **Extract the Simulation Index (`lfpy_idx`):** This is the most critical step of the function. The snapping function returns a Pandas DataFrame (`snapped_df`) containing all the biological info about the matched synapses. `map_abstract_synapses_to_segments` deliberately strips away the `(x, y, z)` coordinates, the errors, and the metadata. It isolates exactly one column: `lfpy_idx`.
   * *Code snippet:* `lfpy_indices = snapped_df['lfpy_idx'].astype(int).tolist()`
   * It forces them into integers just in case pandas interpreted them as floats (e.g., `45.0` to `45`), ensuring strict compatibility with NEURON.
4. **Error Handling & Fallbacks:** If the snapping function fails for a specific connection (e.g., the coordinates were too far away, or the DataFrame returned empty), the script doesn't crash. It catches the error, prints a warning (`"Snapping failed or returned empty..."`), and assigns an empty list `[]` to that pre-synaptic cell, allowing the rest of the network to continue building.
5. **Return the Final Dictionary:** It returns a highly condensed dictionary. Instead of complex 3D coordinate matrices, it simply outputs a mapping like `{pre_cell_101: [45, 46, 102], pre_cell_102: [8, 12]}`. 

**Why it’s brilliantly designed:** It establishes a strict boundary between **Morphological math** and **Electrical simulation**. The 3D spatial math is highly memory-intensive and complex. By isolating this translation step at the very end, you allow your final simulation scripts to be incredibly lightweight. Your NEURON/LFPy code won't need to load KDTrees, rotation matrices, or 3D voxel grids; it just needs to read the simple dictionary of integer segment IDs that this function generates.
