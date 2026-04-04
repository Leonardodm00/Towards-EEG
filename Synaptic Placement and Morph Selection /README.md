## 🧠 Morphology Selection & Synaptic Placement Pipeline

### The "Chicken-and-Egg" Problem in 3D Microcircuits
When building a biologically realistic 3D neural network, placing synapses presents a fundamental challenge: **You cannot snap synapses to a specific dendritic tree if you haven't chosen the tree yet, but you shouldn't choose a tree until you know exactly where the network demands those synapses to be placed.**

To solve this, our pipeline splits the problem into two distinct phases managed by two core functions: an **Architect** (`evaluate_and_select_morphology`) that maps out the mathematical demand in empty space, and an **Inspector** (`cell_MorphSelect`) that empirically tests candidate morphologies to see if they can physically support that demand.

---

### 1. The Architect: `evaluate_and_select_morphology`
This function acts as the bridge between the global network adjacency matrix and the single-cell 3D morphology. It calculates the exact layer-by-layer synaptic demand for a post-synaptic neuron before a specific `.hoc` morphology is even selected.

**Step-by-Step Execution:**
1. **Coordinate Standardization:** Ingests the global 3D coordinates of the post-synaptic cell and all its pre-synaptic partners, ensuring standard NumPy array formatting.
2. **Abstract Voxel Placement:** Iterates through every pre-synaptic partner. It loads the 3D density maps (spanning trees) for both cell types, calculates the overlapping probability field (Joint PMF), and probabilistically drops the required number of synapses into this abstract 3D space.
3. **Layer Profiling:** Extracts the absolute Z-coordinates (depths) of all probabilistically placed synapses and bins them against the theoretical cortical layer boundaries (e.g., L1, L2/3, L4).
4. **Handoff:** Generates a strict "Synaptic Demand Profile" (e.g., *150 synapses required in Layer 4, 300 in Layer 5*) and passes this profile to the morphological selector.

---

### 2. The Inspector: `cell_MorphSelect`
This function performs a strict, empirical intersection test. Instead of estimating a dendritic tree's viability based on total cable length, it directly tests if a candidate morphology has the physical capacity (pre-calculated docking sites) to fulfill the Architect's demand profile at a specific cortical depth.

**Step-by-Step Execution:**
1. **Candidate Shuffling:** Takes a list of candidate `.hoc` morphology files for the specific biological cell type and shuffles them to ensure natural variance across the generated network.
2. **Virtual Depth Shifting:** For a candidate cell, it loads its pre-mapped `_mapped_synapses.csv` database (containing all physically valid docking sites on that tree). It adds the global `target_z_pos` to these coordinates, virtually shifting the tree to its final depth in the simulated cortical column.
3. **Empirical Intersection Test:** It counts exactly how many physical docking sites fall within the Z-boundaries of the required layers. 
4. **Pass/Fail Evaluation:** It compares the physically available docking sites against the requested Synaptic Demand Profile. If a candidate tree lacks the capacity in *any* required layer, it is rejected, and the next candidate is evaluated.
5. **Final Selection:** Returns the Neuron ID (`nid`) of the first morphology that successfully passes all layer capacity requirements.

---

### 💡 Why this Architecture is Robust
By running these two functions in sequence, we mathematically guarantee computational stability during the final structural wiring phase. Because `evaluate_and_select_morphology` maps out the statistical blueprint first, and `cell_MorphSelect` explicitly verifies the structural capacity second, **we completely eliminate the risk of "snapping failures"** (where a network attempts to place a synapse in a layer where the chosen dendritic tree has no branches).
