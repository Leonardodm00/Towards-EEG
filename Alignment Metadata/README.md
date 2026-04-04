## 📐 Morphology Alignment Pipeline

When working with reconstructed 3D morphologies, neurons are often oriented randomly in space. To build an organized cortical column, we must mathematically align these neurons so their primary dendritic arbors point "upward" (along the Z-axis). 

This pipeline achieves this through a three-step process: calculating how directional the neuron is (Anisotropy), computing the exact rotation matrix needed to stand it upright, and orchestrating the data into a clean metadata file.

---

### 1. The Filter: `calculate_dendrite_xy_anisotropy`
Not all neurons have a clear "up" and "down" direction; some are highly branched and isotropic (spherical). This function acts as a quality-control filter by quantifying how elongated a neuron's dendritic tree is in the 2D plane.

**How it works:**
1. **Dendritic Isolation:** It strips away the axon, soma, and basal structures, isolating only the dendritic and apical nodes.
2. **Principal Component Analysis (PCA):** It centers the (X, Y) coordinates of the dendrites and computes the covariance matrix. By extracting the eigenvalues ($\lambda_1, \lambda_2$), it identifies the primary and secondary axes of the dendritic spread.
3. **Fractional Anisotropy (FA) Calculation:** It computes a 2D FA score between `0.0` and `1.0`.
   * An FA close to `0.0` means the dendrites are a perfect circle (isotropic).
   * An FA close to `1.0` means the dendrites form a tight, directional bundle (highly anisotropic).
4. **Visualization:** If requested, it renders a 2D plot showing the neuron, the isolated dendrites, and the PCA eigenvectors overlaid as directional arrows.

---

### 2. The Engine: `calculate_z_alignment_math`
Once a neuron is confirmed to be directional, this function calculates the exact 3D rotation matrix required to align its primary dendritic trunk parallel to the global Z-axis.

**How it works:**
1. **Distal Tail Isolation:** It calculates the distance of every dendritic node from the soma and isolates only the most distal tips (e.g., the top 5% or 10% furthest points, defined by the `percentile` threshold).
2. **Vector Calculation:** It calculates the Center of Mass (CoM) of these distal tips. It then creates a unit vector (`v_com`) pointing directly from the soma to this distal center. This vector represents the neuron's "natural" upward direction.
3. **Rotation Matrix Computation:** Using cross products and dot products, it calculates the angle and axis of rotation required to swing `v_com` so that it perfectly aligns with the target `[0, 0, 1]` Z-axis. It converts this into a 3x3 rotation matrix using `scipy.spatial.transform`.
4. **RAM Efficiency:** Because it processes massive NetworkX graphs, it aggressively clears variables and forces garbage collection (`gc.collect()`) after every iteration to prevent memory leaks.

---

### 3. The Orchestrator: `extract_alignment_metadata`
This function is the master pipeline manager. It wraps the previous two functions into a seamless workflow and outputs a clean, ready-to-save database.

**How it works:**
1. **Execution & Filtering:** It runs `calculate_dendrite_xy_anisotropy` on the provided list of neuron IDs. It immediately drops any neurons whose FA score falls below your specified `fa_threshold` (e.g., `0.75`), ensuring only clean, directional neurons are passed to the next step.
2. **Math Handoff:** It passes the surviving neurons to `calculate_z_alignment_math` to extract their rotation matrices and original soma positions.
3. **Data Packaging:** It merges the FA scores, soma coordinates, CoM vectors, and the 3x3 rotation matrices into a single Pandas DataFrame. This DataFrame is returned and can be saved as a CSV to be loaded later by the LFPy mapping script and the synaptic placement pipeline.
