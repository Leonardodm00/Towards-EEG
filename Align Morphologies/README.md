map_and_save_synapses
Purpose: To take a messy cloud of raw synapse coordinates, figure out which part of the neuron's 3D skeleton they belong to, and export clean, simulation-ready data.

How it works:

Data Ingestion & Filtering: It loads a CSV of synapses for a specific neuron. It immediately filters out outgoing synapses (keeping only incoming ones) and drops any rows missing spatial coordinates.

Spatial Indexing (The KD-Tree): It takes the raw, unaligned coordinate points of the neuron's skeleton and builds a cKDTree.  A KD-Tree is a spatial data structure that organizes points in 3D space, allowing the computer to find the "nearest neighbor" of a point in milliseconds rather than brute-force checking every single point against every other point.

Scaling: It takes the raw synapse locations (which are in voxel coordinates) and multiplies them by their physical resolution (8 nm for X and Y, 33 nm for Z) to convert them into real-world nanometer distances.

The "Snapping" Query: It feeds the scaled synapse coordinates into the KD-Tree. For every single synapse, the tree returns the index of the closest physical node on the neuron's skeleton.

Data Translation & Classification: For every matched skeleton node, the function looks up that exact node's aligned coordinates (which have already been rotated and scaled to micrometers in the main loop). It also checks the raw text of the synapse type to classify it as excitatory (exc) or inhibitory (inh).

Export: It packages the segment ID, the clean micrometric coordinates, and the synapse type into a new CSV.







export_neuron_to_hoc
Purpose: To convert a tabular list of 3D points (a DataFrame) into a .hoc script, which is the specific programming language required by the NEURON simulation environment to build compartmental models.

How it works:

Standardizing Geometry: It ensures every point has a radius (r), converting it from nanometers to micrometers if necessary.

Labeling Compartments: It maps the raw anatomical labels (like "basal dendrite" or "apical tuft") into the strict, standard arrays that NEURON expects: soma, axon, or dend.

Tree Traversal (Depth-First Search): The code builds a dictionary of parent-child relationships (children). It finds the root node (where parent p is -1) and uses a "stack" to trace the branches of the neuron from the soma outward.

Section Building (Crucial Step): NEURON simulates electrical flow through continuous "sections." Your code traverses the tree and groups unbranched chains of nodes together into a single section.

Preventing Cable Fractures: When the code hits a branch point (a node with 2 or more children), it starts a new section for each child branch. Crucially, it copies the parent's branch-point node and inserts it as the first node of the new child section. This ensures the 3D geometries physically touch in the simulator, preventing the model from breaking into disconnected pieces.

Writing the HOC Syntax: Finally, it writes the text file. It creates the arrays (e.g., create dend[45]), connects them topologically (connect dend[1](0), soma[0](1)), and fills each section with its 3D coordinates and diameters using the pt3dadd command.










align_neurons_to_neighborhoodPurpose: This is the main orchestrator. It aligns a neuron to a biologically meaningful reference frame, standardizes its units, and triggers the HOC export and synapse mapping.How it works:Translation (Centering): It takes the raw neuron data and translates the entire skeleton so that the soma rests perfectly at the origin point (0, 0, 0).Finding the Local Neighborhood: It calculates the Euclidean distance between the target neuron's soma and all the reference somas provided in a metadata file. It selects the $k$-nearest neighbors.Rotation Averaging: Because brain tissue can be warped, a global rotation matrix isn't always accurate. Instead, this function takes the specific rotation matrices of those $k$-nearest neighbors and averages them together using Scipy's spatial transform module. This creates a localized rotation matrix specific to that neuron's micro-environment.Applying the Alignment: It mathematically rotates the centered neuron using this new average matrix (via a dot product).Unit Conversion: It converts the spatial units from nanometers to micrometers by dividing the coordinates by 1000. NEURON strictly requires micrometers for spatial coordinates.Orchestration: With the neuron now perfectly centered, locally rotated, and scaled, it overwrites the DataFrame and passes the clean data to export_neuron_to_hoc and map_and_save_synapses.Visualization: If requested, it renders a side-by-side interactive 3D plot showing the messy raw skeleton next to the cleanly aligned target, drawing vectors to show how it relates to its neighbors.































