
# Extract synapses

## Overview
The H01 dataset contains millions of synapses spread across hundreds of JSONL shards in Google Cloud Storage. This function performs a "targeted scan"—instead of downloading the entire dataset, it streams each shard, extracts only the relevant synapses for your target neurons, and writes them to disk in real-time.

## Key Features
- Parallel Cloud Streaming: Uses gcsfs and ThreadPoolExecutor to stream and process multiple JSONL shards simultaneously from GCS without local caching.

- Producer-Consumer Architecture: Implements a dedicated Disk Writer Thread using a thread-safe Queue. This decouples the heavy CPU task (JSON parsing) from the I/O task (Disk writing), preventing file-handle bottlenecks.

- Memory Efficiency: Processes shards line-by-line using TextIOWrapper, allowing it to handle files much larger than the available system RAM.

- Bidirectional Extraction: Automatically identifies if a synapse is incoming (target is post-synaptic) or outgoing (target is pre-synaptic).

## How It Works

### 1. GCS Connectivity
The function connects to the public H01 bucket (h01-release) using anonymous credentials. It identifies all .json shards in the exported synapses directory.

### 2. The Writer Thread (Consumer)
To avoid the overhead of opening/closing files repeatedly, a dedicated thread manages all CSV operations.

- It listens to a write_queue.

- As matches are found, it dynamically creates a CSV for each neuron ID.

- It keeps file handles open until the entire scan is complete to maximize write speed.


### 3. Parallel Shard Processing (Producers)

The ThreadPoolExecutor spins up multiple workers to process shards in parallel:

- Parsing: Lines are read as JSON objects.

- Filtering: It checks both pre_synaptic_site and post_synaptic_partner for matches against the target_neuron_ids.Coordinate 

- Extraction: It pulls the raw H01 $X, Y, Z$ coordinates (typically in voxels) and the synapse type.


# Map synapses to segments
This function implemented in the **align morpholgies** folder


This function, map_synapses_to_segments, is a spatial processing tool designed to "attach" synaptic data to a 3D neuron skeleton. In computational neuroanatomy, synapses are often stored as a list of independent points, while the neuron is stored as a connected tree (skeleton). This function bridges that gap.

Here is a breakdown of what happens at each stage of the code:

## 1. Data Loading and Validation
The function first constructs file paths based on the neuron_id.

- It loads the Neuron Skeleton (a CSV containing nodes with x,y,z coordinates and connectivity).

- It loads the Synapse List (a CSV containing the locations and types of synapses).

- Filtering: it specifically keeps only incoming synapses (dendritic inputs) and removes any records with missing coordinate data.

## 2. Spatial Indexing with KD-Tree
Since a neuron can have thousands of nodes and thousands of synapses, checking every synapse against every node would be incredibly slow.

- The KD-Tree: The code uses cKDTree from scipy.spatial. This creates a high-speed spatial index of all the neuron's skeleton nodes.

- Instead of a "brute-force" search, the tree allows the computer to find the nearest neighbor in logarithmic time.

## 3. Voxel-to-Nanometer Scaling
This is a critical step for data alignment. In the H01 dataset, skeletons are often stored in physical nanometers, but raw synapse coordinates are often stored in voxels (the "pixels" of the 3D EM volume).
The code applies a specific conversion factor:

- X and Y: Multiplied by 8.0

- Z: Multiplied by 33.0

This ensures the synapse points "land" on the correct branches of the skeleton. Without this scaling, the synapses would appear clustered near the origin, far away from the neuron.

## 4. The Mapping Logic
The core of the function happens here:

1) Querying: For every synapse coordinate, the KD-Tree finds the index of the closest skeleton node.

2) Identifying Type: It looks at the synapse_type string:

- Type 2: Labeled as exc_syn (Excitatory).

- Type 1: Labeled as inh_syn (Inhibitory).

3) Labeling: It adds a new column, synapse_label, to the neuron's dataframe. The skeleton node closest to a synapse "inherits" that synapse's identity.

## Summary of Output
The function returns the original neuron dataframe, but with a crucial upgrade: every node in the skeleton that acts as a synaptic contact point is now explicitly labeled.

## Why is this useful?

Simulation: You can now tell a simulator like NEURON exactly where to place excitatory or inhibitory conductance models.

Analysis: You can calculate synapse density across different dendritic branches (e.g., comparing apical vs. basal dendrites).

Visualization: It allows you to generate plots where the skeleton "lights up" at every synaptic input.
