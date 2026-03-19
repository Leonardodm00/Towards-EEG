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
