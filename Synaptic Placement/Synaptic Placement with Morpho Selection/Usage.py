import os
import numpy as np
# ==========================================
# 2. COLAB EXECUTION & TESTING BLOCK
# ==========================================

# Define Candidate Morphologies
nids = [
    5160796796, 5175090962, 5175353953, 5187371550, 5189677248, 5190319976, 
    5217039516, 5217171026, 5232282961, 5234649530, 5247526396, 5261835396, 
    5291168786, 5291548513, 5305258792, 5318852286, 5319189326, 5320400202, 
    5335001245, 5335191586, 5365955112, 5421148001, 5421336745, 5436507845, 
    5449224616, 5451122797, 5464747312, 5478134645, 5478616461, 5480136235, 
    5509088748, 5511717264, 5524551172, 5553081571, 5553578082, 5567711747, 
    5581378299, 5581510029, 5582035326, 5598914100, 5610974886, 5628189095, 
    5638002068, 5640659446, 5655289104, 5655917761, 5669189357, 5672738006, 
    5686418450, 5712933306, 5727681025, 5727870456, 5730658896, 5784625240, 
    5814702337, 5829069825, 5845248438, 5872566163, 5889270340, 5917113823, 
    5918005808, 5918705966, 5960640328, 5977227354, 6019146684
]

output_directory = '/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons'
candidate_morph_paths = [os.path.join(output_directory, f"neuron_{nid}_aligned.hoc") for nid in nids]

# Define Layer Boundaries
layer_bounds = {
    'L1':  [-250.0, 0.0],
    'L23': [-1200.0, -250.0],
    'L4':  [-1580.0, -1200.0],
    'L5':  [-2175.0, -1580.0],
    'L6':  [-2770.0, -2175.0]
}

# --- MOCKING THE HYBRID SIMULATION ARRAYS ---
# We mock these variables to emulate what your 'PopulationSuper' class sees
cellindex = 0
pop_soma_pos = {0: {'x': -200, 'y': -100, 'z': -1800}} # Dictionary Format

# Mock pre-synaptic partners (e.g., cell 101 gives 5 synapses, cell 102 gives 5 synapses)
Cell_afferences = {
    0: np.array([[101, 5], [102, 5]]) 
}
pre_partners_matrix = Cell_afferences[cellindex]

# Mock the massive global arrays for the lookup
global_cell_mtypes = {
    101: "population_probability_L4_inh",
    102: "population_probability_L4_inh" 
}
global_cell_coords = {
    101: {'x': -20, 'y': -20, 'z': -1400},
    102: {'x': -40, 'y': 10, 'z': -1450}
}

# General paths
output_prob = '/content/drive/MyDrive/Colab Notebooks/Spanning_trees'
synapses_dir = '/content/drive/MyDrive/Colab Notebooks/Synapse database'
post_mtype = "population_probability_L5_exc"
voxel_size = 10.0
grid_extent = 1000.0
syn_type = 'inh'

# ==========================================
# 3. RUN THE PIPELINE
# ==========================================
best_nid, abstract_synapse_locations = evaluate_and_select_morphology(
    post_cell_index=cellindex,
    post_mtype=post_mtype,
    post_soma_pos=pop_soma_pos[cellindex],
    pre_partners_matrix=pre_partners_matrix,
    cell_mtypes=global_cell_mtypes,
    cell_coords=global_cell_coords,
    layer_bounds=layer_bounds,
    density_maps_dir=output_prob,
    morph_paths=candidate_morph_paths,
    synapses_dir=synapses_dir,
    voxel_size=voxel_size,
    grid_extent=grid_extent
)

if best_nid is not None and len(abstract_synapse_locations) > 0:
    # Snap to the winning morphology
    final_synapse_locations = snap_to_closest_actual_synapses(
        post_neuron_id=best_nid,
        synapses_dir=synapses_dir,
        post_soma_pos=pop_soma_pos[cellindex],
        chosen_centroids=abstract_synapse_locations,
        synapse_type=syn_type
    )

    # Visualize the results
    visualize_final_snapped_synapses_3d(
        post_neuron_id=best_nid,
        matched_synapses_df=final_synapse_locations,
        input_dir=output_directory,
        metadata_filepath=os.path.join(output_directory, 'alignment_metadata_L5.csv'),
        post_soma_pos=pop_soma_pos[cellindex]
    )
