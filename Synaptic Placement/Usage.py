# ==========================================
# Execution Code
# ==========================================

pre_mtype =  "population_probability_L4_inh"
post_mtype = "population_probability_L5_exc"

output_prob = '/content/drive/MyDrive/Colab Notebooks/Spanning_trees'
num_synapses = 10
post_soma_pos = [-200,-100,-200]  # exc
pre_soma_pos = [-20,-20,-20]  # inh

voxel_size = 10 #[um]
grid_extent = 500.0 #[um]

# Assume this function returns the centroids and their corresponding normalized probabilities
centroids, normalized_probabilities = calculate_synaptic_overlap_locations(
    pre_mtype  = pre_mtype,
    post_mtype = post_mtype,
    pre_soma_pos= pre_soma_pos,
    post_soma_pos = post_soma_pos,
    density_maps_dir = output_prob,
    num_synapses =num_synapses,
    voxel_size=voxel_size,
    grid_extent=grid_extent
)

# Call the updated plotting function passing the normalized_probabilities array
debug_plot_synaptic_overlap(
    pre_mtype,
    post_mtype,
    output_prob,
    pre_soma_pos,
    post_soma_pos,
    centroids,
    normalized_probabilities, # Added this argument
    voxel_size=voxel_size
)




# ==========================================
# Run the probabilistic sampling (Function from previous prompt)
# ==========================================
chosen_synapse_locations = distribute_synapses_probabilistically(centroids, normalized_probabilities, N_SYN)
visualize_joint_spanning_histo(normalized_probabilities)
# ==========================================
# Plot 1: Standard PMF Field Shape
# ==========================================
# This plots the abstract mathematical interaction zone.
visualize_joint_spanning_pmf(centroids, normalized_probabilities, voxel_size)

# ==========================================
# Plot 2: Final Verification against Blueprints
# ==========================================
# This plots the chosen synapses in global space against the biological blueprint context.
visualize_synapse_distribution_verification(
    pre_mtype, post_mtype, output_prob,
    pre_soma_pos, post_soma_pos, chosen_synapse_locations, voxel_size=voxel_size
)






# Variables from your setup
post_neuron_id = 4852204352 # Example post-synaptic ID

syn_type = 'inh' # Change to 'inh' if the pre-synaptic neuron is inhibitory
synapses_dir = '/content/drive/MyDrive/Colab Notebooks/Synapse database'
output_directory = '/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons'
output_filename = 'alignment_metadata_L5.csv'
input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons'

pre_mtype =  "population_probability_L4_inh"
post_mtype = "population_probability_L5_exc"

output_prob = '/content/drive/MyDrive/Colab Notebooks/Spanning_trees'
metadata_filepath = os.path.join(output_directory, output_filename)
# Execute the snapping function
final_synapse_locations = snap_to_closest_actual_synapses(
    post_neuron_id=post_neuron_id,
    synapses_dir=synapses_dir,
    post_soma_pos = post_soma_pos,
    chosen_centroids=chosen_synapse_locations,
    synapse_type=syn_type
)


visualize_final_snapped_synapses_3d(
    post_neuron_id,
    final_synapse_locations,
    input_dir,
    metadata_filepath,
    post_soma_pos,
    pre_mtype=pre_mtype,          # NEW: Pre-synaptic identifier
    density_maps_dir=output_prob,   # NEW: Path to the .npy density maps
    pre_soma_pos=pre_soma_pos,       # NEW: Global position of the pre-soma
    voxel_size=voxel_size,         # NEW: Required to translate grid indices to um
    k_neighbors=3)









# Execute the full pipeline
matched_synapses = execute_synaptic_placement_pipeline(
    pre_mtype=pre_mtype,
    post_mtype=post_mtype,
    pre_soma_pos=pre_soma_pos,
    post_soma_pos=post_soma_pos,
    density_maps_dir=output_prob,
    num_synapses=num_synapses,
    post_neuron_id=post_neuron_id,
    synapses_dir=synapses_dir,
    synapse_type=syn_type,
    voxel_size=voxel_size,
    grid_extent=grid_extent
)

# You can then pass 'matched_synapses' directly to your 3D visualization function
if matched_synapses is not None:
    visualize_final_snapped_synapses_3d(
        post_neuron_id=post_neuron_id,
        matched_synapses_df=matched_synapses,
        input_dir=input_dir,
        metadata_filepath=metadata_filepath,
        post_soma_pos=post_soma_pos,
        pre_mtype=pre_mtype,
        density_maps_dir=output_prob,
        pre_soma_pos=pre_soma_pos,
        voxel_size=voxel_size,
        k_neighbors=3
    )








