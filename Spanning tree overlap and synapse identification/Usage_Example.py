# ==========================================
# Execution Code
# ==========================================

pre_mtype =  "population_probability_L4_inh"
post_mtype = "population_probability_L5_exc"

output_prob = '/content/drive/MyDrive/Colab Notebooks/Spanning_trees'
num_synapses = 10
post_soma_pos = [-200,-100,-200]  # exc
pre_soma_pos = [-20,-20,-20]  # inh

# Assume this function returns the centroids and their corresponding normalized probabilities
centroids, normalized_probabilities = calculate_synaptic_overlap_locations(
    pre_mtype  = pre_mtype, 
    post_mtype = post_mtype, 
    pre_soma_pos= pre_soma_pos, 
    post_soma_pos = post_soma_pos, 
    density_maps_dir = output_prob, 
    num_synapses =num_synapses, 
    voxel_size=5.0, 
    grid_extent=1000.0
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
    voxel_size=5.0
)
