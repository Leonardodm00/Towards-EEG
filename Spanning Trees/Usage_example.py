#--- Example Execution ---
output_directory = '/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons'
output_filename = 'alignment_metadata_L5.csv'
output_prob = '/content/drive/MyDrive/Colab Notebooks/Spanning_trees'

metadata_filepath = os.path.join(output_directory, output_filename)


neuron_ids = [



         1885968871, 2771884890, 2903920410, 3006113857, 3222499167, 3223274014, 3265995691, 3295621595, 3396165104, 3399040143, 3469723286, 3484339026, 3485434276, 3543911915, 3559797036, 3643825567, 3644964664, 3688373256, 3803721169, 3892014018, 3892992905, 3908191917, 3919434378, 4038943832, 4051822360, 4065911312, 4066116171, 4066933831, 4127879232, 4139939668, 4184267343, 4242364929, 4271363320, 4315706800, 4329299008, 4330073983, 4361071433, 4373379460, 4386813257, 4387645213, 4401836842, 4402377014, 4415590654, 4418540025, 4431330530, 4443609959, 4446194392, 4446355067, 4472637539, 4474812411, 4489020400, 4489064511, 4489121710, 4490129250, 4532048217, 4534224460, 4535261682, 4561746646, 4592350377, 4606732344, 4633875577, 4634679023, 4664115273, 4693477092, 4706252842, 4709888554, 4722635073


              ]

voxel_size = 10 #[um]
grid_extent = 1000.0 #[um]

axon_field, dend_field = calculate_spanning_probabilities_jitter(
    neuron_ids=neuron_ids,
    metadata_filepath=metadata_filepath,
    input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons',
    voxel_size=voxel_size,      # 10x10x10 um voxels
    grid_extent=grid_extent    # 2000 um bounding box across X, Y, Z
)




# Save
save_probability_fields(axon_field, dend_field, output_prob, base_name="population_probability_L5_exc", save_tiff=True)


# --- Usage Examples ---
# For quick, memory-safe paper projections:
#plot_spanning_field(axon_field,voxel_size=voxel_size, structure_type='axon',grid_extent=grid_extent, plot_mode='2d', cmap='Reds', projection_type='max',overlay_neuron_ids=neuron_ids, metadata_filepath=metadata_filepath, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons')

# For deep-dive interactive 3D exploration:
plot_spanning_field(dend_field,voxel_size=voxel_size, structure_type='dend',grid_extent=grid_extent, plot_mode='3d', cmap='Blues', min_density_threshold=1e-6,overlay_neuron_ids=neuron_ids, metadata_filepath=metadata_filepath, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons')
