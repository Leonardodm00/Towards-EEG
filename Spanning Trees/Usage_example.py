#--- Example Execution ---

output_directory = '/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons'
output_filename = 'alignment_metadata_L5.csv'

metadata_filepath = os.path.join(output_directory, output_filename)


neuron_ids = [



          1885968871, 2771884890, 


              ]

voxel_size = 5 #[um]
grid_extent = 200.0 #[um]

axon_field, dend_field = calculate_spanning_probabilities_jitter(
    neuron_ids=neuron_ids,
    metadata_filepath=metadata_filepath,
    input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons',
    voxel_size=voxel_size,      # 10x10x10 um voxels
    grid_extent=grid_extent    # 2000 um bounding box across X, Y, Z
)


# --- Usage Examples ---
# For quick, memory-safe paper projections:
#plot_spanning_field(axon_field,voxel_size=voxel_size, structure_type='axon',grid_extent=grid_extent, plot_mode='2d', cmap='Reds', projection_type='max',overlay_neuron_ids=neuron_ids, metadata_filepath=metadata_filepath, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons')

# For deep-dive interactive 3D exploration:
plot_spanning_field(dend_field,voxel_size=voxel_size, structure_type='dend',grid_extent=grid_extent, plot_mode='3d', cmap='Blues', min_density_threshold=1e-6,overlay_neuron_ids=neuron_ids, metadata_filepath=metadata_filepath, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons')
