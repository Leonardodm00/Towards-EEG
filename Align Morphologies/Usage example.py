output_directory = '/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons'
output_filename = 'alignment_metadata_L6.csv'
metadata_filepath = os.path.join(output_directory, output_filename)


neuron_ids = [
              
              ...

  
              ]
algn_neuron =align_neurons_to_neighborhood(neuron_ids, metadata_filepath, input_dir='/content/drive/MyDrive/Colab Notebooks/Reconstructed neurons', k_neighbors=3, show_plot=False)
