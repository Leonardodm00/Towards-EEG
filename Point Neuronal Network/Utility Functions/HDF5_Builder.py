import h5py
import numpy as np

def dictionary_to_fast_h5(spike_dict, total_raw_neurons, h5_path):
    """
    Converts a dictionary of {raw_idx: [spike_times]} into an O(1) HDF5 database.
    
    spike_dict: Dictionary with raw_idx as keys and arrays/lists of spike times as values.
    total_raw_neurons: The absolute maximum number of neurons in your circuit 
                       (e.g., if raw_idx goes from 0 to 314,000, this is 314,001).
    h5_path: Where to save the file.
    """
    print("Compiling dictionary into flat arrays...")
    
    # Pre-allocate the index pointer array
    # Size is N + 1 to easily calculate lengths via indptr[i+1] - indptr[i]
    indptr = np.zeros(total_raw_neurons + 1, dtype=np.int64)
    
    all_times = []
    current_pointer = 0
    
    # We must iterate sequentially through every possible raw_idx
    for raw_idx in range(total_raw_neurons):
        indptr[raw_idx] = current_pointer
        
        if raw_idx in spike_dict:
            # Grab the spikes, ensure they are floats, and ensure they are sorted in time
            spikes = np.array(spike_dict[raw_idx], dtype=np.float32)
            spikes = np.sort(spikes) 
            
            all_times.append(spikes)
            current_pointer += len(spikes)
            
    # Set the very last pointer to the total number of spikes
    indptr[-1] = current_pointer
    
    # Concatenate all lists of spikes into one massive 1D array
    if all_times:
        times_array = np.concatenate(all_times)
    else:
        times_array = np.array([], dtype=np.float32)
        
    print(f"Saving optimized database to {h5_path}...")
    with h5py.File(h5_path, 'w') as f:
        # Save to disk with compression
        f.create_dataset('times', data=times_array, compression='gzip', compression_opts=4)
        f.create_dataset('indptr', data=indptr, compression='gzip', compression_opts=4)
        
    print(f"Done. Total spikes saved: {current_pointer}")




# Save the spikes
dictionary_to_fast_h5(
    spike_dict=my_simulation_spikes, 
    total_raw_neurons=300000, # Replace with your actual len(cell_mtypes)
    h5_path='microcircuit_spikes.h5'
)



