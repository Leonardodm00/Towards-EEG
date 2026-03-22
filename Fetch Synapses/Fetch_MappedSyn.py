def fetch_mapped_synapse_indices(syn_df, syn_type, z_min=-np.inf, z_max=np.inf):
    """
    Retrieves pre-calculated LFPy segment indices from a mapped synapse database 
    based on biological classification and spatial constraints.

    This function bypasses the need for stochastic or area-based synapse placement 
    by leveraging 'pre-baked' spatial snapping data. It allows for the deterministic 
    reconstruction of a connectome within specific cortical layer boundaries.

    Parameters:
    -----------
    syn_df : pandas.DataFrame
        The mapped synapse DataFrame containing at least 'synapse_type', 
        'z' (aligned depth in um), and 'lfpy_idx' (1D segment index).
    syn_type : str
        The biological class of synapses to fetch. Must be 'exc' (excitatory) 
        or 'inh' (inhibitory).
    z_min : float, optional
        The lower (more negative) vertical boundary in micrometers. 
        Defaults to -infinity.
    z_max : float, optional
        The upper (closer to 0) vertical boundary in micrometers. 
        Defaults to +infinity.

    Returns:
    --------
    numpy.ndarray
        An array of integers representing the 1D LFPy segment indices 
        where synapses should be instantiated.
    """
    if syn_type not in ['exc', 'inh']:
        raise ValueError(f"Invalid syn_type '{syn_type}'. Must be 'exc' or 'inh'.")

    # 1. Classification Filtering
    # We isolate only the synapses that match the requested biological type.
    type_mask = syn_df['synapse_type'] == syn_type
    
    # 2. Spatial Filtering (Cortical Layer Slicing)
    # We define a vertical window. Only synapses whose aligned 'z' coordinate 
    # falls within [z_min, z_max] are selected.
    depth_mask = (syn_df['z'] >= z_min) & (syn_df['z'] <= z_max)
    
    # 3. Vectorized Index Extraction
    # We apply both masks and extract the 'lfpy_idx' column as an integer array.
    filtered_df = syn_df[type_mask & depth_mask]
    syn_indices = filtered_df['lfpy_idx'].values.astype(int)
    
    return syn_indices
