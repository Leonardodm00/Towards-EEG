def build_adjacency_matrix(cell_mtypes, cell_coords, conn_data):
    """
    Builds a binary adjacency matrix for a 3D microcircuit.

    Parameters:
    - cell_mtypes: numpy array of string m-types for each cell (length N)
    - cell_coords: numpy array of shape (N, 3) containing x, y, z coordinates
    - conn_data: the dictionary loaded from your 'conn.pkl'

    Returns:
    - adj_matrix: (N, N) binary numpy array (1 = connection, 0 = no connection)
    """
    N = len(cell_mtypes)

    # 1. Calculate the full N x N 3D Euclidean distance matrix instantly
    dist_matrix = cdist(cell_coords, cell_coords, metric='euclidean')

    # Initialize the probability matrix with zeros
    prob_matrix = np.zeros((N, N))

    # 2. Extract unique m-types to loop over pathways instead of individual cells
    unique_mtypes = np.unique(cell_mtypes)

    for pre_mtype in unique_mtypes:
        pre_idx = np.where(cell_mtypes == pre_mtype)[0]

        for post_mtype in unique_mtypes:
            post_idx = np.where(cell_mtypes == post_mtype)[0]

            # Create a meshgrid to safely index the sub-matrix
            idx_grid = np.ix_(pre_idx, post_idx)

            # Check if this projection pathway exists in your data
            if post_mtype in conn_data.get('best_fit', {}).get(pre_mtype, {}):
                
                d_sub = dist_matrix[idx_grid]
                fit_type = conn_data['best_fit'][pre_mtype][post_mtype]

                # --- Apply the specific mathematical fit (WITH FLOAT CASTING) ---
                if fit_type == 'exp':
                    a0 = float(conn_data['a0mat_exp'][pre_mtype][post_mtype])
                    l = float(conn_data['lmat_exp'][pre_mtype][post_mtype])
                    d0 = float(conn_data['d0_exp'][pre_mtype][post_mtype])
                    
                    p_sub = a0 * np.exp(-np.maximum(d_sub - d0, 0) / l)
                    
                elif fit_type == 'gauss':
                    a0 = float(conn_data['a0mat_gauss'][pre_mtype][post_mtype])
                    l = float(conn_data['lmat_gauss'][pre_mtype][post_mtype])
                    x0 = float(conn_data['x0_gauss'][pre_mtype][post_mtype])
                    
                    p_sub = a0 * np.exp(-((d_sub - x0)**2) / (l**2))
                    
                else:
                    # --- Fallback: Step-function empirical probability bins ---
                    
                    # Initialize all to 0.0 (safely covers any distance > 375.0)
                    p_sub = np.zeros_like(d_sub)
                    
                    # Define the distance thresholds and their corresponding dictionary keys
                    dist_bins = [
                        (12.5, 'pmat12um'),   (25.0, 'pmat25um'),   (50.0, 'pmat50um'),
                        (75.0, 'pmat75um'),   (100.0, 'pmat100um'), (125.0, 'pmat125um'),
                        (150.0, 'pmat150um'), (175.0, 'pmat175um'), (200.0, 'pmat200um'),
                        (225.0, 'pmat225um'), (250.0, 'pmat250um'), (275.0, 'pmat275um'),
                        (300.0, 'pmat300um'), (325.0, 'pmat325um'), (350.0, 'pmat350um'),
                        (375.0, 'pmat375um')
                    ]
                    
                    prev_d = -1.0 # Start below 0 to safely capture absolute 0.0 distance
                    
                    for max_d, bin_key in dist_bins:
                        # Use try/except to safely fetch the bin probability if it's missing or null
                        try:
                            bin_prob = float(conn_data[bin_key][pre_mtype][post_mtype])
                        except (KeyError, ValueError, TypeError):
                            bin_prob = 0.0
                            
                        # Create a boolean mask for distances falling within this specific bin
                        mask = (d_sub > prev_d) & (d_sub <= max_d)
                        
                        # Apply the probability to those specific distances
                        p_sub[mask] = bin_prob
                        
                        # Move the lower boundary up for the next loop iteration
                        prev_d = max_d
                
                # Assign the calculated probabilities back into the main probability matrix
                prob_matrix[idx_grid] = p_sub

    # 3. Prevent autapses (zero out the diagonal)
    np.fill_diagonal(prob_matrix, 0.0)

    # 4. Roll the dice to create the binary adjacency matrix
    random_matrix = np.random.rand(N, N)
    adj_matrix = (random_matrix < prob_matrix).astype(int)

    return adj_matrix
