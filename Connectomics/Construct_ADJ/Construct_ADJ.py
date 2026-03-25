import numpy as np
from scipy.spatial.distance import cdist

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
    # This avoids millions of slow Python iterations
    dist_matrix = cdist(cell_coords, cell_coords, metric='euclidean')
    
    # Initialize the probability matrix with zeros
    prob_matrix = np.zeros((N, N))
    
    # 2. Extract unique m-types to loop over pathways instead of individual cells
    unique_mtypes = np.unique(cell_mtypes)
    
    for pre_mtype in unique_mtypes:
        # Find all indices of pre-synaptic cells of this m-type
        pre_idx = np.where(cell_mtypes == pre_mtype)[0]
        
        for post_mtype in unique_mtypes:
            post_idx = np.where(cell_mtypes == post_mtype)[0]
            
            # Create a meshgrid to safely index the sub-matrix of these specific pre/post cells
            idx_grid = np.ix_(pre_idx, post_idx)
            
            # Check if this projection pathway exists in your data
            if post_mtype in conn_data['best_fit'].get(pre_mtype, {}):
                
                # Get the distances specific to this pre/post population pair
                d_sub = dist_matrix[idx_grid]
                
                fit_type = conn_data['best_fit'][pre_mtype][post_mtype]
                
                # --- Apply the specific mathematical fit ---
                if fit_type == 'exp':
                    a0 = conn_data['a0mat_exp'][pre_mtype][post_mtype]
                    l = conn_data['lmat_exp'][pre_mtype][post_mtype]
                    d0 = conn_data['d0_exp'][pre_mtype][post_mtype]
                    
                    # Exponential decay: P(d) = A0 * exp(-(d - d0)/lambda)
                    # Note: Usually probability doesn't exceed A0 for d < d0
                    p_sub = a0 * np.exp(-np.maximum(d_sub - d0, 0) / l)
                    
                elif fit_type == 'gauss':
                    a0 = conn_data['a0mat_gauss'][pre_mtype][post_mtype]
                    l = conn_data['lmat_gauss'][pre_mtype][post_mtype]
                    x0 = conn_data['x0_gauss'][pre_mtype][post_mtype]
                    
                    # Gaussian decay: P(d) = A0 * exp(-((d - x0)^2) / lambda^2)
                    p_sub = a0 * np.exp(-((d_sub - x0)**2) / (l**2))
                    
                else:
                    # Fallback to mean probability if fit is unknown or missing
                    mean_p = conn_data['pmat'][pre_mtype][post_mtype]
                    p_sub = np.full_like(d_sub, mean_p)
                
                # Assign the calculated probabilities back into the main probability matrix
                prob_matrix[idx_grid] = p_sub

    # 3. Prevent autapses (zero out the diagonal)
    np.fill_diagonal(prob_matrix, 0.0)
    
    # 4. Roll the dice to create the binary adjacency matrix
    # If the random number [0, 1) is less than the probability, a connection exists (1)
    random_matrix = np.random.rand(N, N)
    adj_matrix = (random_matrix < prob_matrix).astype(int)
    
    return adj_matrix
