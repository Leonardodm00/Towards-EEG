import numpy as np
from scipy.spatial.distance import cdist

import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse

def build_adjacency_matrix(cell_mtypes, cell_coords, conn_data):
    """
    Optimized builder for massive 3D microcircuits using sparse matrices.
    """
    N = len(cell_mtypes)
    unique_mtypes = np.unique(cell_mtypes)
    
    # We store only the (row, col) indices of successful connections
    # This avoids storing the millions of 'zeros' in a dense matrix
    all_rows = []
    all_cols = []

    # 1. Pre-group indices to avoid calling np.where millions of times
    mtype_indices = {m: np.where(cell_mtypes == m)[0] for m in unique_mtypes}

    for pre_mtype in unique_mtypes:
        pre_idx = mtype_indices[pre_mtype]
        coords_pre = cell_coords[pre_idx]

        for post_mtype in unique_mtypes:
            # Only proceed if this pathway exists in our connectivity data
            if post_mtype not in conn_data.get('best_fit', {}).get(pre_mtype, {}):
                continue
                
            post_idx = mtype_indices[post_mtype]
            coords_post = cell_coords[post_idx]

            # 2. Calculate distances ONLY for this sub-block
            # This is the key RAM saver
            d_sub = cdist(coords_pre, coords_post, metric='euclidean')
            
            fit_type = conn_data['best_fit'][pre_mtype][post_mtype]

            # 3. Calculate probabilities for this sub-block
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
                # Fallback: Step-function bins
                p_sub = np.zeros_like(d_sub)
                dist_bins = [
                    (12.5, 'pmat12um'), (25.0, 'pmat25um'), (50.0, 'pmat50um'),
                    (75.0, 'pmat75um'), (100.0, 'pmat100um'), (125.0, 'pmat125um'),
                    (150.0, 'pmat150um'), (175.0, 'pmat175um'), (200.0, 'pmat200um'),
                    (225.0, 'pmat225um'), (250.0, 'pmat250um'), (275.0, 'pmat275um'),
                    (300.0, 'pmat300um'), (325.0, 'pmat325um'), (350.0, 'pmat350um'),
                    (375.0, 'pmat375um')
                ]
                prev_d = -1.0
                for max_d, bin_key in dist_bins:
                    try:
                        bin_prob = float(conn_data[bin_key][pre_mtype][post_mtype])
                    except (KeyError, ValueError, TypeError):
                        bin_prob = 0.0
                    mask = (d_sub > prev_d) & (d_sub <= max_d)
                    p_sub[mask] = bin_prob
                    prev_d = max_d

            # 4. Roll the dice for this sub-block only
            # random_matrix is now just the size of the population pair
            trial = np.random.rand(*p_sub.shape) < p_sub
            
            # 5. Extract successful connection coordinates
            # local_rows/cols are indices within the sub-block
            local_rows, local_cols = np.where(trial)
            
            if len(local_rows) > 0:
                # Map local indices back to global cell indices
                global_rows = pre_idx[local_rows]
                global_cols = post_idx[local_cols]
                
                # Check for and remove autapses (self-connections)
                mask_autapse = global_rows != global_cols
                all_rows.extend(global_rows[mask_autapse])
                all_cols.extend(global_cols[mask_autapse])

            # Explicitly clear sub-matrices from memory
            del d_sub, p_sub, trial

    # 6. Final Assembly: Create a Sparse CSR matrix
    # This stores only the '1's. Total RAM used is proportional to number of connections.
    data = np.ones(len(all_rows), dtype=np.uint8)
    adj_matrix = sparse.csr_matrix((data, (all_rows, all_cols)), shape=(N, N))

    return adj_matrix
