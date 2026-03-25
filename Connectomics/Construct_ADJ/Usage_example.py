import numpy as np
# Assuming the build_adjacency_matrix function is imported or defined here

# 1. Define a small dummy population of 5 cells
cell_mtypes = np.array(['L23_PC', 'L23_PC', 'L4_SS', 'L4_SS', 'L5_TTPC1'])

# Assign random 3D coordinates (x, y, z) in micrometers
cell_coords = np.array([
    [0.0, 0.0, 100.0],
    [10.0, 5.0, 105.0],
    [5.0, -10.0, 300.0],
    [-5.0, 15.0, 310.0],
    [0.0, 0.0, 600.0]
])

# 2. Mock up a simplified snippet of the 'conn.pkl' data structure
conn_data = {
    'best_fit': {
        'L23_PC': {'L4_SS': 'exp', 'L23_PC': 'gauss'},
        'L4_SS': {'L5_TTPC1': 'exp'}
    },
    # Exponential fit parameters
    'a0mat_exp': {'L23_PC': {'L4_SS': 0.15}, 'L4_SS': {'L5_TTPC1': 0.10}},
    'lmat_exp': {'L23_PC': {'L4_SS': 100.0}, 'L4_SS': {'L5_TTPC1': 150.0}},
    'd0_exp': {'L23_PC': {'L4_SS': 20.0}, 'L4_SS': {'L5_TTPC1': 25.0}},
    
    # Gaussian fit parameters
    'a0mat_gauss': {'L23_PC': {'L23_PC': 0.25}},
    'lmat_gauss': {'L23_PC': {'L23_PC': 75.0}},
    'x0_gauss': {'L23_PC': {'L23_PC': 0.0}},
    
    # Fallback mean probabilities for pathways without spatial fits
    'pmat': { 
        'L4_SS': {'L4_SS': 0.05},
        'L5_TTPC1': {'L23_PC': 0.01}
    }
}

# 3. Generate the matrix
adj_matrix = build_adjacency_matrix(cell_mtypes, cell_coords, conn_data)

print("Generated Adjacency Matrix (5x5):")
print(adj_matrix)
