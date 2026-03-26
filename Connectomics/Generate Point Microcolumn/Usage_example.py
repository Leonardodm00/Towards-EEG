# ==========================================
# Example Usage
# ==========================================

column_input = {
    'Layers': {
        'L1':  [-200.0, 0.0],       
        'L23': [-700.0, -200.0],
        'L4':  [-1000.0, -700.0],
        'L5':  [-1600.0, -1000.0],
        'L6':  [-2000.0, -1600.0]   
    },
    'Cells': {
        # Densities expressed in cells per cubic millimeter (cells/mm^3)
        'L23_PC': 35000, 
        'L23_LBC': 2500,
        'L4_SS': 45000,
        'L5_TTPC1': 20000
    },
    'Geometry': {
        'radius': 210.0 # micrometers
    }
}
cell_mtypes, cell_coords = generate_microcolumn_cells(column_input, mtype_fast_lookup)
