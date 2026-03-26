import numpy as np

def generate_microcolumn_cells(input_dict, mtype_fast_lookup):
    """
    Generates cell m-types and 3D coordinates based on cell densities.
    
    Parameters:
    - input_dict: Dictionary containing 'Layers' (Z bounds), 
                  'Cells' (densities in cells/mm^3), and 'Geometry' (radius).
    - mtype_fast_lookup: The O(1) dictionary mapping m-types to their layer.
    
    Returns:
    - cell_mtypes: 1D numpy array of string m-types.
    - cell_coords: 2D numpy array of shape (N, 3) containing (x, y, z).
    """
    
    radius = input_dict['Geometry']['radius']
    
    mtypes_list = []
    coords_list = []
    
    # Pre-calculate the column's cross-sectional area
    area = np.pi * (radius ** 2)
    
    for mtype, density_mm3 in input_dict['Cells'].items():
        if density_mm3 <= 0:
            continue
            
        if mtype not in mtype_fast_lookup:
            print(f"Warning: '{mtype}' not found in lookup dictionary. Skipping.")
            continue
            
        layer = mtype_fast_lookup[mtype][0]
        
        if layer not in input_dict['Layers']:
            print(f"Warning: Z-boundaries for '{layer}' not defined. Skipping.")
            continue
            
        # Unpack bounds
        bounds = input_dict['Layers'][layer]
        z_min, z_max = min(bounds), max(bounds)
        
        # Calculate volume of this specific layer slice (in cubic micrometers)
        layer_height = z_max - z_min
        volume_um3 = area * layer_height
        
        # Convert density from cells/mm^3 to cells/um^3, then calculate total cells
        density_um3 = density_mm3 / 1e9
        count = int(np.round(density_um3 * volume_um3))
        
        if count <= 0:
            continue
            
        # Generate uniform random positions in the cylinder slice
        r_random = radius * np.sqrt(np.random.rand(count))
        theta_random = np.random.rand(count) * 2 * np.pi
        
        x = r_random * np.cos(theta_random)
        y = r_random * np.sin(theta_random)
        z = np.random.uniform(z_min, z_max, count)
        
        # Stack the generated data
        coords = np.column_stack((x, y, z))
        coords_list.append(coords)
        
        mtypes = np.full(count, mtype, dtype=object)
        mtypes_list.append(mtypes)
        
    if mtypes_list:
        cell_mtypes = np.concatenate(mtypes_list)
        cell_coords = np.vstack(coords_list)
    else:
        cell_mtypes = np.array([])
        cell_coords = np.empty((0, 3))
        
    return cell_mtypes, cell_coords
