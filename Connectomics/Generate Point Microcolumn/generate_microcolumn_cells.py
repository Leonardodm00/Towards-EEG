def generate_microcolumn_cells(input_dict, bbp_results, verbose=True):
    """
    Generates cell m-types and 3D coordinates based on top-level densities
    and biological percentage breakdowns.
    
    Parameters:
    - input_dict: Dictionary containing 'Layers' (Z bounds), 
                  'Cells' (densities grouped by Layer -> BioType), 
                  and 'Geometry' (radius).
    - bbp_results: Nested dictionary mapping Layer -> BioType -> MType -> %
    - verbose: Boolean. If True, prints a detailed breakdown of generated cells.
    
    Returns:
    - cell_mtypes: 1D numpy array of string m-types.
    - cell_coords: 2D numpy array of shape (N, 3) containing (x, y, z).
    """
    
    radius = input_dict['Geometry']['radius']
    
    mtypes_list = []
    coords_list = []
    
    # Pre-calculate the column's cross-sectional area
    area = np.pi * (radius ** 2)
    
    if verbose:
        print("BUILDING MICROCIRCUIT...")
        print("="*40)
    
    for layer, bio_types in input_dict['Cells'].items():
        if layer not in input_dict['Layers']:
            if verbose: print(f"Warning: Z-boundaries for '{layer}' not defined. Skipping.")
            continue
            
        # Unpack bounds and calculate layer volume
        bounds = input_dict['Layers'][layer]
        z_min, z_max = min(bounds), max(bounds)
        layer_height = z_max - z_min
        volume_um3 = area * layer_height
        
        if verbose:
            print(f"\n--- {layer} (Height: {layer_height} um) ---")
        
        for raw_bio_type, density_mm3 in bio_types.items():
            if density_mm3 <= 0:
                continue
            
            # 1. Normalize the bio_type key (handles 'inh', 'exc', 'Inhibitory', etc.)
            bio_type = 'Inhibitory' if raw_bio_type.lower().startswith('inh') else 'Excitatory'
            
            # 2. Convert density (cells/mm^3) to cells/um^3, calculate TOTAL cells for this group
            density_um3 = density_mm3 / 1e9
            total_group_count = int(np.round(density_um3 * volume_um3))
            
            if total_group_count <= 0:
                continue
                
            # 3. Check if we have BBP distribution data for this layer and type
            if layer not in bbp_results or bio_type not in bbp_results[layer]:
                if verbose: print(f"  Warning: BBP results for {layer} {bio_type} not found. Skipping.")
                continue
                
            if verbose:
                print(f"  {bio_type} (Target Total: ~{total_group_count} cells):")
                
            # 4. Iterate through the BBP sub-populations
            mtype_data = bbp_results[layer][bio_type]
            
            for mtype, data in mtype_data.items():
                percentage = data['percentage']
                
                # Calculate the exact number of cells for this specific m-type
                count = int(np.round(total_group_count * (percentage / 100.0)))
                
                if count <= 0:
                    continue
                    
                if verbose:
                    print(f"    -> {mtype:<12}: {count:>5} cells ({percentage:>5.2f}%)")
                    
                # 5. Generate uniform random spatial positions
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
                
    # Concatenate everything into the final format
    if mtypes_list:
        cell_mtypes = np.concatenate(mtypes_list)
        cell_coords = np.vstack(coords_list)
        if verbose:
            print("="*40)
            print(f"SUCCESS: Generated {len(cell_mtypes)} total neurons.")
    else:
        cell_mtypes = np.array([])
        cell_coords = np.empty((0, 3))
        if verbose:
            print("WARNING: No cells were generated. Check your input dictionary.")
        
    return cell_mtypes, cell_coords


