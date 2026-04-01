# 1. The Updated Extraction Function (Now includes 'cell_mtypes')
def extract_macro_populations(connectomics_data, name_list):
    """
    Extracts specific macro-populations using the fast lookup table,
    returning only spatial coordinates, mapping indices, and the integrated synapse dictionary.
    """
    # Unpack only the necessary data
    mtype_fast_lookup = connectomics_data['mtype_fast_lookup']
    cell_mtypes = connectomics_data['cell_mtypes']
    cell_coords = connectomics_data['cell_coords']
    
    # NEW: Only grab the integrated synapse_dict
    synapse_dict = connectomics_data['synapse_dict']

    extracted_data = {}

    for name in name_list:
        # 1. Parse the macro-population string (e.g., "L23_exc" -> "L23", "exc")
        parts = name.split('_')
        target_layer = parts[0].upper()
        target_group = parts[1].lower()

        global_indices = []

        # 2. Iterate through the array using the FAST LOOKUP
        for raw_idx, mtype in enumerate(cell_mtypes):
            if mtype not in mtype_fast_lookup:
                continue

            layer, bio_type = mtype_fast_lookup[mtype]

            # 3. Matching Logic
            is_match = False
            if layer == target_layer:
                if target_group == 'exc' and bio_type == 'Excitatory':
                    # Ensure L4_exc doesn't swallow L4_ss if both are requested
                    if target_layer == 'L4' and 'L4_ss' in name_list and 'SS' in mtype:
                        is_match = False
                    else:
                        is_match = True
                elif target_group == 'inh' and bio_type == 'Inhibitory':
                    is_match = True
                elif target_group == 'ss' and 'SS' in mtype:
                    is_match = True

            if is_match:
                global_indices.append(raw_idx)

        # 4. Process the matched sub-population
        global_indices = np.array(global_indices)

        if len(global_indices) == 0:
            print(f"Warning: No cells found for macro-population '{name}'")
            continue

        # Extract Coordinates
        sub_coords = cell_coords[global_indices]

        # Create Mapping: Local (0 to N) -> Raw Global Index
        local_to_raw_map = {local_idx: raw_idx for local_idx, raw_idx in enumerate(global_indices)}

        # NEW: Filter the single synapse_dict (Keys are LOCAL, Values are RAW)
        sub_synapse_dict = {
            local_idx: synapse_dict[raw_idx] 
            for local_idx, raw_idx in enumerate(global_indices) if raw_idx in synapse_dict
        }

        # 5. Pack strictly the requested keys into the return dictionary
        extracted_data[name] = {
            'cell_coords': sub_coords,
            'local_to_raw_map': local_to_raw_map,
            'synapse_dict': sub_synapse_dict
        }

    return extracted_data
