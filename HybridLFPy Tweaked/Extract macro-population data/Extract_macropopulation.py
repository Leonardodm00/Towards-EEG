import numpy as np
import plotly.graph_objects as go
from collections import Counter

# 1. The Updated Extraction Function (Now includes 'cell_mtypes')
def extract_macro_populations_debug(connectomics_data, name_list):
    mtype_fast_lookup = connectomics_data['mtype_fast_lookup']
    cell_mtypes = connectomics_data['cell_mtypes']
    cell_coords = connectomics_data['cell_coords']
    
    extracted_data = {}
    
    for name in name_list:
        parts = name.split('_')
        target_layer = parts[0].upper()
        target_group = parts[1].lower()
        
        global_indices = []
        
        for raw_idx, mtype in enumerate(cell_mtypes):
            if mtype not in mtype_fast_lookup: continue
            layer, bio_type = mtype_fast_lookup[mtype]
            
            is_match = False
            if layer == target_layer:
                if target_group == 'exc' and bio_type == 'Excitatory':
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
                
        global_indices = np.array(global_indices)
        
        if len(global_indices) == 0:
            print(f"Warning: No cells found for '{name}'")
            continue
            
        # Pack the required debug info
        extracted_data[name] = {
            'cell_coords': cell_coords[global_indices],
            'cell_mtypes': cell_mtypes[global_indices]
        }
        
    return extracted_data
