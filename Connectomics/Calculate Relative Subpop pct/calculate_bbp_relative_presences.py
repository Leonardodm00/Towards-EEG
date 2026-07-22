import os

def calculate_bbp_relative_presences(file_path='S1-cells-distributions-Rat.txt'):
    """
    Parses the BBP S1 distribution file and calculates the relative percentage 
    of each m-type among all neurons of the same biological type in its layer.
    """
    
    # Expanded acronym map to include L1-specific inhibitory cells from the BBP data
    acronym_map = {
        # Excitatory Types
        'PC': 'Excitatory', 'SS': 'Excitatory', 'SP': 'Excitatory',
        'TTPC1': 'Excitatory', 'TTPC2': 'Excitatory', 'STPC': 'Excitatory',
        'UTPC': 'Excitatory', 'BPC': 'Excitatory', 'IPC': 'Excitatory',
        'TPC_L1': 'Excitatory', 'TPC_L4': 'Excitatory',
        
        # Inhibitory Types
        'LBC': 'Inhibitory', 'NBC': 'Inhibitory', 'SBC': 'Inhibitory',
        'ChC': 'Inhibitory', 'MC': 'Inhibitory', 'BTC': 'Inhibitory',
        'DBC': 'Inhibitory', 'BP': 'Inhibitory', 'NGC': 'Inhibitory',
        'HAC': 'Inhibitory', 'DAC': 'Inhibitory', 'SAC': 'Inhibitory',
        
        # L1 Specific Inhibitory Types
        'NGC-DA': 'Inhibitory', 'NGC-SA': 'Inhibitory', 
        'DLAC': 'Inhibitory', 'SLAC': 'Inhibitory'
    }

    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return None

    mtype_raw_counts = {}
    
    # 1. Parse the BBP text file
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    mtype = parts[1]
                    m_count = int(parts[4]) # Column 5 'm' is the total m-type count
                    
                    # Since the file lists multiple e-types per m-type, 
                    # we just overwrite with the same 'm_count' total for that m-type
                    mtype_raw_counts[mtype] = m_count

    # 2. Group into layers and sum up the totals for Excitatory/Inhibitory
    layer_totals = {}
    composition = {}
    
    for mtype, count in mtype_raw_counts.items():
        mtype_parts = mtype.split('_', 1)
        layer = mtype_parts[0]
        acronym = mtype_parts[1] if len(mtype_parts) > 1 else mtype
        
        # Handle the TPC exceptions (e.g. 'TPC_L1', 'TPC_L4')
        if acronym.startswith('TPC'):
            acronym = mtype.replace(f"{layer}_", "")
            
        bio_type = acronym_map.get(acronym, 'Unknown')
        
        if layer not in composition:
            composition[layer] = {'Excitatory': {}, 'Inhibitory': {}}
            layer_totals[layer] = {'Excitatory': 0, 'Inhibitory': 0}
            
        if bio_type in ['Excitatory', 'Inhibitory']:
            composition[layer][bio_type][mtype] = count
            layer_totals[layer][bio_type] += count

    # 3. Calculate the relative percentages
    results = {}
    for layer in composition:
        results[layer] = {'Excitatory': {}, 'Inhibitory': {}}
        
        for bio_type in ['Excitatory', 'Inhibitory']:
            total_cells = layer_totals[layer][bio_type]
            
            for mtype, count in composition[layer][bio_type].items():
                perc = (count / total_cells * 100) if total_cells > 0 else 0.0
                results[layer][bio_type][mtype] = {
                    'count': count,
                    'percentage': perc
                }
                
    return results, layer_totals
import json
import csv
import os

def save_composition_results(results, json_path='bbp_composition.json', csv_path='bbp_composition.csv'):
    """
    Saves the nested results dictionary into both JSON and flattened CSV formats.
    """
    # 1. Save as JSON (keeps the Layer -> BioType -> MType structure)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Successfully saved hierarchical data to: {json_path}")

    # 2. Save as CSV (flattens the data for Excel/Pandas)
    # Define headers
    headers = ['Layer', 'Biological_Type', 'MType', 'Count', 'Percentage']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for layer, types in results.items():
            for bio_type, mtypes in types.items():
                for mtype, data in mtypes.items():
                    writer.writerow([
                        layer, 
                        bio_type, 
                        mtype, 
                        data['count'], 
                        f"{data['percentage']:.4f}"
                    ])
    print(f"Successfully saved flattened table to: {csv_path}")
