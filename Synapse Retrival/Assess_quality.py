import os
import pandas as pd

def assess_synapses_for_neurons(neuron_ids, input_dir='./h01_extracted_synapses'):
    """
    Iterates over a list of neuron IDs, finds their corresponding synapse CSV files,
    performs a quality assessment on each, and returns a compiled summary DataFrame.
    """
    all_stats = []

    for nid in neuron_ids:
        csv_path = os.path.join(input_dir, f"neuron_{nid}_synapses.csv")
        
        print(f"\n📊 --- QA Report: Neuron {nid} ---")
        
        # 1. Check if file exists
        if not os.path.exists(csv_path):
            print(f"⚠️ CSV file not found: {csv_path}")
            all_stats.append({'neuron_id': nid, 'status': 'file_not_found'})
            continue
            
        df = pd.read_csv(csv_path)
        
        # 2. Check if file is empty
        if df.empty:
            print(f"⚠️ The file is empty (0 synapses).")
            all_stats.append({'neuron_id': nid, 'status': 'empty_file', 'total': 0})
            continue
        
        # 3. Calculate Stats
        total = len(df)
        dir_counts = df['direction'].fillna('unknown').value_counts().to_dict()
        incoming = dir_counts.get('incoming', 0)
        outgoing = dir_counts.get('outgoing', 0)
        
        type_counts = df['synapse_type'].fillna('unspecified').value_counts().to_dict()
        breakdown = df.groupby('direction')['synapse_type'].value_counts().unstack(fill_value=0).to_dict('index')
        
        # 4. Print the single-neuron report
        print(f"Total Synapses: {total}")
        print("📍 By Direction:")
        print(f"  * Incoming: {incoming}")
        print(f"  * Outgoing: {outgoing}")
        
        print("🧬 By Type (Global):")
        for s_type, count in type_counts.items():
            print(f"  * {s_type}: {count}")
            
        print("🔍 Detailed Breakdown:")
        for direction, types in breakdown.items():
            print(f"  [{direction.upper()}]")
            for s_type, count in types.items():
                if count > 0:
                    print(f"    - {s_type}: {count}")
        print("-" * 40)
        
        # 5. Append to batch summary
        stats_dict = {
            'neuron_id': nid,
            'status': 'success',
            'total': total,
            'incoming': incoming,
            'outgoing': outgoing
        }
        
        # Add the specific type counts (e.g., 'type_1', 'type_2') to the row
        for s_type, count in type_counts.items():
            stats_dict[f'type_{s_type}'] = count
            
        all_stats.append(stats_dict)

    # 6. Compile everything into a final DataFrame
    summary_df = pd.DataFrame(all_stats)
    
    # Reorder columns slightly to make it prettier if it has data
    if not summary_df.empty and 'status' in summary_df.columns:
        # Move neuron_id and status to the front
        cols = ['neuron_id', 'status'] + [c for c in summary_df.columns if c not in ['neuron_id', 'status']]
        summary_df = summary_df[cols]
        
    return summary_df






output_dir = '/content/drive/MyDrive/Colab Notebooks/Synapse database'
target_ids = [ 5390777283]


assess_synapses_for_neurons(target_ids,output_dir)
