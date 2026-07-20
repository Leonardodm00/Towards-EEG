# ==========================================
# Execution and Report Generator
# ==========================================

# Run the parser
bbp_results, bbp_totals = calculate_bbp_relative_presences('S1-cells-distributions-Rat.txt')
# 2. Save the results
if bbp_results:
    save_composition_results(bbp_results)
if bbp_results:
    print("BBP ORIGINAL MICROCIRCUIT COMPOSITION")
    print("="*45)

    # Sort layers (L1, L23, L4, L5, L6)
    for layer in sorted(bbp_results.keys()):
        print(f"\n--- {layer} ---")
        
        for bio_type in ['Excitatory', 'Inhibitory']:
            total = bbp_totals[layer][bio_type]
            if total > 0:
                print(f"  {bio_type} (Total: {total}):")
                
                # Sort m-types by percentage (highest to lowest)
                mtypes_data = bbp_results[layer][bio_type]
                sorted_mtypes = sorted(mtypes_data.items(), key=lambda item: item[1]['percentage'], reverse=True)
                
                for mtype, data in sorted_mtypes:
                    perc = data['percentage']
                    count = data['count']
                    print(f"    • {mtype:<12}: {perc:>6.2f}%  ({count} cells)")
