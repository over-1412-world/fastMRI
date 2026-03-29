"""
Consolidate all Phase 2 experiment results into a single Excel file with:
- Summary sheet (all experiments)
- Experiment detail sheets (1 per experiment)
- Statistical comparison
"""

import pandas as pd
import os
from pathlib import Path
import re

def extract_experiment_info(dirname):
    """Parse phase2 directory name to extract metadata."""
    # Examples:
    # phase2_min_compare_seed3_e1 -> seed=3, epochs=1, type='smoke'
    # phase2_nextaction_axT1post202_e15_seed3 -> data=AXT1POST_202, epochs=15, seed=3
    # phase2_nextaction_data6_e1_seed42 -> data=data6, epochs=1, seed=42
    
    meta = {
        'dirname': dirname,
        'seed': None,
        'epochs': None,
        'data': None,
        'type': 'unknown'
    }
    
    if 'smoke' in dirname:
        meta['type'] = 'smoke'
    elif 'min_compare' in dirname:
        meta['type'] = 'baseline'
    else:
        meta['type'] = 'nextaction'
    
    # Extract seed (e.g., seed42, seed3)
    seed_match = re.search(r'seed(\d+)', dirname)
    if seed_match:
        meta['seed'] = int(seed_match.group(1))
    
    # Extract epochs (e.g., e1, e5, e15, e8)
    epoch_match = re.search(r'_e(\d+)(_|$)', dirname)
    if epoch_match:
        meta['epochs'] = int(epoch_match.group(1))
    
    # Extract data type (e.g., axT1post202, data6, data1, full)
    if 'axT1post202' in dirname:
        meta['data'] = 'AXT1POST_202'
    elif 'data1_' in dirname:
        meta['data'] = 'data1_subset'
    elif 'data6' in dirname:
        meta['data'] = 'data6_subset'
    elif 'full' in dirname:
        meta['data'] = 'full_dataset'
    elif 'min_compare' in dirname:
        meta['data'] = 'min_compare'
    
    return meta

def consolidate_results(output_dir):
    """Read all phase2 results and consolidate."""
    results = []
    experiment_data = {}
    
    # Find all phase2 directories
    for dirname in sorted(os.listdir(output_dir)):
        if not dirname.startswith('phase2_'):
            continue
        
        exp_dir = os.path.join(output_dir, dirname)
        summary_csv = os.path.join(exp_dir, 'phase2_summary.csv')
        
        if not os.path.exists(summary_csv):
            print(f"  Skipping {dirname}: no phase2_summary.csv")
            continue
        
        try:
            df = pd.read_csv(summary_csv)
            meta = extract_experiment_info(dirname)
            
            # Add metadata columns
            for key, val in meta.items():
                df[key] = val
            
            results.append(df)
            experiment_data[dirname] = {
                'meta': meta,
                'data': df
            }
            print(f"  Loaded: {dirname} (epochs={meta['epochs']}, seed={meta['seed']}, data={meta['data']})")
        except Exception as e:
            print(f"  Error loading {dirname}: {e}")
    
    if not results:
        print("No phase2 results found!")
        return None, None
    
    # Combine all results
    df_all = pd.concat(results, ignore_index=True)
    
    # Sort by epochs, seed
    df_all = df_all.sort_values(['epochs', 'seed']).reset_index(drop=True)
    
    return df_all, experiment_data

def main():
    output_dir = r'c:\Users\s2520\fastMRI\outputs'
    excel_file = os.path.join(output_dir, 'phase2_consolidated_results.xlsx')
    
    print("Consolidating Phase 2 results...")
    df_all, exp_data = consolidate_results(output_dir)
    
    if df_all is None:
        return
    
    # Create Excel with multiple sheets
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Sheet 1: Master summary (all experiments, 4 conditions per experiment)
        df_all.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Sheet 2: Aggregated by (epochs, seed, condition)
        df_summary = df_all.groupby(['epochs', 'data', 'condition']).agg({
            'n': 'first',
            'psnr_mean': ['mean', 'std', 'min', 'max'],
            'ssim_mean': ['mean', 'std', 'min', 'max'],
        }).round(6)
        df_summary.to_excel(writer, sheet_name='Summary_by_Config')
        
        # Sheet 3: Best condition per (epochs, data)
        df_best = []
        for (epochs, data), grp in df_all.groupby(['epochs', 'data']):
            for condition in df_all['condition'].unique():
                cond_data = grp[grp['condition'] == condition]
                if not cond_data.empty:
                    df_best.append(cond_data.iloc[0])
        df_best = pd.DataFrame(df_best).sort_values(['epochs', 'data', 'condition'])
        df_best.to_excel(writer, sheet_name='Best_by_Config', index=False)
        
        # Sheet 4: Latest experiment (15-epoch)
        if 'phase2_nextaction_axT1post202_e15_seed3' in exp_data:
            df_e15 = exp_data['phase2_nextaction_axT1post202_e15_seed3']['data']
            df_e15.to_excel(writer, sheet_name='Latest_E15_AXT1POST_202', index=False)
    
    print(f"\nGenerated: {excel_file}")
    print(f"\nSummary (all {len(df_all)} rows):")
    print(df_all[['epochs', 'data', 'condition', 'psnr_mean', 'psnr_std', 'ssim_mean']].to_string())

if __name__ == '__main__':
    main()
