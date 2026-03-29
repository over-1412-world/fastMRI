"""
Consolidate Phase 3 (30-epoch) results and merge with Phase 2 results into Excel.
This script is run after phase3_e30_validation_seed5/phase3_summary.csv is generated.
"""

import pandas as pd
import os
from pathlib import Path

def consolidate_phase3_with_phase2():
    """Combine Phase 2 and Phase 3 results into comprehensive Excel."""
    
    output_dir = r'c:\Users\s2520\fastMRI\outputs'
    
    # 1) Read Phase 2 consolidated results
    phase2_xlsx = os.path.join(output_dir, 'phase2_consolidated_results.xlsx')
    
    try:
        phase2_all = pd.read_excel(phase2_xlsx, sheet_name='All_Results')
        print(f"Loaded Phase 2 consolidated: {len(phase2_all)} rows")
    except Exception as e:
        print(f"Error loading Phase 2: {e}")
        return
    
    # 2) Read Phase 3 results
    phase3_csv = os.path.join(output_dir, 'phase3_e30_validation_seed5', 'phase3_summary.csv')
    
    if not os.path.exists(phase3_csv):
        print(f"Phase 3 results not ready: {phase3_csv}")
        print("Run this script after phase3_e30_validation_seed5/phase3_summary.csv is generated.")
        return
    
    try:
        phase3_df = pd.read_csv(phase3_csv)
        phase3_df['epochs'] = 30
        phase3_df['data'] = 'AXT1POST_202'
        phase3_df['seed'] = 5  # 5 seeds aggregated
        phase3_df['type'] = 'phase3_extended'
        
        print(f"Loaded Phase 3 results: {len(phase3_df)} rows")
    except Exception as e:
        print(f"Error loading Phase 3: {e}")
        return
    
    # 3) Combine Phase 2 and Phase 3
    combined = pd.concat([phase2_all, phase3_df], ignore_index=True)
    
    # 4) Create comprehensive Excel with multiple sheets
    output_xlsx = os.path.join(output_dir, 'phase2_phase3_consolidated_results.xlsx')
    
    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
        # Sheet 1: All results
        combined.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Sheet 2: Phase 3 detailed (30-epoch only)
        phase3_df.to_excel(writer, sheet_name='Phase3_E30_AXT1POST_202', index=False)
        
        # Sheet 3: Comparison table (5e, 15e, 30e side-by-side)
        comparison = []
        for cond in ['fixed_ifft', 'phase_only', 'phase_amp_hermitian', 'phase_amp_no_hermitian']:
            row = {'condition': cond}
            
            # 5-epoch
            data_5e = phase2_all[(phase2_all['epochs'] == 5) & (phase2_all['condition'] == cond)]
            if not data_5e.empty:
                row['e5_psnr'] = f"{data_5e['psnr_mean'].values[0]:.4f}"
                row['e5_ssim'] = f"{data_5e['ssim_mean'].values[0]:.6f}"
            
            # 15-epoch
            data_15e = phase2_all[(phase2_all['epochs'] == 15) & (phase2_all['condition'] == cond)]
            if not data_15e.empty:
                row['e15_psnr'] = f"{data_15e['psnr_mean'].values[0]:.4f}"
                row['e15_ssim'] = f"{data_15e['ssim_mean'].values[0]:.6f}"
            
            # 30-epoch
            data_30e = phase3_df[phase3_df['condition'] == cond]
            if not data_30e.empty:
                row['e30_psnr'] = f"{data_30e['psnr_mean'].values[0]:.4f}"
                row['e30_ssim'] = f"{data_30e['ssim_mean'].values[0]:.6f}"
            
            comparison.append(row)
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison.to_excel(writer, sheet_name='Epoch_Comparison', index=False)
        
        # Sheet 4: Statistical summary by epoch
        summary_by_epoch = combined.groupby('epochs').agg({
            'psnr_mean': ['min', 'max', 'mean', 'std'],
            'ssim_mean': ['min', 'max', 'mean', 'std'],
        }).round(6)
        summary_by_epoch.to_excel(writer, sheet_name='Summary_by_Epoch')
    
    print(f"\nGenerated: {output_xlsx}")
    print(f"Sheets: All_Results, Phase3_E30_AXT1POST_202, Epoch_Comparison, Summary_by_Epoch")
    
    # 5) Print final results
    print("\n" + "="*70)
    print("PHASE 3 RESULTS (30-EPOCH × 5 SEEDS)")
    print("="*70)
    
    if not phase3_df.empty:
        for _, row in phase3_df.iterrows():
            baseline = phase3_df[phase3_df['condition'] == 'fixed_ifft']['psnr_mean'].values[0]
            delta = row['psnr_mean'] - baseline
            print(f"\n{row['condition']:30s}")
            print(f"  PSNR: {row['psnr_mean']:.4f} ± {row['psnr_std']:.4f} ({delta:+.4f} dB)")
            print(f"  SSIM: {row['ssim_mean']:.6f} ± {row['ssim_std']:.6f}")
    else:
        print("Phase 3 results not available yet.")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    consolidate_phase3_with_phase2()
