"""
Phase 4: Full Dataset Experiment Setup
Generates command templates for running phase 4 experiments on complete AXT1POST dataset.

Usage:
    python scripts/phase4_experiment_template.py --output-dir outputs/phase4_commands.txt
"""

import os
from pathlib import Path
import argparse


def generate_phase4_commands():
    """Generate Phase 4 experiment commands"""
    
    commands = {
        'full_dataset_e30_seed3': {
            'description': 'Main experiment: Full AXT1POST dataset, 30-epoch, 3 seeds',
            'cmd': '''python mri_strip_reconst/scripts/run_phase2_min_compare.py \\
  --glob "data/fastmri/brain_multicoil/training/AXT1POST_*.h5" \\
  --val-glob "data/fastmri/brain_multicoil/validation/AXT1POST_*.h5" \\
  --epochs 30 \\
  --seeds 42 43 44 \\
  --batch-size 1 \\
  --workers 0 \\
  --base-save-dir "outputs/phase4_full_axt1post_e30_seed3" \\
  --coil-compress 0 \\
  --target-ky 640 \\
  --target-kx 320 \\
  --save-exclusion-log \\
  --amp-eps 0.1 \\
  --lambda-phys 0.01 \\
  --patience 5''',
            'duration': '24-36 hours',
            'data_scale': '~200 train, ~50 val files (~3200 train, ~800 val slices)'
        },
        
        'ablation_amp_eps_sensitivity': {
            'description': 'Ablation: amp_eps sensitivity (0.05, 0.1, 0.2)',
            'cmd': '''# Run 3 conditions with different amp_eps values
for AMP_EPS in 0.05 0.1 0.2; do
  python mri_strip_reconst/scripts/train_separable_ifft.py \\
    --glob "data/fastmri/brain_multicoil/training/AXT1POST_*.h5" \\
    --val-glob "data/fastmri/brain_multicoil/validation/AXT1POST_*.h5" \\
    --epochs 20 \\
    --seed 42 \\
    --batch-size 1 \\
    --amp-eps $AMP_EPS \\
    --save-dir "outputs/phase4_ablation_amp_eps_$AMP_EPS"
done''',
            'duration': '18-24 hours',
            'note': 'Tests amplitude correction magnitude sensitivity'
        },
        
        'ablation_rank_sensitivity': {
            'description': 'Ablation: Rank sensitivity for coil compression (if applicable)',
            'cmd': '''# Note: Currently coil_compress=0 (no SVD)
# This would be relevant if using coil compression

for RANK in 4 8 16 32; do
  python mri_strip_reconst/scripts/train_separable_ifft.py \\
    --glob "data/fastmri/brain_multicoil/training/AXT1POST_*.h5" \\
    --epochs 20 \\
    --seed 42 \\
    --coil-compress $RANK \\
    --save-dir "outputs/phase4_ablation_rank_$RANK"
done''',
            'duration': '20-30 hours',
            'note': 'Tests coil compression rank effect (optional)'
        },
        
        'quick_validation_knee': {
            'description': 'Quick validation on knee data (different anatomy)',
            'cmd': '''python mri_strip_reconst/scripts/run_phase2_min_compare.py \\
  --glob "data/fastmri/knee_multicoil/training/SAGITTAL_*.h5" \\
  --val-glob "data/fastmri/knee_multicoil/validation/SAGITTAL_*.h5" \\
  --epochs 15 \\
  --seeds 42 43 44 \\
  --batch-size 1 \\
  --base-save-dir "outputs/phase4_knee_sagittal_e15" \\
  --coil-compress 0 \\
  --target-ky 640 \\
  --target-kx 320 \\
  --save-exclusion-log \\
  --amp-eps 0.1 \\
  --lambda-phys 0.01''',
            'duration': '12-18 hours',
            'note': 'Validates generalization to different anatomy'
        }
    }
    
    return commands


def main():
    parser = argparse.ArgumentParser(description='Generate Phase 4 Experiment Commands')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    commands = generate_phase4_commands()
    
    # Generate markdown command file
    cmd_file = output_dir / 'phase4_experiment_commands.md'
    
    with open(cmd_file, 'w', encoding='utf-8') as f:
        f.write("# Phase 4: Experiment Commands\n\n")
        f.write("## Overview\n\n")
        f.write("Phase 4 consists of:\n")
        f.write("1. **Main Experiment**: Full AXT1POST dataset (200+ train files)\n")
        f.write("2. **Ablation Studies**: amp_eps, rank, Hermitian sensitivity\n")
        f.write("3. **Generalization**: Cross-anatomy validation (knee data)\n\n")
        
        f.write("---\n\n")
        
        for exp_name, exp_config in commands.items():
            f.write(f"## {exp_config['description']}\n\n")
            f.write(f"**Experiment ID**: `{exp_name}`\n\n")
            f.write(f"**Estimated Duration**: {exp_config['duration']}\n\n")
            
            if 'data_scale' in exp_config:
                f.write(f"**Data Scale**: {exp_config['data_scale']}\n\n")
            
            if 'note' in exp_config:
                f.write(f"**Note**: {exp_config['note']}\n\n")
            
            f.write("### Command\n\n")
            f.write("```bash\n")
            f.write(exp_config['cmd'])
            f.write("\n```\n\n")
            
            f.write("---\n\n")
    
    print(f"✅ Generated command file: {cmd_file}\n")
    print("Available experiments:")
    for exp_name, exp_config in commands.items():
        print(f"  - {exp_name}: {exp_config['description']}")


if __name__ == '__main__':
    main()
