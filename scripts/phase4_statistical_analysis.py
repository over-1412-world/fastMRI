"""
Phase 4 Statistical Analysis: Compare 5-epoch, 15-epoch, and 30-epoch results
and generate Go/No-Go decision for full dataset experiment.

Usage:
    python scripts/phase4_statistical_analysis.py \
      --e5-csv outputs/phase2_nextaction_axT1post202_e5_seed3/phase2_summary.csv \
      --e15-csv outputs/phase2_nextaction_axT1post202_e15_seed3/phase2_summary.csv \
      --e30-csv outputs/phase3_e30_validation_seed5/phase3_summary.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


class Phase4StatisticalAnalysis:
    def __init__(self, output_dir='outputs/phase4_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self, e5_csv, e15_csv, e30_csv=None):
        """複数の実験結果を読込"""
        df_e5 = pd.read_csv(e5_csv)
        df_e5['epoch'] = 5
        
        df_e15 = pd.read_csv(e15_csv)
        df_e15['epoch'] = 15
        
        dfs = [df_e5, df_e15]
        
        if e30_csv and Path(e30_csv).exists():
            df_e30 = pd.read_csv(e30_csv)
            df_e30['epoch'] = 30
            dfs.append(df_e30)
        
        combined = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded results: {len(combined)} total rows")
        print(f"   Epochs: {sorted(combined['epoch'].unique())}")
        print(f"   Conditions: {list(combined['condition'].unique())}\n")
        
        return combined
    
    def compare_conditions(self, df):
        """条件間の性能差を統計検定"""
        print("=" * 90)
        print("STATISTICAL COMPARISON ACROSS CONDITIONS")
        print("=" * 90 + "\n")
        
        conditions = sorted(df['condition'].unique())
        
        # 各条件の PSNR を抽出（全 epoch 平均）
        psnr_by_condition = {}
        for cond in conditions:
            psnr_values = df[df['condition'] == cond]['psnr_mean'].values
            psnr_by_condition[cond] = psnr_values
        
        # 結果テーブル
        print(f"{'Condition':<35} {'Mean PSNR':<15} {'Std':<10} {'N':<5}")
        print("-" * 90)
        
        for cond in conditions:
            values = psnr_by_condition[cond]
            print(f"{cond:<35} {values.mean():>10.4f} dB  {values.std():>8.4f}   {len(values):>3}")
        
        # Baseline (fixed_ifft) との比較
        baseline_psnr = psnr_by_condition['fixed_ifft']
        
        print(f"\n{'Comparison vs. fixed_ifft (baseline)':^90}")
        print("-" * 90 + "\n")
        
        comparison_results = []
        
        for cond in conditions:
            if cond == 'fixed_ifft':
                continue
            
            test_psnr = psnr_by_condition[cond]
            
            # t検定（対応なし）
            t_stat, p_value = stats.ttest_ind(test_psnr, baseline_psnr)
            
            # 効果量（Cohen's d）
            n1, n2 = len(test_psnr), len(baseline_psnr)
            pooled_std = np.sqrt(((n1-1)*test_psnr.std()**2 + (n2-1)*baseline_psnr.std()**2) / (n1 + n2 - 2))
            cohens_d = (test_psnr.mean() - baseline_psnr.mean()) / (pooled_std + 1e-8)
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            delta = test_psnr.mean() - baseline_psnr.mean()
            
            print(f"{cond:<35} ΔμPSNR: {delta:+8.4f} dB | "
                  f"p={p_value:.4f} {significance:<3} | Cohen's d = {cohens_d:+.3f}")
            
            comparison_results.append({
                'condition': cond,
                'delta_psnr': delta,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significance': significance,
                'mean_psnr': test_psnr.mean(),
            })
        
        return pd.DataFrame(comparison_results)
    
    def analyze_convergence(self, df):
        """Epoch による収束を可視化"""
        print("\n" + "=" * 90)
        print("CONVERGENCE ANALYSIS (5-epoch → 15-epoch → 30-epoch)")
        print("=" * 90 + "\n")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = {
            'fixed_ifft': '#000000',
            'phase_only': '#0066CC',
            'phase_amp_hermitian': '#FF0000',
            'phase_amp_no_hermitian': '#FF9900'
        }
        
        for cond in sorted(df['condition'].unique()):
            df_cond = df[df['condition'] == cond].groupby('epoch').agg({
                'psnr_mean': ['mean', 'std']
            }).reset_index()
            
            df_cond.columns = ['epoch', 'mean_psnr', 'std_psnr']
            
            ax.errorbar(df_cond['epoch'], df_cond['mean_psnr'], 
                       yerr=df_cond['std_psnr'],
                       marker='o', label=cond, linewidth=2.5, markersize=10,
                       color=colors.get(cond, '#666666'), capsize=5, capthick=2)
            
            # データ点をプリント
            for _, row in df_cond.iterrows():
                print(f"  {cond:<35} epoch={row['epoch']:.0f}: "
                      f"{row['mean_psnr']:.4f} ± {row['std_psnr']:.4f} dB")
        
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=13, fontweight='bold')
        ax.set_title('Convergence Analysis: 5-epoch vs 15-epoch vs 30-epoch', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticks([5, 15, 30])
        
        save_path = self.output_dir / 'convergence_analysis.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved convergence plot to {save_path}")
        plt.close()
    
    def estimate_convergence_rate(self, df):
        """収束速度を推定（線形外挿）"""
        print("\n" + "=" * 90)
        print("CONVERGENCE RATE EXTRAPOLATION")
        print("=" * 90 + "\n")
        
        print(f"{'Condition':<35} {'Slope (dB/epoch)':<20} {'Predicted@50e':<15}")
        print("-" * 90)
        
        predictions = {}
        
        for cond in sorted(df['condition'].unique()):
            df_cond = df[df['condition'] == cond].groupby('epoch')['psnr_mean'].mean()
            
            if len(df_cond) >= 2:
                # 線形回帰
                epochs = df_cond.index.values
                psnr = df_cond.values
                
                z = np.polyfit(epochs, psnr, 1)
                slope = z[0]
                
                # 50-epoch への外挿
                max_epoch = epochs.max()
                predicted_50e = psnr[-1] + slope * (50 - max_epoch)
                
                predictions[cond] = {
                    'slope': slope,
                    'predicted_50e': predicted_50e
                }
                
                print(f"{cond:<35} {slope:>+12.6f} dB/e  {predicted_50e:>12.4f} dB")
        
        return predictions
    
    def generate_go_nogo_decision(self, comparison_df, df):
        """Phase 4 への Go/No-Go を判定"""
        print("\n" + "=" * 90)
        print("PHASE 4 GO/NO-GO DECISION")
        print("=" * 90 + "\n")
        
        # 最新の実験結果（最大 epoch）
        max_epoch = df['epoch'].max()
        df_latest = df[df['epoch'] == max_epoch]
        
        baseline_psnr = df_latest[df_latest['condition'] == 'fixed_ifft']['psnr_mean'].values[0]
        best_cond = df_latest.loc[df_latest['psnr_mean'].idxmax()]
        improvement = best_cond['psnr_mean'] - baseline_psnr
        
        print(f"Evaluation at {max_epoch}-epoch:")
        print(f"  Baseline (fixed_ifft) PSNR: {baseline_psnr:.4f} dB")
        print(f"  Best Condition: {best_cond['condition']}")
        print(f"  Best PSNR: {best_cond['psnr_mean']:.4f} dB")
        print(f"  Improvement: {improvement:+.4f} dB\n")
        
        # 判定ロジック
        if improvement >= 0.05:
            decision = "✅ GO TO PHASE 4 FULL DATASET"
            reason = f"Improvement ({improvement:+.4f} dB) exceeds primary threshold (≥0.05 dB)"
            color = "GREEN"
        elif improvement >= 0.02:
            decision = "⚠️  MARGINAL - CONDITIONAL GO"
            reason = f"Improvement ({improvement:+.4f} dB) is positive but below primary threshold"
            color = "YELLOW"
        else:
            decision = "❌ PIVOT TO CONDITIONAL amp_eps STRATEGY"
            reason = f"Improvement ({improvement:+.4f} dB) is insufficient"
            color = "RED"
        
        print(f"Decision: {decision}")
        print(f"Reason: {reason}\n")
        
        # 統計有意性の確認
        if len(comparison_df) > 0:
            print("Statistical Significance (based on latest epoch):")
            for _, row in comparison_df.iterrows():
                print(f"  {row['condition']:<35} "
                      f"p-value={row['p_value']:.4f} {row['significance']}")
        
        # レポート保存
        report_path = self.output_dir / 'phase4_go_nogo_decision.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 90 + "\n")
            f.write("PHASE 4 GO/NO-GO DECISION REPORT\n")
            f.write("=" * 90 + "\n\n")
            f.write(f"Evaluation Epoch: {max_epoch}\n")
            f.write(f"Baseline (fixed_ifft) PSNR: {baseline_psnr:.4f} dB\n")
            f.write(f"Best Condition: {best_cond['condition']}\n")
            f.write(f"Best PSNR: {best_cond['psnr_mean']:.4f} dB\n")
            f.write(f"Improvement: {improvement:+.4f} dB\n\n")
            f.write(f"Decision: {decision}\n")
            f.write(f"Reason: {reason}\n\n")
            
            if len(comparison_df) > 0:
                f.write("Statistical Significance:\n")
                for _, row in comparison_df.iterrows():
                    f.write(f"  {row['condition']:<35} p={row['p_value']:.4f} {row['significance']}\n")
        
        print(f"\n✅ Report saved to {report_path}")
        
        return decision, improvement
    
    def run_full_analysis(self, e5_csv, e15_csv, e30_csv=None):
        """フル分析を実行"""
        print("\n" + "🔬 " * 45)
        print("PHASE 4 STATISTICAL ANALYSIS")
        print("🔬 " * 45 + "\n")
        
        # データロード
        df = self.load_results(e5_csv, e15_csv, e30_csv)
        
        # 比較分析
        comparison_df = self.compare_conditions(df)
        
        # 収束分析
        self.analyze_convergence(df)
        
        # 収束速度推定
        predictions = self.estimate_convergence_rate(df)
        
        # 最終判定
        decision, improvement = self.generate_go_nogo_decision(comparison_df, df)
        
        print("\n" + "=" * 90)
        print("ANALYSIS COMPLETE")
        print("=" * 90)
        
        return {
            'decision': decision,
            'improvement': improvement,
            'comparison': comparison_df,
            'predictions': predictions
        }


def main():
    parser = argparse.ArgumentParser(description='Phase 4 Statistical Analysis')
    parser.add_argument('--e5-csv', required=True, help='Path to 5-epoch summary CSV')
    parser.add_argument('--e15-csv', required=True, help='Path to 15-epoch summary CSV')
    parser.add_argument('--e30-csv', default=None, help='Path to 30-epoch summary CSV')
    parser.add_argument('--output-dir', default='outputs/phase4_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    analyzer = Phase4StatisticalAnalysis(output_dir=args.output_dir)
    results = analyzer.run_full_analysis(args.e5_csv, args.e15_csv, args.e30_csv)
    
    return results


if __name__ == '__main__':
    main()
