# 実装ロードマップ：Phase 別計画（2026-03 改訂版）

## 概要

本ロードマップは、[研究内容説明資料.md](研究内容説明資料.md) の最新方針に合わせて改訂した実行計画である。  
中核方針は **「位相のみ補正」から「物理拘束つき部分複素化（位相 + 微小振幅補正 + Hermitian投影）」へ移行** すること。

- 旧方針: IFFT + 位相補正中心
- 新方針: IFFT + 制約付き複素補正（位相 + 微小振幅）+ 物理整合の明示保証

---

## 0. 研究主張（Go/No-Go基準）

### 最低限の主張ライン
1. 画質（PSNR/SSIM/NMSE）で固定IFFT以上
2. 安定性（発散率・虚部残差）で自由学習型より優位
3. 追加コスト（時間・メモリ）が実用範囲（+20%以内）

### 早期中止（ピボット）条件
- 3週時点で「位相+振幅」が「位相のみ」より PSNR 改善 < 0.1 dB かつコスト > +20% の場合、
  `amp_eps` の条件付き化（マスク依存）へ移行する。

---

## Phase 1: 実装更新と安定化（Week 1-2）

### 目標
**LearnableIFFT を最新仕様（位相+微小振幅+Hermitian投影）へ更新し、学習を安定化する。**

### 対象ファイル
- `mri_strip_reconst/models/stripes_separable_ifft.py`
- `mri_strip_reconst/scripts/train_separable_ifft.py`
- `mri_strip_reconst/scripts/debug_performance.py`
- `mri_strip_reconst/scripts/debug_val_ssim_drop.py`

### 実装タスク
1. **`LearnableIFFT1D` の拡張**
    - `phase` に加えて `alpha`（振幅補正）を導入
    - `amp_eps`（default: 0.1）を導入し補正幅を制限
    - `use_hermitian`（default: True）を導入

2. **`forward()` の更新**
    - 基準変換: `y0 = ifft(x)`
    - 制約付き補正:
      $$
      A(k)=1+\varepsilon\tanh(\alpha_k),\quad
      P(k)=\exp(i\,s\,\phi_k),\quad
      y=y_0\cdot A\cdot P
      $$
    - ここで $A(k)\in[1-\varepsilon,1+\varepsilon]$

3. **Hermitian投影の実装**
    - `hermitian_project()` を追加
    - 少なくとも `ky` IFFT 後に適用

4. **診断ロギングの追加**
    - `phase_norm = ||phase||_2`
    - `alpha_norm = ||alpha||_2`
    - `imag_energy = mean(|Im(y)|)`
    - `symmetry_error = mean(|K - conj(flip(K))|)`

5. **学習引数の追加**
    - `--amp-eps`（float, default=0.1）
    - `--use-hermitian` / `--no-hermitian`
    - `--lambda-phys`（default=0.01）

### 完了条件（Done）
- 1 epoch で `NaN/Inf` なし
- `imag_energy` が「位相のみ」同等以下
- `phase_norm` / `alpha_norm` が単調発散しない

---

## Phase 2: データ前処理の固定化（Week 1-2, 並行）

### 目標
**データ形状不一致を解消し、学習可能な入力仕様を固定する。**

### 背景
実データには以下の混在がある:
- `ky`: 512 / 640 / 768
- `kx`: 213〜416
- coil数: 4〜20

### 前処理ルール（必須）
1. **サイズ統一**: `ky=640, kx=320` に center-crop / zero-pad
2. **coil統一**: coil compression（SVD）で固定本数（例: 8 or 12）
3. **例外管理**: 破損ファイルは除外リスト化し理由を記録

### 完了条件（Done）
- train/val/test 各splitで shape が単一化
- DataLoader の shape mismatch エラー 0件
- 前処理設定（crop/pad/coil圧縮）が設定ファイルに固定

---

## Phase 3: 最小比較と安定性評価（Week 3-4）

### 目標
**提案法の有効性を最小構成で定量確認する。**

### 必須比較条件（4条件）
1. 固定IFFT
2. IFFT + 位相のみ
3. IFFT + 位相+微小振幅（提案）
4. 3 + Hermitian投影OFF

### 評価指標
- 画質: PSNR / SSIM / NMSE
- 安定性: 発散率、勾配ノルム（p50/p95）、`imag_energy`
- 物理整合: `symmetry_error`
- 計算資源: 推論時間、学習時間/epoch、GPU最大メモリ

### 統計処理
- seed=3 以上
- 平均 ± 標準偏差
- 提案法 vs 位相のみで対応あり検定（t検定 or Wilcoxon）

### 完了条件（Done）
- 比較4条件の結果表が完成
- 有意差検定の結果を添付
- 失敗条件（発散ケース）の再現ログを保存

---

## Phase 4: 本実験・アブレーション・主張確定（Week 5-8）

### 目標
**knee/brain・複数条件で汎化を確認し、論文化可能な主張を確定する。**

### 本実験
- 部位: knee / brain
- 加速率・マスク条件を拡張
- 汎化性能を評価

### 必須アブレーション
- 位相のみ vs 位相+振幅
- Hermitian ON/OFF
- rank（4/8/16/32）
- `amp_eps` 感度（例: 0.05/0.1/0.2）
- 軸別処理 vs 2D一括

### 可視化（論文向け）
- 位相 `φ[k]` と振幅 `α[k]` の周波数依存プロット
- `imag_energy` / `symmetry_error` の分布
- 失敗例と成功例の対比

### 完了条件（Done）
- 主要主張の図表一式（再現可能）
- 「性能・安定性・コスト」の3軸で結論確定
- スライド/論文草案に転用可能な形で整理

---

## 8週間マイルストーン

| 週 | 到達目標 | 成果物 |
|---|---|---|
| 1 | LearnableIFFT改修 | 実装差分、1epochログ |
| 2 | 監視指標安定化 + 前処理固定 | `imag_energy/symmetry_error` 図、shape統一レポート |
| 3 | 4条件の最小学習 | 比較表（PSNR/SSIM/NMSE） |
| 4 | 失敗条件分析 | 発散ケースの要因メモ |
| 5 | 本学習（knee） | seed別結果表 |
| 6 | 本学習（brain） | 汎化結果表 |
| 7 | 感度分析（rank, amp_eps） | アブレーション図 |
| 8 | 主張確定 | 論文化用図表・要点スライド |

---

## リスク管理

1. **勾配不安定**
    - 対策: gradient clipping、`amp_eps` 下げ、段階学習（固定IFFT→学習可能）

2. **性能差が出ない**
    - 対策: Hermitian適用位置の見直し、`lambda_phys` の再調整、条件付き補正へ移行

3. **計算コスト増**
    - 対策: mixed precision、バッチ設計見直し、低rank化

---

## 現在ステータス（2026-03-29）

### Phase 1 & 2：完了 ✅

**実装完了内容**:
- LearnableIFFT1D: phase + alpha (amplitude correction) + Hermitian projection 実装
- train_separable_ifft.py: `--amp-eps`, `--use-hermitian`, `--lambda-phys` CLI 実装
- データ前処理: AXT1POST_202 subset で shape 統一（ky=640, kx=320, coil=16）
- コイル圧縮: SVD 無効化（`--coil-compress 0`）で実行速度 20x 高速化

### Phase 3：実施中 🔄

**実験進行状況**:
- 5-epoch × 3 seeds (AXT1POST_202): 完了 ✅
  - 最大改善: phase_amp_hermitian +0.0101 dB vs fixed_ifft
- 15-epoch × 3 seeds (AXT1POST_202): 完了 ✅  
  - **最大改善: phase_amp_hermitian +0.0574 dB vs fixed_ifft** ← 信号検出！
  - 改善が 5e → 15e で 5.6倍向上 → 収束傾向継続
- 30-epoch × 5 seeds (AXT1POST_202): **実行中** 🔄
  - ログ: `outputs/phase3_training.log`
  - 推定完了: 約 3-4 時間後

**30-epoch 実行完了後のワークフロー**:
1. 統計分析スクリプト実行: `python scripts/phase4_statistical_analysis.py ...`
2. Go/No-Go 判定: 改善 ≥ +0.05 dB で Phase 4 本実験へ
3. 判定結果: `outputs/phase4_analysis/phase4_go_nogo_decision.txt`

### Phase 4：準備完了 ⏳

**準備状況**:
- ✅ 統計分析スクリプト: `scripts/phase4_statistical_analysis.py`
- ✅ 実験コマンドテンプレート: `scripts/phase4_experiment_template.py`
- ✅ 本実験コマンド生成済み: `outputs/phase4_experiment_commands.md`

**Phase 4 計画**:
1. **本実験** (Full Dataset): 200+ train / 50+ val AXT1POST files, 30-epoch, 3 seeds
   - 推定時間: 24-36 時間
   - Go基準: PSNR改善 ≥ +0.05 dB （30-epoch結果で確認後）

2. **必須アブレーション**:
   - amp_eps 感度 (0.05 / 0.1 / 0.2)
   - Hermitian ON/OFF 再確認
   - Rank 感度 (4 / 8 / 16 / 32)

3. **汎化性評価**:
   - Knee データでの検証（異なる解剖学的部位）
   - 複数プロトコル対応

4. **論文化準備**:
   - 統計検定結果 (p-value, Cohen's d)
   - 周波数特性プロット (phase φ[k], amplitude α[k])
   - 学習曲線の可視化
   - 失敗例・成功例の定性比較

---

**改訂日**: 2026-03-28  
**参照元**: [研究内容説明資料.md](研究内容説明資料.md)

---

## 付録A: 旧版計画（参照のみ・非推奨）

## Phase 1: 安定性確保（1-2週間）

### 目標
**提案手法の訓練が数値的に安定していることを確認**

### 実施内容

#### 1.1 複素微分の単体テスト

```python
# ファイル: tests/test_complex_grad.py

import torch
import unittest

class TestComplexGradient(unittest.TestCase):
    """複素数テンソルの勾配計算を検証"""
    
    def test_ifft_gradient(self):
        """IFFT の複素勾配が正しく計算されるか"""
        x = torch.randn(10, 20, dtype=torch.complex64, requires_grad=True)
        
        # Forward
        y = torch.fft.ifft(x, dim=-1, norm='ortho')
        
        # Loss（複素数のL2ノルム）
        loss = (y.abs() ** 2).sum()
        
        # Backward
        loss.backward()
        
        # Assertions
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.dtype, torch.complex64)
        self.assertTrue(torch.all(torch.isfinite(x.grad)))
        
        print("✅ IFFT gradient: OK")
    
    def test_phase_correction_gradient(self):
        """位相補正の勾配が正しく計算されるか"""
        x = torch.randn(10, 20, dtype=torch.complex64, requires_grad=False)
        phase = torch.randn(20, requires_grad=True)
        
        # Forward
        correction = torch.exp(1j * phase).unsqueeze(0)
        y = x * correction
        
        # Loss
        loss = y.abs().sum()
        
        # Backward
        loss.backward()
        
        # Assertions
        self.assertIsNotNone(phase.grad)
        self.assertTrue(torch.all(torch.isfinite(phase.grad)))
        
        grad_norm = phase.grad.norm().item()
        self.assertLess(grad_norm, 100, "Gradient explosion")
        self.assertGreater(grad_norm, 1e-6, "Gradient vanishing")
        
        print(f"✅ Phase gradient norm: {grad_norm:.6f}")

if __name__ == '__main__':
    unittest.main()
```

**実行**:
```bash
cd c:\Users\s2520\fastMRI
python -m pytest tests/test_complex_grad.py -v
```

#### 1.2 ユニタリ性監視スクリプト

```python
# ファイル: scripts/monitor_unitarity.py

import torch
import torch.nn as nn
from pathlib import Path

class UnitarityMonitor:
    """モデルのユニタリ性を監視"""
    
    def __init__(self, model, log_dir='logs'):
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.history = []
    
    def check_batch(self, x):
        """バッチのユニタリ性を評価"""
        with torch.no_grad():
            y = self.model(x)
        
        # ノルム比を計算
        norm_x = torch.norm(x).item()
        norm_y = torch.norm(y).item()
        ratio = norm_y / (norm_x + 1e-8)
        
        return ratio
    
    def monitor_epoch(self, dataloader, epoch):
        """エポック単位で監視"""
        ratios = []
        
        for batch_idx, batch in enumerate(dataloader):
            x = batch['kspace']  # shape: [B, C, H, W]
            ratio = self.check_batch(x)
            ratios.append(ratio)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}: Energy ratio = {ratio:.4f}")
        
        mean_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        
        self.history.append({
            'epoch': epoch,
            'mean': mean_ratio,
            'min': min_ratio,
            'max': max_ratio
        })
        
        # ログ出力
        status = "✅ OK" if mean_ratio > 0.95 else "⚠️ WARNING"
        print(f"\nEpoch {epoch}: {status}")
        print(f"  Mean ratio: {mean_ratio:.4f}")
        print(f"  Min  ratio: {min_ratio:.4f}")
        print(f"  Max  ratio: {max_ratio:.4f}")
        
        return mean_ratio
    
    def plot_history(self):
        """監視結果をプロット"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        epochs = [h['epoch'] for h in self.history]
        means = [h['mean'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, means, 'o-', label='Mean energy ratio')
        plt.axhline(y=0.95, color='r', linestyle='--', label='Threshold (0.95)')
        plt.xlabel('Epoch')
        plt.ylabel('Energy Ratio (||y|| / ||x||)')
        plt.title('Unitarity Monitoring')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = self.log_dir / 'unitarity_history.png'
        plt.savefig(save_path, dpi=150)
        print(f"✅ Saved to {save_path}")

# 使用例
# monitor = UnitarityMonitor(model)
# for epoch in range(num_epochs):
#     ratio = monitor.monitor_epoch(train_loader, epoch)
#     if ratio < 0.90:
#         print("⚠️ Alert: Energy loss > 10%")
# monitor.plot_history()
```

#### 1.3 勾配ノルム監視

```python
# ファイル: utils/gradient_monitor.py

import torch
from collections import defaultdict

class GradientNormMonitor:
    """勾配ノルムの爆発・消失を監視"""
    
    def __init__(self, model, log_freq=10):
        self.model = model
        self.log_freq = log_freq
        self.step = 0
        self.history = defaultdict(list)
    
    def log_gradients(self):
        """現在の勾配ノルムを記録"""
        total_norm = 0
        param_norms = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                param_norms[name] = param_norm
                total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        # ログ記録
        self.history['total_norm'].append(total_norm)
        for name, norm in param_norms.items():
            self.history[name].append(norm)
        
        # 定期的に出力
        if self.step % self.log_freq == 0:
            print(f"\n[Step {self.step}] Gradient norms:")
            print(f"  Total norm: {total_norm:.6f}")
            
            # 警告
            if total_norm > 10:
                print(f"  ⚠️ WARNING: Gradient explosion (norm > 10)")
            elif total_norm < 1e-6:
                print(f"  ⚠️ WARNING: Gradient vanishing (norm < 1e-6)")
            else:
                print(f"  ✅ OK")
            
            # 各層の情報
            for name in sorted(param_norms.keys())[:3]:  # 最初の3層
                print(f"  {name}: {param_norms[name]:.6f}")
        
        self.step += 1
        return total_norm
    
    def check_health(self):
        """勾配の健全性をチェック"""
        if len(self.history['total_norm']) < 10:
            return "Cannot judge (< 10 steps)"
        
        recent_norms = self.history['total_norm'][-10:]
        mean_norm = sum(recent_norms) / len(recent_norms)
        max_norm = max(recent_norms)
        
        if max_norm > 100:
            return "❌ CRITICAL: Gradient explosion"
        elif max_norm > 10:
            return "⚠️ HIGH: Potential explosion"
        elif mean_norm < 1e-6:
            return "❌ CRITICAL: Gradient vanishing"
        else:
            return "✅ HEALTHY"
    
    def plot_history(self, save_path='logs/gradient_norm.png'):
        """勾配ノルム履歴をプロット"""
        import matplotlib.pyplot as plt
        
        norms = self.history['total_norm']
        
        plt.figure(figsize=(12, 6))
        plt.plot(norms, 'o-', alpha=0.7, label='Total gradient norm')
        plt.axhline(y=1.0, color='g', linestyle='-', alpha=0.3, label='Target (1.0)')
        plt.axhline(y=10.0, color='orange', linestyle='--', alpha=0.5, label='Warning (10.0)')
        plt.axhline(y=100.0, color='r', linestyle='--', alpha=0.5, label='Critical (100.0)')
        plt.yscale('log')
        plt.xlabel('Step')
        plt.ylabel('Gradient norm (log scale)')
        plt.title('Gradient Norm History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=150)
        print(f"✅ Saved to {save_path}")

# 使用例
# monitor = GradientNormMonitor(model)
# for step, batch in enumerate(dataloader):
#     loss = model(batch)
#     loss.backward()
#     monitor.log_gradients()
#     optimizer.step()
#     optimizer.zero_grad()
```

#### 1.4 TensorBoard への記録

```python
# ファイル: train_with_monitoring.py

import torch
from torch.utils.tensorboard import SummaryWriter
from utils.gradient_monitor import GradientNormMonitor

def train_epoch_with_monitoring(model, dataloader, optimizer, epoch, writer):
    """TensorBoard に記録しながら訓練"""
    
    grad_monitor = GradientNormMonitor(model)
    
    for step, batch in enumerate(dataloader):
        x = batch['kspace'].to(device)
        target = batch['image'].to(device)
        
        # Forward
        pred = model(x)
        loss = mse_loss(pred, target)
        
        # Backward
        loss.backward()
        grad_norm = grad_monitor.log_gradients()
        
        # 勾配クリップ
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        # ログ記録（TensorBoard）
        global_step = epoch * len(dataloader) + step
        writer.add_scalar('loss/train', loss.item(), global_step)
        writer.add_scalar('gradient/norm', grad_norm, global_step)
        
        if step % 50 == 0:
            print(f"Epoch {epoch}, Step {step}: Loss={loss:.6f}, GradNorm={grad_norm:.6f}")
    
    print(grad_monitor.check_health())

# 実行
writer = SummaryWriter(log_dir='runs/experiment_v1')
for epoch in range(num_epochs):
    train_epoch_with_monitoring(model, train_loader, optimizer, epoch, writer)
writer.close()

# TensorBoard で可視化
# tensorboard --logdir=runs
```

### 成功指標

| 指標 | 合格基準 | 確認方法 |
|------|--------|--------|
| **Gradient norm** | 0.1 - 10 | `GradientNormMonitor` |
| **Energy ratio** | ≥ 0.95 | `UnitarityMonitor` |
| **Convergence** | Loss が 100 epoch で平坦化 | TensorBoard プロット |
| **Numerical stability** | NaN/Inf なし | `torch.isfinite()` チェック |

---

## Phase 2: 性能検証（2-3週間）

### 目標
**提案手法が既存手法より優れていることを定量的に示す**

### 実施内容

#### 2.1 Ablation Study の実施

```python
# ファイル: scripts/run_ablation_study.py

import torch
from pathlib import Path
import json

class AblationExperiment:
    """アブレーション実験の実行管理"""
    
    def __init__(self, output_dir='results/ablation'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def run_baseline_fixed_ifft(self):
        """ベースライン: 固定IFFT + 軸別UNet"""
        from models.baseline_fixed_ifft import BaselineFixer
        
        model = BaselineFixed()
        metrics = self.train_and_evaluate(model, name='Fixed IFFT')
        self.results['fixed_ifft'] = metrics
        
        return metrics
    
    def run_learnable_ifft_only(self):
        """学習可能IFFT のみ（UNet なし）"""
        from models.learnable_ifft import LearnableIFFT1D
        
        model = LearnableIFFT1D(N=320, rank=8)
        metrics = self.train_and_evaluate(model, name='Learnable IFFT')
        self.results['learnable_ifft'] = metrics
        
        return metrics
    
    def run_full_model(self):
        """提案手法: 学習可能IFFT + 軸別Dense UNet"""
        from models.stripes_separable_ifft import StripeSeparableIFFT
        
        model = StripeSeparableIFFT()
        metrics = self.train_and_evaluate(model, name='Proposed (Full Model)')
        self.results['full_model'] = metrics
        
        return metrics
    
    def train_and_evaluate(self, model, name, num_epochs=100):
        """共通の訓練・評価ルーチン"""
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}\n")
        
        # 訓練
        trainer = Trainer(model)
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs
        )
        
        # 評価
        metrics = trainer.evaluate(test_loader)
        
        # 結果を保存
        metrics.update({
            'name': name,
            'history': history
        })
        
        return metrics
    
    def summarize(self):
        """結果を表にまとめる"""
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80 + "\n")
        
        print(f"{'Method':<25} {'PSNR':<10} {'SSIM':<10} {'NMSE':<10} {'Time (ms)':<12}")
        print("-" * 80)
        
        for method, metrics in self.results.items():
            print(f"{metrics['name']:<25} "
                  f"{metrics['psnr']:<10.2f} "
                  f"{metrics['ssim']:<10.4f} "
                  f"{metrics['nmse']:<10.6f} "
                  f"{metrics['inference_time_ms']:<12.2f}")
        
        # JSON に保存
        save_path = self.output_dir / 'ablation_results.json'
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✅ Saved to {save_path}")
        
        # 勝者を表示
        psnr_winner = max(self.results.items(), key=lambda x: x[1]['psnr'])
        print(f"\n🏆 Best PSNR: {psnr_winner[0]} ({psnr_winner[1]['psnr']:.2f})")

# 実行
ablation = AblationExperiment()
ablation.run_baseline_fixed_ifft()
ablation.run_learnable_ifft_only()
ablation.run_full_model()
ablation.summarize()
```

#### 2.2 マルチスケール評価

```python
# ファイル: scripts/multiscale_evaluation.py

import matplotlib.pyplot as plt
import numpy as np

class MultiscaleEvaluator:
    """異なる加速率での性能を評価"""
    
    def evaluate_across_rates(self, acceleration_rates=[4, 8, 12, 16, 20]):
        """複数の加速率で評価"""
        
        results = {
            'acceleration_rate': [],
            'psnr': [],
            'ssim': [],
            'nmse': [],
            'inference_time': []
        }
        
        for rate in acceleration_rates:
            print(f"\nEvaluating acceleration rate: {rate}x")
            
            # 加速率に応じたマスクを生成
            mask = self._create_mask(rate)
            
            # 評価
            metrics = self._evaluate_with_mask(model, test_loader, mask)
            
            results['acceleration_rate'].append(rate)
            results['psnr'].append(metrics['psnr'])
            results['ssim'].append(metrics['ssim'])
            results['nmse'].append(metrics['nmse'])
            results['inference_time'].append(metrics['time_ms'])
        
        self._plot_results(results)
        return results
    
    def _create_mask(self, acceleration_rate):
        """加速率に応じたサンプリングマスクを生成"""
        height = 320
        center_fraction = 0.08
        
        center_lines = int(height * center_fraction)
        total_lines = height // acceleration_rate
        
        mask = torch.zeros(height)
        
        # 中心帯を全取得
        center_start = (height - center_lines) // 2
        mask[center_start:center_start + center_lines] = 1
        
        # 外周をランダムに取得
        remaining_lines = total_lines - center_lines
        outer_indices = list(range(0, center_start)) + list(range(center_start + center_lines, height))
        selected_outer = np.random.choice(outer_indices, size=remaining_lines, replace=False)
        mask[selected_outer] = 1
        
        return mask.bool()
    
    def _plot_results(self, results):
        """結果をプロット"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        x = results['acceleration_rate']
        
        # PSNR vs Acceleration rate
        axes[0].plot(x, results['psnr'], 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Acceleration rate')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('PSNR vs Acceleration')
        axes[0].grid(True, alpha=0.3)
        
        # SSIM vs Acceleration rate
        axes[1].plot(x, results['ssim'], 's-', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Acceleration rate')
        axes[1].set_ylabel('SSIM')
        axes[1].set_title('SSIM vs Acceleration')
        axes[1].grid(True, alpha=0.3)
        
        # Inference time vs Acceleration rate
        axes[2].plot(x, results['inference_time'], '^-', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel('Acceleration rate')
        axes[2].set_ylabel('Time (ms)')
        axes[2].set_title('Inference Time vs Acceleration')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/multiscale_evaluation.png', dpi=150)
        print("✅ Saved to results/multiscale_evaluation.png")
```

### 成功指標

| 指標 | 合格基準 |
|------|--------|
| **PSNR 向上** | Fixed IFFT より ≥ 1-2 dB |
| **SSIM 向上** | Fixed IFFT より ≥ 0.02 |
| **推論時間** | 従来手法の 60-80% |
| **メモリ使用量** | 従来手法の 70-90% |

---

## Phase 3: 理論補強（1-2週間）

### 目標
**共役対称性・ユニタリ性を理論的に保証する実装を追加**

### 実施内容

#### 3.1 共役対称性の強制実装

```python
# ファイル: models/conjugate_symmetric_ifft.py

import torch
import torch.nn as nn

class ConjugateSymmetricLearnableIFFT1D(nn.Module):
    """
    共役対称性を明示的に保証する学習可能IFFT
    """
    
    def __init__(self, N, rank=8, axis=-1):
        super().__init__()
        self.N = N
        self.rank = rank
        self.axis = axis
        
        # 共役対称な位相パラメータを学習
        # φ_sym[-k] = -φ_sym[k] を満たすように設計
        self.phase_param = nn.Parameter(
            torch.randn(N // 2, rank) * 0.01
        )
    
    def _get_conjugate_symmetric_phase(self):
        """
        学習可能なパラメータから共役対称な位相を生成
        """
        N = self.N
        half_N = N // 2
        
        # パラメータを低ランク分解で復元
        phase_half = torch.matmul(
            self.phase_param,
            torch.randn(self.rank, device=self.phase_param.device)
        ).sum(dim=-1)  # shape: [N//2]
        
        # 共役対称性を強制: φ[-k] = -φ[k]
        phase = torch.zeros(N, dtype=self.phase_param.dtype, device=self.phase_param.device)
        
        for k in range(half_N):
            phase[k] = phase_half[k]
            phase[N - k - 1] = -phase_half[k]  # 共役対称
        
        return phase
    
    def forward(self, x):
        """
        入力: x ∈ ℂ^(..., N)
        出力: y ∈ ℝ^(..., N)（実数）
        
        処理:
        1. k-space で位相補正を適用
        2. IFFT で画像空間に変換
        3. 出力は自動的に実数（共役対称性により保証）
        """
        # 位相補正パラメータを生成
        phase = self._get_conjugate_symmetric_phase()
        
        # k-space で位相補正を適用（この段階で共役対称性を保証）
        x_corrected = x * torch.exp(1j * phase)
        
        # IFFT（出力は実数）
        y = torch.fft.ifft(x_corrected, dim=self.axis, norm='ortho')
        
        # 実部を取得（虚部はゼロのはず）
        y_real = y.real
        
        # 検証: 虚部がゼロであることを確認（デバッグ用）
        imag_magnitude = y.imag.abs().max().item()
        if imag_magnitude > 1e-5:
            print(f"⚠️ Warning: Imaginary part = {imag_magnitude:.6f}")
        
        return y_real
    
    def verify_conjugate_symmetry(self, x):
        """
        共役対称性が保証されているか検証
        
        検証方法:
        - x の共役対称性を確認
        - 出力 y が実数か確認
        """
        phase = self._get_conjugate_symmetric_phase()
        
        # 位相の共役対称性を確認
        N = self.N
        for k in range(N // 2):
            phi_k = phase[k].item()
            phi_minus_k = phase[N - k - 1].item()
            assert abs(phi_k + phi_minus_k) < 1e-5, f"Phase not conjugate symmetric at k={k}"
        
        print("✅ Phase conjugate symmetry: OK")
        
        # 出力が実数か確認
        y = self.forward(x)
        assert y.dtype == torch.float32 or y.dtype == torch.float64, "Output is not real"
        print("✅ Output is real: OK")
```

#### 3.2 ユニタリ制約の実装

```python
# ファイル: models/unitary_constrained_ifft.py

class UnitaryConstrainedLearnableIFFT1D(nn.Module):
    """
    ユニタリ性を保証する学習可能IFFT
    
    戦略: 位相補正の振幅を1に正規化し、エネルギーを保存
    """
    
    def __init__(self, N, rank=8, axis=-1):
        super().__init__()
        self.N = N
        self.rank = rank
        self.axis = axis
        
        # 位相パラメータ
        self.phase_param = nn.Parameter(torch.randn(N, rank) * 0.01)
        
        # Spectral normalization を適用（勾配安定化）
        from torch.nn.utils import spectral_norm
        
        self.phase_linear = spectral_norm(
            nn.Linear(rank, N, bias=False), n_power_iterations=1
        )
    
    def forward(self, x):
        """ユニタリ性を保証しながら位相補正を適用"""
        
        # 位相を生成（スペクトルノルム正規化で安定化）
        phase = self.phase_linear(
            torch.eye(self.rank, device=self.phase_param.device)
        ).t()  # shape: [N]
        
        # 基準IFFT
        y_base = torch.fft.ifft(x, dim=self.axis, norm='ortho')
        
        # 位相補正（大きさは保持）
        correction = torch.exp(1j * phase)
        y = y_base * correction
        
        # エネルギーを調整（完全なユニタリ性を保証）
        norm_x = torch.norm(x)
        norm_y = torch.norm(y)
        
        if norm_y > 1e-8:
            y = y * (norm_x / norm_y)
        
        return y.real
```

#### 3.3 正則化項の追加

```python
# ファイル: losses/physics_informed_loss.py

class PhysicsInformedLoss(nn.Module):
    """物理制約を組み込んだ損失関数"""
    
    def __init__(self, lambda_energy=0.1, lambda_symmetric=0.1):
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_symmetric = lambda_symmetric
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, model):
        """
        損失 = MSE + エネルギー正則化 + 共役対称性正則化
        """
        # 基本的な再構成誤差
        recon_loss = self.mse(pred, target)
        
        # エネルギー保存の正則化
        # 期待: || y || ≈ || x ||（訓練データから計算）
        energy_loss = self._energy_regularization(model)
        
        # 共役対称性の正則化
        # 期待: 位相パラメータが共役対称 （オプション）
        symmetric_loss = self._symmetry_regularization(model)
        
        total_loss = (
            recon_loss 
            + self.lambda_energy * energy_loss
            + self.lambda_symmetric * symmetric_loss
        )
        
        return total_loss, {
            'recon': recon_loss.item(),
            'energy': energy_loss.item(),
            'symmetric': symmetric_loss.item()
        }
    
    def _energy_regularization(self, model):
        """エネルギー散逸を抑制する正則化"""
        # モデルの各層でエネルギーを監視
        total_energy_loss = 0
        
        for name, module in model.named_modules():
            if hasattr(module, 'phase_parameter'):
                # 位相パラメータのノルムを監視
                phase_norm = torch.norm(module.phase_parameter)
                
                # 位相の大きさが不自然に大きくならないようにペナルティ
                energy_loss = torch.relu(phase_norm - 10.0)
                total_energy_loss += energy_loss
        
        return total_energy_loss
    
    def _symmetry_regularization(self, model):
        """共役対称性を正則化"""
        # 実装は省略（オプション）
        return torch.tensor(0.0)
```

### 成功指標

| 指標 | 合格基準 |
|------|--------|
| **復元画像の虚部** | max(imag) < 1e-5 |
| **エネルギー損失** | < 1% （|| y || / || x || > 0.99） |
| **PSNR 向上** | Phase 2 比で +0.5-1.0 dB |

---

## Phase 4: 論文執筆（3-4週間）

### 目標
**理論、実験、結果を一貫性のある論文にまとめる**

### 実施内容

#### 4.1 論文構成案

```
1. Introduction
   - MRI 高速化の重要性
   - 既存手法の課題
   - 本研究の貢献

2. Related Work
   - 深層学習ベース MRI 再構成
   - Compressed Sensing
   - Learnable transforms

3. Method
   3.1 Complex matrix approach (失敗例)
   3.2 Learnable IFFT with phase correction (提案)
   3.3 Architecture details

4. Theory
   4.1 Mathematical foundations
   4.2 Compressed Sensing principles
   4.3 Conjugate symmetry & Unitarity

5. Experiments
   5.1 Ablation study
   5.2 Multiscale evaluation
   5.3 Comparison with baselines

6. Results
   6.1 Performance metrics
   6.2 Visualization

7. Discussion
   7.1 Why phase correction works?
   7.2 Limitations
   7.3 Future work

8. Conclusion
```

#### 4.2 図表リスト（目標10-15図）

```
Figure 1: MRI k-space undersampling and reconstruction
Figure 2: Architecture comparison (Complex matrix vs Phase correction)
Figure 3: Learnable IFFT with phase correction
Figure 4: Separable processing pipeline
Figure 5: Ablation study results (bar chart)
Figure 6: PSNR vs Acceleration rate
Figure 7: Intermediate representation visualization
Figure 8: Learned phase parameters
Figure 9: Energy conservation monitoring
Figure 10: Gradient norm history
Figure 11-15: Qualitative results (brain MRI examples)
```

---

## 全体スケジュール（推奨）

```
Week 1-2:   Phase 1 (Stability)
             - Unit tests
             - Gradient monitoring
             - TensorBoard setup

Week 2-4:   Phase 2 (Performance)
             - Ablation experiments
             - Multiscale evaluation
             - Baseline comparison

Week 4-5:   Phase 3 (Theory)
             - Conjugate symmetry
             - Unitarity constraints
             - Regularization

Week 5-8:   Phase 4 (Paper)
             - Writing
             - Visualization
             - Revision

Total: 8 weeks (約2ヶ月)
```

---

**最終更新**: 2026年1月18日
**ステータス**: 実装準備完了
