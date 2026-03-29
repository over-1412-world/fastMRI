"""
Train vs Val の性能差の原因を特定するデバッグスクリプト
"""
import os, sys
import torch
import torch.nn as nn

# import path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.stripes_separable_ifft import LearnableIFFT1D, SeparableIFFTStripesNet

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*60)
print("Hypothesis Testing: Train vs Val Gap")
print("="*60)

# --- Hypothesis 1: 補正機能の活動状況 ---
print("\n[Hypothesis 1] LearnableIFFT の補正機能の活動")
print("-" * 60)

ifft = LearnableIFFT1D(N=320, dim=-1, rank=8).to(device)

# ダミー入力
x = torch.randn(2, 16, 640, 320, dtype=torch.complex64).to(device)

# Training mode
ifft.train()
y_train = ifft(x)
print(f"Training mode:")
print(f"  phase norm: {ifft.phase.norm().item():.6f}")
print(f"  phase range: [{ifft.phase.min().item():.6f}, {ifft.phase.max().item():.6f}]")
print(f"  scale: {ifft.scale.item():.6f}")

# Evaluation mode
ifft.eval()
y_eval = ifft(x)
print(f"\nEvaluation mode:")
print(f"  補正は常に適用される（phase に基づく）")

# 差分を確認
diff = (y_train - y_eval).abs().max().item()
print(f"\nMax difference between train and eval: {diff:.8f}")
if diff < 1e-6:
    print("✅ 訓練と評価で同じ（補正は train/eval に依存しない）")
else:
    print("❌ 訓練と評価で異なる")

# --- Hypothesis 2: 補正スケールの問題 ---
print("\n" + "="*60)
print("[Hypothesis 2] 位相補正のスケール")
print("-" * 60)

# 位相補正の推定大きさ
phase_correction = torch.exp(1j * ifft.scale * ifft.phase)
phase_mag = phase_correction.abs()

print(f"Phase scale (scale parameter): {ifft.scale.item():.6f}")
print(f"Phase range: [{ifft.phase.min().item():.6f}, {ifft.phase.max().item():.6f}]")
print(f"Effective phase range: [{(ifft.scale * ifft.phase).min().item():.6f}, {(ifft.scale * ifft.phase).max().item():.6f}] radians")

base_magnitude = y_eval.abs().max().item()
correction_magnitude = (phase_mag - 1.0).abs().max().item()

print(f"\nBase IFFT output magnitude: {base_magnitude:.6f}")
print(f"Phase correction magnitude: {correction_magnitude:.8f}")

if correction_magnitude < 1e-6:
    print("⚠️  位相補正が非常に小さい（ほぼ 1.0 = 補正なし）")
    print("  → phase がまだ初期値（全て0）の可能性")
else:
    print("✅ 位相補正が活動している")

# --- Hypothesis 3: 共役対称性の破壊 ---
print("\n" + "="*60)
print("[Hypothesis 3] 共役対称性の破壊確認")
print("-" * 60)

# k-space の共役対称性をチェック
x_test = torch.randn(1, 16, 64, 64, dtype=torch.complex64).to(device)

# 共役対称パターン（テスト用）
x_sym = x_test.clone()
x_sym = (x_sym + torch.flip(torch.conj(x_sym), dims=[-2, -1])) / 2

ifft_layer = LearnableIFFT1D(N=64, dim=-1, rank=8).to(device)
ifft_layer.eval()

y_sym = ifft_layer(x_sym)
y_imag_max = y_sym.imag.abs().max().item()

print(f"Input symmetry: Conjugate symmetric")
print(f"Output imaginary part max: {y_imag_max:.8f}")

if y_imag_max > 1e-4:
    print("❌ IFFT が共役対称性を破壊（虚部が大きい）")
else:
    print("✅ 共役対称性が保持されている")

# --- Hypothesis 4: BN のバッチサイズ依存性 ---
print("\n" + "="*60)
print("[Hypothesis 4] BatchNorm のバッチサイズ依存性")
print("-" * 60)

model = SeparableIFFTStripesNet(
    in_ch=16, ky_len=640, kx_len=320, 
    base_ch=32, growth=16, n_layers=4, ifft_rank=8
).to(device)

batch_size_small = 2
batch_size_large = 4  # 8ではなく4に減らす（メモリ対策）

x_small = torch.randn(batch_size_small, 16, 640, 320, dtype=torch.complex64).to(device)
x_large = torch.randn(batch_size_large, 16, 640, 320, dtype=torch.complex64).to(device)

model.train()
try:
    y_small = model(x_small)
    print(f"Batch size {batch_size_small}: ✅ Output shape {y_small.shape}")
    print(f"  Output range [{y_small.min():.4f}, {y_small.max():.4f}]")
except RuntimeError as e:
    print(f"Batch size {batch_size_small}: ❌ Error - {str(e)[:100]}")

try:
    y_large = model(x_large)
    print(f"Batch size {batch_size_large}: ✅ Output shape {y_large.shape}")
    print(f"  Output range [{y_large.min():.4f}, {y_large.max():.4f}]")
except RuntimeError as e:
    print(f"Batch size {batch_size_large}: ❌ Error - {str(e)[:100]}")

# --- 総括 ---
print("\n" + "="*60)
print("Summary")
print("="*60)
print("""
Train SSIM が低く Val SSIM が高い場合、以下の順に確認してください：

1️⃣  補正機能の活動（最優先）
   → 上記で補正が微小な場合、scale_factor を 0.01 → 0.1 に増やす
   
2️⃣  補正スケールの適切化
   → correction_ratio が 1% 未満なら、まず scale_factor を調整
   
3️⃣  alpha の学習
   → alpha.grad が流れているか確認（別スクリプト）
   
4️⃣  共役対称性の強制
   → 虚部が大きければ、入力に共役対称性制約を追加
   
5️⃣  バッチサイズの増加
   → 2 → 4 or 8 に増やして BN を安定化
""")
