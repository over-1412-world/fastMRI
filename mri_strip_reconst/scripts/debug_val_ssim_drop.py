"""
Val SSIM 低下の原因を特定するデバッグスクリプト
"""
import os, sys
import torch
import torch.nn as nn
import numpy as np
import h5py

# import path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mri_strip.fft_utils import to_complex, ifft2c, rss, norm995
from models.stripes_separable_ifft import SeparableIFFTStripesNet, LearnableIFFT1D

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("Val SSIM 低下の原因特定")
print("="*70)

# ======================================================================
# (1) 位相パラメータの学習状況を確認
# ======================================================================
print("\n[1] LearnableIFFT1D の位相パラメータの学習状況")
print("-"*70)

ifft_kx = LearnableIFFT1D(N=320, dim=-1).to(device)
ifft_ky = LearnableIFFT1D(N=640, dim=-2).to(device)

print(f"kx phase (初期値):")
print(f"  norm: {ifft_kx.phase.norm().item():.8f}")
print(f"  min/max: [{ifft_kx.phase.min().item():.8f}, {ifft_kx.phase.max().item():.8f}]")

print(f"\nky phase (初期値):")
print(f"  norm: {ifft_ky.phase.norm().item():.8f}")
print(f"  min/max: [{ifft_ky.phase.min().item():.8f}, {ifft_ky.phase.max().item():.8f}]")

# ======================================================================
# (2) 学習済みモデルをロード
# ======================================================================
print("\n[2] 学習済みモデルの状態確認")
print("-"*70)

checkpoint_path = os.path.join(ROOT, "outputs", "separable_ifft", "separable_ifft_best.pt")

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SeparableIFFTStripesNet(
        in_ch=4, ky_len=640, kx_len=320,
        base_ch=32, growth=16, n_layers=4, ifft_rank=8
    ).to(device)
    model.load_state_dict(ckpt["model"])
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"Epoch: {ckpt['epoch']}, Val PSNR: {ckpt['val_psnr']:.2f}, Val SSIM: {ckpt['val_ssim']:.4f}")
    
    # 位相パラメータを確認
    print(f"\nkx phase (学習後):")
    print(f"  norm: {model.ifft_kx.phase.norm().item():.8f}")
    print(f"  min/max: [{model.ifft_kx.phase.min().item():.8f}, {model.ifft_kx.phase.max().item():.8f}]")
    print(f"  値のサンプル: {model.ifft_kx.phase[:5].detach().cpu().numpy()}")
    
    print(f"\nky phase (学習後):")
    print(f"  norm: {model.ifft_ky.phase.norm().item():.8f}")
    print(f"  min/max: [{model.ifft_ky.phase.min().item():.8f}, {model.ifft_ky.phase.max().item():.8f}]")
    print(f"  値のサンプル: {model.ifft_ky.phase[:5].detach().cpu().numpy()}")
    
else:
    print(f"❌ Checkpoint not found: {checkpoint_path}")
    print("スキップします")
    sys.exit(1)

# ======================================================================
# (3) Training mode vs Eval mode の出力差分
# ======================================================================
print("\n[3] Training mode vs Eval mode の出力差分")
print("-"*70)

# ダミーk-space入力
K_dummy = torch.randn(1, 4, 640, 320, dtype=torch.complex64).to(device)

# Training mode
model.train()
y_train = model(K_dummy)

# Eval mode
model.eval()
y_eval = model(K_dummy)

diff = (y_train - y_eval).abs().max().item()
print(f"Training と Eval の出力差分（max）: {diff:.8f}")

if diff > 1e-5:
    print(f"⚠️  差分が大きい → Dropout や BatchNorm が原因の可能性")
else:
    print(f"✅ 差分がほぼ0 → Training と Eval で同じ動作")

# ======================================================================
# (4) BatchNorm の running stats 確認
# ======================================================================
print("\n[4] BatchNorm の running stats")
print("-"*70)

# kx_net の最初の Dense Block の BN を確認
bn_layers = []
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        bn_layers.append((name, module))

if bn_layers:
    for name, bn in bn_layers[:3]:  # 最初の3つだけ表示
        print(f"\n{name}:")
        print(f"  running_mean: {bn.running_mean[:5]}")  # 最初の5つだけ
        print(f"  running_var: {bn.running_var[:5]}")
        print(f"  num_batches_tracked: {bn.num_batches_tracked.item()}")
else:
    print("No BatchNorm layers found")

# ======================================================================
# (5) 実際のデータで評価
# ======================================================================
print("\n[5] 実際のデータでの評価（Training vs Eval）")
print("-"*70)

# テストデータの読み込み
data_dir = "C:/Users/s2520/data/fastmri/brain_multicoil/test/multicoil_test"
h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

if h5_files:
    test_file = os.path.join(data_dir, h5_files[0])
    
    with h5py.File(test_file, 'r') as f:
        ks = to_complex(f["kspace"][0])  # [C, Ky, Kx]
    
    # Coil数を4に合わせる
    ks = ks[:4]
    
    # k-space テンソルに変換
    K_test = torch.from_numpy(ks[None].astype(np.complex64)).to(device)
    
    # Training mode
    model.train()
    img_train = model(K_test)
    
    # Eval mode
    model.eval()
    img_eval = model(K_test)
    
    print(f"Test data shape: {K_test.shape}")
    print(f"\nTraining mode output:")
    print(f"  min/max: [{img_train.min():.4f}, {img_train.max():.4f}]")
    print(f"  mean: {img_train.mean():.4f}, std: {img_train.std():.4f}")
    
    print(f"\nEval mode output:")
    print(f"  min/max: [{img_eval.min():.4f}, {img_eval.max():.4f}]")
    print(f"  mean: {img_eval.mean():.4f}, std: {img_eval.std():.4f}")
    
    diff_data = (img_train - img_eval).abs().max().item()
    print(f"\n実データでの出力差分: {diff_data:.8f}")
    
else:
    print("❌ テストデータが見つかりません")

# ======================================================================
# (6) 総括
# ======================================================================
print("\n" + "="*70)
print("原因の可能性（順位付け）")
print("="*70)

causes = [
    ("BatchNorm の running stats が不安定", 
     "BN は訓練時と評価時で異なる統計量を使用" if diff > 1e-5 else "なし"),
    
    ("Validation set が小さすぎる",
     "Debug mode で 1 file, 16 slices のみ → ノイズが大きい"),
    
    ("位相補正の過学習",
     "Train SSIM は改善しているが Val SSIM は低下"),
    
    ("Loss と SSIM のミスマッチ",
     "MSE Loss は最小化しているが、SSIM メトリクスは別"),
]

for i, (cause, note) in enumerate(causes, 1):
    print(f"\n{i}. {cause}")
    print(f"   → {note}")

print("\n" + "="*70)
print("推奨アクション")
print("="*70)
print("""
1️⃣ Validation set を増やす（--debug なしで全ファイルで実験）
2️⃣ BatchNorm を LayerNorm に置き換え（BN依存を排除）
3️⃣ SSIM を直接最適化する Loss に変更
4️⃣ 位相補正スケール（scale）を調整
""")
