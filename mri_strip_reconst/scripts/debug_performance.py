"""
LearnableIFFT1D の性能とメモリ使用量をプロファイリング
"""
import torch
import torch.nn as nn
import time
import sys
import os

# import path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.stripes_separable_ifft import LearnableIFFT1D

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# テストケース
Ky_len = 640
Kx_len = 320
batch_size = 2
num_coils = 16

print("\n" + "="*60)
print("LearnableIFFT1D Performance Test")
print("="*60)

# --- Forward Pass Test ---
print("\n[Forward Pass]")
print(f"Input shape: [{batch_size}, {num_coils}, {Ky_len}, {Kx_len}]")

ifft_kx = LearnableIFFT1D(N=Kx_len, dim=-1, rank=8).to(device)
print(f"LearnableIFFT1D: phase shape={tuple(ifft_kx.phase.shape)}, alpha shape={tuple(ifft_kx.alpha.shape)}")
print(f"              scale={ifft_kx.scale.item():.6f}, amp_eps={ifft_kx.amp_eps:.3f}")

K = torch.randn(batch_size, num_coils, Ky_len, Kx_len, dtype=torch.complex64).to(device)

# メモリ使用量
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1e6  # MB

t0 = time.perf_counter()
x = ifft_kx(K)
t1 = time.perf_counter()

if torch.cuda.is_available():
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / 1e6
    mem_peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"Memory before: {mem_before:.1f} MB")
    print(f"Memory after: {mem_after:.1f} MB")
    print(f"Memory peak: {mem_peak:.1f} MB")

print(f"Forward pass time: {(t1 - t0)*1000:.2f} ms")

# --- Backward Pass Test ---
print("\n[Backward Pass]")
loss = x.abs().sum()

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated() / 1e6

t0 = time.perf_counter()
loss.backward()
t1 = time.perf_counter()

if torch.cuda.is_available():
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated() / 1e6
    mem_peak = torch.cuda.max_memory_allocated() / 1e6
    print(f"Memory before: {mem_before:.1f} MB")
    print(f"Memory after: {mem_after:.1f} MB")
    print(f"Memory peak: {mem_peak:.1f} MB")

print(f"Backward pass time: {(t1 - t0)*1000:.2f} ms")

# --- Gradient Check ---
print("\n[Gradient Check]")
has_grad = False
if ifft_kx.phase.grad is not None:
    grad_norm_phase = ifft_kx.phase.grad.norm().item()
    print(f"phase gradient norm: {grad_norm_phase:.6f}")
    has_grad = True

if ifft_kx.alpha.grad is not None:
    grad_norm_alpha = ifft_kx.alpha.grad.norm().item()
    print(f"alpha gradient norm: {grad_norm_alpha:.6f}")
    has_grad = True

if ifft_kx.scale.grad is not None:
    grad_scale = ifft_kx.scale.grad.item()
    print(f"scale gradient: {grad_scale:.6f}")
    has_grad = True

if has_grad:
    print(f"✅ Gradient is flowing correctly")
else:
    print(f"❌ No gradient! (all params are None)")

# --- Estimate Full Epoch Time ---
print("\n" + "="*60)
print("Time Estimation")
print("="*60)
forward_time = (t1 - t0)  # backward を forward の時間として使用
est_per_batch = forward_time * 2  # forward + backward (概算)
est_per_epoch_100_batches = est_per_batch * 100
est_per_epoch_500_batches = est_per_batch * 500

print(f"Forward+Backward per batch (est.): {est_per_batch*1000:.0f} ms")
print(f"Full epoch (100 batches, est.): {est_per_epoch_100_batches/60:.1f} min")
print(f"Full epoch (500 batches, est.): {est_per_epoch_500_batches/60:.1f} min")
print(f"\n⚠️ NOTE: これは概算です。実際の学習時間はデータローディングも含みます")
