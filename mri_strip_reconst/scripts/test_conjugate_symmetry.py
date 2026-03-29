"""
共役対称性の強制が虚部削減に効くか検証
"""
import torch
import numpy as np

def enforce_conjugate_symmetry_kspace(ks):
    """
    k-space の共役対称性を強制
    K(ky, kx) ≈ conj(K(-ky, -kx))
    """
    ks_flipped = np.flip(ks, axis=(1, 2))
    ks_conj_flipped = np.conj(ks_flipped)
    ks_sym = (ks + ks_conj_flipped) / 2.0
    return ks_sym

print("="*60)
print("共役対称性の効果を検証")
print("="*60)

# テスト用 k-space
ks = np.random.randn(16, 64, 64).astype(np.complex64) + \
     1j * np.random.randn(16, 64, 64).astype(np.complex64)

# IFFT（共役対称性なし）
x_before = np.fft.ifft2(ks, axes=(-2, -1))
imag_before = np.abs(x_before.imag).max()

# 共役対称性を強制
ks_sym = enforce_conjugate_symmetry_kspace(ks)

# IFFT（共役対称性あり）
x_after = np.fft.ifft2(ks_sym, axes=(-2, -1))
imag_after = np.abs(x_after.imag).max()

print(f"\n Before conjugate symmetry enforcement:")
print(f"  虚部 max: {imag_before:.6f}")

print(f"\n After conjugate symmetry enforcement:")
print(f"  虚部 max: {imag_after:.8f}")

print(f"\n 削減率: {(imag_before - imag_after) / imag_before * 100:.2f}%")

if imag_after < 1e-5:
    print("\n✅ 虚部がほぼ0に削減された（共役対称性が有効）")
else:
    print("\n⚠️ 虚部がまだ残っている")
