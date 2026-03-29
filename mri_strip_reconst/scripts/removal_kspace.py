import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# === ここを正しいファイルパスに ===
file_path = r"C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5"

if not os.path.isfile(file_path):
    raise FileNotFoundError(f"ファイルが存在しません: {file_path}")

print(f"読み込み中: {file_path}")

# === k-space データの読み込み ===
with h5py.File(file_path, 'r') as f:
    k = f['kspace'][10]  # 先頭スライスを取得
    print(f"k-space shape: {k.shape}")  # -> (coils, ky, kx)
    
    k_coil0 = k[10]
    k_mag = np.log1p(np.abs(np.fft.fftshift(k_coil0)))

plt.imshow(k_mag, cmap='gray')
# plt.title('k-space (coil 0)')
plt.axis('off')
plt.show()
