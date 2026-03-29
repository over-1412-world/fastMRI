import os, h5py, numpy as np, matplotlib.pyplot as plt

file_path = r"C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5"

def ifft2c(k):
    x = np.fft.ifftshift(k, axes=(-2,-1))
    x = np.fft.ifft2(x, axes=(-2,-1))
    x = np.fft.fftshift(x, axes=(-2,-1))
    return x

def rss(img_coils):
    return np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))

with h5py.File(file_path, "r") as f:
    k = f["kspace"][0]  # (coils, ky, kx) 先頭スライス

# 1) “見える”k-space表示（最も強いコイルで表示）
coil_idx = int(np.argmax([np.abs(k[i]).max() for i in range(k.shape[0])]))
k_coil = k[coil_idx]
kmag = np.log1p(np.abs(np.fft.fftshift(k_coil)))
# パーセンタイルでコントラスト強調
p1, p99 = np.percentile(kmag, [1, 99])
vmin, vmax = (p1, p99) if p99 > p1 else (kmag.min(), kmag.max())

plt.figure(figsize=(5,5))
plt.imshow(kmag, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
# plt.title(f"k-space (coil {coil_idx}, log|.|, pct stretch)")
plt.axis("off")
plt.show()

# 2) サンプリングマスクの確認（非ゼロ位置）
mask = np.any(np.abs(k) > 0, axis=0)  # (ky, kx)
plt.figure(figsize=(5,5))
plt.imshow(mask, cmap="gray", interpolation="nearest")
plt.title(f"Sampling mask  (rate={mask.mean():.3f})")
plt.axis("off")
plt.show()

# 3) RSS再構成（centered IFFT）
img_coils = ifft2c(k)
img = rss(img_coils)
img = img / (img.max() + 1e-8)

plt.figure(figsize=(5,5))
plt.imshow(img, cmap="gray")
# plt.title("Reconstruction (RSS, full measured k-space)")
plt.axis("off")
plt.show()
