import h5py, numpy as np, matplotlib.pyplot as plt

# === fastMRI データパス ===
file_path = r"C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5"

def ifft2c(k):
    """centered IFFT"""
    x = np.fft.ifftshift(k, axes=(-2,-1))
    x = np.fft.ifft2(x, axes=(-2,-1))
    x = np.fft.fftshift(x, axes=(-2,-1))
    return x

def rss(img_coils):
    """Root-Sum-of-Squares"""
    return np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))

# === データ読み込み ===
with h5py.File(file_path, "r") as f:
    k = f["kspace"][0]  # 先頭スライス (coils, ky, kx)

# === 中心・周辺・全体のマスク作成 ===
ky, kx = k.shape[-2:]
cy, cx = ky // 2, kx // 2
r = ky // 8  # 中心の半径（例: 全体の1/4）

Y, X = np.ogrid[:ky, :kx]
mask_center = ((Y - cy)**2 + (X - cx)**2) < r**2          # 中心だけ
mask_outer = ~mask_center                                 # 外周だけ
mask_full = np.ones_like(mask_center, dtype=bool)         # 全体

# === 各マスクを適用してIFFT ===
imgs = {}
for name, mask in zip(["Low frequency", "High Frequency", "ALL"], [mask_center, mask_outer, mask_full]):
    k_masked = k * mask                                   # k-spaceマスク
    img_coils = ifft2c(k_masked)
    img_rss = rss(img_coils)
    imgs[name] = np.abs(img_rss) / (np.max(img_rss) + 1e-8)

# === 表示 ===
plt.figure(figsize=(12,4))
for i, (title, img) in enumerate(imgs.items()):
    plt.subplot(1,3,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(title, fontsize=13)
    plt.axis('off')

# plt.suptitle("k-space のどの部分を使うかによる画像の違い", fontsize=15)
# plt.tight_layout()
plt.show()
