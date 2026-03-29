import math, h5py, numpy as np, matplotlib.pyplot as plt

# ★あなたの .h5 に変更
FILENAME = r"C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5"
SLICE_IDX = 0  # 見たいスライス番号

def to_complex(a):
    if np.iscomplexobj(a): return a
    if a.ndim >= 1 and a.shape[-1] == 2: return a[...,0] + 1j*a[...,1]
    raise ValueError(f"Not complex nor (...,2) array. shape={a.shape}")

def ifft2c(k):
    k0 = np.fft.ifftshift(k, axes=(-2,-1))                 # 中心化k-spaceを原点へ
    img = np.fft.ifft2(k0, axes=(-2,-1), norm="ortho")     # 2D IFFT
    return np.fft.fftshift(img, axes=(-2,-1))              # 見やすく再中心化

def rss(imgs):  # imgs: [coils,H,W] 複素
    return np.sqrt(np.sum(np.abs(imgs)**2, axis=0))
def norm995(x):  # 表示用スケール
    return x / (np.percentile(x, 99.5) + 1e-8)

# ----- 1) データ読み出し -----
with h5py.File(FILENAME, "r") as f:
    ks = to_complex(f["kspace"][SLICE_IDX])   # [coils, Ky, Kx]
coils, Ky, Kx = ks.shape

# ----- 2) 全ky→RSS（基準画像） -----
imgs_all = np.stack([ifft2c(ks[c]) for c in range(coils)], axis=0)
img_rss = norm995(rss(imgs_all))

# ----- 3) 代表5本の ky を選ぶ -----
picks = [0, Ky//4, Ky//2, 3*Ky//4, Ky-1]  # 低→高周波
labels = ["low (0)", "low-mid (1/4)", "center (1/2)", "mid-high (3/4)", "high (end)"]

# ----- 4) 1本だけ残して“ストリップ”生成 -----
strips = []
for ky_idx in picks:
    ks_line = np.zeros_like(ks)
    ks_line[:, ky_idx, :] = ks[:, ky_idx, :]            # 指定1本のみ残す
    imgs_line = np.stack([ifft2c(ks_line[c]) for c in range(coils)], axis=0)
    strip = norm995(rss(imgs_line))
    strips.append(strip)


# ----- 5) 可視化 -----
cols = 3
rows = math.ceil((1 + len(strips)) / cols)
plt.figure(figsize=(4*cols, 4*rows))

ax = plt.subplot(rows, cols, 1)
ax.imshow(img_rss, cmap="gray"); ax.set_title(f"RSS (all ky) | slice {SLICE_IDX}"); ax.axis("off")

for i, (ky_idx, strip, lab) in enumerate(zip(picks, strips, labels), start=2):
    ax = plt.subplot(rows, cols, i)
    ax.imshow(strip, cmap="gray")
    ax.set_title(f"ky={ky_idx}  [{lab}]")
    ax.axis("off")

plt.tight_layout(); plt.show()
