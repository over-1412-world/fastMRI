import argparse, h5py, numpy as np, matplotlib.pyplot as plt, math
from mri_strip.fft_utils import to_complex, ifft2c, rss, norm995
from mri_strip.sampling import mixed_center_peripheral
from mri_strip.strips import zf_ifft_rss_from_lines

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True)   # h5ファイル
parser.add_argument("--slice", type=int, default=0)   #何枚目のスライスか
parser.add_argument("--lines", type=int, default=4)   #可視化するkyラインの本数
args = parser.parse_args()

#データの読み込み
"""
coils … 受信コイルのチャンネル数
Ky … 位相エンコード方向（ライン方向）
Kx … 読み出し方向（サンプル方向）
"""
with h5py.File(args.file, "r") as f:
    #HDF5から指定スライスの k-space を読み出し、(実部,虚部) 表現を複素数配列に変換して、コイル×Ky×Kxの複素テンソルに整える
    ks = to_complex(f["kspace"][args.slice])  # [coils,Ky,Kx] complex
coils, Ky, Kx = ks.shape

# 基準: all ky の RSS
imgs_all = np.stack([ifft2c(ks[c]) for c in range(coils)], axis=0)
img_rss = norm995(rss(imgs_all))

# 可視化するライン群（例：中心＋周辺の混合）
picks = mixed_center_peripheral(Ky, args.lines)

# 図
cols = 3; rows = math.ceil((1+len(picks))/cols)
plt.figure(figsize=(4*cols, 4*rows))
ax = plt.subplot(rows, cols, 1); ax.imshow(img_rss, cmap="gray"); ax.set_title(f"RSS | slice {args.slice}"); ax.axis("off")

for i, ky in enumerate(picks, start=2):
    strip = zf_ifft_rss_from_lines(ks, [ky])
    ax = plt.subplot(rows, cols, i)
    ax.imshow(strip, cmap="gray"); ax.set_title(f"ky={int(ky)}"); ax.axis("off")

plt.tight_layout(); plt.show()


# import os, h5py, numpy as np
# import matplotlib.pyplot as plt
# import math

# filename = r"C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5"
# SLICE_IDX = 0   #見たいスライス
# SHOW_LINES = 0   #可視化するkyライン本数

# with h5py.File(filename, "r") as f:
#     print("keys:", list(f.keys()))
#     kspace = f["kspace"]
#     print("kspace shape", kspace.shape, "dtype", kspace.dtype)

# #ユーティリティ関数
# def to_complex_np(arr):
#     if np.iscomplexobj(arr):   #既に複素数
#         return arr
#     if arr.ndim >= 1 and arr.shape[-1] == 2:   #[..., 2] = (real, imag)
#         return arr[..., 0] + 1j * arr[..., 1]
#     return ValueError(f"Not complex nor (..., 2) array. shape={arr.shape}")

# def ifft2c_np(k):
#     #中心化k-space→ifft(正規化つき)
#     k0 = np.fft.ifftshift(k, axes=(-2, -1))
#     img = np.fft.ifft2(k0, axes=(-2, -1), norm="ortho")
#     return np.fft.fftshift(img, axes=(-2, -1))

# def rss(img_coils):
#     #img_coils:[coils, H, W]複素画像
#     return np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))

# def normalize_995(x):
#     return x / (np.percentile(x, 99.5) + 1e-8)

# #1スライスを画像化してRSS合成
# with h5py.File(filename, "r") as f:
#     ks = f["kspace"][SLICE_IDX]
#     ks = to_complex_np(ks)

#     #各コイルを画像化
#     imgs = np.stack([ifft2c_np(ks[c]) for c in range(ks.shape[0])], axis=0)
#     img_rss = normalize_995(rss(imgs))

#     plt.imshow(img_rss, cmap="gray"); plt.title(f"RSS| slice {SLICE_IDX}"); plt.axis("off"); plt.show()


# #kyライン1本のストリップを作って可視化
# coils, Ky, Kx = ks.shape
# center = Ky //2
# picks = [center-40, center-10, center+10, center+40]   #例4本
# picks = [max(0, min(Ky-1, p)) for p in picks]



# cols = 4
# rows = math.ceil((1 + len(picks)) / cols)  # 先頭にRSSを置く

# plt.figure(figsize=(4*cols, 4*rows))

# # (0) RSS
# ax = plt.subplot(rows, cols, 1)
# ax.imshow(img_rss, cmap="gray"); ax.set_title("RSS (all ky)"); ax.axis("off")

# # (1..N) ストリップ
# for i, ky_idx in enumerate(picks, start=2):
#     ks_line = np.zeros_like(ks)
#     ks_line[:, ky_idx, :] = ks[:, ky_idx, :]          # その1本だけ残す
#     imgs_line = np.stack([ifft2c_np(ks_line[c]) for c in range(coils)], axis=0)
#     strip = normalize_995(rss(imgs_line))

#     ax = plt.subplot(rows, cols, i)
#     ax.imshow(strip, cmap="gray"); ax.set_title(f"ky={ky_idx}"); ax.axis("off")

# plt.tight_layout(); plt.show()
