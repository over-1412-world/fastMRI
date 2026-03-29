"""
1コイルだけで再構成する最小スクリプト
fastMRIのk-spaceは中心化(DCが中央)なので、ifft2の前にifftshiftを入れる
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

#file
filename=r"C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5"

#見たいスライス/コイル
slice_idx = 0   #何枚面のスライスを使うか
coil_idx = 0   #どのコイルのデータを使うか

with h5py.File(filename, "r") as f:
    # #ルート直下のキーを見る
    # print("top-level keys:", list(f.keys()))

    ks = f["kspace"][()]
    print("kspace shape:", ks.shape, "dtype", ks.dtype)

    #1)1コイルぶんのk-spaceを取り出す
    k = ks[slice_idx, coil_idx]

    print("k.shape:", k.shape)   #配列の数
    print("k.ndim", k.ndim)   #配列の次元
    print("k.dtype", k.dtype)   #データ型

    #(実部/虚部の2チャンネル形式)→複素数へ
    if k.ndim == 3 and k.shape[-1] == 3:
        k = k[..., 0] + 1j * k[..., 1]

    #2)中心化k-space→画像
    #DCが中央にある前提なので、ifft2前にifftshiftする
    """
    k-space（周波数空間）のデータは、DC成分（直流成分＝低周波数）が中央にある場合と、左上にある場合がある。
    np.fft.ifft2 は 低周波が左上にある配置を想定して計算するため、
    「中央配置 → 左上配置」に直す」**必要がある。
    そのために ifftshift を使う
    """
    k0 = np.fft.ifftshift(k, axes=(-2, -1))
    img = np.fft.ifft2(k0, norm="ortho")   #逆フーリエ変換で画像化 / norm="ortho" は直交規格化で、前後でスケールが揃うようにするオプション
    #表示用に再中心化
    img = np.fft.fftshift(img, axes=(-2, -1))

    #3)振幅画像＆正規化
    mag = np.abs(img)
    mag = mag / (np.percentile(mag, 99.5) + 1e-8)

    #4)表示
    plt.imshow(mag, cmap="gray")
    plt.title(f"Brain | slice {slice_idx}, coil {coil_idx}")
    plt.axis("off")
    plt.show()
   


    # kl = len()

# #再帰的に全部表示する関数
# def print_h5_structure(name, obj):
#     print(name, type(obj), getattr(obj, "shape", ""))

# with h5py.File(filename, "r") as f:
#     f.visititems(print_h5_structure)
