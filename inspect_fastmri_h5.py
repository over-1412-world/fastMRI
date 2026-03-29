import h5py
import numpy as np
import matplotlib.pyplot as plt

# あなたの保存したパスに変更
filename = r"C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5"

with h5py.File(filename, "r") as f:
    print("Keys in file:", list(f.keys()))

    # k-space データを取得
    kspace = f["kspace"][()]
    print("k-space shape:", kspace.shape)  # (num_slices, num_coils, height, width)

    # 1スライス分のk-space → 画像に変換
    kspace_slice = kspace[0]  # 1スライス
    img = np.fft.ifft2(kspace_slice, axes=(-2, -1))  # IFFTで画像化
    img = np.abs(img).sum(axis=0)  # coil combine（簡易にsum-of-squares）

    plt.imshow(img, cmap="gray")
    plt.title("Brain MRI (reconstructed slice)")
    plt.axis("off")
    plt.show()
