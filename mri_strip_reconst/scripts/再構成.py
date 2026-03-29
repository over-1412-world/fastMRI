import os, h5py, numpy as np, matplotlib.pyplot as plt

file_path = r"C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5"
assert os.path.isfile(file_path)

def ifft2c(k, axes=(-2,-1)):
    x = np.fft.ifftshift(k, axes=axes)
    x = np.fft.ifft2(x, axes=axes)
    x = np.fft.fftshift(x, axes=axes)
    return x

def rss(img_coils, axis=0):
    return np.sqrt(np.sum(np.abs(img_coils)**2, axis=axis))

with h5py.File(file_path, "r") as f:
    # k: (coils, ky, kx)  ← fastMRIは [slices, coils, ky, kx]
    k = f["kspace"][0]                 # 先頭スライス（全コイル）
    img_coils = ifft2c(k)              # (coils, ny, nx)
    img_rss   = rss(img_coils, axis=0) # (ny, nx) 全コイル合成

img = np.abs(img_rss)
img = img / (img.max() + 1e-12)

plt.figure(figsize=(6,6))
plt.imshow(img, cmap="gray")
# plt.title("Full-coil Reconstruction (RSS)")
plt.axis("off"); plt.show()
