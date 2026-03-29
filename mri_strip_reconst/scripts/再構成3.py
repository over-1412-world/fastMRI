import numpy as np
import matplotlib.pyplot as plt
import h5py
from typing import List, Optional

# === fastMRI データパス ===
file_path = r"C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5"

# ========= ユーティリティ =========
def ifft2c(img: np.ndarray) -> np.ndarray:
    """中心化された2D逆FFT"""
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(img, axes=(-2, -1)), norm="ortho"),
        axes=(-2, -1)
    )

def rss_combine(img_coils: np.ndarray) -> np.ndarray:
    """Root-Sum-of-Squares (RSS) 合成"""
    return np.sqrt(np.sum(np.abs(img_coils)**2, axis=0))

def make_stripe_from_ky(kspace: np.ndarray, ky_index: int) -> np.ndarray:
    """Kyライン1本からストライプ画像を生成"""
    n_coils, Ny, Nx = kspace.shape
    ks = np.zeros_like(kspace, dtype=np.complex64)
    ks[:, ky_index, :] = kspace[:, ky_index, :]

    img_coils = ifft2c(ks)
    stripe = rss_combine(img_coils)

    # 0-1正規化
    mn, mx = np.percentile(stripe, [0.5, 99.5])
    stripe = np.clip((stripe - mn) / max(mx - mn, 1e-8), 0, 1)
    return stripe

def generate_stripes(kspace: np.ndarray, ky_indices: List[int]) -> List[np.ndarray]:
    """複数のKyラインからストライプを生成"""
    return [make_stripe_from_ky(kspace, ky) for ky in ky_indices]

def plot_stripes_2x4(
    stripes: List[np.ndarray],
    ky_indices: List[int],
    titles: Optional[List[str]] = None,
    big_title: str = "Step 2: Stripe Image (Ky Single → IFFT → 16-Coil RSS)"
) -> None:
    """2x4でストライプを表示"""
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
    axes = np.asarray(axes).reshape(rows, cols)
    fig.suptitle(big_title)

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(stripes):
            im = stripes[i]
            ax.imshow(im, cmap="gray", vmin=0.0, vmax=1.0)
            ky = ky_indices[i]
            ax.text(0.02, 0.06, f"Ky={ky}", transform=ax.transAxes,
                    fontsize=12, color="white",
                    bbox=dict(facecolor="black", alpha=0.5, pad=3))
            if titles and i < len(titles):
                ax.text(0.02, 0.90, titles[i], transform=ax.transAxes,
                        fontsize=11, color="white",
                        bbox=dict(facecolor="black", alpha=0.4, pad=3))
        else:
            ax.set_facecolor("black")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ========= fastMRIデータ読み込み =========
with h5py.File(file_path, "r") as f:
    kspace = f["kspace"][()]  # shape = (num_slices, num_coils, Ny, Nx)
    print("kspace shape:", kspace.shape)

# 任意のスライスを選択（例: 10枚目）
slice_idx = 10
kspace_slice = kspace[slice_idx]  # shape = (num_coils, Ny, Nx)

# ========= ストライプ生成 =========
n_coils, Ny, Nx = kspace_slice.shape
ky_center = Ny // 2
ky_indices = [ky_center, ky_center - 20, ky_center - 40, ky_center - 60,
              ky_center + 20, ky_center + 40, ky_center + 60, ky_center + 80]
titles = ["center"] + ["surrounding "] * 7

stripes = generate_stripes(kspace_slice, ky_indices)
plot_stripes_2x4(stripes, ky_indices, titles)
