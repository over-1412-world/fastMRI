import numpy as np

def to_complex(a):
    """[...,2]=(real,imag) を複素数に。既に complex ならそのまま返す。"""
    if np.iscomplexobj(a):
        return a
    if a.ndim >= 1 and a.shape[-1] == 2:
        return a[..., 0] + 1j * a[..., 1]
    raise ValueError(f"Not complex nor (...,2). shape={a.shape}")

def ifft2c(k):
    """中心化k-space -> 2D IFFT（正規化）-> 見やすく再中心化。"""
    k0 = np.fft.ifftshift(k, axes=(-2, -1))
    img = np.fft.ifft2(k0, axes=(-2, -1), norm="ortho")
    return np.fft.fftshift(img, axes=(-2, -1))

def rss(imgs):
    """imgs: [coils,H,W] complex -> RSS 画像"""
    return np.sqrt(np.sum(np.abs(imgs) ** 2, axis=0))

def norm995(x):
    """表示用の0-1正規化（99.5%でスケーリング）"""
    return x / (np.percentile(x, 99.5) + 1e-8)
