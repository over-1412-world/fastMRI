import numpy as np
from .fft_utils import ifft2c, rss, norm995

def pick_lines_with_conjugate(ks, lines):
    """
    ks: [coils, Ky, Kx] complex
    lines: iterable[int]  # 0..Ky-1
    共役対称 (ky と -ky≡Ky-ky) を満たすように抽出
    """
    ks_sel =  np.zeros_like(ks)
    Ky = ks.shape[1]   #k-spaceの縦方向サイズ
    for ky in lines:
        ky = int(ky)
        sym = (-ky) % Ky   # 共役対称の位置(対称ライン)
        ks_sel[:, ky, :] = ks[:, ky, :]
        ks_sel[:, sym, :] = np.conj(ks[:, ky, :]) #np.conj:複素共役

    return ks_sel

def zf_ifft_rss_from_lines(ks, lines):
    """
    ks: [coils, Ky, Kx] complex
    lines: 取得する ky インデックス
    戻り値: RSS 画像（0-1正規化）
    
    共役対称を満たすように抽出し、ゼロ詰めIFFT+RSSでreconを得る

    ks_sel(k-space selected):sel は、MRI の k-space データ（周波数空間データ）から、選択されたラインだけを取り出したもの
    ks.shape[0]: コイル数

    #概要
    各コイルごとに ks_sel（部分的なk-space）を作り、
    それを逆フーリエ変換（ifft2c）して画像に変換し、
    最後に np.stack でまとめている
    """
    ks_sel = pick_lines_with_conjugate(ks, lines)
    imgs = np.stack([ifft2c(ks_sel[c]) for c in range(ks.shape[0])], axis=0)  # [coils, H, W]

    return norm995(rss(imgs))  # [H, W]