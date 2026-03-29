# scripts/recon_multi_lines.py
import argparse, math, h5py, numpy as np, matplotlib.pyplot as plt, os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from mri_strip.fft_utils import to_complex, ifft2c, rss, norm995
from mri_strip.sampling import (
    central_lines, uniform_lines, mixed_center_peripheral,
    variable_density_random, accel_to_S, add_conjugates, mask_from_lines
)
from mri_strip.strips import zf_ifft_rss_from_lines
from mri_strip.metrics import psnr, ssim


def choose_lines(strategy, Ky, S, seed, center_frac):
    if strategy == "central":
        return central_lines(Ky, S)
    if strategy == "uniform":
        return uniform_lines(Ky, S)
    if strategy == "mixed":
        return mixed_center_peripheral(Ky, S, frac_center=0.5)
    if strategy == "vdrand":
        return variable_density_random(Ky, S, center_frac=center_frac, seed=seed)
    raise ValueError("unknown strategy")

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, type=str)
parser.add_argument("--slice", default=0, type=int)
parser.add_argument("--strategy", default="vdrand",
                    choices=["central","uniform","mixed","vdrand"])
parser.add_argument("--S", type=int, default=None, help="取得するky本数（R指定が無ければ必須）")
parser.add_argument("--R", type=float, default=None, help="加速度（例: 4 → およそ4倍短縮）")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--center-frac", type=float, default=0.12)
args = parser.parse_args()

# 1) データ読み出し
with h5py.File(args.file, "r") as f:
    ks = to_complex(f["kspace"][args.slice])  # [coils,Ky,Kx]
coils, Ky, Kx = ks.shape

# 2) 全kyのRSS（基準）
imgs_all = np.stack([ifft2c(ks[c]) for c in range(coils)], axis=0)
img_rss = norm995(rss(imgs_all))

# 3) 取得本数 S を決定（R→Sへ換算）
S = args.S if args.S is not None else accel_to_S(Ky, args.R, min_center=8)
S = int(np.clip(S, 2, Ky))  # 少なくとも2（共役ペアが作りやすい）

# 4) ライン選択 → 共役拡張
base_lines = choose_lines(args.strategy, Ky, S, args.seed, args.center_frac)
lines = add_conjugates(Ky, base_lines)
mask = mask_from_lines(Ky, lines)

# 5) ZF 再構成（共役込み）
img_zf = zf_ifft_rss_from_lines(ks, lines)

# 6) 可視化：基準/マスク/再構成
plt.figure(figsize=(12, 6))

ax = plt.subplot(1,3,1)
ax.imshow(img_rss, cmap="gray"); ax.set_title(f"RSS (all ky) | slice {args.slice}"); ax.axis("off")

ax = plt.subplot(1,3,2)
ax.imshow(mask[np.newaxis, :], aspect="auto", cmap="gray")  # マスクを帯で表示
ax.set_title(f"Mask | {args.strategy} | S={S} -> paired={len(lines)}"); ax.set_xlabel("ky"); ax.set_yticks([])

ax = plt.subplot(1,3,3)
ax.imshow(img_zf, cmap="gray"); ax.set_title("Zero-filled IFFT (RSS)"); ax.axis("off")

plt.tight_layout(); plt.show()

#print(f"Ky={Ky}, chosen S={S}, paired lines={len(lines)}")

# 7) PSNR / SSIM の計算と出力
gt_rss = img_rss      # 基準（全ky）
pred_rss = img_zf     # 再構成（ゼロ詰め）
val_psnr = psnr(gt_rss, pred_rss)
val_ssim = ssim(gt_rss, pred_rss)

print(f"\n[Summary]")
print(f"Ky={Ky}, chosen S={S}, paired lines={len(lines)}")
print(f"PSNR={val_psnr:.2f} dB, SSIM={val_ssim:.4f}")