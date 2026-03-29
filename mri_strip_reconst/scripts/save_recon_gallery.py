#各Sごとの再構成画像を自動で並べて保存（PSNR/SSIM付き）
# scripts/save_recon_gallery.py
import argparse, os, sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from mri_strip.fft_utils import to_complex, ifft2c, rss, norm995
from mri_strip.sampling import (
    central_lines, uniform_lines, mixed_center_peripheral,
    variable_density_random, accel_to_S, add_conjugates
)
from mri_strip.strips import zf_ifft_rss_from_lines
from mri_strip.metrics import psnr, ssim


def choose_lines(strategy, Ky, S, seed, center_frac):
    if strategy == "central":  return central_lines(Ky, S)
    if strategy == "uniform":  return uniform_lines(Ky, S)
    if strategy == "mixed":    return mixed_center_peripheral(Ky, S, frac_center=0.5)
    if strategy == "vdrand":   return variable_density_random(Ky, S, center_frac=center_frac, seed=seed)
    raise ValueError(f"unknown strategy: {strategy}")

def main():
    ap = argparse.ArgumentParser(description="Save gallery of ZF RSS reconstructions with PSNR/SSIM.")
    ap.add_argument("--file", required=True, type=str)
    ap.add_argument("--slice", default=0, type=int)
    ap.add_argument("--strategies", type=str, default="vdrand,central,mixed")
    ap.add_argument("--mode", choices=["S","R"], default="S")
    ap.add_argument("--values", type=str, required=True, help='"20,40,80,160" or "4,6,8" if mode=R')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--center-frac", type=float, default=0.12)
    ap.add_argument("--outdir", type=str, default=os.path.join(ROOT, "outputs", "figures"))
    ap.add_argument("--save_each", action="store_true", help="also save each recon as single PNG")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    values = [float(v) for v in args.values.split(",") if v.strip()]

    # ---- Load k-space for the slice ----
    with h5py.File(args.file, "r") as f:
        ks = to_complex(f["kspace"][args.slice])  # [coils,Ky,Kx]
    coils, Ky, Kx = ks.shape

    # GT (full ky RSS)
    imgs_all = np.stack([ifft2c(ks[c]) for c in range(coils)], axis=0)
    gt = norm995(rss(imgs_all))

    # ---- Prepare canvas ----
    # 1 row for GT + (len(strategies) * len(values)) items
    ncols = len(values)
    nrows = 1 + len(strategies)
    fig = plt.figure(figsize=(3.2*ncols, 3.5*nrows))

    # Row 0: GT
    ax = plt.subplot(nrows, ncols, 1)
    ax.imshow(gt, cmap="gray"); ax.set_title(f"GT (all ky RSS)\nslice={args.slice}"); ax.axis("off")
    for j in range(2, ncols+1):
        ax = plt.subplot(nrows, ncols, j); ax.axis("off")  # 空白（GTは左端だけ表示）

    # Rows for each strategy
    for si, strat in enumerate(strategies):
        for vi, v in enumerate(values):
            if args.mode == "S":
                S = int(v)
            else:
                S = accel_to_S(Ky, R=v, min_center=8)
            S = int(np.clip(S, 2, Ky))

            base_lines = choose_lines(strat, Ky, S, args.seed, args.center_frac)
            lines = add_conjugates(Ky, base_lines)
            pred = zf_ifft_rss_from_lines(ks, lines)

            p = psnr(gt, pred); s = ssim(gt, pred)

            ax = plt.subplot(nrows, ncols, (si+1)*ncols + vi + 1)
            ax.imshow(pred, cmap="gray")
            ttl = f"{strat} | {args.mode}={int(v) if args.mode=='S' else v:g}\nPSNR {p:.2f} dB | SSIM {s:.4f}"
            ax.set_title(ttl, fontsize=9)
            ax.axis("off")

            if args.save_each:
                fname = f"recon_{strat}_{args.mode}{int(v) if args.mode=='S' else v:g}_slice{args.slice}.png"
                plt.imsave(os.path.join(args.outdir, fname), pred, cmap="gray")

    plt.tight_layout()
    outpng = os.path.join(
        args.outdir,
        f"gallery_{os.path.splitext(os.path.basename(args.file))[0]}_sl{args.slice}_{args.mode}.png"
    )
    plt.savefig(outpng, dpi=160)
    plt.show()
    print(f"✅ Saved gallery: {outpng}")
    if args.save_each:
        print(f"✅ Individual PNGs are also saved in: {args.outdir}")

if __name__ == "__main__":
    main()
