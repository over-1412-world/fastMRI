
# PSNRとSSIMの評価を複数ファイル・複数条件でまとめて実行するスクリプト
# （例: 複数の加速度、複数のライン選択戦略での比較）
# 出力: 図（zf_sweep_dual_S.png / zf_sweep_dual_R.png）

# 実行例：python -m scripts.eval_zf_sweep_dual --file "C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5" --slices 0,1 --mode S --values "20,40,80,160" --strategies "vdrand,central,mixed"

# scripts/eval_zf_sweep_dual.py
import argparse, os, sys, glob, csv
import numpy as np
import h5py
import matplotlib.pyplot as plt

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
    raise ValueError(f"unknown strategy: {strategy}")


def parse_int_list(arg):
    if arg is None:
        return None
    return [int(x) for x in str(arg).split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description="Undersampled ZF-RSS evaluation sweep (PSNR & SSIM)")
    ap.add_argument("--file", type=str, required=True)
    ap.add_argument("--slices", type=str, default="0")
    ap.add_argument("--strategies", type=str, default="vdrand,central,mixed")
    ap.add_argument("--mode", choices=["S", "R"], default="S")
    ap.add_argument("--values", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--center-frac", type=float, default=0.12)
    ap.add_argument("--outdir", type=str, default=os.path.join(ROOT, "outputs", "figures"))
    args = ap.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    values = [float(v) for v in args.values.split(",") if v.strip()]
    slice_list = parse_int_list(args.slices)

    os.makedirs(args.outdir, exist_ok=True)

    with h5py.File(args.file, "r") as f:
        ks0 = to_complex(f["kspace"][slice_list[0]])
        coils, Ky, Kx = ks0.shape

    plot_data_psnr = {s: [] for s in strategies}
    plot_data_ssim = {s: [] for s in strategies}

    for strat in strategies:
        for v in values:
            psnr_list, ssim_list = [], []
            for sl in slice_list:
                with h5py.File(args.file, "r") as f:
                    ks = to_complex(f["kspace"][sl])  # [coils,Ky,Kx]
                imgs_all = np.stack([ifft2c(ks[c]) for c in range(coils)], axis=0)
                gt_rss = norm995(rss(imgs_all))

                if args.mode == "S":
                    S = int(v)
                else:
                    S = accel_to_S(Ky, R=v, min_center=8)
                S = int(np.clip(S, 2, Ky))

                base_lines = choose_lines(strat, Ky, S, args.seed, args.center_frac)
                lines = add_conjugates(Ky, base_lines)

                pred_rss = zf_ifft_rss_from_lines(ks, lines)

                psnr_list.append(psnr(gt_rss, pred_rss))
                ssim_list.append(ssim(gt_rss, pred_rss))

            plot_data_psnr[strat].append(np.mean(psnr_list))
            plot_data_ssim[strat].append(np.mean(ssim_list))

    # --- Plotting ---
    xs_sorted = sorted(values)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax2 = ax1.twinx()
    for strat in strategies:
        ax1.plot(xs_sorted, plot_data_psnr[strat], "o-", label=f"{strat} (PSNR)")
        ax2.plot(xs_sorted, plot_data_ssim[strat], "--", label=f"{strat} (SSIM)")

    ax1.set_xlabel(args.mode)
    ax1.set_ylabel("PSNR (dB)")
    ax2.set_ylabel("SSIM")
    ax1.set_title("Zero-filled RSS Reconstruction Sweep")
    ax1.grid(True, linestyle="--", alpha=0.6)
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.95))
    plt.tight_layout()

    outfig = os.path.join(args.outdir, f"zf_sweep_dual_{args.mode}.png")
    plt.savefig(outfig, dpi=160)
    plt.show()

    print(f"✅ Saved figure: {outfig}")


if __name__ == "__main__":
    main()
