"""
PSNRの評価を複数ファイル・複数条件でまとめて実行するスクリプト
（例: 複数の加速度、複数のライン選択戦  略での比較）
出力: CSVファイル（metrics_zf_sweep.csv）、図（zf_sweep.png）       

実行例：python -m scripts.eval_zf_sweep --file "C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5" --slices 0,1 --mode S --values "20,40,80,160,320" --strategies "vdrand,central,mixed"
"""
# scripts/eval_zf_sweep.py
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

def list_files(file, glob_pattern):
    files = []
    if file:
        files = [file]
    if glob_pattern:
        files += sorted(glob.glob(glob_pattern))
    if not files:
        raise SystemExit("No input files. Use --file or --glob.")
    return files

def parse_int_list(arg):
    if arg is None: return None
    return [int(x) for x in str(arg).split(",") if x.strip()!=""]

def main():
    ap = argparse.ArgumentParser(description="Undersampled ZF-RSS evaluation sweep (PSNR/SSIM).")
    ap.add_argument("--file", type=str, help="single h5 file path")
    ap.add_argument("--glob", type=str, help='glob like "C:/.../multicoil_test/*.h5"')
    ap.add_argument("--strategies", type=str, default="vdrand,central,mixed,uniform",
                    help="comma separated: vdrand,central,mixed,uniform")
    ap.add_argument("--mode", choices=["S","R"], default="S",
                    help="Sweep by number of ky lines (S) or acceleration (R)")
    ap.add_argument("--values", type=str, required=True,
                    help='Comma list. Example: --mode S --values "20,40,80,160"  or  --mode R --values "4,6,8"')
    ap.add_argument("--slices", type=str, default="0",
                    help='Comma list of slice indices, e.g. "0,1,2"')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--center-frac", type=float, default=0.12)
    ap.add_argument("--outcsv", type=str, default=os.path.join(ROOT, "outputs", "metrics_zf_sweep.csv"))
    ap.add_argument("--outfig", type=str, default=os.path.join(ROOT, "outputs", "figures", "zf_sweep.png"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    os.makedirs(os.path.dirname(args.outfig), exist_ok=True)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    values = [float(v) for v in args.values.split(",") if v.strip()]
    slice_list = parse_int_list(args.slices)

    files = list_files(args.file, args.glob)

    # CSV header
    write_header = not os.path.exists(args.outcsv)
    with open(args.outcsv, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        if write_header:
            writer.writerow(["file", "slice", "strategy", "mode", "value", "Ky",
                             "S_base", "S_paired", "PSNR", "SSIM"])

        # aggregate for plotting
        plot_data = {}  # key: (strategy) -> {value->[scores...]}

        for fp in files:
            # Probe file once to know Ky and #slices
            with h5py.File(fp, "r") as f:
                kspace = f["kspace"]
                num_slices = kspace.shape[0]
                Ky = kspace.shape[-2]

            for sl in slice_list:
                if sl < 0 or sl >= num_slices:
                    print(f"Skip slice {sl} (out of range) in {os.path.basename(fp)}")
                    continue

                # Load slice k-space as complex
                with h5py.File(fp, "r") as f:
                    ks = to_complex(f["kspace"][sl])  # [coils,Ky,Kx]
                coils, Ky, Kx = ks.shape

                # Ground-truth (pseudo): full ky RSS
                imgs_all = np.stack([ifft2c(ks[c]) for c in range(coils)], axis=0)
                gt_rss = norm995(rss(imgs_all))

                for strat in strategies:
                    for v in values:
                        if args.mode == "S":
                            S = int(v)
                        else:  # mode == "R"
                            S = accel_to_S(Ky, R=v, min_center=8)
                        S = int(np.clip(S, 2, Ky))

                        base_lines = choose_lines(strat, Ky, S, args.seed, args.center_frac)
                        lines = add_conjugates(Ky, base_lines)

                        pred_rss = zf_ifft_rss_from_lines(ks, lines)

                        val_psnr = psnr(gt_rss, pred_rss)
                        val_ssim = ssim(gt_rss, pred_rss)

                        writer.writerow([os.path.basename(fp), sl, strat, args.mode, v, Ky,
                                         int(S), int(len(lines)), float(val_psnr), float(val_ssim)])

                        plot_data.setdefault(strat, {}).setdefault(v, []).append(float(val_psnr))

        # --- Plot: value vs mean PSNR for each strategy ---
        plt.figure(figsize=(7,5))
        # sort x by numeric ordering
        xs_sorted = sorted(values)
        for strat in strategies:
            ys = []
            for v in xs_sorted:
                vals = plot_data.get(strat, {}).get(v, [])
                ys.append(np.mean(vals) if len(vals)>0 else np.nan)
            plt.plot(xs_sorted, ys, marker="o", label=strat)
        plt.xlabel(args.mode)
        plt.ylabel("PSNR (dB)")
        plt.title("Zero-filled RSS Reconstruction (sweep)")
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(args.outfig, dpi=160)
        plt.show()

        print(f"\nSaved CSV to: {args.outcsv}")
        print(f"Saved figure to: {args.outfig}")

if __name__ == "__main__":
    main()
