"""
Phase 2 最小比較（4条件）を自動実行するランナー。

比較条件:
1) fixed_ifft               : LearnableIFFT 凍結 + amp_eps=0
2) phase_only               : 位相のみ（amp_eps=0）
3) phase_amp_hermitian      : 位相+微小振幅 + Hermitian ON
4) phase_amp_no_hermitian   : 位相+微小振幅 + Hermitian OFF

各条件を複数 seed で実行し、best checkpoint から
val_psnr / val_ssim を収集して CSV に集約する。
"""

import os
import sys
import csv
import json
import argparse
import subprocess
from statistics import mean, stdev

import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "train_separable_ifft.py")


def parse_seeds(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def safe_mean(vals):
    return mean(vals) if vals else float("nan")


def safe_std(vals):
    return stdev(vals) if len(vals) >= 2 else 0.0


def build_base_cmd(args, seed: int, save_dir: str):
    cmd = [
        args.python_exe,
        TRAIN_SCRIPT,
        "--glob", args.glob,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--seed", str(seed),
        "--save-dir", save_dir,
        "--workers", str(args.workers),
        "--patience", str(args.patience),
        "--target-ky", str(args.target_ky),
        "--target-kx", str(args.target_kx),
        "--coil-compress", str(args.coil_compress),
        "--amp-eps", str(args.amp_eps),
        "--lambda-phys", str(args.lambda_phys),
        "--min-lr", str(args.min_lr),
        "--step-size", str(args.step_size),
        "--gamma", str(args.gamma),
        "--lr-schedule", args.lr_schedule,
        "--save-log",
    ]

    if args.val_glob:
        cmd.extend(["--val-glob", args.val_glob])
    else:
        cmd.extend(["--val-ratio", str(args.val_ratio)])

    if args.undersample:
        cmd.extend([
            "--undersample",
            "--ns", str(args.ns),
            "--center-frac", str(args.center_frac),
        ])

    if args.min_coils is not None:
        cmd.extend(["--min-coils", str(args.min_coils)])

    if args.save_exclusion_log:
        cmd.append("--save-exclusion-log")

    return cmd


def run_condition(args, condition_name: str, condition_flags: list, seeds: list):
    records = []

    for seed in seeds:
        save_dir = os.path.join(args.base_save_dir, condition_name, f"seed_{seed}")
        os.makedirs(save_dir, exist_ok=True)

        cmd = build_base_cmd(args, seed=seed, save_dir=save_dir) + condition_flags

        print("=" * 80)
        print(f"[RUN] condition={condition_name}, seed={seed}")
        print(" ".join(cmd))

        if not args.dry_run:
            subprocess.run(cmd, check=True)

        ckpt_path = os.path.join(save_dir, "separable_ifft_best.pt")
        if os.path.exists(ckpt_path) and not args.dry_run:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            rec = {
                "condition": condition_name,
                "seed": seed,
                "best_epoch": int(ckpt.get("epoch", -1)),
                "val_psnr": float(ckpt.get("val_psnr", float("nan"))),
                "val_ssim": float(ckpt.get("val_ssim", float("nan"))),
                "save_dir": save_dir,
            }
        else:
            rec = {
                "condition": condition_name,
                "seed": seed,
                "best_epoch": -1,
                "val_psnr": float("nan"),
                "val_ssim": float("nan"),
                "save_dir": save_dir,
            }

        records.append(rec)

    return records


def write_outputs(base_dir: str, all_records: list):
    os.makedirs(base_dir, exist_ok=True)

    detail_csv = os.path.join(base_dir, "phase2_detail.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["condition", "seed", "best_epoch", "val_psnr", "val_ssim", "save_dir"],
        )
        w.writeheader()
        for r in all_records:
            w.writerow(r)

    grouped = {}
    for r in all_records:
        grouped.setdefault(r["condition"], []).append(r)

    summary = []
    for cond, rows in grouped.items():
        psnrs = [x["val_psnr"] for x in rows if x["val_psnr"] == x["val_psnr"]]
        ssims = [x["val_ssim"] for x in rows if x["val_ssim"] == x["val_ssim"]]
        summary.append(
            {
                "condition": cond,
                "n": len(rows),
                "psnr_mean": safe_mean(psnrs),
                "psnr_std": safe_std(psnrs),
                "ssim_mean": safe_mean(ssims),
                "ssim_std": safe_std(ssims),
            }
        )

    summary_csv = os.path.join(base_dir, "phase2_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["condition", "n", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"],
        )
        w.writeheader()
        for r in summary:
            w.writerow(r)

    summary_md = os.path.join(base_dir, "phase2_summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# Phase2 Minimal Comparison Summary\n\n")
        f.write("| condition | n | PSNR mean±std | SSIM mean±std |\n")
        f.write("|---|---:|---:|---:|\n")
        for r in summary:
            f.write(
                f"| {r['condition']} | {r['n']} | "
                f"{r['psnr_mean']:.4f}±{r['psnr_std']:.4f} | "
                f"{r['ssim_mean']:.5f}±{r['ssim_std']:.5f} |\n"
            )

    meta_json = os.path.join(base_dir, "phase2_run_meta.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump({"detail_csv": detail_csv, "summary_csv": summary_csv, "summary_md": summary_md}, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Saved: {detail_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_md}")


def main():
    ap = argparse.ArgumentParser("Run phase2 minimal 4-condition comparison")
    ap.add_argument("--glob", type=str, required=True)
    ap.add_argument("--val-glob", type=str, default=None)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seeds", type=str, default="42,43,44")
    ap.add_argument("--base-save-dir", type=str, default=os.path.join(ROOT, "outputs", "phase2_min_compare"))
    ap.add_argument("--python-exe", type=str, default=sys.executable)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine", "step", "plateau"])
    ap.add_argument("--step-size", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--min-lr", type=float, default=1e-6)

    ap.add_argument("--target-ky", type=int, default=640)
    ap.add_argument("--target-kx", type=int, default=320)
    ap.add_argument("--coil-compress", type=int, default=8)
    ap.add_argument("--min-coils", type=int, default=None)
    ap.add_argument("--save-exclusion-log", action="store_true")

    ap.add_argument("--undersample", action="store_true")
    ap.add_argument("--ns", type=int, default=32)
    ap.add_argument("--center-frac", type=float, default=0.12)

    ap.add_argument("--amp-eps", type=float, default=0.1)
    ap.add_argument("--lambda-phys", type=float, default=0.01)

    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    os.makedirs(args.base_save_dir, exist_ok=True)

    conditions = {
        "fixed_ifft": ["--amp-eps", "0.0", "--freeze-ifft", "--use-hermitian", "--lambda-phys", "0.0"],
        "phase_only": ["--amp-eps", "0.0", "--use-hermitian", "--lambda-phys", str(args.lambda_phys)],
        "phase_amp_hermitian": ["--amp-eps", str(args.amp_eps), "--use-hermitian", "--lambda-phys", str(args.lambda_phys)],
        "phase_amp_no_hermitian": ["--amp-eps", str(args.amp_eps), "--no-hermitian", "--lambda-phys", str(args.lambda_phys)],
    }

    all_records = []
    for cond_name, cond_flags in conditions.items():
        all_records.extend(run_condition(args, cond_name, cond_flags, seeds))

    write_outputs(args.base_save_dir, all_records)


if __name__ == "__main__":
    main()
