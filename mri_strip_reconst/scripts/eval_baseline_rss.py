import os, sys, glob, argparse
import numpy as np
import h5py
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mri_strip.fft_utils import to_complex, ifft2c, rss, norm995
from mri_strip.metrics import psnr as psnr_np, ssim as ssim_np
import torch.nn.functional as F

def expand_globs(g):
    parts = [p.strip() for p in g.split(";") if p.strip()]
    files = []
    for p in parts:
        files.extend(glob.glob(p))
    return sorted(files)

def main():
    ap = argparse.ArgumentParser("Evaluate baseline (ifft2c+RSS+norm) pipeline")
    ap.add_argument("--glob", type=str, required=True,
                    help='例: "C:/.../multicoil_test/*.h5"')
    ap.add_argument("--max-files", type=int, default=20,
                    help="評価に使う最大ファイル数")
    ap.add_argument("--max-slices", type=int, default=5,
                    help="各ファイルから評価する最大スライス数")
    ap.add_argument("--target-h", type=int, default=256)
    ap.add_argument("--target-w", type=int, default=128)
    args = ap.parse_args()

    files = expand_globs(args.glob)
    if not files:
        raise SystemExit(f"No files found for glob: {args.glob}")

    files = files[:args.max_files]
    print(f"Using {len(files)} files.")

    psnrs, ssims = [], []

    for fi, fp in enumerate(files):
        with h5py.File(fp, "r") as f:
            kspace = f["kspace"]  # [slices, coils, Ky, Kx]
            nsl, ncoils, Ky, Kx = kspace.shape

            max_sl = min(args.max_slices, nsl)
            for sl in range(max_sl):
                ks = to_complex(kspace[sl])  # [C,Ky,Kx], complex64
                coils, Ky, Kx = ks.shape

                # ---- GT と同じパイプライン ----
                imgs_all = np.stack([ifft2c(ks[c]) for c in range(coils)], axis=0)  # [C,H,W] complex
                gt = norm995(rss(imgs_all)).astype(np.float32)  # [H,W], 0–1

                gt_t = torch.from_numpy(gt)[None, None]  # [1,1,H,W]
                gt_resized = F.interpolate(
                    gt_t, size=(args.target_h, args.target_w),
                    mode="bilinear", align_corners=False
                )[0,0].numpy()  # [H,W]

                # baseline は「同じもの」をそのまま使う（=理論上は一致）
                recon = gt_resized.copy()

                psnr = psnr_np(gt_resized, recon)
                ssim = ssim_np(gt_resized, recon)
                psnrs.append(float(psnr))
                ssims.append(float(ssim))

        print(f"[{fi+1}/{len(files)}] {os.path.basename(fp)} done.")

    print("Baseline RSS (self-consistency) stats:")
    print(f"  PSNR: mean {np.mean(psnrs):.2f} dB, min {np.min(psnrs):.2f}, max {np.max(psnrs):.2f}")
    print(f"  SSIM: mean {np.mean(ssims):.4f}, min {np.min(ssims):.4f}, max {np.max(ssims):.4f}")

if __name__ == "__main__":
    main()
