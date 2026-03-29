# scripts/infer_and_visualize.py
"""
学習済みモデルを検証セットに対して実行し、
再構成画像 vs GT画像をスライスごとに可視化・保存。
"""
import os, sys, glob, argparse
from pathlib import Path
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mri_strip.fft_utils import to_complex, ifft2c, rss, norm995
from mri_strip.metrics import psnr as psnr_np, ssim as ssim_np
from models.stripes_separable_ifft import SeparableIFFTStripesNet

# ---- Dataset ----
class FastMRIFullKspaceDataset(Dataset):
    def __init__(self, files, ref_Ky=None, ref_Kx=None):
        self.files = files
        self.ref_Ky = ref_Ky
        self.ref_Kx = ref_Kx
        self.index = []
        self.coils = None
        min_coils = None

        for fp in self.files:
            with h5py.File(fp, "r") as f:
                nsl, ncoils, Ky, Kx = f["kspace"].shape
                if min_coils is None:
                    min_coils = ncoils
                    if self.ref_Ky is None:
                        self.ref_Ky = Ky
                    if self.ref_Kx is None:
                        self.ref_Kx = Kx
                else:
                    min_coils = min(min_coils, ncoils)
                for sl in range(nsl):
                    self.index.append((fp, sl))

        self.coils = min_coils
        print(f"[Dataset] {len(self.index)} slices, {self.coils} coils, ref=(Ky={self.ref_Ky}, Kx={self.ref_Kx})")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fp, sl = self.index[i]
        with h5py.File(fp, "r") as f:
            ks_full = to_complex(f["kspace"][sl])
        coils_orig, Ky, Kx = ks_full.shape
        if coils_orig > self.coils:
            ks = ks_full[:self.coils]
        elif coils_orig < self.coils:
            pad = np.zeros((self.coils - coils_orig, Ky, Kx), dtype=ks_full.dtype)
            ks = np.concatenate([ks_full, pad], axis=0)
        else:
            ks = ks_full

        imgs_all = np.stack([ifft2c(ks[c]) for c in range(self.coils)], axis=0)
        gt = norm995(rss(imgs_all)).astype(np.float32)

        # Pad/Crop k-space
        if self.ref_Ky is not None and self.ref_Kx is not None:
            C, Ky_curr, Kx_curr = ks.shape
            if Ky_curr > self.ref_Ky:
                start = (Ky_curr - self.ref_Ky) // 2
                ks = ks[:, start:start+self.ref_Ky, :]
            elif Ky_curr < self.ref_Ky:
                pad_top = (self.ref_Ky - Ky_curr) // 2
                ks_pad = np.zeros((C, self.ref_Ky, Kx_curr), dtype=ks.dtype)
                ks_pad[:, pad_top:pad_top+Ky_curr, :] = ks
                ks = ks_pad

            _, Ky2, Kx2 = ks.shape
            if Kx2 > self.ref_Kx:
                start = (Kx2 - self.ref_Kx) // 2
                ks = ks[:, :, start:start+self.ref_Kx]
            elif Kx2 < self.ref_Kx:
                pad_left = (self.ref_Kx - Kx2) // 2
                ks_pad = np.zeros((C, Ky2, self.ref_Kx), dtype=ks.dtype)
                ks_pad[:, :, pad_left:pad_left+Kx2] = ks
                ks = ks_pad

        K = torch.from_numpy(ks.astype(np.complex64))
        y = torch.from_numpy(gt.astype(np.float32))[None]

        fname = os.path.basename(fp)
        return K, y, fname, sl


def infer_and_save(model, loader, output_dir, device="cuda", max_samples=10):
    """
    モデルで推論し、再構成画像とGT、指標をファイル保存
    max_samples: 処理する最大サンプル数（デバッグ用）
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    results = []
    processed = 0

    with torch.no_grad():
        for batch_idx, (K, y, fname, sl) in enumerate(loader):
            if max_samples and processed >= max_samples:
                break
                
            K = K.to(device)
            y = y.to(device)

            pred = model(K)
            if pred.shape[-2:] != y.shape[-2:]:
                pred = F.interpolate(pred, size=y.shape[-2:], mode="bilinear", align_corners=False)

            p = pred.detach().cpu().numpy()
            g = y.detach().cpu().numpy()

            for b in range(p.shape[0]):
                if max_samples and processed >= max_samples:
                    break
                    
                pred_img = p[b, 0]
                gt_img = g[b, 0]
                
                # Sigmoid 出力 [0,1] を GT の値域にスケーリング
                # GT は norm995 で max > 1 の場合がある
                # スケーリング: pred_img_scaled = pred_img * max(gt_img)
                gt_max = gt_img.max()
                if gt_max > 0:
                    pred_img = pred_img * gt_max
                psnr_val = float(psnr_np(gt_img, pred_img))
                ssim_val = float(ssim_np(gt_img, pred_img))

                # バッチの各要素から正しくfname/slを抽出
                fname_b = fname if isinstance(fname, str) else fname[b] if hasattr(fname, '__getitem__') else fname
                sl_b = int(sl[b].item()) if torch.is_tensor(sl) else int(sl[b]) if hasattr(sl, '__getitem__') else int(sl)

                results.append({
                    "fname": fname_b,
                    "slice": sl_b,
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "pred": pred_img,
                    "gt": gt_img,
                })
                
                processed += 1

                if (batch_idx % 50) == 0:
                    print(f"[{batch_idx}] {fname_b} slice {sl_b} | PSNR {psnr_val:.2f} dB | SSIM {ssim_val:.4f}")

    # 可視化・保存
    num_to_visualize = min(10, len(results))  # デバッグ用に10枚のみ
    for i, res in enumerate(results[:num_to_visualize]):  # 最初の50スライスを可視化
        fname = res["fname"]
        sl = res["slice"]
        pred = res["pred"]
        gt = res["gt"]
        psnr_val = res["psnr"]
        ssim_val = res["ssim"]
        
        # デバッグ: 最初の画像の値を確認
        if i == 0:
            print(f"\n[Debug First Image]")
            print(f"  GT min/max/mean: {gt.min():.4f}/{gt.max():.4f}/{gt.mean():.4f}")
            print(f"  Pred min/max/mean: {pred.min():.4f}/{pred.max():.4f}/{pred.mean():.4f}")

        fig = plt.figure(figsize=(12, 4))
        gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 表示範囲を GT と Pred の実際の値域に合わせる
        vmin_gt = gt.min()
        vmax_gt = gt.max()
        vmin_pred = pred.min()
        vmax_pred = pred.max()

        ax0 = fig.add_subplot(gs[0])
        im0 = ax0.imshow(gt, cmap="gray", vmin=vmin_gt, vmax=vmax_gt)
        ax0.set_title(f"GT")
        ax0.axis("off")
        plt.colorbar(im0, ax=ax0, fraction=0.046)

        ax1 = fig.add_subplot(gs[1])
        im1 = ax1.imshow(pred, cmap="gray", vmin=vmin_gt, vmax=vmax_gt)  # GT と同じ範囲で比較
        ax1.set_title(f"Reconstructed\nPSNR {psnr_val:.2f} dB")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046)

        ax2 = fig.add_subplot(gs[2])
        err = np.abs(gt - pred)
        im2 = ax2.imshow(err, cmap="hot")
        ax2.set_title(f"Error\nSSIM {ssim_val:.4f}")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        fig.suptitle(f"{fname} | Slice {sl}", fontsize=12, y=0.98)

        save_path = os.path.join(output_dir, f"{i:04d}_{fname.replace('.h5', '')}_sl{sl:03d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved: {save_path}")

    # CSV にサマリー保存
    import csv
    csv_path = os.path.join(output_dir, "inference_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fname", "slice", "psnr", "ssim"])
        for res in results:
            w.writerow([res["fname"], res["slice"], res["psnr"], res["ssim"]])
    
    mean_psnr = np.mean([r["psnr"] for r in results])
    mean_ssim = np.mean([r["ssim"] for r in results])
    print(f"\n=== Summary ===")
    print(f"Processed {len(results)} slices")
    print(f"Mean PSNR: {mean_psnr:.2f} dB")
    print(f"Mean SSIM: {mean_ssim:.4f}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved {num_to_visualize} visualization images to {output_dir}")


def main():
    ap = argparse.ArgumentParser("Inference & Visualize")
    ap.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    ap.add_argument("--val-glob", type=str, required=True, help="Validation glob pattern")
    ap.add_argument("--output-dir", type=str, default=None, help="Output directory for images")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint), "inference_results")

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_args = ckpt.get("args", {})

    # Model setup
    ky_len = model_args.get("ky_len", 768)
    kx_len = model_args.get("kx_len", 396)
    in_ch = model_args.get("in_ch", 4)

    model = SeparableIFFTStripesNet(
        in_ch=in_ch,
        ky_len=ky_len,
        kx_len=kx_len,
        base_ch=32,
        growth=16,
        n_layers=4,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    # Dataset & Loader
    val_files = sorted(glob.glob(args.val_glob, recursive=True))
    print(f"Found {len(val_files)} validation files")

    val_ds = FastMRIFullKspaceDataset(val_files, ref_Ky=ky_len, ref_Kx=kx_len)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)

    # Inference & Visualization
    infer_and_save(model, val_loader, args.output_dir, device=device, max_samples=10)


if __name__ == "__main__":
    main()
