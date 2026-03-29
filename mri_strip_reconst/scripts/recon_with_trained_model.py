# scripts/recon_with_trained_model.py
import os, sys, argparse, h5py, numpy as np, torch
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from mri_strip.fft_utils import to_complex, ifft2c, rss, norm995
from mri_strip.sampling import variable_density_random, add_conjugates
from mri_strip.strips import zf_ifft_rss_from_lines
from mri_strip.metrics import psnr as psnr_np, ssim as ssim_np
from models.stripes_cnn import StripesReconstNet
import torch.nn.functional as F

def make_strips(ks, Ky, num_strips=4, center_frac=0.12, seed=0):
    rng = np.random.default_rng(seed)
    base = variable_density_random(Ky, num_strips, center_frac=center_frac, seed=seed)
    strips = []
    for ky in base:
        lines = add_conjugates(Ky, [int(ky)])
        s = zf_ifft_rss_from_lines(ks, lines)   # 0-1
        strips.append(s.astype(np.float32))
    return np.stack(strips, axis=0)  # [C,H,W], 値域0-1

def main():
    ap = argparse.ArgumentParser("Run inference with trained StripesReconstNet")
    ap.add_argument("--file", required=True, type=str)
    ap.add_argument("--slice", default=0, type=int)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--num-strips", type=int, default=4)
    ap.add_argument("--center-frac", type=float, default=0.12)
    ap.add_argument("--target-h", type=int, default=640)
    ap.add_argument("--target-w", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load data ---
    with h5py.File(args.file, "r") as f:
        ks = to_complex(f["kspace"][args.slice])  # [coils,Ky,Kx]
    coils, Ky, Kx = ks.shape

    imgs_all = np.stack([ifft2c(ks[c]) for c in range(coils)], axis=0)
    gt = norm995(rss(imgs_all)).astype(np.float32)      # [H,W] 0-1

    x = make_strips(ks, Ky, num_strips=args.num_strips, center_frac=args.center_frac, seed=args.seed)
    x_t = torch.from_numpy(x).unsqueeze(0)              # [1,C,H,W]
    y_t = torch.from_numpy(gt)[None,None]               # [1,1,H,W]

    # resize to target (学習時サイズに合わせる)
    x_t = F.interpolate(x_t, size=(args.target_h, args.target_w), mode="bilinear", align_corners=False)
    y_t = F.interpolate(y_t, size=(args.target_h, args.target_w), mode="bilinear", align_corners=False)

    # --- load model ---
    ckpt = torch.load(args.ckpt, map_location=device)
    in_ch = x_t.shape[1]
    model = StripesReconstNet(in_ch=in_ch, base_ch=32, growth=16).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        pred = model(x_t.to(device)).cpu().numpy()[0,0]  # [H,W]
        gt_r  = y_t.cpu().numpy()[0,0]
    ps = psnr_np(gt_r, pred); ss = ssim_np(gt_r, pred)

    # ====== Visualization (clean layout, no overlaps) ======
    outpng = args.out or os.path.join(
        ROOT, "outputs", "figures",
        f"infer_{os.path.splitext(os.path.basename(args.file))[0]}_sl{args.slice}.png"
    )
    os.makedirs(os.path.dirname(outpng), exist_ok=True)

    C = min(4, x.shape[0])                # stripの表示枚数（最大4）
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    gs  = fig.add_gridspec(2, 4)          # 上段: strips(最大4)、下段: GT / Pred / Diff / テキスト

    # ---- 上段: 入力ストリップ ----
    for i in range(C):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(x[i], cmap="gray")
        ax.set_title(f"strip #{i}", fontsize=10, pad=6)
        ax.axis("off")

    # ---- 下段: GT / Pred / Diff ----
    ax_gt = fig.add_subplot(gs[1, 0])
    ax_gt.imshow(gt_r, cmap="gray")
    ax_gt.set_title("GT (RSS)", fontsize=11, pad=6)
    ax_gt.axis("off")

    ax_pred = fig.add_subplot(gs[1, 1])
    ax_pred.imshow(pred, cmap="gray")
    ax_pred.set_title("Pred", fontsize=11, pad=6)
    ax_pred.axis("off")

    ax_diff = fig.add_subplot(gs[1, 2])
    diff = np.abs(gt_r - pred)
    ax_diff.imshow(diff / (diff.max() + 1e-8), cmap="magma")
    ax_diff.set_title("Abs Diff (norm)", fontsize=11, pad=6)
    ax_diff.axis("off")

    # 右下をメトリクス表示枠として使う（図と被らない）
    ax_txt = fig.add_subplot(gs[1, 3])
    ax_txt.axis("off")
    ax_txt.text(0.0, 0.9, "Metrics", fontsize=12, weight="bold")
    ax_txt.text(0.0, 0.65, f"PSNR : {ps:.2f} dB", fontsize=11)
    ax_txt.text(0.0, 0.45, f"SSIM : {ss:.4f}", fontsize=11)

    # 図全体のタイトル（上余白も確保）
    fig.suptitle("Stripes → CNN Reconstruction", fontsize=13, y=0.98)

    # 保存＆表示（余白を保ったまま）
    plt.savefig(outpng, dpi=170, bbox_inches="tight")
    plt.show()
    print(f"Saved: {outpng} | PSNR={ps:.2f} dB, SSIM={ss:.4f}")

if __name__ == "__main__":
    main()
