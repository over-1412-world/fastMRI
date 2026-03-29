# mri_strip_reconst/scripts/visualize_val_recon_diff.py
import os, sys, glob, argparse
import numpy as np
import h5py
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---- import path 設定 ----
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mri_strip.fft_utils import to_complex, ifft2c, rss, norm995
from models.stripes_separable_ifft import SeparableIFFTStripesNet


def center_crop_kspace(ks, ref_Ky, ref_Kx):
    """
    ks: np.complex64 [C,Ky,Kx]
    ref_Ky, ref_Kx に中心クロップ
    """
    C, Ky, Kx = ks.shape
    if (Ky, Kx) == (ref_Ky, ref_Kx):
        return ks

    top = max((Ky - ref_Ky) // 2, 0)
    left = max((Kx - ref_Kx) // 2, 0)
    return ks[:, top:top + ref_Ky, left:left + ref_Kx]


def make_triplet_figure(gt, recon, diff, out_path):
    """
    gt, recon, diff: [H,W] numpy (0–1)
    横一列に並べて保存
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(gt, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title("GT")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(recon, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title("Recon")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap="hot", vmin=0.0, vmax=np.max(diff) + 1e-8)
    plt.title("|GT - Recon|")
    plt.axis("off")

    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def infer_in_ch_from_state_dict(state_dict):
    """
    checkpoint["model"] から in_ch (#coils) を推定
    kx_net の最初の Conv の in_channels = 2 * in_ch を利用する
    """
    conv_weight = None
    for k, v in state_dict.items():
        if ".kx_net." in k and v.ndim == 4:
            conv_weight = v
            break
    if conv_weight is None:
        # fallback: 最初に見つかった Conv2d
        for k, v in state_dict.items():
            if v.ndim == 4:  # [out,in,kh,kw]
                conv_weight = v
                break

    if conv_weight is None:
        raise RuntimeError("Conv2d weightが state_dict から見つかりませんでした。")

    c2 = conv_weight.shape[1]  # ここが 2 * in_ch
    if c2 % 2 != 0:
        raise RuntimeError(f"conv in_channels={c2} から in_ch を 2で割り切れません。")

    in_ch = c2 // 2
    print(f"[INFO] inferred in_ch={in_ch} from state_dict")
    return in_ch

def infer_ky_kx_from_state_dict(state_dict):
    """
    checkpoint["model"] から ky_len / kx_len を推定
    ifft_kx.weight: [kx_len, kx_len]
    ifft_ky.weight: [ky_len, ky_len]
    """
    ky_len = None
    kx_len = None
    for k, v in state_dict.items():
        if "ifft_kx.weight" in k:
            kx_len = v.shape[0]
        if "ifft_ky.weight" in k:
            ky_len = v.shape[0]

    if ky_len is None or kx_len is None:
        raise RuntimeError("ifft_kx/ifft_ky の weight から ky_len/kx_len を推定できませんでした。")

    print(f"[INFO] inferred ky_len={ky_len}, kx_len={kx_len} from state_dict")
    return ky_len, kx_len


def main():
    ap = argparse.ArgumentParser("Visualize GT / Recon / Diff (side-by-side)")
    ap.add_argument("--glob", type=str, required=True,
                    help='例: "C:/.../brain_multicoil/test/multicoil_test/*.h5"')
    ap.add_argument("--ckpt", type=str, required=True,
                    help="学習済み checkpoint (separable_ifft_best.pt)")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="画像の保存先ディレクトリ")
    ap.add_argument("--num-files", type=int, default=3,
                    help="可視化に使うファイル数")
    ap.add_argument("--slices-per-file", type=int, default=2,
                    help="各ファイルから可視化するスライス数")
    ap.add_argument("--target-h", type=int, default=256,
                    help="評価と同じ高さにリサイズ (例: 256)")
    ap.add_argument("--target-w", type=int, default=128,
                    help="評価と同じ幅にリサイズ (例: 128)")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- checkpoint 読み込み ----
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = ckpt["model"]
    ckpt_args = ckpt.get("args", {})

    in_ch = infer_in_ch_from_state_dict(sd)
    ky_len, kx_len = infer_ky_kx_from_state_dict(sd)
    base_ch = ckpt_args.get("base_ch", 32)
    growth = ckpt_args.get("growth", 16)
    n_layers = ckpt_args.get("n_layers", 4)

    # ---- glob からファイル一覧 ----
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files found for glob: {args.glob}")

    print(f"[INFO] model expects Ky,Kx = ({ky_len},{kx_len})")
    print(f"Found {len(files)} files, using {args.num_files} files.")

    # ---- モデル構築（学習時と同じ ky_len/kx_len）----
    model = SeparableIFFTStripesNet(
        in_ch=in_ch,
        base_ch=base_ch,
        growth=growth,
        n_layers=n_layers,
        ky_len=ky_len,
        kx_len=kx_len
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    print("Loaded checkpoint:", args.ckpt)

    # ---- 可視化ループ ----
    use_files = files[: args.num_files]
    for fi, fp in enumerate(use_files):
        with h5py.File(fp, "r") as f:
            nsl, ncoils, Ky_full, Kx_full = f["kspace"].shape

        # モデルの想定より小さい k-space はスキップ
        if Ky_full < ky_len or Kx_full < kx_len:
            print(f"[{fi+1}/{len(use_files)}] file={os.path.basename(fp)} "
                  f"[SKIP] size mismatch: file has (Ky,Kx)=({Ky_full},{Kx_full}), "
                  f"model expects ({ky_len},{kx_len})")
            continue

        max_slices = min(args.slices_per_file, nsl)

        for sl in range(max_slices):
            print(f"[{fi+1}/{len(use_files)}] file={os.path.basename(fp)}, slice={sl}")

            # k-space 読み込み
            with h5py.File(fp, "r") as f:
                ks_full = to_complex(f["kspace"][sl])  # [C_full,Ky_full,Kx_full]

            C_full, Ky, Kx = ks_full.shape

            # coil 数を in_ch に揃える
            if C_full > in_ch:
                ks = ks_full[:in_ch]
            elif C_full < in_ch:
                pad = np.zeros((in_ch - C_full, Ky, Kx), dtype=ks_full.dtype)
                ks = np.concatenate([ks_full, pad], axis=0)
            else:
                ks = ks_full

            # Ky,Kx を ky_len,kx_len に中心クロップ（Ky,Kx >= ky_len,kx_len は保証済）
            ks = center_crop_kspace(ks, ky_len, kx_len)
            _, Ky_c, Kx_c = ks.shape

            # ---- GT 画像（クロップ済み ks から ifft2c+RSS+norm995）----
            imgs_all = np.stack([ifft2c(ks[c]) for c in range(in_ch)], axis=0)  # [C,H,W] complex
            gt = norm995(rss(imgs_all)).astype(np.float32)  # [H,W], 0–1
            gt_t = torch.from_numpy(gt)[None, None]  # [1,1,H,W]

            # ターゲットサイズにリサイズ（学習と条件合わせ）
            if args.target_h is not None and args.target_w is not None:
                gt_t = F.interpolate(
                    gt_t,
                    size=(args.target_h, args.target_w),
                    mode="bilinear",
                    align_corners=False
                )

            # ---- モデルで再構成 ----
            K_t = torch.from_numpy(ks)[None].to(device)  # [1,C,Ky_c,Kx_c] complex
            with torch.no_grad():
                pred = model(K_t)  # [1,1,Ky_c,Kx_c]
                if pred.shape[-2:] != (args.target_h, args.target_w):
                    pred = F.interpolate(
                        pred,
                        size=(args.target_h, args.target_w),
                        mode="bilinear",
                        align_corners=False
                    )

            # numpy へ
            gt_np = gt_t[0, 0].cpu().numpy()
            recon_np = pred[0, 0].cpu().numpy()
            gt_np = np.clip(gt_np, 0.0, 1.0)
            recon_np = np.clip(recon_np, 0.0, 1.0)
            diff_np = np.abs(gt_np - recon_np)

            base = f"{os.path.splitext(os.path.basename(fp))[0]}_sl{sl}"
            out_path = os.path.join(args.out_dir, f"{base}_triplet.png")
            make_triplet_figure(gt_np, recon_np, diff_np, out_path)
    
    print("recon stats:", recon_np.min(), recon_np.max(), recon_np.mean())

    print("Done. Saved triplet images to:", args.out_dir)


if __name__ == "__main__":
    main()
