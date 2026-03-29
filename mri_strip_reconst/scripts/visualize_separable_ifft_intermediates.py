# mri_strip_reconst/scripts/visualize_separable_ifft_intermediates.py
import os, sys, glob, argparse
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt

# ---- import path 設定 ----
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mri_strip.fft_utils import to_complex, ifft2c, rss, norm995
from models.stripes_separable_ifft import SeparableIFFTStripesNet


def save_gray_image(img2d: np.ndarray, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.imshow(img2d, cmap="gray")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def load_kspace_and_gt(fp, sl, target_h=None, target_w=None):
    with h5py.File(fp, "r") as f:
        ks = to_complex(f["kspace"][sl])  # [C,Ky,Kx]

    coils, Ky, Kx = ks.shape

    imgs_all = np.stack([ifft2c(ks[c]) for c in range(coils)], axis=0)  # [C,H,W]
    gt = norm995(rss(imgs_all)).astype(np.float32)[None]  # [1,H,W]

    gt_t = torch.from_numpy(gt)

    if target_h is not None and target_w is not None:
        gt_t = torch.nn.functional.interpolate(
            gt_t.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

    return ks, gt_t


def main():
    ap = argparse.ArgumentParser("Visualize intermediates of SeparableIFFTStripesNet")
    ap.add_argument("--glob", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--num-files", type=int, default=3)
    ap.add_argument("--slices-per-file", type=int, default=2)
    ap.add_argument("--target-h", type=int, default=256)
    ap.add_argument("--target-w", type=int, default=128)
    args = ap.parse_args()


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- checkpoint 読み込み ----
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get("args", {})
    state_dict = ckpt["model"]

    # ① state_dict から in_ch を推定（kx_net の最初の conv を見る）
    axis_ch_in = None
    for k, v in state_dict.items():
        if k.endswith("kx_net.e1.layers.0.0.weight"):  # Conv2d の weight [out, in, kH, kW]
            axis_ch_in = v.shape[1]
            break
    if axis_ch_in is None:
        raise RuntimeError("Failed to infer in_ch from state_dict.")
    in_ch = axis_ch_in // 2  # real/imag の 2ch を含んでいるので /2
    print(f"[INFO] inferred in_ch={in_ch} from state_dict")

    # ② データファイルから (Ky, Kx) を推定
    files = sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files found for glob: {args.glob}")
    with h5py.File(files[0], "r") as f:
        _, _, Ky, Kx = f["kspace"].shape  # [slices, coils, Ky, Kx]
    print(f"[INFO] inferred (Ky,Kx)=({Ky},{Kx}) from {os.path.basename(files[0])}")

    # ③ モデル生成（ky_len, kx_len を渡す）
    model = SeparableIFFTStripesNet(
        in_ch=in_ch,
        ky_len=Ky,
        kx_len=Kx,
        base_ch=ckpt_args.get("base_ch", 32),
        growth=ckpt_args.get("growth", 16),
        n_layers=ckpt_args.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Loaded checkpoint:", args.ckpt)
    print(f"Found {len(files)} files, using {min(args.num_files, len(files))} files.")



    # ---- loop ----
    for fi, fp in enumerate(files[:args.num_files]):
        with h5py.File(fp, "r") as f:
            nsl = f["kspace"].shape[0]

        max_sl = min(args.slices_per_file, nsl)

        for sl in range(max_sl):
            print(f"[{fi+1}/{args.num_files}] file={os.path.basename(fp)}, slice={sl}")

            ks, gt = load_kspace_and_gt(fp, sl, target_h=args.target_h, target_w=args.target_w)
            coils = ks.shape[0]

            # ---- coil mismatch check ----
            if coils != in_ch:
                print(f"  [SKIP] coil mismatch: file has {coils}, model expects {in_ch}")
                continue

            K_t = torch.from_numpy(ks)[None].to(device)

            with torch.no_grad():
                img_hat, hybrid_kx, hybrid_kx_p, img_ky = model(K_t, return_intermediates=True)

            # ---- magnitude ----
            def rss_mag(xc):
                mag = torch.sqrt((xc.abs() ** 2).sum(dim=1, keepdim=True) + 1e-12)
                mag = mag[0, 0].cpu().numpy()
                vmin, vmax = np.percentile(mag, [1, 99])
                return np.clip((mag - vmin) / (vmax - vmin + 1e-12), 0, 1)

            kx_ifft_mag = rss_mag(hybrid_kx)
            kx_net_mag = rss_mag(hybrid_kx_p)
            ky_ifft_mag = rss_mag(img_ky)
            recon = np.clip(img_hat[0, 0].cpu().numpy(), 0, 1)
            gt_np = np.clip(gt[0].cpu().numpy(), 0, 1)

            base = f"{os.path.splitext(os.path.basename(fp))[0]}_sl{sl}"

            save_gray_image(kx_ifft_mag, os.path.join(args.out_dir, f"{base}_kx_ifft.png"))
            save_gray_image(kx_net_mag,  os.path.join(args.out_dir, f"{base}_kx_net.png"))
            save_gray_image(ky_ifft_mag, os.path.join(args.out_dir, f"{base}_ky_ifft.png"))
            save_gray_image(recon,       os.path.join(args.out_dir, f"{base}_recon.png"))
            save_gray_image(gt_np,       os.path.join(args.out_dir, f"{base}_gt.png"))

    print("Done. Saved images to:", args.out_dir)


if __name__ == "__main__":
    main()
