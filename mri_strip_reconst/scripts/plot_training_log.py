# scripts/plot_training_log.py
import os, sys, argparse, csv
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser("Plot curves from train_log.csv")
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--out", default=None, type=str)
    args = ap.parse_args()

    # 読み込み
    keys = []
    rows = []
    with open(args.csv, newline="") as f:
        r = csv.reader(f)
        keys = next(r)
        for row in r:
            rows.append(row)
    data = {k: [] for k in keys}
    for row in rows:
        for k, v in zip(keys, row):
            try:
                data[k].append(float(v))
            except:
                data[k].append(v)

    epoch = np.array(data["epoch"], dtype=float)
    tr_loss = np.array(data["train_loss"], dtype=float)
    va_loss = np.array(data["val_loss"], dtype=float)
    tr_psnr = np.array(data["train_psnr"], dtype=float)
    va_psnr = np.array(data["val_psnr"], dtype=float)
    tr_ssim = np.array(data["train_ssim"], dtype=float)
    va_ssim = np.array(data["val_ssim"], dtype=float)
    lrs = np.array(data["lr"], dtype=float)

    fig = plt.figure(figsize=(12,8))
    gs  = fig.add_gridspec(2,2)

    ax = fig.add_subplot(gs[0,0]); ax.plot(epoch, tr_loss, label="train"); ax.plot(epoch, va_loss, label="val")
    ax.set_title("Loss"); ax.set_xlabel("epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0,1]); ax.plot(epoch, tr_psnr, label="train"); ax.plot(epoch, va_psnr, label="val")
    ax.set_title("PSNR (dB)"); ax.set_xlabel("epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1,0]); ax.plot(epoch, tr_ssim, label="train"); ax.plot(epoch, va_ssim, label="val")
    ax.set_title("SSIM"); ax.set_xlabel("epoch"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1,1]); ax.plot(epoch, lrs)
    ax.set_title("Learning Rate"); ax.set_xlabel("epoch"); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    outpng = args.out or os.path.join(os.path.dirname(args.csv), "training_curves.png")
    plt.savefig(outpng, dpi=160, bbox_inches="tight")
    print(f"Saved: {outpng}")

if __name__ == "__main__":
    main()
