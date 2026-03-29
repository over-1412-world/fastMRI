# scripts/train_stripes_cnn.py
import os, sys, glob, argparse, time, random, csv
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import chain

# ---- CUDA 設定（速度安定化）----
print("CUDA:", torch.cuda.is_available(),
      "Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# import path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mri_strip.fft_utils import to_complex, ifft2c, rss, norm995
from mri_strip.sampling import variable_density_random, add_conjugates
from mri_strip.strips import zf_ifft_rss_from_lines
from mri_strip.metrics import psnr as psnr_np, ssim as ssim_np
from models.stripes_cnn import StripesReconstNet


# -----------------------------
# Dataset (サイズ統一リサイズ付き)
# -----------------------------
class FastMRIStripsDataset(Dataset):
    """
    fastMRI brain_multicoil の h5 から (inputs, target) を作成。
    inputs: [C=num_strips, H, W]  … 各ky 1本(+共役)の ZF-RSS strip をC枚スタック
    target: [1, H, W]            … 全kyのRSS（擬似GT）
    すべて 0–1 に正規化済み。最後に (target_h, target_w) へバイリニアでリサイズ。
    """
    def __init__(self, files, num_strips=4, center_frac=0.12, seed=0,
                 shuffle_lines=True, target_h=640, target_w=320):
        self.files = files
        self.num_strips = int(num_strips)
        self.center_frac = float(center_frac)
        self.rng = np.random.default_rng(seed)
        self.shuffle_lines = shuffle_lines
        self.th = int(target_h)
        self.tw = int(target_w)

        # (file, slice) のインデックス
        self.index = []
        for fp in self.files:
            with h5py.File(fp, "r") as f:
                nsl = f["kspace"].shape[0]
            for sl in range(nsl):
                self.index.append((fp, sl))

    def __len__(self):
        return len(self.index)

    def _make_strips(self, ks, Ky):
        """num_strips 本の ky を選んで strip を作成し [C,H,W] で返す"""
        base = variable_density_random(Ky, self.num_strips,
                                       center_frac=self.center_frac,
                                       seed=self.rng.integers(1e9))
        if self.shuffle_lines:
            self.rng.shuffle(base)
        strips = []
        for ky in base:
            lines = add_conjugates(Ky, [int(ky)])
            s = zf_ifft_rss_from_lines(ks, lines)  # 0-1, [H,W]
            strips.append(s.astype(np.float32))
        return np.stack(strips, axis=0)  # [C,H,W]

    def __getitem__(self, i):
        fp, sl = self.index[i]
        with h5py.File(fp, "r") as f:
            ks = to_complex(f["kspace"][sl])  # [coils,Ky,Kx] complex64
        coils, Ky, Kx = ks.shape

        # GT（全ky）：RSS→0-1
        imgs_all = np.stack([ifft2c(ks[c]) for c in range(coils)], axis=0)
        gt = norm995(rss(imgs_all)).astype(np.float32)  # [H,W]

        # 入力ストリップ
        x = self._make_strips(ks, Ky)  # [C,H,W]

        # torch に変換
        x = torch.from_numpy(x)          # [C,H,W]
        y = torch.from_numpy(gt)[None]   # [1,H,W]

        # サイズ統一（すべて (th, tw) に）
        x = F.interpolate(x.unsqueeze(0), size=(self.th, self.tw),
                          mode="bilinear", align_corners=False).squeeze(0)
        y = F.interpolate(y.unsqueeze(0), size=(self.th, self.tw),
                          mode="bilinear", align_corners=False).squeeze(0)
        return x, y, os.path.basename(fp), sl


# -----------------------------
# 1 epoch 実行
# -----------------------------
def run_epoch(model, loader, optimizer=None, device="cuda", tag="train"):
    is_train = optimizer is not None   #optimizer があれば学習モード、無ければ評価モード
    model.train(is_train)
    loss_fn = nn.MSELoss()   #損失関数:平均二乗誤差
    #各種集計用に初期化
    total_loss, n_batches = 0.0, 0
    psnr_list, ssim_list = [], []
    #最初のバッチを読み込んでウォームアップ
    print(f"[{tag}] warmup: building first batch ...", flush=True)
    it = iter(loader)   #DataLoaderのイテレータを作成
    t0 = time.perf_counter()   #開始時間を記録
    try:
        first = next(it)   #最初のバッチを取得
    except StopIteration:
        # データが空の場合はゼロを返す
        return 0.0, 0.0, 0.0
    t1 = time.perf_counter()   #取得完了時間を記録
    print(f"[{tag}] first batch fetched in {t1 - t0:.2f}s | x={tuple(first[0].shape)}, y={tuple(first[1].shape)}",
          flush=True)

    # 逐次処理（メモリに溜め込まない）
    #chain([first], it) で、first + 残りバッチをまとめてループ
    for batch_idx, (x, y, fname, sl) in enumerate(chain([first], it), start=1):
        t_fetch = time.perf_counter()   #バッチ取得完了時間
        #入力と教師をGPUへ転送
        x = x.to(device, non_blocking=True)   # [B,C,H,W]
        y = y.to(device, non_blocking=True)   # [B,1,H,W]

        #学習時のみ勾配をリセット
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        #順伝搬:モデル出力（再構成画像）を得る
        pred = model(x)                       # [B,1,H,W]
        #損失計算(MSE)
        loss = loss_fn(pred, y)

        if is_train:
            #逆伝搬
            loss.backward()
            #勾配ノルムを5以下にクリップして爆発防止
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            #パラメータ更新
            optimizer.step()
        t_step = time.perf_counter()   #ステップ終了時刻

        total_loss += float(loss.item()); n_batches += 1   #バッチごとの損失を集計

        # CPUへ戻して PSNR/SSIM　の計算
        with torch.no_grad():
            #TensorをNumPy配列に変換
            p = pred.detach().cpu().numpy()
            g = y.detach().cpu().numpy()
            #バッチ内の各画像についてPSNR/SSIMを計算してリストに追加
            for b in range(p.shape[0]):
                psnr_list.append(float(psnr_np(g[b,0], p[b,0])))   #psnr値
                ssim_list.append(float(ssim_np(g[b,0], p[b,0])))   #ssim値

        # デバッグ出力（頻度高め）
        if (batch_idx % 5) == 0 or batch_idx == 1:
            print(f"[{tag}] batch {batch_idx}/{len(loader)} | step {(t_step - t_fetch):.2f}s | loss {loss.item():.4f}",
                  flush=True)
    #エポック平均値を算出
    avg_loss = total_loss / max(1, n_batches)
    m_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    m_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0

    #学習又は検証エポックの平均結果を返す
    return avg_loss, m_psnr, m_ssim


# -----------------------------
# main
# -----------------------------
def main():
    # -----------------------------
    # 引数パーサ設定
    # -----------------------------
    ap = argparse.ArgumentParser("Train StripesReconstNet on fastMRI brain (ZF strip inputs)")
    ap.add_argument("--glob", type=str, required=True,
                    help='例: "C:/.../multicoil_train/*.h5"（; 区切りで複数指定可）')
    ap.add_argument("--val-glob", type=str, default=None,
                    help="検証セットを分離指定したい場合のglob（未指定なら --glob を split）")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="val-glob 未指定時の簡易split比率")
    ap.add_argument("--num-strips", type=int, default=4, help="入力ストリップ本数（チャネル数）")
    ap.add_argument("--center-frac", type=float, default=0.12)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-h", type=int, default=640, help="統一リサイズ後の高さ（Ky方向）")
    ap.add_argument("--target-w", type=int, default=320, help="統一リサイズ後の幅（Kx方向）")
    ap.add_argument("--save-dir", type=str, default=os.path.join(ROOT, "outputs", "checkpoints"))
    ap.add_argument("--workers", type=int, default=0)  # Windowsは0が安定
    ap.add_argument("--debug", action="store_true", help="少数ファイル＆詳細ログでの動作確認モード")

    # EarlyStopping / Scheduler / Logging 関連オプション
    ap.add_argument("--patience", type=int, default=5, help="EarlyStopping の猶予エポック")
    ap.add_argument("--early-stop-metric", type=str, default="score",
                    choices=["score", "psnr", "ssim", "neg_val_loss"],
                    help="停止判定に使う指標（score=PSNR+10*SSIM を最大化）")
    ap.add_argument("--lr-schedule", type=str, default="none",
                    choices=["none", "cosine", "step", "plateau"],
                    help="学習率スケジューラ")
    ap.add_argument("--step-size", type=int, default=10, help="StepLR の step_size")
    ap.add_argument("--gamma", type=float, default=0.5, help="Step/Plateau の gamma")
    ap.add_argument("--min-lr", type=float, default=1e-6, help="最小学習率（Cosine/Plateau で使用）")
    ap.add_argument("--save-log", action="store_true", help="CSVで学習ログを保存（曲線描画用）")

    #ここで実際にコマンドライン引数を解析
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # glob は ; 区切りで複数対応
    def expand_globs(g):
        parts = [p.strip() for p in g.split(";") if p.strip()]
        files = []
        for p in parts:
            files.extend(glob.glob(p))
        return sorted(files)

    files = expand_globs(args.glob)
    if not files:
        raise SystemExit(f"No files found for glob: {args.glob}")

    if args.val_glob:
        val_files = expand_globs(args.val_glob)
        if not val_files:
            raise SystemExit(f"No files found for val-glob: {args.val_glob}")
        train_files = files
    else:
        n_val = max(1, int(len(files) * args.val_ratio))
        val_files = files[:n_val]
        train_files = files[n_val:]

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # 追加：--debug のときは極小に
    if args.debug:
        train_files = train_files[:2]
        val_files   = val_files[:1]
        print(f"[DEBUG] reducing to Train {len(train_files)} files, Val {len(val_files)}", flush=True)

    train_ds = FastMRIStripsDataset(train_files, num_strips=args.num_strips,
                                    center_frac=args.center_frac, seed=args.seed,
                                    shuffle_lines=True, target_h=args.target_h, target_w=args.target_w)
    val_ds   = FastMRIStripsDataset(val_files,   num_strips=args.num_strips,
                                    center_frac=args.center_frac, seed=args.seed+1,
                                    shuffle_lines=False, target_h=args.target_h, target_w=args.target_w)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=max(1, args.batch_size//2), shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # モデル & Optimizer & Scheduler
    model = StripesReconstNet(in_ch=args.num_strips, base_ch=32, growth=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    elif args.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.gamma, patience=2,
            min_lr=args.min_lr, verbose=True
        )

    # ログCSV
    log_path = os.path.join(args.save_dir, "train_log.csv") if args.save_log else None
    if log_path:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","train_loss","train_psnr","train_ssim",
                        "val_loss","val_psnr","val_ssim","lr","score"])

    def score_fn(psnr, ssim):
        return float(psnr) + 10.0 * float(ssim)

    def pick_metric(va_loss, va_psnr, va_ssim):
        if args.early_stop_metric == "psnr":
            return va_psnr
        if args.early_stop_metric == "ssim":
            return va_ssim
        if args.early_stop_metric == "neg_val_loss":
            return -va_loss
        return score_fn(va_psnr, va_ssim)  # score

    best_metric = -1e9
    best_epoch = 0
    patience_ctr = 0

    # ---- 学習ループ ----
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_psnr, tr_ssim = run_epoch(model, train_loader, optimizer, device, tag="train")
        va_loss, va_psnr, va_ssim = run_epoch(model, val_loader, None, device, tag="val")
        dt = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:02d}] "
              f"train: loss {tr_loss:.4f} | PSNR {tr_psnr:.2f} dB | SSIM {tr_ssim:.4f}  ||  "
              f"val: loss {va_loss:.4f} | PSNR {va_psnr:.2f} dB | SSIM {va_ssim:.4f}  | lr {curr_lr:.2e}  ({dt:.1f}s)")

        # ログ追記
        if log_path:
            with open(log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, tr_loss, tr_psnr, tr_ssim, va_loss, va_psnr, va_ssim, curr_lr,
                            score_fn(va_psnr, va_ssim)])

        # 早期終了判定
        curr_metric = pick_metric(va_loss, va_psnr, va_ssim)
        if curr_metric > best_metric:
            best_metric = curr_metric
            best_epoch = epoch
            patience_ctr = 0
            ckpt = os.path.join(args.save_dir, "stripes_cnn_best.pt")
            torch.save({"model": model.state_dict(),
                        "args": vars(args),
                        "epoch": epoch,
                        "val_psnr": va_psnr,
                        "val_ssim": va_ssim}, ckpt)
            print(f"  -> saved: {ckpt}")
        else:
            patience_ctr += 1

        # スケジューラ更新
        if scheduler is not None:
            if args.lr_schedule == "plateau":
                scheduler.step(va_loss)
            else:
                scheduler.step()

        # patience 超えたら打ち切り
        if patience_ctr >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best metric {best_metric:.4f})")
            break

    # last 保存（最終状態）
    last = os.path.join(args.save_dir, "stripes_cnn_last.pt")
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "val_psnr": va_psnr,
        "val_ssim": va_ssim
    }, last)
    print(f"Done. Saved last checkpoint to {last} (epoch {epoch}, val_psnr={va_psnr:.2f}, val_ssim={va_ssim:.4f})")
    if log_path:
        print(f"Saved log to {log_path}")


if __name__ == "__main__":
    main()
