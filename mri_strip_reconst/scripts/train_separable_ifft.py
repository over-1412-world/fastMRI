# scripts/train_separable_ifft.py
import os, sys, glob, argparse, time, random, csv
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from pytorch_msssim import ssim as ssim_loss

# ---- CUDA 設定（速度安定化）----
print("CUDA:", torch.cuda.is_available(),
      "Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# import path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# fastMRI / 再構成ユーティリティ
from mri_strip.fft_utils import to_complex, ifft2c, rss, norm995
from mri_strip.metrics import psnr as psnr_np, ssim as ssim_np

# 新モデル
from models.stripes_separable_ifft import SeparableIFFTStripesNet


# -----------------------------
# Ky 方向サンプリングマスク生成
# -----------------------------
def make_ky_mask(ny=640, ns=32, center_frac=0.12, rng=None):
    """
    Ky 方向の 1D サンプリングマスクを生成
      ny: Ky 本数 (例: 640)
      ns: 取得したい Ky ライン本数
      center_frac: 中心帯の割合（0.12 なら Ny*0.12 本が中心）
    戻り値: shape (ny,) の bool 配列
    """
    if rng is None:
        rng = np.random.default_rng()

    if ns >= ny:
        # フルサンプル
        return np.ones(ny, dtype=bool)

    mask = np.zeros(ny, dtype=bool)

    n_center = int(ny * center_frac)
    n_center = max(1, min(n_center, ny))
    start = (ny - n_center) // 2
    center_idx = np.arange(start, start + n_center)

    # 中心帯からある程度必ずサンプル
    n_center_pick = min(max(1, ns // 4), n_center)
    center_pick = rng.choice(center_idx, size=n_center_pick, replace=False)
    mask[center_pick] = True

    # 残り本数を外側からランダムサンプル
    remaining = ns - n_center_pick
    if remaining > 0:
        all_idx = np.arange(ny)
        outer_idx = np.setdiff1d(all_idx, center_idx)
        if remaining > len(outer_idx):
            remaining = len(outer_idx)
        if remaining > 0:
            outer_pick = rng.choice(outer_idx, size=remaining, replace=False)
            mask[outer_pick] = True

    return mask


def enforce_conjugate_symmetry_kspace(ks):
    """
    k-space の共役対称性を強制
    K(ky, kx) ≈ conj(K(-ky, -kx))
    
    方法: (K + conj(K_flipped)) / 2 で平均を取る
    """
    # k-space は [C, Ky, Kx] の形状
    # 共役対称: ks[c, ky, kx] = conj(ks[c, -ky, -kx])
    ks_flipped = np.flip(ks, axis=(1, 2))  # [C, Ky, Kx] を (ky, kx) で反転
    ks_conj_flipped = np.conj(ks_flipped)
    
    # 平均を取る（共役対称性を強制）
    ks_sym = (ks + ks_conj_flipped) / 2.0
    return ks_sym


def center_crop_or_pad_kspace(ks, target_Ky, target_Kx):
    """
    ks: np.complex64 [C,Ky,Kx]
    target_Ky, target_Kx に合わせて
      - 大きければ中心クロップ
      - 小さければゼロパディング
    して返す
    """
    C, Ky, Kx = ks.shape

    # --- Ky方向 ---
    if Ky > target_Ky:
        start = (Ky - target_Ky) // 2
        ks = ks[:, start:start + target_Ky, :]
    elif Ky < target_Ky:
        pad_top = (target_Ky - Ky) // 2
        pad_bottom = target_Ky - Ky - pad_top
        ks_pad = np.zeros((C, target_Ky, Kx), dtype=ks.dtype)
        ks_pad[:, pad_top:pad_top + Ky, :] = ks
        ks = ks_pad

    # --- Kx方向 ---
    _, Ky2, Kx2 = ks.shape
    if Kx2 > target_Kx:
        start = (Kx2 - target_Kx) // 2
        ks = ks[:, :, start:start + target_Kx]
    elif Kx2 < target_Kx:
        pad_left = (target_Kx - Kx2) // 2
        pad_right = target_Kx - Kx2 - pad_left
        ks_pad = np.zeros((C, Ky2, target_Kx), dtype=ks.dtype)
        ks_pad[:, :, pad_left:pad_left + Kx2] = ks
        ks = ks_pad

    return ks


def coil_compress_svd(ks, target_coils):
    """
    SVD による coil compression。
    ks: np.complex64 [C, Ky, Kx] -> [target_coils, Ky, Kx]
    """
    C, Ky, Kx = ks.shape
    if target_coils is None or target_coils <= 0:
        return ks

    if C == target_coils:
        return ks

    if C < target_coils:
        pad = np.zeros((target_coils - C, Ky, Kx), dtype=ks.dtype)
        return np.concatenate([ks, pad], axis=0)

    x = ks.reshape(C, -1)  # [C, N]
    u, _, _ = np.linalg.svd(x, full_matrices=False)
    basis = np.conj(u[:, :target_coils]).T  # [target_coils, C]
    compressed = basis @ x  # [target_coils, N]
    return compressed.reshape(target_coils, Ky, Kx).astype(np.complex64, copy=False)


def inspect_kspace_file(fp):
    """ファイルの k-space 形状情報を取得。失敗時は (None, reason) を返す。"""
    try:
        with h5py.File(fp, "r") as f:
            if "kspace" not in f:
                return None, "missing_kspace"
            shape = f["kspace"].shape
            if len(shape) != 4:
                return None, f"invalid_kspace_shape:{shape}"
            nsl, ncoils, Ky, Kx = shape
            if nsl <= 0 or ncoils <= 0 or Ky <= 0 or Kx <= 0:
                return None, f"non_positive_shape:{shape}"
        return {"file": fp, "slices": nsl, "coils": ncoils, "Ky": Ky, "Kx": Kx}, None
    except Exception as e:
        return None, f"read_error:{type(e).__name__}"


# -----------------------------
# Dataset: full/undersampled k-space + RSS GT
# -----------------------------
class FastMRIFullKspaceDataset(Dataset):
    """
    fastMRI brain_multicoil の h5 から
      入力: （必要なら Ky サンプリングした）complex k-space [C, Ky, Kx]
      教師: （使用する coil だけから作った）RSS 画像 [1, H, W] (0–1 正規化済み)

    ★ポイント
    - ファイルごとに coil 数が違っても OK。
      → このクラスの中で「使うコイル数 self.coils」に揃える
        （基本は全ファイルの最小コイル数 or expected_coils との min）。
    - ref_Ky, ref_Kx が指定されていれば、k-space をそのサイズに
      center crop / pad してモデルに渡す。
    """
    def __init__(self, files, target_h=None, target_w=None,
                 expected_coils=None,
                 undersample=False, ns=32, center_frac=0.12, seed=42,
                 ref_Ky=None, ref_Kx=None, min_coils=None,
                 coil_compress_to=None):
        self.files = files
        self.th = target_h
        self.tw = target_w
        self.ref_Ky = ref_Ky
        self.ref_Kx = ref_Kx

        self.undersample = undersample
        self.ns = ns
        self.center_frac = center_frac
        self.rng = np.random.default_rng(seed)
        self.coil_compress_to = coil_compress_to

        self.index = []
        self.coils = None  # この Dataset で実際に使うコイル数

        # ---- この Dataset 内の最小コイル数を調べる ----
        min_coils = None
        for fp in self.files:
            with h5py.File(fp, "r") as f:
                kshape = f["kspace"].shape  # [slices, coils, Ky, Kx]
                nsl, ncoils, Ky, Kx = kshape

            if min_coils is None:
                min_coils = ncoils
                # ref_Ky / ref_Kx が指定されていなければ、このファイルを基準にする
                if self.ref_Ky is None:
                    self.ref_Ky = Ky
                if self.ref_Kx is None:
                    self.ref_Kx = Kx
            else:
                min_coils = min(min_coils, ncoils)

            for sl in range(nsl):
                self.index.append((fp, sl))

        # expected_coils / coil_compress_to と突き合わせて使用 coil 数を決定
        candidate = min_coils
        if expected_coils is not None:
            candidate = min(candidate, expected_coils)
        if self.coil_compress_to is not None:
            candidate = min(candidate, int(self.coil_compress_to))
        self.coils = int(candidate)

        print(
            f"[FastMRIFullKspaceDataset] total slices: {len(self.index)}, "
            f"min_coils={min_coils}, using_coils={self.coils}, "
            f"undersample={self.undersample}, ns={self.ns}, "
            f"center_frac={self.center_frac}, ref_Ky={self.ref_Ky}, ref_Kx={self.ref_Kx}"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fp, sl = self.index[i]
        with h5py.File(fp, "r") as f:
            ks_full = to_complex(f["kspace"][sl])  # numpy complex64, [C,Ky,Kx]

        coils_orig, Ky, Kx = ks_full.shape

        # ---- coil 数を self.coils に揃える（SVD圧縮優先）----
        if self.coil_compress_to is not None:
            ks = coil_compress_svd(ks_full, self.coils)
        else:
            if coils_orig > self.coils:
                ks = ks_full[: self.coils]  # 先頭 self.coils を使用
            elif coils_orig < self.coils:
                pad = np.zeros((self.coils - coils_orig, Ky, Kx), dtype=ks_full.dtype)
                ks = np.concatenate([ks_full, pad], axis=0)
            else:
                ks = ks_full

        assert ks.shape[0] == self.coils

        # ---- 教師画像(RSS) : 「使うコイル」だけから作る ----
        imgs_all = np.stack([ifft2c(ks[c]) for c in range(self.coils)], axis=0)  # [C,H,W] complex
        gt = norm995(rss(imgs_all)).astype(np.float32)  # [H,W], 0-1

        # ---- 入力 k-space : 必要なら Ky サンプリング ----
        ks_in = ks
        Ky_curr, Kx_curr = ks_in.shape[1], ks_in.shape[2]

        if self.undersample:
            mask = make_ky_mask(ny=Ky_curr, ns=self.ns,
                                center_frac=self.center_frac, rng=self.rng)
            # [C,Ky,Kx] * [1,Ky,1]
            ks_in = ks_in * mask[None, :, None]

        # ---- 共役対称性を強制（虚部削減）---- 
        ks_in = enforce_conjugate_symmetry_kspace(ks_in)

        # ---- 必要なら (Ky,Kx) を ref_Ky, ref_Kx に center crop / pad ----
        if (self.ref_Ky is not None) and (self.ref_Kx is not None):
            ks_in = center_crop_or_pad_kspace(ks_in, self.ref_Ky, self.ref_Kx)

        # torch へ変換（複素型を明示的に保証）
        K = torch.from_numpy(ks_in.astype(np.complex64))  # complex64 tensor, [C,Ky',Kx']
        y = torch.from_numpy(gt.astype(np.float32))[None]    # [1,H,W]

        # 画像側を統一サイズにリサイズ（バッチでの collate エラーを防止）
        # ref_Ky, ref_Kx がある場合は、モデル出力サイズも統一されるため
        # GT も同じサイズに揃える
        if self.ref_Ky is not None and self.ref_Kx is not None:
            y = F.interpolate(
                y.unsqueeze(0), size=(self.ref_Ky, self.ref_Kx),
                mode="bilinear", align_corners=False
            ).squeeze(0)
        elif self.th is not None and self.tw is not None:
            y = F.interpolate(
                y.unsqueeze(0), size=(self.th, self.tw),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        return K, y, os.path.basename(fp), sl


# -----------------------------
# 1 epoch 実行
# -----------------------------
def run_epoch(model, loader, optimizer=None, device="cuda", tag="train", lambda_phys: float = 0.0):
    is_train = optimizer is not None
    model.train(is_train)

    mse_fn = nn.MSELoss()
    # SSIM Loss: 1 - SSIM（最小化対象に変換）
    # data_range=1.0 を指定（画像が0-1正規化済みのため）

    total_loss, n_batches = 0.0, 0
    psnr_list, ssim_list = [], []
    diag_acc = {
        "kx_phase_norm": 0.0,
        "kx_alpha_norm": 0.0,
        "kx_imag_energy": 0.0,
        "kx_symmetry_error": 0.0,
        "ky_phase_norm": 0.0,
        "ky_alpha_norm": 0.0,
        "ky_imag_energy": 0.0,
        "ky_symmetry_error": 0.0,
    }

    # warmup: 最初のバッチ読み込み
    print(f"[{tag}] warmup: building first batch ...", flush=True)
    it = iter(loader)
    t0 = time.perf_counter()
    try:
        first = next(it)
    except StopIteration:
        return 0.0, 0.0, 0.0
    t1 = time.perf_counter()
    print(f"[{tag}] first batch fetched in {t1 - t0:.2f}s | "
          f"x={tuple(first[0].shape)}, y={tuple(first[1].shape)}", flush=True)

    for batch_idx, (K, y, fname, sl) in enumerate(chain([first], it), start=1):
        t_fetch = time.perf_counter()

        # 入力: complex k-space [B,C,Ky,Kx]
        K = K.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)  # [B,1,th,tw]（例: 256x128）

        # 複素型が保持されているか確認（デバッグ用）
        if batch_idx == 1 and is_train:
            print(f"[DEBUG] K dtype={K.dtype}, K shape={K.shape}, is_complex={torch.is_complex(K)}", flush=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # モデルは内部で 1D IFFT(kx/ky) + 軸ネットを通して画像を出す
        pred, diagnostics = model(K, return_diagnostics=True)  # [B,1,H,W], 0-1
        
        # 出力値の範囲確認（デバッグ用）
        if batch_idx == 1 and is_train:
            print(f"[DEBUG] pred min={pred.min():.6f}, max={pred.max():.6f}, mean={pred.mean():.6f}", flush=True)
            print(f"[DEBUG] y min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}", flush=True)

        # GT とサイズが異なる場合は、GT のサイズに合わせてリサイズ（例: 256x128）
        if pred.shape[-2:] != y.shape[-2:]:
            pred = F.interpolate(
                pred, size=y.shape[-2:], mode="bilinear", align_corners=False
            )

        # 混合 Loss: MSE + SSIM Loss の加重和
        # SSIM Loss は pytorch_msssim を使用
        # ssim_loss は 0-1の値を返し、1に近いほど類似度が高い
        # (1 - ssim_loss) で最小化対象に変換
        mse_loss = mse_fn(pred, y)
        try:
            ssim_val = ssim_loss(pred, y, data_range=1.0, size_average=True)  # 0-1, high is good
            ssim_loss_val = 1.0 - ssim_val  # 最小化対象
            loss = 0.5 * mse_loss + 0.5 * ssim_loss_val  # 50:50 混合
        except Exception as e:
            # SSIM計算失敗時はMSEのみ使用（バッチサイズが小さい場合など）
            loss = mse_loss

        # 物理整合正則化（symmetry_error を抑制）
        if lambda_phys > 0.0:
            sym_kx = diagnostics["kx"]["symmetry_error"]
            sym_ky = diagnostics["ky"]["symmetry_error"]
            phys_loss = 0.5 * (sym_kx + sym_ky)
            loss = loss + lambda_phys * phys_loss

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        t_step = time.perf_counter()

        total_loss += float(loss.item())
        n_batches += 1

        # 診断ログ集計
        for prefix in ("kx", "ky"):
            diag_acc[f"{prefix}_phase_norm"] += diagnostics[prefix]["phase_norm"]
            diag_acc[f"{prefix}_alpha_norm"] += diagnostics[prefix]["alpha_norm"]
            diag_acc[f"{prefix}_imag_energy"] += diagnostics[prefix]["imag_energy"]
            diag_acc[f"{prefix}_symmetry_error"] += diagnostics[prefix]["symmetry_error"]

        # CPU に戻して PSNR/SSIM
        with torch.no_grad():
            p = pred.detach().cpu().numpy()
            g = y.detach().cpu().numpy()
            for b in range(p.shape[0]):
                psnr_list.append(float(psnr_np(g[b, 0], p[b, 0])))
                ssim_list.append(float(ssim_np(g[b, 0], p[b, 0])))

        if (batch_idx % 5) == 0 or batch_idx == 1:
            print(f"[{tag}] batch {batch_idx}/{len(loader)} | "
                  f"step {(t_step - t_fetch):.2f}s | loss {loss.item():.4f}",
                  flush=True)

    avg_loss = total_loss / max(1, n_batches)
    m_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    m_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    m_diag = {k: (v / max(1, n_batches)) for k, v in diag_acc.items()}

    return avg_loss, m_psnr, m_ssim, m_diag


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        "Train Separable IFFT DenseNet on fastMRI brain (full or undersampled k-space inputs)"
    )
    ap.add_argument("--glob", type=str, required=True,
                    help='例: "/path/to/multicoil_train/*.h5"（; 区切りで複数指定可）')
    ap.add_argument("--val-glob", type=str, default=None,
                    help="検証セットを別globで指定する場合。未指定なら --glob を split")
    ap.add_argument("--val-ratio", type=float, default=0.1,
                    help="val-glob 未指定時の split 比率")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="初期学習率（デフォルト 5e-5, 勾配消失対策。LearnableIFFTは超低値推奨）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-h", type=int, default=None,
                    help="GT画像のみリサイズする高さ（未指定ならそのまま）")
    ap.add_argument("--target-w", type=int, default=None,
                    help="GT画像のみリサイズする幅")
    ap.add_argument("--save-dir", type=str, default=os.path.join(ROOT, "outputs", "separable_ifft"))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--debug", action="store_true",
                    help="少数ファイルで動作確認するモード")

    # サンプリング関連オプション
    ap.add_argument("--undersample", action="store_true",
                    help="指定すると Ky 方向サンプリング（間引き）を行う")
    ap.add_argument("--ns", type=int, default=32,
                    help="Ky 方向で残すライン本数 (ns)")
    ap.add_argument("--center-frac", type=float, default=0.12,
                    help="中心帯の割合 (0.12 なら Ny*0.12 本が中心)")

    # EarlyStopping / Scheduler / Logging
    ap.add_argument("--patience", type=int, default=10,
                    help="EarlyStopping の猶予エポック（デフォルト 10, 初期学習を見守る）")
    ap.add_argument("--early-stop-metric", type=str, default="score",
                    choices=["score", "psnr", "ssim", "neg_val_loss"],
                    help="停止判定に使う指標（score=PSNR+10*SSIM を最大化）")
    ap.add_argument("--lr-schedule", type=str, default="none",
                    choices=["none", "cosine", "step", "plateau"],
                    help="学習率スケジューラ種別")
    ap.add_argument("--step-size", type=int, default=10,
                    help="StepLR の step_size")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="Step/Plateau の gamma")
    ap.add_argument("--min-lr", type=float, default=1e-6,
                    help="最小学習率（Cosine/Plateau で使用）")
    ap.add_argument("--save-log", action="store_true",
                    help="CSV で学習ログを保存")
    ap.add_argument("--min-coils", type=int, default=None,
                help="この値未満の coil 数のファイルは除外する（例: 16）")
    ap.add_argument("--target-ky", type=int, default=640,
                    help="k-space の Ky を固定する値（center crop / pad）")
    ap.add_argument("--target-kx", type=int, default=320,
                    help="k-space の Kx を固定する値（center crop / pad）")
    ap.add_argument("--coil-compress", type=int, default=8,
                    help="SVD coil compression 先の coil 数。0以下で無効")
    ap.add_argument("--save-exclusion-log", action="store_true",
                    help="除外ファイルと理由を CSV に保存")
    ap.add_argument("--amp-eps", type=float, default=0.1,
                    help="LearnableIFFT の微小振幅補正係数 ε")
    ap.add_argument("--use-hermitian", dest="use_hermitian", action="store_true",
                    help="ky 段で Hermitian 投影を有効化")
    ap.add_argument("--no-hermitian", dest="use_hermitian", action="store_false",
                    help="Hermitian 投影を無効化")
    ap.add_argument("--lambda-phys", type=float, default=0.01,
                    help="物理整合正則化（symmetry_error）重み")
    ap.add_argument("--freeze-ifft", action="store_true",
                    help="LearnableIFFT のパラメータを凍結（固定IFFT相当）")
    ap.set_defaults(use_hermitian=True)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # glob を展開
    def expand_globs(g):
        """
        セミコロン区切りのグロブを展開。再帰パターン(**)も有効化する。
        例:
          "C:/data/brain_multicoil/train/**/*.h5;D:/extra/*.h5"
        """
        parts = [p.strip() for p in g.split(";") if p.strip()]
        files = []
        for p in parts:
            # Python の glob を再帰対応で使用
            files.extend(glob.glob(p, recursive=True))
        return sorted(files)

    files = expand_globs(args.glob)
    if not files:
        raise SystemExit(f"No files found for glob: {args.glob}")

    random.shuffle(files)

    if args.val_glob:
        val_files = expand_globs(args.val_glob)
        if not val_files:
            raise SystemExit(f"No files found for val-glob: {args.val_glob}")
        train_files = files
    else:
        n_val = max(1, int(len(files) * args.val_ratio))
        val_files = files[:n_val]
        train_files = files[n_val:]

    # 破損ファイル・形状不正を train/val 両方で事前除外
    candidates = list(dict.fromkeys(train_files + val_files))
    inspected, excluded = [], []
    for fp in candidates:
        info, reason = inspect_kspace_file(fp)
        if info is None:
            excluded.append((fp, reason))
        else:
            if args.min_coils is not None and info["coils"] < args.min_coils:
                excluded.append((fp, f"coils_below_min:{info['coils']}<{args.min_coils}"))
            else:
                inspected.append(info)

    valid_set = {x["file"] for x in inspected}
    train_files = [fp for fp in train_files if fp in valid_set]
    val_files = [fp for fp in val_files if fp in valid_set]

    if not train_files:
        raise SystemExit("No valid train files after inspection/filtering.")
    if not val_files:
        raise SystemExit("No valid val files after inspection/filtering.")

    if excluded:
        print(f"Excluded files: {len(excluded)}")
        if args.save_exclusion_log:
            excl_path = os.path.join(args.save_dir, "excluded_files.csv")
            with open(excl_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["file", "reason"])
                w.writerows(excluded)
            print(f"Saved exclusion log: {excl_path}")

    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    if args.debug:
        train_files = train_files[:2]
        val_files   = val_files[:1]
        print(f"[DEBUG] reducing to Train {len(train_files)} files, Val {len(val_files)}", flush=True)

    # ---- train_files + val_files から coil 統計を取る ----
    info_map = {x["file"]: x for x in inspected}
    all_files_for_coils = list(dict.fromkeys(train_files + val_files))
    global_min_coils = min(info_map[fp]["coils"] for fp in all_files_for_coils)
    use_svd_compress = args.coil_compress is not None and args.coil_compress > 0
    if use_svd_compress:
        using_coils = min(global_min_coils, int(args.coil_compress))
    else:
        using_coils = global_min_coils

    example_Ky, example_Kx = int(args.target_ky), int(args.target_kx)

    print(f"Global minimum coils={global_min_coils}, using_coils={using_coils}, target (Ky,Kx)=({example_Ky},{example_Kx})")

    # ---- Dataset / DataLoader ----
    train_ds = FastMRIFullKspaceDataset(
        train_files,
        target_h=args.target_h,
        target_w=args.target_w,
        expected_coils=using_coils,
        undersample=args.undersample,
        ns=args.ns,
        center_frac=args.center_frac,
        seed=args.seed,
        ref_Ky=example_Ky,
        ref_Kx=example_Kx,
        min_coils=args.min_coils,
        coil_compress_to=(using_coils if use_svd_compress else None),
    )
    val_ds = FastMRIFullKspaceDataset(
        val_files,
        target_h=args.target_h,
        target_w=args.target_w,
        expected_coils=using_coils,
        undersample=args.undersample,  # 評価も同じサンプリング条件で
        ns=args.ns,
        center_frac=args.center_frac,
        seed=args.seed + 1,            # 一応別シードにしておく
        ref_Ky=example_Ky,
        ref_Kx=example_Kx,
        min_coils=args.min_coils,
        coil_compress_to=(using_coils if use_svd_compress else None),
    )

    # 念のため train/val で coils が一致しているか確認
    assert train_ds.coils == val_ds.coils, f"train coils={train_ds.coils}, val coils={val_ds.coils}"
    used_coils = train_ds.coils
    print(f"Using coils={used_coils} for model input.")


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # モデル / Optimizer / Scheduler
    model = SeparableIFFTStripesNet(
        in_ch=used_coils,
        ky_len=example_Ky,
        kx_len=example_Kx,
        base_ch=32,
        growth=16,
        n_layers=4,
        ifft_rank=8,  # 低ランク補正を使用（高速化）
        amp_eps=args.amp_eps,
        use_hermitian=args.use_hermitian,
    ).to(device)

    if args.freeze_ifft:
        for p in model.ifft_kx.parameters():
            p.requires_grad = False
        for p in model.ifft_ky.parameters():
            p.requires_grad = False
        print("[INFO] LearnableIFFT parameters are frozen (--freeze-ifft).")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise SystemExit("No trainable parameters found.")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-5)

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

    # ログ CSV
    log_path = os.path.join(args.save_dir, "train_log.csv") if args.save_log else None
    if log_path:
        with open(log_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "train_psnr", "train_ssim",
                        "val_loss", "val_psnr", "val_ssim", "lr", "score",
                        "tr_kx_phase_norm", "tr_kx_alpha_norm", "tr_kx_imag_energy", "tr_kx_symmetry_error",
                        "tr_ky_phase_norm", "tr_ky_alpha_norm", "tr_ky_imag_energy", "tr_ky_symmetry_error",
                        "va_kx_phase_norm", "va_kx_alpha_norm", "va_kx_imag_energy", "va_kx_symmetry_error",
                        "va_ky_phase_norm", "va_ky_alpha_norm", "va_ky_imag_energy", "va_ky_symmetry_error"])

    def score_fn(psnr, ssim):
        return float(psnr) + 10.0 * float(ssim)

    def pick_metric(va_loss, va_psnr, va_ssim):
        if args.early_stop_metric == "psnr":
            return va_psnr
        if args.early_stop_metric == "ssim":
            return va_ssim
        if args.early_stop_metric == "neg_val_loss":
            return -va_loss
        return score_fn(va_psnr, va_ssim)

    best_metric = -1e9
    best_epoch = 0
    patience_ctr = 0

    # ---- 学習ループ ----
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        print(f"\n[Epoch {epoch:02d}] Starting...", flush=True)
        if torch.cuda.is_available():
            print(f"  Device memory: {torch.cuda.memory_allocated()/1e9:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB", flush=True)
        
        tr_loss, tr_psnr, tr_ssim, tr_diag = run_epoch(
            model, train_loader, optimizer, device, tag="train", lambda_phys=args.lambda_phys
        )
        va_loss, va_psnr, va_ssim, va_diag = run_epoch(
            model, val_loader, None, device, tag="val", lambda_phys=0.0
        )
        dt = time.time() - t0

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch:02d}] "
              f"train: loss {tr_loss:.4f} | PSNR {tr_psnr:.2f} dB | SSIM {tr_ssim:.4f}  ||  "
              f"val: loss {va_loss:.4f} | PSNR {va_psnr:.2f} dB | SSIM {va_ssim:.4f}  | lr {curr_lr:.2e}  ({dt:.1f}s)")
        print(f"           diag: tr_imag(kx/ky)=({tr_diag['kx_imag_energy']:.3e}/{tr_diag['ky_imag_energy']:.3e}), "
              f"tr_sym(kx/ky)=({tr_diag['kx_symmetry_error']:.3e}/{tr_diag['ky_symmetry_error']:.3e})")

        if log_path:
            with open(log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, tr_loss, tr_psnr, tr_ssim,
                            va_loss, va_psnr, va_ssim, curr_lr,
                        score_fn(va_psnr, va_ssim),
                        tr_diag["kx_phase_norm"], tr_diag["kx_alpha_norm"], tr_diag["kx_imag_energy"], tr_diag["kx_symmetry_error"],
                        tr_diag["ky_phase_norm"], tr_diag["ky_alpha_norm"], tr_diag["ky_imag_energy"], tr_diag["ky_symmetry_error"],
                        va_diag["kx_phase_norm"], va_diag["kx_alpha_norm"], va_diag["kx_imag_energy"], va_diag["kx_symmetry_error"],
                        va_diag["ky_phase_norm"], va_diag["ky_alpha_norm"], va_diag["ky_imag_energy"], va_diag["ky_symmetry_error"]])

        # 早期終了判定
        curr_metric = pick_metric(va_loss, va_psnr, va_ssim)
        if curr_metric > best_metric:
            best_metric = curr_metric
            best_epoch = epoch
            patience_ctr = 0
            ckpt = os.path.join(args.save_dir, "separable_ifft_best.pt")
            torch.save({
                "model": model.state_dict(),
                "args": {
                    **vars(args),
                    "in_ch": used_coils,
                    "ky_len": example_Ky,
                    "kx_len": example_Kx,
                },
                "epoch": epoch,
                "val_psnr": va_psnr,
                "val_ssim": va_ssim
            }, ckpt)
            print(f"  -> saved: {ckpt}")
        else:
            patience_ctr += 1

        # スケジューラ更新
        if scheduler is not None:
            if args.lr_schedule == "plateau":
                scheduler.step(va_loss)
            else:
                scheduler.step()

        if patience_ctr >= args.patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, best metric {best_metric:.4f})")
            break

    # last 保存
    last = os.path.join(args.save_dir, "separable_ifft_last.pt")
    torch.save({
        "model": model.state_dict(),
        "args": {
            **vars(args),
            "in_ch": used_coils,
            "ky_len": example_Ky,
            "kx_len": example_Kx,
        },
        "epoch": epoch,
        "val_psnr": va_psnr,
        "val_ssim": va_ssim
    }, last)
    print(f"Done. Saved last checkpoint to {last} (epoch {epoch}, val_psnr={va_psnr:.2f}, val_ssim={va_ssim:.4f})")
    if log_path:
        print(f"Saved log to {log_path}")


if __name__ == "__main__":
    main()
