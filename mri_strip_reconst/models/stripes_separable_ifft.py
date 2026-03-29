# models/stripes_separable_ifft.py
"""
Separable IFFT + Axis-wise Dense Encoder–Decoder for fastMRI k-space.

2D 逆フーリエ変換を
  (1) kx 方向 1D IFFT + kx 用 Dense Encoder–Decoder
  (2) ky 方向 1D IFFT + ky 用 Dense Encoder–Decoder
に分解し，最後に 1ch 画像 (0–1) を出力するモデル。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableIFFT1D(nn.Module):
    """
    1D IFFT に学習可能な位相補正を加える層。
    
    ⭐ シンプル版：
    - 基準: torch.fft.ifft（物理的 IFFT）
    - 補正: per-frequency 学習可能な位相シフト φ_k
    - y = IFFT(x) * exp(i * φ)  （軸毎に異なる位相補正）
    
    利点：
    - 共役対称性を保証（IFFT は既に保証）
    - 位相補正だけなので数値的に安定
    - 物理的解釈が明確
    """
    def __init__(
        self,
        N: int,
        dim: int = -1,
        dtype=torch.complex64,
        rank: int = 8,
        amp_eps: float = 0.1,
        use_hermitian: bool = False,
        phase_init_scale: float = 0.0,
        alpha_init_scale: float = 0.0,
    ):
        super().__init__()
        self.N = N
        self.dim = dim
        self.dtype = dtype
        self.rank = rank
        self.amp_eps = float(amp_eps)
        self.use_hermitian = bool(use_hermitian)
        
        # 学習可能な位相シフト: shape [N]
        # 初期値: 0 (何もしない)
        self.phase = nn.Parameter(torch.zeros(N, dtype=torch.float32))
        if phase_init_scale > 0:
            nn.init.normal_(self.phase, mean=0.0, std=phase_init_scale)

        # 学習可能な微小振幅補正パラメータ: A(k)=1+eps*tanh(alpha)
        self.alpha = nn.Parameter(torch.zeros(N, dtype=torch.float32))
        if alpha_init_scale > 0:
            nn.init.normal_(self.alpha, mean=0.0, std=alpha_init_scale)
        
        # スケール調整用（学習可能に）
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    @staticmethod
    def hermitian_project_2d(x: torch.Tensor) -> torch.Tensor:
        """
        2D Hermitian 投影を適用し、x を共役対称に近づける。
        x[..., ky, kx] ~= conj(x[..., -ky, -kx])
        """
        x_flip_conj = torch.conj(torch.flip(x, dims=(-2, -1)))
        return 0.5 * (x + x_flip_conj)

    @staticmethod
    def symmetry_error_2d(x: torch.Tensor) -> torch.Tensor:
        """平均共役対称誤差 mean(|K - conj(flip(K))|) を返す。"""
        x_flip_conj = torch.conj(torch.flip(x, dims=(-2, -1)))
        return (x - x_flip_conj).abs().mean()

    def _expand_1d_param(self, p: torch.Tensor, x_ndim: int) -> torch.Tensor:
        """1D パラメータ [N] を対象軸にブロードキャスト可能な形へ整形。"""
        shape = [1] * x_ndim
        dim = self.dim if self.dim >= 0 else x_ndim + self.dim
        shape[dim] = self.N
        return p.reshape(shape)

    def diagnostics(self, y: torch.Tensor) -> dict:
        """本層の診断値（phase/alpha norm, imag_energy, symmetry_error）を返す。"""
        with torch.no_grad():
            return {
                "phase_norm": float(self.phase.norm(p=2).item()),
                "alpha_norm": float(self.alpha.norm(p=2).item()),
                "imag_energy": float(y.imag.abs().mean().item()),
                "symmetry_error": float(self.symmetry_error_2d(y).item()),
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: complex tensor [B, C, Ky, Kx] など
        
        処理:
        1) 基準の IFFT を適用: y0 = torch.fft.ifft(x, dim=self.dim)
        2) 学習可能な位相補正を追加: y = y0 * exp(i * scale * phase)
        """
        assert torch.is_complex(x), "入力は complex tensor である必要があります"

        # --- (1) 基準の IFFT（高速、共役対称性保証）---
        y = torch.fft.ifft(x, dim=self.dim)

        # --- (2) 制約付き複素補正（位相 + 微小振幅） ---
        # P(k)=exp(i*scale*phase), A(k)=1+eps*tanh(alpha)
        phase_reshaped = self._expand_1d_param(self.phase, x.ndim)
        alpha_reshaped = self._expand_1d_param(self.alpha, x.ndim)

        phase_correction = torch.exp(1j * self.scale * phase_reshaped)
        amp_correction = 1.0 + self.amp_eps * torch.tanh(alpha_reshaped)
        y = y * phase_correction * amp_correction

        # --- (3) 任意で 2D Hermitian 投影 ---
        if self.use_hermitian:
            y = self.hermitian_project_2d(y)

        return y


def center_crop_to(x, ref):
    """
    x を ref と同じ H,W になるように中心クロップする。
    すでに同じサイズならそのまま返す。
    """
    _, _, H, W = x.shape
    _, _, Hr, Wr = ref.shape

    if (H, W) == (Hr, Wr):
        return x

    # enc側の方が大きい前提で中心から切り出す（1ピクセル差などを吸収）
    top = max((H - Hr) // 2, 0)
    left = max((W - Wr) // 2, 0)
    return x[:, :, top:top + Hr, left:left + Wr]


# -----------------------------
# Dense Block / Transition (元モデルと同じ構造)
# -----------------------------
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers=4):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=8, num_channels=growth_rate),
                    nn.ReLU(inplace=True),
                )
            )
            channels += growth_rate
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x):
        return self.deconv(x)

class SeparableIFFTStripesNet(nn.Module):
    """
    入力: complex k-space [B, C, Ky, Kx]  (C = #coils)
    ...
    """
    def __init__(self, in_ch: int, base_ch: int = 32, growth: int = 16, n_layers: int = 4):
        super().__init__()
        # 期待する coil 数を保存しておく（可視化で使う）
        self.in_ch = in_ch

        # 複素を real/imag の2chにするので 2*in_ch
        axis_ch = in_ch * 2

        self.kx_net = AxisDenseUNet(
            in_ch=axis_ch, base_ch=base_ch, growth=growth,
            n_layers=n_layers, out_ch=axis_ch
        )
        self.ky_net = AxisDenseUNet(
            in_ch=axis_ch, base_ch=base_ch, growth=growth,
            n_layers=n_layers, out_ch=axis_ch
        )

        self.head = nn.Sequential(
            nn.Conv2d(axis_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )

# -----------------------------
# 複素数 <-> 実2ch ユーティリティ
# -----------------------------
def complex_to_2ch(x: torch.Tensor) -> torch.Tensor:
    """
    complex [B, C, H, W] -> real [B, 2C, H, W]
    (real/imag をチャネル方向に連結)
    """
    real = x.real
    imag = x.imag
    return torch.cat([real, imag], dim=1)


def ch2_to_complex(x: torch.Tensor) -> torch.Tensor:
    """
    real [B, 2C, H, W] -> complex [B, C, H, W]
    """
    c2 = x.size(1)
    assert c2 % 2 == 0, "チャネル数は 2C である必要があります"
    c = c2 // 2
    real = x[:, :c, :, :]
    imag = x[:, c:, :, :]
    return torch.complex(real, imag)


# -----------------------------
# 「軸用」Dense Encoder–Decoder
# -----------------------------
class AxisDenseUNet(nn.Module):
    """
    軽量版 Dense Encoder–Decoder
    
    メモリ削減のため層数を減らし、チャネル数も調整
    """
    def __init__(self, in_ch, base_ch=32, growth=16, n_layers=4, out_ch=None):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch

        # Encoder: 2層（元々3層）
        self.e1 = DenseBlock(in_ch, growth, n_layers=2)  # 1層削減
        ch_e1 = in_ch + growth * 2
        self.td1 = TransitionDown(ch_e1, base_ch)

        self.e2 = DenseBlock(base_ch, growth, n_layers=2)  # 1層削減
        ch_e2 = base_ch + growth * 2
        self.td2 = TransitionDown(ch_e2, base_ch * 2)

        # Bottleneck（軽量化）
        self.bottleneck = DenseBlock(base_ch * 2, growth, n_layers=2)  # 1層削減
        ch_bn = base_ch * 2 + growth * 2

        # Decoder: 2層（元々3層）
        self.tu2 = TransitionUp(ch_bn, base_ch)
        self.d2 = DenseBlock(base_ch + ch_e2, growth, n_layers=2)
        ch_d2 = base_ch + ch_e2 + growth * 2

        self.tu1 = TransitionUp(ch_d2, base_ch)
        self.d1 = DenseBlock(base_ch + ch_e1, growth, n_layers=2)
        ch_d1 = base_ch + ch_e1 + growth * 2

        # 出力層
        self.out = nn.Conv2d(ch_d1, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.e1(x)
        t1 = self.td1(e1)

        e2 = self.e2(t1)
        t2 = self.td2(e2)

        b = self.bottleneck(t2)

        # Decoder
        u2 = self.tu2(b)
        e2_c = center_crop_to(e2, u2)
        d2 = self.d2(torch.cat([u2, e2_c], dim=1))

        u1 = self.tu1(d2)
        e1_c = center_crop_to(e1, u1)
        d1 = self.d1(torch.cat([u1, e1_c], dim=1))

        out = self.out(d1)
        return out

class SeparableIFFTStripesNet(nn.Module):
    """
    入力: complex k-space [B, C, Ky, Kx]  (C = #coils)

    処理:
      1) Learned 1D IFFT (kx)   : K --L-IFFT(kx)--> x
      2) kx-Net (AxisDenseUNet) : x --(kx-Net)--> x'
      3) Learned 1D IFFT (ky)   : x' --L-IFFT(ky)--> y
      4) ky-Net (AxisDenseUNet) : y --(ky-Net)--> y'
      5) Conv head              : y' → 1ch 画像（0–1）

    ※ パターンB: IFFT を学習可能な線形変換 (LearnableIFFT1D) に置き換え、
       初期値として物理 IFFT を使う。
    """

    def __init__(
        self,
        in_ch: int,   # #coils
        ky_len: int,  # Ky (例: 640)
        kx_len: int,  # Kx (例: 320)
        base_ch: int = 32,
        growth: int = 16,
        n_layers: int = 4,
        ifft_rank: int = 8,  # LearnableIFFT の補正ランク
        amp_eps: float = 0.1,
        use_hermitian: bool = True,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.ky_len = ky_len
        self.kx_len = kx_len

        # 複素を real/imag の2chにするので 2*in_ch
        axis_ch = in_ch * 2

        # --- Learnable IFFT 層（高速化版）--- #
        # 補正ランクを指定して低ランク分解を使用
        # kx段では通常 Hermitian OFF、ky段で ON を推奨
        self.ifft_kx = LearnableIFFT1D(
            N=kx_len, dim=-1, rank=ifft_rank,
            amp_eps=amp_eps, use_hermitian=False,
        )
        self.ifft_ky = LearnableIFFT1D(
            N=ky_len, dim=-2, rank=ifft_rank,
            amp_eps=amp_eps, use_hermitian=use_hermitian,
        )

        # --- 軸ごとの Dense UNet --- #
        self.kx_net = AxisDenseUNet(
            in_ch=axis_ch, base_ch=base_ch, growth=growth,
            n_layers=n_layers, out_ch=axis_ch
        )
        self.ky_net = AxisDenseUNet(
            in_ch=axis_ch, base_ch=base_ch, growth=growth,
            n_layers=n_layers, out_ch=axis_ch
        )

        # --- coil + real/imag 情報を 1ch 画像に落とすヘッド --- #
        self.head = nn.Sequential(
            nn.Conv2d(axis_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, kernel_size=1),
            nn.Sigmoid()  # 0–1 出力（推論時に GT の max でスケーリングする）
        )

    def forward(
        self,
        K: torch.Tensor,
        return_intermediates: bool = False,
        return_diagnostics: bool = False,
    ):
        """
        K: complex tensor [B, C, Ky, Kx]
        return_intermediates=True のとき、中間空間も返す
        """
        assert torch.is_complex(K), "入力Kは complex 型である必要があります"

        B, C, Ky, Kx = K.shape
        assert C == self.in_ch, f"expected {self.in_ch} coils, got {C}"
        assert Ky == self.ky_len and Kx == self.kx_len, \
            f"expected (Ky,Kx)=({self.ky_len},{self.kx_len}), got ({Ky},{Kx})"

        # --- (1) kx 方向 Learnable IFFT --- #
        # ここが既存版の torch.fft.ifft(K, dim=-1) の置き換え
        x = self.ifft_kx(K)                    # [B,C,Ky,Kx] complex

        # Real/Imag に分解して軸ネットへ
        x2 = complex_to_2ch(x)                 # [B,2C,Ky,Kx] real
        x2p = self.kx_net(x2)                  # [B,2C,Ky,Kx]
        xp = ch2_to_complex(x2p)               # [B,C,Ky,Kx] complex

        # --- (2) ky 方向 Learnable IFFT --- #
        y = self.ifft_ky(xp)                   # [B,C,Ky,Kx] complex

        y2 = complex_to_2ch(y)                 # [B,2C,Ky,Kx]
        y2p = self.ky_net(y2)                  # [B,2C,Ky,Kx]

        # --- (3) 実画像へ (coil 結合も含む) --- #
        img_hat = self.head(y2p)               # [B,1,Ky,Kx]（あとでリサイズして 256x128 に合わせる）

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                "kx": self.ifft_kx.diagnostics(x),
                "ky": self.ifft_ky.diagnostics(y),
            }

        if return_intermediates and return_diagnostics:
            hybrid_kx   = x
            hybrid_kx_p = xp
            img_ky      = y
            return img_hat, hybrid_kx, hybrid_kx_p, img_ky, diagnostics

        if return_intermediates:
            # 中間も可視化用に返す（magnitude にするのは外でやる前提でもOK）
            hybrid_kx   = x          # kx IFFT（learned）後 [B,C,Ky,Kx]
            hybrid_kx_p = xp         # kx-Net 後 [B,C,Ky,Kx]
            img_ky      = y          # ky IFFT（learned）後 [B,C,Ky,Kx]
            return img_hat, hybrid_kx, hybrid_kx_p, img_ky

        if return_diagnostics:
            return img_hat, diagnostics

        return img_hat

# # -----------------------------
# # 本体: Separable IFFT + Axis-wise Dense UNet
# # -----------------------------
# class SeparableIFFTStripesNet(nn.Module):
#     """
#     入力: complex k-space [B, C, Ky, Kx]  (C = #coils)
#     処理:
#       K --ifft(kx)--> x --(kx-Net)--> x'
#         --ifft(ky)--> y --(ky-Net)--> y'
#         --Conv--> I_hat

#     出力: I_hat [B, 1, H, W] (0–1に正規化)
#     """

#     def __init__(self, in_ch: int, base_ch: int = 32, growth: int = 16, n_layers: int = 4):
#         super().__init__()
#         # 複素を real/imag の2chにするので 2*in_ch
#         axis_ch = in_ch * 2

#         self.kx_net = AxisDenseUNet(
#             in_ch=axis_ch, base_ch=base_ch, growth=growth,
#             n_layers=n_layers, out_ch=axis_ch
#         )
#         self.ky_net = AxisDenseUNet(
#             in_ch=axis_ch, base_ch=base_ch, growth=growth,
#             n_layers=n_layers, out_ch=axis_ch
#         )

#         # coil + real/imag 情報を 1ch 画像に落とすヘッド
#         self.head = nn.Sequential(
#             nn.Conv2d(axis_ch, base_ch, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(base_ch, 1, kernel_size=1),
#             nn.Sigmoid()  # 0–1 出力
#         )

#     def forward(self, K: torch.Tensor, return_intermediates: bool = False):
#         """
#         K: complex tensor [B, C, Ky, Kx]
#         return_intermediates=True のとき、中間空間も返す
#         """
#         assert torch.is_complex(K), "入力Kは complex 型である必要があります"

#         # --- (1) kx 方向 1D IFFT --- #
#         x = torch.fft.ifft(K, dim=-1)            # [B,C,H,W] complex

#         x2 = complex_to_2ch(x)                   # [B,2C,H,W] real
#         x2p = self.kx_net(x2)                    # [B,2C,H,W]
#         xp = ch2_to_complex(x2p)                 # [B,C,H,W] complex

#         # --- (2) ky 方向 1D IFFT --- #
#         y = torch.fft.ifft(xp, dim=-2)           # [B,C,H,W] complex

#         y2 = complex_to_2ch(y)                   # [B,2C,H,W]
#         y2p = self.ky_net(y2)                    # [B,2C,H,W]

#         # --- (3) 実画像へ (coil 結合も含む) --- #
#         img_hat = self.head(y2p)                 # [B,1,H,W], 0–1

#         if return_intermediates:
#             # 可視化用に magnitude を返すと扱いやすい
#             hybrid_kx   = x.abs()    # kx IFFT 後 [B,C,H,W]
#             hybrid_kx_p = xp.abs()   # kx-Net 後 [B,C,H,W]
#             img_ky      = y.abs()    # ky IFFT 後 [B,C,H,W]
#             return img_hat, hybrid_kx, hybrid_kx_p, img_ky

#         return img_hat
