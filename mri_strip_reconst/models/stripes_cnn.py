# mri_strip_reconst/models/stripes_cnn.py
"""
Dense Encoder–Decoder CNN for MRI Strip Reconstruction
based on DDNet structure from 門田遼 (卒業論文, 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Dense Block
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
                    nn.BatchNorm2d(growth_rate),
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


# -----------------------------
# Transition Down (Encoder side)
# -----------------------------
class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


# -----------------------------
# Transition Up (Decoder side)
# -----------------------------
class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

    def forward(self, x):
        return self.deconv(x)


# -----------------------------
# Main DDNet-like Model
# -----------------------------
class StripesReconstNet(nn.Module):
    """
    MRI版 Dense Encoder–Decoder
    入力: ストリップ画像群 (N, C, H, W)
    出力: 再構成画像 (N, 1, H, W)
    """

    def __init__(self, in_ch=4, base_ch=32, growth=16):
        """
        Args:
            in_ch: 入力ストリップ数 (kyラインセット数)
            base_ch: ベースチャネル数
            growth: Denseブロックの成長率
        """
        super().__init__()

        # Encoder
        self.e1 = DenseBlock(in_ch, growth, n_layers=4)
        ch_e1 = in_ch + growth * 4
        self.td1 = TransitionDown(ch_e1, base_ch)

        self.e2 = DenseBlock(base_ch, growth, n_layers=4)
        ch_e2 = base_ch + growth * 4
        self.td2 = TransitionDown(ch_e2, base_ch * 2)

        self.e3 = DenseBlock(base_ch * 2, growth, n_layers=4)
        ch_e3 = base_ch * 2 + growth * 4
        self.td3 = TransitionDown(ch_e3, base_ch * 4)

        # Bottleneck
        self.bottleneck = DenseBlock(base_ch * 4, growth, n_layers=4)
        ch_bn = base_ch * 4 + growth * 4

        # Decoder
        self.tu3 = TransitionUp(ch_bn, base_ch * 4)
        self.d3 = DenseBlock(base_ch * 4 + ch_e3, growth, n_layers=4)
        ch_d3 = base_ch * 4 + ch_e3 + growth * 4

        self.tu2 = TransitionUp(ch_d3, base_ch * 2)
        self.d2 = DenseBlock(base_ch * 2 + ch_e2, growth, n_layers=4)
        ch_d2 = base_ch * 2 + ch_e2 + growth * 4

        self.tu1 = TransitionUp(ch_d2, base_ch)
        self.d1 = DenseBlock(base_ch + ch_e1, growth, n_layers=4)
        ch_d1 = base_ch + ch_e1 + growth * 4

        # Output layer
        self.out = nn.Sequential(
            nn.Conv2d(ch_d1, 1, kernel_size=1),
            nn.Sigmoid()  # 正規化出力 (0–1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        t1 = self.td1(e1)

        e2 = self.e2(t1)
        t2 = self.td2(e2)

        e3 = self.e3(t2)
        t3 = self.td3(e3)

        # Bottleneck
        b = self.bottleneck(t3)

        # Decoder with skip connections
        u3 = self.tu3(b)
        d3 = self.d3(torch.cat([u3, e3], dim=1))

        u2 = self.tu2(d3)
        d2 = self.d2(torch.cat([u2, e2], dim=1))

        u1 = self.tu1(d2)
        d1 = self.d1(torch.cat([u1, e1], dim=1))

        return self.out(d1)


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    model = StripesReconstNet(in_ch=4, base_ch=32, growth=16)
    x = torch.randn(1, 4, 256, 256)
    y = model(x)
    print(f"input {x.shape} → output {y.shape}")
