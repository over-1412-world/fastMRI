import torch
import fastmri.fftc as fftc

# 複素テンソルを作る
img = (torch.randn(1,1,64,64) + 1j*torch.randn(1,1,64,64)).to(torch.complex64)

# 2チャネル形式 [B,C,H,W,2] へ変換
img_ri = torch.view_as_real(img)          # 実部・虚部を分離
kspace_ri = fftc.fft2c_new(img_ri)        # FFT（2チャネル形式）
recon_ri  = fftc.ifft2c_new(kspace_ri)    # IFFT（2チャネル形式）
recon     = torch.view_as_complex(recon_ri)

print(img.shape, kspace_ri.shape, recon.shape)
print("MAE:", (img - recon).abs().mean().item())
