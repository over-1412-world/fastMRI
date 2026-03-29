import torch
import sys, os
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from models.stripes_separable_ifft import SeparableIFFTStripesNet

# Load checkpoint
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(r'C:\Users\s2520\fastMRI\mri_strip_reconst\outputs\separable_ifft\separable_ifft_best.pt', map_location=device)
model_args = ckpt.get('args', {})
print('Model args:', model_args)

# Check IFFT parameters
model = SeparableIFFTStripesNet(
    in_ch=model_args.get('in_ch', 4),
    ky_len=model_args.get('ky_len', 768),
    kx_len=model_args.get('kx_len', 396),
    base_ch=32,
    growth=16,
    n_layers=4,
).to(device)
model.load_state_dict(ckpt['model'])

print(f'\n=== IFFT Parameters ===')
print(f'IFFT kx scale: {model.ifft_kx.scale.item():.6f}')
print(f'IFFT ky scale: {model.ifft_ky.scale.item():.6f}')
print(f'IFFT kx phase mean/std: {model.ifft_kx.phase.mean().item():.6f}/{model.ifft_kx.phase.std().item():.6f}')
print(f'IFFT ky phase mean/std: {model.ifft_ky.phase.mean().item():.6f}/{model.ifft_ky.phase.std().item():.6f}')

# Test forward pass with dummy data
print(f'\n=== Test Forward Pass ===')
B, C, Ky, Kx = 1, model_args.get('in_ch', 4), model_args.get('ky_len', 768), model_args.get('kx_len', 396)
dummy_k = torch.randn(B, C, Ky, Kx, dtype=torch.complex64).to(device)
with torch.no_grad():
    output = model(dummy_k)
print(f'Output shape: {output.shape}')
print(f'Output min/max/mean: {output.min().item():.6f}/{output.max().item():.6f}/{output.mean().item():.6f}')
