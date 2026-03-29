import os, sys
from pathlib import Path

# Ensure workspace root is on sys.path to import tests.create_temp_data
WS_ROOT = Path(__file__).resolve().parents[2]  # c:\Users\s2520\fastMRI
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

try:
    from tests.create_temp_data import create_temp_data
except Exception as e:
    print("Failed to import create_temp_data from tests:", e)
    raise

out_root = WS_ROOT / "mri_strip_reconst" / "data" / "synthetic"
out_root.mkdir(parents=True, exist_ok=True)

knee_root, brain_root, meta = create_temp_data(out_root)
print("Synthetic brain data created at:", brain_root)
print("Train glob:", str(brain_root / "multicoil_train" / "*.h5"))
print("Val   glob:", str(brain_root / "multicoil_val" / "*.h5"))
