import h5py, numpy as np, os

file_path = r"C:\Users\s2520\data\fastmri\brain_multicoil\test\multicoil_test\file_brain_AXFLAIR_200_6002527.h5"
with h5py.File(file_path, "r") as f:
    k = f["kspace"][0]  # (coils, ky, kx)
    print("shape:", k.shape, "dtype:", k.dtype)
    a = np.abs(k)
    print("max|k|:", a.max(), "min|k|:", a.min())
    print("nonzero ratio:", np.count_nonzero(a)/a.size)
    # コイル別のmax
    print("per-coil max:", [np.abs(k[i]).max() for i in range(k.shape[0])])
