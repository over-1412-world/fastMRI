import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# interactive 3D
import pandas as pd
import plotly.express as px

# ================================
# 0. 設定
# ================================
data_glob = r"C:/Users/s2520/data/fastmri/brain_multicoil/train/multicoil_train_batch_0/multicoil_train/*.h5"
output_dir = "outputs/Image_recognition_report3"
os.makedirs(output_dir, exist_ok=True)
print("保存先:", os.path.abspath(output_dir))

# クラス定義：中心 vs 周辺
CENTER_RANGE = (0.40, 0.60)   # 中央40-60%
EDGE_RANGES  = [(0.00, 0.20), (0.80, 1.00)]  # 端0-20%, 80-100%

# PCAの次元
SUBSPACE_DIM = 3

# メモリが厳しければ制限
max_files = None  # 例: 50

# ================================
# 1. データ読み込み（reconstruction_rss）
# ================================
h5_files = glob.glob(data_glob)
if max_files is not None:
    h5_files = h5_files[:max_files]
if len(h5_files) == 0:
    raise FileNotFoundError("h5ファイルが見つかりません")

print("見つかったファイル数:", len(h5_files))

target_shape = None
X_center_list = []
X_edge_list = []
used_files = 0

def normalize01(vol):
    vol = vol.astype(np.float32)
    vol = vol - vol.min()
    vol = vol / (vol.max() + 1e-8)
    return vol

def slice_indices(T, a, b):
    s = int(np.floor(T * a))
    e = int(np.ceil (T * b))
    s = max(s, 0); e = min(e, T)
    return np.arange(s, e)

for path in h5_files:
    with h5py.File(path, "r") as f:
        if "reconstruction_rss" not in f:
            continue
        vol = f["reconstruction_rss"][()]  # (T,H,W)

    vol = normalize01(vol)
    T, H, W = vol.shape

    if target_shape is None:
        target_shape = (H, W)
        print("基準サイズ:", target_shape)

    if (H, W) != target_shape:
        print("[SKIP size]", path, (H, W))
        continue

    # 中心スライス
    idx_c = slice_indices(T, *CENTER_RANGE)

    # 周辺スライス（2つの範囲を結合）
    idx_e = []
    for a, b in EDGE_RANGES:
        idx_e.append(slice_indices(T, a, b))
    idx_e = np.unique(np.concatenate(idx_e))

    # flattenして積む（各スライスを1サンプルとする）
    X_center_list.append(vol[idx_c].reshape(len(idx_c), -1))
    X_edge_list.append(vol[idx_e].reshape(len(idx_e), -1))
    used_files += 1

X1 = np.concatenate(X_center_list, axis=0)  # Class1
X2 = np.concatenate(X_edge_list, axis=0)    # Class2

print("\n===== 使用データ集計 =====")
print("採用ボリューム数:", used_files)
print("Class1(中心) サンプル数:", X1.shape[0])
print("Class2(周辺) サンプル数:", X2.shape[0])
print("次元:", X1.shape[1])
print("=========================\n")

# ================================
# 2. S1, S2（各クラスの3次元部分空間）
#   PCAの成分（基底）を取る：U1, U2 ∈ R^{D×3}
# ================================
pca1 = PCA(n_components=SUBSPACE_DIM, svd_solver="randomized")
pca2 = PCA(n_components=SUBSPACE_DIM, svd_solver="randomized")

pca1.fit(X1)
pca2.fit(X2)

U1 = pca1.components_.T  # (D,3)
U2 = pca2.components_.T  # (D,3)

# ================================
# 3. 正準角（principal/canonical angles）
#    cos(theta_i) = singular values of U1^T U2
# ================================
M = U1.T @ U2
svals = np.linalg.svd(M, compute_uv=False)
svals = np.clip(svals, -1.0, 1.0)
thetas = np.arccos(svals)  # rad
thetas_deg = thetas * 180.0 / np.pi

print("=== 正準角（deg）===")
for i, t in enumerate(thetas_deg, 1):
    print(f"theta{i}: {t:.3f} deg")

# 保存（テキスト）
with open(os.path.join(output_dir, "canonical_angles.txt"), "w", encoding="utf-8") as f:
    f.write("Canonical angles (degrees)\n")
    for i, t in enumerate(thetas_deg, 1):
        f.write(f"theta{i}: {t:.6f}\n")

# ================================
# 4. 差分部分空間（difference subspace）※省メモリ版
#   D×Dの射影行列 (U U^T) を作らずに計算する
#   (I - U2U2^T)U1 = U1 - U2(U2^T U1)
#   (I - U1U1^T)U2 = U2 - U1(U1^T U2)
# ================================
# U1, U2: (D,3)

# 小さい行列（3x3）だけ作る
A21 = U2.T @ U1  # (3,3)
A12 = U1.T @ U2  # (3,3)

# 差分成分（D,3）
D1 = U1 - U2 @ A21
D2 = U2 - U1 @ A12

# 2つを結合して (D,6)
B = np.concatenate([D1, D2], axis=1).astype(np.float32)

# 直交化 → 上位3次元を差分部分空間基底に
# BはD×6で巨大ではない（102400×6）のでOK
Q, _ = np.linalg.qr(B)         # Q: (D,6)
Ud = Q[:, :SUBSPACE_DIM]       # (D,3)


# ================================
# 5. 差分部分空間の基底ベクトルを画像化（3枚）
# ================================
H, W = target_shape
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i in range(3):
    img = Ud[:, i].reshape(H, W)
    # 見やすいように正規化（符号は任意なので絶対値ではなくmin-max）
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(f"Diff basis {i+1}")
    axes[i].axis("off")
plt.tight_layout()
save_diff_basis = os.path.join(output_dir, "diff_subspace_basis_images.png")
fig.savefig(save_diff_basis, dpi=300)
plt.show()
plt.close(fig)
print("[保存] 差分部分空間 基底画像:", save_diff_basis)

# ================================
# 6. 各クラスを差分部分空間へ射影（3次元）して3Dプロット
# ================================
Z1 = X1 @ Ud  # (N1,3)
Z2 = X2 @ Ud  # (N2,3)

# --- matplotlib PNG（固定視点） ---
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(Z1[:,0], Z1[:,1], Z1[:,2], s=6, alpha=0.5, label="Class1(center)")
ax.scatter(Z2[:,0], Z2[:,1], Z2[:,2], s=6, alpha=0.5, label="Class2(edge)")
ax.set_xlabel("d1"); ax.set_ylabel("d2"); ax.set_zlabel("d3")
ax.set_title("Projection onto Difference Subspace (3D)")
ax.view_init(elev=20, azim=40)
ax.legend()
save_png = os.path.join(output_dir, "diff_subspace_3d.png")
fig.savefig(save_png, dpi=300)
plt.show()
plt.close(fig)
print("[保存] 3D PNG:", save_png)

# --- plotly HTML（回せる） ---
df1 = pd.DataFrame({"d1": Z1[:,0], "d2": Z1[:,1], "d3": Z1[:,2], "class": "center"})
df2 = pd.DataFrame({"d1": Z2[:,0], "d2": Z2[:,1], "d3": Z2[:,2], "class": "edge"})
df = pd.concat([df1, df2], axis=0, ignore_index=True)

fig_html = px.scatter_3d(df, x="d1", y="d2", z="d3", color="class",
                         title="Projection onto Difference Subspace (Interactive)")
save_html = os.path.join(output_dir, "diff_subspace_3d_interactive.html")
fig_html.write_html(save_html)
print("[保存] 3D interactive HTML:", save_html)

# ================================
# 7. 出力一覧
# ================================
print("\n出力フォルダ内のファイル一覧:")
for f in os.listdir(output_dir):
    print(" -", f)
