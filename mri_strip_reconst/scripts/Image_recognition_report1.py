import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

# ここからインタラクティブ3D用
import pandas as pd
import plotly.express as px

# ================================
# 0. 保存フォルダ作成
# ================================
output_dir = "outputs/Image_recognition_report1"
os.makedirs(output_dir, exist_ok=True)
print(f"保存先フォルダ: {os.path.abspath(output_dir)}")

# ================================
# 1. fastMRI brain multicoil のファイル取得
# ================================
h5_files = glob.glob(
    r"C:/Users/s2520/data/fastmri/brain_multicoil/train/multicoil_train_batch_0/multicoil_train/*.h5"
)

if len(h5_files) == 0:
    raise FileNotFoundError("指定フォルダに .h5 ファイルがありません")

print(f"見つかったファイル数: {len(h5_files)}")

# メモリ制限をかけたいときは max_files = 20 などにする
max_files = None
if max_files is not None:
    h5_files = h5_files[:max_files]

# ================================
# 2. 読み込み（同一サイズのみ採用）
# ================================
all_volumes = []
used_files = 0
used_slices = 0
target_shape = None

for i, h5_path in enumerate(h5_files):
    with h5py.File(h5_path, "r") as f:
        if "reconstruction_rss" in f:
            vol = f["reconstruction_rss"][()]
        else:
            print(f"[SKIP] {h5_path}（画像なし）")
            continue

    vol = vol.astype(np.float32)
    vol = vol - vol.min()
    vol = vol / (vol.max() + 1e-8)

    # 最初のボリュームの形を基準にする
    if target_shape is None:
        _, H0, W0 = vol.shape
        target_shape = (H0, W0)
        print(f"\n基準画像サイズ: {target_shape}")

    _, H, W = vol.shape
    if (H, W) != target_shape:
        print(f"[SKIP] {h5_path} shape={(H, W)}")
        continue

    all_volumes.append(vol)
    used_files += 1
    used_slices += vol.shape[0]

    print(f"[採用] {h5_path} | slices={vol.shape[0]}")

# ================================
# 3. 結合と報告
# ================================
vol_all = np.concatenate(all_volumes, axis=0)
T_total, H, W = vol_all.shape

print("\n===== 使用データ集計 =====")
print(f"採用ボリューム数: {used_files}")
print(f"総スライス数: {used_slices}")
print(f"結合後 shape: {vol_all.shape}")
print("=============================\n")

# ================================
# 4. PCA の入力行列へ reshape
# ================================
X = vol_all.reshape(T_total, -1)
print("PCA 入力 X shape:", X.shape)

# ================================
# 5. PCA（9主成分）
# ================================
pca = PCA(n_components=9)
scores = pca.fit_transform(X)
components = pca.components_

print("寄与率:", pca.explained_variance_ratio_)

# ================================
# 6. 主成分画像（PC1〜PC9） PNG 保存
# ================================
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.ravel()

for i in range(9):
    pc_img = components[i].reshape(H, W)
    pc_img = (pc_img - pc_img.min()) / (pc_img.max() - pc_img.min() + 1e-8)

    ax = axes[i]
    ax.imshow(pc_img, cmap="gray")
    ax.set_title(f"PC{i+1}")
    ax.axis("off")

plt.suptitle("Principal Component Images (PC1–PC9)")
plt.tight_layout()

save_pc = os.path.join(output_dir, "PC1_9_images.png")
fig.savefig(save_pc, dpi=300)
print(f"[保存完了] 主成分画像 PNG: {save_pc}")

plt.show()
plt.close(fig)

# ================================
# 7. 3Dプロット PNG 保存関数
# ================================
def plot_3d_and_save_png(scores, idxs, title, filename):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        scores[:, idxs[0]],
        scores[:, idxs[1]],
        scores[:, idxs[2]],
        s=8,
        c=np.arange(scores.shape[0]),
        cmap="viridis"
    )

    ax.set_xlabel(f"PC{idxs[0] + 1}")
    ax.set_ylabel(f"PC{idxs[1] + 1}")
    ax.set_zlabel(f"PC{idxs[2] + 1}")
    ax.set_title(title)

    ax.view_init(elev=20, azim=40)

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=300)
    print(f"[保存完了] 3Dプロット PNG: {save_path}")

    plt.show()
    plt.close(fig)

# ================================
# 8. インタラクティブ3D（HTML）保存関数
# ================================
def save_interactive_3d(scores, idxs, title, filename):
    df = pd.DataFrame({
        f"PC{idxs[0]+1}": scores[:, idxs[0]],
        f"PC{idxs[1]+1}": scores[:, idxs[1]],
        f"PC{idxs[2]+1}": scores[:, idxs[2]],
        "index": np.arange(scores.shape[0])
    })

    fig = px.scatter_3d(
        df,
        x=f"PC{idxs[0]+1}",
        y=f"PC{idxs[1]+1}",
        z=f"PC{idxs[2]+1}",
        color="index",
        title=title
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=f"PC{idxs[0]+1}",
            yaxis_title=f"PC{idxs[1]+1}",
            zaxis_title=f"PC{idxs[2]+1}",
        )
    )

    save_path = os.path.join(output_dir, filename)
    fig.write_html(save_path)
    print(f"[保存完了] インタラクティブ3D HTML: {save_path}")

# ================================
# 9. 3Dプロットを PNG & HTML 両方保存
# ================================
# PC1-3
plot_3d_and_save_png(scores, (0, 1, 2), "PC1–PC3 3D Projection", "PC1_3_3D.png")
save_interactive_3d(scores, (0, 1, 2), "PC1–PC3 3D Projection", "PC1_3_3D_interactive.html")

# PC4-6
plot_3d_and_save_png(scores, (3, 4, 5), "PC4–PC6 3D Projection", "PC4_6_3D.png")
save_interactive_3d(scores, (3, 4, 5), "PC4–PC6 3D Projection", "PC4_6_3D_interactive.html")

# PC7-9
plot_3d_and_save_png(scores, (6, 7, 8), "PC7–PC9 3D Projection", "PC7_9_3D.png")
save_interactive_3d(scores, (6, 7, 8), "PC7–PC9 3D Projection", "PC7_9_3D_interactive.html")

# ================================
# 10. 出力ファイル一覧
# ================================
print("\n出力フォルダ内のファイル一覧:")
for f in os.listdir(output_dir):
    print(" -", f)
