import h5py
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os

# インタラクティブ3D用
import pandas as pd
import plotly.express as px

# ================================
# 0. 保存フォルダ作成
# ================================
output_dir = "outputs/Image_recognition_report2"
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
# 2. 読み込み（同一サイズのみ採用）＋4クラスのラベル生成
#    クラスは「各ボリューム内のスライス位置」で4分割
#    0〜25%: class 0, 25〜50%: class 1,
#    50〜75%: class 2, 75〜100%: class 3
# ================================
all_volumes = []
all_labels = []
used_files = 0
used_slices = 0
target_shape = None

for i, h5_path in enumerate(h5_files):
    with h5py.File(h5_path, "r") as f:
        if "reconstruction_rss" in f:
            vol = f["reconstruction_rss"][()]  # shape: (T, H, W)
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

    T = vol.shape[0]

    # スライス位置に応じて 4 クラスに分割
    # idx = 0,...,T-1 に対し、class = floor(4 * idx / T)
    idx = np.arange(T)
    labels_vol = (4 * idx) // T
    labels_vol[labels_vol > 3] = 3  # 念のため保険

    all_volumes.append(vol)
    all_labels.append(labels_vol)

    used_files += 1
    used_slices += T

    print(f"[採用] {h5_path} | slices={T}")

# ================================
# 3. 結合と報告
# ================================
if len(all_volumes) == 0:
    raise RuntimeError("有効なボリュームがありませんでした。")

vol_all = np.concatenate(all_volumes, axis=0)   # (N, H, W)
labels = np.concatenate(all_labels, axis=0)     # (N,)

N_total, H, W = vol_all.shape

print("\n===== 使用データ集計 =====")
print(f"採用ボリューム数: {used_files}")
print(f"総スライス数: {used_slices}")
print(f"結合後 shape: {vol_all.shape}")
print("=============================\n")

classes = np.unique(labels)
print("クラス一覧:", classes)
if len(classes) != 4:
    print("※クラス数が4以外になっています（スライス数が極端に少ないボリュームが多い？）")

# ================================
# 4. PCA/LDA の入力行列へ reshape
# ================================
X = vol_all.reshape(N_total, -1).astype(np.float32)  # (N, D)
N, D = X.shape
print("PCA/LDA 入力 X shape:", X.shape)

IMAGE_H, IMAGE_W = H, W

# ================================
# 5. PCA（主成分分析）
#    ・高次元なのでランク落ち対策として PCA→LDA のパイプラインを採用
#    ・n_components は LDA 用に 50 次元まで圧縮（＋PC1〜3はここから利用）
# ================================
N_PCA_FOR_LDA = 50  # LDA に渡す PCA 次元数（必要に応じて調整）

n_pca = max(3, N_PCA_FOR_LDA)
pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=0)
scores_pca = pca.fit_transform(X)  # (N, n_pca)
components = pca.components_       # (n_pca, D)

print("PCA 寄与率（先頭10成分）:", pca.explained_variance_ratio_[:10])

# ================================
# 6. 3D プロット（PNG）汎用関数
# ================================
def plot_3d_and_save_png(X_3d, labels, axis_labels, title, filename):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    for cls in np.unique(labels):
        idx = (labels == cls)
        ax.scatter(
            X_3d[idx, 0],
            X_3d[idx, 1],
            X_3d[idx, 2],
            s=8,
            alpha=0.6,
            label=f"Class {cls}",
        )

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(title)
    ax.legend()

    ax.view_init(elev=20, azim=40)

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=300)
    print(f"[保存完了] 3Dプロット PNG: {save_path}")

    plt.close(fig)

# ================================
# 7. インタラクティブ 3D（HTML）汎用関数
# ================================
def save_interactive_3d(X_3d, labels, axis_labels, title, filename):
    """
    X_3d: shape (N, 3)
    labels: shape (N,)
    """
    df = pd.DataFrame({
        axis_labels[0]: X_3d[:, 0],
        axis_labels[1]: X_3d[:, 1],
        axis_labels[2]: X_3d[:, 2],
        "class": labels.astype(int),
    })

    fig = px.scatter_3d(
        df,
        x=axis_labels[0],
        y=axis_labels[1],
        z=axis_labels[2],
        color="class",
        title=title,
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
        )
    )

    save_path = os.path.join(output_dir, filename)
    fig.write_html(save_path)
    print(f"[保存完了] インタラクティブ3D HTML: {save_path}")

# ================================
# 8. PCA 3D プロット（設問3）PNG & HTML
# ================================
scores_pca_3d = scores_pca[:, :3]

# PNG
plot_3d_and_save_png(
    scores_pca_3d,
    labels,
    axis_labels=("PC1", "PC2", "PC3"),
    title="PCA 3D Projection (PC1–PC3)",
    filename="PCA_PC1_3_3D.png",
)

# HTML（ぐるぐる）
save_interactive_3d(
    scores_pca_3d,
    labels,
    axis_labels=("PC1", "PC2", "PC3"),
    title="PCA 3D Projection (PC1–PC3)",
    filename="PCA_PC1_3_3D_interactive.html",
)

# ================================
# 9. PCA 基底ベクトル（PC1〜PC3）を画像化（設問4）
# ================================
def save_basis_images_from_components(components, prefix, image_h, image_w, out_dir):
    K = min(3, components.shape[0])  # 先頭3成分
    for i in range(K):
        vec = components[i]          # (D,)
        img = vec.reshape(image_h, image_w)

        # 0-1 に正規化して表示しやすくする
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img_norm = (img - vmin) / (vmax - vmin)
        else:
            img_norm = np.zeros_like(img)

        plt.imshow(img_norm, cmap="gray")
        plt.colorbar()
        plt.title(f"{prefix} basis {i+1}")
        save_path = os.path.join(out_dir, f"{prefix}_basis_{i+1}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print("保存:", save_path)

# PCA 基底 (PC1〜PC3)
save_basis_images_from_components(
    components,
    prefix="PCA",
    image_h=IMAGE_H,
    image_w=IMAGE_W,
    out_dir=output_dir,
)

# ================================
# 10. LDA（重判別分析）
#     PCA で N_PCA_FOR_LDA 次元に圧縮した特徴に対して LDA
#     → ランク落ち防止の Fisherface 的アプローチ
# ================================
X_pca_for_lda = scores_pca[:, :N_PCA_FOR_LDA]  # (N, N_PCA_FOR_LDA)

lda = LinearDiscriminantAnalysis(n_components=3)
scores_lda = lda.fit_transform(X_pca_for_lda, labels)  # (N, 3)

print("LDA n_components:", lda.n_components)

# ================================
# 11. LDA 3D プロット（設問1）PNG & HTML
# ================================
# PNG
plot_3d_and_save_png(
    scores_lda,
    labels,
    axis_labels=("LD1", "LD2", "LD3"),
    title="LDA 3D Projection (LD1–LD3)",
    filename="LDA_LD1_3_3D.png",
)

# HTML（ぐるぐる）
save_interactive_3d(
    scores_lda,
    labels,
    axis_labels=("LD1", "LD2", "LD3"),
    title="LDA 3D Projection (LD1–LD3)",
    filename="LDA_LD1_3_3D_interactive.html",
)

# ================================
# 12. LDA 判別ベクトルを元の画像空間に戻して可視化（設問2）
#     Fisherface と同様:
#         B_pixel = P^T @ W
#       P: PCA の基底 (N_PCA_FOR_LDA, D)
#       W: LDA の基底 (N_PCA_FOR_LDA, 3)
# ================================
# LDA が学習した基底（PCA空間内）
W_pca = lda.scalings_[:, :3]  # shape: (N_PCA_FOR_LDA, 3)

# PCA の基底（LDA に使った分だけ）
P_for_lda = components[:N_PCA_FOR_LDA, :]  # (N_PCA_FOR_LDA, D)

# 元のピクセル空間での判別ベクトル
B_pixel = P_for_lda.T @ W_pca  # (D, 3)

# 画像として保存
for i in range(3):
    vec = B_pixel[:, i]
    img = vec.reshape(IMAGE_H, IMAGE_W)

    vmin, vmax = img.min(), img.max()
    if vmax > vmin:
        img_norm = (img - vmin) / (vmax - vmin)
    else:
        img_norm = np.zeros_like(img)

    plt.imshow(img_norm, cmap="gray")
    plt.colorbar()
    plt.title(f"LDA basis {i+1} (mapped to pixel space)")
    save_path = os.path.join(output_dir, f"LDA_basis_{i+1}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print("保存:", save_path)

# ================================
# 13. 出力ファイル一覧
# ================================
print("\n出力フォルダ内のファイル一覧:")
for f in os.listdir(output_dir):
    print(" -", f)

print("\n=== レポート2用の PCA / LDA 図と 3Dぐるぐる HTML の生成が完了しました ===")
