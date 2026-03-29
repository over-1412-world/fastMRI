# """
# MRIのk-spaceの縦方向からどのラインを取得するかを決めるためのサンプリングパターン生成関数群

# """

import numpy as np

def central_lines(Ky: int, S: int) -> np.ndarray:
    """中心から連続S本を選ぶ（端はクリップ）"""
    c = Ky // 2
    half = S // 2
    start = c - half
    lines = np.arange(start, start + S)
    return np.clip(lines, 0, Ky - 1).astype(int)

def uniform_lines(Ky: int, S: int) -> np.ndarray:
    """0..Ky-1 を等間隔で S 本"""
    return np.linspace(0, Ky - 1, S, dtype=int)

def mixed_center_peripheral(Ky: int, S: int, frac_center: float = 0.5) -> np.ndarray:
    """
    中心連続Sc本 + 周辺からSp本を均等抽出（中心帯を少し避ける）
    - Ky: ky次元長
    - S : 総本数
    - frac_center: 中心に割く比率（0〜1）
    """
    S  = int(S)
    Sc = max(1, int(round(S * frac_center)))
    Sp = max(0, S - Sc)

    center = central_lines(Ky, Sc)

    if Sp > 0:
        # 中心±margin を避けて周辺候補を作る（例：±Ky/10）
        margin = max(2, Ky // 10)
        left  = np.arange(0, max(0, Ky // 2 - margin))
        right = np.arange(min(Ky, Ky // 2 + margin), Ky)
        periph_candidates = np.concatenate([left, right])
        if periph_candidates.size == 0:
            periph = np.array([], dtype=int)
        else:
            idx = np.linspace(0, periph_candidates.size - 1, Sp, dtype=int)
            periph = periph_candidates[idx]
    else:
        periph = np.array([], dtype=int)

    return np.unique(np.concatenate([center, periph])).astype(int)

def variable_density_random(Ky, S, center_frac=0.12, seed=0):
    """
    中心は密、端は疎になるよう確率重みをつけてサンプリング。
    center_frac: 中心に重みを寄せる幅（0..1）。0.12 くらいから。
    """
    rng = np.random.default_rng(seed)
    ky = np.arange(Ky)
    center = Ky // 2
    # 中心で重み最大のガウス分布的ウェイト
    sigma = (center_frac * Ky) / 2.355  # FWHM→σ
    w = np.exp(-0.5 * ((ky - center) / (sigma + 1e-8)) ** 2)
    w = w / w.sum()
    lines = rng.choice(ky, size=S, replace=False, p=w)
    return np.sort(lines.astype(int))

def accel_to_S(Ky, R, min_center=8):
    """
    加速度R(=何本に1本取るか)から、取得本数Sを見積もる。
    min_center: 低周波の最低確保ライン数（Rに関わらず確保したい時に使用）
    """
    S = max(min_center, int(np.round(Ky / R)))
    return min(S, Ky)

def add_conjugates(Ky, lines):
    """ ky と (-ky)≡(Ky-ky) を両方含むように拡張 """
    lines = np.array(lines, dtype=int)
    conj = (-lines) % Ky
    all_lines = np.unique(np.concatenate([lines, conj]))
    return np.sort(all_lines)

def mask_from_lines(Ky, lines):
    m = np.zeros(Ky, dtype=bool)
    m[np.array(lines, dtype=int)] = True
    return m


# import numpy as np

# def central_lines(Ky, S):
#     """
#     Ky: k-spaceの縦方向サイズ
#     S: 取得するライン数 (偶数)
#     戻り値: 取得する ky インデックスのリスト
#     k-space の中央 S ラインを取得する
#     """
#     c = Ky //2   #全体の真ん中
#     half = S //2    #取りたい本数の半分
#     lines = np.arange(c - half, c - half + S)

#     return np.clip(lines, 0, Ky -1)   #インデックスは0からなので、Ky-1

# #k-space全体の範囲から、均等間隔にS本のラインを抽出
# def uniform_lins(Ky, S):
#     return np.linspace(0, Ky-1, S, dtype=int)

# def mixed_lines(Ky, S, frac_center = 0.5):
#     Sc = max(1, int(S * frac_center))  # 中央からとる本数(低周波)、最低1本
#     Sp = S - Sc                     # 周辺からとる本数（高周波）
#     center = central_lines(Ky, Sc)
#     #周辺から均等に抽出
#     #[1:-1]で両端を除く    
#     periph = np.linspace(0, Ky-1, Sp+2, dtype=int)[1:-1] if Sp >0 else np.array([], dtype=int)
    
#     return np.unique(np.concatenate([center, periph]))  #両端を除く
