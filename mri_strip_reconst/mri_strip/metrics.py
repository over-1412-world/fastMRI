import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

#PSNR(ピーク信号対雑音比)とSSIM(構造類似度指数)を計算する関数
def psnr(gt, pred):
    # 想定: 0-1 正規化
    return peak_signal_noise_ratio(gt, pred, data_range=1.0)

def ssim(gt, pred):
    return structural_similarity(gt, pred, data_range=1.0)
