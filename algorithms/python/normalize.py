# -*- coding: utf-8 -*-
"""
数据标准化 / 归一化 — Python 实现
Min-Max / Z-score / Robust / Decimal / 向量归一化
"""

import numpy as np


def minmax_scale(data, feature_range=(0, 1)):
    """Min-Max 归一化"""
    X = np.asarray(data, dtype=float)
    data_min = np.min(X, axis=0)
    data_max = np.max(X, axis=0)
    data_range = data_max - data_min
    data_range[data_range == 0] = 1e-10
    scale = (feature_range[1] - feature_range[0]) / data_range
    return X * scale + (feature_range[0] - data_min * scale)


def zscore_scale(data):
    """Z-score 标准化（均值为 0，标准差为 1）"""
    X = np.asarray(data, dtype=float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    std[std == 0] = 1e-10
    return (X - mean) / std


def robust_scale(data):
    """Robust 缩放（中位数 + IQR，对异常值鲁棒）"""
    X = np.asarray(data, dtype=float)
    median = np.median(X, axis=0)
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1e-10
    return (X - median) / iqr


def vector_normalize(data):
    """向量归一化（TOPSIS 专用，除以 L2 范数）"""
    X = np.asarray(data, dtype=float)
    norm = np.sqrt(np.sum(X ** 2, axis=0))
    norm[norm == 0] = 1e-10
    return X / norm


def decimal_scaling(data):
    """小数定标标准化（除以 10 的幂次使所有值在 [-1, 1]）"""
    X = np.asarray(data, dtype=float)
    max_abs = np.max(np.abs(X), axis=0)
    powers = np.ceil(np.log10(max_abs + 1e-10))
    divisors = 10 ** powers
    return X / divisors


def handle_missing(df, strategy='median', columns=None):
    """缺失值处理"""
    import pandas as pd
    df_filled = df.copy()
    cols = columns or df.columns
    for col in cols:
        if df[col].isnull().sum() == 0:
            continue
        if strategy == 'mean':
            df_filled[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df_filled[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df_filled[col] = df[col].fillna(df[col].mode()[0])
        elif strategy == 'interpolate':
            df_filled[col] = df[col].interpolate(method='linear')
        elif strategy == 'forward':
            df_filled[col] = df[col].fillna(method='ffill')
    return df_filled


def outlier_detect(data, method='iqr'):
    """异常值检测"""
    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, m = X.shape
    mask = np.zeros((n, m), dtype=bool)

    for j in range(m):
        col = X[:, j]
        if method == 'iqr':
            q1, q3 = np.percentile(col, 25), np.percentile(col, 75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask[:, j] = (col < lower) | (col > upper)
        elif method == 'zscore':
            z = np.abs(zscore_scale(col))
            mask[:, j] = z > 3

    return mask


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    data = np.array([[1, 10, 100],
                     [2, 20, 200],
                     [3, 30, 300]], dtype=float)

    print("=" * 60)
    print("数据标准化方法对比")
    print("=" * 60)

    print(f"\n原始数据:\n{data}")

    print(f"\nMin-Max [0,1]:\n{minmax_scale(data)}")
    print(f"\nZ-score:\n{zscore_scale(data)}")
    print(f"\nRobust (中位数+IQR):\n{robust_scale(data)}")
    print(f"\n向量归一化 (L2):\n{vector_normalize(data)}")
    print(f"\n小数定标标准化:\n{decimal_scaling(data)}")

    # 使用方法选择建议
    print("\n--- 选择建议 ---")
    print("  数据无异常值, 需比较大小      → Min-Max")
    print("  数据有异常值, 需正态分布      → Z-score (先去异常)")
    print("  数据有异常值, 不想去异常      → Robust Scaler")
    print("  TOPSIS 评价模型              → 向量归一化 (L2)")
