# -*- coding: utf-8 -*-
"""
熵权法 — Python 实现
客观定权：利用信息熵计算指标权重

用法：
    python entropy_weight.py
"""

import numpy as np


def entropy_weight(data, benefit_cols=None, cost_cols=None):
    """
    熵权法计算指标权重

    Parameters
    ----------
    data : ndarray (m, n)
        决策矩阵，m个样本，n个指标
    benefit_cols : list of int, optional
        极大型指标列（越大越好），未指定的列默认为极大型
    cost_cols : list of int, optional
        极小型指标列（越小越好）

    Returns
    -------
    weights : ndarray (n,)
        熵权
    entropy_values : ndarray (n,)
        各指标信息熵
    """
    X = np.asarray(data, dtype=float)
    m, n = X.shape

    benefit_cols = benefit_cols or list(range(n))
    cost_cols = cost_cols or []

    # 正向化：极小型 → 极大型
    X_pos = X.copy()
    for j in cost_cols:
        # 若指标值可能为0，使用 1/(1+x) 或平移
        col_max = np.max(X_pos[:, j])
        X_pos[:, j] = col_max - X_pos[:, j]

    # 负值平移（确保所有值 > 0）
    for j in range(n):
        col_min = np.min(X_pos[:, j])
        if col_min <= 0:
            X_pos[:, j] += abs(col_min) + 1e-6

    # 归一化 P_ij = x_ij / Σx_ij
    P = X_pos / (np.sum(X_pos, axis=0) + 1e-10)

    # 计算信息熵 e_j = -k * Σ(P_ij * ln(P_ij)), k = 1/ln(m)
    k = 1.0 / np.log(m)
    epsilon = 1e-10  # 避免 log(0)
    entropy_values = -k * np.sum(P * np.log(P + epsilon), axis=0)

    # 熵权 w_j = (1 - e_j) / Σ(1 - e_j)
    d = 1 - entropy_values  # 信息效用值
    weights = d / np.sum(d)

    return weights, entropy_values


def entropy_topsis(data, benefit_cols=None, cost_cols=None):
    """
    熵权TOPSIS：熵权法定权 + TOPSIS 排序（常用组合）

    Returns
    -------
    scores, ranking, weights
    """
    from topsis import topsis

    weights, entropy_vals = entropy_weight(data, benefit_cols, cost_cols)
    scores, ranking, rank_order, details = topsis(
        data, weights=weights,
        benefit_cols=benefit_cols or [],
        cost_cols=cost_cols or [],
    )
    return scores, ranking, weights, details


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    # 示例数据：5个样本，4个指标
    # 指标1-2: 极大型, 指标3: 极小型, 指标4: 极大型
    data = np.array([
        [8.0, 7.0, 2.0, 6.0],
        [6.0, 8.0, 3.0, 5.0],
        [9.0, 6.0, 1.5, 8.0],
        [7.0, 9.0, 2.5, 7.0],
        [5.0, 5.0, 4.0, 4.0],
    ])

    print("=" * 60)
    print("熵权法示例")
    print("=" * 60)
    print(f"\n原始数据 (5样本 × 4指标):\n{data}")

    weights, entropy_vals = entropy_weight(data, cost_cols=[2])

    print(f"\n信息熵值: {entropy_vals}")
    print(f"信息效用值 (1-e): {1 - entropy_vals}")
    print(f"熵权: {weights}")

    # 验证：熵权之和为1
    print(f"\n权重之和: {np.sum(weights):.4f}")

    # 解释
    print("\n--- 权重解读 ---")
    for i, (e, w) in enumerate(zip(entropy_vals, weights)):
        print(f"  指标{i+1}: 信息熵={e:.4f}, 熵权={w:.4f}")
        if w > 0.3:
            print(f"    → 该指标数据变异大，区分度高，权重较大")
        else:
            print(f"    → 该指标数据较一致，区分度低")

    # --- 熵权TOPSIS ---
    print("\n" + "=" * 60)
    print("熵权TOPSIS 组合示例")
    print("=" * 60)

    scores, ranking, w, details = entropy_topsis(data, cost_cols=[2])
    for j, rank in enumerate(ranking):
        print(f"  第{rank}名: 样本{j+1}, 贴近度={scores[j]:.4f}")
    print(f"  使用的熵权: {w}")
