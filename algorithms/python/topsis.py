# -*- coding: utf-8 -*-
"""
TOPSIS 法（优劣解距离法）— Python 实现
支持多种指标正向化 + 加权标准化

用法：
    python topsis.py
"""

import numpy as np


def topsis(data, weights=None, benefit_cols=None, cost_cols=None,
           target_cols=None, target_values=None,
           interval_cols=None, interval_bounds=None):
    """
    TOPSIS 综合评价

    Parameters
    ----------
    data : ndarray (m, n)
        决策矩阵，m 个方案，n 个指标
    weights : ndarray (n,), optional
        指标权重，默认等权重
    benefit_cols : list of int
        极大型（越大越好）指标列索引
    cost_cols : list of int
        极小型（越小越好）指标列索引
    target_cols : list of int
        中间型指标列索引
    target_values : list of float
        中间型指标的最佳值
    interval_cols : list of int
        区间型指标列索引
    interval_bounds : list of tuple (low, high)
        区间型指标的最佳区间

    Returns
    -------
    scores : ndarray (m,)
        相对贴近度 C_i（0~1，越大越优）
    ranking : ndarray (m,)
        排序（1 为最优）
    details : dict
        中间计算结果
    """
    X = np.asarray(data, dtype=float)
    m, n = X.shape

    benefit_cols = benefit_cols or []
    cost_cols = cost_cols or []
    target_cols = target_cols or []
    target_values = target_values or []
    interval_cols = interval_cols or []
    interval_bounds = interval_bounds or []

    # Step 1: 正向化
    X_pos = X.copy()

    # 极小型 → 极大型
    for j in cost_cols:
        X_pos[:, j] = np.max(X_pos[:, j]) - X_pos[:, j]

    # 中间型 → 极大型
    for idx, j in enumerate(target_cols):
        best = target_values[idx]
        M = np.max(np.abs(X_pos[:, j] - best))
        if M > 0:
            X_pos[:, j] = 1 - np.abs(X_pos[:, j] - best) / M

    # 区间型 → 极大型
    for idx, j in enumerate(interval_cols):
        low, high = interval_bounds[idx]
        M = max(low - np.min(X_pos[:, j]), np.max(X_pos[:, j]) - high)
        if M > 0:
            for i in range(m):
                if X_pos[i, j] < low:
                    X_pos[i, j] = 1 - (low - X_pos[i, j]) / M
                elif X_pos[i, j] > high:
                    X_pos[i, j] = 1 - (X_pos[i, j] - high) / M
                else:
                    X_pos[i, j] = 1

    # Step 2: 向量归一化 (SSQ 标准化)
    Z = X_pos / np.sqrt(np.sum(X_pos ** 2, axis=0) + 1e-10)

    # Step 3: 加权
    if weights is None:
        weights = np.ones(n) / n
    weights = np.asarray(weights, dtype=float)
    Z_weighted = Z * weights

    # Step 4: 正负理想解
    Z_plus = np.max(Z_weighted, axis=0)
    Z_minus = np.min(Z_weighted, axis=0)

    # Step 5: 计算距离（欧氏距离）
    D_plus = np.sqrt(np.sum((Z_weighted - Z_plus) ** 2, axis=1))
    D_minus = np.sqrt(np.sum((Z_weighted - Z_minus) ** 2, axis=1))

    # Step 6: 贴近度
    scores = D_minus / (D_plus + D_minus + 1e-10)

    # 排序（大到小）
    ranking = np.argsort(-scores) + 1
    rank_order = np.argsort(-scores)

    details = {
        'X_positive': X_pos,
        'Z_normalized': Z,
        'Z_weighted': Z_weighted,
        'Z_plus': Z_plus,
        'Z_minus': Z_minus,
        'D_plus': D_plus,
        'D_minus': D_minus,
    }

    return scores, ranking, rank_order, details


def topsis_sensitivity(data, weights, col_to_perturb, perturbations):
    """
    TOPSIS 灵敏性分析
    """
    results = []
    for delta in perturbations:
        w_pert = weights.copy()
        w_pert[col_to_perturb] *= (1 + delta)
        # 重新归一化
        w_pert = w_pert / np.sum(w_pert)
        scores, ranking, _, _ = topsis(data, weights=w_pert)
        results.append({'delta': delta, 'scores': scores.copy(), 'ranking': ranking.copy()})
    return results


# ==========================================
# 示例运行
# ==========================================

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 60)
    print("TOPSIS 法示例")
    print("=" * 60)

    # 示例：4个方案，5个指标
    # 指标1-2: 极大型(越大越好), 指标3: 极小型(越小越好),
    # 指标4: 中间型(越接近5越好), 指标5: 区间型([3,5]最好)
    data = np.array([
        [8.0, 7.0, 2.0, 4.5, 4.0],   # 方案1
        [6.0, 8.0, 3.0, 5.2, 3.5],   # 方案2
        [9.0, 6.0, 1.5, 4.8, 5.5],   # 方案3
        [7.0, 9.0, 2.5, 3.5, 2.0],   # 方案4
    ])

    print(f"\n原始数据 (4方案 × 5指标):\n{data}")

    scores, ranking, rank_order, details = topsis(
        data,
        benefit_cols=[0, 1],
        cost_cols=[2],
        target_cols=[3],
        target_values=[5],
        interval_cols=[4],
        interval_bounds=[(3, 5)],
    )

    print(f"\n正理想解: {details['Z_plus']}")
    print(f"负理想解: {details['Z_minus']}")
    print(f"\n距离 D+: {details['D_plus']}")
    print(f"距离 D-: {details['D_minus']}")
    print(f"\n贴近度 C: {scores}")
    print(f"排名 (1最优): {ranking}")

    print("\n--- 结果排序 ---")
    for r in rank_order:
        print(f"  第{ranking[r]}名: 方案{r+1}, 贴近度={scores[r]:.4f}")

    # --- 灵敏性分析 ---
    print("\n" + "=" * 60)
    print("灵敏性分析 — 扰动第1个指标权重")
    print("=" * 60)

    weights = np.ones(5) / 5
    perturbations = [-0.3, -0.1, 0, 0.1, 0.3]
    sens_results = topsis_sensitivity(data, weights, 0, perturbations)

    for r in sens_results:
        print(f"\n  权重扰动 {r['delta']:+5.0%}:")
        print(f"    贴近度: {r['scores']}")
        print(f"    排名: {r['ranking']}")
