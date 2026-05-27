# -*- coding: utf-8 -*-
"""
模糊综合评价 — Python 实现
支持多种模糊算子 + 隶属函数计算
"""

import numpy as np


def fuzzy_evaluate(R, W, operator='weighted_avg'):
    """
    模糊综合评价

    Parameters
    ----------
    R : list of ndarray
        模糊评价矩阵列表（每个指标对应一个），shape: (m_alternatives, k_levels)
    W : ndarray
        权重向量
    operator : str
        'weighted_avg' — 加权平均型 M(·,+)
        'main_factor'  — 主因素决定型 M(∧,∨)
        'main_factor_highlight' — 主因素突出型 M(·,∨)
        'weighted_fuzzy' — 加权模糊型 M(∧,+)

    Returns
    -------
    result : dict
        'B': 综合评判结果向量 (各等级隶属度)
        'result': 最大隶属度对应的等级索引
        'score': 加权得分（若levels为数值）
    """
    n_alt = len(R)  # 方案数
    k = R[0].shape[1]  # 评价等级数
    W = np.asarray(W)

    # 聚合评价矩阵（多指标加权综合为一个评价矩阵）
    # R_combined[i,j] = 方案i在第j等级的隶属度
    R_matrix = np.stack([R[i] for i in range(n_alt)])  # (n_alt, n_criteria, k)

    B_list = []
    for i in range(n_alt):
        if operator == 'weighted_avg':
            B = W @ R_matrix[i]
        elif operator == 'main_factor':
            B = np.array([np.max(np.minimum(W, R_matrix[i, :, j])) for j in range(k)])
        elif operator == 'main_factor_highlight':
            B = np.array([np.max(W * R_matrix[i, :, j]) for j in range(k)])
        elif operator == 'weighted_fuzzy':
            B = np.array([np.minimum(1, np.sum(np.minimum(W, R_matrix[i, :, j]))) for j in range(k)])
        else:
            raise ValueError(f"Unknown operator: {operator}")
        B = B / np.sum(B)  # 归一化
        B_list.append(B)

    results = []
    for B in B_list:
        max_idx = np.argmax(B)
        results.append({
            'B': B,
            'max_grade_idx': max_idx,
        })
    return results


def triangular_mf(x, params):
    """三角隶属函数"""
    a, b, c = params
    if x <= a or x >= c:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return (c - x) / (c - b)
    return 0


def trapezoidal_mf(x, params):
    """梯形隶属函数"""
    a, b, c, d = params
    if x <= a or x >= d:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1
    elif c < x <= d:
        return (d - x) / (d - c)
    return 0


def gaussian_mf(x, params):
    """高斯隶属函数"""
    mean, sigma = params
    return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def compute_membership(values, mf_type, mf_params):
    """
    批量计算隶属度

    Parameters
    ----------
    values : array-like
        观测值
    mf_type : str
        'triangular' / 'trapezoidal' / 'gaussian'
    mf_params : list of tuple
        每个等级的隶属函数参数列表

    Returns
    -------
    R : ndarray (k_levels,)
        各等级的隶属度向量
    """
    mf_func = {'triangular': triangular_mf, 'trapezoidal': trapezoidal_mf,
               'gaussian': gaussian_mf}[mf_type]
    k = len(mf_params)
    R = np.zeros(k)
    for j in range(k):
        R[j] = mf_func(values, mf_params[j])
    return R


if __name__ == '__main__':
    print("=" * 60)
    print("模糊综合评价示例")
    print("=" * 60)

    # 3个方案，4个评价指标，5个等级（很差/差/一般/好/很好）
    # 每个指标的值已经通过隶属函数转化为隶属度
    R0 = np.array([
        [0.2, 0.3, 0.3, 0.1, 0.1],  # 指标1 对各等级的隶属度
        [0.1, 0.2, 0.4, 0.2, 0.1],  # 指标2
        [0.0, 0.1, 0.3, 0.4, 0.2],  # 指标3
        [0.1, 0.1, 0.3, 0.3, 0.2],  # 指标4
    ])

    R1 = np.array([
        [0.1, 0.4, 0.3, 0.1, 0.1],
        [0.2, 0.3, 0.3, 0.1, 0.1],
        [0.0, 0.2, 0.4, 0.3, 0.1],
        [0.1, 0.3, 0.3, 0.2, 0.1],
    ])

    R = [R0, R1]
    W = np.array([0.3, 0.3, 0.2, 0.2])  # 指标权重

    results = fuzzy_evaluate(R, W, operator='weighted_avg')

    levels = ['很差', '差', '一般', '好', '很好']
    for i, r in enumerate(results):
        print(f"\n方案{i+1}:")
        for j, (level, b) in enumerate(zip(levels, r['B'])):
            bar = '█' * int(b * 20)
            print(f"  {level}: {b:.3f} {bar}")
        print(f"  综合评定: {levels[r['max_grade_idx']]}")

    # --- 三角隶属函数示例 ---
    print("\n" + "=" * 60)
    print("三角隶属函数计算示例")
    print("=" * 60)

    # 对值 6.5 计算其属于 3 个等级的隶属度
    mf_params = [(0, 4, 8), (4, 8, 12), (8, 12, 16)]  # 3个等级
    R_example = compute_membership(6.5, 'triangular', mf_params)
    level_names = ['低', '中', '高']
    for name, val in zip(level_names, R_example):
        print(f"  属于'{name}'的隶属度: {val:.3f}")
