# -*- coding: utf-8 -*-
"""
层次分析法 (AHP) — Python 实现
支持：几何平均法（默认）/ 特征值法 / 算术平均法
含一致性检验

用法：
    python ahp.py
"""

import numpy as np


def ahp_weights(judgment_matrix, method='geometric'):
    """
    计算 AHP 权重

    Parameters
    ----------
    judgment_matrix : ndarray (n, n)
        判断矩阵，必须满足正互反性 (a[i,j] = 1/a[j,i])
    method : str
        'geometric' — 几何平均法（推荐）
        'eigenvalue' — 特征值法（最精确）
        'arithmetic' — 算术平均法

    Returns
    -------
    weights : ndarray (n,)
        权重向量
    lambda_max : float
        最大特征值
    CI : float
        一致性指标
    CR : float
        一致性比率
    is_consistent : bool
        是否通过一致性检验 (CR < 0.10)
    """
    A = np.asarray(judgment_matrix, dtype=float)
    n = A.shape[0]

    # 检查正互反性
    if A.shape[0] != A.shape[1]:
        raise ValueError("判断矩阵必须是方阵")
    if not np.all(A > 0):
        raise ValueError("判断矩阵元素必须为正")
    if not np.allclose(np.diag(A), 1):
        raise ValueError("对角线元素必须为1")

    # 计算权重
    if method == 'geometric':
        # 几何平均法
        geo_mean = np.power(np.prod(A, axis=1), 1/n)
        weights = geo_mean / (np.sum(geo_mean) + 1e-10)
    elif method == 'arithmetic':
        # 算术平均法（列归一化后取行平均）
        col_sum = np.sum(A, axis=0)
        normalized = A / col_sum
        weights = np.mean(normalized, axis=1)
    elif method == 'eigenvalue':
        # 特征值法
        eigenvalues, eigenvectors = np.linalg.eig(A)
        max_idx = np.argmax(np.real(eigenvalues))
        weights = np.real(eigenvectors[:, max_idx])
        weights = weights / np.sum(weights)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 计算最大特征值
    # λ_max = (1/n) * Σ[(A·w)_i / w_i]
    Aw = A @ weights
    lambda_max = np.mean(Aw / (weights + 1e-10))

    # 一致性检验
    RI_table = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
                7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51,
                12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59}

    CI = (lambda_max - n) / (n - 1) if n > 1 else 0
    RI = RI_table.get(n, 1.59)
    CR = CI / RI if RI > 0 and n > 2 else 0
    is_consistent = CR < 0.10

    return weights, lambda_max, CI, CR, is_consistent


def ahp_hierarchy(criteria_matrix, sub_matrices):
    """
    层次总排序

    Parameters
    ----------
    criteria_matrix : ndarray
        准则层判断矩阵
    sub_matrices : list of ndarray
        各准则下子指标的判断矩阵列表

    Returns
    -------
    total_weights : ndarray
        各指标相对于总目标的合成权重
    """
    # 准则层权重
    criteria_weights, _, _, _, _ = ahp_weights(criteria_matrix)

    # 子层权重矩阵
    sub_weights_list = []
    for sub_mat in sub_matrices:
        w, _, _, _, _ = ahp_weights(sub_mat)
        sub_weights_list.append(w)

    # 组合权重
    total_weights = np.zeros(len(sub_weights_list[0]))
    for i, w in enumerate(sub_weights_list):
        total_weights += criteria_weights[i] * w

    return total_weights


def sensitivity_check(judgment_matrix, perturb_index, perturb_range=(-0.2, 0.2, 5)):
    """
    灵敏性检查：扰动单个判断值，观察 CR 变化
    """
    A = np.asarray(judgment_matrix, dtype=float)
    n = A.shape[0]
    i, j = perturb_index

    results = []
    for p in np.linspace(perturb_range[0], perturb_range[1], perturb_range[2]):
        A_mod = A.copy()
        A_mod[i, j] *= (1 + p)
        A_mod[j, i] = 1 / A_mod[i, j]

        _, lam, CI, CR, ok = ahp_weights(A_mod)
        results.append({'perturbation': p, 'lambda_max': lam, 'CR': CR, 'consistent': ok})

    return results


# ==========================================
# 示例运行
# ==========================================

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    # --- 示例1: 单层AHP ---
    # 3个指标的两两比较矩阵
    #      C1  C2  C3
    # C1 [  1   2   4  ]
    # C2 [ 1/2  1   3  ]
    # C3 [ 1/4 1/3  1  ]
    #
    # 含义：C1比C2稍微重要(2)，C1比C3略强于明显重要(4)，C2比C3稍微重要(3)

    A = np.array([
        [1,   2,   4],
        [1/2, 1,   3],
        [1/4, 1/3, 1]
    ])

    print("=" * 60)
    print("层次分析法 (AHP) 示例")
    print("=" * 60)
    print(f"\n判断矩阵:\n{A}")

    w, lam, ci, cr, ok = ahp_weights(A, method='geometric')

    print(f"\n几何平均法权重: {w}")
    print(f"最大特征值 λ_max: {lam:.4f}")
    print(f"一致性指标 CI: {ci:.4f}")
    print(f"一致性比率 CR: {cr:.4f}")
    print(f"一致性检验: {'通过' if ok else '不通过'} (CR {'<' if ok else '≥'} 0.10)")

    # 对比三种方法
    print("\n--- 三种方法对比 ---")
    for method in ['geometric', 'arithmetic', 'eigenvalue']:
        w_m, _, _, _, _ = ahp_weights(A, method=method)
        print(f"  {method:12s}: {w_m}")

    # --- 示例2: 层次总排序 ---
    print("\n" + "=" * 60)
    print("层次总排序示例")
    print("=" * 60)

    # 准则层：3个准则
    criteria = np.array([
        [1,   3,   5],
        [1/3, 1,   3],
        [1/5, 1/3, 1]
    ])

    # 各准则下4个指标的判断矩阵
    sub1 = np.array([
        [1,   2,   4,   6],
        [1/2, 1,   3,   5],
        [1/4, 1/3, 1,   3],
        [1/6, 1/5, 1/3, 1]
    ])
    sub2 = np.array([
        [1,   1/3, 2,   4],
        [3,   1,   4,   6],
        [1/2, 1/4, 1,   3],
        [1/4, 1/6, 1/3, 1]
    ])
    sub3 = np.array([
        [1,   5,   3,   7],
        [1/5, 1,   1/3, 2],
        [1/3, 3,   1,   5],
        [1/7, 1/2, 1/5, 1]
    ])

    total_w = ahp_hierarchy(criteria, [sub1, sub2, sub3])
    print(f"合成权重: {total_w}")

    # --- 示例3: 灵敏性检查 ---
    print("\n" + "=" * 60)
    print("灵敏性检查")
    print("=" * 60)

    results = sensitivity_check(A, (0, 1))
    for r in results:
        flag = "✓" if r['consistent'] else "✗"
        print(f"  扰动 {r['perturbation']:+5.0%}: CR={r['CR']:.4f} {flag}")
