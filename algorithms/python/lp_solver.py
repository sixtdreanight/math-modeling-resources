# -*- coding: utf-8 -*-
"""
线性规划 / 整数规划 — Python 实现
基于 scipy.optimize.linprog 和 PuLP

用法：
    pip install pulp
    python lp_solver.py
"""

import numpy as np


def solve_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='highs'):
    """
    线性规划求解

    minimize   c^T x
    subject to A_ub @ x <= b_ub
               A_eq @ x == b_eq
               bounds[i] = (lower, upper)

    Parameters
    ----------
    c : 目标函数系数
    A_ub, b_ub : 不等式约束
    A_eq, b_eq : 等式约束
    bounds : 变量上下界
    method : 'highs' (推荐) / 'simplex' / 'interior-point'

    Returns
    -------
    dict: {x, fun, success, message}
    """
    from scipy.optimize import linprog

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method=method)
    return {
        'x': result.x,
        'fun': result.fun,
        'success': result.success,
        'message': result.message,
    }


def solve_ilp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
              bounds=None, maximize=False):
    """
    整数规划求解（使用 PuLP）

    Parameters
    ----------
    c : 目标函数系数
    A_ub, b_ub : 不等式约束 (A_ub @ x <= b_ub)
    A_eq, b_eq : 等式约束
    bounds : 变量上下界，None 则为 (0, None) for ≥0
    maximize : True 最大化, False 最小化

    Returns
    -------
    dict: {x, obj_val, status}
    """
    import pulp

    n = len(c)
    prob = pulp.LpProblem("ILP", pulp.LpMaximize if maximize else pulp.LpMinimize)

    # 决策变量（整数）
    x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=None, cat='Integer') for i in range(n)]

    # 变量上下界
    if bounds:
        for i, (lb, ub) in enumerate(bounds):
            if lb is not None:
                x[i].lowBound = lb
            if ub is not None:
                x[i].upBound = ub

    # 目标函数
    prob += pulp.lpSum([c[i] * x[i] for i in range(n)])

    # 不等式约束
    if A_ub is not None:
        for i in range(len(b_ub)):
            prob += pulp.lpSum([A_ub[i, j] * x[j] for j in range(n)]) <= b_ub[i]

    # 等式约束
    if A_eq is not None:
        for i in range(len(b_eq)):
            prob += pulp.lpSum([A_eq[i, j] * x[j] for j in range(n)]) == b_eq[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    x_vals = np.array([pulp.value(x[i]) for i in range(n)])
    status = pulp.LpStatus[prob.status]

    return {
        'x': x_vals,
        'obj_val': pulp.value(prob.objective),
        'status': status,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("线性规划 / 整数规划示例")
    print("=" * 60)

    # --- 示例1: 线性规划 ---
    # maximize 2x1 + 3x2
    # s.t.    x1 + 2x2 <= 8
    #         4x1     <= 16
    #             4x2 <= 12
    #         x1, x2 >= 0

    print("\n[线性规划]")
    c = [-2, -3]  # 取负号转为最小化
    A_ub = [[1, 2], [4, 0], [0, 4]]
    b_ub = [8, 16, 12]
    bounds = [(0, None), (0, None)]

    result = solve_lp(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    print(f"  最优解: x1={result['x'][0]:.1f}, x2={result['x'][1]:.1f}")
    print(f"  最优值: {-result['fun']:.0f}")  # 还原最大化

    # --- 示例2: 整数规划 (0-1背包) ---
    # 容量 15, 物品价值 v=[8,10,6,3], 重量 w=[5,6,4,1]
    # maximize 8x1 + 10x2 + 6x3 + 3x4
    # s.t.     5x1 + 6x2 + 4x3 + x4 <= 15
    #          xi ∈ {0, 1}

    print("\n[0-1 整数规划 / 背包问题]")
    values = [8, 10, 6, 3]
    weights = [5, 6, 4, 1]
    capacity = 15
    n_items = len(values)

    c = [-v for v in values]  # 取负号，最小化
    A_ub = [weights]
    b_ub = [capacity]
    bounds = [(0, 1) for _ in range(n_items)]

    result = solve_ilp(c, A_ub=np.array([weights]), b_ub=[capacity],
                       bounds=bounds, maximize=False)
    print(f"  选择的物品: {result['x'].astype(int)}")
    print(f"  总价值: {-result['obj_val']:.0f}")
    print(f"  总重量: {np.dot(result['x'], weights):.0f} / {capacity}")
