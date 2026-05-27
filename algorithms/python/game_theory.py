# -*- coding: utf-8 -*-
"""
博弈论 — Python 实现
功能：纳什均衡 / 演化博弈 / 拍卖模型

用法：
    python game_theory.py
"""

import numpy as np
from scipy.optimize import linprog


def nash_equilibrium_pure(payoff_matrix):
    """
    纯策略纳什均衡（2人有限策略博弈）

    Parameters
    ----------
    payoff_matrix : ndarray (m, n, 2)
        支付矩阵, payoff[i, j, 0] = 玩家1的支付, payoff[i, j, 1] = 玩家2的支付

    Returns
    -------
    list of tuple: 纯策略纳什均衡 [(i*, j*), ...]
    """
    m, n = payoff_matrix.shape[:2]
    equilibria = []

    for i in range(m):
        for j in range(n):
            # 玩家1：给定玩家2选j，玩家1选i是否最优？
            is_best1 = np.all(payoff_matrix[i, j, 0] >= payoff_matrix[:, j, 0])
            # 玩家2：给定玩家1选i，玩家2选j是否最优？
            is_best2 = np.all(payoff_matrix[i, j, 1] >= payoff_matrix[i, :, 1])
            if is_best1 and is_best2:
                equilibria.append((i, j))

    return equilibria


def mixed_nash_2x2(payoff_player1):
    """
    2×2 博弈混合策略纳什均衡（解析公式）

    A = [[a11, a12],    B = [[b11, b12],
         [a21, a22]]         [b21, b22]]

    Returns:
        p (玩家1选策略1的概率), q (玩家2选策略1的概率)
    """
    A = np.asarray(payoff_player1)
    # 玩家1的策略：p * A[0,j] + (1-p) * A[1,j]
    # 玩家1的混合策略使玩家2无差异
    # 玩家2选策略1: p*a11 + (1-p)*a21
    # 玩家2选策略2: p*a12 + (1-p)*a22
    # 令两者相等 → p = (a22 - a21) / (a11 - a12 - a21 + a22)

    a11, a12 = A[0, 0], A[0, 1]
    a21, a22 = A[1, 0], A[1, 1]
    denom = a11 - a12 - a21 + a22
    if abs(denom) < 1e-10:
        return None
    p = (a22 - a21) / denom
    p = np.clip(p, 0, 1)
    return p


def evolutionary_game(A, x0=None, steps=100, dt=0.1):
    """
    演化博弈：复制者动态方程
    dp/dt = p_i * (f_i - avg_f)
    """
    A = np.asarray(A)
    n = A.shape[0]
    if x0 is None:
        x0 = np.ones(n) / n

    x = x0.copy()
    history = [x.copy()]

    for _ in range(steps):
        fitness = A @ x  # f_i = Σ A_ij * x_j
        avg_fitness = x @ fitness  # avg = Σ x_i * f_i
        dx = x * (fitness - avg_fitness)  # 复制者动态
        x = x + dt * dx
        x = np.clip(x, 0, 1)
        x = x / np.sum(x)
        history.append(x.copy())

    return np.array(history)


def first_price_auction(n_bidders, values_dist, n_sim=50000):
    """
    一价密封拍卖模拟（蒙特卡洛）
    每个投标人独立从 values_dist 抽样估值 v_i
    均衡策略: b(v) = v * (n-1)/n （均匀分布假设）

    Returns:
        dict: 期望收益、买家剩余、卖方收入
    """
    n = n_bidders
    values = values_dist(n_sim, n)  # shape (n_sim, n)

    # 最高估值者赢，支付其出价
    winner_idx = np.argmax(values, axis=1)
    second_highest = np.sort(values, axis=1)[:, -2]

    # 理论均衡出价
    bids = values * (n - 1) / n
    revenue = bids[np.arange(n_sim), winner_idx]
    surplus = values[np.arange(n_sim), winner_idx] - revenue

    return {
        'expected_revenue': np.mean(revenue),
        'expected_surplus': np.mean(surplus),
        'efficiency': np.mean(values[np.arange(n_sim), winner_idx]),
    }


if __name__ == '__main__':
    np.random.seed(42)
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 60)
    print("博弈论示例")
    print("=" * 60)

    # --- 囚徒困境 ---
    # 玩家1的支付矩阵
    #       C(合作)  D(背叛)
    # C      (-1)     (-10)
    # D      (0)      (-5)
    payoff = np.array([
        [[-1, -1], [-10, 0]],  # 玩家1选C
        [[0, -10], [-5, -5]],  # 玩家1选D
    ])
    ne = nash_equilibrium_pure(payoff)
    print(f"\n[囚徒困境] 纯策略纳什均衡: {ne}")
    print("  → (D, D) = (背叛, 背叛) 是唯一的纳什均衡")

    # --- 演化博弈 ---
    # 鹰鸽博弈支付矩阵
    #     鹰    鸽
    # 鹰 (0,0) (4,1)
    # 鸽 (1,4) (2,2)
    A_hawk_dove = np.array([[0, 4], [1, 2]])
    history = evolutionary_game(A_hawk_dove, x0=np.array([0.3, 0.7]), steps=200, dt=0.05)

    print(f"\n[演化博弈 - 鹰鸽博弈] 初值 x0=[0.3, 0.7]")
    print(f"  开始: 鹰={history[0,0]:.2f}, 鸽={history[0,1]:.2f}")
    print(f"  结束: 鹰={history[-1,0]:.3f}, 鸽={history[-1,1]:.3f}")

    # --- 一价拍卖 ---
    print(f"\n[一价拍卖模拟] n=3 投标人, 估值~U(0,100)")
    auction = first_price_auction(3, lambda s, n: np.random.uniform(0, 100, (s, n)), n_sim=20000)
    print(f"  卖方期望收入: {auction['expected_revenue']:.1f}")
    print(f"  买方期望剩余: {auction['expected_surplus']:.1f}")
