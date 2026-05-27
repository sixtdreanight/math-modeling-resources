# -*- coding: utf-8 -*-
"""
蒙特卡洛模拟 — Python 实现
功能：π估计 / 定积分 / 优化 / 排队系统模拟 / 风险分析

用法：
    python monte_carlo.py
"""

import numpy as np


def mc_pi(n=100000):
    """用蒙特卡洛法估计 π（经典示例）"""
    points = np.random.rand(n, 2)
    inside = np.sum(np.sum(points ** 2, axis=1) <= 1)
    pi_est = 4 * inside / n
    return pi_est


def mc_integrate(f, a, b, n=100000):
    """
    蒙特卡洛定积分 ∫[a,b] f(x)dx

    Returns
    -------
    估计值, 标准误差
    """
    x = np.random.uniform(a, b, n)
    y = f(x)
    mean_val = np.mean(y)
    std_err = np.std(y) / np.sqrt(n)
    return (b - a) * mean_val, std_err


def mc_optimize(obj_func, bounds, n=50000, maximize=True):
    """
    蒙特卡洛随机搜索优化
    当目标函数非凸/非光滑/无梯度时使用

    Parameters
    ----------
    obj_func : callable
        目标函数 f(x)，x 为向量
    bounds : list of tuple
        各变量范围 [(lb1, ub1), ...]
    n : int
        随机采样数
    maximize : bool

    Returns
    -------
    最优解, 最优值
    """
    dim = len(bounds)
    best_x = None
    best_val = -np.inf if maximize else np.inf

    for _ in range(n):
        x = np.array([np.random.uniform(lb, ub) for lb, ub in bounds])
        val = obj_func(x)
        if maximize:
            if val > best_val:
                best_val = val
                best_x = x.copy()
        else:
            if val < best_val:
                best_val = val
                best_x = x.copy()

    return best_x, best_val


def mc_queue_sim(arrival_rate, service_rate, num_customers=10000):
    """
    M/M/1 排队系统蒙特卡洛模拟

    Parameters
    ----------
    arrival_rate : float
        到达率 λ
    service_rate : float
        服务率 μ (需 μ > λ 否则队无限长)
    num_customers : int
        模拟顾客数

    Returns
    -------
    dict: 平均等待时间、平均队列长度、服务器利用率等
    """
    np.random.seed(42)
    rng = np.random.RandomState(42)

    # 生成到达间隔（指数分布）和服务时间
    inter_arrival = rng.exponential(1 / arrival_rate, num_customers)
    arrival_times = np.cumsum(inter_arrival)
    service_times = rng.exponential(1 / service_rate, num_customers)

    # 模拟
    wait_times = np.zeros(num_customers)
    departure_times = np.zeros(num_customers)

    departure_times[0] = arrival_times[0] + service_times[0]

    for i in range(1, num_customers):
        start_service = max(arrival_times[i], departure_times[i-1])
        wait_times[i] = start_service - arrival_times[i]
        departure_times[i] = start_service + service_times[i]

    total_time = max(departure_times[-1], arrival_times[-1])

    return {
        'avg_wait': np.mean(wait_times),
        'max_wait': np.max(wait_times),
        'utilization': np.sum(service_times) / total_time,
        'theoretical_avg_wait': arrival_rate / (service_rate * (service_rate - arrival_rate)),
        'theoretical_utilization': arrival_rate / service_rate,
    }


def mc_risk_analysis(cash_flows, discount_rate=0.05, n_sim=10000):
    """
    投资风险评估：现金流服从正态分布，模拟 NPV 分布

    Parameters
    ----------
    cash_flows : list of tuple (mean, std)
        每年现金流 (均值, 标准差)
    discount_rate : float
        折现率
    n_sim : int
        模拟次数

    Returns
    -------
    dict: NPV 均值、标准差、VaR、亏损概率
    """
    T = len(cash_flows)
    npvs = np.zeros(n_sim)

    t_values = np.arange(T)
    for t in range(T):
        mean, std = cash_flows[t]
        sim_cf = np.random.normal(mean, std, n_sim)
        npvs += sim_cf / (1 + discount_rate) ** t_values[t]

    return {
        'mean_npv': np.mean(npvs),
        'std_npv': np.std(npvs),
        'VaR_95': np.percentile(npvs, 5),   # 5% VaR
        'CVaR_95': npvs[npvs <= np.percentile(npvs, 5)].mean(),  # 条件VaR
        'prob_loss': np.mean(npvs < 0),
    }


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    print("=" * 60)
    print("蒙特卡洛模拟示例")
    print("=" * 60)

    # 1. 估计 π
    pi_est = mc_pi(500000)
    print(f"\n[π 估计] {pi_est:.6f} (误差: {abs(pi_est - np.pi):.6f})")

    # 2. 定积分
    f = lambda x: np.sin(x)
    integral, stderr = mc_integrate(f, 0, np.pi, n=200000)
    print(f"\n[定积分] ∫sin(x)dx [0,π] = {integral:.6f}, 标准误={stderr:.6f} (精确值=2)")

    # 3. 随机优化
    def rastrigin(x):
        return 20 + x[0]**2 + x[1]**2 - 10*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))
    best_x, best_val = mc_optimize(rastrigin, [(-5, 5), (-5, 5)], n=20000, maximize=False)
    print(f"\n[随机优化] Rastrigin函数 最优解=({best_x[0]:.3f}, {best_x[1]:.3f}), 值={best_val:.3f}")

    # 4. 排队模拟
    q_result = mc_queue_sim(arrival_rate=5, service_rate=8, num_customers=5000)
    print(f"\n[M/M/1 排队] λ=5, μ=8")
    print(f"  平均等待: {q_result['avg_wait']:.3f} (理论值: {q_result['theoretical_avg_wait']:.3f})")
    print(f"  利用率: {q_result['utilization']:.3f} (理论值: {q_result['theoretical_utilization']:.3f})")

    # 5. 风险分析
    cash_flows = [(-100, 5), (30, 8), (35, 10), (40, 12), (45, 15), (50, 18)]
    risk = mc_risk_analysis(cash_flows, discount_rate=0.08, n_sim=20000)
    print(f"\n[投资风险] 6年期项目")
    print(f"  NPV均值: {risk['mean_npv']:.1f}, 标准差: {risk['std_npv']:.1f}")
    print(f"  VaR(95%): {risk['VaR_95']:.1f}, 亏损概率: {risk['prob_loss']:.1%}")
