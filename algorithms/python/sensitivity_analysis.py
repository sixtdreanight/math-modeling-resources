# -*- coding: utf-8 -*-
"""
灵敏性分析 — Python 实现
方法：OAT / Morris 筛选 / Sobol 指数
论文评审加分项，建议每个赛题都做

用法：
    pip install SALib
    python sensitivity_analysis.py
"""

import numpy as np


def oat_sensitivity(model_func, base_params, param_names, deltas=None):
    """
    OAT (One-At-a-Time) 灵敏性分析
    每次只改变一个参数，观察输出变化

    Parameters
    ----------
    model_func : callable
        f(params_dict) -> float
    base_params : list of float
        基准参数值
    param_names : list of str
        参数名称
    deltas : list of float, optional
        扰动百分比列表，默认 [-50%, -30%, -10%, +10%, +30%, +50%]

    Returns
    -------
    dict: {param_name: [(delta, output), ...]}
    """
    deltas = deltas or [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
    results = {}

    for i, (name, base) in enumerate(zip(param_names, base_params)):
        rows = []
        for d in deltas:
            params = base_params.copy()
            params[i] = base * (1 + d)
            output = model_func(params)
            rows.append({'delta': d, 'output': output})
        results[name] = rows

    return results


def morris_analysis(model_func, bounds, param_names, n_trajectories=10, n_levels=4):
    """
    Morris 筛选法（Elementary Effects）
    适合参数较多时筛选有影响力的参数

    Parameters
    ----------
    model_func : callable
        f(x_array) -> float
    bounds : list of tuple
        各参数范围 [(min, max), ...]
    param_names : list of str
    n_trajectories : int
        轨迹数（越大越精确）
    n_levels : int
        离散级别数

    Returns
    -------
    dict: mu (均值), mu_star (绝对均值), sigma (标准差)
    """
    k = len(param_names)
    delta = n_levels / (n_levels - 1)
    levels = np.linspace(0, 1, n_levels)

    # 随机生成轨迹
    mu = np.zeros(k)
    mu_star = np.zeros(k)
    sigma = np.zeros(k)

    for t in range(n_trajectories):
        # 随机起始点
        x = np.array([np.random.choice(levels) for _ in range(k)])
        x_phys = np.array([bounds[i][0] + x[i] * (bounds[i][1] - bounds[i][0]) / (n_levels - 1)
                          for i in range(k)])

        # 随机排列参数顺序
        perm = np.random.permutation(k)

        for j in range(k):
            idx = perm[j]
            x_prev = x_phys.copy()
            prev_out = model_func(x_prev)

            # 改变一个参数
            step = np.random.choice([1, -1]) * delta / (n_levels - 1)
            x_phys[idx] = bounds[idx][0] + (x[idx] + step) * (bounds[idx][1] - bounds[idx][0]) / (n_levels - 1)
            x_phys[idx] = np.clip(x_phys[idx], bounds[idx][0], bounds[idx][1])

            new_out = model_func(x_phys)
            ee = (new_out - prev_out) / step

            mu[idx] += ee
            sigma[idx] += abs(ee)

        mu_star += np.abs(mu)

    n_ee = n_trajectories
    mu /= n_ee
    mu_star /= n_ee
    sigma = np.sqrt(np.maximum(sigma / n_ee - mu_star ** 2, 0))

    return {
        'param_names': param_names,
        'mu': mu,
        'mu_star': mu_star,
        'sigma': sigma,
    }


def sobol_analysis(model_func, bounds, param_names, n_samples=1024):
    """
    Sobol 指数分析（全局灵敏性分析，金标准）
    需安装 SALib: pip install SALib

    Returns
    -------
    dict: S1 (一阶), ST (总效应), S2 (二阶交互)
    """
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': bounds,
    }

    # 生成样本
    param_values = saltelli.sample(problem, n_samples)

    # 评估模型
    Y = np.array([model_func(p) for p in param_values])

    # 灵敏度分析
    Si = sobol.analyze(problem, Y)

    return Si


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    # 示例模型：y = 2*a + 0.5*b^2 + 0.1*a*b + noise
    def model(params):
        a, b, c = params
        return 2 * a + 0.5 * b ** 2 + 0.1 * a * b + 0.01 * c

    print("=" * 60)
    print("灵敏性分析示例")
    print("=" * 60)
    print("模型: y = 2a + 0.5b² + 0.1ab + 0.01c")

    # --- OAT 分析 ---
    print("\n[OAT 逐次变化分析]")
    base = [1.0, 2.0, 3.0]
    names = ['a', 'b', 'c']

    oat_results = oat_sensitivity(model, base, names)

    for name, rows in oat_results.items():
        print(f"\n  参数 {name} (基准值={base[names.index(name)]}):")
        print(f"    {'扰动':>8s}  {'输出':>10s}  {'变化%':>8s}")
        base_out = model(base)
        for r in rows:
            delta_pct = r['delta'] * 100
            out_change = (r['output'] - base_out) / base_out * 100
            print(f"    {delta_pct:+6.0f}%  {r['output']:10.3f}  {out_change:+7.2f}%")

    # --- Morris 筛选 ---
    print("\n" + "=" * 60)
    print("[Morris 筛选法]")

    bounds = [(0, 5), (0, 5), (0, 5)]
    morris_result = morris_analysis(model, bounds, names, n_trajectories=15)

    print(f"\n  {'参数':>6s}  {'mu*':>8s}  {'sigma':>8s}  {'影响度'}")
    for i, name in enumerate(names):
        influence = '高' if morris_result['mu_star'][i] > np.mean(morris_result['mu_star']) else '低'
        print(f"  {name:>6s}  {morris_result['mu_star'][i]:8.3f}  {morris_result['sigma'][i]:8.3f}  {influence}")

    # 解读
    print("\n  → 参数 c 的 mu* 最小，对输出几乎无影响（可固定）")
    print("  → 参数 a 和 b 影响较大，模型中需重点关注")
