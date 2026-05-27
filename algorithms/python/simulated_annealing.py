# -*- coding: utf-8 -*-
"""
模拟退火 (SA) — Python 实现
功能：指数退火 / 自适应步长

用法：
    python simulated_annealing.py
"""

import numpy as np


class SimulatedAnnealing:
    """模拟退火算法"""

    def __init__(self, obj_func, bounds, T_start=1000, T_end=0.01,
                 cooling_rate=0.95, max_iter=5000, maximize=True):
        self.obj_func = obj_func
        self.bounds = np.asarray(bounds)
        self.dim = len(bounds)
        self.T_start = T_start
        self.T_end = T_end
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.maximize = maximize

    def _generate_neighbor(self, x, T):
        """生成邻域解，自适应步长"""
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        scale = (ub - lb) * (T / self.T_start + 0.01)
        delta = np.random.normal(0, 1, self.dim) * scale
        return np.clip(x + delta, lb, ub)

    def run(self, verbose=True):
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]

        # 随机初始解
        x = np.random.uniform(lb, ub)
        f_x = self.obj_func(x)
        best_x = x.copy()
        best_f = f_x

        T = self.T_start
        history = [best_f]
        accept_count = 0

        for t in range(self.max_iter):
            x_new = self._generate_neighbor(x, T)
            f_new = self.obj_func(x_new)

            delta = f_new - f_x
            if not self.maximize:
                delta = -delta

            # Metropolis 准则
            if delta > 0 or np.random.rand() < np.exp(delta / (T + 1e-10)):
                x = x_new
                f_x = f_new
                accept_count += 1

                if (self.maximize and f_x > best_f) or (not self.maximize and f_x < best_f):
                    best_x = x.copy()
                    best_f = f_x

            # 退火
            T *= self.cooling_rate
            if T < self.T_end:
                T = self.T_end

            history.append(best_f)

            if verbose and t % 1000 == 0:
                print(f"  Iter {t:4d}: T={T:.4f}, Best={best_f:.6f}")

        return {
            'x_best': best_x,
            'f_best': best_f,
            'history': history,
            'accept_rate': accept_count / self.max_iter,
        }


if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("模拟退火 (SA) 示例")
    print("=" * 60)

    # Rosenbrock 函数
    def rosenbrock(x):
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    bounds = [(-3, 3), (-3, 3)]
    sa = SimulatedAnnealing(rosenbrock, bounds, T_start=100, cooling_rate=0.98,
                            max_iter=3000, maximize=False)
    result = sa.run(verbose=True)

    print(f"\n最优解: x1={result['x_best'][0]:.6f}, x2={result['x_best'][1]:.6f}")
    print(f"最优值: f={result['f_best']:.6f}")
    print(f"接受率: {result['accept_rate']:.2%}")
    print(f"全局最优 ≈ (1, 1), f=0")
