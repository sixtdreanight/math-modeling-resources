# -*- coding: utf-8 -*-
"""
粒子群优化 (PSO) — Python 实现
功能：标准PSO / 自适应惯性权重 / 速度钳制

用法：
    python particle_swarm.py
"""

import numpy as np


class ParticleSwarm:
    """粒子群优化算法"""

    def __init__(self, obj_func, bounds, n_particles=50, max_iter=200,
                 w_start=0.9, w_end=0.4, c1=2.0, c2=2.0, maximize=True):
        self.obj_func = obj_func
        self.bounds = np.asarray(bounds)
        self.dim = len(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.maximize = maximize

    @property
    def lb(self):
        return self.bounds[:, 0]

    @property
    def ub(self):
        return self.bounds[:, 1]

    def _init_swarm(self):
        """初始化粒子群"""
        pos = np.random.uniform(self.lb, self.ub, (self.n_particles, self.dim))
        vel = np.zeros((self.n_particles, self.dim))
        pbest_pos = pos.copy()
        pbest_val = np.array([self.obj_func(p) for p in pos])

        idx = np.argmax(pbest_val) if self.maximize else np.argmin(pbest_val)
        return pos, vel, pbest_pos, pbest_val, pbest_pos[idx].copy(), pbest_val[idx]

    def _update_velocity(self, vel, pos, pbest_pos, gbest_pos, t):
        """更新粒子速度"""
        w = self.w_start - (self.w_start - self.w_end) * t / self.max_iter
        r1 = np.random.rand(self.n_particles, self.dim)
        r2 = np.random.rand(self.n_particles, self.dim)
        vel = w * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
        max_vel = 0.2 * (self.ub - self.lb)
        return np.clip(vel, -max_vel, max_vel), w

    def _update_best(self, pos, pbest_pos, pbest_val, gbest_pos, gbest_val):
        """更新个体最优和全局最优"""
        current_val = np.array([self.obj_func(p) for p in pos])
        if self.maximize:
            improved = current_val > pbest_val
            curr_best_idx = np.argmax(current_val)
            if current_val[curr_best_idx] > gbest_val:
                gbest_val = current_val[curr_best_idx]
                gbest_pos = pos[curr_best_idx].copy()
        else:
            improved = current_val < pbest_val
            curr_best_idx = np.argmin(current_val)
            if current_val[curr_best_idx] < gbest_val:
                gbest_val = current_val[curr_best_idx]
                gbest_pos = pos[curr_best_idx].copy()
        pbest_pos[improved] = pos[improved].copy()
        pbest_val[improved] = current_val[improved]
        return pbest_pos, pbest_val, gbest_pos, gbest_val

    def run(self, verbose=True):
        pos, vel, pbest_pos, pbest_val, gbest_pos, gbest_val = self._init_swarm()
        history = [gbest_val]

        for t in range(self.max_iter):
            vel, _ = self._update_velocity(vel, pos, pbest_pos, gbest_pos, t)
            pos = np.clip(pos + vel, self.lb, self.ub)
            pbest_pos, pbest_val, gbest_pos, gbest_val = (
                self._update_best(pos, pbest_pos, pbest_val, gbest_pos, gbest_val))
            history.append(gbest_val)

            if verbose and t % 50 == 0:
                print(f"  Iter {t:3d}: Best = {gbest_val:.6f}")

        return {'x_best': gbest_pos, 'f_best': gbest_val, 'history': history}


if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("粒子群优化 (PSO) 示例")
    print("=" * 60)

    def ackley(x):
        a, b, c = 20, 0.2, 2 * np.pi
        s1 = np.sum(x ** 2)
        s2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(s1 / len(x))) - np.exp(s2 / len(x)) + a + np.exp(1)

    bounds = [(-5, 5), (-5, 5)]
    pso = ParticleSwarm(ackley, bounds, n_particles=50, max_iter=200, maximize=False)
    result = pso.run(verbose=True)

    print(f"\n最优解: x1={result['x_best'][0]:.6f}, x2={result['x_best'][1]:.6f}")
    print(f"最优值: f={result['f_best']:.6f}")
    print(f"全局最优 ≈ (0, 0), f=0")
