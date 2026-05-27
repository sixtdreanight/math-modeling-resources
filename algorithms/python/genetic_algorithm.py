# -*- coding: utf-8 -*-
"""
遗传算法 (GA) — Python 实现
功能：二进制编码 / 轮盘赌选择 / 交叉 / 变异 / 精英保留

用法：
    python genetic_algorithm.py
"""

import numpy as np


class GeneticAlgorithm:
    """通用遗传算法求解器

    Parameters
    ----------
    obj_func : callable
        目标函数 f(x) -> float
    bounds : list of tuple
        各变量范围
    config : dict, optional
        {'pop_size': 100, 'max_gen': 200, 'crossover_prob': 0.8,
         'mutation_prob': 0.1, 'elite_ratio': 0.05, 'maximize': True}
        可直接传入配置字典替代多个参数。
    也可使用关键字参数逐个设置，config 中的值会覆盖关键字默认值。
    """

    def __init__(self, obj_func, bounds, config=None, **kwargs):
        self.obj_func = obj_func
        self.bounds = np.asarray(bounds)
        self.dim = len(bounds)

        # 合并 config 和 kwargs
        cfg = {'pop_size': 100, 'max_gen': 200, 'crossover_prob': 0.8,
               'mutation_prob': 0.1, 'elite_ratio': 0.05, 'maximize': True}
        if config:
            cfg.update(config)
        cfg.update(kwargs)

        self.pop_size = cfg['pop_size']
        self.max_gen = cfg['max_gen']
        self.crossover_prob = cfg['crossover_prob']
        self.mutation_prob = cfg['mutation_prob']
        self.n_elite = max(1, int(self.pop_size * cfg['elite_ratio']))
        self.maximize = cfg['maximize']

    def _init_population(self):
        pop = np.zeros((self.pop_size, self.dim))
        for i in range(self.dim):
            lb, ub = self.bounds[i]
            pop[:, i] = np.random.uniform(lb, ub, self.pop_size)
        return pop

    def _evaluate(self, pop):
        return np.array([self.obj_func(ind) for ind in pop])

    def _select(self, pop, fitness):
        """轮盘赌选择"""
        if self.maximize:
            fitness = fitness - np.min(fitness) + 1e-10
        else:
            fitness = np.max(fitness) - fitness + 1e-10
        probs = fitness / np.sum(fitness)
        indices = np.random.choice(len(pop), size=self.pop_size - self.n_elite, p=probs)
        return pop[indices]

    def _crossover(self, parent1, parent2):
        """模拟二进制交叉 (SBX)"""
        if np.random.rand() > self.crossover_prob:
            return parent1.copy(), parent2.copy()

        eta = 20  # 分布指数
        child1, child2 = parent1.copy(), parent2.copy()

        for i in range(self.dim):
            if np.random.rand() < 0.5:
                u = np.random.rand()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                lb, ub = self.bounds[i]
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
                child1[i] = np.clip(child1[i], lb, ub)
                child2[i] = np.clip(child2[i], lb, ub)

        return child1, child2

    def _mutate(self, individual):
        """多项式变异"""
        eta = 20
        for i in range(self.dim):
            if np.random.rand() < self.mutation_prob:
                u = np.random.rand()
                lb, ub = self.bounds[i]
                delta = (individual[i] - lb) / (ub - lb) if ub != lb else 0.5

                if u <= 0.5:
                    delta_q = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                individual[i] += delta_q * (ub - lb)
                individual[i] = np.clip(individual[i], lb, ub)
        return individual

    def run(self, verbose=True):
        """运行遗传算法"""
        pop = self._init_population()
        best_val_history = []

        for gen in range(self.max_gen):
            fitness = self._evaluate(pop)
            elite_idx = np.argsort(fitness)

            if self.maximize:
                elite_idx = elite_idx[::-1]
            best_val = fitness[elite_idx[0]]

            elites = pop[elite_idx[:self.n_elite]].copy()

            if verbose and gen % 40 == 0:
                print(f"  Gen {gen:3d}: Best = {best_val:.6f}")

            best_val_history.append(best_val)

            # 选择和繁殖
            selected = self._select(pop, fitness)
            offspring = []

            for i in range(0, len(selected) - 1, 2):
                c1, c2 = self._crossover(selected[i], selected[i+1])
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                offspring.extend([c1, c2])

            # 合并精英和后代
            offspring = np.array(offspring[:self.pop_size - self.n_elite])
            pop = np.vstack([elites, offspring])

        fitness = self._evaluate(pop)
        best_idx = np.argmax(fitness) if self.maximize else np.argmin(fitness)

        return {
            'x_best': pop[best_idx],
            'f_best': fitness[best_idx],
            'history': best_val_history,
        }


if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("遗传算法 (GA) 示例")
    print("=" * 60)

    # 测试函数：Rastrigin
    def rastrigin(x):
        return 20 + x[0]**2 + x[1]**2 - 10*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))

    bounds = [(-5, 5), (-5, 5)]
    ga = GeneticAlgorithm(rastrigin, bounds, pop_size=100, max_gen=200, maximize=False)
    result = ga.run(verbose=True)

    print(f"\n最优解: x1={result['x_best'][0]:.6f}, x2={result['x_best'][1]:.6f}")
    print(f"最优值: f={result['f_best']:.6f}")
    print(f"全局最优 ≈ (0, 0), f=0")
