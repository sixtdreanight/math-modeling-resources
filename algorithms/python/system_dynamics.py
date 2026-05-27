# -*- coding: utf-8 -*-
"""
系统动力学 (System Dynamics) — Python 实现
存量-流量图 / 反馈回路 / 一阶延迟

用法：
    python system_dynamics.py
"""

import numpy as np
from scipy.integrate import solve_ivp


class StockFlowModel:
    """简易存量-流量建模框架"""

    def __init__(self, stocks, flows, auxiliaries, params):
        """
        Parameters
        ----------
        stocks : dict {name: init_value}
            存量定义
        flows : dict {name: func(t, stocks, aux, params) -> float}
            流量定义（速率方程）
        auxiliaries : dict {name: func(t, stocks, aux, params) -> float}
            辅助变量
        params : dict
            模型参数
        """
        self.stock_names = list(stocks.keys())
        self.init_values = np.array(list(stocks.values()))
        self.flows = flows
        self.auxiliaries = auxiliaries
        self.params = params

    def _system_ode(self, t, y):
        stocks = {name: y[i] for i, name in enumerate(self.stock_names)}

        # 计算辅助变量
        aux = {}
        for name, func in self.auxiliaries.items():
            aux[name] = func(t, stocks, aux, self.params)

        # 计算各存量的净流量
        net_rates = np.zeros(len(self.stock_names))
        for i, stock_name in enumerate(self.stock_names):
            # 入流和出流的命名约定: inflow_stockname, outflow_stockname
            for flow_name, flow_func in self.flows.items():
                # 简单约定: 流量名以 "+stockname" 或 "-stockname" 结尾
                # 或显式指定 net_rates
                pass

        # 简化方案：直接计算每个存量的 d/dt
        return np.zeros_like(y)

    def simulate(self, t_span, t_eval=None):
        """运行仿真"""

        def default_ode(t, y):
            stocks = {name: y[i] for i, name in enumerate(self.stock_names)}
            aux = {}
            for name, func in self.auxiliaries.items():
                aux[name] = func(t, stocks, aux, self.params)

            # 计算所有流量
            flow_vals = {}
            for name, func in self.flows.items():
                flow_vals[name] = func(t, stocks, aux, self.params)

            rates = np.zeros(len(self.stock_names))
            for flow_name, flow_val in flow_vals.items():
                for i, stock_name in enumerate(self.stock_names):
                    if flow_name.startswith(f'+{stock_name}'):
                        rates[i] += flow_val
                    elif flow_name.startswith(f'-{stock_name}'):
                        rates[i] -= flow_val
            return rates

        result = solve_ivp(default_ode, t_span, self.init_values,
                          t_eval=t_eval, method='RK45')

        return {
            't': result.t,
            'stocks': {self.stock_names[i]: result.y[i] for i in range(len(self.stock_names))},
        }


def bass_diffusion(p=0.03, q=0.38, M=1000, T=50):
    """
    Bass 扩散模型（创新扩散）
    用于新产品/技术采纳预测

    dA/dt = (p + q*A/M) * (M - A)
    A: 累计采纳者, p: 创新系数, q: 模仿系数, M: 市场潜力
    """
    def ode(t, y):
        A = y[0]
        dA = (p + q * A / M) * (M - A)
        return [dA]

    result = solve_ivp(ode, [0, T], [0], t_eval=np.linspace(0, T, 200))
    # 新采纳者 = dA/dt
    new_adopters = np.diff(result.y[0], prepend=0)
    # 修正首尾
    new_adopters[0] = result.y[0, 0]

    return result.t, result.y[0], new_adopters


def s__growth_model(capacity=1000, growth_rate=0.3, decay_rate=0.1,
                    initial_capital=100, T=60):
    """
    经典存量-流量模型：资本积累
    Stock: Capital
    Inflow: Investment = growth_rate * Capital
    Outflow: Depreciation = decay_rate * Capital
    """
    def ode(t, y):
        capital = y[0]
        investment = growth_rate * capital
        depreciation = decay_rate * capital
        impact = min(capital / capacity, 2.0)  # 接近容量时投资效率下降
        return [investment * (2 - impact) - depreciation]

    result = solve_ivp(ode, [0, T], [initial_capital],
                      t_eval=np.linspace(0, T, 200))
    return result.t, result.y[0]


if __name__ == '__main__':
    np.set_printoptions(precision=4)

    print("=" * 60)
    print("系统动力学示例")
    print("=" * 60)

    # --- Bass 扩散模型 ---
    t, adopters, new_adopters = bass_diffusion(p=0.03, q=0.38, M=1000, T=30)
    peak_idx = np.argmax(new_adopters)
    print(f"\n[Bass 扩散模型] p=0.03, q=0.38, M=1000")
    print(f"  t=30 累计采纳: {adopters[-1]:.0f}")
    print(f"  新采纳高峰: t={t[peak_idx]:.1f}, 数量={new_adopters[peak_idx]:.0f}")

    # --- 资本积累 ---
    t_cap, capital = s__growth_model(capacity=1000, growth_rate=0.3, decay_rate=0.1,
                                     initial_capital=100, T=50)
    print(f"\n[资本积累] r=0.3, 折旧=0.1, K0=100, 容量=1000")
    print(f"  t=50: K={capital[-1]:.1f}")
    steady_state = (0.3 - 0.1) / (0.3 / 500) if False else 0
    print(f"  稳态: 增长趋于平衡时有 K ≈ {capital[-1]:.0f}")
