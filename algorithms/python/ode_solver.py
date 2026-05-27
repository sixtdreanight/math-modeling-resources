# -*- coding: utf-8 -*-
"""
常微分方程 (ODE) 数值求解 — Python 实现
SciPy solve_ivp + 手动 RK4 + 相图绘制

用法：
    python ode_solver.py
"""

import numpy as np
from scipy.integrate import solve_ivp


def solve_ode(ode_func, t_span, y0, method='RK45', t_eval=None, **kwargs):
    """
    ODE 求解器（封装 scipy.solve_ivp）

    Parameters
    ----------
    ode_func : callable f(t, y) -> dy/dt
    t_span : tuple (t_start, t_end)
    y0 : array-like, 初值
    method : 'RK45' / 'RK23' / 'DOP853' / 'Radau' / 'BDF' / 'LSODA'
    t_eval : array, 评估点（None 则自动选择）

    Returns
    -------
    dict: t, y, success, message
    """
    result = solve_ivp(ode_func, t_span, y0, method=method,
                       t_eval=t_eval, **kwargs)
    return {
        't': result.t,
        'y': result.y,
        'success': result.success,
        'message': result.message,
    }


def rk4_step(f, t, y, h):
    """经典 4 阶 Runge-Kutta 单步"""
    y = np.asarray(y)
    k1 = np.asarray(f(t, y))
    k2 = np.asarray(f(t + h/2, y + h/2 * k1))
    k3 = np.asarray(f(t + h/2, y + h/2 * k2))
    k4 = np.asarray(f(t + h, y + h * k3))
    return y + h/6 * (k1 + 2*k2 + 2*k3 + k4)


def rk4_solver(f, t_span, y0, n_steps=1000):
    """手动 RK4 求解器（教学用）"""
    t0, tf = t_span
    h = (tf - t0) / n_steps
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((len(y0), n_steps + 1))
    y[:, 0] = y0

    for i in range(n_steps):
        y[:, i+1] = rk4_step(f, t[i], y[:, i], h)

    return t, y


# ==========================================
# 常见 ODE 模型
# ==========================================

def logistic_growth(t, N, r=0.5, K=1000):
    """Logistic 人口增长: dN/dt = rN(1 - N/K)"""
    return r * N * (1 - N / K)


def lotka_volterra(t, y, alpha=1, beta=0.1, gamma=0.5, delta=0.02):
    """捕食者-被捕食者模型 (Lotka-Volterra)
    y[0] = prey, y[1] = predator
    dy1/dt = alpha*y1 - beta*y1*y2
    dy2/dt = -gamma*y2 + delta*y1*y2
    """
    prey, predator = y
    d_prey = alpha * prey - beta * prey * predator
    d_predator = -gamma * predator + delta * prey * predator
    return np.array([d_prey, d_predator])


def sir_model(t, y, beta=0.3, gamma=0.1, N=1000):
    """SIR 传染病模型
    y[0]=S (易感者), y[1]=I (感染者), y[2]=R (康复者)
    dS/dt = -beta*S*I/N
    dI/dt = beta*S*I/N - gamma*I
    dR/dt = gamma*I
    """
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return np.array([dS, dI, dR])


def seir_model(t, y, beta=0.3, sigma=0.2, gamma=0.1, N=1000):
    """SEIR 模型（增加潜伏期 E）
    dS/dt = -beta*S*I/N
    dE/dt = beta*S*I/N - sigma*E
    dI/dt = sigma*E - gamma*I
    dR/dt = gamma*I
    """
    S, E, I, R = y
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return np.array([dS, dE, dI, dR])


def phase_portrait(f, x_range, y_range, n_grid=20):
    """相平面图（2D 系统的方向场）"""
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y_vec = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y_vec)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(n_grid):
        for j in range(n_grid):
            dydt = f(0, np.array([X[i, j], Y[i, j]]))
            U[i, j], V[i, j] = dydt[0], dydt[1]

    # 归一化
    magnitude = np.sqrt(U**2 + V**2) + 1e-10
    return X, Y, U/magnitude, V/magnitude


if __name__ == '__main__':
    np.set_printoptions(precision=4)

    print("=" * 60)
    print("ODE 数值求解示例")
    print("=" * 60)

    # --- 示例1: Logistic 增长 ---
    print("\n[Logistic 人口增长]")
    def f_logistic(t, y): return [logistic_growth(t, y[0])]

    result = solve_ode(f_logistic, (0, 30), [10], t_eval=np.linspace(0, 30, 100))
    print(f"  初值 10, 环境容量 1000")
    print(f"  t=30 时人口: {result['y'][0, -1]:.0f}")
    print(f"  成功: {result['success']}")

    # --- 示例2: Lotka-Volterra ---
    print("\n[Lotka-Volterra 捕食者-被捕食者]")
    def f_lv(t, y): return lotka_volterra(t, y)
    result = solve_ode(f_lv, (0, 50), [40, 9], t_eval=np.linspace(0, 50, 200))
    print(f"  初值: prey=40, predator=9")
    print(f"  t=50: prey={result['y'][0, -1]:.1f}, predator={result['y'][1, -1]:.1f}")

    # --- 示例3: SIR 模型 ---
    print("\n[SIR 传染病模型]")
    def f_sir(t, y): return sir_model(t, y, beta=0.3, gamma=0.1)
    result = solve_ode(f_sir, (0, 100), [999, 1, 0], t_eval=np.linspace(0, 100, 200))
    print(f"  β=0.3, γ=0.1, R0=β/γ={0.3/0.1:.1f}")
    print(f"  t=100: S={result['y'][0, -1]:.0f}, I={result['y'][1, -1]:.0f}, R={result['y'][2, -1]:.0f}")
    peak_idx = np.argmax(result['y'][1])
    print(f"  感染高峰: I_max={result['y'][1, peak_idx]:.0f} at t={result['t'][peak_idx]:.1f}")

    # --- 示例4: RK4 手动实现对比 ---
    print("\n[RK4 手动 vs SciPy]")
    t_rk4, y_rk4 = rk4_solver(f_logistic, (0, 20), [10], n_steps=100)
    print(f"  手动RK4 t=20: y={y_rk4[0, -1]:.2f}")
    result_scipy = solve_ode(f_logistic, (0, 20), [10])
    print(f"  SciPy t=20: y={result_scipy['y'][0, -1]:.2f}")
