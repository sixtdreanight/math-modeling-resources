# 机理分析模型（微分方程/物理建模）

## 总览

机理分析是国赛A题和美赛A题的核心方法。与数据驱动的方法不同，机理分析从问题背后的内在规律（物理/化学/生物/经济）出发建立数学模型。

## 模型列表

| 模型 | 文件 | 适用场景 |
|------|------|---------|
| ODE基础 | `ode_basics.md` | 带时间演化的连续系统 |
| PDE数值解 | `pde_numerical.md` | 空间+时间多维系统 |
| 稳定性分析 | `stability_analysis.md` | 系统长期行为判别 |
| 人口/生态模型 | `population_model.md` | 人口增长、种群竞争 |
| 传染病模型 | `infectious_disease.md` | SIR/SEIR、传染病传播 |
| 扩散模型 | `diffusion_model.md` | 热传导、污染物扩散 |
| 量纲分析 | `dimensional_analysis.md` | 物理量关系推导（A题高频） |
| 运动学/动力学 | `kinematic_model.md` | 运动轨迹、力学系统 |
| 流体力学基础 | `fluid_mechanics_basics.md` | 简化 N-S 方程应用 |

## 建模流程

```
问题分析 → 基本假设 → 建立方程 → 求解 → 参数估计 → 模型检验
    │          │           │          │          │          │
    ▼          ▼           ▼          ▼          ▼          ▼
 物理规律   合理简化   微分/差分   解析/数值   数据拟合   灵敏性分析
 守恒定律   无量纲化    方程组      解        参数识别   稳定性验证
```

## A题常见物理背景

| 场景 | 常用方程 | 涉及模型 |
|------|---------|---------|
| 热传导 | 傅里叶热传导方程 (PDE) | pde_numerical |
| 流体流动 | Navier-Stokes简化解 | fluid_mechanics_basics |
| 振动/波动 | 波动方程 | ode_basics |
| 扩散过程 | Fick扩散定律 | diffusion_model |
| 动力学 | 牛顿第二定律 | kinematic_model |
| 化学反应 | 反应速率方程 | ode_basics |

## 数值方法速查

| 方法 | 适用 | Python库 | MATLAB函数 |
|------|------|---------|-----------|
| 欧拉法 | 简单ODE | 自写 | 自写 |
| 龙格-库塔法(RK4/RK45) | ODE求解 | scipy.solve_ivp | ode45 |
| 有限差分法 | PDE | 自写 | 自写 |
| 有限元法 | 复杂PDE | FEniCS | PDE Toolbox |

## 代码索引

Python 实现在 `algorithms/python/`：
- `ode_solver.py` — ODE求解（scipy.solve_ivp）
- `finite_difference.py` — 有限差分法解 PDE
- `sir_model.py` — SIR/SEIR 传染病模型

MATLAB 实现在 `algorithms/matlab/`：
- `ode_solver.m` — ODE求解（ode45/ode15s）
- `finite_difference.m` — 有限差分法

## 常见赛题

- 嫦娥三号软着陆轨道设计（国赛A，2014）
- 制动器试验台控制方法（国赛A，2009）
- 储油罐变位识别（国赛A，2010）
- 美赛A题各年连续问题
- 传染病传播预测（2020年前后高频出现）
