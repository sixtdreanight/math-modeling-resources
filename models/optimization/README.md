# 优化与控制模型

## 总览

优化问题在数学建模竞赛中**每届必出**，是最高频的模型类别。核心思想：在约束条件下寻找使目标函数最优的决策方案。

## 模型列表

| 模型 | 文件 | 赛题类型 | 关键词 |
|------|------|---------|--------|
| 线性规划 | `linear_programming.md` | 国赛B、美赛B/D | 单纯形法、对偶理论、灵敏度分析 |
| 整数规划 | `integer_programming.md` | 国赛B、美赛B | 分支定界、0-1规划、指派问题、蒙特卡洛 |
| 非线性规划 | `nonlinear_programming.md` | 国赛A/B、美赛A/B | KKT条件、拉格朗日乘子法、罚函数法 |
| 动态规划 | `dynamic_programming.md` | 国赛B、美赛B | 最优性原理、背包问题、设备更新 |
| 多目标优化 | `multi_objective.md` | 美赛E/B | Pareto最优、NSGA-II、加权法、ε-约束法 |
| 图论优化 | `graph_theory.md` | 国赛B、美赛B/D | Dijkstra/Floyd、Kruskal/Prim、最大流、TSP |
| 排队论 | `queuing_theory.md` | 美赛D、国赛B | M/M/1、M/M/s、Little公式、排队优化 |
| 博弈论 | `game_theory.md` | 美赛B/F、MathorCup | 纳什均衡、演化博弈、拍卖机制 |
| 投资组合优化 | `portfolio_optimization.md` | 大湾区杯、国赛C | Markowitz模型、有效前沿、VaR |
| 调度优化 | `scheduling.md` | MathorCup、美赛B | 车间调度、人员排班、约束满足 |
| 智能优化算法 | `intelligent_algorithm.md` | 所有赛题 | 遗传算法、粒子群、模拟退火、蚁群 |

## 模型选择决策树

```
                    问题类型
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      连续变量      离散变量      网络结构
          │            │            │
    ┌─────┴──┐    ┌───┴───┐    ┌──┴──┐
    │线性目标│    │小规模 │    │图论 │
    └────┬───┘    └───┬───┘    └──┬──┘
         │            │           │
    线性规划      分支定界     graph_theory
    ┌────┴───┐      ┌─┴──┐     Dijkstra/Floyd
    │非线性  │      │大规模│   最小生成树/最大流
    │目标    │      └─┬──┘
    └──┬───┘         │
    非线性规划      智能算法
    KKT/罚函数      GA/PSO/SA
```

## 代码索引

Python 实现在 `algorithms/python/`：
- `lp_solver.py` — 线性规划（scipy.optimize.linprog）
- `integer_programming.py` — 整数规划（PuLP）
- `genetic_algorithm.py` — 遗传算法
- `particle_swarm.py` — 粒子群算法
- `simulated_annealing.py` — 模拟退火
- `dijkstra.py` / `floyd.py` — 最短路径
- `kruskal.py` / `prim.py` — 最小生成树

MATLAB 实现在 `algorithms/matlab/`：
- `lp_solver.m` — 线性规划（linprog/fmincon）
- `ga_optimize.m` — 遗传算法（ga 工具箱）
- `dijkstra.m` / `floyd.m` — 最短路径

## 相关赛事指南

- [美赛 B 题（离散/优化）](../../competitions/mcm-icm.md)
- [国赛 B 题（优化决策）](../../competitions/cumcm.md)
- [MathorCup（企业优化题）](../../competitions/mathorcup.md)
