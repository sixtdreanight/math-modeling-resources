# 线性规划

## 1. 模型原理

线性规划（Linear Programming）研究线性约束条件下线性目标函数的极值问题，是运筹学最基本也是最重要的模型。

### 标准形式

$$
\begin{aligned}
\min \quad & \mathbf{c}^T \mathbf{x} \\
\text{s.t.} \quad & \mathbf{A}\mathbf{x} \leq \mathbf{b} \\
& \mathbf{x} \geq \mathbf{0}
\end{aligned}
$$

### 核心概念

| 概念 | 说明 |
|------|------|
| 决策变量 $\mathbf{x}$ | 需要确定的最优量 |
| 目标函数 $\mathbf{c}^T\mathbf{x}$ | 需要最优化的指标（成本/利润等） |
| 约束条件 | 资源限制、技术要求等线性关系 |
| 可行域 | 所有满足约束的解构成的多面体 |
| 最优解 | 可行域的顶点之一 |

### 对偶理论

每个 LP 有一个对偶问题，两者最优值相等：

- 原问题：$\min \mathbf{c}^T\mathbf{x}$ s.t. $\mathbf{A}\mathbf{x} \geq \mathbf{b}$, $\mathbf{x} \geq 0$
- 对偶问题：$\max \mathbf{b}^T\mathbf{y}$ s.t. $\mathbf{A}^T\mathbf{y} \leq \mathbf{c}$, $\mathbf{y} \geq 0$

对偶变量的经济学意义：**影子价格**

## 2. 适用场景

| 赛事 | 典型场景 |
|------|---------|
| 国赛 B | 生产决策、资源分配、运输调度 |
| 美赛 D | 运筹优化、人员排班 |
| MathorCup | 企业供应链优化、投资组合 |
| 所有赛事 | 任何可量化为线性关系的资源分配问题 |

**典型赛题**：
- DVD在线租赁（国赛B，2005）
- 生产企业决策（国赛C，2021）
- 美赛D题运筹网络类

## 3. 建模步骤

1. 确定决策变量 —— 什么是可以被控制的？
2. 写出目标函数 —— 想要最大化/最小化什么？
3. 列出约束条件 —— 有哪些资源限制？
4. 求解 → Python: `scipy.optimize.linprog` / MATLAB: `linprog`
5. 灵敏度分析 —— 系���变动时解的变化？

## 4. 代码实现

Python: `algorithms/python/lp_solver.py` (封装 scipy.optimize.linprog)
MATLAB: 使用 `linprog(f, A, b, Aeq, beq, lb, ub)`

## 5. 注意事项

- **优点**：最优化理论最成熟的方法，求解快速且保证全局最优
- **缺点**：现实问题大多非线性，强行线性化可能失真
- **整数约束**：若变量必须为整数，使用整数规划（ILP/MILP），不能简单四舍五入
- **无解/无界**：若线性规划无可行解，检查约束条件是否矛盾；若无界，检查是否遗漏约束
- **灵敏度分析**：论文中必须分析参数变化对最优解的影响

## 6. 论文呈现建议

- 数学形式写清楚：决策变量定义、目标函数、约束条件
- 用表格列出最优解
- 必须做灵敏度分析（目标系数、约束右端项变化）
- 报告影子价格的经济解释（对偶变量>=0的资源为稀缺资源）

## 7. 参考资料

- Dantzig, G.B. (1963). Linear Programming and Extensions. Princeton University Press.
- references/ 目录下 MATLAB 优化相关代码
