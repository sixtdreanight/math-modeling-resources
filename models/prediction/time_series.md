# 时间序列分析（指数平滑 + ARIMA）

## 1. 模型原理

时间序列分析通过挖掘数据自身的时序规律（趋势、季节、周期）进行预测，不需要外部自变量。

### 指数平滑法

| 类型 | 适用 | 公式 |
|------|------|------|
| 一次平滑 | 无趋势无季节性 | $\hat{y}_{t+1} = \alpha y_t + (1-\alpha)\hat{y}_t$ |
| 二次平滑 (Holt) | 有趋势无季节性 | 增加趋势项 $b_t$ |
| 三次平滑 (Holt-Winters) | 有趋势有季节性 | 增加季节项 $s_t$ |

### ARIMA(p,d,q)

- **AR(p)**：自回归，$y_t = c + \phi_1 y_{t-1} + ... + \phi_p y_{t-p} + \epsilon_t$
- **I(d)**：差分阶数 d 使序列平稳
- **MA(q)**：移动平均，$y_t = \mu + \epsilon_t + \theta_1 \epsilon_{t-1} + ... + \theta_q \epsilon_{t-q}$

### SARIMA（季节版本）

SARIMA(p,d,q)(P,D,Q)s，其中 P/D/Q 为季节项，s 为周期长度。

## 2. 建模步骤

1. **绘制时序图** —— 观察趋势和季节性
2. **平稳性检验** —— ADF 检验（p<0.05 则平稳）
3. **若不平滑** —— 差分。通常 d=1 或 2
4. **定阶** —— ACF/PACF 图观察，或 AIC 自动搜索
5. **参数估计** —— 最大似然估计
6. **残差诊断** —— 残差应为白噪声（Ljung-Box p>0.05）
7. **预测 + 置信区间**

## 3. 模型选择指南

| 条件 | 推荐模型 |
|------|---------|
| 数据 < 10个 | 灰色预测 GM(1,1) |
| 数据 10-50, 无季节 | 指数平滑 |
| 数据 > 30, 有趋势 | ARIMA |
| 数据 > 30, 有季节 | SARIMA 或 Holt-Winters |
| 数据 > 100, 复杂 | LSTM/GRU |

## 4. 代码实现

Python: `algorithms/python/arima.py` (自动定阶 + 白噪声诊断)
MATLAB: `algorithms/matlab/arima_forecast.m`

## 5. 注意事项

- **优点**：仅需时序数据，无需外部变量；解释性强
- **缺点**：无法捕捉非线性、突变式变化
- **常见坑**：
  - 不检验平稳性直接建模 → 可能伪回归
  - 差分过多 → 丢失信息
  - 不看残差 → 模型可能欠拟合
- **ARIMA 不适用于**：有结构断点、长记忆性（可考虑计量方法）

## 6. 论文呈现建议

- ADF 检验表
- ACF/PACF 相关图（标注显著性边界）
- 拟合值与实际值的时序对比图
- 预测区间的带状图（置信区间）
- 残差白噪声检验结果

## 7. 参考资料

- Box, G.E.P., Jenkins, G.M. (1976). Time Series Analysis: Forecasting and Control.
- Hyndman, R.J., Athanasopoulos, G. (2018). Forecasting: Principles and Practice.
