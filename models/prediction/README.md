# 预测与预报模型

## 总览

预测是数学建模中最常见的需求之一。从赛题趋势看，预测模型正从传统统计方法向深度学习演进，但经典方法（灰色预测、ARIMA）因解释性强、计算快，仍是主力。

## 模型列表

| 模型 | 文件 | 适用条件 | 数据需求 |
|------|------|---------|---------|
| 灰色预测 | `grey_prediction.md` | 小样本(4-10个)、短期 | 极少 |
| 时间序列分析 | `time_series.md` | 时序数据、中短期 | 中等(30+) |
| 回归分析 | `regression.md` | 有自变量、关系明确 | 取决于自变量数 |
| 马尔可夫预测 | `markov.md` | 状态转移、无后效性 | 状态序列 |
| 神经网络预测 | `neural_network_prediction.md` | 复杂非线性、大量数据 | 大(数百+) |
| 深度学习预测 | `deep_learning_prediction.md` | 长序列、高维数据 | 大量(数千+) |
| 插值与拟合 | `interpolation_fitting.md` | 数据补全、曲线平滑 | 任意 |
| 组合预测 | `integrated_forecasting.md` | 提高精度、降低风险 | 同上 |

## 精度评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| MAE | `mean(abs(y_true - y_pred))` | 平均绝对误差 |
| MAPE | `mean(abs((y_true - y_pred) / y_true)) * 100` | 平均绝对百分比误差 |
| RMSE | `sqrt(mean((y_true - y_pred)^2))` | 均方根误差 |
| R² | `1 - SS_res / SS_tot` | 拟合优度 |
| 后验差比值 C | S2/S1 | 灰色预测专用 |

## 建模步骤

1. **数据审核** — 趋势、季节、噪声、缺失、异常
2. **数据预处理** — 平稳化、差分、归一化
3. **模型选择** — 根据数据量和特性选择
4. **参数估计** — 拟合模型
5. **残差检验** — 白噪声检验、Q检验
6. **预测与区间估计**

## 代码索引

Python 实现在 `algorithms/python/`：
- `grey_model.py` — GM(1,1)灰色预测
- `exponential_smoothing.py` — 指数平滑（一/二/三次）
- `arima.py` — ARIMA/SARIMA（statsmodels）
- `lstm_predictor.py` — Keras LSTM 预测

MATLAB 实现在 `algorithms/matlab/`：
- `grey_model.m` — GM(1,1)灰色预测
- `exponential_smoothing.m` — 指数平滑
- `arima_model.m` — ARIMA（Econometric Modeler/代码）

## 常见赛题

- 中国人口增长预测（国赛A，2007）
- 太阳影子定位（国赛A，2015）
- 美赛C题 数据洞察类
- 美赛E题 环境趋势预测
