# 通用辅助方法

## 总览

这些方法不单独构成模型，但在所有赛题中都会用到，是建模的"基础设施"。其中**灵敏性分析**是论文评审的加分项，建议每题都做。

## 模型列表

| 模型 | 文件 | 用途 | 重要性 |
|------|------|------|--------|
| 灵敏性分析 | `sensitivity_analysis.md` | 检验模型对参数变化的鲁棒性 | ★★★★★ |
| 数据预处理 | `data_preprocessing.md` | 缺失值、异常值、标准化、编码 | ★★★★★ |
| 特征工程 | `feature_engineering.md` | 降维、选择、构造 | ★★★★ |
| 相关性分析 | `correlation_analysis.md` | Pearson/Spearman/Kendall/偏相关 | ★★★★ |
| 稳健性分析 | `robust_analysis.md` | 更换变量度量、更换样本、更换方法 | ★★★★ |
| 不确定性分析 | `uncertainty_analysis.md` | 参数不确定性传播、置信区间 | ★★★ |

## 灵敏性分析：必做加分项

**为什么重要**：美赛和国赛评审中，"模型的灵敏性/稳健性分析"是独立评分点。不做灵敏性分析 = 放弃这部分分数。

常用方法对比：

| 方法 | 适用范围 | 计算成本 | 说明 |
|------|---------|---------|------|
| OAT(一次变一个) | 参数少(<5) | 低 | 最简单，但忽略参数交互 |
| Morris筛选法 | 参数多(5-50) | 中 | 筛选有影响力的参数 |
| Sobol指数 | 任何参数数 | 高 | 总效应和主效应的完整分解 |
| LHS(拉丁超立方) | 参数多 | 中 | 高效采样，常配合Sobol |
| FAST/eFAST | 参数多 | 高 | 周期采样、傅里叶分解 |

## 数据预处理流程

```
原始数据
│
├── 缺失值处理
│   ├── 删除（缺失>50%）
│   ├── 均值/中位数填充
│   ├── KNN填充
│   └── 多重插补(MICE)
│
├── 异常值检测
│   ├── 3σ准则
│   ├── IQR（箱线图）
│   ├── Grubbs检验
│   └── 马氏距离
│
├── 标准化/归一化
│   ├── Min-Max（到[0,1]）
│   ├── Z-score（到标准正态）
│   └── Robust Scaler（对异常值鲁棒）
│
└── 编码
    ├── One-Hot（无序类别）
    └── Label/Ordinal（有序类别）
```

## 代码索引

Python 实现在 `algorithms/python/`：
- `sensitivity_analysis.py` — 灵敏性分析（Morris + Sobol）
- `normalize.py` — 数据标准化
- `handle_missing.py` — 缺失值处理
- `outlier_detection.py` — 异常值检测
- `pca.py` — 主成分分析降维

MATLAB 实现在 `algorithms/matlab/`：
- `normalize.m` — 数据标准化
- `handle_missing.m` — 缺失值处理
- `pca_analysis.m` — 主成分分析
