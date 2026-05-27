# 分类与判别模型

## 总览

分类模型在国赛B/C题和美赛C题（大数据/数据洞察）中频繁出现。近年来深度学习和集成方法的引入使分类能力大幅提升。

## 模型列表

| 模型 | 文件 | 类型 | 特点 |
|------|------|------|------|
| 支持向量机(SVM) | `svm.md` | 有监督 | 小样本分类效果好、核技巧 |
| 决策树/随机森林 | `decision_tree_random_forest.md` | 有监督 | 可解释性强、不易过拟合 |
| XGBoost/LightGBM | `xgboost_lightgbm.md` | 有监督 | C题核心竞争力、表格数据SOTA |
| 聚类分析 | `clustering.md` | 无监督 | K-means/层次/DBSCAN/高斯混合 |
| Fisher判别 | `fisher_discriminant.md` | 有监督 | 经典线性判别、降维 |
| 逻辑回归 | `logistic_regression.md` | 有监督 | 二分类/多分类、概率输出 |
| 集成学习 | `ensemble_stacking.md` | 有监督 | Stacking/Blending/Bagging/Boosting |
| NLP基础 | `nlp_basics.md` | — | TF-IDF/LDA/Word2Vec、文本分类 |
| 贝叶斯方法 | `bayesian_methods.md` | 有监督/无监督 | 朴素贝叶斯、贝叶斯推断、MCMC |

## 模型选择建议

```
                    有标签？(y已知)
                       │
          ┌────────────┴────────────┐
          ▼ 是                     ▼ 否
    数据量规模？               聚类分析
          │                  K-means/DBSCAN
    ┌─────┴─────┐
    ▼           ▼
  小/中等      大数据
    │           │
  需要可解    追求精度
  释性？       │
  ┌──┴──┐   XGBoost/LightGBM
  ▼     ▼    Stacking
 SVM  决策树
 LR   随机森林
```

## C题数据分析全流程

```
原始数据 → 数据清洗 → 特征工程 → 模型训练 → 评估 → 预测/分类
               │           │           │
               ▼           ▼           ▼
           缺失值处理   TF-IDF      CV调参
           异常值检测   PCA降维      GridSearch
           标准化      特征选择     集成Stacking
```

## 代码索引

Python 实现在 `algorithms/python/`：
- `svm_classifier.py` — SVM分类（sklearn.svm）
- `kmeans.py` — K-means聚类
- `decision_tree.py` — 决策树
- `random_forest.py` — 随机森林
- `xgboost_model.py` — XGBoost
- `lightgbm_model.py` — LightGBM
- `fisher_discriminant.py` — Fisher判别
- `nlp_lda.py` — LDA主题模型
- `bayesian.py` — 贝叶斯推断

MATLAB 实现在 `algorithms/matlab/`：
- `svm_classifier.m` — SVM分类
- `kmeans.m` — K-means聚类
- `fisher_discriminant.m` — Fisher判别

## 常见赛题

- DNA序列分类（国赛B，2000）
- 碎纸片的拼接复原（国赛B，2013）
- 美赛C题（大数据/数据洞察，近年每年必出）
