# MCM/ICM 美赛指南

## 基本信息

| 项目 | 详情 |
|------|------|
| 全称 | Mathematical/Interdisciplinary Contest in Modeling |
| 主办方 | COMAP（美国数学及其应用联合会） |
| 时间 | 每年 1 月底 — 2 月初（4天4夜） |
| 报名费 | $100/队 |
| 参赛对象 | 本科生为主 |
| 语言 | 英文 |
| 官网 | https://www.comap.com/contests/mcm-icm |

## 六种题型

### MCM（数学建模竞赛）

| 题型 | 方向 | 典型特征 | 常用方法 |
|------|------|---------|---------|
| **A题 — 连续型** | 物理/工程 | 微分方程、连续系统、物理过程 | ODE/PDE、量纲分析、数值模拟 |
| **B题 — 离散型** | 优化/决策 | 离散事件、网络、调度 | 图论、整数规划、排队论、博弈论 |
| **C题 — 数据洞察** | 大数据 | 数据挖掘、模式识别、统计分析 | XGBoost、NLP、时间序列、聚类 |

### ICM（交叉学科建模竞赛）

| 题型 | 方向 | 典型特征 | 常用方法 |
|------|------|---------|---------|
| **D题 — 运筹学/网络科学** | 系统优化 | 网络流、调度、排队 | 优化、图论、排队论、蒙特卡洛 |
| **E题 — 可持续性** | 环境科学 | 环境评价、资源管理、生态 | 评价模型、系统动力学、微分方程 |
| **F题 — 政策** | 政策分析 | 社会政策、经济影响、法规 | 评价模型、博弈论、统计推断 |

## 奖项设置

| 奖项 | 英文 | 占比 | 含金量 |
|------|------|------|--------|
| 特等奖 | Outstanding Winner (O奖) | <1% | ★★★★★ |
| 特等奖提名 | Finalist (F奖) | 2% | ★★★★☆ |
| 一等奖 | Meritorious Winner (M奖) | 7% | ★★★★ |
| 二等奖 | Honorable Mention (H奖) | 24% | ★★★ |
| 成功参赛 | Successful Participant (S奖) | ~65% | ★★ |

## 2026年新规

1. **页数上限**：论文调整为 **25 页**（含摘要页和附录）
2. **AI 使用**：首次**明确允许使用 AI 工具**，但必须在附录中声明使用情况
3. **数据引用**：必须注明数据来源

## 选题策略

| 因素 | 建议 |
|------|------|
| 数学基础强 | A题（连续物理）或 B题（离散优化） |
| 编程能力强 | C题（大数据） |
| 写作能力强 | E/F题（文字量大，数理要求相对低） |
| 跨学科背景 | D题（运筹）或 E题（环境） |
| C题选的人最多 | 竞争激烈但题目最接近日常数据分析 |

## 论文结构

```
1. Summary（摘要页，独立一页，最重要！）
2. Table of Contents
3. Introduction
   3.1 Problem Background
   3.2 Problem Restatement
   3.3 Literature Review
   3.4 Our Work
4. Assumptions and Justifications
5. Notations（符号表）
6. Model I: xxx
   6.1 Model Establishment
   6.2 Model Solution
   6.3 Results and Analysis
7. Model II: xxx
   ...
8. Sensitivity Analysis（灵敏性分析，必做！）
9. Model Evaluation
   9.1 Strengths
   9.2 Weaknesses
10. Conclusion
11. References
12. Appendix
    12.1 Code
    12.2 Additional Figures/Tables
    12.3 AI Usage Report（2026新增）
```

## 备赛时间线

```
提前6个月 → 学习模型基础（本知识库 + 参考书）
提前4个月 → 精读 O 奖论文 5-10 篇，模仿写作风格
提前2个月 → 团队模拟赛（至少完整做 2-3 次）
提前1个月 → 准备模板、代码库、工具环境
提前1周  → 调整作息，检查工具
比赛期间 → 第一天：选题+理解+文献 → 第二天：建模+求解 → 第三天：写作 → 第四天：修改+翻译+润色+提交
```

## 相关资源

- 本知识库模板：`templates/latex/mcm_template.tex`
- COMAP 官网历年特等奖论文：https://www.comap.com/contests/mcm-icm
