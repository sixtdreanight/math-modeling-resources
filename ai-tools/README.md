# AI 辅助数学建模工具集

## 概述

2026 年起美赛、国赛等主流赛事相继放开 AI 工具使用。AI 不是"自动驾驶仪"，而是**第四位队友**——帮你查资料、写代码、润色文字，但核心建模和判断仍然由人主导。

## 目录结构

```
ai-tools/
├── README.md                    ← 你在这里
├── prompts/                     # 提示词模板库
│   ├── stage-prompts.md         # 分阶段提示词（分析→建模→代码→写作）
│   └── anti-ai-writing.md       # 去AI化写作40个技巧
├── skills/                      # Claude Code / Codex CLI Skill 配置
│   └── claude-skill.md          # 数学建模 Skill 安装与使用指南
├── workflow/                    # AI 辅助全流程
│   └── ai-workflow.md           # 96小时 AI 辅助时间线与操作指南
└── references/                  # 外部参考
```

## 快速开始

### 1. 安装数学建模 Skill（推荐）

在项目根目录创建 `.claude/skills/math-modeling/SKILL.md`，让 Claude Code 在建模时自动加载专业知识。

详见 [`skills/claude-skill.md`](skills/claude-skill.md)

### 2. 使用提示词模板

按竞赛阶段选用提示词：
- **选题阶段** → [`stage-prompts.md`](prompts/stage-prompts.md) 第 1 节
- **建模阶段** → 第 2 节
- **代码阶段** → 第 3 节
- **写作阶段** → 第 4 节
- **去 AI 味** → [`anti-ai-writing.md`](prompts/anti-ai-writing.md)

### 3. 遵循 AI 使用规范

- 美赛 2026：论文后必须附 **AI 使用报告**（使用工具、目的、范围）
- 核心建模和推导不可完全依赖 AI
- 所有 AI 生成的公式和代码必须人工验证

## 同类开源项目参考

| 项目 | 地址 | 特点 |
|------|------|------|
| **mathmodel-skill** | `github.com/handsomeZR-netizen/mathmodel-skill` | 10阶段工程化 Skill，支持 Claude Code/Codex CLI，含 91 篇获奖论文数据 |
| **MM-Agent** | `github.com/usail-hkust/LLM-MM-Agent` | NeurIPS 论文方案，GPT-4o 驱动，2025 美赛 Finalist |
| **categorical-meta-prompting** | `github.com/manutej/categorical-meta-prompting` | 范畴论基础的结构化提示词工程 |

## AI 使用红线

- 绝不直接复制 AI 生成的段落到论文
- 绝不无验证使用 AI 生成的代码/公式
- 绝不让 AI 编造参考文献
- 必须提交真实的 AI 使用报告
