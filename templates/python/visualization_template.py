# -*- coding: utf-8 -*-
"""
数学建模 — 论文级可视化模板
功能：折线图 / 散点图 / 柱状图 / 热力图 / 3D 曲面 / 子图布局 / 双 Y 轴
Python 3.7+, matplotlib, seaborn
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# 全局配置 — 论文级美观设置
# ==========================================

plt.rcParams.update({
    # 中文字体支持
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,   # 负号正常显示
    # 尺寸和分辨率
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    # 字体
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    # 样式
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    # 线条
    'lines.linewidth': 2,
    'lines.markersize': 6,
})

# 论文常用配色
COLORS = {
    'blue': '#2563EB',
    'red': '#DC2626',
    'green': '#16A34A',
    'orange': '#EA580C',
    'purple': '#7C3AED',
    'teal': '#0D9488',
    'pink': '#DB2777',
}

PALETTE = list(COLORS.values())


def set_style(style='paper'):
    """切换样式: 'paper' 论文 / 'presentation' 演示 / 'dark' 深色"""
    if style == 'presentation':
        plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18})
    elif style == 'dark':
        plt.style.use('dark_background')
    else:
        plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 14})


# ==========================================
# 1. 折线图
# ==========================================

def plot_line(x, y_list, labels=None, xlabel='X', ylabel='Y',
              title=None, filename=None):
    """
    多线折线图
    y_list: 多个 y 数据列表 [[y1], [y2], ...]
    labels: 对应图例标签
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = labels or [f'Series {i+1}' for i in range(len(y_list))]

    for i, y in enumerate(y_list):
        ax.plot(x, y, color=PALETTE[i % len(PALETTE)],
                marker='o', markersize=4, label=labels[i],
                markevery=max(1, len(x) // 10))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=True, fancybox=True, shadow=True)

    if filename:
        fig.savefig(filename)
    return fig, ax


# ==========================================
# 2. 散点图（含拟合线）
# ==========================================

def plot_scatter(x, y, fit_degree=None, xlabel='X', ylabel='Y',
                 title=None, filename=None, alpha=0.6):
    """散点图 + 可选多项式拟合线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=alpha, color=COLORS['blue'], s=30,
               edgecolors='white', linewidth=0.5)

    if fit_degree is not None:
        z = np.polyfit(x, y, fit_degree)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(x), max(x), 200)
        ax.plot(x_smooth, p(x_smooth), color=COLORS['red'],
                linewidth=2.5, label=f'Fit (degree={fit_degree})')
        # 添加 R²
        r2 = 1 - np.sum((y - p(x))**2) / np.sum((y - np.mean(y))**2)
        ax.legend(title=f'$R^2 = {r2:.4f}$')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if filename:
        fig.savefig(filename)
    return fig, ax


# ==========================================
# 3. 柱状图
# ==========================================

def plot_bar(categories, values, xlabel='Category', ylabel='Value',
             title=None, filename=None, color=None, horizontal=False):
    """柱状图"""
    fig, ax = plt.subplots(figsize=(8, 5))
    color = color or PALETTE[:len(categories)]

    if horizontal:
        ax.barh(categories, values, color=color, edgecolor='white')
    else:
        ax.bar(categories, values, color=color, edgecolor='white')

    # 柱顶标注数值
    for i, v in enumerate(values):
        if horizontal:
            ax.text(v + max(values) * 0.01, i, f'{v:.2f}', va='center')
        else:
            ax.text(i, v + max(values) * 0.01, f'{v:.2f}', ha='center', fontsize=9)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if filename:
        fig.savefig(filename)
    return fig, ax


# ==========================================
# 4. 热力图（相关矩阵 / 矩阵可视化）
# ==========================================

def plot_heatmap(data, x_labels=None, y_labels=None, title=None,
                 cmap='RdBu_r', annot=True, filename=None):
    """
    热力图
    data: 2D 数组或 DataFrame
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, aspect='auto', cmap=cmap)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)

    if annot and isinstance(data, np.ndarray):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = 'white' if abs(data[i, j]) > np.max(np.abs(data)) / 2 else 'black'
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                        fontsize=8, color=color)

    if x_labels:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    if y_labels:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

    if title:
        ax.set_title(title)

    if filename:
        fig.savefig(filename)
    return fig, ax


# ==========================================
# 5. 3D 曲面图
# ==========================================

def plot_surface_3d(X, Y, Z, title=None, filename=None):
    """3D 曲面图"""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.9,
                           linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)

    if filename:
        fig.savefig(filename)
    return fig, ax


# ==========================================
# 6. 子图布局（多图并排）
# ==========================================

def plot_subplots(plot_funcs, nrows, ncols, figsize=None,
                  titles=None, filename=None):
    """
    通用子图布局
    plot_funcs: 列表，每个元素为 (func, args, kwargs)
      其中 func 接受 (fig, ax, **kwargs) 在指定子图上画图
    """
    figsize = figsize or (5 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    titles = titles or [f'Subplot {i+1}' for i in range(len(plot_funcs))]

    for i, (func, args, kwargs) in enumerate(plot_funcs):
        func(axes[i], *args, **(kwargs or {}))
        axes[i].set_title(titles[i] if i < len(titles) else '')

    # 隐藏多余的子图
    for j in range(len(plot_funcs), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if filename:
        fig.savefig(filename)
    return fig, axes


# ==========================================
# 7. 双 Y 轴图
# ==========================================

def plot_double_y(x, y1, y2, label1='Y1', label2='Y2',
                  title=None, filename=None):
    """双Y轴图 — 适用于不同量纲的两组数据"""
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1, color2 = COLORS['blue'], COLORS['red']

    ax1.plot(x, y1, color=color1, marker='o', markersize=4)
    ax1.set_xlabel('X')
    ax1.set_ylabel(label1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color=color2, marker='s', markersize=4)
    ax2.set_ylabel(label2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # 图例合并
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color1, marker='o', label=label1),
        Line2D([0], [0], color=color2, marker='s', label=label2),
    ]
    ax1.legend(handles=legend_elements)

    if title:
        ax1.set_title(title)

    if filename:
        fig.savefig(filename)
    return fig, ax1, ax2


# ==========================================
# 8. 灵敏性分析专用图
# ==========================================

def plot_sensitivity(param_name, param_values, output_values,
                     baseline=None, xlabel=None, ylabel='Output',
                     title=None, filename=None):
    """
    灵敏性分析单参数图
    baseline: 基准参数值，画参考线
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(param_values, output_values, color=COLORS['blue'],
            marker='o', linewidth=2)

    if baseline is not None:
        ax.axvline(x=baseline, color=COLORS['red'], linestyle='--',
                   linewidth=1.5, label=f'Baseline = {baseline}')
        ax.legend()

    ax.set_xlabel(xlabel or param_name)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if filename:
        fig.savefig(filename)
    return fig, ax


# ==========================================
# 示例运行
# ==========================================

if __name__ == '__main__':
    np.random.seed(42)

    # --- 折线图示例 ---
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x) + np.random.normal(0, 0.1, 50)
    y2 = np.cos(x) + np.random.normal(0, 0.1, 50)
    fig, ax = plot_line(x, [y1, y2], labels=['sin(x)', 'cos(x)'],
                        title='Line Plot Example', xlabel='Time', ylabel='Value')
    plt.show()

    # --- 散点图示例 ---
    x_scatter = np.random.uniform(0, 10, 100)
    y_scatter = 2 * x_scatter + 1 + np.random.normal(0, 1.5, 100)
    fig, ax = plot_scatter(x_scatter, y_scatter, fit_degree=1,
                           title='Scatter with Linear Fit')
    plt.show()

    print("可视化模板加载完成。所有函数可独立调用。")
