# -*- coding: utf-8 -*-
"""
灰色预测 GM(1,1) — Python 实现
支持：基本 GM(1,1) / 残差修正 / 级比检验 / 后验差检验

用法：
    python grey_model.py
"""

import numpy as np


def gm11(series, forecast_num=1):
    """
    GM(1,1) 灰色预测

    Parameters
    ----------
    series : array-like
        原始序列（至少4个数据点）
    forecast_num : int
        预测期数

    Returns
    -------
    dict: {
        'forecast': 预测值列表,
        'fitted': 拟合值列表,
        'residuals': 残差,
        'relative_errors': 相对误差 (%),
        'C': 后验差比值,
        'P': 小误差概率,
        'grade': 精度等级 (1-4),
        'params': {'a': 发展系数, 'b': 灰色作用量}
    }
    """
    X0 = np.asarray(series, dtype=float)
    n = len(X0)

    # 级比检验
    lambda_k = X0[:-1] / X0[1:]
    lower = np.exp(-2 / (n + 1))
    upper = np.exp(2 / (n + 1))
    lambdas_ok = np.all((lambda_k > lower) & (lambda_k < upper))

    # 1-AGO（一次累加生成）
    X1 = np.cumsum(X0)

    # 紧邻均值生成 Z1
    Z1 = 0.5 * (X1[1:] + X1[:-1])

    # 最小二乘估计 [a, b]^T = (B^T B)^{-1} B^T Y
    B = np.column_stack([-Z1, np.ones(n - 1)])
    Y = X0[1:]
    ab = np.linalg.lstsq(B, Y, rcond=None)[0]
    a, b = ab[0], ab[1]

    # 发育系数为零的防护
    if abs(a) < 1e-10:
        a = 1e-10

    # 时间响应式
    k_values = np.arange(n + forecast_num)

    # 预测 AGO 值
    X1_pred = (X0[0] - b / a) * np.exp(-a * k_values) + b / a

    # 还原（一次累减）
    X0_pred = np.zeros_like(X1_pred)
    X0_pred[0] = X0[0]
    X0_pred[1:] = X1_pred[1:] - X1_pred[:-1]

    # 拟合值和预测值
    fitted = X0_pred[:n]
    forecast = X0_pred[n:]

    # 残差分析
    residuals = X0 - fitted
    relative_errors = np.abs(residuals / (X0 + 1e-10)) * 100

    # 后验差检验
    S1 = np.std(X0)          # 原始序列标准差
    S2 = np.std(residuals)   # 残差标准差
    C = S2 / S1 if S1 > 0 else 0  # 后验差比值

    # 小误差概率
    mean_residual = np.mean(residuals)
    P_count = np.sum(np.abs(residuals - mean_residual) < 0.6745 * S1)
    P = P_count / n

    # 精度等级
    if C < 0.35 and P > 0.95:
        grade = 1
        grade_name = '好 (Good)'
    elif C < 0.50 and P > 0.80:
        grade = 2
        grade_name = '合格 (Qualified)'
    elif C < 0.65 and P > 0.70:
        grade = 3
        grade_name = '勉强 (Barely)'
    else:
        grade = 4
        grade_name = '不合格 (Unqualified)'

    return {
        'forecast': forecast,
        'fitted': fitted,
        'residuals': residuals,
        'relative_errors': relative_errors,
        'C': C,
        'P': P,
        'grade': grade,
        'grade_name': grade_name,
        'lambdas_ok': lambdas_ok,
        'params': {'a': a, 'b': b},
    }


def post_residual_gm11(series, forecast_num=1):
    """
    残差修正 GM(1,1)：对残差序列再做灰色预测修正
    """
    # 第一次拟合
    result = gm11(series, forecast_num)
    residuals = result['residuals']

    # 对残差绝对值做 GM(1,1)
    abs_residuals = np.abs(residuals)
    if np.all(abs_residuals[1:] > 0) and len(abs_residuals) >= 4:
        try:
            residual_result = gm11(abs_residuals, forecast_num)
            sign = np.sign(result['residuals'])
            correction = sign[0] * residual_result['fitted']  # 符号修正

            result['fitted'] = result['fitted'] + correction
            result['residuals'] = series - result['fitted']
            result['relative_errors'] = np.abs(result['residuals'] / series) * 100
        except np.linalg.LinAlgError:
            pass  # 残差序列不可建模

    return result


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    # 示例：10年人口数据（单位：万）
    data = np.array([89677, 90859, 92148, 93769, 95050,
                     96450, 97812, 99256, 100729, 102034])

    print("=" * 60)
    print("灰色预测 GM(1,1) 示例")
    print("=" * 60)
    print(f"原始序列: {data}")

    result = gm11(data, forecast_num=3)

    print(f"\n级比检验: {'通过' if result['lambdas_ok'] else '未通过（建议做平移变换或使用其他模型）'}")
    print(f"发展系数 a: {result['params']['a']:.6f}")
    print(f"灰色作用量 b: {result['params']['b']:.2f}")

    # 发展系数诊断
    a_val = result['params']['a']
    if abs(a_val) < 0.3:
        print("  → a ∈ (-0.3, 0.3), 适合中长期预测")
    elif 0.3 <= abs(a_val) <= 0.5:
        print("  → a ∈ [0.3, 0.5], 仅适合短期预测")
    else:
        print("  → |a| > 0.5, 不建议用于预测，只能定性分析")

    print(f"\n后验差检验:")
    print(f"  后验差比值 C: {result['C']:.4f}")
    print(f"  小误差概率 P: {result['P']:.4f}")
    print(f"  精度等级: {result['grade']}级 — {result['grade_name']}")

    print(f"\n拟合结果:")
    print(f"  拟合值:     {result['fitted'].astype(int)}")
    print(f"  残差:       {result['residuals'].astype(int)}")
    print(f"  相对误差(%): {result['relative_errors'].round(2)}")
    print(f"  平均相对误差: {np.mean(result['relative_errors']):.2f}%")

    print(f"\n预测值（未来3期）: {result['forecast'].astype(int)}")
