# -*- coding: utf-8 -*-
"""
时间序列分析 (ARIMA/SARIMA) — Python 实现
基于 statsmodels，含自动定阶 AIC 搜索 + 残差诊断

用法：
    需要安装: pip install statsmodels
    python arima.py
"""

import numpy as np


def auto_arima(series, max_p=5, max_d=2, max_q=5, criterion='aic', verbose=False):
    """
    自动 ARIMA 定阶（网格搜索最小 AIC/BIC）

    Returns
    -------
    best_order : tuple (p, d, q)
    best_model : statsmodels ARIMA 对象
    best_aic : float
    """
    from statsmodels.tsa.arima.model import ARIMA

    best_aic = np.inf
    best_order = (0, 0, 0)
    best_model = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    val = fitted.aic if criterion == 'aic' else fitted.bic
                    if val < best_aic:
                        best_aic = val
                        best_order = (p, d, q)
                        best_model = fitted
                        if verbose:
                            print(f"  ARIMA{p,d,q}: {criterion.upper()}={val:.1f} ✓ (新最优)")
                except (ValueError, np.linalg.LinAlgError, Exception) as e:
                    if verbose:
                        print(f"  ARIMA{p,d,q}: failed ({e})")

    return best_order, best_model, best_aic


def arima_forecast(series, order=None, forecast_num=5, auto_order=True, max_p=5, max_d=2, max_q=5):
    """
    ARIMA 预测

    Parameters
    ----------
    series : array-like
        时间序列
    order : tuple (p, d, q), optional
        ARIMA 阶数，None 则自动定阶
    forecast_num : int
        预测步数
    auto_order : bool
        是否自动搜索最优阶数
    max_p, max_d, max_q : int
        自动搜索范围

    Returns
    -------
    dict: 包含拟合值、预测值、置信区间、模型摘要
    """
    from statsmodels.tsa.arima.model import ARIMA

    series = np.asarray(series, dtype=float)

    if order is None and auto_order:
        order, model, aic = auto_arima(series, max_p, max_d, max_q, verbose=False)
        print(f"[自动定阶] ARIMA{order}, AIC={aic:.1f}")
    elif order is None:
        order = (1, 0, 0)
        model = ARIMA(series, order=order).fit()
    else:
        model = ARIMA(series, order=order).fit()

    # 预测
    forecast_result = model.get_forecast(steps=forecast_num)
    forecast = forecast_result.predicted_mean
    ci = forecast_result.conf_int(alpha=0.05)

    # 拟合值
    fitted_values = model.fittedvalues
    residuals = series - fitted_values

    return {
        'order': order,
        'forecast': forecast,
        'ci_lower': ci[:, 0],
        'ci_upper': ci[:, 1],
        'fitted': fitted_values,
        'residuals': residuals,
        'aic': model.aic,
        'bic': model.bic,
        'model': model,
    }


def check_residual_white_noise(residuals, lags=None):
    """
    残差白噪声检验（Ljung-Box 检验）
    H0: 残差为白噪声（希望 p > 0.05，不拒绝原假设）
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    lag_default = min(10, len(residuals) // 5)
    lags = lags or lag_default
    lb_result = acorr_ljungbox(residuals.dropna(), lags=[lags], return_df=True)
    return lb_result


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    # 生成示例时间序列（趋势 + 季节 + 噪声）
    np.random.seed(42)
    t = np.arange(60)
    trend = 0.3 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 1.5, 60)
    series = 50 + trend + seasonal + noise

    print("=" * 60)
    print("ARIMA 时间序列预测")
    print("=" * 60)
    print(f"序列长度: {len(series)}, 前10个值: {series[:10].round(2)}")

    # 自动定阶预测
    result = arima_forecast(series, forecast_num=6, max_p=3, max_d=2, max_q=3)

    print(f"\n最优模型: ARIMA{result['order']}")
    print(f"AIC: {result['aic']:.1f}, BIC: {result['bic']:.1f}")

    print(f"\n未来6期预测值: {result['forecast'].round(2)}")
    print(f"95% 置信区间下界: {result['ci_lower'].round(2)}")
    print(f"95% 置信区间上界: {result['ci_upper'].round(2)}")

    # 残差诊断
    try:
        resid = result['residuals'].dropna()
        lb = check_residual_white_noise(resid)
        p_value = lb['lb_pvalue'].values[0]
        is_white = p_value > 0.05
        print(f"\nLjung-Box 白噪声检验: p={p_value:.3f}, "
              f"{'残差为白噪声 ✓' if is_white else '残差非白噪声 ✗（模型可能需改进）'}")
    except Exception as e:
        print(f"\n白噪声检验跳过: {e}")
