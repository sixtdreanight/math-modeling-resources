function [y_forecast, result] = arima_forecast(y, p, d, q, steps)
% ARIMA 时间序列预测 — MATLAB 实现
%
% Input:
%   y     - 时间序列 (向量)
%   p     - AR 阶数 (默认 1)
%   d     - 差分阶数 (默认 0)
%   q     - MA 阶数 (默认 1)
%   steps - 预测步数 (默认 5)
%
% Output:
%   y_forecast - 预测值
%   result     - 结构体: order, aic, bic, residuals, fitted
%
% 用法:
%   t = (1:60)';
%   y = 50 + 0.3*t + 5*sin(2*pi*t/12);
%   [yf, r] = arima_forecast(y, 2, 1, 2, 6);

if nargin < 2 || isempty(p), p = 1; end
if nargin < 3 || isempty(d), d = 0; end
if nargin < 4 || isempty(q), q = 1; end
if nargin < 5 || isempty(steps), steps = 5; end

y = y(:);  % 列向量

model = arima('ARLags', 1:p, 'D', d, 'MALags', 1:q);
try
    fitted_model = estimate(model, y, 'Display', 'off');
    [y_forecast, ~, forecast_mse] = forecast(fitted_model, steps, 'Y0', y);

    y_fitted = infer(fitted_model, y);
    residuals = y - y_fitted;

    [aic_val, bic_val] = aicbic(fitted_model.logL, ...
        numel(fitted_model.AR{:}) + numel(fitted_model.MA{:}) + 1, length(y));

    result.order = [p, d, q];
    result.aic = aic_val;
    result.bic = bic_val;
    result.residuals = residuals;
    result.fitted = y_fitted;
    result.model = fitted_model;
    result.forecast_mse = forecast_mse;
catch
    y_forecast = NaN(steps, 1);
    result = struct('error', '模型估计失败，尝试调整 p, d, q');
end

end

% ==========================================
% 示例运行
% ==========================================

t = (1:60)';
trend = 0.3 * t;
seasonal = 5 * sin(2*pi*t/12);
rng(42);
noise = 1.5 * randn(60, 1);
y = 50 + trend + seasonal + noise;

[yf, r] = arima_forecast(y, 2, 1, 1, 6);

fprintf('========== ARIMA 预测 ==========\n');
fprintf('模型阶数: ARIMA(%d,%d,%d)\n', r.order(1), r.order(2), r.order(3));
fprintf('AIC: %.2f  BIC: %.2f\n', r.aic, r.bic);
fprintf('\n未来6期预测值:\n');
for i = 1:length(yf)
    fprintf('  期%d: %.4f\n', i, yf(i));
end
fprintf('\n预测标准差: %.4f\n', sqrt(r.forecast_mse));
