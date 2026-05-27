function result = grey_model(series, forecast_num)
% GM(1,1) 灰色预测 — MATLAB 实现
%
% Input:
%   series       - 原始序列 (至少4个数据点)
%   forecast_num - 预测期数
%
% Output:
%   result.forecast      - 预测值
%   result.fitted        - 拟合值
%   result.residuals     - 残差
%   result.relative_err  - 相对误差 (%)
%   result.C             - 后验差比值
%   result.P             - 小误差概率
%   result.grade         - 精度等级 (1-4)
%
% 用法:
%   data = [89677 90859 92148 93769 95050 96450 97812 99256 100729 102034];
%   r = grey_model(data, 3);

if nargin < 2
    forecast_num = 1;
end

X0 = series(:)';  % 转为行向量
n = length(X0);

% 级比检验
lambda_k = X0(1:end-1) ./ X0(2:end);
lower = exp(-2/(n+1));
upper = exp(2/(n+1));
lambdas_ok = all(lambda_k > lower & lambda_k < upper);

% 1-AGO (一次累加)
X1 = cumsum(X0);

% 紧邻均值生成
Z1 = 0.5 * (X1(1:end-1) + X1(2:end));

% 最小二乘估计
B = [-Z1' ones(n-1,1)];
Y = X0(2:end)';
ab = (B' * B) \ (B' * Y);  % 或 inv(B' * B) * B' * Y
a = ab(1);
b = ab(2);

% 发育系数为零的防护
if abs(a) < 1e-10
    a = 1e-10;
end

% 时间响应式
k_values = 0:(n + forecast_num - 1);
X1_pred = (X0(1) - b/a) * exp(-a * k_values) + b/a;

% 还原
X0_pred = zeros(1, n + forecast_num);
X0_pred(1) = X0(1);
X0_pred(2:end) = X1_pred(2:end) - X1_pred(1:end-1);

% 拟合值和预测值
fitted = X0_pred(1:n);
forecast = X0_pred(n+1:end);

% 残差分析
residuals = X0 - fitted;
relative_err = abs(residuals ./ (X0 + 1e-10)) * 100;

% 后验差检验
S1 = std(X0);
S2 = std(residuals);
C = S2 / S1;

% 小误差概率
mean_residual = mean(residuals);
P_count = sum(abs(residuals - mean_residual) < 0.6745 * S1);
P = P_count / n;

% 精度等级
if C < 0.35 && P > 0.95
    grade = 1;
    grade_name = '好 (Good)';
elseif C < 0.50 && P > 0.80
    grade = 2;
    grade_name = '合格 (Qualified)';
elseif C < 0.65 && P > 0.70
    grade = 3;
    grade_name = '勉强 (Barely)';
else
    grade = 4;
    grade_name = '不合格 (Unqualified)';
end

% 输出结果
result.forecast = forecast;
result.fitted = fitted;
result.residuals = residuals;
result.relative_err = relative_err;
result.C = C;
result.P = P;
result.grade = grade;
result.grade_name = grade_name;
result.lambdas_ok = lambdas_ok;
result.params.a = a;
result.params.b = b;

end

% ==========================================
% 示例运行 (直接执行本文件)
% ==========================================

data = [89677 90859 92148 93769 95050 96450 97812 99256 100729 102034];

r = grey_model(data, 3);

fprintf('========== 灰色预测 GM(1,1) ==========\n');
fprintf('原始序列: '); fprintf('%d ', data); fprintf('\n');
fprintf('\n级比检验: %s\n', cond2str(r.lambdas_ok));
fprintf('发展系数 a: %.6f\n', r.params.a);
fprintf('灰色作用量 b: %.2f\n', r.params.b);
fprintf('\n后验差比值 C: %.4f\n', r.C);
fprintf('小误差概率 P: %.4f\n', r.P);
fprintf('精度等级: %d级 — %s\n', r.grade, r.grade_name);
fprintf('\n拟合值: '); fprintf('%d ', round(r.fitted)); fprintf('\n');
fprintf('残差:   '); fprintf('%d ', round(r.residuals)); fprintf('\n');
fprintf('相对误差(%%): '); fprintf('%.2f ', r.relative_err); fprintf('\n');
fprintf('平均相对误差: %.2f%%\n', mean(r.relative_err));
fprintf('\n预测值(未来3期): '); fprintf('%d ', round(r.forecast)); fprintf('\n');

function s = cond2str(ok)
    if ok
        s = '通过';
    else
        s = '未通过（建议做平移变换或使用其他模型）';
    end
end
