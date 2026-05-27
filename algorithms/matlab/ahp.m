function [weights, lambda_max, CI, CR, is_consistent] = ahp(judgment_matrix, method)
% AHP 层次分析法 — MATLAB 实现
%
% Input:
%   judgment_matrix - 判断矩阵 (n×n)，需满足正互反性
%   method - 'geometric'(几何平均法,默认) / 'eigen' (特征值法) / 'arithmetic'(算术平均法)
%
% Output:
%   weights      - 权重向量 (n×1)
%   lambda_max   - 最大特征值
%   CI           - 一致性指标
%   CR           - 一致性比率
%   is_consistent- 是否通过一致性检验 (CR < 0.10)
%
% 用法:
%   A = [1 2 4; 1/2 1 3; 1/4 1/3 1];
%   [w, lam, ci, cr, ok] = ahp(A);

if nargin < 2
    method = 'geometric';
end

A = judgment_matrix;
n = size(A, 1);

% 随机一致性指标 RI
RI_table = [0 0 0.58 0.90 1.12 1.24 1.32 1.41 1.45 1.49 1.51 1.54 1.56 1.58 1.59];
if n <= length(RI_table)
    RI = RI_table(n);
else
    RI = 1.59;
end

% 计算权重
switch lower(method)
    case 'geometric'
        % 几何平均法
        geo_mean = prod(A, 2) .^ (1/n);
        weights = geo_mean / sum(geo_mean);

    case 'arithmetic'
        % 算术平均法
        col_sum = sum(A, 1);
        normalized = A ./ col_sum;
        weights = mean(normalized, 2);

    case 'eigen'
        % 特征值法
        [V, D] = eig(A);
        eigenvalues = diag(D);
        [~, max_idx] = max(real(eigenvalues));
        weights = real(V(:, max_idx));
        weights = weights / sum(weights);

    otherwise
        error('Unknown method: %s', method);
end

% 最大特征值
Aw = A * weights;
lambda_max = mean(Aw ./ weights);

% 一致性检验
if n > 1
    CI = (lambda_max - n) / (n - 1);
else
    CI = 0;
end

if RI > 0 && n > 2
    CR = CI / RI;
else
    CR = 0;
end

is_consistent = CR < 0.10;

end

% ==========================================
% 示例运行 (直接执行本文件)
% ==========================================

% 3个指标两两比较
A = [1   2   4;
     1/2 1   3;
     1/4 1/3 1];

[w, lam, ci, cr, ok] = ahp(A);

fprintf('========== 层次分析法 (AHP) ==========\n');
fprintf('判断矩阵:\n'); disp(A);
fprintf('权重: '); fprintf('%.4f ', w); fprintf('\n');
fprintf('λ_max: %.4f\n', lam);
fprintf('CI: %.4f\n', ci);
fprintf('CR: %.4f\n', cr);
fprintf('一致性检验: %s\n', cond2str(ok));

% 三种方法对比
fprintf('\n--- 三种方法对比 ---\n');
methods = {'geometric', 'arithmetic', 'eigen'};
for i = 1:length(methods)
    [w_m, ~, ~, ~, ~] = ahp(A, methods{i});
    fprintf('  %-12s: ', methods{i});
    fprintf('%.4f ', w_m);
    fprintf('\n');
end

function s = cond2str(ok)
    if ok
        s = '通过 (CR < 0.10)';
    else
        s = '不通过 (CR >= 0.10)';
    end
end
