function normalized = normalize_data(data, method)
% 数据标准化 — MATLAB 实现
%
% Input:
%   data   - 原始数据矩阵 (m×n)
%   method - 'minmax' / 'zscore' / 'robust' / 'l2'
%
% Output:
%   normalized - 标准化后的数据矩阵
%
% 用法:
%   X = [1 10 100; 2 20 200; 3 30 300];
%   X_norm = normalize_data(X, 'minmax');

if nargin < 2
    method = 'zscore';
end

X = data;
[m, n] = size(X);

switch lower(method)
    case 'minmax'
        x_min = min(X, [], 1);
        x_max = max(X, [], 1);
        denom = x_max - x_min;
        denom(denom == 0) = 1e-10;
        normalized = (X - x_min) ./ denom;

    case 'zscore'
        mu = mean(X, 1);
        sigma = std(X, 0, 1);  % 0 = N-1 归一化
        sigma(sigma == 0) = 1e-10;
        normalized = (X - mu) ./ sigma;

    case 'robust'
        med = median(X, 1);
        q1 = prctile(X, 25, 1);
        q3 = prctile(X, 75, 1);
        iqr = q3 - q1;
        iqr(iqr == 0) = 1e-10;
        normalized = (X - med) ./ iqr;

    case 'l2'
        norms = sqrt(sum(X .^ 2, 1));
        norms(norms == 0) = 1e-10;
        normalized = X ./ norms;

    otherwise
        error('Unknown method: %s', method);
end

end

% ==========================================
% 示例运行
% ==========================================

X = [1 10 100; 2 20 200; 3 30 300];

fprintf('========== 数据标准化 ==========\n');
fprintf('原始数据:\n'); disp(X);

fprintf('\nMin-Max 归一化 [0,1]:\n'); disp(normalize_data(X, 'minmax'));
fprintf('Z-score 标准化:\n'); disp(normalize_data(X, 'zscore'));
fprintf('Robust 缩放:\n'); disp(normalize_data(X, 'robust'));
fprintf('向量归一化 (L2):\n'); disp(normalize_data(X, 'l2'));
