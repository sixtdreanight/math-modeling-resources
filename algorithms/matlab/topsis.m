function [scores, ranking, details] = topsis(data, weights, varargin)
% TOPSIS 法（优劣解距离法）— MATLAB 实现
%
% Input:
%   data    - 决策矩阵 (m方案 × n指标)
%   weights - 权重向量 (1×n), 默认等权
%   可选参数(键值对):
%     'cost_cols'       - 极小型指标列索引
%     'target_cols'     - 中间型指标列索引
%     'target_values'   - 中间型最佳值
%     'interval_cols'   - 区间型指标列索引
%     'interval_bounds' - 区间型最佳区间
%
% Output:
%   scores  - 相对贴近度 C (m×1)
%   ranking - 排名 (1最优)
%   details - 中间结果结构体
%
% 用法:
%   data = [8 7 2 4.5 4; 6 8 3 5.2 3.5; 9 6 1.5 4.8 5.5; 7 9 2.5 3.5 2];
%   [s, r, d] = topsis(data, [], 'cost_cols', 3, 'target_cols', 4, 'target_values', 5, 'interval_cols', 5, 'interval_bounds', [3 5]);

% 解析参数
p = inputParser;
addRequired(p, 'data');
addRequired(p, 'weights');
addParameter(p, 'cost_cols', []);
addParameter(p, 'target_cols', []);
addParameter(p, 'target_values', []);
addParameter(p, 'interval_cols', []);
addParameter(p, 'interval_bounds', []);
parse(p, data, weights, varargin{:});

cost_cols = p.Results.cost_cols;
target_cols = p.Results.target_cols;
target_values = p.Results.target_values;
interval_cols = p.Results.interval_cols;
interval_bounds = p.Results.interval_bounds;

X = data;
[m, n] = size(X);

% Step 1: 正向化
X_pos = X;

% 极小型 → 极大型
for idx = 1:length(cost_cols)
    j = cost_cols(idx);
    X_pos(:, j) = max(X_pos(:, j)) - X_pos(:, j);
end

% 中间型 → 极大型
for idx = 1:length(target_cols)
    j = target_cols(idx);
    best = target_values(idx);
    M = max(abs(X_pos(:, j) - best));
    if M > 0
        X_pos(:, j) = 1 - abs(X_pos(:, j) - best) / M;
    end
end

% 区间型 → 极大型
for idx = 1:length(interval_cols)
    j = interval_cols(idx);
    low = interval_bounds(idx, 1);
    high = interval_bounds(idx, 2);
    M = max([low - min(X_pos(:, j)), max(X_pos(:, j)) - high]);
    if M > 0
        for i = 1:m
            if X_pos(i, j) < low
                X_pos(i, j) = 1 - (low - X_pos(i, j)) / M;
            elseif X_pos(i, j) > high
                X_pos(i, j) = 1 - (X_pos(i, j) - high) / M;
            else
                X_pos(i, j) = 1;
            end
        end
    end
end

% Step 2: 向量归一化
Z = X_pos ./ sqrt(sum(X_pos .^ 2, 1));

% Step 3: 加权
if isempty(weights) || all(weights == 0)
    weights = ones(1, n) / n;
end
Z_weighted = Z .* weights;

% Step 4: 正负理想解
Z_plus = max(Z_weighted, [], 1);
Z_minus = min(Z_weighted, [], 1);

% Step 5: 欧氏距离
D_plus = sqrt(sum((Z_weighted - Z_plus) .^ 2, 2));
D_minus = sqrt(sum((Z_weighted - Z_minus) .^ 2, 2));

% Step 6: 贴近度
scores = D_minus ./ (D_plus + D_minus);

% 排序
[~, idx] = sort(scores, 'descend');
ranking = zeros(m, 1);
for i = 1:m
    ranking(idx(i)) = i;
end

details.X_positive = X_pos;
details.Z_normalized = Z;
details.Z_weighted = Z_weighted;
details.Z_plus = Z_plus;
details.Z_minus = Z_minus;
details.D_plus = D_plus;
details.D_minus = D_minus;

end

% ==========================================
% 示例运行 (直接执行本文件)
% ==========================================

% 4个方案，5个指标
% 指标1-2: 极大型, 指标3: 极小型, 指标4: 中间型(最佳=5), 指标5: 区间型([3,5])
data = [8.0 7.0 2.0 4.5 4.0;
        6.0 8.0 3.0 5.2 3.5;
        9.0 6.0 1.5 4.8 5.5;
        7.0 9.0 2.5 3.5 2.0];

[scores, ranking, details] = topsis(data, [], ...
    'cost_cols', 3, ...
    'target_cols', 4, 'target_values', 5, ...
    'interval_cols', 5, 'interval_bounds', [3 5]);

fprintf('========== TOPSIS 法 ==========\n');
fprintf('正理想解 Z+: '); fprintf('%.4f ', details.Z_plus); fprintf('\n');
fprintf('负理想解 Z-: '); fprintf('%.4f ', details.Z_minus); fprintf('\n');
fprintf('\n方案  贴近度  排名\n');
fprintf('----  ------  ----\n');
for i = 1:length(scores)
    fprintf('  %d    %.4f    %d\n', i, scores(i), ranking(i));
end
