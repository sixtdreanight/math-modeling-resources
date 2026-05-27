function result = kmeans_cluster(X, k, method)
% K-means 聚类 — MATLAB 实现
%
% Input:
%   X      - 数据矩阵 (n×p)
%   k      - 聚类数 (默认 3)
%   method - 'standard' (默认) / 'kmeans++'
%
% Output:
%   result.labels     - 聚类标签
%   result.centers    - 簇中心
%   result.inertia    - 簇内平方和 (SSE)
%   result.silhouette - 轮廓系数均值
%
% 用法:
%   X = [randn(100,2); randn(100,2)+3];
%   r = kmeans_cluster(X, 2);

if nargin < 2 || isempty(k), k = 3; end
if nargin < 3 || isempty(method), method = 'standard'; end

% 标准化
X_norm = normalize_data(X, 'zscore');

% K-means
rng(42);
if strcmpi(method, 'kmeans++')
    [labels, centers] = kmeans(X_norm, k, 'Replicates', 10, 'Start', 'plus');
else
    [labels, centers] = kmeans(X_norm, k, 'Replicates', 10);
end

% 簇内平方和 (inertia)
inertia = 0;
for j = 1:k
    cluster_points = X_norm(labels == j, :);
    inertia = inertia + sum(sum((cluster_points - centers(j,:)).^2, 2));
end

% 轮廓系数
silhouette_val = silhouette(X_norm, labels);
avg_silhouette = mean(silhouette_val);

% 将中心还原到原始尺度
X_mean = mean(X, 1);
X_std = std(X, 0, 1);
centers_orig = centers .* X_std + X_mean;

result.labels = labels;
result.centers = centers_orig;
result.centers_norm = centers;
result.inertia = inertia;
result.silhouette = avg_silhouette;

end

% ==========================================
% 示例运行
% ==========================================

rng(42);

% 生成3簇数据
X = [randn(100,2) * 0.6 + [0, 0];
     randn(100,2) * 0.8 + [3, 0];
     randn(100,2) * 0.7 + [1.5, 3]];

fprintf('========== K-means 聚类 ==========\n');

% 肘部法则
fprintf('肘部法则:\n');
fprintf('  K  SSE\n');
for k_test = 1:6
    r_k = kmeans_cluster(X, k_test);
    marker = '';
    if k_test == 3, marker = ' <-- 最优'; end
    fprintf('  %d   %.2f%s\n', k_test, r_k.inertia, marker);
end

% 最终聚类
r = kmeans_cluster(X, 3);
fprintf('\n聚类结果 (K=3):\n');
fprintf('  轮廓系数: %.4f\n', r.silhouette);
fprintf('  Inertia: %.2f\n', r.inertia);
fprintf('  各簇大小: ');
for j = 1:3
    fprintf('%d ', sum(r.labels == j));
end
fprintf('\n');
