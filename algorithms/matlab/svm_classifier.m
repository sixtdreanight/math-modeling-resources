function result = svm_classifier(X, y, kernel, test_ratio)
% SVM 分类器 — MATLAB 实现
%
% Input:
%   X         - 特征矩阵 (n×p)
%   y         - 标签向量 (n×1)
%   kernel    - 'linear' / 'rbf' / 'polynomial' (default: 'rbf')
%   test_ratio- 测试集比例 (default: 0.3)
%
% Output:
%   result.accuracy     - 分类准确率
%   result.model        - 训练好的 SVM 模型
%   result.confusion    - 混淆矩阵
%   result.y_pred       - 预测标签
%
% 用法:
%   X = randn(100, 3); y = [ones(50,1); 2*ones(50,1)];
%   r = svm_classifier(X, y, 'rbf', 0.3);

if nargin < 3 || isempty(kernel), kernel = 'rbf'; end
if nargin < 4 || isempty(test_ratio), test_ratio = 0.3; end

% 划分训练/测试集
n = size(X, 1);
cv = cvpartition(y, 'HoldOut', test_ratio);
X_train_orig = X(training(cv), :);
y_train = y(training(cv), :);
X_test_orig = X(test(cv), :);
y_test = y(test(cv), :);

% 标准化（仅在训练集上 fit，避免数据泄露）
[train_mu, train_sigma] = deal(mean(X_train_orig, 1), std(X_train_orig, 0, 1));
X_train = (X_train_orig - train_mu) ./ (train_sigma + 1e-10);
X_test = (X_test_orig - train_mu) ./ (train_sigma + 1e-10);

% 训练 SVM (使用 fitcecoc 处理多分类)
if numel(unique(y)) > 2
    template = templateSVM('KernelFunction', kernel, 'Standardize', true, ...
                           'KernelScale', 'auto');
    model = fitcecoc(X_train, y_train, 'Learners', template);
else
    model = fitcsvm(X_train, y_train, 'KernelFunction', kernel, ...
                    'Standardize', true, 'KernelScale', 'auto');
end

% 预测
y_pred = predict(model, X_test);
accuracy = sum(y_pred == y_test) / length(y_test);

% 混淆矩阵
classes = unique(y);
confusion = confusionmat(y_test, y_pred);

% 输出
result.model = model;
result.accuracy = accuracy;
result.confusion = confusion;
result.y_pred = y_pred;
result.y_test = y_test;

end

% ==========================================
% 示例运行
% ==========================================

rng(42);

% 生成3类数据
n = 300;
X = [randn(100,3) + 0;
     randn(100,3) + 3;
     randn(100,3) + [0, 3, 0]];
y = [ones(100,1); 2*ones(100,1); 3*ones(100,1)];

r = svm_classifier(X, y, 'rbf', 0.3);

fprintf('========== SVM 分类器 ==========\n');
fprintf('准确率: %.2f%%\n', r.accuracy * 100);
fprintf('\n混淆矩阵:\n');
disp(r.confusion);
fprintf('\n分类报告:\n');
for c = 1:size(r.confusion, 1)
    tp = r.confusion(c, c);
    precision = tp / sum(r.confusion(:, c));
    recall = tp / sum(r.confusion(c, :));
    fprintf('  类别 %d: Precision=%.3f, Recall=%.3f\n', c, precision, recall);
end
