# -*- coding: utf-8 -*-
"""
支持向量机 (SVM) — Python 实现
基于 sklearn.svm, 含 GridSearchCV 参数优化

用法：
    python svm_classifier.py
"""

import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


def svm_classifier(X, y, kernel='rbf', C=1.0, gamma='scale', cv=5,
                   param_search=False, test_size=0.3):
    """
    SVM 分类器

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    y : ndarray (n_samples,)
    kernel : 'linear' / 'rbf' / 'poly' / 'sigmoid'
    C : float, 正则化参数
    gamma : 'scale' / 'auto' / float
    cv : int, 交叉验证折数
    param_search : bool, 是否网格搜索最优参数
    test_size : float, 测试集比例

    Returns
    -------
    dict: model, accuracy, report, best_params
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # 标准化（仅在训练集上fit，避免数据泄露）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if param_search:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear'],
        }
        clf = GridSearchCV(SVC(), param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)
        best_params = clf.best_params_
    else:
        clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        clf.fit(X_train, y_train)
        best_params = {'kernel': kernel, 'C': C, 'gamma': gamma}

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        'model': clf,
        'scaler': scaler,
        'accuracy': acc,
        'y_test': y_test,
        'y_pred': y_pred,
        'best_params': best_params,
        'report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }


def svm_regressor(X, y, kernel='rbf', C=1.0, epsilon=0.1,
                  test_size=0.3):
    """SVM 回归 (SVR)"""
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, r2_score

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=42
    )

    reg = SVR(kernel=kernel, C=C, epsilon=epsilon)
    reg.fit(X_train, y_train)
    y_pred_scaled = reg.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    return {
        'model': reg,
        'mse': mean_squared_error(y_test_orig, y_pred),
        'r2': r2_score(y_test_orig, y_pred),
        'y_test': y_test_orig,
        'y_pred': y_pred,
    }


if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("SVM 支持向量机示例")
    print("=" * 60)

    # --- SVM 分类 ---
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=300, n_features=4, n_informative=3,
                               n_redundant=0, n_classes=3, n_clusters_per_class=1,
                               random_state=42)

    # 参数搜索
    result = svm_classifier(X, y, param_search=True, test_size=0.3)

    print(f"\n[分类] 最优参数: {result['best_params']}")
    print(f"准确率: {result['accuracy']:.4f}")
    print(f"\n分类报告:\n{result['report']}")
    print(f"混淆矩阵:\n{result['confusion_matrix']}")

    # 交叉验证
    X_scaled = result['scaler'].transform(X)
    cv_scores = cross_val_score(SVC(**result['best_params']), X_scaled, y, cv=5)
    print(f"\n5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
