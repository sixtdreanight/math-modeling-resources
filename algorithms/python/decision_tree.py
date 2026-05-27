# -*- coding: utf-8 -*-
"""
决策树 / 随机森林 — Python 实现
sklearn.ensemble.RandomForestClassifier/Regressor

用法：
    python decision_tree.py
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score


def random_forest_classifier(X, y, n_estimators=200, max_depth=None, param_search=False,
                             test_size=0.3):
    """随机森林分类"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if param_search:
        params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5],
        }
        rf = GridSearchCV(RandomForestClassifier(random_state=42), params,
                          cv=3, scoring='accuracy', n_jobs=-1)
        rf.fit(X_train, y_train)
    else:
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    cv_scores = cross_val_score(rf.best_estimator_ if param_search else rf,
                                X, y, cv=5, scoring='accuracy')

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': (rf.best_estimator_.feature_importances_
                               if param_search else rf.feature_importances_),
        'model': rf,
    }


def random_forest_regressor(X, y, n_estimators=200, test_size=0.3):
    """随机森林回归"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    return {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'feature_importance': rf.feature_importances_,
    }


if __name__ == '__main__':
    from sklearn.datasets import make_classification, make_regression
    np.random.seed(42)

    print("=" * 60)
    print("随机森林示例")
    print("=" * 60)

    # 分类
    X_cls, y_cls = make_classification(n_samples=500, n_features=8, n_informative=5,
                                       n_redundant=2, n_classes=3, n_clusters_per_class=1,
                                       random_state=42)

    result_cls = random_forest_classifier(X_cls, y_cls, n_estimators=150)
    print(f"\n[分类] 准确率: {result_cls['accuracy']:.4f}, CV: {result_cls['cv_mean']:.4f} (+/- {result_cls['cv_std']:.4f})")
    top5 = np.argsort(result_cls['feature_importance'])[-5:][::-1]
    print(f"  特征重要性 top5: {list(top5)}")

    # 回归
    X_reg, y_reg = make_regression(n_samples=300, n_features=6, noise=15, random_state=42)
    result_reg = random_forest_regressor(X_reg, y_reg)
    print(f"\n[回归] RMSE: {result_reg['rmse']:.4f}, R²: {result_reg['r2']:.4f}")
