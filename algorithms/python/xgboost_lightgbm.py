# -*- coding: utf-8 -*-
"""
XGBoost / LightGBM — Python 实现
参数：GridSearchCV / 特征重要性 / 学习曲线

用法：
    pip install xgboost lightgbm
    python xgboost_lightgbm.py
"""

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.datasets import make_classification, make_regression


def xgboost_classifier(X, y, param_search=False, test_size=0.3):
    """XGBoost 分类"""
    import xgboost as xgb

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    if param_search:
        params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
        }
        model = GridSearchCV(xgb.XGBClassifier(eval_metric='logloss', random_state=42),
                             params, cv=3, scoring='accuracy', n_jobs=-1)
        model.fit(X_train, y_train)
    else:
        model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                  subsample=0.8, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model.best_estimator_ if param_search else model,
                                X, y, cv=5, scoring='accuracy')

    return {
        'model': model,
        'accuracy': acc,
        'cv_mean': cv_scores.mean(),
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_importance': model.best_estimator_.feature_importances_ if param_search
                              else model.feature_importances_,
    }


def lightgbm_classifier(X, y, test_size=0.3):
    """LightGBM 分类"""
    import lightgbm as lgb

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                               subsample=0.8, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    return {
        'model': model,
        'accuracy': acc,
        'cv_mean': cv_scores.mean(),
        'feature_importance': model.feature_importances_,
    }


def xgboost_regressor(X, y, test_size=0.3):
    """XGBoost 回归"""
    import xgboost as xgb

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                             subsample=0.8, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        'model': model,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'feature_importance': model.feature_importances_,
    }


if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("XGBoost / LightGBM 示例")
    print("=" * 60)

    # --- 分类 ---
    X_cls, y_cls = make_classification(n_samples=500, n_features=10, n_informative=6,
                                       n_redundant=2, n_classes=3, n_clusters_per_class=1,
                                       random_state=42)

    try:
        result_xgb = xgboost_classifier(X_cls, y_cls, param_search=False)
        print(f"\n[XGBoost 分类] 准确率: {result_xgb['accuracy']:.4f}, CV: {result_xgb['cv_mean']:.4f}")
        print(f"  特征重要性 (top5): {np.argsort(result_xgb['feature_importance'])[-5:][::-1]}")
    except ImportError:
        print("\n[XGBoost] 未安装 (pip install xgboost)")

    try:
        result_lgb = lightgbm_classifier(X_cls, y_cls)
        print(f"\n[LightGBM 分类] 准确率: {result_lgb['accuracy']:.4f}, CV: {result_lgb['cv_mean']:.4f}")
    except ImportError:
        print("[LightGBM] 未安装 (pip install lightgbm)")

    # --- 回归 ---
    X_reg, y_reg = make_regression(n_samples=300, n_features=8, noise=10, random_state=42)

    try:
        result_reg = xgboost_regressor(X_reg, y_reg)
        print(f"\n[XGBoost 回归] RMSE: {result_reg['rmse']:.2f}, R²: {result_reg['r2']:.4f}")
    except ImportError:
        pass
