# -*- coding: utf-8 -*-
"""
K-means 聚类 + 层次聚类 + DBSCAN — Python 实现

用法：
    python clustering.py
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


def kmeans_cluster(X, n_clusters=3, n_init=10, random_state=42):
    """
    K-means 聚类

    Returns: labels, model, inertia, silhouette
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0

    return {
        'labels': labels,
        'centers': scaler.inverse_transform(model.cluster_centers_),
        'inertia': model.inertia_,
        'silhouette': silhouette,
        'model': model,
        'scaler': scaler,
    }


def optimal_k_elbow(X, k_range=range(1, 11)):
    """
    肘部法则找最优 K 值
    同时计算轮廓系数
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    silhouettes = []
    calinskis = []

    for k in k_range:
        if k == 1:
            km = KMeans(n_clusters=1, n_init=10, random_state=42)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(0)
            calinskis.append(0)
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X_scaled)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_scaled, labels))
            calinskis.append(calinski_harabasz_score(X_scaled, labels))

    # 肘部检测：找曲率最大的点
    inertias = np.array(inertias)
    k_vals = np.array(list(k_range))
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    elbow_k = k_vals[1 + np.argmax(diffs2)]

    return {
        'k_range': k_range,
        'inertias': inertias,
        'silhouettes': silhouettes,
        'calinskis': calinskis,
        'suggested_k': elbow_k,
    }


def hierarchical_cluster(X, n_clusters=3, linkage='ward'):
    """层次聚类"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0

    return {'labels': labels, 'silhouette': silhouette}


def dbscan_cluster(X, eps=0.5, min_samples=5):
    """DBSCAN 密度聚类"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    silhouette = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0

    return {
        'labels': labels,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'silhouette': silhouette,
    }


if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 60)
    print("聚类分析示例")
    print("=" * 60)

    # 生成 3 簇数据
    from sklearn.datasets import make_blobs
    X, y_true = make_blobs(n_samples=300, centers=3, n_features=2,
                           cluster_std=[0.6, 1.0, 0.8], random_state=42)

    # --- K-means ---
    result = kmeans_cluster(X, n_clusters=3)
    print(f"\n[K-means] 聚类数=3")
    print(f"  inertia: {result['inertia']:.2f}")
    print(f"  silhouette: {result['silhouette']:.4f}")
    print(f"  簇中心:\n{result['centers']}")

    # --- 肘部法则找最优K ---
    opt = optimal_k_elbow(X)
    print(f"\n[肘部法则] 建议 K = {opt['suggested_k']}")
    print(f"  K=2 silhouette: {opt['silhouettes'][1]:.4f}")
    print(f"  K=3 silhouette: {opt['silhouettes'][2]:.4f}")
    print(f"  K=4 silhouette: {opt['silhouettes'][3]:.4f}")

    # --- 层次聚类 ---
    hier = hierarchical_cluster(X, n_clusters=3)
    print(f"\n[层次聚类 (Ward)] silhouette: {hier['silhouette']:.4f}")

    # --- DBSCAN ---
    db = dbscan_cluster(X, eps=0.6, min_samples=5)
    print(f"\n[DBSCAN] 聚类数={db['n_clusters']}, 噪声点={db['n_noise']}, silhouette={db['silhouette']:.4f}")
