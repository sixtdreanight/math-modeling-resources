# -*- coding: utf-8 -*-
"""
数学建模 — 数据处理通用模板
功能：读数据 → 缺失值处理 → 异常值检测 → 标准化/归一化 → 编码 → 输出
Python 3.7+, pandas, numpy, scipy, scikit-learn
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# ==========================================
# 1. 数据读取
# ==========================================

def load_data(filepath):
    """自动识别文件格式并读取"""
    ext = filepath.suffix.lower() if hasattr(filepath, 'suffix') else filepath.split('.')[-1]
    if ext == '.csv':
        return pd.read_csv(filepath, encoding='utf-8-sig')
    elif ext in ('.xlsx', '.xls'):
        return pd.read_excel(filepath)
    elif ext == '.txt':
        return pd.read_csv(filepath, sep='\t', encoding='utf-8-sig')
    elif ext == '.json':
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ==========================================
# 2. 数据概览
# ==========================================

def data_summary(df):
    """输出数据基本信息"""
    print("=" * 60)
    print("数据基本信息")
    print("=" * 60)
    print(f"形状 (行, 列): {df.shape}")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
    print(f"\n缺失值占比:\n{(df.isnull().sum() / len(df) * 100).round(2)}%")
    print(f"\n描述性统计:\n{df.describe().round(2)}")
    print("=" * 60)


# ==========================================
# 3. 缺失值处理
# ==========================================

def handle_missing(df, strategy='auto', threshold_col=0.5, threshold_row=0.3):
    """
    缺失值处理
    strategy: 'drop' 删除 / 'mean' 均值填充 / 'median' 中位数填充 / 'knn' KNN填充 / 'mice' 多重插补 / 'auto' 自动
    threshold_col: 缺失超过此比例的列直接删除
    threshold_row: 缺失超过此比例的行直接删除
    """
    n_rows, n_cols = df.shape
    df_clean = df.copy()

    # 删除缺失率过高的列
    missing_ratio = df_clean.isnull().sum() / n_rows
    cols_to_drop = missing_ratio[missing_ratio > threshold_col].index.tolist()
    if cols_to_drop:
        print(f"[删除列] 缺失率>{threshold_col:.0%}: {cols_to_drop}")
        df_clean = df_clean.drop(columns=cols_to_drop)

    # 删除缺失率过高的行
    row_missing = df_clean.isnull().sum(axis=1) / len(df_clean.columns)
    rows_to_drop = (row_missing > threshold_row).sum()
    if rows_to_drop > 0:
        print(f"[删除行] 缺失率>{threshold_row:.0%}: {rows_to_drop} 行")
        df_clean = df_clean[row_missing <= threshold_row]

    # 分离数值列和非数值列
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns

    if strategy == 'auto':
        # 自动策略：少量缺失用中位数，中等用KNN
        num_missing = df_clean[num_cols].isnull().sum().sum()
        if num_missing == 0:
            print("[缺失值处理] 无缺失值，跳过")
            return df_clean
        ratio = num_missing / (df_clean[num_cols].shape[0] * len(num_cols))
        if ratio < 0.05:
            strategy = 'median'
        elif ratio < 0.20:
            strategy = 'knn'
        else:
            strategy = 'median'
        print(f"[自动选择] 缺失比例={ratio:.2%}, 策略={strategy}")

    if len(num_cols) > 0:
        if strategy == 'mean':
            imp = SimpleImputer(strategy='mean')
        elif strategy == 'median':
            imp = SimpleImputer(strategy='median')
        elif strategy == 'knn':
            imp = KNNImputer(n_neighbors=5)
        elif strategy == 'mice':
            imp = IterativeImputer(max_iter=10, random_state=42)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        df_clean[num_cols] = imp.fit_transform(df_clean[num_cols])

    # 分类变量用众数填充
    if len(cat_cols) > 0:
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')

    print(f"[缺失值处理完成] 策略={strategy}, 剩余缺失={df_clean.isnull().sum().sum()}")
    return df_clean


# ==========================================
# 4. 异常值检测
# ==========================================

def detect_outliers(df, method='iqr', threshold=3.0):
    """
    异常值检测
    method: 'iqr' (IQR箱线图) / 'zscore' (Z分数) / 'grubbs' (Grubbs检验)
    threshold: zscore用3, iqr用1.5
    返回异常值字典 {列名: [异常行索引列表]}
    """
    outliers = {}
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        series = df[col].dropna()
        if method == 'zscore':
            z = np.abs(stats.zscore(series))
            outlier_idx = series.index[z > threshold].tolist()
        elif method == 'iqr':
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - threshold * IQR, Q3 + threshold * IQR
            outlier_idx = series.index[(series < lower) | (series > upper)].tolist()
        elif method == 'grubbs':
            # 简化版Grubbs：单侧检验
            outlier_idx = []
            data = series.values.copy()
            for _ in range(min(5, len(data) // 10)):  # 最多检测5个异常值
                mean, std = np.mean(data), np.std(data, ddof=1)
                z = np.abs((data - mean) / std)
                max_z_idx = np.argmax(z)
                # Grubbs临界值近似
                n = len(data)
                t_crit = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
                g_crit = (n - 1) / np.sqrt(n) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
                if z[max_z_idx] > g_crit:
                    original_idx = series.index[np.isin(series.values, data[max_z_idx])].tolist()
                    outlier_idx.extend(original_idx)
                    data = np.delete(data, max_z_idx)
                else:
                    break
        if outlier_idx:
            outliers[col] = outlier_idx

    if outliers:
        total = sum(len(v) for v in outliers.values())
        print(f"[异常值检测] 共检测到 {total} 个异常值 (方法={method})")
        for col, idx in outliers.items():
            print(f"  {col}: {len(idx)} 个")
    else:
        print("[异常值检测] 未检测到异常值")
    return outliers


def handle_outliers(df, outliers, strategy='cap'):
    """
    处理异常值
    strategy: 'cap' 截断 / 'remove' 删除 / 'mean' 替换为均值
    """
    df_clean = df.copy()
    if strategy == 'remove':
        all_outlier_idx = set()
        for indices in outliers.values():
            all_outlier_idx.update(indices)
        df_clean = df_clean.drop(index=all_outlier_idx)
        print(f"[异常值处理] 删除了 {len(all_outlier_idx)} 行")
    elif strategy == 'cap':
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower, upper)
        print("[异常值处理] 已截断到 IQR 范围")
    elif strategy == 'mean':
        for col, indices in outliers.items():
            mean_val = df_clean[col].mean()
            df_clean.loc[indices, col] = mean_val
        print("[异常值处理] 已替换为均值")
    return df_clean


# ==========================================
# 5. 标准化 / 归一化
# ==========================================

def scale_data(df, method='standard', exclude_cols=None):
    """
    数据缩放
    method: 'standard' (Z-score) / 'minmax' ([0,1]) / 'robust' (对异常值鲁棒)
    exclude_cols: 不参与缩放的列名列表
    """
    exclude_cols = exclude_cols or []
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}")

    df_scaled = df.copy()
    if len(num_cols) > 0:
        df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"[数据缩放] 方法={method}, 处理了 {len(num_cols)} 列")
    return df_scaled, scaler


# ==========================================
# 6. 分类变量编码
# ==========================================

def encode_categorical(df, cols=None, method='onehot'):
    """
    分类变量编码
    method: 'onehot' / 'label'
    """
    if cols is None:
        cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(cols) == 0:
        return df, {}

    df_encoded = df.copy()
    encoders = {}

    for col in cols:
        if method == 'onehot':
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
            encoders[col] = 'onehot'
        elif method == 'label':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    print(f"[编码完成] 方法={method}, 处理了 {len(cols)} 列")
    return df_encoded, encoders


# ==========================================
# 7. 完整处理流程
# ==========================================

def full_pipeline(filepath, scale='standard', missing_strategy='auto',
                  outlier_method='iqr', outlier_strategy='cap',
                  encode_method='onehot', exclude_cols=None):
    """
    一键执行完整数据处理流程
    """
    print("=" * 60)
    print("数学建模 — 数据处理流水线")
    print("=" * 60)

    # Step 1: 加载
    df = load_data(filepath)
    data_summary(df)

    # Step 2: 缺失值
    df = handle_missing(df, strategy=missing_strategy)

    # Step 3: 异常值
    outliers = detect_outliers(df, method=outlier_method)
    df = handle_outliers(df, outliers, strategy=outlier_strategy)

    # Step 4: 编码
    df, encoders = encode_categorical(df, method=encode_method)

    # Step 5: 缩放
    df, scaler = scale_data(df, method=scale, exclude_cols=exclude_cols)

    print("=" * 60)
    print(f"处理完成! 最终形状: {df.shape}")
    print("=" * 60)
    return df


# ==========================================
# 示例运行
# ==========================================

if __name__ == '__main__':
    # 创建示例数据并演示流程
    np.random.seed(42)

    df = pd.DataFrame({
        'x1': np.random.normal(10, 2, 100),
        'x2': np.random.normal(5, 1, 100),
        'x3': np.random.exponential(3, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
    })

    # 人为制造缺失值和异常值
    df.loc[np.random.choice(100, 10, replace=False), 'x1'] = np.nan
    df.loc[np.random.choice(100, 5, replace=False), 'x2'] = np.nan
    df.loc[0, 'x3'] = 100  # 极端异常值

    print("原始数据:")
    print(df.head(10))

    data_summary(df)

    df_clean = handle_missing(df, strategy='auto')
    outliers = detect_outliers(df_clean, method='iqr')
    df_clean = handle_outliers(df_clean, outliers, strategy='cap')
    df_clean, encoders = encode_categorical(df_clean, method='onehot')
    df_clean, scaler = scale_data(df_clean, method='standard')

    print("\n处理后的数据:")
    print(df_clean.head(10))
    print("\n各方法可独立使用，可根据实际需求组合调用。")
