import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Any, Dict, Tuple, List, Union

import category_encoders as ce


# --- 缺失值处理 ---
def impute_mean(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, None]:
    """使用均值填充指定列的缺失值。"""
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行均值填充。")
        return df, None
    if df[column].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(df[column]):
        imputer = SimpleImputer(strategy='mean')
        # 优化: 使用 .ravel() 将输出转为一维数组，保证赋值的稳定性
        df[column] = imputer.fit_transform(df[[column]]).ravel()
        print(f"列 '{column}' 的缺失值已通过均值填充。")
    return df, None


def impute_median(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, None]:
    """使用中位数填充指定列的缺失值。"""
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行中位数填充。")
        return df, None
    if df[column].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(df[column]):
        imputer = SimpleImputer(strategy='median')
        # 优化: 使用 .ravel() 将输出转为一维数组
        df[column] = imputer.fit_transform(df[[column]]).ravel()
        print(f"列 '{column}' 的缺失值已通过中位数填充。")
    return df, None


def impute_most_frequent(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, None]:
    """使用众数填充指定列的缺失值。"""
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行众数填充。")
        return df, None
    if df[column].isnull().sum() > 0:
        imputer = SimpleImputer(strategy='most_frequent')
        # 优化: 先将列转为object类型，确保imputer能正确处理混合类型。
        # 接着使用 .ravel() 将输出转为一维数组。
        df[column] = imputer.fit_transform(df[[column]].astype(object)).ravel()
        print(f"列 '{column}' 的缺失值已通过众数填充。")
    return df, None


def delete_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, None]:
    """删除指定列。"""
    if column in df.columns:
        df = df.drop(columns=[column])
        print(f"列 '{column}' 已被删除。")
    else:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法删除。")
    return df, None


def delete_rows_with_missing_in_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, None]:
    """删除在指定列中包含缺失值的行。"""
    if column in df.columns:
        original_len = len(df)
        df = df.dropna(subset=[column])
        print(f"在列 '{column}' 中包含缺失值的 {original_len - len(df)} 行已被删除。")
    else:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法基于此列删除行。")
    return df, None


# --- 数据缩放 ---
def standard_scale_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, StandardScaler]:
    """对指定数值列进行标准化缩放。"""
    scaler = StandardScaler()
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行标准化。")
        return df, scaler
    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = scaler.fit_transform(df[[column]]).ravel()
        print(f"列 '{column}' 已进行标准化缩放。")
    else:
        print(f"警告: 列 '{column}' 不是数值类型，无法执行标准化。")
    return df, scaler


def min_max_scale_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """对指定数值列进行最小-最大缩放。"""
    scaler = MinMaxScaler()
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行MinMax缩放。")
        return df, scaler
    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = scaler.fit_transform(df[[column]]).ravel()
        print(f"列 '{column}' 已进行MinMax缩放。")
    else:
        print(f"警告: 列 '{column}' 不是数值类型，无法执行MinMax缩放。")
    return df, scaler


# --- 类别特征编码 ---
def one_hot_encode_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """对指定类别列进行独热编码。"""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行独热编码。")
        return df, encoder

    try:
        # 确保处理的是二维数组
        encoded_data = encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column]), index=df.index)
        df = df.drop(columns=[column])
        df = pd.concat([df, encoded_df], axis=1)
        print(f"列 '{column}' 已进行独热编码，生成了 {len(encoded_df.columns)} 个新特征。")
    except Exception as e:
        print(f"对列 '{column}' 进行独热编码失败: {e}")
    return df, encoder


def label_encode_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    """对指定类别列进行标签编码。"""
    encoder = LabelEncoder()
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行标签编码。")
        return df, encoder

    # LabelEncoder 不能直接处理NaN，将其转换为字符串'NaN'作为一个独立的类别
    if df[column].isnull().any():
        df[column] = df[column].astype(str)

    try:
        df[column] = encoder.fit_transform(df[column])
        print(f"列 '{column}' 已进行标签编码。")
    except Exception as e:
        print(f"对列 '{column}' 进行标签编码失败: {e}")

    return df, encoder


def target_encode_column(df: pd.DataFrame, column: str, target_metric_name: str) -> Tuple[
    pd.DataFrame, ce.TargetEncoder]:
    """对指定类别列进行目标编码。"""
    # handle_unknown='value' 和 smoothing 参数可以增加编码的稳定性
    encoder = ce.TargetEncoder(cols=[column], handle_unknown='value', smoothing=5)
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行目标编码。")
        return df, encoder
    if target_metric_name not in df.columns:
        print(f"警告: 目标列 '{target_metric_name}' 不在DataFrame中，无法执行目标编码。")
        return df, encoder

    try:
        # TargetEncoder可以自动处理NaN
        df[column] = encoder.fit_transform(X=df[column], y=df[target_metric_name])
        print(f"列 '{column}' 已进行目标编码。")
    except Exception as e:
        print(f"对列 '{column}' 进行目标编码失败: {e}")

    return df, encoder


# --- 离群值处理 (初步) ---
def cap_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> Tuple[pd.DataFrame, None]:
    """使用IQR方法对指定列的离群值进行封顶处理。"""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        print(f"警告: 列 '{column}' 无效或非数值，无法处理离群值。")
        return df, None

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    original_sum_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    if original_sum_outliers > 0:
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        print(f"列 '{column}' 的离群值已使用IQR方法进行封顶处理。检测到并处理了 {original_sum_outliers} 个离群点。")
    else:
        print(f"列 '{column}' 未检测到明显离群值 (基于IQR factor={factor})。")
    return df, None


# 方法映射表，方便LLM选择和程序调用
PREPROCESSING_METHODS_MAP = {
    # 缺失值处理
    "impute_mean": impute_mean,
    "impute_median": impute_median,
    "impute_most_frequent": impute_most_frequent,
    "delete_column": delete_column,
    "delete_rows_with_missing_in_column": delete_rows_with_missing_in_column,

    # 数据缩放
    "standard_scale_column": standard_scale_column,
    "min_max_scale_column": min_max_scale_column,

    # 类别特征编码
    "one_hot_encode_column": one_hot_encode_column,
    "label_encode_column": label_encode_column,
    "target_encode_column": target_encode_column,

    # 离群值处理
    "cap_outliers_iqr": cap_outliers_iqr,
}
