import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Any, Dict, Tuple, List, Union


# --- 缺失值处理 ---
def impute_mean(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """使用均值填充指定列的缺失值。"""
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行均值填充。")
        return df
    if df[column].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(df[column]):
        imputer = SimpleImputer(strategy='mean')
        df[column] = imputer.fit_transform(df[[column]])
        print(f"列 '{column}' 的缺失值已通过均值填充。")
    return df


def impute_median(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """使用中位数填充指定列的缺失值。"""
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行中位数填充。")
        return df
    if df[column].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(df[column]):
        imputer = SimpleImputer(strategy='median')
        df[column] = imputer.fit_transform(df[[column]])
        print(f"列 '{column}' 的缺失值已通过中位数填充。")
    return df


def impute_most_frequent(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """使用众数填充指定列的缺失值。"""
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行众数填充。")
        return df
    if df[column].isnull().sum() > 0:
        imputer = SimpleImputer(strategy='most_frequent')
        df[column] = imputer.fit_transform(df[[column]])
        print(f"列 '{column}' 的缺失值已通过众数填充。")
    return df


def delete_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """删除指定列。"""
    if column in df.columns:
        df = df.drop(columns=[column])
        print(f"列 '{column}' 已被删除。")
    else:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法删除。")
    return df


def delete_rows_with_missing_in_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """删除在指定列中包含缺失值的行。"""
    if column in df.columns:
        original_len = len(df)
        df = df.dropna(subset=[column])
        print(f"在列 '{column}' 中包含缺失值的 {original_len - len(df)} 行已被删除。")
    else:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法基于此列删除行。")
    return df


# --- 数据缩放 ---
def standard_scale_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, StandardScaler]:
    """对指定数值列进行标准化缩放。"""
    scaler = StandardScaler()
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行标准化。")
        return df, scaler  # 返回原始df和未拟合的scaler
    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = scaler.fit_transform(df[[column]])
        print(f"列 '{column}' 已进行标准化缩放。")
    else:
        print(f"警告: 列 '{column}' 不是数值类型，无法执行标准化。")
    return df, scaler  # 返回处理后的df和拟合后的scaler


def min_max_scale_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """对指定数值列进行最小-最大缩放。"""
    scaler = MinMaxScaler()
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行MinMax缩放。")
        return df, scaler
    if pd.api.types.is_numeric_dtype(df[column]):
        df[column] = scaler.fit_transform(df[[column]])
        print(f"列 '{column}' 已进行MinMax缩放。")
    else:
        print(f"警告: 列 '{column}' 不是数值类型，无法执行MinMax缩放。")
    return df, scaler


# --- 类别特征编码 ---
def one_hot_encode_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """对指定类别列进行独热编码。"""
    encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False for dense array
    if column not in df.columns:
        print(f"警告: 列 '{column}' 不在DataFrame中，无法执行独热编码。")
        return df, encoder

    # 确保列是字符串类型，并且填充NaN为特定标记（如果需要）

    try:
        encoded_data = encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column]), index=df.index)
        df = df.drop(columns=[column])
        df = pd.concat([df, encoded_df], axis=1)
        print(f"列 '{column}' 已进行独热编码。")
    except Exception as e:
        print(f"对列 '{column}' 进行独热编码失败: {e}")
    return df, encoder


# --- 离群值处理 (简单) ---
def cap_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    """使用IQR方法对指定列的离群值进行封顶处理。"""
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        print(f"警告: 列 '{column}' 无效或非数值，无法处理离群值。")
        return df

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
    return df


# 方法映射表，方便LLM选择和程序调用
PREPROCESSING_METHODS_MAP = {
    "impute_mean": impute_mean,
    "impute_median": impute_median,
    "impute_most_frequent": impute_most_frequent,
    "delete_column": delete_column,
    "delete_rows_with_missing_in_column": delete_rows_with_missing_in_column,
    "standard_scale_column": standard_scale_column,
    "min_max_scale_column": min_max_scale_column,
    "one_hot_encode_column": one_hot_encode_column,
    "cap_outliers_iqr": cap_outliers_iqr,
}