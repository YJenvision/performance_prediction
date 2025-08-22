# methods.py
import pandas as pd
import numpy as np
from typing import Any, Dict, Tuple, List, Union

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder,
    RobustScaler, PowerTransformer
)
from sklearn.impute import SimpleImputer
import category_encoders as ce


# ---------- 缺失值处理 ----------
def impute_mean(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """使用均值填充指定列的缺失值。"""
    message = f"特征 '{column}' 无需处理或非数值类型，跳过均值填充。"
    if column not in df.columns:
        message = f"特征 '{column}' 不存在，跳过均值填充。"
        return df, {'message': message}
    if df[column].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(df[column]):
        imputer = SimpleImputer(strategy='mean')
        df[column] = imputer.fit_transform(df[[column]]).ravel()
        message = f"对数值型特征 '{column}' 应用了均值填充。"
    return df, {'message': message}


def impute_median(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """使用中位数填充指定列的缺失值。"""
    message = f"特征 '{column}' 无需处理或非数值类型，跳过中位数填充。"
    if column not in df.columns:
        message = f"特征 '{column}' 不存在，跳过中位数填充。"
        return df, {'message': message}
    if df[column].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(df[column]):
        imputer = SimpleImputer(strategy='median')
        df[column] = imputer.fit_transform(df[[column]]).ravel()
        message = f"对数值型特征 '{column}' 应用了中位数填充。"
    return df, {'message': message}


def impute_most_frequent(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """使用众数填充指定列的缺失值。"""
    message = f"特征 '{column}' 无缺失值，跳过众数填充。"
    if column not in df.columns:
        message = f"特征 '{column}' 不存在，跳过众数填充。"
        return df, {'message': message}
    if df[column].isnull().sum() == 0:
        return df, {'message': message}

    # 保持原始数据类型
    original_dtype = df[column].dtype

    # 对数值或类别型数据统一使用SimpleImputer，更稳健
    imputer = SimpleImputer(strategy='most_frequent')
    df[column] = imputer.fit_transform(df[[column]]).ravel()
    message = f"对特征 '{column}' 应用了众数填充。"
    # 尝试恢复原始类型，失败则忽略
    try:
        df[column] = df[column].astype(original_dtype)
    except (ValueError, TypeError):
        pass

    return df, {'message': message}


def delete_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """删除指定的列。"""
    message = f"特征 '{column}' 不存在，无需删除。"
    if column in df.columns:
        df = df.drop(columns=[column])
        message = f"根据计划，删除了特征 '{column}'。"
    return df, {'message': message}


def delete_rows_with_missing_in_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """删除在指定列有缺失值的行。"""
    message = f"特征 '{column}' 不存在，无法删除行。"
    if column in df.columns:
        initial_rows = len(df)
        df = df.dropna(subset=[column])
        removed_count = initial_rows - len(df)
        message = f"删除了特征 '{column}' 中 {removed_count} 个包含缺失值的样本行。"
    return df, {'message': message}


def add_missing_indicator(df: pd.DataFrame, column: str, suffix: str = "_ismissing") -> Tuple[
    pd.DataFrame, Dict[str, str]]:
    """为指定列添加一个缺失指示器特征。"""
    message = f"特征 '{column}' 不存在，无法添加缺失指示器。"
    if column in df.columns:
        df[f"{column}{suffix}"] = df[column].isna().astype(int)
        message = f"为特征 '{column}' 添加了缺失指示器 '{column}{suffix}'。"
    return df, {'message': message}


def impute_auto(df: pd.DataFrame, column: str, skew_threshold: float = 1.0) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    自适应填充：
    通过偏度值（skewness）选择填充策略符合统计学原理：
    当数据分布接近对称（偏度绝对值 < 1.0）时，均值是更有效的集中趋势度量
    当数据显著偏斜（偏度绝对值 ≥ 1.0）时，中位数对极端值更稳健
    - 对数值列，根据偏度自动选择均值或中位数。
    - 对类别列，使用众数。
    """
    if column not in df.columns or df[column].isnull().sum() == 0:
        message = f"特征 '{column}' 不存在或无缺失值，跳过自动填充。"
        return df, {'message': message}

    ser = df[column]
    if pd.api.types.is_numeric_dtype(ser):
        # 根据偏度选择均/中位数
        skew_val = ser.dropna().skew()
        if abs(float(skew_val)) >= skew_threshold:
            impute_median(df, column)  # 执行填充
            message = f"特征 '{column}' 的偏度值为 {skew_val:.2f}，数据呈偏态分布，应用中位数进行填充。"
            return df, {'message': message}

        impute_mean(df, column)  # 执行填充
        message = f"特征 '{column}' 的偏度值为 {skew_val:.2f}，数据分布近似对称，应用均值进行填充。"
        return df, {'message': message}

    # 类别/二元 -> 众数
    impute_most_frequent(df, column)  # 执行填充
    message = f"特征 '{column}' 为类别型数据，应用众数进行填充。"
    return df, {'message': message}


# ---------- 缩放与变换 ----------
def standard_scale_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """对指定列进行标准化。"""
    scaler = StandardScaler()
    message = f"特征 '{column}' 无需处理或非数值类型，跳过标准化缩放。"
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        df[column] = scaler.fit_transform(df[[column]]).ravel()
        message = f"对数值型特征 '{column}' 应用了标准化缩放。"
    return df, {'message': message, 'fitted_object': scaler}


def min_max_scale_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """对指定列进行归一化。"""
    scaler = MinMaxScaler()
    message = f"特征 '{column}' 无需处理或非数值类型，跳过归一化缩放。"
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        df[column] = scaler.fit_transform(df[[column]]).ravel()
        message = f"对数值型特征 '{column}' 应用了归一化缩放。"
    return df, {'message': message, 'fitted_object': scaler}


def robust_scale_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """对指定列进行鲁棒缩放，对离群值不敏感。"""
    scaler = RobustScaler()
    message = f"特征 '{column}' 无需处理或非数值类型，跳过鲁棒缩放。"
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        df[column] = scaler.fit_transform(df[[column]]).ravel()
        message = f"对数值型特征 '{column}' 应用了鲁棒缩放。"
    return df, {'message': message, 'fitted_object': scaler}


def yeo_johnson_transform_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """对指定列进行Yeo-Johnson变换，使其更接近正态分布。"""
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    message = f"特征 '{column}' 无需处理或非数值类型，跳过Yeo-Johnson变换。"
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        col_data = df[[column]].astype(float)
        try:
            df[column] = pt.fit_transform(col_data).ravel()
            message = f"对数值型特征 '{column}' 应用了Yeo-Johnson变换。"
        except Exception:
            message = f"特征 '{column}' 应用Yeo-Johnson变换失败，跳过操作。"
            pass
    return df, {'message': message, 'fitted_object': pt}


# ---------- 类别编码 ----------
def one_hot_encode_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """对指定列进行独热编码。"""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=int)
    message = f"特征 '{column}' 不存在，跳过独热编码。"
    if column in df.columns:
        encoded = encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]), index=df.index)
        df = df.drop(columns=[column])
        df = pd.concat([df, encoded_df], axis=1)
        message = f"对类别型特征 '{column}' 应用了独热编码。"
    return df, {'message': message, 'fitted_object': encoder}


def label_encode_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """对指定列进行标签编码。"""
    encoder = LabelEncoder()
    message = f"特征 '{column}' 不存在，跳过标签编码。"
    if column in df.columns:
        series = df[column]
        if series.isnull().any():
            series = series.astype(str).fillna('__NULL__')
        df[column] = encoder.fit_transform(series)
        message = f"对类别型特征 '{column}' 应用了标签编码。"
    return df, {'message': message, 'fitted_object': encoder}


def target_encode_column(df: pd.DataFrame, column: str, target_metric_name: str) -> Tuple[
    pd.DataFrame, Dict[str, Any]]:
    """对指定列进行目标编码。"""
    encoder = ce.TargetEncoder(cols=[column], handle_unknown='value', smoothing=5)
    message = f"特征 '{column}' 或目标列 '{target_metric_name}' 不存在，跳过目标编码。"
    if column in df.columns and target_metric_name in df.columns:
        df[column] = encoder.fit_transform(X=df[column], y=df[target_metric_name])
        message = f"对类别型特征 '{column}' 应用了目标编码。"
    return df, {'message': message, 'fitted_object': encoder}


def frequency_encode_column(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """对指定列进行频率编码。"""
    message = f"特征 '{column}' 不存在，跳过频率编码。"
    freq_map = {}
    if column in df.columns:
        counts = df[column].value_counts(normalize=True, dropna=False)
        freq_map = counts.to_dict()
        df[column] = df[column].map(freq_map)
        message = f"对类别型特征 '{column}' 应用了频率编码。"
    return df, {'message': message, 'fitted_object': freq_map}


def rare_label_collapse(df: pd.DataFrame, column: str, min_freq: float = 0.01, rare_label: str = "__RARE__") -> Tuple[
    pd.DataFrame, Dict[str, Any]]:
    """将指定列中的稀有类别归并为一个标签。"""
    mapping = {}
    message = f"特征 '{column}' 不存在，跳过稀有标签合并。"
    if column in df.columns:
        vc = df[column].value_counts(normalize=True, dropna=False)
        rare_categories = vc[vc < min_freq].index
        mapping = {cat: rare_label for cat in rare_categories}
        if rare_categories.any():
            df[column] = df[column].replace(mapping)
            message = f"对类别型特征 '{column}' 的稀有标签进行了合并。"
        else:
            message = f"特征 '{column}' 中未发现稀有标签，跳过合并操作。"
    return df, {'message': message, 'fitted_object': mapping}


# ---------- 异常值处理 ----------
def cap_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """使用IQR规则对指定列的离群值进行封顶。"""
    message = f"特征 '{column}' 无需处理或非数值类型，跳过IQR封顶。"
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        col_data = df[column].dropna()
        if not col_data.empty:
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 - factor * IQR
                df[column] = df[column].clip(lower_bound, upper_bound)
                message = f"对数值型特征 '{column}' 的异常值应用了IQR封顶。"
            else:
                message = f"特征 '{column}' 的IQR为0，跳过IQR封顶。"
    return df, {'message': message}


def winsorize_by_quantile(df: pd.DataFrame, column: str, lower_q: float = 0.01, upper_q: float = 0.99) -> Tuple[
    pd.DataFrame, Dict[str, Any]]:
    """使用分位数对指定列进行缩尾处理。"""
    message = f"特征 '{column}' 无需处理或非数值类型，跳过分位数缩尾。"
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        col_data = df[column].dropna()
        if not col_data.empty:
            lower_limit = col_data.quantile(lower_q)
            upper_limit = col_data.quantile(upper_q)
            df[column] = df[column].clip(lower_limit, upper_limit)
            message = f"对数值型特征 '{column}' 的异常值应用了分位数缩尾。"
    return df, {'message': message}


# ---------- 方法映射字典 ----------
PREPROCESSING_METHODS_MAP = {
    # 缺失值处理
    "impute_mean": impute_mean,
    "impute_median": impute_median,
    "impute_most_frequent": impute_most_frequent,
    "impute_auto": impute_auto,
    "add_missing_indicator": add_missing_indicator,
    "delete_column": delete_column,
    "delete_rows_with_missing_in_column": delete_rows_with_missing_in_column,

    # 缩放/变换
    "standard_scale_column": standard_scale_column,
    "min_max_scale_column": min_max_scale_column,
    "robust_scale_column": robust_scale_column,
    "yeo_johnson_transform_column": yeo_johnson_transform_column,

    # 类别编码
    "rare_label_collapse": rare_label_collapse,
    "one_hot_encode_column": one_hot_encode_column,
    "label_encode_column": label_encode_column,
    "frequency_encode_column": frequency_encode_column,
    "target_encode_column": target_encode_column,

    # 异常值处理
    "cap_outliers_iqr": cap_outliers_iqr,
    "winsorize_by_quantile": winsorize_by_quantile,

    # 空操作
    "no_action": lambda df, column, **k: (df, {'message': f"特征 '{column}' 无需操作。"}),
}
