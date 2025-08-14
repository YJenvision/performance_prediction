import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# 定义一个阈值，用于判断是否为低基数类别特征。
# 如果特征的唯一值数量低于或等于此阈值，画像中将列出具体的唯一值。
LOW_CARDINALITY_THRESHOLD = 10


def generate_data_profile(df: pd.DataFrame, target_metric: str = None) -> Dict[str, Any]:
    """
    为DataFrame生成一个为大语言模型（LLM）消耗而优化的、精简高效的数据画像。

    这个最终版本的画像有以下核心改进：
    1.  **语义化类型（Semantic Types）**: 将Pandas的物理类型抽象为逻辑类型（'numeric', 'categorical', 'binary', 'datetime', 'empty'），为智能体提供更直接的决策依据。
    2.  **决策导向的统计信息**:
        - 对数值型特征，提供关键的统计摘要，并新增 **偏度(skewness)** 和 **峰度(kurtosis)**，为变换和异常值处理提供关键依据。
        - 对类别型特征，提供基数（cardinality），并对低基数特征直接展示其唯一值。
    3.  **极致精简**: 画像中不包含无用信息，有效缩短输入给智能体的上下文长度。
    4.  **逻辑清晰**: 数值型（int, float）将始终被视为 'numeric'，不再根据基数转换为 'categorical'，使类型判断更加一致和稳健。

    参数:
    - df: 输入的Pandas DataFrame。
    - target_metric: 目标列的名称，该列将被排除在画像之外。

    返回:
    - 一个包含优化后数据画像信息的字典。
    """
    profile = {
        "summary": {
            "num_rows": len(df),
            "num_cols": len(df.columns),
        },
        "column_profiles": {}
    }

    for col_name in df.columns:
        # 目标列不参与特征画像的生成
        if col_name == target_metric:
            continue

        series = df[col_name]
        non_null_data = series.dropna()
        col_profile = {}

        # 1. 处理全空列
        if non_null_data.empty:
            col_profile['type'] = 'empty'
            profile["column_profiles"][col_name] = col_profile
            continue

        # 2. 计算通用信息
        missing_percentage = series.isnull().mean() * 100
        if missing_percentage > 0:
            col_profile['missing_percentage'] = round(missing_percentage, 2)

        nunique = non_null_data.nunique()

        # 3. 根据数据类型进行针对性画像
        if pd.api.types.is_numeric_dtype(non_null_data) and not pd.api.types.is_bool_dtype(non_null_data):
            col_profile['type'] = 'numeric'
            stats = non_null_data.describe()
            col_profile['stats'] = {
                'mean': round(stats.get('mean', 0), 3),
                'std': round(stats.get('std', 0), 3),
                'min': round(stats.get('min', 0), 3),
                'median': round(non_null_data.median(), 3),
                'max': round(stats.get('max', 0), 3),
                'skewness': round(non_null_data.skew(), 3),  # 新增偏度
                'kurtosis': round(non_null_data.kurt(), 3)  # 新增峰度
            }

            q1 = non_null_data.quantile(0.25)
            q3 = non_null_data.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = non_null_data[(non_null_data < lower_bound) | (non_null_data > upper_bound)]
                if not outliers.empty:
                    outlier_percentage = len(outliers) / len(non_null_data) * 100
                    col_profile['outlier_percentage'] = round(outlier_percentage, 2)

        elif pd.api.types.is_bool_dtype(non_null_data) or nunique == 2:
            col_profile['type'] = 'binary'
            col_profile['cardinality'] = 2
            col_profile['unique_values'] = [str(v) for v in non_null_data.unique()]

        elif pd.api.types.is_datetime64_any_dtype(non_null_data):
            col_profile['type'] = 'datetime'
            col_profile['cardinality'] = nunique
            try:
                col_profile['stats'] = {
                    'min': non_null_data.min().isoformat(),
                    'max': non_null_data.max().isoformat()
                }
            except TypeError:
                col_profile['stats'] = {
                    'min': str(non_null_data.min()),
                    'max': str(non_null_data.max())
                }
        else:
            col_profile['type'] = 'categorical'
            col_profile['cardinality'] = nunique
            if nunique <= LOW_CARDINALITY_THRESHOLD:
                col_profile['unique_values'] = [str(v) for v in non_null_data.unique()]

        profile["column_profiles"][col_name] = col_profile

    return profile


def _percent(x: float) -> float:
    """将比率转换为百分比的辅助函数"""
    return round(float(x) * 100.0, 2)


def generate_iterative_profile(
        df: pd.DataFrame,
        target_metric: Optional[str] = None,
        candidate_thresholds: tuple = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
        topk_missing_cols: int = 40
) -> Dict[str, Any]:
    """
    为数据样本清洗生成一个轻量级的画像 (此函数与您提供的版本基本一致，保持不变)。
    """
    features = [c for c in df.columns if c != target_metric]
    n_rows, n_cols = len(df), len(features) if features else 0

    if n_rows == 0 or n_cols == 0:
        return {
            "数据摘要": {"数据总行数": n_rows, "特征列数": n_cols, "全局缺失率(%)": 0.0},
            "行缺失率分布": {
                "均值": 0.0, "中位数": 0.0, "90%分位数行缺失率(p90)": 0.0,
                "95%分位数行缺失率(p95)": 0.0, "最大值": 0.0,
                "候选阈值": list(candidate_thresholds), "位于各阈值上方的行数": {}
            },
            "特征缺失率(%)详情(降序)": {}
        }

    miss_mat = df[features].isna()
    row_missing_ratio = miss_mat.mean(axis=1).astype(float)
    col_missing_pct = miss_mat.mean(axis=0).apply(_percent).to_dict()

    top_cols = sorted(col_missing_pct.items(), key=lambda kv: kv[1], reverse=True)[:topk_missing_cols]

    rows_above = {
        str(th): int((row_missing_ratio > th).sum())
        for th in candidate_thresholds
    }

    def q(p):
        return float(row_missing_ratio.quantile(p)) if not row_missing_ratio.empty else 0.0

    profile = {
        "数据摘要": {
            "数据总行数": n_rows,
            "特征列数": n_cols,
            "全局缺失率(%)": _percent(miss_mat.values.sum() / (n_rows * n_cols))
        },
        "行缺失率分布": {
            "均值": round(float(row_missing_ratio.mean()), 4),
            "中位数": round(float(row_missing_ratio.median()), 4),
            "90%分位数行缺失率(p90)": round(q(0.90), 4),
            "95%分位数行缺失率(p95)": round(q(0.95), 4),
            "最大值": round(float(row_missing_ratio.max()) if n_rows > 0 else 0.0, 4),
            "候选阈值": list(candidate_thresholds),
            "位于各阈值上方的行数": rows_above
        },
        "特征缺失率(%)详情(降序)": dict(top_cols)
    }

    return profile
