import pandas as pd
import numpy as np
from typing import Dict, Any


def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    生成DataFrame的数据画像。数据探索阶段，了解数据集的基本情况和质量帮助辅助。

    参数:
    - df: 输入的Pandas DataFrame。

    返回:
    - 一个包含数据画像信息的字典。
        包括：
            总行数和总列数
            每列的详细统计信息（数据类型、缺失值、唯一值等）
            数值列的统计量（均值、标准差、分位数等）
    {
    "num_rows": 总行数,
    "num_cols": 总列数,
    "column_details": {
        "列名1": {
            "dtype": 数据类型,
            "missing_percentage": 缺失值百分比,
            "unique_values": 唯一值数量,

            # 数值列特有
            "potential_outliers_iqr": 基于IQR的潜在离群值数量,
            "potential_outliers_percentage_iqr": 基于IQR的潜在离群值百分比,

        },
        # 其他列...
    }
}
    """

    profile = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "column_details": {}
    }

    for col in df.columns:
        col_data = df[col]

        # 获取特征列中的非空数据用于类型判断
        non_null_data = col_data.dropna()

        details = {
            "dtype": str(non_null_data.dtype) if not non_null_data.empty else "allNull",
        }

        # 缺失值计算方法：使用isna()而不是isnull()
        missing_percentage = float(round(col_data.isna().mean() * 100, 2))

        # 仅当缺失值百分比不为 0 时添加该字段
        if missing_percentage:
            details["missing_percentage"] = missing_percentage

        # 仅当唯一值数量不为 0 时添加 unique_values 字段
        unique_values = int(non_null_data.nunique()) if not non_null_data.empty else 0
        if unique_values:
            details["unique_values"] = unique_values

        # 使用非空数据判断类型
        if not non_null_data.empty and pd.api.types.is_numeric_dtype(non_null_data):
            # 简单离群值检测 (基于IQR)
            Q1 = non_null_data.quantile(0.25)
            Q3 = non_null_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            potential_outliers_iqr = int(len(outliers))
            potential_outliers_percentage_iqr = float(
                round(len(outliers) / len(col_data.dropna()) * 100 if len(col_data.dropna()) > 0 else 0, 2)
            )
            if potential_outliers_iqr > 0:
                details["potential_outliers_iqr"] = potential_outliers_iqr
            if potential_outliers_percentage_iqr > 0:
                details["potential_outliers_percentage_iqr"] = potential_outliers_percentage_iqr

        profile["column_details"][col] = details

    return profile
