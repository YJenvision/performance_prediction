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
            布尔列的统计量（True/False计数）
            分类/对象列的频数统计
    {
    "num_rows": 总行数,
    "num_cols": 总列数,
    "column_details": {
        "列名1": {
            "dtype": 数据类型,
            "missing_percentage": 缺失值百分比,
            "unique_values": 唯一值数量,

            # 数值列特有
            "mean": 均值,
            "std": 标准差,
            "min": 最小值,
            "25%": 25分位数,
            "50%": 50分位数,
            "75%": 75分位数,
            "max": 最大值,
            "potential_outliers_iqr": 基于IQR的潜在离群值数量,
            "potential_outliers_percentage_iqr": 基于IQR的潜在离群值百分比,

            # 布尔列特有
            "true_count": True计数,
            "false_count": False计数,

            # 分类列特有
            "top_values": 前5高频值
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
            # details["mean"] = float(round(non_null_data.mean(), 3))
            # details["std"] = float(round(non_null_data.std(), 3))
            # details["min"] = float(round(non_null_data.min(), 3))
            # details["25%"] = float(round(non_null_data.quantile(0.25), 3))
            # details["50%"] = float(round(non_null_data.quantile(0.50), 3))
            # details["75%"] = float(round(non_null_data.quantile(0.75), 3))
            # details["max"] = float(round(non_null_data.max(), 3))
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

        elif pd.api.types.is_bool_dtype(non_null_data):  # 布尔类型处理
            details["true_percentage"] = float(round(non_null_data.mean() * 100, 2)) if not non_null_data.isnull().all() else None
            details["false_percentage"] = float(
                round((1 - non_null_data.mean()) * 100, 2)) if not non_null_data.isnull().all() else None

        profile["column_details"][col] = details

    return profile


if __name__ == '__main__':

    data = {
        'A': [1, 2, np.nan, 4, 5, 1, 2, 30],  # 包含NaN和离群值
        'B': ['x', 'y', 'x', 'z', 'y', 'x', 'x', 'y'],
        'C': [1.1, 2.2, 3.3, None, ' ', 1.1, 2.2, 3.3],
        'D': [True, False, True, True, None, ' ', False, True],
        'E': [None, None, None, None, None, ' ', None, None],
        'F': [None, None, None, None, None, None, None, None],  # 全是NaN
        'G': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # 全是NaN
        'H': [np.nan, np.nan, np.nan, np.nan, np.nan, 00, np.nan, np.nan],
    }
    sample_df = pd.DataFrame(data)

    # 预处理操作在分析数据画像前data_loader进行
    # 预处理：将列名全部转换为大写
    sample_df.columns = sample_df.columns.str.upper()
    # 预处理：将空字符串' '、'(null)'替换为NaN
    sample_df = sample_df.replace(r'^\s*$', np.nan, regex=True)
    sample_df = sample_df.replace(r'(null)', np.nan, regex=True)

    profile = generate_data_profile(sample_df)
    import json

    print(json.dumps(profile, indent=4))
