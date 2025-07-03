import pandas as pd
import numpy as np
from typing import Dict, Any, List

# 定义一个阈值，用于判断是否为低基数类别特征。
# 如果特征的唯一值数量低于或等于此阈值，画像中将列出具体的唯一值。
LOW_CARDINALITY_THRESHOLD = 10


def generate_data_profile(df: pd.DataFrame, target_metric: str = None) -> Dict[str, Any]:
    """
    为DataFrame生成一个为大语言模型（LLM）消耗而优化的、精简高效的数据画像。

    这个最终版本的画像有以下核心改进：
    1.  **语义化类型（Semantic Types）**: 将Pandas的物理类型抽象为逻辑类型（'numeric', 'categorical', 'binary', 'datetime', 'empty'），为智能体提供更直接的决策依据。
    2.  **决策导向的统计信息**:
        - 对数值型特征，提供关键的统计摘要（均值、中位数、标准差等），帮助智能体选择最合适的缺失值填充策略。
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
        # 检查是否为数值型 (并排除布尔型)
        if pd.api.types.is_numeric_dtype(non_null_data) and not pd.api.types.is_bool_dtype(non_null_data):
            # 所有整型和浮点型都视为 'numeric'
            col_profile['type'] = 'numeric'
            stats = non_null_data.describe()
            col_profile['stats'] = {
                'mean': round(stats.get('mean', 0), 3),
                'std': round(stats.get('std', 0), 3),
                'min': round(stats.get('min', 0), 3),
                'median': round(non_null_data.median(), 3),
                'max': round(stats.get('max', 0), 3),
            }

            # 离群值检测 (基于IQR)
            q1 = non_null_data.quantile(0.25)
            q3 = non_null_data.quantile(0.75)
            iqr = q3 - q1
            # 仅在IQR大于0时计算，避免除零错误或无效边界
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = non_null_data[(non_null_data < lower_bound) | (non_null_data > upper_bound)]
                # 仅在存在离群值时报告
                if not outliers.empty:
                    outlier_percentage = len(outliers) / len(non_null_data) * 100
                    col_profile['outlier_percentage'] = round(outlier_percentage, 2)

        # 检查是否为布尔型或二元特征
        elif pd.api.types.is_bool_dtype(non_null_data) or nunique == 2:
            col_profile['type'] = 'binary'
            col_profile['cardinality'] = 2
            col_profile['unique_values'] = [str(v) for v in non_null_data.unique()]

        # 检查是否为日期时间型
        elif pd.api.types.is_datetime64_any_dtype(non_null_data):
            col_profile['type'] = 'datetime'
            col_profile['cardinality'] = nunique
            try:
                col_profile['stats'] = {
                    'min': non_null_data.min().isoformat(),
                    'max': non_null_data.max().isoformat()
                }
            except TypeError:  # 兼容带时区的时间
                col_profile['stats'] = {
                    'min': str(non_null_data.min()),
                    'max': str(non_null_data.max())
                }

        # 其他所有情况都视为类别型
        else:
            col_profile['type'] = 'categorical'
            col_profile['cardinality'] = nunique
            # 仅在低基数时展示唯一值，以节省空间
            if nunique <= LOW_CARDINALITY_THRESHOLD:
                col_profile['unique_values'] = [str(v) for v in non_null_data.unique()]

        profile["column_profiles"][col_name] = col_profile

    return profile
