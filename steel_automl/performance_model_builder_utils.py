# @Time    : 2025/7/22 10:41
# @Author  : ZhangJingLiang
# @Email   : jinglianglink@qq.com
# @Project : performance_prediction_agent

import os
import pickle
from typing import Dict, Any

import pandas as pd


def _generate_filename_prefix(request_params: Dict[str, Any], timestamp_str: str) -> str:
    """
    根据请求参数生成标准化的文件名头部。
    命名标准: 目标性能_数据时间范围_牌号_机组_出钢记号_钢种_时间
    """

    def format_param(param_value: Any) -> str:
        """用于格式化列表或将单个值格式化为字符串的帮助函数。"""
        if param_value is None:
            return ""
        if isinstance(param_value, list):
            return "-".join(map(str, param_value))
        return str(param_value)

    parts = [
        format_param(request_params.get("target_metric")),
        format_param(request_params.get("time_range")),
        format_param(request_params.get("sg_sign")),
        format_param(request_params.get("product_unit_no")),
        format_param(request_params.get("st_no")),
        format_param(request_params.get("steel_grade")),
        timestamp_str,
    ]
    # 过滤掉空字符串部分并用下划线连接
    prefix = "_".join(filter(None, parts))
    return prefix.replace(" ", "").replace("/", "-")


def _generate_data_filename(filename_prefix: str, data_type_str: str) -> str:
    """
    根据文件名头部和数据类型生成标准化的数据文件名。
    """
    return f"{filename_prefix}_{data_type_str}.csv"


def _save_dataframe(df: pd.DataFrame, data_type_name: str, filename_prefix: str, run_dir: str) -> str:
    """
    保存DataFrame到指定目录并返回完整路径。
    """
    if df is None or df.empty:
        return ""
    data_dir = os.path.join(run_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    filename = _generate_data_filename(filename_prefix, data_type_name)
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"数据已保存: {filepath}")
    return filepath


def _save_fitted_objects(objects: Dict[str, Any], filename_prefix: str, run_dir: str, object_type: str) -> str:
    """
    使用pickle保存拟合的对象（如预处理器、特征生成器）。
    这些对象对于在部署后转换新数据至关重要。

    参数:
    - objects: 包含已拟合转换器的字典。
    - filename_prefix: 标准化的文件名头部。
    - run_dir: 本次运行的根目录。
    - object_type: 对象的类型描述 (e.g., 'preprocessors', 'feature_generators')。

    返回:
    - 保存文件的完整路径，如果失败则返回空字符串。
    """
    if not objects:
        return ""

    run_dir = os.path.join(run_dir, object_type)

    # 确保主运行目录存在
    os.makedirs(run_dir, exist_ok=True)

    # 定义文件名，例如: ..._preprocessors.pkl
    filename = f"{filename_prefix}_{object_type}.pkl"
    filepath = os.path.join(run_dir, filename)

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(objects, f)
        print(f"拟合的对象已保存: {filepath}")
        return filepath
    except Exception as e:
        print(f"错误: 保存拟合的对象到 {filepath} 失败: {e}")
        return ""
