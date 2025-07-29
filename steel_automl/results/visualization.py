# @Time    : 2025/7/29 09:16
# @Author  : ZhangJingLiang
# @Email   : jinglianglink@qq.com
# @Project : performance_prediction_agent_stream

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Union, Any


def _set_plot_style(title: str, xlabel: str, ylabel: str, ax: plt.Axes):
    """设置图表的通用样式。"""
    # 中文字体设置，确保图表能正确显示中文
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 统一所有字体大小
    ax.set_title(title, fontsize=16, pad=20, linespacing=1.8, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=14, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    # 统一刻度标签字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)


def _add_stats_textbox(ax: plt.Axes, text: str):
    """在图表左上角添加一个信息框。"""
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.7)
    # 统一信息框字体大小
    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)


def _set_legend_style(ax: plt.Axes, loc: str = 'best'):
    """设置图例的统一样式。"""
    legend = ax.legend(loc=loc, frameon=True, fancybox=True, shadow=True,
                       borderpad=1, fontsize=12)  # 统一图例字体大小
    if legend:
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
    return legend


def _generate_plot_filename(request_params: Dict[str, Any], model_name: str, timestamp_str: str, dataset_name: str,
                            plot_type: str) -> str:
    """
    生成标准化的图表文件名。
    命名逻辑: 目标性能_数据时间范围_牌号_机组_出钢记号_钢种_当前时间_模型算法_{dataset_name}_{plot_type}.png
    """

    def format_param(param_value: Any) -> str:
        if param_value is None:
            return ""
        if isinstance(param_value, list):
            return "-".join(map(str, param_value))
        return str(param_value).replace('/', '-')

    parts = [
        format_param(request_params.get("target_metric")),
        format_param(request_params.get("time_range")),
        format_param(request_params.get("sg_sign")),
        format_param(request_params.get("product_unit_no")),
        format_param(request_params.get("st_no")),
        format_param(request_params.get("steel_grade")),
        timestamp_str,
        model_name,
        dataset_name,
        plot_type
    ]
    filename = "_".join(filter(None, parts)) + ".png"
    return filename.replace(" ", "")


def plot_prediction_vs_actual(
        y_true: pd.Series,
        y_pred: np.ndarray,
        acceptable_error: Dict[str, Union[str, float]],
        target_metric: str,
        model_name: str,
        dataset_name: str,
        request_params: Dict[str, Any],
        timestamp_str: str,
        output_dir: str
) -> str:
    """
    绘制真实值与预测值的对比散点图，并高亮显示在可接受误差范围内的点。
    """
    # 创建特定类型的子文件夹
    plot_specific_dir = os.path.join(output_dir, "prediction_vs_actual")
    os.makedirs(plot_specific_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    error_type = acceptable_error.get("type", "percentage")
    error_value = acceptable_error.get("value", 10)

    abs_error = np.abs(y_pred - y_true)

    if error_type == 'percentage':
        with np.errstate(divide='ignore', invalid='ignore'):
            # 仅在 y_true 不为0的地方计算百分比误差
            non_zero_mask = y_true != 0
            is_within_bounds = np.zeros_like(y_true, dtype=bool)
            is_within_bounds[non_zero_mask] = (abs_error[non_zero_mask] / y_true.loc[non_zero_mask]) <= (
                    error_value / 100)
            # 如果真实值为0，则只有预测值也为0才算在界内
            is_within_bounds[~non_zero_mask] = y_pred[~non_zero_mask] == 0
        error_str = f"±{error_value}%"
    else:  # value
        is_within_bounds = abs_error <= error_value
        error_str = f"±{error_value}"

    data = pd.DataFrame({'真实值': y_true, '预测值': y_pred, 'in_bounds': is_within_bounds})
    in_bounds_data = data[data['in_bounds']]
    out_of_bounds_data = data[~data['in_bounds']]

    ax.scatter(out_of_bounds_data['真实值'], out_of_bounds_data['预测值'],
               color='royalblue', alpha=0.6, s=50, label=f'误差 > {error_str}')
    ax.scatter(in_bounds_data['真实值'], in_bounds_data['预测值'],
               color='red', alpha=0.7, s=50, label=f'误差 ≤ {error_str}')

    lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', lw=2.5, label='理想线 (y = x)')

    if error_type == 'percentage':
        ax.plot([lim_min, lim_max], [lim_min * (1 + error_value / 100), lim_max * (1 + error_value / 100)], 'g--', lw=2,
                label=f'误差边界线 (±{error_value}%)')
        ax.plot([lim_min, lim_max], [lim_min * (1 - error_value / 100), lim_max * (1 - error_value / 100)], 'g--', lw=2)
    else:
        ax.plot([lim_min, lim_max], [lim_min + error_value, lim_max + error_value], 'g--', lw=2,
                label=f'误差边界线 (±{error_value})')
        ax.plot([lim_min, lim_max], [lim_min - error_value, lim_max - error_value], 'g--', lw=2)

    title = f'{model_name} 在 {dataset_name} 上的预测结果\n(目标: {target_metric})'
    _set_plot_style(title, '真实值', '预测值', ax)

    total_samples = len(data)
    in_bounds_count = len(in_bounds_data)
    in_bounds_perc = (in_bounds_count / total_samples) * 100 if total_samples > 0 else 0

    stats_text = (
        f"总样本数: {total_samples}\n"
        f"符合误差区间样本数: {in_bounds_count} ({in_bounds_perc:.2f}%)\n"
        f"超出误差区间样本数: {total_samples - in_bounds_count} ({100 - in_bounds_perc:.2f}%)"
    )
    _add_stats_textbox(ax, stats_text)
    _set_legend_style(ax, 'lower right')

    filename = _generate_plot_filename(request_params, model_name, timestamp_str, dataset_name,
                                       "prediction_vs_actual_plot")
    # 使用新的子文件夹路径保存文件
    filepath = os.path.join(plot_specific_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return filepath


def plot_error_distribution(
        y_true: pd.Series,
        y_pred: np.ndarray,
        acceptable_error: Dict[str, Union[str, float]],
        target_metric: str,
        model_name: str,
        dataset_name: str,
        request_params: Dict[str, Any],
        timestamp_str: str,
        output_dir: str
) -> str:
    """
    绘制预测误差的分布直方图。
    智能切换：如果误差类型是'percentage'，则绘制百分比误差分布；否则绘制绝对误差分布。
    """
    # 创建特定类型的子文件夹
    plot_specific_dir = os.path.join(output_dir, "error_distribution")
    os.makedirs(plot_specific_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    error_type = acceptable_error.get("type", "percentage")
    error_value = float(acceptable_error.get("value", 10))

    xlabel, title, plot_data, upper_bound, lower_bound = '', '', pd.Series(), 0.0, 0.0

    if error_type == 'percentage':
        xlabel = '预测百分比误差 (%)'
        title = f'{model_name} 在 {dataset_name} 上的预测百分比误差分布\n(目标: {target_metric})'
        # 安全地计算百分比误差，避免除以零
        y_true_safe = y_true.replace(0, np.nan)
        percentage_error = ((y_pred - y_true) / y_true_safe * 100)
        plot_data = percentage_error.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        upper_bound, lower_bound = error_value, -error_value
        is_within_bounds = np.abs(percentage_error) <= error_value
    else:  # 'value'
        xlabel = '预测误差 (预测值 - 真实值)'
        title = f'{model_name} 在 {dataset_name} 上的预测误差分布\n(目标: {target_metric})'
        absolute_error = y_pred - y_true
        plot_data = pd.Series(absolute_error)
        upper_bound, lower_bound = error_value, -error_value
        is_within_bounds = np.abs(absolute_error) <= error_value

    sns.histplot(plot_data, kde=True, ax=ax, bins=50, color='skyblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='零误差线 (Error = 0)')
    ax.axvline(upper_bound, color='green', linestyle='--', linewidth=2, label=f'可接受误差上界 ({upper_bound:+.2f})')
    ax.axvline(lower_bound, color='green', linestyle='--', linewidth=2, label=f'可接受误差下界 ({lower_bound:+.2f})')
    _set_plot_style(title, xlabel, '频数', ax)

    total_samples = len(y_true)
    in_bounds_count = np.sum(is_within_bounds)
    in_bounds_perc = (in_bounds_count / total_samples) * 100 if total_samples > 0 else 0

    stats_text = (
        f"总样本数: {total_samples}\n"
        f"符合误差区间样本数: {in_bounds_count} ({in_bounds_perc:.2f}%)\n"
        f"超出误差区间样本数: {total_samples - in_bounds_count} ({100 - in_bounds_perc:.2f}%)"
    )
    _add_stats_textbox(ax, stats_text)
    _set_legend_style(ax, 'upper right')

    filename = _generate_plot_filename(request_params, model_name, timestamp_str, dataset_name,
                                       "error_distribution_plot")
    # 使用新的子文件夹路径保存文件
    filepath = os.path.join(plot_specific_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return filepath


def plot_value_distribution(
        y_true: pd.Series,
        y_pred: np.ndarray,
        target_metric: str,
        model_name: str,
        dataset_name: str,
        request_params: Dict[str, Any],
        timestamp_str: str,
        output_dir: str
) -> str:
    """
    绘制真实值与预测值的数值分布对比直方图。
    """
    # 创建特定类型的子文件夹
    plot_specific_dir = os.path.join(output_dir, "value_distribution")
    os.makedirs(plot_specific_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- 1. 计算统计数据 ---
    stats_true = {
        "N": len(y_true),
    }
    stats_pred = {
        "N": len(y_pred),
    }

    # --- 2. 创建图例标签 ---
    label_true = f"实际值 (样本数:{stats_true['N']})"
    label_pred = f"预测值 (样本数:{stats_pred['N']})"

    # --- 3. 自适应计算分箱 ---
    # 合并数据以确定全局范围和最优分箱
    combined_data = np.concatenate((y_true, y_pred))
    # 使用 'auto' 策略，让 numpy 根据 Freedman-Diaconis rule 或 Sturges' formula 自动选择最佳分箱数
    bins = np.histogram_bin_edges(combined_data, bins='auto')

    # --- 4. 绘图 (移除边框: lw=0) ---
    # 绘制真实值分布
    sns.histplot(y_true, bins=bins, ax=ax, color="C0", label=label_true, alpha=0.7, kde=False, lw=0)
    # 绘制预测值分布
    sns.histplot(y_pred, bins=bins, ax=ax, color="C1", label=label_pred, alpha=0.6, kde=False, lw=0)

    # --- 5. 设置图表样式 ---
    title = f'{model_name} 在 {dataset_name} 上真实值与预测值分布对比\n(目标: {target_metric})'
    _set_plot_style(title, f'{target_metric} 数值', '频数', ax)
    _set_legend_style(ax, loc='upper right')

    # --- 6. 保存文件 ---
    filename = _generate_plot_filename(request_params, model_name, timestamp_str, dataset_name,
                                       "value_distribution_plot")
    # 使用新的子文件夹路径保存文件 ---
    filepath = os.path.join(plot_specific_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return filepath
