import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Union


def _set_plot_style(title: str, xlabel: str, ylabel: str, ax: plt.Axes):
    """设置图表的通用样式。"""
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


def _set_legend_style(ax: plt.Axes, loc: str = 'lower right'):
    """设置图例的统一样式。"""
    legend = ax.legend(loc=loc, frameon=True, fancybox=True, shadow=True,
                       borderpad=1, fontsize=12)  # 统一图例字体大小
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    return legend


def plot_prediction_vs_actual(
        y_true: pd.Series,
        y_pred: np.ndarray,
        acceptable_error: Dict[str, Union[str, float]],
        target_metric: str,
        model_name: str,
        dataset_name: str,
        output_dir: str = "automl_runs//visualization"
) -> str:
    """
    绘制真实值与预测值的对比散点图，并高亮显示在可接受误差范围内的点。
    新增功能：在图上显示样本统计信息。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        ax.plot([lim_min, lim_max], [lim_min * (1 + error_value / 100), lim_max * (1 + error_value / 100)], 'g--',
                lw=2, label=f'误差边界线 (±{error_value}%)')
        ax.plot([lim_min, lim_max], [lim_min * (1 - error_value / 100), lim_max * (1 - error_value / 100)], 'g--', lw=2)
    else:
        ax.plot([lim_min, lim_max], [lim_min + error_value, lim_max + error_value], 'g--', lw=2,
                label=f'误差边界线 (±{error_value})')
        ax.plot([lim_min, lim_max], [lim_min - error_value, lim_max - error_value], 'g--', lw=2)

    title = f'{model_name} 在 {dataset_name} 上的预测结果\n(目标: {target_metric})'
    _set_plot_style(title, '真实值', '预测值', ax)

    total_samples = len(data)
    in_bounds_count = len(in_bounds_data)
    out_of_bounds_count = len(out_of_bounds_data)
    in_bounds_perc = (in_bounds_count / total_samples) * 100 if total_samples > 0 else 0

    stats_text = (
        f"总样本数: {total_samples}\n"
        f"符合误差区间样本数: {in_bounds_count} ({in_bounds_perc:.2f}%)\n"
        f"超出误差区间样本数: {out_of_bounds_count} ({100 - in_bounds_perc:.2f}%)"
    )
    _add_stats_textbox(ax, stats_text)

    _set_legend_style(ax, 'lower right')

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{dataset_name}_prediction_vs_actual_plot_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"预测值vs真实值图表已保存至: {filepath}")
    return filepath


def plot_error_distribution(
        y_true: pd.Series,
        y_pred: np.ndarray,
        acceptable_error: Dict[str, Union[str, float]],
        target_metric: str,
        model_name: str,
        dataset_name: str,
        output_dir: str = "automl_runs//visualization"
) -> str:
    """
    绘制预测误差的分布直方图。
    智能切换：如果误差类型是'percentage'，则绘制百分比误差分布；否则绘制绝对误差分布。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    error_type = acceptable_error.get("type", "percentage")
    error_value = float(acceptable_error.get("value", 10))

    xlabel, title, plot_data, upper_bound, lower_bound = '', '', pd.Series(), 0.0, 0.0

    if error_type == 'percentage':
        # --- 绘制百分比误差分布 ---
        xlabel = f'预测百分比误差 (%)'
        title = f'{model_name} 在 {dataset_name} 上的预测百分比误差分布\n(目标: {target_metric})'

        # 替换y_true中的0为np.nan以安全地进行除法运算
        y_true_safe = y_true.replace(0, np.nan)
        percentage_error = ((y_pred - y_true) / y_true_safe * 100)

        # 移除因y_true为0而产生的NaN/inf值，这些点不参与分布图绘制
        plot_data = percentage_error.dropna().replace([np.inf, -np.inf], np.nan).dropna()

        # 边界是固定的百分比值
        upper_bound, lower_bound = error_value, -error_value
        is_within_bounds = np.abs(percentage_error) <= error_value

    else:  # 'value'
        # --- 绘制绝对误差分布 ---
        xlabel = '预测误差 (预测值 - 真实值)'
        title = f'{model_name} 在 {dataset_name} 上的预测误差分布\n(目标: {target_metric})'
        absolute_error = y_pred - y_true
        plot_data = pd.Series(absolute_error)
        upper_bound, lower_bound = error_value, -error_value
        is_within_bounds = np.abs(absolute_error) <= error_value

    # 绘制误差分布图
    sns.histplot(plot_data, kde=True, ax=ax, bins=50, color='skyblue', edgecolor='black')

    # 绘制参考线
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='零误差线 (Error = 0)')
    ax.axvline(upper_bound, color='green', linestyle='--', linewidth=2, label=f'可接受误差上界 ({upper_bound:+.2f})')
    ax.axvline(lower_bound, color='green', linestyle='--', linewidth=2, label=f'可接受误差下界 ({lower_bound:+.2f})')

    _set_plot_style(title, xlabel, '频数', ax)

    # 添加统计信息文本框
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

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{model_name}_{dataset_name}_error_distribution_plot_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"误差分布图表已保存至: {filepath}")
    return filepath
