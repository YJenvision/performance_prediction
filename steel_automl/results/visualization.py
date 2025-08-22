import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Union, Any, Tuple
import io
import base64
import shap


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


def _generate_plot_filename(model_name: str, dataset_name: str) -> str:
    """
    生成标准化的图表文件名。
    命名逻辑: {model_name}_{dataset_name}_{plot_type}.png
    """

    def format_param(param_value: Any) -> str:
        if param_value is None:
            return ""
        if isinstance(param_value, list):
            return "-".join(map(str, param_value))
        return str(param_value).replace('/', '-')

    parts = [
        model_name,
        dataset_name,
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
        output_dir: str,
        run_specific_dir_name: str  # 【新增】传入当前运行的顶级目录名
) -> str:
    """
    绘制真实值与预测值的对比散点图。
    : 现在返回图片的 URL 路径。
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
            non_zero_mask = y_true != 0
            is_within_bounds = np.zeros_like(y_true, dtype=bool)
            is_within_bounds[non_zero_mask] = (abs_error[non_zero_mask] / y_true.loc[non_zero_mask]) <= (
                    error_value / 100)
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

    filename = _generate_plot_filename(model_name, dataset_name)
    filepath = os.path.join(plot_specific_dir, filename)

    # 1. 保存到文件
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 2. 构建并返回可访问的 URL 路径
    # URL 格式: /runs/{run_specific_dir_name}/visualization/prediction_vs_actual/{filename}
    # 注意：这里使用正斜杠 '/' 来确保 URL 格式正确
    image_url = f"runs/{run_specific_dir_name}/visualization/prediction_vs_actual/{filename}"

    return image_url


def plot_error_distribution(
        y_true: pd.Series,
        y_pred: np.ndarray,
        acceptable_error: Dict[str, Union[str, float]],
        target_metric: str,
        model_name: str,
        dataset_name: str,
        request_params: Dict[str, Any],
        timestamp_str: str,
        output_dir: str,
        run_specific_dir_name: str  # 【新增】
) -> str:
    """
    绘制预测误差的分布直方图。
    : 现在返回图片的 URL 路径。
    """
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

    filename = _generate_plot_filename(model_name, dataset_name)
    filepath = os.path.join(plot_specific_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 构建并返回 URL
    image_url = f"runs/{run_specific_dir_name}/visualization/error_distribution/{filename}"
    return image_url


def plot_value_distribution(
        y_true: pd.Series,
        y_pred: np.ndarray,
        target_metric: str,
        model_name: str,
        dataset_name: str,
        request_params: Dict[str, Any],
        timestamp_str: str,
        output_dir: str,
        run_specific_dir_name: str  # 【新增】
) -> str:
    """
    绘制真实值与预测值的数值分布对比直方图。
    : 现在返回图片的 URL 路径。
    """
    plot_specific_dir = os.path.join(output_dir, "value_distribution")
    os.makedirs(plot_specific_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    stats_true = {"N": len(y_true)}
    stats_pred = {"N": len(y_pred)}
    label_true = f"实际值 (样本数:{stats_true['N']})"
    label_pred = f"预测值 (样本数:{stats_pred['N']})"

    combined_data = np.concatenate((y_true, y_pred))
    bins = np.histogram_bin_edges(combined_data, bins='auto')

    sns.histplot(y_true, bins=bins, ax=ax, color="C0", label=label_true, alpha=0.7, kde=False, lw=0)
    sns.histplot(y_pred, bins=bins, ax=ax, color="C1", label=label_pred, alpha=0.6, kde=False, lw=0)

    title = f'{model_name} 在 {dataset_name} 上真实值与预测值分布对比\n(目标: {target_metric})'
    _set_plot_style(title, f'{target_metric} 数值', '频数', ax)
    _set_legend_style(ax, loc='upper right')

    filename = _generate_plot_filename(model_name, dataset_name)
    filepath = os.path.join(plot_specific_dir, filename)

    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 构建并返回 URL
    image_url = f"runs/{run_specific_dir_name}/visualization/value_distribution/{filename}"
    return image_url


def plot_shap_summary_combined(
        shap_values: np.ndarray,
        X: pd.DataFrame,
        model_name: str,
        dataset_name: str,
        output_dir: str,
        run_specific_dir_name: str  # 【新增】
) -> str:
    """
    绘制组合的SHAP蜂巢图和特征重要性条形图，并返回图片的 URL 路径。
    """
    plot_specific_dir = os.path.join(output_dir, "feature_importance")
    os.makedirs(plot_specific_dir, exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    max_display = min(30, X.shape[1])
    fig_height = max(8, max_display * 0.4)
    fig, ax1 = plt.subplots(figsize=(10, fig_height), dpi=200)

    shap.summary_plot(shap_values, X, plot_type="dot", max_display=max_display, show=False, color_bar=True)
    ax1 = plt.gca()
    ax2 = ax1.twiny()
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False, color_bar=False)

    for bar in ax2.patches:
        bar.set_alpha(0.2)

    ax1.set_xlabel('Shapley Value (对模型输出的影响)', fontsize=12)
    ax2.set_xlabel('Mean |SHAP Value| (平均影响幅度)', fontsize=12)
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()
    ax1.set_ylabel('特征', fontsize=12)
    plt.tight_layout(pad=1.5)

    filename = f"{model_name}_{dataset_name}_SHAP_Combined_Summary.png"
    filepath = os.path.join(plot_specific_dir, filename)

    plt.savefig(filepath, bbox_inches='tight')
    plt.close(fig)

    # 构建并返回 URL
    image_url = f"runs/{run_specific_dir_name}/visualization/feature_importance/{filename}"
    return image_url
