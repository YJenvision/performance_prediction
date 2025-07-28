# @Time    : 2025/7/2
# @Author  : ZhangJingLiang
# @Email   : jinglianglink@qq.com
# @Project : performance_prediction_agent

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import json
import os


class ResultHandler:
    def __init__(self, pipeline_summary: Dict[str, Any], run_specific_dir: str):
        """
        初始化结果处理器。
        """
        self.pipeline_summary = pipeline_summary
        self.run_specific_dir = run_specific_dir  # 保存专属运行目录
        self.final_model_info: Optional[Dict[str, Any]] = None
        self.evaluation_metrics: Optional[Dict[str, float]] = None
        self.feature_importances: Optional[pd.Series] = None
        self.overall_status: str = pipeline_summary.get("final_status", "unknown")

    def add_model_details(self, model_name: str, best_hyperparams: Optional[Dict[str, Any]]):
        """
        添加最终选择和训练的模型的详细信息。
        """
        self.final_model_info = {
            "model_name": model_name,
            "best_hyperparameters": best_hyperparams
        }

    def add_evaluation_metrics(self, metrics: Optional[Dict[str, float]]):
        """
        添加最终模型的评估指标。
        """
        self.evaluation_metrics = metrics

    def add_feature_importances(self, importances: Optional[pd.Series]):
        """
        添加最终模型的特征重要性。
        """
        self.feature_importances = importances

    def compile_final_result(self) -> Dict[str, Any]:
        """
        汇编所有结果信息。
        """
        final_result_package = {
            "status": self.overall_status,
            "user_request": self.pipeline_summary.get("user_request", {}),
            "pipeline_run_id": self.pipeline_summary.get("pipeline_id", "N/A"),
            "modeling_start_time": self.pipeline_summary.get("start_time"),
            "modeling_end_time": self.pipeline_summary.get("end_time"),
            "modeling_duration_seconds": self.pipeline_summary.get("duration_seconds"),
            "selected_model": self.final_model_info,
            "evaluation_metrics": self.evaluation_metrics,
            "feature_importances_top10": self.feature_importances.head(
                10).to_dict() if self.feature_importances is not None and not self.feature_importances.empty else None,
            "pipeline_stages_summary": self.pipeline_summary.get("stages", [])
        }
        return {k: v for k, v in final_result_package.items() if v is not None}

    def save_final_result(self, filepath: Optional[str] = None) -> str:
        """
        将最终结果保存到JSON文件。
        """
        result_package = self.compile_final_result()
        if filepath is None:
            # 构建结果文件的保存路径
            results_dir = os.path.join(self.run_specific_dir, "result_info")
            os.makedirs(results_dir, exist_ok=True)

            # 使用 pipeline_run_id (即文件夹名) 来命名结果文件，确保关联性
            run_id = result_package.get('pipeline_run_id', 'unknown_run')
            filename = f"{run_id}_result_summary.json"
            filepath = os.path.join(results_dir, filename)

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_package, f, indent=4, ensure_ascii=False, default=str)
            print(f"最终结果已保存到: {filepath}")
        except Exception as e:
            print(f"保存最终结果失败: {e}")
        return filepath
