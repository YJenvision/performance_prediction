from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import json
import os  # 导入os模块


class ResultHandler:
    def __init__(self, pipeline_summary: Dict[str, Any]):
        """
        初始化结果处理器。

        参数:
        - pipeline_summary: PipelineBuilder生成的Pipeline摘要。
        """
        self.pipeline_summary = pipeline_summary
        self.final_model_info: Optional[Dict[str, Any]] = None
        self.evaluation_metrics: Optional[Dict[str, float]] = None
        self.feature_importances: Optional[pd.Series] = None
        self.overall_status: str = pipeline_summary.get("final_status", "unknown")

    def add_model_details(self, model_name: str, best_hyperparams: Optional[Dict[str, Any]],
                          model_object_ref: Optional[str] = None):
        """
        添加最终选择和训练的模型的详细信息。

        参数:
        - model_name: 最终模型的名称。
        - best_hyperparams: 模型训练后得到的最佳超参数。
        - model_object_ref: 存储的实际模型对象的引用或路径 (可选)。
        """
        self.final_model_info = {
            "model_name": model_name,
            "best_hyperparameters": best_hyperparams,
            "model_object_reference": model_object_ref if model_object_ref else "Not explicitly saved in this version"
        }
        print(f"结果处理器: 添加模型详情 - {model_name}")

    def add_evaluation_metrics(self, metrics: Optional[Dict[str, float]]):
        """
        添加最终模型的评估指标。

        参数:
        - metrics: 包含评估指标 (如 MSE, R2) 的字典。
        """
        self.evaluation_metrics = metrics
        if metrics:
            print(f"结果处理器: 添加评估指标 - R2 Score: {metrics.get('r2', 'N/A')}")

    def add_feature_importances(self, importances: Optional[pd.Series]):
        """
        添加最终模型的特征重要性。

        参数:
        - importances: Pandas Series，索引为特征名，值为重要性分数。
        """
        self.feature_importances = importances
        if importances is not None:
            print(
                f"结果处理器: 添加特征重要性 (Top 3): \n{importances.head(3).to_dict() if not importances.empty else 'No importances'}")

    def compile_final_result(self) -> Dict[str, Any]:
        """
        汇编所有结果信息。

        返回:
        - 一个包含所有关键结果的字典。
        """
        final_result_package = {
            "status": self.overall_status,  # 最终执行状态
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

        # 清理掉值为None的键，使输出更简洁
        return {k: v for k, v in final_result_package.items() if v is not None}

    def save_final_result(self, filepath: Optional[str] = None) -> str:
        """
        将最终结果保存到JSON文件。

        参数:
        - filepath: 保存文件的路径。如果为None，则生成默认文件名。

        返回:
        - 实际保存的文件路径。
        """
        result_package = self.compile_final_result()
        if filepath is None:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                       "automl_runs//result_info")
            os.makedirs(results_dir, exist_ok=True)
            filepath = os.path.join(results_dir,
                                    f"{result_package.get('pipeline_run_id', 'unknown_run')}.json")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_package, f, indent=4, ensure_ascii=False, default=str)
            print(f"最终结果已保存到: {filepath}")
        except Exception as e:
            print(f"保存最终结果失败: {e}")
        return filepath
