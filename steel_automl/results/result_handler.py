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
                                       "results_output")
            os.makedirs(results_dir, exist_ok=True)
            filepath = os.path.join(results_dir,
                                    f"final_result_{result_package.get('pipeline_run_id', 'unknown_run')}.json")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_package, f, indent=4, ensure_ascii=False,
                          default=str)  # default=str for pd.Series, datetime
            print(f"最终结果已保存到: {filepath}")
        except Exception as e:
            print(f"保存最终结果失败: {e}")
        return filepath


if __name__ == '__main__':
    # 模拟 PipelineBuilder 的输出
    mock_pipeline_summary = {
        "pipeline_id": "automl_20240525103000_TESTSG002",
        "user_request": {"sg_sign": "TESTSG002", "target_metric": "TensileStrength"},
        "start_time": "2024-05-25T10:30:00.000000",
        "end_time": "2024-05-25T10:35:00.000000",
        "duration_seconds": 300.0,
        "final_status": "success",
        "stages": [
            {"stage_name": "data_acquisition", "status": "success"},
            {"stage_name": "data_preprocessing", "status": "success"},
            {"stage_name": "feature_engineering", "status": "success"},
            {"stage_name": "model_selection", "status": "success"},
            {"stage_name": "model_training_evaluation", "status": "success"}
        ]
    }

    result_handler = ResultHandler(pipeline_summary=mock_pipeline_summary)

    result_handler.add_model_details(
        model_name="RandomForestRegressor",
        best_hyperparams={"n_estimators": 100, "max_depth": 10},
        model_object_ref="path/to/saved_model.pkl"  # 模拟
    )
    result_handler.add_evaluation_metrics(
        metrics={"mse": 25.5, "rmse": 5.05, "r2": 0.85}
    )

    # 模拟特征重要性
    mock_features = [f'feat_{i}' for i in range(10)]
    mock_importances = pd.Series(np.random.rand(10), index=mock_features).sort_values(ascending=False)
    result_handler.add_feature_importances(importances=mock_importances)

    final_output = result_handler.compile_final_result()
    print("\n编译后的最终结果包:")
    print(json.dumps(final_output, indent=2, default=str))

    # 测试保存
    saved_file = result_handler.save_final_result()
    print(f"结果包保存路径: {saved_file}")
    if os.path.exists(saved_file):
        print("结果文件创建成功。")
        # os.remove(saved_file) # 清理
    else:
        print("结果文件创建失败。")
