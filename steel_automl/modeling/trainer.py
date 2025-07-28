import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, List, Optional, Generator
import os
import joblib
from datetime import datetime
import traceback

from config import DEFAULT_RANDOM_STATE
from steel_automl.modeling.algorithms.random_forest import RandomForestModel
from steel_automl.modeling.algorithms.xgboost_model import XGBoostModel
from steel_automl.results.visualization import plot_prediction_vs_actual, plot_error_distribution

# 模型算法库中模型类映射表
MODEL_CLASSES = {
    "RandomForestRegressor": RandomForestModel,
    "XGBoostRegressor": XGBoostModel,
}


class ModelTrainer:
    def __init__(self, selected_model_name: str, model_info: Dict[str, Any], hpo_config: Dict[str, Any],
                 automl_plan: Dict[str, Any], run_specific_dir: str):
        """
        初始化模型训练器。
        MODIFIED: 添加 run_specific_dir 参数
        """
        if selected_model_name not in MODEL_CLASSES:
            raise ValueError(f"不支持的模型: {selected_model_name}。")

        self.model_name = selected_model_name
        self.model_class = MODEL_CLASSES[selected_model_name]
        self.hyperparam_suggestions = model_info.get("hyperparameter_suggestions", {})
        self.hpo_config = hpo_config
        self.acceptable_error = automl_plan.get("model_plan", {}).get("acceptable_error")
        self.request_params = automl_plan.get("user_request_details", {})
        self.target_metric = self.request_params.get("target_metric", "Unknown Target")
        self.run_timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
        self.current_stage = "模型训练与评估"

        # MODIFIED: 保存专属运行目录路径
        self.run_specific_dir = run_specific_dir

        self.model_instance = self.model_class(hyperparameters={})
        self.training_log: List[Dict[str, Any]] = []
        self.evaluation_results: Dict[str, Any] = {"train": None, "test": None, "artifacts": {}}
        self.trained_model_object = None
        self.feature_importances: Optional[pd.Series] = None

    def _generate_artifact_base_filename(self) -> str:
        """
        生成标准化的产物（模型、数据）文件名基础部分。
        命名标准: 目标性能_数据时间范围_牌号_机组_出钢记号_钢种_当前时间_模型算法
        """

        def format_param(param_value: Any) -> str:
            if param_value is None:
                return ""
            if isinstance(param_value, list):
                return "-".join(map(str, param_value))
            return str(param_value).replace('/', '-')

        parts = [
            format_param(self.request_params.get("target_metric")),
            format_param(self.request_params.get("time_range")),
            format_param(self.request_params.get("sg_sign")),
            format_param(self.request_params.get("product_unit_no")),
            format_param(self.request_params.get("st_no")),
            format_param(self.request_params.get("steel_grade")),
            self.run_timestamp_str,
            self.model_name,
        ]
        base_filename = "_".join(filter(None, parts))
        return base_filename.replace(" ", "")

    def _evaluate_and_visualize(self, X_data: pd.DataFrame, y_data: pd.Series, dataset_name: str) -> Tuple[Dict, str]:
        """
        封装评估和可视化的通用函数。
        """
        metrics, predictions = self.model_instance.evaluate(X_data, y_data, self.acceptable_error)

        if predictions is not None and self.acceptable_error and metrics is not None:
            # 为可视化图表指定正确的输出目录
            visualization_dir = os.path.join(self.run_specific_dir, "visualization")

            pred_vs_actual_path = plot_prediction_vs_actual(
                y_true=y_data, y_pred=predictions, acceptable_error=self.acceptable_error,
                target_metric=self.target_metric, model_name=self.model_name, dataset_name=dataset_name,
                request_params=self.request_params, timestamp_str=self.run_timestamp_str,
                output_dir=visualization_dir
            )
            metrics["prediction_plot_path"] = pred_vs_actual_path

            error_dist_path = plot_error_distribution(
                y_true=y_data, y_pred=predictions, acceptable_error=self.acceptable_error,
                target_metric=self.target_metric, model_name=self.model_name, dataset_name=dataset_name,
                request_params=self.request_params, timestamp_str=self.run_timestamp_str,
                output_dir=visualization_dir
            )
            metrics["error_distribution_plot_path"] = error_dist_path

        if dataset_name == "测试集" and predictions is not None:
            # 使用 run_specific_dir 构建评估数据的保存路径
            data_dir = os.path.join(self.run_specific_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            filename = f"{self._generate_artifact_base_filename()}_测试集数据评估结果.csv"
            filepath = os.path.join(data_dir, filename)

            results_df = X_data.copy()
            results_df[f'true_{self.target_metric}'] = y_data
            results_df[f'predict_{self.target_metric}'] = predictions
            results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            self.evaluation_results["artifacts"]["test_data_with_predictions_path"] = filepath

        return metrics, f"对 {dataset_name} 的评估和可视化完成。"

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, test_size: float) -> Generator[
        Dict[str, Any], None, bool]:
        """
        执行模型训练、评估、并保存过程产物。现在是一个生成器。
        """
        # 1. 数据划分
        yield {"type": "status_update",
               "payload": {"stage": self.current_stage, "status": "running", "detail": "划分数据集..."}}
        log_entry = {"step": "data_split", "test_size": test_size, "random_state": DEFAULT_RANDOM_STATE}
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                random_state=DEFAULT_RANDOM_STATE)
            log_entry["status"] = "success"
            log_entry["train_shape"] = X_train.shape
            log_entry["test_shape"] = X_test.shape
            self.training_log.append(log_entry)
            detail = f"数据集成功划分为训练集 ({X_train.shape[0]}条) 和测试集 ({X_test.shape[0]}条)。"
            yield {"type": "status_update",
                   "payload": {"stage": self.current_stage, "status": "running", "detail": detail}}
        except Exception as e:
            log_entry["status"] = "failed"
            log_entry["error"] = str(e)
            self.training_log.append(log_entry)
            error_msg = f"数据划分失败: {e}"
            yield {"type": "error",
                   "payload": {"stage": self.current_stage, "detail": error_msg + traceback.format_exc()}}
            return False

        # 2. 模型训练与超参数优化
        detail = f"开始训练模型: {self.model_name}..."
        if self.hpo_config:
            detail += f" (使用 {self.hpo_config.get('method', 'HPO')}进行超参数优化)"
        yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "running", "detail": detail}}
        train_log_entry = {"step": "model_training", "model_name": self.model_name, "hpo_config": self.hpo_config}
        try:
            param_grid_for_tuning = self.hyperparam_suggestions
            if not param_grid_for_tuning:
                yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "running",
                                                            "detail": "警告: 未提供超参数搜索空间，将使用默认参数训练。"}}
                self.hpo_config = None

            self.model_instance.train(X_train, y_train, hpo_config=self.hpo_config, param_grid=param_grid_for_tuning)
            self.trained_model_object = self.model_instance.model
            train_log_entry["status"] = "success"
            if self.model_instance.best_params_:
                train_log_entry["best_hyperparameters"] = self.model_instance.best_params_
                yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "running",
                                                            "detail": f"超参数优化完成，最佳参数: {self.model_instance.best_params_}"}}

            self.training_log.append(train_log_entry)

            # 3. 模型训练成功后，保存模型
            # 使用 run_specific_dir 构建模型的保存路径
            model_dir = os.path.join(self.run_specific_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            filename = f"{self._generate_artifact_base_filename()}.pkl"
            filepath = os.path.join(model_dir, filename)
            joblib.dump(self.trained_model_object, filepath)
            self.evaluation_results["artifacts"]["model_path"] = filepath
            self.training_log.append({"step": "model_saving", "status": "success", "path": filepath})
            yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "running",
                                                        "detail": f"模型训练完成并已保存。"}}

        except Exception as e:
            train_log_entry["status"] = "failed"
            train_log_entry["error"] = str(e)
            self.training_log.append(train_log_entry)
            error_msg = f"模型 {self.model_name} 训练失败: {e}"
            # 打印回溯以进行服务器端调试
            traceback.print_exc()
            yield {"type": "error",
                   "payload": {"stage": self.current_stage, "detail": error_msg + traceback.format_exc()}}
            return False

        # 3. 模型评估
        yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "running",
                                                    "detail": "开始在测试集和训练集上评估模型性能..."}}
        try:
            train_metrics, train_log = self._evaluate_and_visualize(X_train, y_train, "训练集")
            self.evaluation_results["train"] = train_metrics
            self.training_log.append(
                {"step": "model_evaluation_train", "status": "success", "details": train_log, "metrics": train_metrics})
            yield {"type": "status_update",
                   "payload": {"stage": self.current_stage, "status": "running", "detail": "训练集评估完成。"}}

            test_metrics, test_log = self._evaluate_and_visualize(X_test, y_test, "测试集")
            self.evaluation_results["test"] = test_metrics
            self.training_log.append(
                {"step": "model_evaluation_test", "status": "success", "details": test_log, "metrics": test_metrics})
            yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "running",
                                                        "detail": f"测试集评估完成。评估图表、数据已生成。"}}


        except Exception as e:
            self.training_log.append({"step": "model_evaluation", "status": "failed", "error": str(e)})
            error_msg = f"模型 {self.model_name} 评估失败: {e}"
            yield {"type": "error",
                   "payload": {"stage": self.current_stage, "detail": error_msg + traceback.format_exc()}}
            return False

        # 4. 特征重要性提取
        yield {"type": "status_update",
               "payload": {"stage": self.current_stage, "status": "running", "detail": "正在提取特征重要性..."}}
        fi_log_entry = {"step": "feature_importance_extraction", "model_name": self.model_name}
        try:
            feature_names = X_train.columns.tolist()
            self.feature_importances = self.model_instance.get_feature_importances(feature_names)
            if self.feature_importances is not None:
                fi_log_entry["status"] = "success"
                # This print is for server-side logging
                print(f"特征重要性 (前5): \n{self.feature_importances.head()}")
                yield {"type": "status_update",
                       "payload": {"stage": self.current_stage, "status": "running", "detail": "特征重要性提取成功。"}}
            else:
                fi_log_entry["status"] = "not_available"
                yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "running",
                                                            "detail": "当前模型不支持特征重要性提取。"}}
            self.training_log.append(fi_log_entry)
        except Exception as e:
            fi_log_entry["status"] = "failed"
            fi_log_entry["error"] = str(e)
            self.training_log.append(fi_log_entry)
            # This is not a critical failure, so we just yield a status update, not an error
            yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "warning",
                                                        "detail": f"提取特征重要性时发生错误: {e}"}}

        return True
