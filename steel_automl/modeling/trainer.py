import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, List, Optional
import os
import joblib
from datetime import datetime

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
                 automl_plan: Dict[str, Any]):
        """
        初始化模型训练器。
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
            pred_vs_actual_path = plot_prediction_vs_actual(
                y_true=y_data, y_pred=predictions, acceptable_error=self.acceptable_error,
                target_metric=self.target_metric, model_name=self.model_name, dataset_name=dataset_name,
                request_params=self.request_params, timestamp_str=self.run_timestamp_str
            )
            metrics["prediction_plot_path"] = pred_vs_actual_path

            error_dist_path = plot_error_distribution(
                y_true=y_data, y_pred=predictions, acceptable_error=self.acceptable_error,
                target_metric=self.target_metric, model_name=self.model_name, dataset_name=dataset_name,
                request_params=self.request_params, timestamp_str=self.run_timestamp_str
            )
            metrics["error_distribution_plot_path"] = error_dist_path

        if dataset_name == "测试集" and predictions is not None:
            data_dir = "automl_runs\\data"
            os.makedirs(data_dir, exist_ok=True)
            filename = f"{self._generate_artifact_base_filename()}_测试集数据评估结果.csv"
            filepath = os.path.join(data_dir, filename)

            results_df = X_data.copy()
            results_df[f'true_{self.target_metric}'] = y_data
            results_df[f'predict_{self.target_metric}'] = predictions
            results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"测试集评估数据已保存: {filepath}")
            self.evaluation_results["artifacts"]["test_data_with_predictions_path"] = filepath

        return metrics, f"对 {dataset_name} 的评估和可视化完成。"

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, test_size: float) -> None:
        """
        执行模型训练、评估、并保存产物。
        """
        print(f"\n--- 开始训练模型: {self.model_name} ---")
        log_entry = {"step": "data_split", "test_size": test_size, "random_state": DEFAULT_RANDOM_STATE}
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=DEFAULT_RANDOM_STATE)
            log_entry["status"] = "success"
            log_entry["train_shape"] = X_train.shape
            log_entry["test_shape"] = X_test.shape
            self.training_log.append(log_entry)
            print(f"数据已划分为训练集 ({X_train.shape}) 和测试集 ({X_test.shape})，测试集比例: {test_size:.2f}。")
        except Exception as e:
            log_entry["status"] = "failed"
            log_entry["error"] = str(e)
            self.training_log.append(log_entry)
            print(f"数据划分失败: {e}")
            return

        train_log_entry = {"step": "model_training", "model_name": self.model_name, "hpo_config": self.hpo_config}
        try:
            param_grid_for_tuning = self.hyperparam_suggestions
            if not param_grid_for_tuning:
                print("警告: 未提供有效的超参数范围用于调优，将使用初始/默认参数训练。")
                self.hpo_config = None

            self.model_instance.train(X_train, y_train, hpo_config=self.hpo_config, param_grid=param_grid_for_tuning)
            self.trained_model_object = self.model_instance.model
            train_log_entry["status"] = "success"
            if self.model_instance.best_params_:
                train_log_entry["best_hyperparameters"] = self.model_instance.best_params_
            self.training_log.append(train_log_entry)

            # 模型训练成功后，保存模型
            model_dir = "automl_runs\\models"
            os.makedirs(model_dir, exist_ok=True)
            filename = f"{self._generate_artifact_base_filename()}.pkl"
            filepath = os.path.join(model_dir, filename)
            joblib.dump(self.trained_model_object, filepath)
            print(f"训练好的模型已保存: {filepath}")
            self.evaluation_results["artifacts"]["model_path"] = filepath
            self.training_log.append({"step": "model_saving", "status": "success", "path": filepath})

        except Exception as e:
            train_log_entry["status"] = "failed"
            train_log_entry["error"] = str(e)
            self.training_log.append(train_log_entry)
            print(f"模型 {self.model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return

        print("\n--- 正在评估模型性能 ---")
        try:
            test_metrics, test_log = self._evaluate_and_visualize(X_test, y_test, "测试集")
            self.evaluation_results["test"] = test_metrics
            self.training_log.append({"step": "model_evaluation_test", "status": "success", "details": test_log, "metrics": test_metrics})

            train_metrics, train_log = self._evaluate_and_visualize(X_train, y_train, "训练集")
            self.evaluation_results["train"] = train_metrics
            self.training_log.append({"step": "model_evaluation_train", "status": "success", "details": train_log, "metrics": train_metrics})
        except Exception as e:
            self.training_log.append({"step": "model_evaluation", "status": "failed", "error": str(e)})
            print(f"模型 {self.model_name} 评估失败: {e}")
            return

        fi_log_entry = {"step": "feature_importance_extraction", "model_name": self.model_name}
        try:
            feature_names = X_train.columns.tolist()
            self.feature_importances = self.model_instance.get_feature_importances(feature_names)
            if self.feature_importances is not None:
                fi_log_entry["status"] = "success"
                print(f"特征重要性 (前5): \n{self.feature_importances.head()}")
            else:
                fi_log_entry["status"] = "not_available"
            self.training_log.append(fi_log_entry)
        except Exception as e:
            fi_log_entry["status"] = "failed"
            fi_log_entry["error"] = str(e)
            self.training_log.append(fi_log_entry)
            print(f"获取模型 {self.model_name} 特征重要性失败: {e}")

        print(f"--- 模型 {self.model_name} 训练与评估完成 ---")
