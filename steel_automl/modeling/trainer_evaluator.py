import pandas as pd
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from typing import Dict, Any, Tuple, List, Optional, Generator
import os
import joblib
from datetime import datetime
import traceback

from config import DEFAULT_RANDOM_STATE
from steel_automl.modeling.algorithms.random_forest import RandomForestModel
from steel_automl.modeling.algorithms.xgboost_model import XGBoostModel
from steel_automl.modeling.algorithms.lightgbm_model import LightGBMModel
from steel_automl.results.visualization import plot_prediction_vs_actual, plot_error_distribution, \
    plot_value_distribution

# 模型算法库中模型类映射表
MODEL_CLASSES = {
    "RandomForestRegressor": RandomForestModel,
    "XGBoostRegressor": XGBoostModel,
    "LightGBMRegressor": LightGBMModel,
}


class ModelTrainer:
    """
    模型训练与评估器。

    该类负责执行完整的模型训练流程，包括数据划分、超参数优化、模型训练、
    模型评估和特征重要性提取。它被设计为一个生成器，可以流式地返回
    每个子任务的状态和结果，以支持前端的渐进式呈现。
    """

    def __init__(self, selected_model_name: str, model_info: Dict[str, Any],
                 automl_plan: Dict[str, Any], run_specific_dir: str):
        """
        初始化模型训练器。
        """
        if selected_model_name not in MODEL_CLASSES:
            raise ValueError(f"不支持的模型: {selected_model_name}。")

        self.model_name = selected_model_name
        self.model_class = MODEL_CLASSES[selected_model_name]
        self.hyperparam_suggestions = model_info.get("hyperparameter_suggestions", {})

        # 从automl_plan中提取详细计划
        self.automl_plan = automl_plan
        self.model_plan = self.automl_plan.get("model_plan", {})
        self.data_split_plan = self.model_plan.get("data_split_plan", {"method": "sequential", "test_size": 0.2})
        self.cv_plan = self.model_plan.get("cv_plan", {"method": "time_series", "k_folds": 3})
        self.hpo_config = self.model_plan.get("hpo_config", {"method": "RandomizedSearchCV", "n_iter": 30})
        self.acceptable_error = self.model_plan.get("acceptable_error")

        self.request_params = self.automl_plan.get("user_request_details", {})
        self.target_metric = self.request_params.get("target_metric", "Unknown Target")
        self.run_timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S')
        self.current_stage = "模型训练与评估"
        self.run_specific_dir = run_specific_dir

        self.model_instance = self.model_class(hyperparameters={})
        self.training_log: List[Dict[str, Any]] = []
        self.evaluation_results: Dict[str, Any] = {"train": None, "test": None, "artifacts": {}}
        self.trained_model_object = None
        self.feature_importances: Optional[pd.Series] = None

    def _generate_artifact_base_filename(self) -> str:
        parts = [self.model_name]
        return "_".join(filter(None, parts)).replace(" ", "")

    def _evaluate_and_visualize(self, X_data: pd.DataFrame, y_data: pd.Series, dataset_name: str) -> Tuple[Dict, str]:
        """
        封装评估和可视化的通用函数。
        """
        metrics, predictions = self.model_instance.evaluate(X_data, y_data, self.acceptable_error)

        if predictions is not None and self.acceptable_error and metrics is not None:
            visualization_dir = os.path.join(self.run_specific_dir, "visualization")

            # 绘制 预测值 vs 真实值 散点图
            pred_vs_actual_path = plot_prediction_vs_actual(
                y_true=y_data, y_pred=predictions, acceptable_error=self.acceptable_error,
                target_metric=self.target_metric, model_name=self.model_name, dataset_name=dataset_name,
                request_params=self.request_params, timestamp_str=self.run_timestamp_str,
                output_dir=visualization_dir
            )
            metrics["prediction_plot_path"] = pred_vs_actual_path

            # 绘制误差分布图
            error_dist_path = plot_error_distribution(
                y_true=y_data, y_pred=predictions, acceptable_error=self.acceptable_error,
                target_metric=self.target_metric, model_name=self.model_name, dataset_name=dataset_name,
                request_params=self.request_params, timestamp_str=self.run_timestamp_str,
                output_dir=visualization_dir
            )
            metrics["error_distribution_plot_path"] = error_dist_path

            # 绘制真实值与预测值分布对比图
            value_dist_path = plot_value_distribution(
                y_true=y_data, y_pred=predictions,
                target_metric=self.target_metric, model_name=self.model_name, dataset_name=dataset_name,
                request_params=self.request_params, timestamp_str=self.run_timestamp_str,
                output_dir=visualization_dir
            )
            metrics["value_distribution_plot_path"] = value_dist_path

        if dataset_name == "测试集" and predictions is not None:
            data_dir = os.path.join(self.run_specific_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            filename = f"#7{self._generate_artifact_base_filename()}_测试集数据预测结果.csv"
            filepath = os.path.join(data_dir, filename)

            results_df = X_data.copy()
            results_df[f'true_{self.target_metric}'] = y_data
            results_df[f'predict_{self.target_metric}'] = predictions
            results_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            self.evaluation_results["artifacts"]["test_data_with_predictions_path"] = filepath

        return metrics, f"对 {dataset_name} 的评估和可视化完成。"

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> Generator[Dict[str, Any], None, bool]:
        """
        **核心修改方法**
        执行模型训练、评估、并保存过程产物。
        这是一个生成器，它将每个子任务的结果通过 `substage_result` 消息逐步返回。
        """
        # === 子任务 1: 数据划分 ===
        substage_title_split = "数据划分"
        split_method = self.data_split_plan.get("method", "sequential")
        test_size = self.data_split_plan.get("test_size", 0.2)

        yield {"type": "status_update",
               "payload": {"stage": self.current_stage, "status": "running",
                           "detail": f"正在执行数据划分 (方法: {split_method}, 测试集比例: {test_size})..."}}

        log_entry = {"step": substage_title_split, "plan": self.data_split_plan}

        try:
            if split_method == 'sequential':
                # 顺序切分 (假设数据已按时间排序)
                split_index = int(len(X) * (1 - test_size))
                X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
                y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
                log_entry["detail"] = "Data was split sequentially."
            else:  # 'random'
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=DEFAULT_RANDOM_STATE
                )
                log_entry["detail"] = "Data was split randomly."

            log_entry["status"] = "success"
            log_entry["train_shape"] = X_train.shape
            log_entry["test_shape"] = X_test.shape
            self.training_log.append(log_entry)

            result_data = {
                "detail": f"数据集成功划分为训练集 ({X_train.shape[0]}条) 和测试集 ({X_test.shape[0]}条)。方法: {split_method}",
                "train_shape": {"rows": X_train.shape[0], "columns": X_train.shape[1]},
                "test_shape": {"rows": X_test.shape[0], "columns": X_test.shape[1]}
            }
            yield {"type": "substage_result", "payload": {
                "stage": self.current_stage, "substage_title": substage_title_split,
                "data": result_data
            }}
        except Exception as e:
            log_entry["status"] = "failed"
            log_entry["error"] = str(e)
            self.training_log.append(log_entry)
            error_msg = f"数据划分失败: {e}"
            yield {"type": "error",
                   "payload": {"stage": self.current_stage, "detail": error_msg + traceback.format_exc()}}
            return False

        # === 子任务 2: 模型训练与超参数优化 ===
        substage_title_train = "模型训练"

        # --- 子任务 2.1: 准备交叉验证策略 ---
        cv_method = self.cv_plan.get("method", "time_series")
        k_folds = self.cv_plan.get("k_folds", 5)
        cv = None
        try:
            if cv_method == 'time_series':
                cv = TimeSeriesSplit(n_splits=k_folds)
                cv_detail_msg = f"使用时序{k_folds}折交叉验证。"
            else:  # 'random'
                cv = KFold(n_splits=k_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
                cv_detail_msg = f"使用随机{k_folds}折交叉验证。"

            self.training_log.append({"step": "cv_setup", "status": "success", "plan": self.cv_plan})
            yield {"type": "substage_result", "payload": {
                "stage": self.current_stage, "substage_title": "交叉验证策略",
                "data": {"detail": cv_detail_msg}
            }}
        except Exception as e:
            self.training_log.append({"step": "cv_setup", "status": "failed", "error": str(e)})
            error_msg = f"设置交叉验证策略失败: {e}"
            yield {"type": "error",
                   "payload": {"stage": self.current_stage, "detail": error_msg + traceback.format_exc()}}
            return False

        # --- 子任务 2.2: 开始训练 ---
        detail = f"开始训练模型: {self.model_name}"
        if self.hpo_config:
            detail += f" (使用 {self.hpo_config.get('method', 'HPO')} 进行超参数优化)"
        yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "running", "detail": detail}}

        train_log_entry = {"step": substage_title_train, "model_name": self.model_name, "hpo_config": self.hpo_config}
        try:
            param_grid_for_tuning = self.hyperparam_suggestions
            if not param_grid_for_tuning:
                yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "running",
                                                            "detail": "未提供超参数搜索空间，将使用默认参数训练。"}}
                self.hpo_config = None

            # 将准备好的cv对象放入hpo_config中
            if self.hpo_config:
                self.hpo_config['cv'] = cv
                if 'cv_folds' in self.hpo_config:
                    del self.hpo_config['cv_folds']

            self.model_instance.train(X_train, y_train, hpo_config=self.hpo_config, param_grid=param_grid_for_tuning)
            self.trained_model_object = self.model_instance.model
            train_log_entry["status"] = "success"

            # 训练完成后，立即发送一个子任务结果
            train_result_data = {"status": "成功", "model_name": self.model_name}
            if self.model_instance.best_params_:
                train_log_entry["best_hyperparameters"] = self.model_instance.best_params_
                train_result_data["best_hyperparameters"] = self.model_instance.best_params_

            yield {"type": "substage_result", "payload": {
                "stage": self.current_stage, "substage_title": "超参数优化结果" if self.hpo_config else "模型训练完成",
                "data": train_result_data
            }}
            self.training_log.append(train_log_entry)

            # --- 子任务 2.3: 模型保存 ---
            model_dir = os.path.join(self.run_specific_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            filename = f"{self._generate_artifact_base_filename()}.pkl"
            filepath = os.path.join(model_dir, filename)
            joblib.dump(self.trained_model_object, filepath)

            self.evaluation_results["artifacts"]["model_path"] = filepath
            self.training_log.append({"step": "model_saving", "status": "success", "path": filepath})

            yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "running",
                                                        "detail": "正在保存模型文件..."}}

        except Exception as e:
            train_log_entry["status"] = "failed"
            train_log_entry["error"] = str(e)
            self.training_log.append(train_log_entry)
            error_msg = f"模型 {self.model_name} 训练失败: {e}"
            traceback.print_exc()
            yield {"type": "error",
                   "payload": {"stage": self.current_stage, "detail": error_msg + traceback.format_exc()}}
            return False

        # === 子任务 3: 模型评估 ===
        yield {"type": "status_update", "payload": {"stage": self.current_stage, "status": "running",
                                                    "detail": "开始在测试集和训练集上评估模型性能..."}}
        try:
            # --- 子任务 3.1: 训练集评估 ---
            train_metrics, train_log = self._evaluate_and_visualize(X_train, y_train, "训练集")
            self.evaluation_results["train"] = train_metrics
            self.training_log.append(
                {"step": "model_evaluation_train", "status": "success", "details": train_log, "metrics": train_metrics})
            yield {"type": "substage_result", "payload": {
                "stage": self.current_stage, "substage_title": "模型评估 (训练集)",
                "data": train_metrics
            }}

            # --- 子任务 3.2: 测试集评估 ---
            test_metrics, test_log = self._evaluate_and_visualize(X_test, y_test, "测试集")
            self.evaluation_results["test"] = test_metrics
            self.training_log.append(
                {"step": "model_evaluation_test", "status": "success", "details": test_log, "metrics": test_metrics})
            yield {"type": "substage_result", "payload": {
                "stage": self.current_stage, "substage_title": "模型评估 (测试集)",
                "data": test_metrics
            }}

        except Exception as e:
            self.training_log.append({"step": "model_evaluation", "status": "failed", "error": str(e)})
            error_msg = f"模型 {self.model_name} 评估失败: {e}"
            yield {"type": "error",
                   "payload": {"stage": self.current_stage, "detail": error_msg + traceback.format_exc()}}
            return False

        # === 子任务 4: 特征重要性提取 ===
        substage_title_fi = "特征重要性"
        yield {"type": "status_update",
               "payload": {"stage": self.current_stage, "status": "running", "detail": "正在提取特征重要性..."}}
        fi_log_entry = {"step": substage_title_fi, "model_name": self.model_name}
        fi_result_data = {}
        try:
            feature_names = X_train.columns.tolist()
            self.feature_importances = self.model_instance.get_feature_importances(feature_names)

            if self.feature_importances is not None:
                fi_log_entry["status"] = "success"
                # 为了方便JSON序列化和前端展示，转换为字典
                fi_result_data = self.feature_importances.head(15).to_dict()
            else:
                fi_log_entry["status"] = "not_available"
                fi_result_data = {"message": "当前模型不支持特征重要性提取。"}

            self.training_log.append(fi_log_entry)
            yield {"type": "substage_result", "payload": {
                "stage": self.current_stage, "substage_title": substage_title_fi,
                "data": fi_result_data
            }}
        except Exception as e:
            fi_log_entry["status"] = "failed"
            fi_log_entry["error"] = str(e)
            self.training_log.append(fi_log_entry)
            error_detail = f"提取特征重要性时发生错误: {e}"
            # 这不是一个致命错误，因此只发送一个子任务结果，而不是全局错误
            yield {"type": "substage_result", "payload": {
                "stage": self.current_stage, "substage_title": substage_title_fi,
                "data": {"status": "失败", "error": error_detail}
            }}

        # 所有步骤成功完成
        return True
