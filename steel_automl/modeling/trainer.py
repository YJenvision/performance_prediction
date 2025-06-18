import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, List, Optional
from config import DEFAULT_RANDOM_STATE
from steel_automl.modeling.algorithms.random_forest import RandomForestModel
from steel_automl.modeling.algorithms.xgboost_model import XGBoostModel

# 模型类映射表
MODEL_CLASSES = {
    "RandomForestRegressor": RandomForestModel,
    "XGBoostRegressor": XGBoostModel,
}


class ModelTrainer:
    def __init__(self, selected_model_name: str, model_info: Dict[str, Any], hpo_config: Dict[str, Any]):
        """
        初始化模型训练器。

        参数:
        - selected_model_name: 选择的模型名称。
        - model_info: 包含该模型超参数建议范围的字典。
        - hpo_config: 超参数优化(HPO)的配置。
        """
        if selected_model_name not in MODEL_CLASSES:
            raise ValueError(f"不支持的模型: {selected_model_name}。")

        self.model_name = selected_model_name
        self.model_class = MODEL_CLASSES[selected_model_name]
        self.hyperparam_suggestions = model_info.get("hyperparameter_suggestions", {})
        self.hpo_config = hpo_config

        # 初始超参数在这里可以不设置，因为调优流程会直接使用范围
        self.initial_hyperparams = {}

        self.model_instance = self.model_class(hyperparameters=self.initial_hyperparams)
        self.training_log: List[Dict[str, Any]] = []
        self.evaluation_results: Optional[Dict[str, float]] = None
        self.trained_model_object = None
        self.feature_importances: Optional[pd.Series] = None

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, test_size: float) -> None:
        """
        执行模型训练和评估。

        参数:
        - X: 特征DataFrame。
        - y: 目标Series。
        - test_size: 从LLM计划中动态获取的测试集划分比例。
        """
        print(f"\n--- 开始训练模型: {self.model_name} ---")
        log_entry = {"step": "data_split", "test_size": test_size, "random_state": DEFAULT_RANDOM_STATE}
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                random_state=DEFAULT_RANDOM_STATE)
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

        # 训练模型
        train_log_entry = {"step": "model_training", "model_name": self.model_name, "hpo_config": self.hpo_config}
        try:
            # param_grid 就是从LLM获取的建议，直接传递给模型
            param_grid_for_tuning = self.hyperparam_suggestions

            if not param_grid_for_tuning:
                print("警告: 未提供有效的超参数范围用于调优，将使用初始/默认参数训练。")
                self.hpo_config = None  # 关闭调优

            self.model_instance.train(
                X_train, y_train,
                hpo_config=self.hpo_config,
                param_grid=param_grid_for_tuning
            )

            self.trained_model_object = self.model_instance.model
            train_log_entry["status"] = "success"
            if self.model_instance.best_params_:
                train_log_entry["best_hyperparameters"] = self.model_instance.best_params_
            self.training_log.append(train_log_entry)
        except Exception as e:
            train_log_entry["status"] = "failed"
            train_log_entry["error"] = str(e)
            self.training_log.append(train_log_entry)
            print(f"模型 {self.model_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # 评估模型
        eval_log_entry = {"step": "model_evaluation", "model_name": self.model_name}
        try:
            self.evaluation_results = self.model_instance.evaluate(X_test, y_test)
            eval_log_entry["status"] = "success"
            eval_log_entry["metrics"] = self.evaluation_results
            self.training_log.append(eval_log_entry)
        except Exception as e:
            eval_log_entry["status"] = "failed"
            eval_log_entry["error"] = str(e)
            self.training_log.append(eval_log_entry)
            print(f"模型 {self.model_name} 评估失败: {e}")
            return

        # 获取特征重要性
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
