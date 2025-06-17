import json

import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, List, Optional
from config import DEFAULT_RANDOM_STATE
from steel_automl.modeling.algorithms.random_forest import RandomForestModel
from steel_automl.modeling.algorithms.xgboost_model import XGBoostModel


# 如果算法模型库有更多模型，都在这里导入或通过映射表动态加载
MODEL_CLASSES = {
    "RandomForestRegressor": RandomForestModel,
    "XGBoostRegressor": XGBoostModel,
    # 后续添加其他模型类
}


class ModelTrainer:
    def __init__(self, selected_model_name: str, model_hyperparams_suggestions: Dict[str, Any]):
        """
        初始化模型训练器。

        参数:
        - selected_model_name: LLM选择的模型名称。
        - model_hyperparams_suggestions: LLM建议的该模型的超参数范围或值。
        """
        if selected_model_name not in MODEL_CLASSES:
            raise ValueError(f"不支持的模型: {selected_model_name}。当前系统支持的模型有: {list(MODEL_CLASSES.keys())}")

        self.model_name = selected_model_name
        self.model_class = MODEL_CLASSES[selected_model_name]
        self.hyperparam_suggestions = model_hyperparams_suggestions  # 这是GridSearchCV的param_grid

        # 从suggestions中提取固定参数 (如果LLM建议的是固定值而不是范围)
        # 简化：假设suggestions直接是param_grid，或者对于非调优情况是固定参数
        # 如果suggestions包含固定值，应将其传递给模型类的构造函数
        # 例如，如果suggestions是 {"n_estimators": 100} 而不是 {"n_estimators": [100,200]}
        # 这里我们假设，如果进行调优，suggestions就是param_grid；如果不调优，它就是固定参数。
        self.initial_hyperparams = {}
        if self.hyperparam_suggestions:
            # 尝试提取固定参数 (如果不是列表形式的范围)
            for key, value in self.hyperparam_suggestions.items():
                if not isinstance(value, list) and not isinstance(value, tuple):  # 不是范围/列表
                    self.initial_hyperparams[key] = value

        self.model_instance = self.model_class(hyperparameters=self.initial_hyperparams)
        self.training_log: List[Dict[str, Any]] = []
        self.evaluation_results: Optional[Dict[str, float]] = None
        self.trained_model_object = None  # 存储实际训练好的模型对象
        self.feature_importances: Optional[pd.Series] = None

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                           tune_hyperparameters: bool = True) -> None:
        """
        执行模型训练和评估。

        参数:
        - X: 特征DataFrame。
        - y: 目标Series。
        - test_size: 测试集划分比例。
        - tune_hyperparameters: 是否执行超参数调优 (基于LLM的建议)。
        """
        X.to_csv('X.csv', index=False)
        print(f"\n--- 开始训练模型: {self.model_name} ---")
        log_entry = {"step": "data_split", "test_size": test_size, "random_state": DEFAULT_RANDOM_STATE}
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                random_state=DEFAULT_RANDOM_STATE)
            log_entry["status"] = "success"
            log_entry["train_shape"] = X_train.shape
            log_entry["test_shape"] = X_test.shape
            self.training_log.append(log_entry)
            print(f"数据已划分为训练集 ({X_train.shape}) 和测试集 ({X_test.shape})。")
        except Exception as e:
            log_entry["status"] = "failed"
            log_entry["error"] = str(e)
            self.training_log.append(log_entry)
            print(f"数据划分失败: {e}")
            return

        # 训练模型
        train_log_entry = {"step": "model_training", "model_name": self.model_name}
        try:
            param_grid_for_tuning = None
            if tune_hyperparameters:
                # 确保hyperparam_suggestions是适合GridSearchCV的格式 (值是列表)
                param_grid_for_tuning = {}
                for key, value in self.hyperparam_suggestions.items():
                    if isinstance(value, list) or isinstance(value, tuple):
                        param_grid_for_tuning[key] = value
                    else:  # 如果是单个值，也包装成列表给GridSearchCV
                        param_grid_for_tuning[key] = [value]

                if not param_grid_for_tuning:  # 如果LLM没给范围，则不调优
                    print("警告: 未提供有效的超参数范围用于调优，将使用初始/默认参数训练。")
                    tune_hyperparameters = False  # 关闭调优

            self.model_instance.train(X_train, y_train,
                                      tune_hyperparameters=tune_hyperparameters,
                                      param_grid=param_grid_for_tuning if tune_hyperparameters else None)

            self.trained_model_object = self.model_instance.model  # 获取训练好的底层模型对象
            train_log_entry["status"] = "success"
            if self.model_instance.best_params_:
                train_log_entry["best_hyperparameters"] = self.model_instance.best_params_
            self.training_log.append(train_log_entry)
        except Exception as e:
            train_log_entry["status"] = "failed"
            train_log_entry["error"] = str(e)
            self.training_log.append(train_log_entry)
            print(f"模型 {self.model_name} 训练失败: {e}")
            return

        # 评估模型
        eval_log_entry = {"step": "model_evaluation", "model_name": self.model_name}
        try:
            self.evaluation_results = self.model_instance.evaluate(X_test, y_test)
            if self.evaluation_results:
                eval_log_entry["status"] = "success"
                eval_log_entry["metrics"] = self.evaluation_results
            else:
                eval_log_entry["status"] = "failed"
                eval_log_entry["error"] = "Evaluation returned no results."
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
            # 确保X_train.columns是列表
            feature_names = X_train.columns.tolist() if isinstance(X_train.columns, pd.Index) else X_train.columns
            self.feature_importances = self.model_instance.get_feature_importances(feature_names)
            if self.feature_importances is not None:
                fi_log_entry["status"] = "success"
                # fi_log_entry["importances"] = self.feature_importances.to_dict() # 可能太长，不直接存log
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
