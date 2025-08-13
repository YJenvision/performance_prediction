# @Time    : 2025/8/4 15:08
# @Author  : ZhangJingLiang
# @Email   : jinglianglink@qq.com
# @Project : performance_prediction_agent_stream


import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scipy.stats import randint, uniform
from config import DEFAULT_RANDOM_STATE

import os

os.environ['LOKY_MAX_CPU_COUNT'] = '16'  # 设置临时的CPU核心数


class LightGBMModel:
    """
    一个封装了LightGBM回归模型的类，支持超参数优化。
    """

    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        """
        初始化LightGBM回归模型。
        """
        self.model_name = "LightGBMRegressor"
        self.model: Optional[Any] = None  # 可以是LGBMRegressor实例或SearchCV对象
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.best_params_: Optional[Dict[str, Any]] = None
        # LightGBM默认在CPU上运行，可以通过 'device': 'gpu' 来启用GPU
        self.hyperparameters.setdefault('device', 'cpu')
        self.hyperparameters.setdefault('verbose', -1)  # 减少不必要的日志输出

    def _prepare_bayesian_search_space(self, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """将建议的参数范围转换为BayesSearchCV兼容的搜索空间。"""
        search_space = {}
        for param, values in param_grid.items():
            if not isinstance(values, list) or len(values) < 2:
                search_space[param] = Categorical([values])
                continue

            lower, upper = values[0], values[1]
            param_type = 'log-uniform' if len(values) > 2 and values[2] == 'log' else 'uniform'

            if isinstance(lower, int) and isinstance(upper, int):
                search_space[param] = Integer(lower, upper, prior=param_type)
            elif isinstance(lower, float) or isinstance(upper, float):
                search_space[param] = Real(float(lower), float(upper), prior=param_type)
            else:
                search_space[param] = Categorical(values)
        return search_space

    def _prepare_random_search_distributions(self, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """将建议的范围转换为scipy.stats分布，用于RandomizedSearchCV。"""
        distributions = {}
        for param, values in param_grid.items():
            if not isinstance(values, list) or len(values) < 2:
                distributions[param] = values
                continue

            lower, upper = values[0], values[1]

            if isinstance(lower, int) and isinstance(upper, int):
                if lower >= upper:
                    distributions[param] = [lower]
                else:
                    distributions[param] = randint(lower, upper + 1)
            elif isinstance(lower, float) or isinstance(upper, float):
                if lower >= upper:
                    distributions[param] = [lower]
                else:
                    distributions[param] = uniform(loc=float(lower), scale=float(upper) - float(lower))
            else:
                distributions[param] = values
        return distributions

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              hpo_config: Optional[Dict[str, Any]] = None,
              param_grid: Optional[Dict[str, Any]] = None) -> None:
        """
        训练模型，支持多种超参数优化方法。
        """
        should_tune = hpo_config and param_grid
        hpo_method = hpo_config.get("method") if should_tune else None

        base_model = lgb.LGBMRegressor(random_state=DEFAULT_RANDOM_STATE, **self.hyperparameters)

        if hpo_method:
            print(f"开始为 {self.model_name} 进行超参数调优 (方法: {hpo_method})...")

            valid_param_grid = {k: v for k, v in param_grid.items() if k in base_model.get_params().keys()}
            if not valid_param_grid:
                print("警告: 没有有效的超参数网格用于调优，将使用默认/初始参数训练。")
                should_tune = False
            else:
                search_cv = None
                cv_strategy = hpo_config.get("cv", 3)
                scoring = hpo_config.get("scoring_metric", 'neg_mean_squared_error')
                n_iter = hpo_config.get("n_iter", 30)

                if hpo_method == "GridSearchCV":
                    search_cv = GridSearchCV(estimator=base_model, param_grid=valid_param_grid, cv=cv_strategy,
                                             scoring=scoring, n_jobs=-1, verbose=1)
                elif hpo_method == "RandomizedSearchCV":
                    param_distributions = self._prepare_random_search_distributions(valid_param_grid)
                    search_cv = RandomizedSearchCV(estimator=base_model, param_distributions=param_distributions,
                                                   n_iter=n_iter, cv=cv_strategy, scoring=scoring, n_jobs=-1, verbose=1,
                                                   random_state=DEFAULT_RANDOM_STATE)
                elif hpo_method == "BayesianOptimization":
                    search_space = self._prepare_bayesian_search_space(valid_param_grid)
                    search_cv = BayesSearchCV(estimator=base_model, search_spaces=search_space, n_iter=n_iter,
                                              cv=cv_strategy, scoring=scoring, n_jobs=-1, verbose=1,
                                              random_state=DEFAULT_RANDOM_STATE)

                if search_cv:
                    search_cv.fit(X_train, y_train)
                    self.model = search_cv.best_estimator_
                    self.best_params_ = search_cv.best_params_
                    if isinstance(self.best_params_, dict):
                        self.best_params_ = dict(self.best_params_)
                    print(f"{self.model_name} 最佳超参数: {self.best_params_}")
                else:
                    should_tune = False

        if not should_tune:
            print(f"使用固定参数为 {self.model_name} 进行训练...")
            self.model = base_model
            self.model.fit(X_train, y_train)
            self.best_params_ = self.hyperparameters

        print(f"{self.model_name} 训练完成。")

    def predict(self, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        """使用训练好的模型进行预测。"""
        if not self.model:
            print("错误: 模型尚未训练，无法进行预测。")
            return None
        return self.model.predict(X_test)

    def evaluate(self, X_data: pd.DataFrame, y_data: pd.Series, acceptable_error: Optional[Dict[str, Any]] = None) -> \
            Tuple[Optional[Dict[str, float]], Optional[np.ndarray]]:
        """评估模型性能，并返回指标和预测结果。"""
        predictions = self.predict(X_data)
        if predictions is None:
            return None, None

        mse = mean_squared_error(y_data, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_data, predictions)
        mae = mean_absolute_error(y_data, predictions)

        metrics = {"MSE": mse, "RMSE": rmse, "R2": r2, "MAE": mae}

        if acceptable_error:
            error_type = acceptable_error.get("type", "percentage")
            error_value = acceptable_error.get("value", 5)
            abs_error = np.abs(predictions - y_data)

            if error_type == 'percentage':
                with np.errstate(divide='ignore', invalid='ignore'):
                    non_zero_mask = y_data != 0
                    is_within_bounds = np.zeros_like(y_data, dtype=bool)
                    is_within_bounds[non_zero_mask] = \
                        (abs_error[non_zero_mask] / y_data[non_zero_mask]) <= (error_value / 100)
                    is_within_bounds[~non_zero_mask] = abs_error[~non_zero_mask] <= 0
            else:  # value
                is_within_bounds = abs_error <= error_value

            proportion = np.mean(is_within_bounds)
            metrics["proportion_in_acceptable_range"] = proportion

        return metrics, predictions

    def get_feature_importances(self, feature_names: List[str]) -> Optional[pd.Series]:
        """获取并返回模型的特征重要性。"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return pd.Series(importances, index=feature_names).sort_values(ascending=False)
        else:
            print(f"错误: 模型 {self.model_name} 未训练或不支持特征重要性。")
            return None
