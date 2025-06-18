from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scipy.stats import randint, uniform
from config import DEFAULT_RANDOM_STATE


class RandomForestModel:
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        """
        初始化随机森林回归模型。

        参数:
        - hyperparameters: 模型的固定超参数。
        """
        self.model_name = "RandomForestRegressor"
        self.model: Optional[Any] = None  # Can be Regressor or a SearchCV object
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.best_params_: Optional[Dict[str, Any]] = None

    def _prepare_bayesian_search_space(self, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """将建议的参数范围转换为BayesSearchCV兼容的搜索空间。"""
        search_space = {}
        for param, values in param_grid.items():
            if not isinstance(values, list) or len(values) < 2:
                search_space[param] = Categorical([values])
                continue

            lower, upper = values[0], values[1]
            param_type = 'log' if len(values) > 2 and values[2] == 'log' else 'uniform'

            if isinstance(lower, int) and isinstance(upper, int):
                search_space[param] = Integer(lower, upper, prior=param_type)
            elif isinstance(lower, float) or isinstance(upper, float):
                search_space[param] = Real(float(lower), float(upper), prior=param_type)
            else:
                search_space[param] = Categorical(values)
        return search_space

    def _prepare_random_search_distributions(self, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """
        将建议的范围转换为scipy.stats分布，用于RandomizedSearchCV。
        """
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
                    # randint 从 [low, high) 中采样，所以 high 需要是 upper + 1 才能包含 upper
                    distributions[param] = randint(lower, upper + 1)
            elif isinstance(lower, float) or isinstance(upper, float):
                if lower >= upper:
                    distributions[param] = [lower]
                else:
                    # uniform 从 [loc, loc + scale] 中采样
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

        if hpo_method:
            print(f"开始为 {self.model_name} 进行超参数调优 (方法: {hpo_method})...")
            rf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, **self.hyperparameters)

            valid_param_grid = {k: v for k, v in param_grid.items() if k in rf.get_params().keys()}
            if not valid_param_grid:
                print("警告: 没有有效的超参数网格用于调优，将使用默认/初始参数训练。")
                should_tune = False
            else:
                search_cv = None
                cv_folds = hpo_config.get("cv_folds", 3)
                scoring = hpo_config.get("scoring_metric", 'neg_mean_squared_error')
                n_iter = hpo_config.get("n_iter", 30)

                if hpo_method == "GridSearchCV":
                    search_cv = GridSearchCV(
                        estimator=rf, param_grid=valid_param_grid, cv=cv_folds,
                        scoring=scoring, n_jobs=-1, verbose=1
                    )
                elif hpo_method == "RandomizedSearchCV":
                    # **[FIX]** 使用新的辅助函数来创建正确的分布
                    param_distributions = self._prepare_random_search_distributions(valid_param_grid)
                    print(f"RandomizedSearchCV开始，为随机搜索准备的分布: {param_distributions}")
                    search_cv = RandomizedSearchCV(
                        estimator=rf, param_distributions=param_distributions, n_iter=n_iter,
                        cv=cv_folds, scoring=scoring, n_jobs=-1, verbose=1,
                        random_state=DEFAULT_RANDOM_STATE
                    )
                elif hpo_method == "BayesianOptimization":
                    print(f"BayesianOptimization开始，为贝叶斯优化准备搜索空间: {valid_param_grid}")
                    search_space = self._prepare_bayesian_search_space(valid_param_grid)
                    search_cv = BayesSearchCV(
                        estimator=rf, search_spaces=search_space, n_iter=n_iter,
                        cv=cv_folds, scoring=scoring, n_jobs=-1, verbose=1,
                        random_state=DEFAULT_RANDOM_STATE
                    )

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
            self.model = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, **self.hyperparameters)
            self.model.fit(X_train, y_train)
            self.best_params_ = self.hyperparameters

        print(f"{self.model_name} 训练完成。")

    def predict(self, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        """进行预测。"""
        if self.model:
            return self.model.predict(X_test)
        else:
            print("错误: 模型尚未训练。")
            return None

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Optional[Dict[str, float]]:
        """评估模型性能。"""
        predictions = self.predict(X_test)
        if predictions is not None:
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            print(f"{self.model_name} 评估结果: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
            return {"mse": mse, "rmse": rmse, "r2": r2}
        return None

    def get_feature_importances(self, feature_names: List[str]) -> Optional[pd.Series]:
        """获取特征重要性。"""
        if self.model and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            return pd.Series(importances, index=feature_names).sort_values(ascending=False)
        else:
            print(f"错误: 模型 {self.model_name} 未训练或不支持特征重要性。")
            return None
