from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from config import DEFAULT_RANDOM_STATE


class RandomForestModel:
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        """
        初始化随机森林回归模型。

        参数:
        - hyperparameters: 模型的超参数。如果为None，则使用默认参数或后续通过GridSearch确定。
        """
        self.model_name = "RandomForestRegressor"
        self.model: Optional[RandomForestRegressor] = None
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.best_params_: Optional[Dict[str, Any]] = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, tune_hyperparameters: bool = False,
              param_grid: Optional[Dict[str, Any]] = None) -> None:
        """
        训练模型。

        参数:
        - X_train: 训练特征。
        - y_train: 训练目标。
        - tune_hyperparameters: 是否进行超参数调优。
        - param_grid: 超参数调优的搜索网格 (如果tune_hyperparameters为True)。
        """
        if tune_hyperparameters and param_grid:
            print(f"开始为 {self.model_name} 进行超参数调优...")
            # 确保param_grid中的参数名与RandomForestRegressor的参数名一致
            rf = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, **self.hyperparameters)  # 可以传入初始参数

            # 过滤掉param_grid中不属于模型合法参数的键
            valid_param_grid = {k: v for k, v in param_grid.items() if k in rf.get_params().keys()}
            if len(valid_param_grid) != len(param_grid):
                print(
                    f"警告: 以下参数在param_grid中但不是 {self.model_name} 的有效参数，已被忽略: {set(param_grid.keys()) - set(valid_param_grid.keys())}")

            if not valid_param_grid:
                print("警告: 没有有效的超参数网格用于调优，将使用默认/初始参数训练。")
                self.model = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, **self.hyperparameters)
            else:
                grid_search = GridSearchCV(estimator=rf, param_grid=valid_param_grid, cv=3,
                                           scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_
                self.best_params_ = grid_search.best_params_
                print(f"{self.model_name} 最佳超参数: {self.best_params_}")
        else:
            self.model = RandomForestRegressor(random_state=DEFAULT_RANDOM_STATE, **self.hyperparameters)
            self.model.fit(X_train, y_train)
            self.best_params_ = self.hyperparameters  # 如果不调优，最佳参数就是初始参数

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
