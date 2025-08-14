import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scipy.stats import randint, uniform
from config import DEFAULT_RANDOM_STATE
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

try:
    import torch

    # IS_CUDA_AVAILABLE = torch.cuda.is_available()
    # 使用GPU训练和预测的情况下，模型在GPU上，数据在CPU上，导致数据交换成本过高，导致效率变慢，目前Windows解决失败，Linux计划使用cudf解决。

    # 均使用CPU进行模型的训练和预测。
    IS_CUDA_AVAILABLE = False

except ImportError:
    IS_CUDA_AVAILABLE = False
    print("警告: 未安装PyTorch。无法准确检测CUDA GPU。将依赖XGBoost的内部检测。")


class XGBoostModel:
    """
    一个封装了XGBoost回归模型的类，支持超参数优化和GPU加速。
    """

    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None):
        """
        初始化XGBoost回归模型。

        会自动检测GPU可用性，并相应地设置 'device'。

        参数:
        - hyperparameters: 模型的固定超参数。
        """
        self.model_name = "XGBoostRegressor"
        self.model: Optional[Any] = None  # 可以是XGBRegressor实例或SearchCV对象
        self.hyperparameters = hyperparameters if hyperparameters else {}
        self.best_params_: Optional[Dict[str, Any]] = None
        self.booster = None  # 存储原始booster对象用于GPU预测

        # 更新动态设备选择逻辑以兼容 XGBoost >= 2.0.0
        if IS_CUDA_AVAILABLE:
            self.device_params = {'device': 'cuda', 'tree_method': 'hist'}
            # print("信息: 检测到CUDA GPU。XGBoost将使用 device='cuda' 进行训练。")
        else:
            # 明确指定使用 CPU
            self.device_params = {'device': 'cpu', 'tree_method': 'hist'}
            # print("信息: 未检测到CUDA GPU或PyTorch未安装。XGBoost将使用CPU。")

        # 将设备参数与用户提供的超参数合并
        # 移除任何已弃用的参数
        deprecated_params = ['predictor', 'gpu_id']
        for param in deprecated_params:
            if param in self.hyperparameters:
                print(f"警告: 移除已弃用的参数 '{param}'")
                del self.hyperparameters[param]

        # 如果存在旧的gpu_hist参数，更新为新的格式
        if self.hyperparameters.get('tree_method') == 'gpu_hist':
            self.hyperparameters['tree_method'] = 'hist'
            if IS_CUDA_AVAILABLE:
                self.hyperparameters['device'] = 'cuda'

        self.hyperparameters.update(self.device_params)

    def _prepare_bayesian_search_space(self, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """将LLM建议的参数范围转换为BayesSearchCV兼容的搜索空间。"""
        search_space = {}
        for param, values in param_grid.items():
            if not isinstance(values, list) or len(values) < 2:
                search_space[param] = Categorical([values])
                continue

            lower, upper = values[0], values[1]
            # 检查是否有 'log' 标志
            param_type = 'log-uniform' if len(values) > 2 and values[2] == 'log' else 'uniform'

            if isinstance(lower, int) and isinstance(upper, int):
                search_space[param] = Integer(lower, upper, prior=param_type)
            elif isinstance(lower, float) or isinstance(upper, float):
                search_space[param] = Real(float(lower), float(upper), prior=param_type)
            else:  # 处理字符串或其他分类类型
                search_space[param] = Categorical(values)
        return search_space

    def _prepare_random_search_distributions(self, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """将LLM建议的范围转换为scipy.stats分布，用于RandomizedSearchCV。"""
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

        # 清理param_grid中的已弃用参数
        if param_grid:
            deprecated_params = ['predictor', 'gpu_id']
            for param in deprecated_params:
                if param in param_grid:
                    del param_grid[param]

            # 更新gpu_hist到新格式
            if 'tree_method' in param_grid and 'gpu_hist' in param_grid['tree_method']:
                param_grid['tree_method'] = ['hist']
                if IS_CUDA_AVAILABLE and 'device' not in param_grid:
                    param_grid['device'] = ['cuda']

        base_model = xgb.XGBRegressor(random_state=DEFAULT_RANDOM_STATE, **self.hyperparameters)

        if hpo_method:
            print(f"开始为 {self.model_name} 进行超参数调优 (方法: {hpo_method})...")

            valid_param_grid = {k: v for k, v in param_grid.items() if k in base_model.get_params().keys()}
            if not valid_param_grid:
                print("警告: 没有有效的超参数网格用于调优，将使用默认/初始参数训练。")
                should_tune = False
            else:
                search_cv = None
                # [修改] 优先使用传入的cv对象，如果没有则回退到固定的折数
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

        # 训练后获取booster对象以供GPU预测使用
        if self.model and hasattr(self.model, 'get_booster'):
            self.booster = self.model.get_booster()

        print(f"{self.model_name} 训练完成。")

    def predict(self, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        """
        使用训练好的模型进行预测。
        修复：确保GPU模型的预测不会产生设备不匹配警告。
        """
        if not self.model:
            print("错误: 模型尚未训练，无法进行预测。")
            return None

        is_gpu_model = self.model.get_params().get('device') == 'cuda'

        # 对于GPU模型，使用booster的inplace_predict以避免设备不匹配警告
        if is_gpu_model and self.booster:
            try:
                # 使用inplace_predict避免设备不匹配
                # 将DataFrame转换为numpy数组进行预测
                X_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
                predictions = self.booster.inplace_predict(X_array)

                # 如果结果是cupy数组，转换回numpy
                if hasattr(predictions, 'get'):
                    predictions = predictions.get()

                return predictions
            except Exception as e:
                # 如果inplace_predict失败，回退到标准预测
                print(f"GPU inplace_predict失败，回退到标准预测: {e}")
                predictions = self.model.predict(X_test)
        else:
            # 对于CPU模型，直接预测
            predictions = self.model.predict(X_test)

        # 处理cupy数组
        if hasattr(predictions, 'get'):
            return predictions.get()

        return predictions

    def evaluate(self, X_data: pd.DataFrame, y_data: pd.Series, acceptable_error: Optional[Dict[str, Any]] = None) -> \
            Tuple[Optional[Dict[str, float]], Optional[np.ndarray]]:
        """
        评估模型性能，并返回指标和预测结果。

        参数:
        - X_data: 特征数据。
        - y_data: 真实标签数据。
        - acceptable_error: 包含误差类型和值的字典。

        返回:
        - 一个包含评估指标的字典。
        - 预测结果的Numpy数组。
        """
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
                # 避免y_true为0时的除法错误
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Create a boolean mask for y_data != 0
                    non_zero_mask = y_data != 0
                    # Initialize proportion array with False
                    is_within_bounds = np.zeros_like(y_data, dtype=bool)
                    # Calculate proportion only where y_data is not zero
                    is_within_bounds[non_zero_mask] = \
                        (abs_error[non_zero_mask] / y_data[non_zero_mask]) <= (error_value / 100)
                    # For cases where y_data is zero, check against absolute error
                    is_within_bounds[~non_zero_mask] = abs_error[~non_zero_mask] <= 0  # Or some small tolerance
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
