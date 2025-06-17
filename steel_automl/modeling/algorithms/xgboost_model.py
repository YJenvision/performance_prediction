from xgboost import XGBRegressor
import xgboost as xgb  # 新增导入
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from config import DEFAULT_RANDOM_STATE


class XGBoostModel:
    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """
        初始化XGBoost回归模型（GPU版本）。

        参数:
        - hyperparameters: 模型的超参数。如果为None，则使用默认参数或后续通过GridSearch确定。
        - device: 使用的设备，"cuda"表示GPU，"cuda:0"/"cuda:1"等表示指定GPU，"cpu"表示CPU
        """
        self.model_name = "XGBoostRegressor"
        self.model: Optional[XGBRegressor] = None
        self.hyperparameters = hyperparameters if hyperparameters else {}

        # 设置新版本XGBoost的GPU参数
        self.hyperparameters['tree_method'] = 'hist'  # 使用直方图算法
        self.hyperparameters['device'] = device  # 设置设备为GPU

        # 移除旧版本的参数（如果存在）
        self.hyperparameters.pop('gpu_id', None)
        self.hyperparameters.pop('predictor', None)

        # XGBoost的random_state参数名为seed（在新版本中可能直接使用random_state）
        if 'random_state' not in self.hyperparameters and 'seed' not in self.hyperparameters:
            self.hyperparameters['random_state'] = DEFAULT_RANDOM_STATE
        elif 'seed' in self.hyperparameters and 'random_state' not in self.hyperparameters:
            # 新版本更倾向于使用random_state
            self.hyperparameters['random_state'] = self.hyperparameters.pop('seed')

        self.best_params_: Optional[Dict[str, Any]] = None

        # 检查GPU是否可用
        self._check_gpu_availability()

    def _check_gpu_availability(self):
        """检查GPU是否可用"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode != 0:
                print("警告: 未检测到NVIDIA GPU，但将继续尝试使用GPU模式。")
                print("如果遇到错误，请确保已安装CUDA和正确版本的XGBoost GPU版本。")
            else:
                print(f"✓ 检测到NVIDIA GPU，将使用设备: {self.hyperparameters['device']}")
        except Exception as e:
            print(f"警告: 无法检查GPU状态: {e}")
            print("请确保已安装CUDA和正确版本的XGBoost GPU版本。")

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
        current_params = self.hyperparameters.copy()

        # 确保使用正确的参数名
        if 'seed' in current_params and 'random_state' not in current_params:
            current_params['random_state'] = current_params.pop('seed')
        if 'random_state' not in current_params:
            current_params['random_state'] = DEFAULT_RANDOM_STATE

        # 确保GPU参数始终存在
        current_params['tree_method'] = 'hist'
        if 'device' not in current_params:
            current_params['device'] = 'cuda'

        # 移除旧版本参数
        current_params.pop('gpu_id', None)
        current_params.pop('predictor', None)

        if tune_hyperparameters and param_grid:
            print(f"开始为 {self.model_name} 进行超参数调优（GPU模式: {current_params['device']}）...")

            temp_param_grid = param_grid.copy()

            # 移除旧版本参数名
            if 'seed' in temp_param_grid and 'random_state' not in temp_param_grid:
                temp_param_grid['random_state'] = temp_param_grid.pop('seed')

            # 确保不会覆盖GPU参数
            temp_param_grid.pop('tree_method', None)
            temp_param_grid.pop('device', None)
            temp_param_grid.pop('gpu_id', None)
            temp_param_grid.pop('predictor', None)

            xgbr = XGBRegressor(**current_params)

            # 过滤掉param_grid中不属于模型合法参数的键
            valid_param_grid = {k: v for k, v in temp_param_grid.items() if k in xgbr.get_params().keys()}
            if len(valid_param_grid) != len(temp_param_grid):
                print(
                    f"警告: 以下参数在param_grid中但不是 {self.model_name} 的有效参数，已被忽略: {set(temp_param_grid.keys()) - set(valid_param_grid.keys())}")

            if not valid_param_grid:
                print("警告: 没有有效的超参数网格用于调优，将使用默认/初始参数训练。")
                self.model = XGBRegressor(**current_params)
            else:
                grid_search = GridSearchCV(estimator=xgbr, param_grid=valid_param_grid, cv=3,
                                           scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
                try:
                    grid_search.fit(X_train, y_train)
                    self.model = grid_search.best_estimator_
                    self.best_params_ = grid_search.best_params_
                    # 添加GPU参数到best_params_
                    self.best_params_['tree_method'] = 'hist'
                    self.best_params_['device'] = current_params['device']
                    print(f"{self.model_name} 最佳超参数: {self.best_params_}")
                except Exception as e:
                    print(f"GPU训练失败: {e}")
                    print("请确保已安装支持GPU的XGBoost版本和CUDA。")
                    raise
        else:
            self.model = XGBRegressor(**current_params)
            try:
                self.model.fit(X_train, y_train)
                self.best_params_ = current_params
                print(f"{self.model_name} 训练完成（GPU模式: {current_params['device']}）。")
            except Exception as e:
                print(f"GPU训练失败: {e}")
                print("请确保已安装支持GPU的XGBoost版本和CUDA。")
                raise

    def predict(self, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        """进行预测（使用GPU）。"""
        if self.model:
            try:
                # 直接使用模型预测，设备设置已在模型初始化时完成
                return self.model.predict(X_test)
            except Exception as e:
                print(f"GPU预测失败: {e}")
                raise
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


if __name__ == '__main__':
    from sklearn.datasets import make_regression
    import xgboost

    # 显示XGBoost版本
    print(f"XGBoost版本: {xgboost.__version__}")

    # 检查GPU环境
    print("\n=== GPU环境检查 ===")
    try:
        import subprocess

        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 检测到NVIDIA GPU")
        else:
            print("✗ 未检测到NVIDIA GPU")
    except:
        print("✗ 无法运行nvidia-smi命令")

    print("\n开始测试GPU版本的XGBoost...\n")

    X, y = make_regression(n_samples=1000, n_features=20, random_state=DEFAULT_RANDOM_STATE)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_s = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_s, test_size=0.2, random_state=DEFAULT_RANDOM_STATE)

    # 1. 使用默认参数训练（GPU模式）
    print("\n--- 测试默认参数训练（GPU模式）---")
    xgb_model_default = XGBoostModel()
    xgb_model_default.train(X_train, y_train)
    xgb_model_default.evaluate(X_test, y_test)
    importances_default = xgb_model_default.get_feature_importances(X_train.columns.tolist())
    if importances_default is not None:
        print("特征重要性 (默认):")
        print(importances_default.head())

    # 2. 使用指定超参数训练（GPU模式）
    print("\n--- 测试指定超参数训练（GPU模式）---")
    custom_params_xgb = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5}
    xgb_model_custom = XGBoostModel(hyperparameters=custom_params_xgb)
    xgb_model_custom.train(X_train, y_train)
    xgb_model_custom.evaluate(X_test, y_test)

    # 3. 进行超参数调优训练（GPU模式）
    print("\n--- 测试超参数调优训练（GPU模式）---")
    param_grid_xgb = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    xgb_model_tuned = XGBoostModel()
    xgb_model_tuned.train(X_train, y_train, tune_hyperparameters=True, param_grid=param_grid_xgb)
    xgb_model_tuned.evaluate(X_test, y_test)
    if xgb_model_tuned.best_params_:
        print(f"调优后的最佳参数: {xgb_model_tuned.best_params_}")
