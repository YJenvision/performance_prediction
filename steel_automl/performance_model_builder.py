# @Time    : 2025/5/16
# @Author  : ZhangJingLiang
# @Email   : jinglianglink@qq.com
# @Project : performance_prediction_agent

import time
import os
import pickle
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from config import DB_CONFIG
from steel_automl.data_acquisition.data_loader import DataLoader
from steel_automl.data_preprocessing.preprocessor import DataPreprocessor
from steel_automl.feature_engineering.feature_generator import FeatureGenerator
from steel_automl.model_selection.selector import ModelSelector
from steel_automl.modeling.trainer import ModelTrainer
from steel_automl.pipeline.pipeline_builder import PipelineBuilder
from steel_automl.results.result_handler import ResultHandler


def _generate_filename_prefix(request_params: Dict[str, Any], timestamp_str: str) -> str:
    """
    根据请求参数生成标准化的文件名头部。
    命名标准: 目标性能_数据时间范围_牌号_机组_出钢记号_钢种_时间
    """

    def format_param(param_value: Any) -> str:
        """用于格式化列表或将单个值格式化为字符串的帮助函数。"""
        if param_value is None:
            return ""
        if isinstance(param_value, list):
            return "-".join(map(str, param_value))
        return str(param_value)

    parts = [
        format_param(request_params.get("target_metric")),
        format_param(request_params.get("time_range")),
        format_param(request_params.get("sg_sign")),
        format_param(request_params.get("product_unit_no")),
        format_param(request_params.get("st_no")),
        format_param(request_params.get("steel_grade")),
        timestamp_str,
    ]
    # 过滤掉空字符串部分并用下划线连接
    prefix = "_".join(filter(None, parts))
    return prefix.replace(" ", "").replace("/", "-")


def _generate_data_filename(filename_prefix: str, data_type_str: str) -> str:
    """
    根据文件名头部和数据类型生成标准化的数据文件名。
    """
    return f"{filename_prefix}_{data_type_str}.csv"


def _save_dataframe(df: pd.DataFrame, data_type_name: str, filename_prefix: str, run_dir: str) -> str:
    """
    保存DataFrame到指定目录并返回完整路径。
    """
    if df is None or df.empty:
        return ""
    data_dir = os.path.join(run_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    filename = _generate_data_filename(filename_prefix, data_type_name)
    filepath = os.path.join(data_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"数据已保存: {filepath}")
    return filepath


def _save_fitted_objects(objects: Dict[str, Any], filename_prefix: str, run_dir: str, object_type: str) -> str:
    """
    使用pickle保存拟合的对象（如预处理器、特征生成器）。
    这些对象对于在部署后转换新数据至关重要。

    参数:
    - objects: 包含已拟合转换器的字典。
    - filename_prefix: 标准化的文件名头部。
    - run_dir: 本次运行的根目录。
    - object_type: 对象的类型描述 (e.g., 'preprocessors', 'feature_generators')。

    返回:
    - 保存文件的完整路径，如果失败则返回空字符串。
    """
    if not objects:
        return ""

    run_dir = os.path.join(run_dir, object_type)

    # 确保主运行目录存在
    os.makedirs(run_dir, exist_ok=True)

    # 定义文件名，例如: ..._preprocessors.pkl
    filename = f"{filename_prefix}_{object_type}.pkl"
    filepath = os.path.join(run_dir, filename)

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(objects, f)
        print(f"拟合的对象已保存: {filepath}")
        return filepath
    except Exception as e:
        print(f"错误: 保存拟合的对象到 {filepath} 失败: {e}")
        return ""


def performanceModelBuilder(request_params: Dict[str, str]) -> Dict[str, Any]:
    """
    性能预报模型AutoML主流程。

    参数:
    - request_params: 包含用户请求的字典。
    返回:
    - 一个包含模型信息与评估结果的字典。
    """
    start_time_total = time.time()
    run_timestamp = datetime.now()
    run_timestamp_str = run_timestamp.strftime('%Y%m%d%H%M%S')
    run_dir = "automl_runs"
    os.makedirs(run_dir, exist_ok=True)

    # 生成本次运行所有产物的标准化文件名头部
    filename_prefix = _generate_filename_prefix(request_params, run_timestamp_str)

    user_request = request_params.get("user_request", "用户具体请求描述为空。")
    target_metric = request_params.get("target_metric")

    if not target_metric:
        return {"status": "failed", "error": "请求参数不完整 (目标性能字段target_metric必须提供)。"}

    pipeline_recorder = PipelineBuilder(user_request_details=request_params)
    final_result_package = {"status": "pending"}

    try:
        # 1. 数据获取
        print("\n模块1: 数据获取...")
        data_loader = DataLoader(db_config=DB_CONFIG)
        raw_data, sql_query = data_loader.fetch_data(
            request_params.get("sg_sign"),
            target_metric,
            request_params.get("time_range"),
            request_params.get("product_unit_no"),
            request_params.get("st_no"),
            request_params.get("steel_grade")
        )

        raw_data_path = _save_dataframe(raw_data, "原始数据集", filename_prefix, run_dir)

        if raw_data is None or raw_data.empty:
            pipeline_recorder.add_stage("data_acquisition", "failed",
                                        {"error": "未能获取到数据或数据为空", "sql_query": sql_query},
                                        artifacts={"raw_data_path": raw_data_path})
            pipeline_recorder.set_final_status("failed")
            return ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()

        pipeline_recorder.add_stage("data_acquisition", "success",
                                    {"rows_fetched": len(raw_data), "columns": list(raw_data.columns)},
                                    artifacts={"sql_query": sql_query, "raw_data_path": raw_data_path})

        # 验证1: 目标列是否存在
        if target_metric not in raw_data.columns:
            pipeline_recorder.add_stage("data_validation", "failed",
                                        {"error": f"目标列 '{target_metric}' 在获取的数据中不存在，无法进行训练。"})
            pipeline_recorder.set_final_status("failed")
            final_result_package = ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()
            return final_result_package

        # 验证2: 目标列全为空值
        raw_data[target_metric] = raw_data[target_metric].replace(
            [r'^\s*$', r'\(?null\)?', 'null', 'nan'],
            np.nan,
            regex=True
        )
        if raw_data[target_metric].isnull().all():
            pipeline_recorder.add_stage("data_validation", "failed",
                                        {"error": f"目标列 '{target_metric}' 全部为空值，无法进行训练。"})
            pipeline_recorder.set_final_status("failed")
            final_result_package = ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()
            return final_result_package

        # 验证3: 目标列为常量列
        if raw_data[target_metric].nunique() == 1:
            pipeline_recorder.add_stage("data_validation", "failed",
                                        {"error": f"目标列 '{target_metric}' 是常量列（所有值相同），无法进行训练。"})
            pipeline_recorder.set_final_status("failed")
            final_result_package = ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()
            return final_result_package

        # 2. 数据预处理
        print("\n模块2: 数据预处理...")
        preprocessor = DataPreprocessor(user_request=user_request, target_metric=target_metric)
        df_processed, preprocessing_steps, fitted_preproc_objects = preprocessor.preprocess_data(raw_data.copy())

        preprocessed_data_path = _save_dataframe(df_processed, "经过清洗后的数据集", filename_prefix, run_dir)

        # 保存拟合的预处理对象，并将路径添加到产物记录中
        preproc_artifacts = {
            "fitted_objects_keys": list(fitted_preproc_objects.keys()),
            "preprocessed_data_path": preprocessed_data_path
        }
        saved_preproc_path = _save_fitted_objects(fitted_preproc_objects, filename_prefix, run_dir, "preprocessors")
        if saved_preproc_path:
            preproc_artifacts["fitted_preprocessors_path"] = saved_preproc_path

        # 检查预处理步骤中是否有严重错误导致无法继续
        critical_preprocessing_failed = False
        if df_processed.empty and not raw_data.empty:
            critical_preprocessing_failed = True
            print("错误: 数据预处理后DataFrame为空。")
        # 检查preprocessing_steps中的错误状态
        num_failed_preprocessing_steps = sum(1 for step in preprocessing_steps if
                                             isinstance(step, dict) and step.get("status") == "failed" and step.get(
                                                 "step") != "llm_generate_preprocessing_plan")  # 不算智能体本身的失败
        if num_failed_preprocessing_steps > 0 and not any(
                step.get("operation") == "no_action" for step in preprocessing_steps if
                isinstance(step, dict) and step.get("plan")):  # 有实际操作失败
            llm_plan_failed = any(
                step.get("step") == "llm_generate_preprocessing_plan" and step.get("status") == "failed" for step in
                preprocessing_steps)
            if llm_plan_failed:
                critical_preprocessing_failed = True
                print("错误: 智能体未能成功生成预处理计划。")

        pipeline_recorder.add_stage("data_preprocessing",
                                    "failed" if critical_preprocessing_failed else "success",
                                    {"steps_details": preprocessing_steps},
                                    artifacts=preproc_artifacts)
        if critical_preprocessing_failed:
            pipeline_recorder.set_final_status("failed")
            return ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()

        # 3. 特征工程
        print("\n模块3: 特征工程...")
        preprocessed_feature_cols = [col for col in df_processed.columns if col != target_metric]
        feature_generator = FeatureGenerator(user_request=user_request, target_metric=target_metric,
                                             preprocessed_columns=preprocessed_feature_cols)
        df_engineered, fe_steps, fitted_fe_objects = feature_generator.generate_features(df_processed.copy())

        engineered_data_path = _save_dataframe(df_engineered, "经过特征工程后的最终数据集", filename_prefix, run_dir)

        # 保存拟合的特征工程对象，并将路径添加到产物记录中
        fe_artifacts = {
            "fitted_objects_keys": list(fitted_fe_objects.keys()),
            "engineered_data_path": engineered_data_path
        }
        saved_fe_path = _save_fitted_objects(fitted_fe_objects, filename_prefix, run_dir, "feature_generators")
        if saved_fe_path:
            fe_artifacts["fitted_feature_generators_path"] = saved_fe_path

        critical_fe_failed = False
        if df_engineered.empty and not df_processed.empty:
            critical_fe_failed = True
        llm_fe_plan_failed = any(
            step.get("step") == "llm_generate_fe_plan" and step.get("status") == "failed" for step in fe_steps)
        if llm_fe_plan_failed:
            critical_fe_failed = True
            print("错误: 智能体未能成功生成特征工程计划。")

        pipeline_recorder.add_stage("feature_engineering",
                                    "failed" if critical_fe_failed else "success",
                                    {"steps_details": fe_steps},
                                    artifacts=fe_artifacts)
        if critical_fe_failed:
            pipeline_recorder.set_final_status("failed")
            return ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()

        # 准备建模数据
        X = df_engineered.drop(columns=[target_metric], errors='ignore')
        y = df_engineered[target_metric]

        if X.empty:
            pipeline_recorder.add_stage("data_preparation_for_modeling", "failed",
                                        {"error": "特征集X为空，无法进行模型训练。"})
            pipeline_recorder.set_final_status("failed")
            return ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()

        # 4. 模型选择与计划制定
        print("\n模块4: 模型选择与计划制定...")
        model_selector = ModelSelector(
            user_request=user_request,
            target_metric=target_metric,
            num_features=X.shape[1],
            num_samples=X.shape[0]
        )
        automl_plan, model_selection_log = model_selector.select_model_and_plan()

        if not automl_plan or "model_recommendations" not in automl_plan or not automl_plan["model_recommendations"]:
            pipeline_recorder.add_stage("model_selection_planning", "failed",
                                        {"log": model_selection_log, "error": "未能生成任何有效的AutoML计划。"})
            pipeline_recorder.set_final_status("failed")
            return ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()

        automl_plan["user_request_details"] = request_params
        pipeline_recorder.add_stage("model_selection_planning", "success",
                                    {"model_plan": automl_plan, "log": model_selection_log})

        # 5. Pipeline构建 (决策)
        plan_details = automl_plan["model_plan"]
        model_recommendations = automl_plan["model_recommendations"]
        chosen_model_name = list(model_recommendations.keys())[0]
        chosen_model_info = model_recommendations[chosen_model_name]
        data_split_ratio = plan_details.get("data_split_ratio", 0.2)
        hpo_config = plan_details.get("hpo_config", {"method": "RandomizedSearchCV", "n_iter": 30})
        pipeline_recorder.set_model_name(chosen_model_name)
        pipeline_recorder.add_stage("decision_making", "success",
                                    {"chosen_model": chosen_model_name, "data_split_ratio": data_split_ratio,
                                     "hpo_config": hpo_config})

        # 6. 模型训练与评估
        print(f"\n模块6: 模型训练与评估 (模型: {chosen_model_name})...")
        trainer = ModelTrainer(selected_model_name=chosen_model_name, model_info=chosen_model_info,
                               hpo_config=hpo_config, automl_plan=automl_plan)
        trainer.train_and_evaluate(X.copy(), y.copy(), test_size=data_split_ratio)

        training_artifacts = {}
        if trainer.evaluation_results:
            saved_artifacts = trainer.evaluation_results.get("artifacts", {})
            training_artifacts.update(saved_artifacts)

        # 为了 pipeline 记录文件的逻辑性，过滤掉 evaluation_results 中的 artifacts 字段
        evaluation_results = {
            key: value for key, value in trainer.evaluation_results.items() if key != "artifacts"
        }

        pipeline_recorder.add_stage("model_training_evaluation",
                                    "success" if trainer.evaluation_results and trainer.evaluation_results.get(
                                        "test") else "failed",
                                    {"training_log": trainer.training_log, "evaluation_metrics": evaluation_results},
                                    artifacts=training_artifacts)

        if not trainer.evaluation_results or not trainer.evaluation_results.get("test"):
            pipeline_recorder.set_final_status("failed")
            return ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()

        # 7. 结果汇总
        print("\n模块7: 结果汇总...")
        pipeline_recorder.set_final_status("success")
        result_handler = ResultHandler(pipeline_summary=pipeline_recorder.get_pipeline_summary())
        result_handler.add_model_details(
            model_name=chosen_model_name,
            best_hyperparams=trainer.model_instance.best_params_ if trainer.model_instance else None
        )
        result_handler.add_evaluation_metrics(metrics=trainer.evaluation_results)
        result_handler.add_feature_importances(importances=trainer.feature_importances)
        final_result_package = result_handler.compile_final_result()
        result_handler.save_final_result()

        return final_result_package

    except Exception as e:
        print(f"AutoML流程执行过程中发生未捕获的严重错误: {e}")
        import traceback
        traceback.print_exc()

        pipeline_recorder.add_stage("global_error_handler", "failed",
                                    {"error_type": type(e).__name__, "message": str(e),
                                     "traceback": traceback.format_exc()})
        pipeline_recorder.set_final_status("failed")

        final_result_package = ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()
        if "status" not in final_result_package or final_result_package["status"] != "failed":
            final_result_package["status"] = "failed"
        if "error_details" not in final_result_package:
            final_result_package["error_details"] = {"message": str(e), "type": type(e).__name__}

        # 即使失败，也尝试保存最终的pipeline记录
        if 'pipeline_summary' in final_result_package:
            try:
                summary_path = os.path.join(run_dir, "failed_pipeline_summary.json")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(final_result_package['pipeline_summary'], f, indent=4, ensure_ascii=False)
                print(f"失败的pipeline摘要已保存至: {summary_path}")
            except Exception as save_e:
                print(f"保存失败的pipeline摘要时出错: {save_e}")

        return final_result_package
