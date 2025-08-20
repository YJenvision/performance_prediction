# @Time    : 2025/5/16
# @Author  : ZhangJingLiang
# @Email   : jinglianglink@qq.com
# @Project : performance_prediction_agent

import time
import os
import pickle
from datetime import datetime
from typing import Dict, Any, List, Tuple, Generator

import pandas as pd

from config import DB_CONFIG
from steel_automl.data_acquisition.data_loader import DataLoader
from steel_automl.data_preprocessing.preprocessor import DataPreprocessor
from steel_automl.feature_engineering.feature_generator import FeatureGenerator
from steel_automl.model_selection.selector import ModelSelector
from steel_automl.modeling.trainer_evaluator import ModelTrainer
from steel_automl.performance_model_builder_utils import _generate_filename, _save_dataframe, \
    _save_fitted_objects
from steel_automl.pipeline.pipeline_builder import PipelineBuilder
from steel_automl.results.result_handler import ResultHandler


def _consume_sub_generator(generator: Generator) -> Generator[Dict[str, Any], None, Tuple[Any, bool]]:
    """
    一个辅助生成器，用于消费子生成器，传递其中间消息，并检查是否有错误。
    它本身是一个生成器，最终会 return 一个元组 (final_return_value, has_failed)。
    """
    final_return_value = None
    has_failed = False
    while True:
        try:
            chunk = next(generator)
            payload = chunk.get("payload", {})
            if chunk.get("type") == "error" or \
                    (chunk.get("type") == "status_update" and payload.get("status") == "failed"):
                has_failed = True

            yield chunk

            if has_failed:
                # 如果检测到失败，立即停止消费并返回
                return None, True
        except StopIteration as e:
            # 子生成器正常结束，捕获其返回值
            final_return_value = e.value
            break
    return final_return_value, has_failed


def performanceModelBuilder(
        request_params: Dict[str, Any]
) -> Generator[Dict[str, Any], None, None]:
    """
    性能预报模型AutoML主流程，生成器。
    """
    start_time_total = time.time()
    run_timestamp = datetime.now()
    run_timestamp_str = run_timestamp.strftime('%Y%m%d%H%M%S')
    run_dir = "automl_runs"
    os.makedirs(run_dir, exist_ok=True)

    # 根据请求参数和时间戳为本次运行生成一个专属的文件夹名称
    run_folder_name = _generate_filename(request_params, run_timestamp_str)
    # 创建本次运行的根目录
    run_specific_dir = os.path.join("automl_runs", run_folder_name)
    os.makedirs(run_specific_dir, exist_ok=True)

    user_request = request_params.get("user_request", "用户具体请求描述为空。")
    target_metric = request_params.get("target_metric")

    pipeline_recorder = PipelineBuilder(user_request_details=request_params)

    try:
        # 1. 数据收集
        current_stage = "数据收集"
        data_loader = DataLoader(db_config=DB_CONFIG)

        # 连接数据库，从数据库中加载数据。
        fetch_data_generator = data_loader.fetch_data(
            sg_sign=request_params.get("sg_sign"),
            target_metric=request_params.get("target_metric"),
            time_range=request_params.get("time_range"),
            product_unit_no=request_params.get("product_unit_no"),
            st_no=request_params.get("st_no"),
            steel_grade=request_params.get("steel_grade")
        )

        returned_value, has_failed = yield from _consume_sub_generator(fetch_data_generator)
        if has_failed:
            return

        raw_data, sql_query = returned_value

        if raw_data is None or raw_data.empty:
            error_detail = "未能获取到有效数据或数据集为空。"
            yield {"type": "error",
                   "payload": {"stage": current_stage, "detail": error_detail + "查询条件为：" + sql_query}}
            return

        if target_metric not in raw_data.columns:
            error_detail = f"获取的数据中不包含目标性能字段 '{target_metric}'。"
            yield {"type": "error",
                   "payload": {"stage": current_stage, "detail": error_detail + "查询条件为：" + sql_query}}
            return

        # 如果存在时间列，则按时间排序
        if 'REC_REVISE_TIME' in raw_data.columns:
            yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                        "detail": "数据包含时间戳，正在按时间顺序排序以支持时序建模..."}}
            try:
                # 确保时间列是可排序的类型
                raw_data['REC_REVISE_TIME'] = pd.to_datetime(raw_data['REC_REVISE_TIME'])
                raw_data.sort_values(by='REC_REVISE_TIME', inplace=True)
                raw_data.reset_index(drop=True, inplace=True)
                yield {"type": "thinking_stream", "payload": f"按时间列 'REC_REVISE_TIME' 升序排序成功。"}
            except Exception as e:
                yield {"type": "error", "payload": {"stage": current_stage,
                                                    "detail": f"按时间列 'REC_REVISE_TIME' 排序时失败: {e}"}}

        _save_dataframe(raw_data, "#1原始数据集", run_specific_dir)
        pipeline_recorder.add_stage(current_stage, "success", {"sql_query": sql_query, "num_rows": len(raw_data)})

        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "success", "detail": "完成数据收集。"}}

        # 2. 数据探索与预处理
        current_stage = "数据探索与预处理"

        preprocessor = DataPreprocessor(user_request=user_request, target_metric=target_metric)
        # 调用预处理器时传入文件名和路径参数
        preprocess_generator = preprocessor.preprocess_data(raw_data.copy(), run_specific_dir)

        returned_value, has_failed = yield from _consume_sub_generator(preprocess_generator)

        if has_failed:
            pipeline_recorder.add_stage(current_stage, "failed", {"details": "数据探索与预处理流程因错误中断。"})
            pipeline_recorder.set_final_status("failed")
            return
        # 修改点：更新返回值的解包逻辑
        df_processed, preprocessing_steps, fitted_preproc_objects, preprocessed_data_path = returned_value

        preproc_artifacts = {
            "fitted_objects_keys": list(fitted_preproc_objects.keys()),
            "preprocessed_data_path": preprocessed_data_path
        }

        # 这里的 'fitted_preproc_objects' 是一个包含了所有预处理阶段（缺失值填充、变换、编码、缩放等）的已拟合转换器（fitters）的字典。
        # 将这个字典作为一个整体保存，是确保模型部署时能够重现完全相同的预处理流程的最佳实践。
        # 文件将保存在指定的 'preprocessors' 文件夹内。
        if fitted_preproc_objects:
            saved_preproc_path = _save_fitted_objects(fitted_preproc_objects, run_specific_dir,
                                                      "preprocessors")
            if saved_preproc_path:
                preproc_artifacts["fitted_preprocessors_path"] = saved_preproc_path

        critical_preprocessing_failed = False

        if df_processed.empty and not raw_data.empty:
            critical_preprocessing_failed = True
            yield {"type": "error",
                   "payload": {"stage": current_stage, "detail": "经过有效特征筛选和预处理后的数据集为空，流程中止。"}}
            return

        llm_plan_failed = any(
            step.get("step") == "生成数据预处理计划" and step.get("status") == "failed" for step in preprocessing_steps
        )
        if llm_plan_failed:
            critical_preprocessing_failed = True

        pipeline_recorder.add_stage(current_stage,
                                    "failed" if critical_preprocessing_failed else "success",
                                    {"steps_details": preprocessing_steps},
                                    artifacts=preproc_artifacts)

        if critical_preprocessing_failed:
            yield {"type": "error", "payload": {"stage": current_stage, "detail": "关键预处理步骤失败，流程中止。"}}
            pipeline_recorder.set_final_status("failed")
            return

        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "success", "detail": "完成数据的探索与预处理。"}}

        # 3. 特征工程
        current_stage = "特征工程"

        preprocessed_feature_cols = [col for col in df_processed.columns if col != target_metric]
        feature_generator = FeatureGenerator(user_request=user_request, target_metric=target_metric,
                                             preprocessed_columns=preprocessed_feature_cols)

        fe_generator = feature_generator.generate_features(df_processed.copy())
        returned_value, has_failed = yield from _consume_sub_generator(fe_generator)

        if has_failed:
            pipeline_recorder.add_stage(current_stage, "failed", {"details": "特征构造流程因错误中断。"})
            return

        df_engineered, fe_steps, fitted_fe_objects = returned_value

        engineered_data_path = _save_dataframe(df_engineered, "#6经过特征构造后的数据集",
                                               run_specific_dir)

        fe_artifacts = {
            "fitted_objects_keys": list(fitted_fe_objects.keys()),
            "engineered_data_path": engineered_data_path
        }

        if fitted_fe_objects:
            saved_fe_path = _save_fitted_objects(fitted_fe_objects, run_specific_dir,
                                                 "feature_generators")
            if saved_fe_path:
                fe_artifacts["fitted_feature_generators_path"] = saved_fe_path

        if df_engineered.empty and not df_processed.empty:
            yield {"type": "error", "payload": {"stage": current_stage, "detail": "特征构造后数据集为空，流程中止。"}}
            return
        pipeline_recorder.add_stage(current_stage, "success", {"steps_details": fe_steps}, artifacts=fe_artifacts)

        X = df_engineered.drop(columns=[target_metric], errors='ignore')
        y = df_engineered[target_metric]

        if X.empty:
            yield {"type": "error", "payload": {"stage": current_stage, "detail": "训练特征集为空，无法进行模型训练。"}}
            return

        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "success", "detail": "完成特征构造。"}}

        # 4. 模型选择与计划制定
        current_stage = "模型选择与计划制定"

        model_selector = ModelSelector(
            user_request=user_request,
            target_metric=target_metric,
            num_features=X.shape[1],
            num_samples=X.shape[0]
        )

        plan_generator = model_selector.select_model_and_plan()
        returned_value, has_failed = yield from _consume_sub_generator(plan_generator)

        if has_failed:
            pipeline_recorder.add_stage("model_selection_planning", "failed", {"details": "模型选择流程因错误中断。"})
            return

        automl_plan, model_selection_log = returned_value

        # 将完整的用户请求参数添加到automl_plan中，
        # 以确保后续模块（ModelTrainer）可以访问它们以生成正确格式的文件名。
        if automl_plan:
            automl_plan["user_request_details"] = request_params

        if not automl_plan or "model_recommendations" not in automl_plan or not automl_plan["model_recommendations"]:
            error_msg = "未能生成有效的模型算法计划。"
            pipeline_recorder.add_stage("model_selection_planning", "failed",
                                        {"log": model_selection_log, "error": error_msg})
            yield {"type": "error", "payload": {"stage": current_stage, "detail": error_msg}}
            return

        pipeline_recorder.add_stage("model_selection_planning", "success",
                                    {"model_plan": automl_plan, "log": model_selection_log})

        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "success", "detail": "完成模型选择与计划制定。"}}

        # 5. Pipeline构建
        plan_details = automl_plan["model_plan"]
        model_recommendations = automl_plan["model_recommendations"]
        chosen_model_name = list(model_recommendations.keys())[0]
        chosen_model_info = model_recommendations[chosen_model_name]

        # 从新的计划结构中提取信息
        data_split_plan = plan_details.get("data_split_plan", {"method": "sequential", "test_size": 0.2})
        cv_plan = plan_details.get("cv_plan", {"method": "time_series", "k_folds": 3})
        hpo_config = plan_details.get("hpo_config", {"method": "RandomizedSearchCV", "n_iter": 30})

        pipeline_recorder.set_model_name(chosen_model_name)
        pipeline_recorder.add_stage("decision_making", "success",
                                    {"chosen_model": chosen_model_name,
                                     "data_split_plan": data_split_plan,
                                     "cv_plan": cv_plan,
                                     "hpo_config": hpo_config})

        # 6. 模型训练与评估
        current_stage = "模型训练与评估"

        trainer = ModelTrainer(selected_model_name=chosen_model_name,
                               model_info=chosen_model_info,
                               automl_plan=automl_plan,
                               run_specific_dir=run_specific_dir)

        # train_and_evaluate 不再直接接收 test_size，它会从 automl_plan 中读取完整的划分计划
        train_generator = trainer.train_and_evaluate(X.copy(), y.copy())
        training_succeeded, has_failed = yield from _consume_sub_generator(train_generator)

        if has_failed or not training_succeeded:
            error_msg = "模型训练和评估流程失败。"
            pipeline_recorder.add_stage("model_training_evaluation", "failed",
                                        {"training_log": trainer.training_log, "error": error_msg})
            pipeline_recorder.set_final_status("failed")
            # 训练器应该已经给出了错误消息
            return

        # 训练成功，现在记录结果
        training_artifacts = {}
        if trainer.evaluation_results:
            saved_artifacts = trainer.evaluation_results.get("artifacts", {})
            training_artifacts.update(saved_artifacts)

        evaluation_results = {
            key: value for key, value in trainer.evaluation_results.items() if key != "artifacts"
        }

        pipeline_recorder.add_stage("model_training_evaluation", "success",
                                    {"training_log": trainer.training_log, "evaluation_metrics": evaluation_results},
                                    artifacts=training_artifacts)

        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "success",
                           "detail": f"{chosen_model_name}模型训练和评估成功完成。"}}

        # 7. 结果汇总
        current_stage = "结果汇总"
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "正在汇总最终结果..."}}

        pipeline_recorder.set_final_status("success")
        result_handler = ResultHandler(pipeline_summary=pipeline_recorder.get_pipeline_summary(),
                                       run_specific_dir=run_specific_dir)
        result_handler.add_model_details(
            model_name=chosen_model_name,
            best_hyperparams=trainer.model_instance.best_params_ if trainer.model_instance else None
        )
        result_handler.add_evaluation_metrics(metrics=trainer.evaluation_results)
        result_handler.add_feature_importances(importances=trainer.feature_importances)
        result_handler.save_final_result()

        yield {"type": "stage_completed", "payload": {"stage": current_stage, "status": "success",
                                                      "detail": "性能预报智能体的建模与评估流程成功完成。"}}
        return

    except Exception as e:
        import traceback
        error_msg = f"智能体在AutoML流程中发生严重错误: {e}"
        print(error_msg)
        print(traceback.format_exc())
        yield {"type": "error",
               "payload": {"stage": "全局错误", "detail": error_msg + traceback.format_exc()}}

        pipeline_recorder.add_stage("global_error_handler", "failed",
                                    {"error_type": type(e).__name__, "message": str(e),
                                     "traceback": traceback.format_exc()})
        pipeline_recorder.set_final_status("failed")

        # 即使失败，编译并返回到目前为止的结果
        ResultHandler(
            pipeline_recorder.get_pipeline_summary(),
            run_specific_dir=run_specific_dir
        ).save_final_result()
        return
