import time
from typing import Dict, Any
from config import DB_CONFIG
from steel_automl.data_acquisition.data_loader import DataLoader
from steel_automl.data_preprocessing.preprocessor import DataPreprocessor
from steel_automl.feature_engineering.feature_generator import FeatureGenerator
from steel_automl.model_selection.selector import ModelSelector
from steel_automl.modeling.trainer import ModelTrainer
from steel_automl.pipeline.pipeline_builder import PipelineBuilder
from steel_automl.results.result_handler import ResultHandler


def performanceModelBuilder(request_params: Dict[str, str]) -> Dict[str, Any]:
    """
    钢铁产品性能预报回归模型AutoML主流程。

    参数:
    - request_params: 包含用户请求的字典，应包括:
        - user_request: str (用户的原始自然语言建模请求描述)
        - sg_sign: str (牌号)
        - target_metric: str (目标性能指标)
        - time_range: str (数据时间范围, e.g., "2023-01-01_to_2025-12-31")
        - product_unit_no: str (机组/产线号)
        - st_no: str (出钢记号)
        - 后续逻辑待补充完善
    返回:
    - 一个包含模型信息与评估结果的字典。
    """
    start_time_total = time.time()  # 记录本次建模评估流程执行时间

    user_request = request_params.get("user_request", "用户具体请求描述为空。")
    sg_sign = request_params.get("sg_sign")
    target_metric = request_params.get("target_metric")
    time_range = request_params.get("time_range")
    product_unit_no = request_params.get("product_unit_no")
    st_no = request_params.get("st_no")

    if not all([target_metric]):
        return {"status": "failed", "error": "请求参数不完整 (目标性能字段target_metric必须提供)。"}

    # 0. 初始化Pipeline记录器
    pipeline_recorder = PipelineBuilder(user_request_details=request_params)
    final_result_package = {"status": "pending"}  # 初始化最终结果

    try:
        # 1. 数据获取
        print("\n模块1: 数据获取...")
        data_loader = DataLoader(db_config=DB_CONFIG)
        # 数据库取数
        raw_data = data_loader.fetch_data(sg_sign, target_metric, time_range, product_unit_no, st_no)
        # 从excel中取数
        # raw_data = data_loader.fetch_data_from_excel(sg_sign, target_metric, time_range, product_unit_no, st_no)
        if raw_data is None or raw_data.empty:
            pipeline_recorder.add_stage("data_acquisition", "failed", {"error": "未能获取到数据或数据为空"})
            pipeline_recorder.set_final_status("failed")
            final_result_package = ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()
            return final_result_package
        pipeline_recorder.add_stage("data_acquisition", "success",
                                    {"rows_fetched": len(raw_data), "columns": list(raw_data.columns)})

        # 确保目标列存在
        if target_metric not in raw_data.columns:
            pipeline_recorder.add_stage("data_validation", "failed",
                                        {"error": f"目标列 '{target_metric}' 在获取的数据中不存在。"})
            pipeline_recorder.set_final_status("failed")
            final_result_package = ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()
            return final_result_package

        # 2. 数据预处理
        print("\n模块2: 数据预处理...")
        preprocessor = DataPreprocessor(user_request=user_request, target_metric=target_metric)
        df_processed, preprocessing_steps, fitted_preproc_objects = preprocessor.preprocess_data(
            raw_data.copy())  # 使用副本

        # 检查预处理步骤中是否有严重错误导致无法继续
        # (一个简单的检查：如果df_processed为空但原始数据不为空，或者关键步骤失败)
        critical_preprocessing_failed = False
        if df_processed.empty and not raw_data.empty:
            critical_preprocessing_failed = True
            print("错误: 数据预处理后DataFrame为空。")
        # 检查preprocessing_steps中的错误状态
        num_failed_preprocessing_steps = sum(1 for step in preprocessing_steps if
                                             isinstance(step, dict) and step.get("status") == "failed" and step.get(
                                                 "step") != "llm_generate_preprocessing_plan")  # 不算LLM本身的失败
        if num_failed_preprocessing_steps > 0 and not any(
                step.get("operation") == "no_action" for step in preprocessing_steps if
                isinstance(step, dict) and step.get("plan")):  # 有实际操作失败
            # LLM计划本身失败，也算严重错误
            llm_plan_failed = any(
                step.get("step") == "llm_generate_preprocessing_plan" and step.get("status") == "failed" for step in
                preprocessing_steps)
            if llm_plan_failed:
                critical_preprocessing_failed = True
                print("错误: LLM未能成功生成预处理计划。")

        pipeline_recorder.add_stage("data_preprocessing",
                                    "failed" if critical_preprocessing_failed else "success",
                                    {"steps_details": preprocessing_steps},
                                    {"fitted_objects_keys": list(fitted_preproc_objects.keys())})
        if critical_preprocessing_failed:
            pipeline_recorder.set_final_status("failed")
            final_result_package = ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()
            return final_result_package

        # 3. 特征工程
        print("\n模块3: 特征工程...")
        # 获取预处理后的特征列 (排除目标列)
        preprocessed_feature_cols = [col for col in df_processed.columns if col != target_metric]
        feature_generator = FeatureGenerator(user_request=user_request, target_metric=target_metric,
                                             preprocessed_columns=preprocessed_feature_cols)
        df_engineered, fe_steps, fitted_fe_objects = feature_generator.generate_features(df_processed.copy())  # 使用副本

        # 类似地检查特征工程的关键失败
        critical_fe_failed = False
        if df_engineered.empty and not df_processed.empty:
            critical_fe_failed = True
        llm_fe_plan_failed = any(
            step.get("step") == "llm_generate_fe_plan" and step.get("status") == "failed" for step in fe_steps)
        if llm_fe_plan_failed:
            critical_fe_failed = True
            print("错误: LLM未能成功生成特征工程计划。")

        pipeline_recorder.add_stage("feature_engineering",
                                    "failed" if critical_fe_failed else "success",
                                    {"steps_details": fe_steps},
                                    {"fitted_objects_keys": list(fitted_fe_objects.keys())})
        if critical_fe_failed:
            pipeline_recorder.set_final_status("failed")
            final_result_package = ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()
            return final_result_package

        # 准备建模数据
        X = df_engineered.drop(columns=[target_metric], errors='ignore')
        y = df_engineered[target_metric]

        if X.empty:
            pipeline_recorder.add_stage("data_preparation_for_modeling", "failed",
                                        {"error": "特征集X为空，无法进行模型训练。可能所有特征都被删除或处理错误。"})
            pipeline_recorder.set_final_status("failed")
            final_result_package = ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()
            return final_result_package

        # 4. 模型选择
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

        pipeline_recorder.add_stage("model_selection_planning", "success",
                                    {"model_plan": automl_plan, "log": model_selection_log})

        # 5. Pipeline构建 (此处简化为记录选择的模型和参数)
        # 实际的scikit-learn Pipeline对象可以将预处理器和模型串联起来
        # 目前只选择一个模型进行训练 (例如，系统推荐的第一个，或基于某种策略选择)
        # 简单开始，选择推荐列表中的第一个模型
        # 从计划中提取信息
        plan_details = automl_plan["model_plan"]
        model_recommendations = automl_plan["model_recommendations"]

        # 简单策略：选择推荐列表中的第一个模型进行训练
        chosen_model_name = list(model_recommendations.keys())[0]
        chosen_model_info = model_recommendations[chosen_model_name]
        data_split_ratio = plan_details.get("data_split_ratio", 0.2)  # 使用计划的划分比例，提供默认值
        hpo_config = plan_details.get("hpo_config", {"method": "RandomizedSearchCV", "n_iter": 30})  # 使用计划的HPO配置

        pipeline_recorder.add_stage("decision_making", "success",
                                    {"chosen_model": chosen_model_name,
                                     "data_split_ratio": data_split_ratio,
                                     "hpo_config": hpo_config})

        print(f"\n模块6: 模型训练与评估 (模型: {chosen_model_name})...")
        trainer = ModelTrainer(
            selected_model_name=chosen_model_name,
            model_info=chosen_model_info,
            hpo_config=hpo_config
        )
        # 将动态的划分比例传入
        trainer.train_and_evaluate(X.copy(), y.copy(), test_size=data_split_ratio)

        pipeline_recorder.add_stage("model_training_evaluation",
                                    "success" if trainer.evaluation_results else "failed",
                                    {"training_log": trainer.training_log, "evaluation_metrics": trainer.evaluation_results})

        if not trainer.evaluation_results:
            # 错误处理
            pipeline_recorder.set_final_status("failed")
            # 可以在这里保存pipeline摘要
            pipeline_recorder.save_pipeline_summary()
            return {"status": "failed", "error": "模型训练或评估失败。"}

        # 7. 结果汇总
        print("\n模块7: 结果汇总...")
        pipeline_recorder.set_final_status("success")  # 如果流程走到这里，认为是成功的

        # 创建ResultHandler实例并填充信息
        result_handler = ResultHandler(pipeline_summary=pipeline_recorder.get_pipeline_summary())
        result_handler.add_model_details(
            model_name=chosen_model_name,
            best_hyperparams=trainer.model_instance.best_params_ if trainer.model_instance else None
            # model_object_ref: 实际保存模型到文件并记录路径
        )
        result_handler.add_evaluation_metrics(metrics=trainer.evaluation_results)
        result_handler.add_feature_importances(importances=trainer.feature_importances)

        final_result_package = result_handler.compile_final_result()

        # 保存Pipeline摘要和最终结果
        pipeline_recorder.save_pipeline_summary()
        result_handler.save_final_result()

        total_duration = time.time() - start_time_total
        print(f"\n--- 整个AutoML流程完成 ---")
        print(f"总耗时: {total_duration:.2f} 秒")
        print(f"最终状态: {final_result_package.get('status')}")
        if final_result_package.get('status') == 'success':
            print(f"最终模型: {final_result_package.get('selected_model', {}).get('model_name')}")
            print(f"评估R2分数: {final_result_package.get('evaluation_metrics', {}).get('r2')}")

        return final_result_package

    except Exception as e:
        print(f"AutoML流程执行过程中发生未捕获的严重错误: {e}")
        import traceback
        traceback.print_exc()  # 打印完整的堆栈跟踪

        pipeline_recorder.add_stage("global_error_handler", "failed",
                                    {"error_type": type(e).__name__, "message": str(e),
                                     "traceback": traceback.format_exc()})
        pipeline_recorder.set_final_status("failed")

        # 尝试保存已有的pipeline信息
        pipeline_recorder.save_pipeline_summary()

        # 返回错误信息
        final_result_package = ResultHandler(pipeline_recorder.get_pipeline_summary()).compile_final_result()
        if "status" not in final_result_package or final_result_package["status"] != "failed":
            final_result_package["status"] = "failed"
        if "error_details" not in final_result_package:
            final_result_package["error_details"] = {"message": str(e), "type": type(e).__name__}

        return final_result_package
