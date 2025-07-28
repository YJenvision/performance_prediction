import json
from typing import Dict, Any, Tuple, List, Generator

import pandas as pd
from pandas import DataFrame

from config import FEATURE_ENGINEERING_KB_NAME
from knowledge_base.kb_service import KnowledgeBaseService
from llm_utils import call_llm
from steel_automl.feature_engineering.methods import FEATURE_ENGINEERING_METHODS_MAP
from prompts.prompt_manager import get_prompt


class FeatureGenerator:
    """
    一个由LLM驱动的智能特征生成器。

    它通过以下步骤工作：
    1. 从知识库中检索与当前任务相关的特征工程知识。
    2. 让智能体根据用户需求、现有特征和检索到的知识，制定一个特征工程计划。
       这个计划的核心是让LLM动态地将理论公式中的元素映射到数据中的实际列名。
    3. 执行智能体生成的计划，创建新的特征。
    """

    def __init__(self, user_request: str, target_metric: str, preprocessed_columns: List[str]):
        """
        初始化特征生成器。

        参数:
        - user_request: 用户的原始自然语言建模请求。
        - target_metric: 目标性能指标的列名。
        - preprocessed_columns: 经过预处理后的特征列列表。
        """
        self.user_request = user_request
        self.target_metric = target_metric
        self.preprocessed_columns = preprocessed_columns
        # 初始化知识库服务，并加载您提供的知识
        self.fe_kb_service = KnowledgeBaseService(FEATURE_ENGINEERING_KB_NAME)
        # self.history_kb_service = KnowledgeBaseService(HISTORICAL_CASES_KB_NAME)  # 用于参考历史案例
        self.applied_steps = []
        self.fitted_objects = {}  # 用于存储需要拟合的对象，如多项式转换器

    def _get_knowledge_snippets_with_metadata(self, context_query: str, k: int = 3) -> str:
        """从特征工程和历史案例知识库检索信息。"""
        fe_snippets = self.fe_kb_service.search(context_query, k=k)
        # history_snippets = self.history_kb_service.search(context_query, k=k)

        formatted_snippets = ""
        if fe_snippets:
            for i, snippet in enumerate(fe_snippets):
                formatted_snippets += f"{i + 1}. {snippet.get('metadata', {})}\n"
        else:
            formatted_snippets += "未从特征工程知识库中检索到相关信息。\n"
        return formatted_snippets

    def _generate_llm_prompt_for_fe_plan(self, current_features_str: str) -> Tuple[str, str]:
        """为特征工程计划生成系统和用户提示词。"""
        # 从知识库检索相关知识
        kb_query_context = f"为回归任务生成特征，目标是'{self.target_metric}'，现有特征包括化学成分和工艺参数。"
        knowledge_snippets = self._get_knowledge_snippets_with_metadata(kb_query_context)

        # 将知识片段格式化，以便LLM清晰地理解
        formatted_knowledge = json.dumps(knowledge_snippets, indent=2, ensure_ascii=False)

        # 从配置文件加载提示词
        system_prompt = get_prompt('feature_generator.generate_fe_plan.system')
        user_prompt = get_prompt(
            'feature_generator.generate_fe_plan.user',
            user_request=self.user_request,
            target_metric=self.target_metric,
            current_features_str=current_features_str,
            formatted_knowledge=formatted_knowledge
        )
        return system_prompt, user_prompt

    def _execute_feature_engineering_plan(self, df: pd.DataFrame, plan: List[Dict[str, Any]]) -> Generator[
        Dict[str, Any], None, pd.DataFrame]:
        """
        根据制定的计划执行特征工程步骤，现在是一个生成器，可以流式返回每一步的结果。
        """
        df_engineered = df.copy()
        current_stage = "特征工程"

        for i, step_details in enumerate(plan):
            operation = step_details.get("operation")
            params = step_details.get("params", {})
            step_log = {"operation": operation, "params": params, "status": "failed"}

            yield {"type": "status_update", "payload": {
                "stage": current_stage, "status": "running",
                "detail": f"执行步骤 {i + 1}/{len(plan)}: {operation}"
            }}

            try:
                if operation == "no_action":
                    step_log["status"] = "no_action"
                    result_data = f"步骤 {i + 1}: 无需执行操作。"
                elif operation in FEATURE_ENGINEERING_METHODS_MAP:
                    method_to_call = FEATURE_ENGINEERING_METHODS_MAP[operation]
                    original_cols = set(df_engineered.columns)

                    if operation == "create_polynomial_features":
                        df_engineered, fitted_obj = method_to_call(df_engineered, **params)
                        self.fitted_objects[f"poly_{'_'.join(params.get('columns', []))}"] = fitted_obj
                    else:
                        df_engineered = method_to_call(df_engineered, **params)

                    new_cols = set(df_engineered.columns) - original_cols
                    step_log["status"] = "success"
                    step_log["new_features"] = list(new_cols)
                    result_data = {
                        "operation": operation,
                        "params": params,
                        "status": "成功",
                        "new_features": list(new_cols) if new_cols else "无新增特征（例如，部分操作是替换或修改现有列）"
                    }
                else:
                    step_log["error"] = f"Unknown operation: {operation}"
                    result_data = {"operation": operation, "status": "失败", "error": f"未知操作: {operation}"}

            except Exception as e:
                step_log["error"] = str(e)
                result_data = {"operation": operation, "status": "失败", "error": str(e)}

            self.applied_steps.append(step_log)
            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": f"执行步骤: {operation}",
                "data": result_data
            }}

        return df_engineered

    def generate_features(self, df: pd.DataFrame) -> Generator[
        Dict[str, Any], None, Tuple[DataFrame, List[Dict], Dict]]:
        """
        执行特征工程的主流程，现在是一个生成器。

        返回:
        - (通过 yield) 流式思考和状态更新。
        - (通过 return) df_engineered, applied_steps, fitted_objects
        """
        current_stage = "特征工程"
        self.applied_steps = []
        self.fitted_objects = {}
        current_feature_names = [col for col in df.columns if col != self.target_metric]
        current_features_str = ", ".join(current_feature_names)

        # 子任务 1: 制定特征工程计划
        yield {"type": "status_update", "payload": {
            "stage": current_stage, "status": "running",
            "detail": "正在分析数据并制定特征工程计划..."
        }}
        system_prompt, user_prompt = self._generate_llm_prompt_for_fe_plan(current_features_str)

        llm_gen = call_llm(system_prompt, user_prompt, model="ds_v3")
        llm_response_str = ""
        while True:
            try:
                chunk = next(llm_gen)
                if chunk.get("type") == "error":
                    yield chunk
                    self.applied_steps.append({"step": "llm_generate_fe_plan", "status": "failed",
                                               "error": "智能体调用失败", "raw_response": ""})
                    return df, self.applied_steps, self.fitted_objects
                yield chunk
            except StopIteration as e:
                llm_response_str = e.value
                break

        if "Agent failed" in llm_response_str:
            yield {"type": "error",
                   "payload": {"stage": current_stage, "detail": "特征工程计划制定失败：智能体返回错误。"}}
            self.applied_steps.append({"step": "llm_generate_fe_plan", "status": "failed", "error": "智能体调用失败",
                                       "raw_response": llm_response_str})
            return df, self.applied_steps, self.fitted_objects

        feature_engineering_plan = []
        try:
            feature_engineering_plan = json.loads(llm_response_str)
            if not isinstance(feature_engineering_plan, list):
                raise ValueError("智能体响应不是一个列表。")

            self.applied_steps.append(
                {"step": "llm_generate_fe_plan", "status": "success", "plan": feature_engineering_plan})
            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "特征工程计划",
                "data": feature_engineering_plan
            }}
        except (json.JSONDecodeError, ValueError) as e:
            error_msg = f"生成的计划无效: {e}"
            yield {"type": "error", "payload": {"stage": current_stage, "detail": "特征工程计划生成失败：" + error_msg}}
            self.applied_steps.append(
                {"step": "llm_generate_fe_plan", "status": "failed", "error": str(e), "raw_response": llm_response_str})
            return df, self.applied_steps, self.fitted_objects

        # 子任务 2: 执行特征工程计划
        if not feature_engineering_plan or (
                len(feature_engineering_plan) == 1 and feature_engineering_plan[0].get("operation") == "no_action"):
            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "执行计划",
                "data": "评估后认为无需新增特征。"
            }}
        else:
            yield {"type": "status_update", "payload": {
                "stage": current_stage, "status": "running",
                "detail": "正在执行特征工程计划..."
            }}
            execution_generator = self._execute_feature_engineering_plan(df, feature_engineering_plan)
            df = yield from execution_generator

        return df, self.applied_steps, self.fitted_objects
