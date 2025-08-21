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
    1. 从知识库中检索与当前任务相关的特征构造知识。
    2. 让智能体根据用户需求、现有特征和检索到的知识，制定一个特征构造计划。计划的核心是动态地将理论公式中的元素映射到数据中的实际列名。
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
        self.fe_kb_service = KnowledgeBaseService(FEATURE_ENGINEERING_KB_NAME)
        self.applied_steps = []
        self.fitted_objects = {}

    def _get_knowledge_snippets_with_metadata(self, context_query: str, k: int = 3) -> str:
        """从特征工程知识库检索信息。"""
        fe_snippets = self.fe_kb_service.search(context_query, k=k)
        formatted_snippets = ""
        if fe_snippets:
            for i, snippet in enumerate(fe_snippets):
                formatted_snippets += f"{i + 1}. {snippet.get('metadata', {})}\n"
        else:
            formatted_snippets += "未从特征工程知识库中检索到相关信息。\n"
        return formatted_snippets

    def _generate_llm_prompt_for_fe_plan(self, current_features_str: str) -> Tuple[str, str]:
        """为特征工程计划生成系统和用户提示词。"""
        kb_query_context = f"为回归任务生成特征，目标是'{self.target_metric}'，现有特征包括化学成分和工艺参数。"
        knowledge_snippets = self._get_knowledge_snippets_with_metadata(kb_query_context)
        formatted_knowledge = json.dumps(knowledge_snippets, indent=2, ensure_ascii=False)
        system_prompt = get_prompt('feature_generator.generate_fe_plan.system')
        user_prompt = get_prompt(
            'feature_generator.generate_fe_plan.user',
            user_request=self.user_request,
            target_metric=self.target_metric,
            current_features_str=current_features_str,
            formatted_knowledge=formatted_knowledge
        )
        return system_prompt, user_prompt

    def _generate_step_description(self, operation: str, params: Dict[str, Any]) -> str:
        """
        根据操作和参数生成一个人类可读的步骤描述。
        """
        if operation == "apply_knowledge_based_formula":
            formula = params.get('formula_template', 'N/A')
            new_feature = params.get('new_feature_name', 'N/A')
            cols = list(params.get('column_mapping', {}).values())
            return f"应用领域知识公式 '{formula}' 来创建新特征 '{new_feature}'。该公式将使用 {'、 '.join(cols)} 等特征进行计算。"

        elif operation == "create_polynomial_features":
            columns = params.get('columns', [])
            degree = params.get('degree', 2)
            interaction_only = params.get('interaction_only', False)
            desc = f"为 {', '.join(columns)} 等特征创建最高 {degree} 次的多项式/交互特征"
            if interaction_only:
                desc += " (仅交互项)。"
            else:
                desc += "。"
            return desc

        elif operation == "create_ratio_features":
            num_col = params.get('numerator_col', 'N/A')
            den_col = params.get('denominator_col', 'N/A')
            new_col = params.get('new_col_name', 'N/A')
            return f"通过计算 '{num_col}' 与 '{den_col}' 的比值来创建新特征 '{new_col}'。"

        elif operation == "no_action":
            return "评估后认为无需执行任何特征构造操作。"

        else:
            return f"执行一个自定义或未知操作: {operation}"

    def _execute_feature_engineering_plan(self, df: pd.DataFrame, plan: List[Dict[str, Any]]) -> Generator[
        Dict[str, Any], None, pd.DataFrame]:
        """
        根据制定的计划执行特征构造步骤，一个生成器，可以流式返回每一步的结果。
        """
        df_engineered = df.copy()
        current_stage = "特征工程"
        total_new_features = []

        yield {"type": "status_update", "payload": {
            "stage": current_stage, "status": "running",
            "detail": "正在执行特征构造计划..."
        }}

        for i, step_details in enumerate(plan):
            operation = step_details.get("operation")
            params = step_details.get("params", {})
            step_log = {"operation": operation, "params": params, "status": "failed"}

            # 使用新方法生成更详细、更具可读性的描述
            step_description = self._generate_step_description(operation, params)
            detail_message = f"执行特征构造计划 {i + 1}/{len(plan)}: {step_description}"

            yield {"type": "thinking_stream", "payload": detail_message}

            try:
                if operation == "no_action":
                    step_log["status"] = "no_action"
                elif operation in FEATURE_ENGINEERING_METHODS_MAP:
                    method_to_call = FEATURE_ENGINEERING_METHODS_MAP[operation]
                    original_cols = set(df_engineered.columns)

                    if operation == "create_polynomial_features":
                        df_engineered, fitted_obj = method_to_call(df_engineered, **params)
                        self.fitted_objects[f"poly_{'_'.join(params.get('columns', []))}"] = fitted_obj
                    else:
                        df_engineered = method_to_call(df_engineered, **params)

                    new_cols = list(set(df_engineered.columns) - original_cols)
                    step_log["status"] = "success"
                    step_log["new_features"] = new_cols
                    total_new_features.extend(new_cols)
                else:
                    step_log["error"] = f"Unknown operation: {operation}"

            except Exception as e:
                step_log["error"] = str(e)
                yield {"type": "status_update", "payload": {
                    "stage": current_stage, 'status': 'running',
                    "detail": f"执行步骤 '{operation}' 时出错: {e}"
                }}

            self.applied_steps.append(step_log)

        # 在所有步骤执行完毕后，提供一个总结性的成果
        if total_new_features:
            yield {"type": "substage_result", "payload": {
                "stage": current_stage,
                "substage_title": "特征构造执行成果",
                "data": f"特征工程执行完毕，共创建了 {len(total_new_features)} 个新特征: {'、'.join(total_new_features)}。"
            }}
        elif any(s.get("operation") != "no_action" for s in plan):
            yield {"type": "substage_result", "payload": {
                "stage": current_stage,
                "substage_title": "特征构造执行成果",
                "data": "特征工程计划已执行，但未生成新的特征列。"
            }}

        return df_engineered

    def generate_features(self, df: pd.DataFrame) -> Generator[
        Dict[str, Any], None, Tuple[DataFrame, List[Dict], Dict]]:
        """
        执行特征构造的主流程，现在是一个生成器。
        """
        current_stage = "特征工程"
        self.applied_steps = []
        self.fitted_objects = {}
        current_feature_names = [col for col in df.columns if col != self.target_metric]
        current_features_str = ", ".join(current_feature_names)

        yield {"type": "status_update", "payload": {
            "stage": current_stage, "status": "running",
            "detail": "正在分析数据并制定特征构造计划..."
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
                                               "error": "智能体调用失败"})
                    return df, self.applied_steps, self.fitted_objects
                yield chunk
            except StopIteration as e:
                llm_response_str = e.value
                break

        if "Agent failed" in llm_response_str:
            yield {"type": "error",
                   "payload": {"stage": current_stage, "detail": "特征构造计划制定失败：智能体返回错误。"}}
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
        except (json.JSONDecodeError, ValueError) as e:
            error_msg = f"生成的特征构造计划无效: {e}"
            yield {"type": "error", "payload": {"stage": current_stage, "detail": "特征构造计划制定失败：" + error_msg}}
            self.applied_steps.append(
                {"step": "llm_generate_fe_plan", "status": "failed", "error": str(e), "raw_response": llm_response_str})
            return df, self.applied_steps, self.fitted_objects

        if not feature_engineering_plan or (
                len(feature_engineering_plan) == 1 and feature_engineering_plan[0].get("operation") == "no_action"):
            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "特征构造计划评估",
                "data": "经智能体评估，当前特征已足够，无需构造新特征。"
            }}
        else:
            execution_generator = self._execute_feature_engineering_plan(df, feature_engineering_plan)
            df = yield from execution_generator

        return df, self.applied_steps, self.fitted_objects
