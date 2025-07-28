# @Time    : 2025/7/2
# @Author  : ZhangJingLiang
# @Email   : jinglianglink@qq.com
# @Project : performance_prediction_agent

import pandas as pd
import json
import numpy as np
import re
from typing import Dict, Any, Tuple, List, Generator

from llm_utils import call_llm
from knowledge_base.kb_service import KnowledgeBaseService
from config import PREPROCESSING_KB_NAME, PROFESSIONAL_KNOWLEDGE_KB_NAME
from steel_automl.data_preprocessing.methods import PREPROCESSING_METHODS_MAP
from steel_automl.utils.data_utils import generate_data_profile
from prompts.prompt_manager import get_prompt


class DataPreprocessor:
    def __init__(self, user_request: str, target_metric: str):
        """
        初始化数据预处理器。
        """
        self.user_request = user_request
        self.target_metric = target_metric
        # 加载两个知识库：一个用于通用的预处理策略，一个用于特定领域的业务知识
        self.preprocessing_kb = KnowledgeBaseService(PREPROCESSING_KB_NAME)
        self.domain_kb = KnowledgeBaseService(PROFESSIONAL_KNOWLEDGE_KB_NAME)
        self.applied_steps = []
        self.fitted_objects = {}

    def _generate_prompt_for_screening(self, all_columns: List[str], perf_metrics_info: str,
                                       unsuitable_features_info: str) -> Tuple[str, str]:
        """为第一阶段的特征粗筛提示词。"""
        # 从配置文件获取提示词并格式化
        system_prompt = get_prompt(
            'preprocessor.coarse_feature_screening.system',
            target_metric=self.target_metric,
            user_request=self.user_request
        )

        user_prompt = f"""
请根据以下信息，为数据集执行特征粗筛任务。

**当前数据集所有列:**
{json.dumps(all_columns, indent=2)}

**用户原始请求:**
"{self.user_request}"

**领域知识库参考:**
1.  **性能指标列表 (除目标列 `{self.target_metric}` 外，其余均应删除):**
{perf_metrics_info}
2.  **经验上不适用作训练的特征列表 (应删除):**
{unsuitable_features_info}

请严格遵循系统指令，分析以上所有信息，并以指定的JSON格式返回你的决策。
"""
        return system_prompt, user_prompt

    def _coarse_grained_feature_screening(self, df: pd.DataFrame) -> Generator[
        Dict[str, Any], None, Tuple[pd.DataFrame, List[Dict]]]:
        """执行基于规则和知识的特征粗筛, 现在是一个生成器。"""
        steps_log = []
        df_screened = df.copy()
        current_stage = "数据预处理"

        # 子任务1: 删除常量列和全空列
        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                    "detail": "特征粗筛：正在检查并删除常量列和全空列..."}}

        null_placeholders = [r'^\s*$', r'\(?null\)?', 'null', 'nan']
        for placeholder in null_placeholders:
            df_screened.replace(to_replace=placeholder, value=np.nan, regex=True, inplace=True)

        initial_cols = set(df_screened.columns)
        rule_based_removed_cols = []

        for col in df_screened.columns:
            if col == self.target_metric:
                continue
            if df_screened[col].isnull().all():
                rule_based_removed_cols.append(col)
                continue
            if df_screened[col].dropna().nunique() <= 1:
                rule_based_removed_cols.append(col)

        if rule_based_removed_cols:
            df_screened.drop(columns=rule_based_removed_cols, inplace=True)
            steps_log.append({
                "step": "特征粗筛-基于规则",
                "status": "success",
                "details": "删除了常量列、几乎全为常量或全为空的列。",
                "removed_columns": rule_based_removed_cols
            })

            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "特征粗筛(规则)",
                "data": f"移除了{len(rule_based_removed_cols)}个常量或空列：{rule_based_removed_cols}"
            }}

        # 子任务2: 基于领域知识和用户需求的特征删除
        detail = "正在基于用户需求使用知识库进行特征筛选..."
        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                    "detail": "特征粗筛：正在基于需求使用领域知识进行智能筛选..."}}

        # 动态精简知识库信息
        unsuitable_features_docs = self.domain_kb.search("经验意义上不适用作训练字段的特征列", k=1)
        unsuitable_features_info = "无相关知识"

        if unsuitable_features_docs and "metadata" in unsuitable_features_docs[0]:
            metadata = unsuitable_features_docs[0]["metadata"]
            if isinstance(metadata, dict) and 'fields' in metadata:
                pre_removed_set = set(rule_based_removed_cols)
                original_fields = metadata.get('fields', [])
                filtered_fields = [f for f in original_fields if f.get("field_code") not in pre_removed_set]
                streamlined_metadata = metadata.copy()
                streamlined_metadata['fields'] = filtered_fields
                streamlined_metadata['description'] = metadata.get('description', '') + " (注意: 此知识列表已动态精简)。"
                unsuitable_features_info = json.dumps(streamlined_metadata, indent=2, ensure_ascii=False)
            else:
                unsuitable_features_info = str(metadata)

        perf_metrics_docs = self.domain_kb.search("数据预处理过程时的目标性能指标字段列表", k=1)
        perf_metrics_info = perf_metrics_docs[0].get("metadata", "无相关知识") if perf_metrics_docs else "无相关知识"
        if isinstance(perf_metrics_info, dict):
            perf_metrics_info = json.dumps(perf_metrics_info, indent=2, ensure_ascii=False)

        current_columns = df_screened.columns.tolist()
        system_prompt, user_prompt = self._generate_prompt_for_screening(current_columns, str(perf_metrics_info),
                                                                         unsuitable_features_info)

        # 流式消费LLM调用
        llm_gen = call_llm(system_prompt, user_prompt, model="ds_v3")
        llm_response_str = ""
        while True:
            try:
                chunk = next(llm_gen)
                yield chunk  # 直接传递思考流和错误
            except StopIteration as e:
                llm_response_str = e.value
                break

        print(f"\n智能体粗筛决策原始响应:\n{llm_response_str}")

        try:
            decision = json.loads(llm_response_str)
            cols_to_delete_by_llm = decision.get("columns_to_delete", [])

            if self.target_metric in cols_to_delete_by_llm:
                print(f"警告: 智能体建议删除目标列 '{self.target_metric}'，此操作已被阻止。")
                cols_to_delete_by_llm.remove(self.target_metric)

            valid_cols_to_delete = [col for col in cols_to_delete_by_llm if col in df_screened.columns]

            if valid_cols_to_delete:
                df_screened.drop(columns=valid_cols_to_delete, inplace=True)
                steps_log.append({
                    "step": "特征粗筛-基于领域经验知识的智能体决策", "status": "success",
                    "details": "基于领域知识和用户需求，删除了其他性能指标和不适用特征。",
                    "removed_columns": valid_cols_to_delete, "reasoning_source": "智能体基于领域经验知识的决策。"
                })

                yield {"type": "substage_result", "payload": {
                    "stage": current_stage, "substage_title": "特征粗筛(智能体决策)",
                    "data": {f"基于需求使用领域知识移除了{len(valid_cols_to_delete)}个特征:{valid_cols_to_delete}"
                             }}}
            else:
                print("智能体决策未建议删除任何特征。")
                steps_log.append({
                    "step": "特征粗筛-智能体基于领域经验知识的决策", "status": "no_action",
                    "details": "分析后未发现需要基于知识库删除的列。"
                })
        except (json.JSONDecodeError, AttributeError) as e:
            error_msg = f"解析智能体的粗筛决策失败: {e}。跳过此步骤。"
            print(f"错误: {error_msg}")
            steps_log.append({
                "step": "特征粗筛-基于领域经验知识的智能体决策", "status": "failed",
                "error": error_msg, "raw_response": llm_response_str
            })
            yield {"type": "error",
                   "payload": {"stage": current_stage, "detail": error_msg}}

        return df_screened, steps_log

    def _generate_llm_prompt_for_preprocessing_plan(self, data_profile_str: str, columns_to_process: List[str]) -> \
            Tuple[str, str]:
        # 从配置文件获取提示词并格式化
        system_prompt = get_prompt(
            'preprocessor.detailed_preprocessing_plan.system',
            user_request=self.user_request
        )

        user_prompt = f"""
请为以下经过粗筛后的数据制定详细的预处理计划。

**数据画像:**
{data_profile_str}

请为以下列制定预处理计划: {json.dumps(columns_to_process)}。
目标列 '{self.target_metric}' 不应包含在你的计划中。
请严格按照系统提示中要求的JSON格式（每个特征列对应一个操作列表）输出。
"""
        return system_prompt, user_prompt

    def _execute_preprocessing_plan(self, df: pd.DataFrame, plan: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """根据制定的计划，执行预处理步骤"""
        df_processed = df.copy()
        PROCESSING_ORDER = [
            "delete_column", "delete_rows_with_missing_in_column",
            "impute_mean", "impute_median", "impute_most_frequent",
            "cap_outliers_iqr",
            "one_hot_encode_column", "label_encode_column", "target_encode_column",
            "no_action"
        ]

        if not isinstance(plan, dict):
            error_msg = f"预处理计划格式无效，期望是字典，但得到的是 {type(plan)}。跳过执行。"
            print(f"错误: {error_msg}")
            self.applied_steps.append({"step": "执行详细预处理计划", "status": "failed", "error": error_msg})
            return df_processed

        # 按列和操作顺序执行
        for operation_type in PROCESSING_ORDER:
            sorted_columns = sorted(plan.keys())
            for column in sorted_columns:
                if column not in df_processed.columns:
                    continue

                steps = plan.get(column, [])
                for step in steps:
                    if step.get("operation") == operation_type:
                        operation = step.get("operation")
                        params = step.get("params", {})
                        print(f"处理列'{column}'，操作为'{operation}'")

                        if column == self.target_metric and operation not in ['delete_rows_with_missing_in_column']:
                            continue

                        step_details = {"column": column, "operation": operation, "params": params, "status": "failed"}
                        try:
                            if operation == "no_action":
                                step_details["status"] = "no_action"
                            elif operation in PREPROCESSING_METHODS_MAP:
                                method_to_call = PREPROCESSING_METHODS_MAP[operation]
                                if operation == 'target_encode_column':
                                    params['target_metric_name'] = self.target_metric
                                result = method_to_call(df_processed, column, **params)
                                fitted_obj = None
                                if isinstance(result, tuple) and len(result) == 2:
                                    df_processed, fitted_obj = result
                                    if fitted_obj:
                                        object_key = f"{column}_{operation}"
                                        self.fitted_objects[object_key] = fitted_obj
                                        print(f"    -> 已拟合的对象 '{object_key}' 已保存。")
                                else:
                                    df_processed = result
                                step_details["status"] = "success"
                            else:
                                step_details["error"] = f"Unknown operation: {operation}"
                        except Exception as e:
                            step_details["error"] = str(e)
                            print(f"    -> 错误: 处理列'{column}'，操作'{operation}'失败: {e}")
                        self.applied_steps.append(step_details)
        return df_processed

    def preprocess_data(self, df: pd.DataFrame) -> Generator[
        Dict[str, Any], None, Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]]:
        """数据预处理主流程，现在是一个生成器。"""
        self.applied_steps = []
        self.fitted_objects = {}
        current_stage = "数据预处理"

        # 阶段1: 目标列清洗
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "正在清洗目标列..."}}

        initial_rows = len(df)
        df.dropna(subset=[self.target_metric], inplace=True)
        rows_after_na_drop = len(df)
        dropped_na_count = initial_rows - rows_after_na_drop

        invalid_values_to_drop = [0]
        mask_invalid = df[self.target_metric].isin(invalid_values_to_drop)
        dropped_invalid_count = mask_invalid.sum()
        df = df[~mask_invalid]

        if dropped_na_count > 0 or dropped_invalid_count > 0:
            self.applied_steps.append({
                "step": "目标列清洗", "status": "success",
                "details": f"从目标列'{self.target_metric}'中移除了包含空值或指定无效值的行。",
                "removed_na_rows": int(dropped_na_count),
                "removed_invalid_rows": int(dropped_invalid_count),
                "remaining_rows": len(df)
            })
            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "目标列清洗",
                "data": f"移除了 {dropped_na_count} 个空值行和 {dropped_invalid_count} 个无效值行。剩余 {len(df)} 行。"}
                   }

        if df.empty:
            error_msg = "清洗目标列后，数据集为空。预处理中止。"
            yield {"type": "error", "payload": {"stage": current_stage, "detail": error_msg}}
            self.applied_steps.append({"step": "目标列清洗", "status": "failed", "error": error_msg})
            return pd.DataFrame(), self.applied_steps, self.fitted_objects

        # 阶段2: 特征粗筛 (调用子生成器)
        screening_gen = self._coarse_grained_feature_screening(df)
        df_screened, screening_steps = yield from screening_gen

        self.applied_steps.extend(screening_steps)

        if df_screened.empty or self.target_metric not in df_screened.columns:
            error_msg = "特征粗筛后数据为空或目标列被移除，预处理中止。"
            yield {"type": "error", "payload": {"stage": current_stage, "detail": error_msg}}
            self.applied_steps.append({"step": "特征粗筛", "status": "failed", "error": error_msg})
            return pd.DataFrame(), self.applied_steps, self.fitted_objects

        # 阶段3: 详细预处理
        columns_to_process = [col for col in df_screened.columns if col != self.target_metric]
        if not columns_to_process:
            return df_screened, self.applied_steps, self.fitted_objects
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "正在生成数据画像..."}}

        data_profile = generate_data_profile(df_screened, target_metric=self.target_metric)
        data_profile_str = json.dumps(data_profile, indent=2, ensure_ascii=False)
        self.applied_steps.append({"step": "生成数据画像", "status": "success"})

        yield {"type": "substage_result", "payload": {
            "stage": current_stage, "substage_title": "数据画像",
            "data": data_profile
        }}

        detail = "正在制定详细预处理计划..."
        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running", "detail": detail}}
        system_prompt, user_prompt = self._generate_llm_prompt_for_preprocessing_plan(data_profile_str,
                                                                                      columns_to_process)

        max_retries = 3
        llm_plan = None
        for i in range(max_retries):
            llm_gen = call_llm(system_prompt, user_prompt, model="ds_v3")
            llm_response_str = ""
            while True:
                try:
                    chunk = next(llm_gen)
                    yield chunk
                except StopIteration as e:
                    llm_response_str = e.value
                    break

            print(f"\n智能体详细计划原始响应 (尝试 {i + 1}/{max_retries}):\n{llm_response_str}")
            try:
                llm_plan = json.loads(llm_response_str)
                if isinstance(llm_plan, dict):
                    self.applied_steps.append({"step": "生成详细预处理计划", "status": "success", "plan": llm_plan})
                    break
                else:
                    llm_plan = None
            except json.JSONDecodeError as e:
                print(f"解析详细计划失败: {e}")
                if i == max_retries - 1:
                    error_msg = f"智能体响应解析失败: {e}"
                    self.applied_steps.append(
                        {"step": "生成详细预处理计划", "status": "failed", "error": error_msg,
                         "raw_response": llm_response_str})
                    yield {"type": "error",
                           "payload": {"stage": current_stage, "detail": error_msg}}
                    return df_screened, self.applied_steps, self.fitted_objects

        yield {"type": "substage_result", "payload": {
            "stage": current_stage, "substage_title": "详细预处理计划",
            "data": llm_plan
        }}

        detail = "正在执行详细预处理计划..."
        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running", "detail": detail}}
        df_processed = self._execute_preprocessing_plan(df_screened.copy(), llm_plan)

        # 最终清理
        final_cols_to_drop = [col for col in df_processed.columns if
                              df_processed[col].dropna().nunique() <= 1 and col != self.target_metric]
        if final_cols_to_drop:
            df_processed = df_processed.drop(columns=final_cols_to_drop)
            print(f"最终清理数据预处理后变成常量的列: {final_cols_to_drop}")
            self.applied_steps.append(
                {"step": "最终清理预处理后变成常量的列", "status": "success", "removed_columns": final_cols_to_drop})

        print("--- 数据预处理完成 ---")
        return df_processed, self.applied_steps, self.fitted_objects
