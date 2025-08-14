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
from steel_automl.utils.data_utils import generate_data_profile, generate_iterative_profile
from prompts.prompt_manager import get_prompt


class DataPreprocessor:
    """
    数据预处理类，用于数据清洗、有效特征筛选和预处理流程。它结合了规则式处理、领域知识库和大语言模型生成预处理策略
    """

    def __init__(self, user_request: str, target_metric: str):
        """
        初始化数据探索与预处理器。
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

        system_prompt = get_prompt(
            'preprocessor.feature_screening.system',
            target_metric=self.target_metric,
            user_request=self.user_request
        )

        user_prompt = f"""
请根据以下信息，为数据集执行有效特征筛选任务。

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

    def _feature_screening(self, df: pd.DataFrame) -> Generator[
        Dict[str, Any], None, Tuple[pd.DataFrame, List[Dict]]]:
        """执行基于规则和经验知识的有效特征筛选, 生成器。"""
        steps_log = []
        df_screened = df.copy()
        current_stage = "数据探索与预处理"

        # 子任务1: 删除常量列和全空列
        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                    "detail": "有效特征筛选：正在检查并移除常量列和全空列..."}}

        null_placeholders = [r'^\s*$', r'\(?null\)?', 'null', 'nan']
        for placeholder in null_placeholders:
            df_screened.replace(to_replace=placeholder, value=np.nan, regex=True, inplace=True)

        initial_cols = set(df_screened.columns)
        rule_based_removed_cols = []

        for col in df_screened.columns:
            if col == self.target_metric:
                continue  # 跳过目标列
            if df_screened[col].isnull().all():  # 全空列
                rule_based_removed_cols.append(col)
                continue
            if df_screened[col].dropna().nunique() <= 1:  # 去除空值（NaN）后，剩余值的唯一值数量 ≤ 1 的列
                rule_based_removed_cols.append(col)

        if rule_based_removed_cols:
            df_screened.drop(columns=rule_based_removed_cols, inplace=True)
            steps_log.append({
                "step": "基于规则的有效特征筛选",
                "status": "success",
                "details": "移除常量列和全空列。",
                "removed_columns": rule_based_removed_cols
            })

            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "基于规则的有效特征筛选",
                "data": f"移除了{len(rule_based_removed_cols)}个常量或空列：{rule_based_removed_cols}"
            }}

        # 子任务2: 基于经验知识和需求的有效特征筛选
        detail = "正在基于需求使用经验知识进行特征筛选..."
        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                    "detail": "有效特征筛选：正在基于需求使用经验知识进行特征筛选..."}}

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

        llm_gen = call_llm(system_prompt, user_prompt, model="ds_v3")
        llm_response_str = ""
        while True:
            try:
                chunk = next(llm_gen)
                yield chunk
            except StopIteration as e:
                llm_response_str = e.value
                break

        try:
            decision = json.loads(llm_response_str)
            cols_to_delete_by_llm = decision.get("columns_to_delete", [])

            if self.target_metric in cols_to_delete_by_llm:  # 强制阻止智能体建议删除目标列的操作
                cols_to_delete_by_llm.remove(self.target_metric)

            valid_cols_to_delete = [col for col in cols_to_delete_by_llm if col in df_screened.columns]

            if valid_cols_to_delete:
                df_screened.drop(columns=valid_cols_to_delete, inplace=True)
                steps_log.append({
                    "step": "基于需求和经验知识的有效特征动态筛选", "status": "success",
                    "details": "基于需求和经验知识删除了其他性能指标和不适用特征。",
                    "removed_columns": valid_cols_to_delete
                })

                yield {"type": "substage_result", "payload": {
                    "stage": current_stage, "substage_title": "基于需求和经验知识的有效特征动态筛选",
                    "data": f"基于需求使用经验知识动态移除了{len(valid_cols_to_delete)}个特征:{valid_cols_to_delete}"
                }}
            else:
                steps_log.append({
                    "step": "基于需求和经验知识的有效特征动态筛选", "status": "no_action",
                    "details": "分析后未发现需要基于经验知识动态删除的列。"
                })
        except (json.JSONDecodeError, AttributeError) as e:
            error_msg = f"解析智能体的基于需求和经验知识的有效特征动态筛选决策失败: {e}。跳过此步骤。"
            print(f"错误: {error_msg}")
            steps_log.append({
                "step": "基于需求和经验知识的有效特征动态筛选", "status": "failed",
                "error": error_msg, "raw_response": llm_response_str
            })
            yield {"type": "error",
                   "payload": {"stage": current_stage, "detail": error_msg}}

        return df_screened, steps_log

    def _decide_row_pruning_threshold(self, iter_profile: Dict[str, Any]) -> float:
        """
        让 LLM 在严格边界内“只选阈值”；失败时回退到启发式。
        """
        system_prompt = get_prompt(
            'preprocessor.row_pruning.system',
            user_request=self.user_request
        )

        user_prompt = f"""
你将基于下面的数据样本清洗画像，**仅选择一个行缺失比例阈值**，超过该阈值的样本行应当删除。

数据样本清洗画像（行分布与Top缺失列摘要）：
{json.dumps(iter_profile, indent=2, ensure_ascii=False)}

**请输出 JSON，形如：**
{{"row_drop_threshold": 0.6, "second_pass": false, "reason": "..." }}
- 只允许阈值 ∈ [0.3, 0.9]，最多保留两位小数。
- 若需要两次删行（更保守地先大后小），可将 second_pass 设为 true（我会在第二轮再询问你阈值）。
"""
        llm_gen = call_llm(system_prompt, user_prompt, model="ds_v3")
        llm_resp = ""
        while True:
            try:
                chunk = next(llm_gen)
                yield chunk  # 允许生成器向外传事件
            except StopIteration as e:
                llm_resp = e.value
                break

        try:
            print(llm_resp)
            obj = json.loads(llm_resp)
            th = float(obj.get("row_drop_threshold", 0.6))
            th = max(0.3, min(0.9, round(th, 2)))   # 阈值范围强制约束
            second = bool(obj.get("second_pass", False))
            return th, second
        except Exception:
            # 回退：基于分位点的启发式
            rp = iter_profile.get("row_profile", {})
            p90 = float(rp.get("p90", 0.0))
            p95 = float(rp.get("p95", 0.0))
            # 若尾部很重，选更激进阈值，否则温和
            if p95 >= 0.7:
                return 0.7, False
            if p90 >= 0.6:
                return 0.6, False
            return 0.5, False

    def _drop_rows_by_ratio(self, df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        feats = [c for c in df.columns if c != self.target_metric]
        if not feats:
            return df, {"removed": 0, "threshold": threshold}
        r = df[feats].isna().mean(axis=1).astype(float)
        mask = r > threshold
        removed = int(mask.sum())
        df2 = df.loc[~mask].copy()
        return df2, {"removed": removed, "threshold": threshold}

    def _generate_llm_prompt_for_preprocessing_plan(self, data_profile_str: str, columns_to_process: List[str]) -> \
            Tuple[str, str]:
        # 从配置文件获取提示词并格式化
        system_prompt = get_prompt(
            'preprocessor.detailed_preprocessing_plan.system',
            user_request=self.user_request
        )

        user_prompt = f"""
现在，请为以下经过有效特征筛选后的数据制定详细的数据预处理计划。

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
            self.applied_steps.append({"step": "执行数据预处理计划", "status": "failed", "error": error_msg})
            return df_processed

        # 按列和操作顺序执行
        for operation_type in PROCESSING_ORDER:  # 按顺序执行操作类型
            sorted_columns = sorted(plan.keys())  # 按列名排序，确保确定性
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
                                result = method_to_call(df_processed, column, **params)  # 执行操作
                                fitted_obj = None
                                if isinstance(result, tuple) and len(result) == 2:  # 若返回(数据, 拟合对象)
                                    df_processed, fitted_obj = result
                                    if fitted_obj:
                                        object_key = f"{column}_{operation}"
                                        self.fitted_objects[object_key] = fitted_obj  # 保存拟合对象（如编码器）
                                        print(f"已拟合的对象 '{object_key}' 已保存。")
                                else:
                                    df_processed = result
                                step_details["status"] = "success"
                            else:
                                step_details["error"] = f"Unknown operation: {operation}"
                        except Exception as e:
                            step_details["error"] = str(e)
                            print(f"错误: 处理列'{column}'，操作'{operation}'失败: {e}")
                        self.applied_steps.append(step_details)
        return df_processed

    def preprocess_data(self, df: pd.DataFrame) -> Generator[
        Dict[str, Any], None, Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]]:
        """数据探索与预处理主流程，生成器。"""
        self.applied_steps = []
        self.fitted_objects = {}
        current_stage = "数据探索与预处理"

        # 阶段1: 目标列清洗
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "正在检视并清洗目标性能列..."}}

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
                "step": "目标性能列的检视和清洗", "status": "success",
                "details": f"从目标性能列'{self.target_metric}'中移除了包含空值或指定无效值（0）的行。",
                "removed_na_rows": int(dropped_na_count),
                "removed_invalid_rows": int(dropped_invalid_count),
                "remaining_rows": len(df)
            })

            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "目标性能列的检视和清洗",
                "data": f"移除了 {dropped_na_count} 个{self.target_metric}为空值的行和 {dropped_invalid_count} 个{self.target_metric}为无效值（0）的行。剩余 {len(df)} 行。"}
                   }

        yield {"type": "substage_result",
               "payload": {"stage": current_stage, "substage_title": "目标性能列的检视和清洗",
                           "data": f"目标性能列'{self.target_metric}'无需清洗。列值均有效。"}
               }

        if df.empty:
            error_msg = "清洗目标性能列后，数据集为空。预处理中止。"
            yield {"type": "error", "payload": {"stage": current_stage, "detail": error_msg}}
            self.applied_steps.append({"step": "目标性能列的检视和清洗", "status": "failed", "error": error_msg})
            return pd.DataFrame(), self.applied_steps, self.fitted_objects

        # 阶段2: 有效特征筛选
        screening_gen = self._feature_screening(df)
        df_screened, screening_steps = yield from screening_gen
        self.applied_steps.extend(screening_steps)

        if df_screened.empty or self.target_metric not in df_screened.columns:
            err = "有效特征筛选后数据为空或目标列被移除。"
            yield {"type": "error", "payload": {"stage": current_stage, "detail": err}}
            self.applied_steps.append({"step": "有效特征筛选", "status": "failed", "error": err})
            return pd.DataFrame(), self.applied_steps, self.fitted_objects

        # 阶段3: 数据样本清洗
        # 第1轮：行缺失阈值选择与删行（可能1到2次）
        df_iter = df_screened.copy()

        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                    "detail": "正在生成数据样本清洗画像..."}}
        iter_profile = generate_iterative_profile(df_iter, target_metric=self.target_metric, prev_profile=None)
        self.applied_steps.append({"step": "生成数据样本清洗画像", "status": "success", "profile": iter_profile})
        yield {"type": "substage_result", "payload": {
            "stage": current_stage, "substage_title": "数据样本清洗画像", "data": iter_profile
        }}

        # 让LLM选阈值（含回退）
        th, second_pass = None, False
        decider = self._decide_row_pruning_threshold(iter_profile)
        while True:
            try:
                chunk = next(decider)
                yield chunk
            except StopIteration as e:
                th, second_pass = e.value if isinstance(e.value, tuple) else (0.6, False)
                break

        df_iter, meta = self._drop_rows_by_ratio(df_iter, th)
        self.applied_steps.append({"step": "数据样本清洗", "status": "success", **meta})
        yield {"type": "substage_result", "payload": {
            "stage": current_stage, "substage_title": "数据样本清洗",
            "data": {"threshold": th, "removed_rows": meta["removed"]}
        }}

        if second_pass:
            # 第2轮更保守的删行：重新画像再询问
            iter_profile = generate_iterative_profile(df_iter, target_metric=self.target_metric,
                                                      prev_profile=iter_profile)
            self.applied_steps.append({"step": "生成数据样本清洗画像#2", "status": "success", "profile": iter_profile})
            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "数据样本清洗画像#2", "data": iter_profile
            }}
            decider2 = self._decide_row_pruning_threshold(iter_profile)
            while True:
                try:
                    chunk = next(decider2)
                    yield chunk
                except StopIteration as e:
                    th2, _ = e.value if isinstance(e.value, tuple) else (th, False)
                    break
            df_iter, meta2 = self._drop_rows_by_ratio(df_iter, th2)
            self.applied_steps.append({"step": "数据样本清洗#2", "status": "success", **meta2})
            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "数据样本清洗#2",
                "data": {"threshold": th2, "removed_rows": meta2["removed"]}
            }}

        # 阶段4: 数据预处理
        columns_to_process = [col for col in df_iter.columns if col != self.target_metric]
        if not columns_to_process:
            return df_iter, self.applied_steps, self.fitted_objects
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "正在生成数据画像..."}}

        data_profile = generate_data_profile(df_iter, target_metric=self.target_metric)
        data_profile_str = json.dumps(data_profile, indent=2, ensure_ascii=False)
        self.applied_steps.append({"step": "生成数据画像", "status": "success"})
        print("数据画像:", data_profile_str)
        yield {"type": "substage_result", "payload": {
            "stage": current_stage, "substage_title": "数据画像",
            "data": data_profile_str
        }}

        detail = "正在制定数据预处理计划..."
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

            try:
                llm_plan = json.loads(llm_response_str)
                if isinstance(llm_plan, dict):
                    self.applied_steps.append({"step": "生成数据预处理计划", "status": "success", "plan": llm_plan})
                    break
                else:
                    llm_plan = None
            except json.JSONDecodeError as e:
                if i == max_retries - 1:
                    error_msg = f"智能体响应解析失败: {e}"
                    self.applied_steps.append(
                        {"step": "生成数据预处理计划", "status": "failed", "error": error_msg,
                         "raw_response": llm_response_str})
                    yield {"type": "error",
                           "payload": {"stage": current_stage, "detail": error_msg}}
                    return df_iter, self.applied_steps, self.fitted_objects

        yield {"type": "substage_result", "payload": {
            "stage": current_stage, "substage_title": "数据预处理计划",
            "data": llm_plan
        }}

        detail = "正在执行数据预处理计划..."
        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running", "detail": detail}}
        df_processed = self._execute_preprocessing_plan(df_iter.copy(), llm_plan)

        # 后处理
        detail = "正在检视数据并进行后处理过程..."
        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running", "detail": detail}}
        final_cols_to_drop = [col for col in df_processed.columns if
                              df_processed[col].dropna().nunique() <= 1 and col != self.target_metric]
        if final_cols_to_drop:
            df_processed = df_processed.drop(columns=final_cols_to_drop)
            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "去除预处理计划后变成常量的特征",
                "data": final_cols_to_drop
            }}
            self.applied_steps.append(
                {"step": "去除预处理计划后变成常量的特征", "status": "success", "removed_columns": final_cols_to_drop})

        yield {"type": "substage_result", "payload": {
            "stage": current_stage, "substage_title": "检视后无需进行后处理过程"
        }}

        return df_processed, self.applied_steps, self.fitted_objects
