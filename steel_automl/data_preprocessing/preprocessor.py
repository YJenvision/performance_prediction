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
# 导入用于保存数据帧的工具函数
from steel_automl.performance_model_builder_utils import _save_dataframe


class DataPreprocessor:
    """
    数据预处理类，用于数据清洗、有效特征初步筛选和预处理流程。它结合了规则式处理、领域知识库和大语言模型生成预处理策略
    采用多阶段、迭代式的方法，结合规则、领域知识和LLM进行智能决策。
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
        self.fitted_objects = {}
        self.PROCESSING_ORDER = [
            # 删除操作最先
            "delete_column", "delete_rows_with_missing_in_column",
            # 缺失值指示器与填充
            "add_missing_indicator", "impute_auto", "impute_mean", "impute_median", "impute_most_frequent",
            # 异常值与变换
            "winsorize_by_quantile", "cap_outliers_iqr", "yeo_johnson_transform_column",
            # 类别编码
            "rare_label_collapse", "frequency_encode_column", "target_encode_column", "one_hot_encode_column",
            "label_encode_column",
            # 数值缩放
            "robust_scale_column", "standard_scale_column", "min_max_scale_column",
            # 空操作
            "no_action"
        ]

    def _call_llm_for_plan(self, system_prompt: str, user_prompt: str, stage_name: str) -> Generator[
        Dict[str, Any], None, Tuple[Dict[str, Any], Dict[str, Any]]]:
        """一个通用的LLM调用和解析的生成器函数, 返回计划和日志条目。"""
        max_retries = 3
        llm_plan = None
        log_entry = {}
        llm_response_str = ""
        for i in range(max_retries):
            llm_gen = call_llm(system_prompt, user_prompt, model="ds_v3")
            try:
                # 消费子生成器并向外传递消息
                llm_response_str = yield from llm_gen
            except StopIteration as e:
                llm_response_str = e.value

            try:
                # 尝试解析响应
                parsed_json = json.loads(llm_response_str)
                if isinstance(parsed_json, dict):
                    llm_plan = parsed_json
                    log_entry = {"step": stage_name, "status": "success", "plan": llm_plan}
                    break  # 成功解析并验证格式后退出循环
                else:
                    raise json.JSONDecodeError("LLM response is not a dictionary.", llm_response_str, 0)
            except json.JSONDecodeError as e:
                if i == max_retries - 1:
                    error_msg = f"在第 {i + 1} 次尝试后，智能体响应解析失败: {e}"
                    log_entry = {"step": stage_name, "status": "failed", "error": error_msg,
                                 "raw_response": llm_response_str}
                    yield {"type": "error", "payload": {"stage": "数据探索与预处理", "detail": error_msg}}
                    # 在最终失败时返回一个空的plan和失败日志
                    return {}, log_entry

        # 如果循环正常结束（因为break），llm_plan将包含有效计划
        return llm_plan if llm_plan is not None else {}, log_entry

    def _generate_prompt_for_screening(self, all_columns: List[str], perf_metrics_info: str,
                                       unsuitable_features_info: str) -> Tuple[str, str]:

        system_prompt = get_prompt(
            'preprocessor.feature_screening.system',
            target_metric=self.target_metric,
            user_request=self.user_request
        )

        user_prompt = f"""
请根据以下信息，为数据集执行有效特征初步筛选任务。

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
        """执行基于规则和经验知识的有效特征初步筛选, 生成器。"""
        steps_log = []
        df_screened = df.copy()
        current_stage = "数据探索与预处理"

        # 子任务1: 删除常量列和全空列
        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                    "detail": "有效特征初步筛选：正在检查并移除常量特征和全空特征..."}}

        yield {"type": "thinking_stream",
               "payload": "首先我将对数据集中的所有空取值进行标准化处理，将其中的nan、null、(null)、单个或多个空字符串等表示空情形的数据进行统一化，随后我将检查并移除常量特征和全空特征..."}
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
                "step": "基于规则的有效特征初步筛选",
                "status": "success",
                "details": "移除常量特征和全空特征。",
                "removed_columns": rule_based_removed_cols
            })

            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "基于规则的有效特征初步筛选成果",
                "data": f"移除了 {len(rule_based_removed_cols)} 个常量特征或空特征：{rule_based_removed_cols}"
            }}

        # 子任务2: 基于经验知识和需求的有效特征初步筛选
        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                    "detail": "有效特征初步筛选：正在基于需求使用经验知识进行特征筛选..."}}

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
                    "stage": current_stage, "substage_title": "基于需求和经验知识的有效特征动态筛选成果",
                    "data": f"基于需求使用经验知识动态移除了 {len(valid_cols_to_delete)} 个特征：{valid_cols_to_delete}"
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

    def _decide_row_pruning_threshold(self, iter_profile: Dict[str, Any]) -> Generator[
        Dict[str, Any], None, Tuple[float, bool]]:
        """
        智能体在严格边界内“只选阈值”；失败时回退到启发式。
        """
        system_prompt = get_prompt(
            'preprocessor.row_pruning.system',
            user_request=self.user_request
        )

        user_prompt = f"""
我的建模需求是：{self.user_request}
你将基于下面的数据样本清洗画像，**仅选择一个行缺失比例阈值**，超过该阈值的样本行应当删除。

数据样本清洗画像（行分布与Top缺失列摘要）：
{json.dumps(iter_profile, indent=2, ensure_ascii=False)}
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
            th = max(0.3, min(0.9, round(th, 2)))  # 阈值范围强制约束
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

    def _execute_plan(self, df: pd.DataFrame, plan: Dict[str, List[Dict[str, Any]]], stage_name: str) -> Generator[
        Dict[str, Any], None, Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
        """根据制定的计划，按预设顺序执行预处理步骤，并流式返回执行结果和日志。"""
        df_processed = df.copy()
        execution_logs = []

        if not isinstance(plan, dict):
            error_msg = f"预处理计划格式无效（应为字典），跳过执行。计划内容: {plan}"
            execution_logs.append({"step": stage_name, "status": "failed", "error": error_msg})
            return df_processed, execution_logs
        yield {'type': 'status_update',
               'payload': {'stage': '数据探索与预处理', 'status': 'running',
                           'detail': f'正在执行{stage_name}策略...'}}

        # 按预定义的顺序遍历所有可能的操作
        for operation_type in self.PROCESSING_ORDER:
            # 遍历所有列，执行当前类型的操作
            for column, steps in plan.items():
                if column not in df_processed.columns and operation_type not in ['one_hot_encode_column',
                                                                                 'delete_column']:
                    continue

                for step in steps:
                    if step.get("operation") == operation_type:
                        params = step.get("params", {})

                        step_details = {"column": column, "operation": operation_type, "params": params,
                                        "status": "failed"}
                        try:
                            method_to_call = PREPROCESSING_METHODS_MAP[operation_type]

                            # 为target_encode注入目标列名
                            if operation_type == 'target_encode_column':
                                params['target_metric_name'] = self.target_metric

                            result = method_to_call(df_processed, column, **params)

                            result_obj = None
                            if isinstance(result, tuple) and len(result) == 2:
                                df_processed, result_obj = result
                                # 检查返回的是否为执行结果消息
                                if isinstance(result_obj, dict) and 'message' in result_obj:
                                    # 根据阶段名称决定返回格式
                                    if stage_name == "数据精加工":
                                        yield {
                                            'type': 'thinking_stream',
                                            'payload': result_obj['message']}
                                    else:
                                        yield {
                                            'type': 'thinking_stream',
                                            'payload': result_obj['message']}
                                    # 如果结果对象只包含消息，则它不是一个拟合对象
                                    if list(result_obj.keys()) == ['message']:
                                        result_obj = None

                                # 存储拟合对象
                                if result_obj is not None:
                                    object_key = f"{column}_{operation_type}"
                                    self.fitted_objects[object_key] = result_obj

                            else:
                                df_processed = result

                            step_details["status"] = "success"
                        except Exception as e:
                            step_details["error"] = str(e)
                            print(f"错误: 在阶段'{stage_name}'处理列'{column}'，操作'{operation_type}'失败: {e}")

                        execution_logs.append(step_details)
        return df_processed, execution_logs

    def preprocess_data(self, df: pd.DataFrame, run_specific_dir: str) -> Generator[
        Dict[str, Any], None, Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], str]]:
        """数据探索与预处理主流程，生成器。"""
        self.step_details = {}
        self.fitted_objects = {}
        current_stage = "数据探索与预处理"

        # 阶段1: 目标列清洗
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "正在检视目标性能特征列..."}}

        initial_rows = len(df)
        df.dropna(subset=[self.target_metric], inplace=True)
        rows_after_na_drop = len(df)
        dropped_na_count = initial_rows - rows_after_na_drop

        invalid_values_to_drop = [0]
        mask_invalid = df[self.target_metric].isin(invalid_values_to_drop)
        dropped_invalid_count = mask_invalid.sum()
        df = df[~mask_invalid]

        target_cleaning_log = []
        if dropped_na_count > 0 or dropped_invalid_count > 0:
            target_cleaning_log.append({
                "step": "目标性能特征列的检视和清洗", "status": "success",
                "details": f"从目标性能特征列'{self.target_metric}'中移除了包含空值或指定无效值（0）的行。",
                "removed_na_rows": int(dropped_na_count),
                "removed_invalid_rows": int(dropped_invalid_count),
                "remaining_rows": len(df)
            })
            yield {"type": "thinking_stream",
                   "payload": f"经过检视，已经移除了 {dropped_na_count} 个 {self.target_metric} 为空值的数据样本和 {dropped_invalid_count} 个 {self.target_metric} 为无效值（0）的数据样本。剩余数据样本 {len(df)} 个。"}
        else:
            yield {"type": "thinking_stream",
                   "payload": f"经过检视，所有数据样本的目标性能特征列 '{self.target_metric}' 取值均有效，无需清洗。"}
        self.step_details["目标列清洗"] = target_cleaning_log

        if df.empty:
            error_msg = "清洗目标性能列后，数据集为空。预处理中止。"
            yield {"type": "error", "payload": {"stage": current_stage, "detail": error_msg}}
            self.step_details["目标列清洗"].append({"step": "目标性能列的检视和清洗", "status": "failed", "error": error_msg})
            return pd.DataFrame(), self.step_details, self.fitted_objects, ""

        # 阶段2: 有效特征初步筛选
        df_screened, screening_steps = yield from self._feature_screening(df)
        self.step_details["有效特征初步筛选"] = screening_steps

        # 保存经过有效特征初步筛选后的数据集
        _save_dataframe(df_screened, "#2经过有效特征初步筛选后的数据集", run_specific_dir)

        if df_screened.empty or self.target_metric not in df_screened.columns:
            err = "有效特征初步筛选后数据为空或目标列被移除。"
            yield {"type": "error", "payload": {"stage": current_stage, "detail": err}}
            self.step_details["有效特征初步筛选"].append({"step": "有效特征初步筛选", "status": "failed", "error": err})
            return pd.DataFrame(), self.step_details, self.fitted_objects, ""

        # 阶段3: 数据样本清洗
        df_iter = df_screened.copy()
        row_cleaning_steps = []

        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                    "detail": "正在检视数据样本并进行清洗..."}}
        iter_profile = generate_iterative_profile(df_iter, target_metric=self.target_metric)
        row_cleaning_steps.append({"step": "生成数据样本清洗画像", "status": "success", "profile": iter_profile})

        # 让LLM选阈值（含回退）
        th, second_pass = yield from self._decide_row_pruning_threshold(iter_profile)

        df_iter, meta = self._drop_rows_by_ratio(df_iter, th)
        row_cleaning_steps.append({"step": "数据样本清洗", "status": "success", **meta})
        yield {"type": "substage_result", "payload": {
            "stage": current_stage, "substage_title": "数据样本清洗成果",
            "data": f"根据数据样本的检视情况，为了平衡数据保留和清洗效果，选择阈值 {th} 筛除了 {meta['removed']} 个相对高特征缺失率样本。"
        }}

        if second_pass:
            iter_profile_2 = generate_iterative_profile(df_iter, target_metric=self.target_metric)
            row_cleaning_steps.append({"step": "生成数据样本清洗画像#2", "status": "success", "profile": iter_profile_2})
            th2, _ = yield from self._decide_row_pruning_threshold(iter_profile_2)
            df_iter, meta2 = self._drop_rows_by_ratio(df_iter, th2)
            row_cleaning_steps.append({"step": "数据样本清洗#2", "status": "success", **meta2})
            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "数据样本清洗#2成果",
                "data": f"根据最新一个版本的数据样本检视情况，本轮数据样本清洗选择阈值 {th2} 筛除了 {meta2['removed']} 个相对高特征缺失率样本。"
            }}
        self.step_details["数据样本清洗"] = row_cleaning_steps
        df_current = df_iter.copy()

        # 保存经过数据样本清洗后的数据集
        _save_dataframe(df_current, "#3经过数据样本清洗后的数据集", run_specific_dir)

        # 阶段4: 缺失值处理
        missing_value_steps = {}
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "正在进行缺失值处理策略制定..."}}
        profile_after_cleaning = generate_data_profile(df_current, self.target_metric)

        sys_prompt = get_prompt('preprocessor.missing_value_plan.system', user_request=self.user_request)
        user_prompt = f"""请为以下数据画像制定缺失值处理计划。
**数据画像:**
{json.dumps(profile_after_cleaning, indent=2, ensure_ascii=False)}"""

        missing_plan, plan_log = yield from self._call_llm_for_plan(sys_prompt, user_prompt, "生成缺失值处理计划")
        missing_value_steps["计划制定"] = plan_log

        if missing_plan:
            df_current, execution_logs = yield from self._execute_plan(df_current, missing_plan, "缺失值处理")
            missing_value_steps["计划执行"] = execution_logs
        self.step_details["缺失值处理"] = missing_value_steps

        # 保存经过缺失值处理后的数据集
        _save_dataframe(df_current, "#4经过缺失值处理后的数据集", run_specific_dir)

        # 阶段5: 数据精加工 (异常/变换/编码/缩放)
        final_processing_steps = {}
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "正在进行数据精加工策略制定..."}}
        profile_after_imputation = generate_data_profile(df_current, self.target_metric)

        sys_prompt = get_prompt('preprocessor.final_processing_plan.system', user_request=self.user_request)
        user_prompt = f"""请为以下数据画像制定精加工计划。
**用户的原始建模需求**：{self.user_request}
**数据画像:**
{json.dumps(profile_after_imputation, indent=2, ensure_ascii=False)}"""

        final_plan, final_plan_log = yield from self._call_llm_for_plan(sys_prompt, user_prompt, "生成数据精加工计划")
        final_processing_steps["计划制定"] = final_plan_log

        if final_plan:
            df_current, execution_logs = yield from self._execute_plan(df_current, final_plan, "数据精加工")
            final_processing_steps["计划执行"] = execution_logs
        self.step_details["数据精加工"] = final_processing_steps

        # 后处理
        post_processing_steps = []
        detail = "正在检视数据并进行后处理过程..."
        yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running", "detail": detail}}
        final_cols_to_drop = [col for col in df_current.columns if
                              df_current[col].dropna().nunique() <= 1 and col != self.target_metric]
        if final_cols_to_drop:
            df_current = df_current.drop(columns=final_cols_to_drop)
            yield {"type": "thinking_stream", "payload": "我要检查并且筛除执行预处理计划之后变成常量的特征列..."}
            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "后处理过程成果",
                "data": f"后处理过程执行完毕，共去除 {len(final_cols_to_drop)} 个变成常量的特征列。它们是：{'、'.join(final_cols_to_drop)}"
            }}
            post_processing_steps.append(
                {"step": "去除预处理计划后变成常量的特征", "status": "success", "removed_columns": final_cols_to_drop})
        else:
            yield {"type": "thinking_stream", "payload": "我要检查并且筛除执行预处理计划之后变成常量的特征列..."}
            yield {"type": "thinking_stream", "payload": "没有发现变成常量的特征列。"}
        self.step_details["后处理"] = post_processing_steps

        # 保存最终预处理后的数据集
        final_data_path = _save_dataframe(df_current, "#5经过数据预处理后的数据集", run_specific_dir)

        return df_current, self.step_details, self.fitted_objects, final_data_path
