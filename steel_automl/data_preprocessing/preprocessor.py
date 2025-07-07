# @Time    : 2025/7/2
# @Author  : ZhangJingLiang
# @Email   : jinglianglink@qq.com
# @Project : performance_prediction_agent

import pandas as pd
import json
import numpy as np
import re
from typing import Dict, Any, Tuple, List
from llm_utils import call_llm
from knowledge_base.kb_service import KnowledgeBaseService
from config import PREPROCESSING_KB_NAME, PROFESSIONAL_KNOWLEDGE_KB_NAME
from steel_automl.data_preprocessing.methods import PREPROCESSING_METHODS_MAP
from steel_automl.utils.data_utils import generate_data_profile


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
        system_prompt = f"""
你是一名资深的钢铁行业数据科学家，当前正在执行AutoML流程中的特征粗筛任务。
你的目标是根据领域知识和用户的特殊要求，识别出在进入详细预处理步骤之前就应该被删除的特征列。

**决策规则:**
1.  **删除其他性能指标**: 数据中可能包含多个潜在的性能指标列。除了本次任务的唯一目标 `{self.target_metric}` 之外，所有其他的性能指标都应被删除，因为它们是标签而非特征。
2.  **删除不适用特征**: 删除那些根据领域知识不适合直接用于模型训练的特征，例如唯一ID、与目标无关的时间戳、或已知会引入噪声的高基数类别特征。
3.  **用户需求优先**: 这是最高优先级的规则。仔细分析用户的原始请求。如果用户明确要求“保留”或“使用”某个通常会被规则1或2删除的列，你必须遵守用户的指令，不要将其列入删除名单。

**输入信息:**
- 任务目标列 (绝不能删除): `{self.target_metric}`
- 用户的原始请求: `"{self.user_request}"`
- 领域知识库参考 (包含“性能指标”和“不适用特征”的列表)。

**输出格式要求:**
你必须返回一个且仅一个合法的JSON对象。该对象只有一个键 `columns_to_delete`，其值是一个包含所有根据上述规则决定删除的列名的列表。
如果没有任何列需要删除，请返回 `{{ "columns_to_delete": [] }}`。

**示例输出:**
`{{ "columns_to_delete": ["ST_NO", "SIGN_CODE", "SIGN_LINE_NO"] }}`

确保你的回复中不包含任何解释性文字、代码块标记或其他非JSON内容。
"""

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

    def _coarse_grained_feature_screening(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        """执行基于规则和知识的特征粗筛。"""
        steps_log = []
        df_screened = df.copy()

        # --- 步骤①: 删除常量列和全空列 ---
        print("  粗筛步骤①: 检查并删除常量列和全空列")
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
            print(f"    移除了 {len(rule_based_removed_cols)} 个常量或空列: {rule_based_removed_cols}")
        else:
            print("    未发现需要通过规则删除的常量或空列。")

        # --- 步骤② & ③: 基于领域知识和用户需求的特征删除 ---
        print("  粗筛步骤②&③: 基于知识库和用户需求，使用智能体进行特征筛选...")

        # 动态精简知识库信息
        print("    优化步骤: 动态精简知识库信息以减少上下文长度...")
        unsuitable_features_docs = self.domain_kb.search("经验意义上不适用作训练字段的特征列", k=1)
        unsuitable_features_info = "无相关知识"

        if unsuitable_features_docs and "metadata" in unsuitable_features_docs[0]:
            metadata = unsuitable_features_docs[0]["metadata"]
            if isinstance(metadata, dict) and 'fields' in metadata:
                pre_removed_set = set(rule_based_removed_cols)
                original_fields = metadata.get('fields', [])
                filtered_fields = [f for f in original_fields if f.get("field_code") not in pre_removed_set]
                if len(filtered_fields) < len(original_fields):
                    print(
                        f"      成功精简'不适用特征'知识: 从 {len(original_fields)} 个字段减少到 {len(filtered_fields)} 个。")
                    streamlined_metadata = metadata.copy()
                    streamlined_metadata['fields'] = filtered_fields
                    streamlined_metadata['description'] = metadata.get('description', '') + " (注意: 此知识列表已动态精简)。"
                    unsuitable_features_info = json.dumps(streamlined_metadata, indent=2, ensure_ascii=False)
                else:
                    unsuitable_features_info = json.dumps(metadata, indent=2, ensure_ascii=False)
            else:
                unsuitable_features_info = str(metadata)

        perf_metrics_docs = self.domain_kb.search("数据预处理过程时的目标性能指标字段列表", k=1)
        perf_metrics_info = perf_metrics_docs[0].get("metadata", "无相关知识") if perf_metrics_docs else "无相关知识"
        if isinstance(perf_metrics_info, dict):
            perf_metrics_info = json.dumps(perf_metrics_info, indent=2, ensure_ascii=False)

        current_columns = df_screened.columns.tolist()
        system_prompt, user_prompt = self._generate_prompt_for_screening(current_columns, str(perf_metrics_info),
                                                                         unsuitable_features_info)

        llm_response_str = call_llm(system_prompt, user_prompt)
        print(f"\n    智能体粗筛决策原始响应:\n{llm_response_str}")

        try:
            decision = json.loads(llm_response_str)
            cols_to_delete_by_llm = decision.get("columns_to_delete", [])

            if self.target_metric in cols_to_delete_by_llm:
                print(f"    警告: 智能体建议删除目标列 '{self.target_metric}'，此操作已被阻止。")
                cols_to_delete_by_llm.remove(self.target_metric)

            valid_cols_to_delete = [col for col in cols_to_delete_by_llm if col in df_screened.columns]

            if valid_cols_to_delete:
                df_screened.drop(columns=valid_cols_to_delete, inplace=True)
                steps_log.append({
                    "step": "特征粗筛-基于领域经验知识的智能体决策", "status": "success",
                    "details": "基于领域知识和用户需求，删除了其他性能指标和不适用特征。",
                    "removed_columns": valid_cols_to_delete, "reasoning_source": "智能体基于领域经验知识的决策。"
                })
                print(f"    智能体决策移除了 {len(valid_cols_to_delete)} 个特征: {valid_cols_to_delete}")
            else:
                print("    智能体决策未建议删除任何特征。")
                steps_log.append({
                    "step": "特征粗筛-智能体基于领域经验知识的决策", "status": "no_action",
                    "details": "分析后未发现需要基于知识库删除的列。"
                })
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"    错误: 解析智能体的粗筛决策失败: {e}。跳过此步骤。")
            steps_log.append({
                "step": "特征粗筛-基于领域经验知识的智能体决策", "status": "failed",
                "error": f"解析智能体响应失败： {e}", "raw_response": llm_response_str
            })

        return df_screened, steps_log

    def _generate_llm_prompt_for_preprocessing_plan(self, data_profile_str: str, columns_to_process: List[str]) -> \
            Tuple[str, str]:
        system_prompt = f"""
你是一位顶级的钢铁行业数据科学家，专门负责制定数据预处理策略。
你的任务是根据用户需求、数据画像和知识库经验，为每个待处理的特征列定义一个或多个预处理步骤。

**任务上下文:**
- 用户原始请求摘要: "{self.user_request}"，其具有最高优先级，你需要结合其意图调整策略。

**可用的预处理操作:**
1.  **缺失值处理** (存在缺失值的特征列均需处理):
    - 'delete_column': 删除该列 (缺失比例过高)。
    - 'delete_rows_with_missing_in_column': 删除在该列有缺失值的行 (缺失比例非常低，且样本充足时)。
    - 'impute_mean': 均值填充 (适用于接近正态分布的数值型特征)。
    - 'impute_median': 中位数填充 (适用于有偏分布或有离群值的数值型特征)。
    - 'impute_most_frequent': 众数填充。

2.  **类别特征编码** (dtype 为 'object', 'bool' 等非数值型的特征):
    - 'one_hot_encode_column': 独热编码。适用于低基数特征。
    - 'label_encode_column': 标签编码。适用于有序类别或树模型中的中低基数特征。
    - 'target_encode_column': 目标编码。适用于中高基数特征，能有效利用目标信息。
    - 'delete_column': 删除该列。适用于基数过高、噪音大或与目标无关的类别特征。

3.  **离群值处理** (数值型):
    - 'cap_outliers_iqr': 基于IQR进行封顶 (可选参数 'factor', 默认为1.5)。

4.  **其他操作**:
    - 'no_action': 不执行任何操作。

**决策逻辑:**
- **逻辑顺序**: 如果为单列制定多个操作，请务必遵循逻辑顺序：缺失值处理 -> 离群值处理 -> 编码。
- **缺失值处理**:
    - 查看 `missing_percentage`。
    - 高缺失率: 如果任一列的 `missing_percentage` 非常高，应优先决策 `delete_column`。
    - 比较 `stats.mean` 和 `stats.median`：若两者差异显著，暗示数据偏斜，应使用 `impute_median`；否则 `impute_mean` 是合理选择。
    - `type: 'empty'`: 必须决策 `delete_column`
- **离群值处理**:
    - 查看 `outlier_percentage`。若该值存在且显著，应添加 `cap_outliers_iqr` 操作。
- **类别特征编码策略**:
    - **极低基数 (cardinality <= 5)** 或 **布尔型 (type == 'binary')**: 优先使用 `label_encode_column`，因为它最高效。
    - **低基数 (5 < cardinality <= 10)**: 优先使用 `one_hot_encode_column`，这是最安全、最通用的方法，可以避免错误的顺序假设。
    - **中基数 (10 < cardinality <= 50)**: 这是一个权衡区。`target_encode_column` 是一个强大的选择，可以直接捕捉特征与目标的关系。同时，对于树模型算法，`label_encode_column` 也是一个可行的、更简单的备选方案。
    - **高基数 (cardinality > 50)**: 强烈推荐 `target_encode_column`，以避免`one_hot_encode_column`导致的维度爆炸。如果特征的业务意义不明确或可能引入大量噪音，`delete_column` 也是一个合理的防御性策略。

**输出格式要求:**
严格按照以下JSON格式返回你的计划。JSON对象中，每个键是列名，对应的值是一个**操作列表**。

**示例JSON输出:**
{{
  "feature_A": [
    {{ "operation": "impute_median" }},
    {{ "operation": "cap_outliers_iqr", "params": {{ "factor": 2.0 }} }}
  ],
  "feature_B_categorical": [
    {{ "operation": "impute_most_frequent" }},
    {{ "operation": "target_encode_column" }}
  ],
  "feature_C_to_drop": [
    {{ "operation": "delete_column" }}
  ]
}}

请确保你的回复是且仅是一个合法的JSON对象。不要包含任何解释性文字或代码块标记。
"""
        """详细数据预处理计划系统和用户提示词"""
        user_prompt = f"""
请为以下经过粗筛后的数据制定详细的预处理计划。

**数据画像:**
```json
{data_profile_str}
```

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
            "one_hot_encode_column", "label_encode_column", "target_encode_column",
            "cap_outliers_iqr", "no_action"
        ]

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
                        print(f"  处理列'{column}'，操作为'{operation}'")

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

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
        """数据预处理主流程，包含目标列清洗、特征粗筛和详细处理三个阶段。"""
        print("\n--- 开始数据预处理 ---")
        self.applied_steps = []
        self.fitted_objects = {}

        # 目标列清洗
        print("\n--- 预处理准备阶段: 清洗目标列 ---")

        # 删除目标列中包含空值或无效值（如0）的行
        initial_rows = len(df)

        # 子步骤 A: 删除空值行
        df.dropna(subset=[self.target_metric], inplace=True)
        rows_after_na_drop = len(df)
        dropped_na_count = initial_rows - rows_after_na_drop

        # 子步骤 B: 删除目标值为0的行。
        # 依赖具体业务。在某些目标性能预测任务中0是有效值。
        invalid_values_to_drop = [0]
        mask_invalid = df[self.target_metric].isin(invalid_values_to_drop)
        dropped_invalid_count = mask_invalid.sum()
        df = df[~mask_invalid]

        if dropped_na_count > 0 or dropped_invalid_count > 0:
            self.applied_steps.append({
                "step": "目标列清洗", "status": "success",
                "details": f"从目标列 '{self.target_metric}' 中移除了包含空值或指定无效值的行。",
                "removed_na_rows": int(dropped_na_count),
                "removed_invalid_rows": int(dropped_invalid_count),
                "remaining_rows": len(df)
            })

        if df.empty:
            error_msg = "清洗目标列后，数据集为空。预处理中止。"
            print(f"错误: {error_msg}")
            self.applied_steps.append({"step": "目标列清洗", "status": "failed", "error": error_msg})
            return pd.DataFrame(), self.applied_steps, self.fitted_objects

        # --- 阶段一: 基于规则和知识的特征粗筛 ---
        print("\n--- 预处理阶段一: 特征粗筛 ---")
        df_screened, screening_steps = self._coarse_grained_feature_screening(df)
        self.applied_steps.extend(screening_steps)

        if df_screened.empty or self.target_metric not in df_screened.columns:
            print("错误: 特征粗筛后数据为空或目标列被移除，预处理中止。")
            self.applied_steps.append({"error": "粗筛后数据为空或目标列被移除，预处理中止。"})
            return pd.DataFrame(), self.applied_steps, self.fitted_objects
        print("--- 特征粗筛完成 ---")

        # --- 阶段二: 对剩余特征进行详细预处理 ---
        print("\n--- 预处理阶段二: 详细预处理计划 ---")
        columns_to_process = [col for col in df_screened.columns if col != self.target_metric]
        if not columns_to_process:
            print("错误: 粗筛后没有特征列可供预处理，将直接返回数据。")
            self.applied_steps.append({"error": "粗筛后没有特征列可供预处理，预处理中止。"})
            return df_screened, self.applied_steps, self.fitted_objects

        print("步骤1: 为剩余特征生成优化的数据画像...")
        # 调用优化后的画像生成函数
        data_profile = generate_data_profile(df_screened, target_metric=self.target_metric)
        data_profile_str = json.dumps(data_profile, indent=2, ensure_ascii=False)
        self.applied_steps.append({"step": "生成数据画像", "status": "success", "data_profile": data_profile})
        print(f"数据画像：\n{data_profile_str}")

        print("步骤2: 智能体为剩余特征制定详细预处理计划...")
        system_prompt, user_prompt = self._generate_llm_prompt_for_preprocessing_plan(data_profile_str,
                                                                                      columns_to_process)

        max_retries = 3
        llm_plan = None
        for i in range(max_retries):
            llm_response_str = call_llm(system_prompt, user_prompt)
            print(f"\n智能体详细计划原始响应 (尝试 {i + 1}/{max_retries}):\n{llm_response_str}")
            try:
                llm_plan = json.loads(llm_response_str)
                if isinstance(llm_plan, dict):
                    break
                else:
                    llm_plan = None
            except json.JSONDecodeError as e:
                print(f"解析详细计划失败: {e}")
                if i == max_retries - 1:
                    self.applied_steps.append(
                        {"step": "生成详细预处理计划", "status": "failed", "error": f"智能体响应解析失败: {e}",
                         "raw_response": llm_response_str})
                    return df_screened, self.applied_steps, self.fitted_objects

        self.applied_steps.append({"step": "生成详细预处理计划", "status": "success", "plan": llm_plan})

        print("步骤3: 执行详细预处理计划...")
        df_processed = self._execute_preprocessing_plan(df_screened, llm_plan)

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
