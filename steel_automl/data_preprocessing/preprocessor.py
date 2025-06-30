import pandas as pd
import json
import numpy as np
import re
from typing import Dict, Any, Tuple, List
# 假设 llm_utils, config, methods, data_utils 文件在项目路径下
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
`{{ "columns_to_delete": ["ST_NO", "SM_TM", "TS_N"] }}`

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
        print("  粗筛步骤①: 检查并删除常量列和全空列...")
        # 统一处理各种形式的空值
        null_placeholders = [r'^\s*$', r'\(?null\)?', 'null', 'nan']
        for placeholder in null_placeholders:
            df_screened.replace(to_replace=placeholder, value=np.nan, regex=True, inplace=True)

        initial_cols = set(df_screened.columns)
        cols_to_remove = []

        for col in df_screened.columns:
            # 跳过目标列
            if col == self.target_metric:
                continue

            # 删除全为空的列
            if df_screened[col].isnull().all():
                cols_to_remove.append(col)
                continue

            # 删除只有一个唯一值（非空）或全部值相同的列
            if df_screened[col].dropna().nunique() <= 1:
                cols_to_remove.append(col)

        if cols_to_remove:
            df_screened.drop(columns=cols_to_remove, inplace=True)
            steps_log.append({
                "step": "特征粗筛-规则删除",
                "status": "success",
                "details": "删除了常量列、几乎全为常量或全为空的列。",
                "removed_columns": cols_to_remove
            })
            print(f"    移除了 {len(cols_to_remove)} 个常量或空列: {cols_to_remove}")
        else:
            print("    未发现需要通过规则删除的常量或空列。")

        # --- 步骤② & ③: 基于领域知识和用户需求的特征删除 ---
        print("  粗筛步骤②&③: 基于知识库和用户需求，使用智能体进行特征筛选...")

        # 从业务数据库知识库检索信息
        perf_metrics_docs = self.domain_kb.search("数据预处理过程时的目标性能指标字段列表", k=1)
        unsuitable_features_docs = self.domain_kb.search("经验意义上不适用作训练字段的特征列", k=1)
        perf_metrics_info = perf_metrics_docs[0].get("metadata", "")
        unsuitable_features_info = unsuitable_features_docs[0].get("metadata", "")
        current_columns = df_screened.columns.tolist()
        system_prompt, user_prompt = self._generate_prompt_for_screening(current_columns, perf_metrics_info,
                                                                         unsuitable_features_info)

        llm_response_str = call_llm(system_prompt, user_prompt)
        print(f"\n    智能体粗筛决策原始响应:\n{llm_response_str}")

        try:
            decision = json.loads(llm_response_str)
            cols_to_delete_by_llm = decision.get("columns_to_delete", [])

            # 确保目标列不会被意外删除
            if self.target_metric in cols_to_delete_by_llm:
                print(f"    警告: LLM建议删除目标列 '{self.target_metric}'，此操作已被阻止。")
                cols_to_delete_by_llm.remove(self.target_metric)

            valid_cols_to_delete = [col for col in cols_to_delete_by_llm if col in df_screened.columns]

            if valid_cols_to_delete:
                df_screened.drop(columns=valid_cols_to_delete, inplace=True)
                steps_log.append({
                    "step": "特征粗筛-智能体决策删除",
                    "status": "success",
                    "details": "基于领域知识和用户需求，删除了其他性能指标和不适用特征。",
                    "removed_columns": valid_cols_to_delete,
                    "reasoning_source": "LLM with Domain Knowledge"
                })
                print(f"    LLM决策移除了 {len(valid_cols_to_delete)} 个特征: {valid_cols_to_delete}")
            else:
                print("    LLM决策未建议删除任何特征。")
                steps_log.append({
                    "step": "特征粗筛-智能体决策删除",
                    "status": "no_action",
                    "details": "智能体分析后未发现需要基于知识库删除的列。"
                })

        except (json.JSONDecodeError, AttributeError) as e:
            print(f"    错误: 解析智能体的粗筛决策失败: {e}。跳过此步骤。")
            steps_log.append({
                "step": "特征粗筛-智能体决策删除",
                "status": "failed",
                "error": f"Failed to parse Agent response: {e}",
                "raw_response": llm_response_str
            })

        return df_screened, steps_log

    def _generate_llm_prompt_for_preprocessing_plan(self, data_profile_str: str, columns_to_process: List[str]) -> \
            Tuple[str, str]:
        """数据预处理计划系统和用户提示词"""
        system_prompt = f"""
你是一位顶级的钢铁行业数据科学家，专门负责制定数据预处理策略。
你的任务是根据用户需求、数据画像和知识库经验，为每个待处理的特征列定义一个或多个预处理步骤。

用户原始请求摘要: "{self.user_request}"
目标性能指标: "{self.target_metric}" (此列通常不参与特征预处理)

可用的预处理操作:
1.  缺失值处理 (存在缺失值的特征列均需进行缺失值处理):
    - 'delete_column': 删除该列 (缺失比例过高时)。
    - 'delete_rows_with_missing_in_column': 删除在该列有缺失值的行 (缺失比例非常低时)。
    - 'impute_mean': 均值填充 (数值型)。
    - 'impute_median': 中位数填充 (数值型)。
    - 'impute_most_frequent': 众数填充 (数值型或类别型)。
2.  类别特征编码 (所有非数值型特征(object类、bool类等)均采取删除策略):
    - 'delete_column': 删除该列。
3.  离群值处理 (数值型):
    - 'cap_outliers_iqr': 基于IQR进行封顶 (可选参数 'factor', 默认为1.5)。
4.  其他操作:
    - 'no_action': 不执行任何操作。

决策逻辑:
- **多步骤处理**: 一个列可以应用多个操作，例如可以先填充缺失值，再进行离群值处理。
- **操作顺序**: 在为单列制定计划时，如果设计多个操作，请务必遵循逻辑顺序：缺失值处理 -> 离群值处理。
- **数据画像**: 仔细分析数据类型、缺失比例、唯一值和分布来选择最合适的操作组合。
- **知识库**: 优先采纳知识库中的成功策略。

输出格式要求:
严格按照以下JSON格式返回你的计划。JSON对象中，每个键是列名，对应的值是一个**操作列表**。

示例JSON输出:
{{
  "column_name1": [
    {{ "operation": "impute_median" }},
    {{ "operation": "standard_scale_column" }}
  ],
  "column_name2": [
    {{ 
      "operation": "cap_outliers_iqr", 
      "params": {{ "factor": 2.0 }}
    }},
    {{ "operation": "min_max_scale_column" }}
  ],
  "column_name3": [
    {{ "operation": "delete_column" }}
  ],
  "column_name4": [
    {{ "operation": "no_action" }}
  ]
}}

请确保你的回复是且仅是一个合法的JSON对象。不要包含任何解释性文字或代码块标记。
"""
        user_prompt = f"""
请为以下经过粗筛后的数据制定详细的预处理计划。

数据画像:
{data_profile_str}

请为以下列制定预处理计划: {', '.join(columns_to_process)}.
目标列 '{self.target_metric}' 不参与预处理。
请严格按照系统提示中要求的JSON格式（每个特征列对应一个操作列表）输出。
"""
        return system_prompt, user_prompt

    def _execute_preprocessing_plan(self, df: pd.DataFrame, plan: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """根据制定的计划，执行预处理步骤。(此部分保持不变)"""
        df_processed = df.copy()
        # 优化点：定义一个固定的、逻辑正确的处理顺序
        PROCESSING_ORDER = [
            # 阶段1：删除
            "delete_column",
            # 阶段2：插补
            "delete_rows_with_missing_in_column", "impute_mean", "impute_median", "impute_most_frequent",
            # 阶段3：离群值处理
            "cap_outliers_iqr",
            # 阶段4：缩放
            "standard_scale_column", "min_max_scale_column",
            # (编码可以放在这里，如果添加的话)
            # "one_hot_encode_column",
            # 阶段6：无操作
            "no_action"
        ]
        for operation_type in PROCESSING_ORDER:
            for column, steps in plan.items():
                if column not in df_processed.columns: continue
                for step in steps:
                    if step.get("operation") == operation_type:
                        # ... (原有执行细节不变)
                        operation = step.get("operation")
                        params = step.get("params", {})
                        print(f"  处理列'{column}'，操作为'{operation}'")
                        if column == self.target_metric: continue
                        step_details = {"column": column, "operation": operation, "params": params, "status": "failed"}
                        try:
                            if operation == "no_action":
                                step_details["status"] = "no_action"
                            elif operation in PREPROCESSING_METHODS_MAP:
                                method_to_call = PREPROCESSING_METHODS_MAP[operation]
                                df_processed = method_to_call(df_processed, column, **params)
                                step_details["status"] = "success"
                            else:
                                step_details["error"] = f"Unknown operation: {operation}"
                        except Exception as e:
                            step_details["error"] = str(e)
                        self.applied_steps.append(step_details)
        return df_processed

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
        """
        数据预处理主流程，包含特征粗筛和详细处理两个阶段。
        """
        print("\n--- 开始数据预处理 ---")
        self.applied_steps = []
        self.fitted_objects = {}

        if df is None or df.empty:
            print("错误: 输入的DataFrame为空，无法进行预处理。")
            self.applied_steps.append({"error": "输入的 DataFrame 为空。"})
            return pd.DataFrame(), self.applied_steps, self.fitted_objects

        # --- 阶段一: 基于规则和知识的特征粗筛 ---
        print("\n--- 预处理阶段一: 特征粗筛 ---")
        df_screened, screening_steps = self._coarse_grained_feature_screening(df)
        self.applied_steps.extend(screening_steps)

        if df_screened.empty or self.target_metric not in df_screened.columns:
            print("错误: 特征粗筛后数据为空或目标列被移除，预处理中止。")
            self.applied_steps.append({"error": "粗筛后数据无效。"})
            return pd.DataFrame(), self.applied_steps, self.fitted_objects
        print("--- 特征粗筛完成 ---")

        # --- 阶段二: 对剩余特征进行详细预处理 ---
        print("\n--- 预处理阶段二: 详细预处理计划 ---")
        columns_to_process = [col for col in df_screened.columns if col != self.target_metric]
        if not columns_to_process:
            print("警告: 粗筛后没有特征列可供预处理，将直接返回数据。")
            self.applied_steps.append({"warning": "No feature columns left after screening."})
            return df_screened, self.applied_steps, self.fitted_objects

        print("步骤1: 为剩余特征生成数据画像...")
        data_profile = generate_data_profile(df_screened[columns_to_process])
        data_profile_str = json.dumps(data_profile, indent=2, ensure_ascii=False)
        self.applied_steps.append({"step": "生成数据画像", "status": "success"})
        print(f"数据画像：\n{data_profile_str}")

        print("步骤2: 局部智能体为剩余特征制定详细预处理计划...")
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
                    break  # 成功解析则跳出循环
                else:
                    llm_plan = None
            except json.JSONDecodeError as e:
                print(f"解析详细计划失败: {e}")
                if i == max_retries - 1:
                    self.applied_steps.append(
                        {"step": "生成详细预处理计划", "status": "failed", "error": f"解析失败: {e}"})
                    return df_screened, self.applied_steps, self.fitted_objects

        if not llm_plan:
            print("错误: 无法获取有效的详细预处理计划。")
            return df_screened, self.applied_steps, self.fitted_objects

        self.applied_steps.append({"step": "生成详细预处理计划", "status": "success", "plan": llm_plan})

        print("步骤3: 执行详细预处理计划...")
        df_processed = self._execute_preprocessing_plan(df_screened, llm_plan)

        print("--- 数据预处理完成 ---")
        return df_processed, self.applied_steps, self.fitted_objects
