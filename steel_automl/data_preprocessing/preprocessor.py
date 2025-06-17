import pandas as pd
import json
import numpy as np
from typing import Dict, Any, Tuple, List
from llm_utils import call_llm
from knowledge_base.kb_service import KnowledgeBaseService
from config import PREPROCESSING_KB_NAME
from steel_automl.data_preprocessing.methods import PREPROCESSING_METHODS_MAP
from steel_automl.utils.data_utils import generate_data_profile


class DataPreprocessor:
    def __init__(self, user_request: str, target_metric: str):
        """
        初始化数据预处理器。

        参数:
        - user_request: 用户的原始自然语言建模请求。
        - target_metric: 目标性能指标的列名。
        """
        self.user_request = user_request
        self.target_metric = target_metric
        self.kb_service = KnowledgeBaseService(PREPROCESSING_KB_NAME)
        self.applied_steps = []  # 记录应用的预处理步骤
        self.fitted_objects = {}  # 存储拟合的scaler, encoder等对象

    def _get_knowledge_snippets(self, context_query: str, k: int = 1) -> str:
        """从知识库检索片段并格式化为字符串。"""
        snippets = self.kb_service.search(context_query, k=k)
        if not snippets:
            return "未从知识库中检索到相关信息。"

        formatted_snippets = "\n知识库参考信息:\n"
        for i, snippet in enumerate(snippets):
            formatted_snippets += f"{i + 1}. {snippet.get('metadata', {}).get('strategy', '未知策略')}: {snippet.get('metadata', {}).get('reason', snippet.get('text_for_embedding', '无详细描述'))}\n"
        return formatted_snippets

    # """
    # 2.  数据缩放 (数值型):
    #     - 'standard_scale_column': 标准化 (Z-score)。
    #     - 'min_max_scale_column': 最小-最大缩放。

    # - 'custom_code': 提供Python代码片段来执行特定操作。
    # """
    def _generate_llm_prompt_for_preprocessing_plan(self, data_profile_str: str, columns_to_process: List[str]) -> \
            Tuple[str, str]:
        """为数据预处理计划生成系统和用户提示词。"""
        system_prompt = f"""
你是一位顶级的钢铁行业数据科学家，专门负责制定数据预处理策略。
你的任务是根据用户需求、数据画像和知识库经验，为每个待处理的特征列定义一个或多个预处理步骤。

用户原始请求摘要: "{self.user_request}"
目标性能指标: "{self.target_metric}" (此列通常不参与特征预处理)

可用的预处理操作:
1.  缺失值处理 (存在缺失值的特征列均需进行缺失值处理):
    - 'delete_column': 删除该列 (缺失比例过高时)。
    - 'delete_rows_with_missing_in_column': 删除在该列有缺失值的行 (缺失比例极低时)。
    - 'impute_mean': 均值填充 (数值型)。
    - 'impute_median': 中位数填充 (数值型)。
    - 'impute_most_frequent': 众数填充 (数值型或类别型)。
2.  类别特征编码 (所有非数值型特征(object类、bool类、allNull类等)均采取删除策略):
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
严格按照以下JSON格式返回你的计划。JSON对象中，每个键是列名，对应的值是一个**操作列表**。每个操作都是一个包含 'operation' 和可选 'params' 的字典。

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

        kb_query_context = f"数据预处理策略咨询：用户请求 '{self.user_request}', 目标指标 '{self.target_metric}', 待处理列 {columns_to_process}。请提供相关经验。"
        knowledge_snippets = self._get_knowledge_snippets(kb_query_context)

        user_prompt = f"""
请为以下数据制定预处理计划。

用户原始请求:
{self.user_request}

数据画像:
{data_profile_str}

知识库信息：
{knowledge_snippets}

请为以下列制定预处理计划: {', '.join(columns_to_process)}.
目标列 '{self.target_metric}' 通常不直接参与这些预处理步骤。
请严格按照系统提示中要求的JSON格式（每列对应一个操作列表）输出。
"""
        return system_prompt, user_prompt

    def _execute_preprocessing_plan(self, df: pd.DataFrame, plan: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        根据制定的计划，按照预设的阶段顺序执行预处理步骤。
        """
        df_processed = df.copy()

        # **优化点：定义一个固定的、逻辑正确的处理顺序**
        PROCESSING_ORDER = [
            # 阶段1：删除
            "delete_column",
            # 阶段2：插补
            "delete_rows_with_missing_in_column", "impute_mean", "impute_median", "impute_most_frequent",
            # 阶段3：自定义代码和转换
            "custom_code",
            # 阶段4：离群值处理
            "cap_outliers_iqr",
            # 阶段5：缩放
            "standard_scale_column", "min_max_scale_column",
            # (编码可以放在这里，如果添加的话)
            # "one_hot_encode_column",
            # 阶段6：无操作
            "no_action"
        ]

        # 按预设顺序遍历操作类型
        for operation_type in PROCESSING_ORDER:
            # 遍历计划中的每一列
            for column, steps in plan.items():
                # 检查列是否存在
                if column not in df_processed.columns:
                    continue

                # 查找当前列是否有匹配当前阶段的操作
                for step in steps:
                    if step.get("operation") == operation_type:
                        details = step
                        operation = details.get("operation")
                        params = details.get("params", {})

                        print(f"阶段 '{operation_type}': 处理列'{column}'，操作为'{operation}', 参数为'{params}'")

                        if column == self.target_metric:
                            print(f"警告: 计划对目标列 '{self.target_metric}' 执行操作 '{operation}'。跳过此操作。")
                            self.applied_steps.append({
                                "column": column, "operation": "跳过目标列的修改。", "details": operation
                            })
                            continue

                        step_details = {"column": column, "operation": operation, "params": params, "status": "failed"}

                        try:
                            if operation == "no_action":
                                step_details["status"] = "no_action"
                            elif operation == "custom_code":
                                code_snippet_str = params.get("code_snippet")
                                if code_snippet_str:
                                    # 安全警告: eval/exec存在风险，生产环境需要沙箱化。
                                    if "lambda" in code_snippet_str:
                                        custom_func = eval(code_snippet_str, {"pd": pd, "np": np})
                                        df_processed[column] = df_processed[column].apply(custom_func)
                                        step_details["status"] = "success"
                                    else:
                                        step_details["error"] = "Custom code execution skipped (not a simple lambda)."
                                else:
                                    step_details["error"] = "Custom code snippet is empty."
                            elif operation in PREPROCESSING_METHODS_MAP:
                                method_to_call = PREPROCESSING_METHODS_MAP[operation]

                                if operation in ["standard_scale_column", "min_max_scale_column",
                                                 "one_hot_encode_column"]:
                                    df_processed, fitted_obj = method_to_call(df_processed, column, **params)
                                    self.fitted_objects[column + "_" + operation] = fitted_obj
                                else:
                                    df_processed = method_to_call(df_processed, column, **params)
                                step_details["status"] = "success"
                            else:
                                print(f"警告: 未知的预处理操作 '{operation}' for column '{column}'。")
                                step_details["error"] = f"Unknown operation: {operation}"

                        except Exception as e:
                            print(f"处理列 '{column}' 时发生错误 (操作: {operation}): {e}")
                            step_details["error"] = str(e)

                        self.applied_steps.append(step_details)

        return df_processed

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]]:
        print("\n--- 开始数据预处理 ---")
        self.applied_steps = []
        self.fitted_objects = {}

        if df is None or df.empty:
            print("错误: 输入的DataFrame为空，无法进行预处理。")
            self.applied_steps.append({"error": "输入的 DataFrame 为空。"})
            return df, self.applied_steps, self.fitted_objects

        df = df.replace(r'^\s*$', np.nan, regex=True).replace(r'\(?null\)?', np.nan, regex=True).replace('nan', np.nan)
        self.applied_steps.append({
            "step": "替换空白和null字符串为NaN。",
            "status": "success"
        })

        constant_cols = [col for col in df.columns if df[col].dropna().nunique() <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            self.applied_steps.append({
                "step": "删除常量列或空列。",
                "status": "success",
                "removed_columns": constant_cols,
                "removal_reason": "特征列的取值为常量或空。"
            })
            if self.target_metric in constant_cols:
                print(f"错误: 目标列 '{self.target_metric}' 是常量列或全空列，无法进行建模。")
                self.applied_steps.append({"error": f"Target column {self.target_metric} is constant or empty."})
                return df, self.applied_steps, self.fitted_objects

        columns_to_process = [col for col in df.columns if col != self.target_metric]
        if not columns_to_process:
            print("警告: 没有特征列可供预处理。")
            self.applied_steps.append({"warning": "No feature columns to preprocess."})
            return df, self.applied_steps, self.fitted_objects

        print("步骤1: 生成数据画像...")
        data_profile = generate_data_profile(df[columns_to_process])
        data_profile_str = json.dumps(data_profile, indent=2, ensure_ascii=False)
        self.applied_steps.append({"step": "生成数据画像", "status": "success"})
        print(f"数据画像：\n{data_profile_str}")

        print("步骤2: 局部智能体制定预处理计划...")
        system_prompt, user_prompt = self._generate_llm_prompt_for_preprocessing_plan(data_profile_str,
                                                                                      columns_to_process)

        # 添加重试逻辑
        max_retries = 3
        retry_count = 0
        preprocessing_plan = None

        while retry_count < max_retries and preprocessing_plan is None:
            llm_response_str = call_llm(system_prompt, user_prompt)
            print(f"\n原始响应 (预处理计划) [尝试 {retry_count + 1}/{max_retries}]:\n{llm_response_str}")

            try:
                preprocessing_plan = json.loads(llm_response_str)
                if not isinstance(preprocessing_plan, dict):
                    raise ValueError("响应不是一个字典。")

                # 验证新的计划结构
                for col, steps in preprocessing_plan.items():
                    if not isinstance(steps, list):
                        raise ValueError(f"列 '{col}' 的值不是一个操作列表。")
                    for step in steps:
                        if not isinstance(step, dict) or "operation" not in step:
                            raise ValueError(f"列 '{col}' 的一个操作步骤格式不正确。")

            except (json.JSONDecodeError, ValueError) as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"预处理计划解析失败 (尝试 {retry_count}/{max_retries}), 将重试... 错误: {e}")
                    continue

                print(f"错误: 预处理计划解析失败: {e}")
                self.applied_steps.append(
                    {"step": "生成数据预处理计划。", "status": "failed", "error": str(e),
                     "raw_response": llm_response_str})
                return df, self.applied_steps, self.fitted_objects

        self.applied_steps.append({
            "step": "生成数据预处理计划。",
            "status": "success",
            "plan": preprocessing_plan,
            "retry_count": retry_count
        })
        print("LLM预处理计划已生成。")

        print("步骤3: 执行预处理计划...")
        df_processed = self._execute_preprocessing_plan(df, preprocessing_plan)

        if self.target_metric not in df_processed.columns and self.target_metric in df.columns:
            print(f"警告: 目标列 '{self.target_metric}' 在预处理后丢失，将从原始数据中恢复。")
            df_processed[self.target_metric] = df[self.target_metric]

        # 最终再检查一次常量列
        final_constant_cols = [col for col in df_processed.columns if
                               df_processed[col].dropna().nunique() <= 1 and col != self.target_metric]
        if final_constant_cols:
            df_processed = df_processed.drop(columns=final_constant_cols)
            self.applied_steps.append({
                "step": "再次删除常量列或空列。",
                "status": "success",
                "removed_columns": final_constant_cols,
                "removal_reason": "特征列在处理后变为常量或空。"
            })

        print("--- 数据预处理完成 ---")
        return df_processed, self.applied_steps, self.fitted_objects
