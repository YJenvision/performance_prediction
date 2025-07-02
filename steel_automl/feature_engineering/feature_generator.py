import numpy as np
import pandas as pd
import json
from typing import Dict, Any, Tuple, List, Union

from pandas import DataFrame

from llm_utils import call_llm
from knowledge_base.kb_service import KnowledgeBaseService
from config import FEATURE_ENGINEERING_KB_NAME, HISTORICAL_CASES_KB_NAME
from steel_automl.feature_engineering.methods import FEATURE_ENGINEERING_METHODS_MAP


class FeatureGenerator:
    """
    一个由LLM驱动的智能特征生成器。

    它通过以下步骤工作：
    1. 从知识库中检索与当前任务相关的特征工程知识。
    2. 让LLM根据用户需求、现有特征和检索到的知识，制定一个特征工程计划。
       这个计划的核心是让LLM动态地将理论公式中的元素映射到数据中的实际列名。
    3. 执行LLM生成的计划，创建新的特征。
    """

    def __init__(self, user_request: str, target_metric: str, preprocessed_columns: List[str]):
        """
        初始化特征生成器。

        参数:
        - user_request: 用户的原始自然语言建模请求。
        - target_metric: 目标性能指标的列名。
        - preprocessed_columns: 经过预处理后的特征列列表。
        - knowledge_base_data: 用于初始化特征工程知识库的数据。
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

        system_prompt = f"""
你是一位世界顶级的材料科学数据科学家，尤其擅长为机器学习模型设计和创造具有深刻物理意义的特征。你的任务是基于用户需求、当前数据特征和领域知识库，为回归任务制定一个详细、可执行的特征工程计划。

**决策依据:**
1.  **领域知识优先**: 优先使用知识库中提供的公式和方法，这些是经过验证的领域经验。
2.  **动态列名映射**: 你的核心任务之一是解决理论与现实的差距。知识库中的元素名（如 'C', 'Mn'）可能与数据集中的列名（如 'ELM_C', 'ELM_MN'）不完全一致。你必须利用知识库中提供的 `mapping_hints`参考 和你对所提供列名含义的理解，将公式中的 `elements` 准确映射到 `当前可用特征` 列表中的实际列名。如果某个元素在当前特征中找不到任何可能的匹配，你应该放弃使用需要该元素的公式，并在思考过程中说明原因。
3.  **通用性方法**: 除了领域知识，你也可以使用通用的特征工程方法或根据用户的提示来新增特征，如创建多项式特征或比率特征，但前提是这样做有明确的理由（例如，探索特征间的非线性关系或相互作用）。
4.  **用户优先级最高**: 对于用户明确的特征工程需求，你应该优先考虑这些需求。

**可用的特征工程操作:**
1.  `apply_knowledge_based_formula`: 应用知识库中的领域特定公式。这是首选操作。
    - `params`:
        - `formula_template`: str, 从知识库中获取的原始公式模板。
        - `new_feature_name`: str, 从知识库中获取的新特征名称。
        - `column_mapping`: Dict[str, str], 你完成的动态列名映射。Key是公式模板中的占位符（如 "C"），Value是数据集中实际的列名（如 "ELM_C"）。
2.  `create_polynomial_features`: 创建多项式和交互特征。
    - `params`:
        - `columns`: List[str], 需要进行操作的原始数值列名列表。
        - `degree`: int (可选, 默认为2)。
3.  `create_ratio_features`: 创建两个数值列的比率特征。
    - `params`:
        - `numerator_col`: str, 分子列名。
        - `denominator_col`: str, 分母列名。
        - `new_col_name`: str, 新特征的名称。
4.  `no_action`: 如果你认为现有特征已经足够，不需要任何新的特征工程。

**输出格式要求:**
你必须严格按照JSON格式返回一个操作列表。每个操作是一个包含 'operation' 和 'params' 的字典。不要在JSON前后添加任何解释性文字或代码块标记。

**示例JSON输出:**
[
  {{
    "operation": "apply_knowledge_based_formula",
    "params": {{
      "formula_template": "{{C}} + {{Mn}}/6 + ({{Cr}}+{{Mo}}+{{V}})/5 + ({{Ni}}+{{Cu}})/15",
      "new_feature_name": "CE",
      "column_mapping": {{
        "C": "ELM_C",
        "Mn": "ELM_MN",
        "Cr": "ELM_CR",
        "Mo": "ELM_MO",
        "V": "ELM_V",
        "Ni": "ELM_NI",
        "Cu": "ELM_CU"
      }}
    }}
  }},
  {{
    "operation": "create_ratio_features",
    "params": {{
      "numerator_col": "process_param1",
      "denominator_col": "process_param2",
      "new_col_name": "param1_div_param2"
    }}
  }}
]
"""

        user_prompt = f"""
请为以下任务和数据制定一个特征工程计划。

**用户原始请求**:
"{self.user_request}"

**目标预测指标**:
"{self.target_metric}"

**当前可用特征列表**:
{current_features_str}

**可供参考的领域知识库**:
{formatted_knowledge}

请仔细分析以上信息，特别是将知识库中的 `elements` 映射到 `当前可用特征列表`。然后，严格按照系统提示中要求的JSON列表格式，输出你的特征工程计划。
"""
        return system_prompt, user_prompt

    def _execute_feature_engineering_plan(self, df: pd.DataFrame, plan: List[Dict[str, Any]]) -> pd.DataFrame:
        """根据制定的计划执行特征工程步骤。"""
        df_engineered = df.copy()

        for i, step_details in enumerate(plan):
            operation = step_details.get("operation")
            params = step_details.get("params", {})
            print(f"\n执行步骤 {i + 1}: 操作='{operation}', 参数='{params}'")

            step_log = {"operation": operation, "params": params, "status": "failed"}
            try:
                if operation == "no_action":
                    print("=> 无特征工程操作。")
                    step_log["status"] = "no_action"
                elif operation in FEATURE_ENGINEERING_METHODS_MAP:
                    method_to_call = FEATURE_ENGINEERING_METHODS_MAP[operation]
                    # 特殊处理需要返回拟合对象的方法
                    if operation == "create_polynomial_features":
                        df_engineered, fitted_obj = method_to_call(df_engineered, **params)
                        self.fitted_objects[f"poly_{'_'.join(params.get('columns', []))}"] = fitted_obj
                    else:
                        df_engineered = method_to_call(df_engineered, **params)
                    step_log["status"] = "success"
                else:
                    print(f"警告: 未知的特征工程操作 '{operation}'。")
                    step_log["error"] = f"Unknown operation: {operation}"
            except Exception as e:
                print(f"执行操作 '{operation}' 时发生错误: {e}")
                step_log["error"] = str(e)

            self.applied_steps.append(step_log)
        return df_engineered

    def generate_features(self, df: pd.DataFrame) -> Tuple[DataFrame, List[Dict], Dict]:
        """
        执行特征工程的主流程。

        返回:
        - df_engineered: 经过特征工程的DataFrame。
        - applied_steps: 应用的特征工程步骤日志。
        - fitted_objects: 存储的拟合对象。
        """
        print("\n--- 开始智能特征工程 ---")
        self.applied_steps = []
        self.fitted_objects = {}

        current_feature_names = [col for col in df.columns if col != self.target_metric]
        current_features_str = ", ".join(current_feature_names)

        # 1. LLM制定特征工程计划
        print("步骤 1: 智能体 正在制定特征工程计划...")
        system_prompt, user_prompt = self._generate_llm_prompt_for_fe_plan(current_features_str)

        llm_response_str = call_llm(system_prompt, user_prompt)

        print(f"智能体 返回的计划 (原始字符串):\n{llm_response_str}")

        try:
            feature_engineering_plan = json.loads(llm_response_str)
            if not isinstance(feature_engineering_plan, list):
                raise ValueError("智能体响应不是一个列表。")
            self.applied_steps.append({
                "step": "llm_generate_fe_plan", "status": "success", "plan": feature_engineering_plan
            })
            print("=> 智能体特征工程计划已成功生成。")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"错误: LLM返回的计划无效: {e}")
            self.applied_steps.append({
                "step": "llm_generate_fe_plan", "status": "failed", "error": str(e), "raw_response": llm_response_str
            })
            return df, self.applied_steps, self.fitted_objects

        # 2. 执行特征工程计划
        if not feature_engineering_plan or (
                len(feature_engineering_plan) == 1 and feature_engineering_plan[0].get("operation") == "no_action"):
            print("\n步骤 2: 智能体建议不执行新的特征工程。")
        else:
            print("\n步骤 2: 开始执行特征工程计划...")
            df = self._execute_feature_engineering_plan(df, feature_engineering_plan)

        print("\n--- 特征工程完成 ---")
        return df, self.applied_steps, self.fitted_objects
