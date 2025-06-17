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
        # self.history_kb_service = KnowledgeBaseService(HISTORICAL_CASES_KB_NAME)  # 用于参考历史案例
        self.applied_steps = []
        self.fitted_objects = {}  # 例如 PolynomialFeatures 转换器

    def _get_knowledge_snippets(self, context_query: str, k: int = 1) -> str:
        """从特征工程和历史案例知识库检索信息。"""
        fe_snippets = self.fe_kb_service.search(context_query, k=k)
        # history_snippets = self.history_kb_service.search(context_query, k=k)

        formatted_snippets = "\n特征工程知识库参考:\n"
        if fe_snippets:
            for i, snippet in enumerate(fe_snippets):
                formatted_snippets += f"{i + 1}. {snippet.get('metadata', {}).get('feature_name', '未知特征')}: {snippet.get('metadata', {}).get('type', snippet.get('text_for_embedding', '无详细描述'))}\n"
        else:
            formatted_snippets += "未从特征工程知识库中检索到相关信息。\n"

        # if history_snippets:
        #     formatted_snippets += "\n历史案例参考:\n"
        #     for i, snippet in enumerate(history_snippets):
        #         formatted_snippets += f"{i+1}. {snippet.get('metadata', {}).get('case_id', '未知案例')}: {snippet.get('metadata', {}).get('summary', '无描述')}\n"
        # else:
        #     formatted_snippets += "未从历史案例知识库中检索到相关信息。\n"

        return formatted_snippets

    def _generate_llm_prompt_for_fe_plan(self, current_features_str: str) -> Tuple[str, str]:
        """为特征工程计划生成系统和用户提示词。"""
        """
        6.  'custom_code_feature': 如果标准方法不适用，你可以提供Python代码片段来生成新特征。代码片段应能接收整个DataFrame并返回一个包含新特征的Series或DataFrame。新特征的列名应在 'params' 的 'new_column_names' (List[str]) 中指定。
        """
        system_prompt = f"""
你是一位顶级的钢铁材料科学家和数据特征工程师，精通从现有数据中挖掘和创造有价值的预测特征。
你的任务是根据用户需求、当前可用的特征集、以及从钢铁行业特征工程知识库中检索到的经验，提出一个特征工程计划。

用户原始请求摘要: "{self.user_request}"
目标性能指标: "{self.target_metric}"
当前可用特征: {current_features_str}

可用的特征工程操作包括:
1.  'create_polynomial_features': 创建多项式和交互特征。
    参数:
    - 'columns': List[str], 需要进行操作的原始数值列名列表。
    - 'degree': int (可选, 默认为2), 多项式的次数。
    - 'interaction_only': bool (可选, 默认为False), 是否只生成交互项。
2.  'create_ratio_features': 创建两个数值列的比率特征。
    参数:
    - 'numerator_col': str, 分子列名。
    - 'denominator_col': str, 分母列名。
    - 'new_col_name': str, 生成的新特征的名称。
    - 'epsilon': float (可选, 默认为1e-6), 防止除零的小数。
3.  'apply_custom_domain_formula': 应用领域特定的公式。
    参数:
    - 'formula_name': str, 预定义的公式名称 (例如 "carbon_equivalent_ceq", "hall_petch_relation")。
    - 'params': Dict[str, Any], 公式所需的参数，通常是原始列名映射到公式中的变量名 (例如 {{ "C_col": "feature_C", "Mn_col": "feature_Mn" }} for CEQ)。
4.  'select_features_by_importance': (此阶段暂不实现，但LLM可以建议，后续由模型选择或专门的特征选择模块处理) 基于模型重要性选择特征。
5.  'no_action': 不执行任何新的特征工程。

决策依据:
- 领域知识: 考虑钢铁冶金、材料科学的原理，生成有物理意义或经验证明有效的特征。
- 数据特性: 基于当前特征的类型和分布。
- 知识库信息: 优先考虑知识库中提到的有效特征构造方法。
- 用户偏好: 遵守用户请求中关于特征工程的任何指示。

输出格式要求:
严格按照以下JSON格式返回你的计划。这是一个操作列表，每个操作是一个包含 'operation' 和 'params' 的字典。

示例JSON输出:
[
  {{
    "operation": "create_polynomial_features",
    "params": {{ "columns": ["feature_A", "feature_B"], "degree": 2 }}
  }},
  {{
    "operation": "create_ratio_features",
    "params": {{ "numerator_col": "process_param1", "denominator_col": "process_param2", "new_col_name": "param1_div_param2" }}
  }},
  {{
    "operation": "apply_custom_domain_formula",
    "params": {{ "formula_name": "carbon_equivalent_ceq", "params": {{ "C_col": "element_C", "Mn_col": "element_Mn" }} }}
  }}
]
如果不需要特征工程，则返回空列表 `[]` 或包含 `{{ "operation": "no_action" }}` 的列表。
请确保你的回复是且仅是一个合法的JSON列表。不要包含任何解释性文字或代码块标记。
"""
        kb_query_context = f"特征工程策略咨询：用户请求 '{self.user_request}', 目标指标 '{self.target_metric}', 当前特征 {current_features_str}。请提供相关特征构造经验。"
        knowledge_snippets = self._get_knowledge_snippets(kb_query_context)

        user_prompt = f"""
请为以下数据和任务制定特征工程计划。

用户原始请求:
{self.user_request}

当前可用特征列表:
{current_features_str}

{knowledge_snippets}

请严格按照系统提示中要求的JSON列表格式输出你的特征工程操作计划。
如果现有特征已足够，可以建议 "no_action" 或返回空列表。
"""
        return system_prompt, user_prompt

    def _execute_feature_engineering_plan(self, df: pd.DataFrame, plan: List[Dict[str, Any]]) -> pd.DataFrame:
        """根据LLM制定的计划执行特征工程步骤。"""
        df_engineered = df.copy()

        for i, step_details_dict in enumerate(plan):
            operation = step_details_dict.get("operation")
            params = step_details_dict.get("params", {})

            print(f"执行特征工程步骤 {i + 1}: 操作='{operation}', 参数='{params}'")
            step_log = {"operation": operation, "params": params, "status": "failed"}

            try:
                if operation == "no_action":
                    print("无特征工程操作。")
                    step_log["status"] = "no_action"
                elif operation == "custom_code_feature":
                    code_snippet_str = params.get("code_snippet")
                    new_col_names = params.get("new_column_names")  # 应该是列表
                    if code_snippet_str and new_col_names and isinstance(new_col_names, list) and len(
                            new_col_names) > 0:
                        try:
                            # IMPORTANT: Executing arbitrary code from LLM is a security risk.
                            if "lambda df:" in code_snippet_str:  # 简单检查是否为lambda df: ...
                                custom_func = eval(code_snippet_str, {"pd": pd, "np": np,
                                                                      "df_copy": df_engineered.copy()})  # 允许使用pd, np, 和df的副本
                                new_feature_series_or_df = custom_func(df_engineered.copy())  # 传递DataFrame副本

                                if isinstance(new_feature_series_or_df, pd.Series):
                                    if len(new_col_names) == 1:
                                        df_engineered[new_col_names[0]] = new_feature_series_or_df
                                        print(f"通过自定义代码生成了新特征: {new_col_names[0]}")
                                        step_log["status"] = "success"
                                    else:
                                        step_log[
                                            "error"] = "Custom code returned a Series, but multiple new_column_names were specified."
                                        print(f"错误: 自定义代码返回Series，但指定了多个新列名: {new_col_names}")
                                elif isinstance(new_feature_series_or_df, pd.DataFrame):
                                    # 确保返回的DataFrame的列名与new_col_names匹配或可以对应
                                    if len(new_col_names) == len(new_feature_series_or_df.columns):
                                        renamed_cols = dict(zip(new_feature_series_or_df.columns, new_col_names))
                                        new_feature_df_renamed = new_feature_series_or_df.rename(columns=renamed_cols)
                                        for col_name in new_col_names:
                                            df_engineered[col_name] = new_feature_df_renamed[col_name]
                                        print(f"通过自定义代码生成了新特征: {new_col_names}")
                                        step_log["status"] = "success"
                                    else:
                                        step_log[
                                            "error"] = "Mismatch between custom code DataFrame columns and specified new_column_names."
                                        print(f"错误: 自定义代码返回DataFrame的列数与指定新列名数量不符。")
                                else:
                                    step_log["error"] = "Custom code did not return a Pandas Series or DataFrame."
                                    print(f"错误: 自定义代码未返回Series或DataFrame。")
                            else:
                                print(
                                    f"警告: 自定义特征代码不是预期的 'lambda df:' 格式，执行被跳过。代码: {code_snippet_str}")
                                step_log["error"] = "Custom code execution skipped (not a 'lambda df:' expression)."
                        except Exception as e:
                            print(f"执行自定义特征代码失败: {e}")
                            step_log["error"] = str(e)
                    else:
                        print(f"警告: 自定义特征代码片段或新列名为空/格式不正确。")
                        step_log["error"] = "Custom code snippet or new_column_names are invalid."

                elif operation in FEATURE_ENGINEERING_METHODS_MAP:
                    method_to_call = FEATURE_ENGINEERING_METHODS_MAP[operation]
                    if operation == "create_polynomial_features":
                        df_engineered, fitted_obj = method_to_call(df_engineered, **params)
                        self.fitted_objects[f"polynomial_features_{params.get('columns', ['all'])}"] = fitted_obj
                    else:  # 其他方法只返回DataFrame
                        df_engineered = method_to_call(df_engineered, **params)
                    step_log["status"] = "success"
                else:
                    print(f"警告: 未知的特征工程操作 '{operation}'。")
                    step_log["error"] = f"Unknown operation: {operation}"
            except Exception as e:
                print(f"执行特征工程操作 '{operation}' 时发生错误: {e}")
                step_log["error"] = str(e)

            self.applied_steps.append(step_log)

        return df_engineered

    def generate_features(self, df: pd.DataFrame) -> Tuple[DataFrame, List[Union[
        Dict[str, str], Dict[str, str], Dict[str, Union[List[dict], list, str]], Dict[str, str], Dict[str, str], Dict[
            str, Union[List[Dict[str, str]], str]]]], Dict[Any, Any]]:
        """
        执行特征工程的主流程。

        参数:
        - df: 经过预处理的Pandas DataFrame。

        返回:
        - df_engineered: 经过特征工程的DataFrame。
        - applied_steps: 应用的特征工程步骤列表。
        - fitted_objects: 存储的拟合对象 (如PolynomialFeatures转换器)。
        """
        print("\n--- 开始特征工程 ---")
        self.applied_steps = []
        self.fitted_objects = {}

        if df is None or df.empty:
            print("错误: 输入的DataFrame为空，无法进行特征工程。")
            self.applied_steps.append({"error": "Input DataFrame is empty."})
            return df, self.applied_steps, self.fitted_objects

        # 提取当前所有特征列名 (排除目标列)
        current_feature_names = [col for col in df.columns if col != self.target_metric]
        if not current_feature_names:
            print("警告: 没有可用的特征列进行特征工程。")
            self.applied_steps.append({"warning": "No features available for engineering."})
            return df, self.applied_steps, self.fitted_objects

        current_features_str = ", ".join(current_feature_names)

        # 1. LLM分析现有特征和知识库，制定特征工程计划
        print("步骤1: LLM制定特征工程计划...")
        system_prompt, user_prompt = self._generate_llm_prompt_for_fe_plan(current_features_str)

        # print("\n--- LLM System Prompt (Feature Engineering) ---")
        # print(system_prompt)
        # print("\n--- LLM User Prompt (Feature Engineering) ---")
        # print(user_prompt[:1000] + "..." if len(user_prompt) > 1000 else user_prompt)

        llm_response_str = call_llm(system_prompt, user_prompt)

        try:
            feature_engineering_plan = json.loads(llm_response_str)
            if not isinstance(feature_engineering_plan, list):  # 期望是一个操作列表
                # 尝试处理LLM可能返回单个字典的情况（例如只有一个no_action）
                if isinstance(feature_engineering_plan, dict) and feature_engineering_plan.get(
                        "operation") == "no_action":
                    feature_engineering_plan = [feature_engineering_plan]
                else:
                    raise ValueError("LLM响应不是一个列表。")

            # 简单验证计划结构
            for step in feature_engineering_plan:
                if not isinstance(step, dict) or "operation" not in step:
                    raise ValueError(f"计划中的步骤 '{step}' 格式不正确。")

            self.applied_steps.append({
                "step": "llm_generate_fe_plan",
                "status": "success",
                "plan": feature_engineering_plan
            })
            print("LLM特征工程计划已生成。")
        except json.JSONDecodeError as e:
            print(f"错误: LLM返回的特征工程计划不是有效的JSON: {e}")
            print(f"LLM原始输出: {llm_response_str}")
            self.applied_steps.append(
                {"step": "llm_generate_fe_plan", "status": "failed", "error": f"Invalid JSON from LLM: {e}",
                 "raw_response": llm_response_str})
            return df, self.applied_steps, self.fitted_objects
        except ValueError as e:
            print(f"错误: LLM返回的特征工程计划结构不正确: {e}")
            print(f"LLM原始输出: {llm_response_str}")
            self.applied_steps.append(
                {"step": "llm_generate_fe_plan", "status": "failed", "error": f"Invalid plan structure: {e}",
                 "raw_response": llm_response_str})
            return df, self.applied_steps, self.fitted_objects

        # 2. 执行特征工程计划
        if not feature_engineering_plan or (
                len(feature_engineering_plan) == 1 and feature_engineering_plan[0].get("operation") == "no_action"):
            print("LLM建议不执行新的特征工程步骤。")
            if not self.applied_steps or self.applied_steps[-1].get("plan") is None:  # 确保plan被记录
                self.applied_steps.append(
                    {"step": "llm_generate_fe_plan", "status": "success", "plan": [{"operation": "no_action"}]})
        else:
            print("步骤2: 执行特征工程计划...")
            df = self._execute_feature_engineering_plan(df, feature_engineering_plan)

        print("--- 特征工程完成 ---")
        return df, self.applied_steps, self.fitted_objects


if __name__ == '__main__':
    # 假设这是从预处理步骤得到的DataFrame
    data = {
        'feature_C': np.array([0.1, 0.15, 0.12, 0.18, 0.11]),  # 确保是numpy array或Series
        'feature_Mn': np.array([0.5, 0.6, 0.55, 0.65, 0.52]),
        'process_param1': np.array([100.0, 102.0, 99.0, 105.0, 101.0]),
        'process_param2': np.array([50.0, 51.0, 48.0, 52.0, 49.0]),
        'YieldStrength': np.array([450, 480, 460, 500, 455])  # 目标列
    }
    sample_df_preprocessed = pd.DataFrame(data)

    print("预处理后的DataFrame (用于特征工程输入):")
    print(sample_df_preprocessed)

    user_request_example = "请为预测屈服强度构建一些有用的特征，特别是考虑成分之间的交互以及工艺参数的比率。"
    target_metric_example = "YieldStrength"
    preprocessed_cols_example = [col for col in sample_df_preprocessed.columns if col != target_metric_example]

    feature_generator = FeatureGenerator(user_request_example, target_metric_example, preprocessed_cols_example)

    # 运行特征工程 (这将调用LLM)
    # 确保LLM服务和知识库可用
    df_engineered, fe_steps, fe_fitted_objs = feature_generator.generate_features(sample_df_preprocessed.copy())

    print("\n--- 特征工程结果 ---")
    print("特征工程后的DataFrame (前5行):")
    print(df_engineered.head())
    if not df_engineered.empty:
        print("\n特征工程后的DataFrame信息:")
        df_engineered.info()

    print("\n应用的特征工程步骤:")
    for step in fe_steps:
        print(json.dumps(step, indent=2, ensure_ascii=False, default=str))

    print("\n特征工程中拟合的对象:")
    for name, obj in fe_fitted_objs.items():
        print(f"- {name}: {type(obj)}")
