import json
import os
from typing import Dict, Any, Tuple, List, Generator
from llm_utils import call_llm
from knowledge_base.kb_service import KnowledgeBaseService
from config import MODEL_SELECTION_KB_NAME, HISTORICAL_CASES_KB_NAME


class ModelSelector:
    """
    为AutoML流程生成一个完整的算法选择计划。
    该计划包括可接受误差、数据划分策略、超参数优化(HPO)方法、推荐的模型及其超参数范围。
    """

    def __init__(self, user_request: str, target_metric: str, num_features: int, num_samples: int):
        """
        初始化模型选择器。

        参数:
        - user_request: 用户的原始自然语言建模请求。
        - target_metric: 目标性能指标。
        - num_features: 特征数量。
        - num_samples: 样本数量。
        """
        self.user_request = user_request
        self.target_metric = target_metric
        self.num_features = num_features
        self.num_samples = num_samples
        self.ms_kb_service = KnowledgeBaseService(MODEL_SELECTION_KB_NAME)
        self.error_config = self._load_error_config()
        # self.history_kb_service = KnowledgeBaseService(HISTORICAL_CASES_KB_NAME)

        # 系统当前支持的组件
        self.available_models = {
            "RandomForestRegressor": {
                "description": "随机森林回归器，基于决策树的集成学习方法，鲁棒性较好，不易过拟合。",
                "hyperparameters": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
            },
            "XGBoostRegressor": {
                "description": "XGBoost回归器，梯度提升框架的高效实现，通常性能优越但需要仔细调参。",
                "hyperparameters": ["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree"]
            },
        }
        self.available_hpo_methods = {
            "GridSearchCV": {
                "description": "网格搜索。尝试所有给定的超参数组合，精度高但计算成本巨大。仅适用于超参数组合总数非常少（如<50）的场景。"
            },
            "RandomizedSearchCV": {
                "description": "随机搜索。在给定的参数空间中随机采样指定次数(n_iter)的组合。在较大搜索空间中，通常能以更少的计算成本找到较好的参数组合。"
            },
            "BayesianOptimization": {
                "description": "贝叶斯优化。使用高斯过程等代理模型来评估最有希望的超参数组合，从而更智能地探索搜索空间。在寻找最优解方面通常比随机搜索更高效。是大多数情况下的首选。"
            }
        }

    def _load_error_config(self) -> Dict:
        """加载误差范围配置文件。"""
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'error_config.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"default": {"type": "percentage", "value": 5}}

    def _determine_acceptable_error(self) -> Generator[Dict[str, Any], None, Tuple[Dict[str, Any], str]]:
        """
        使用智能体根据用户请求和配置文件确定可接受的误差范围，现在是一个生成器。
        """
        system_prompt = f"""
你是一位领域专家，任务是为机器学习回归模型的性能评估确定一个“可接受的误差范围”。

**你的决策依据:**
1.  **用户明确要求**: 如果用户的请求中明确提到了误差范围（例如，“误差要小于5%”，“精度在10以内”），你必须优先采用用户的要求，并设定来源为 "user_request"。
2.  **配置文件**: 如果用户没有明确要求，请参考提供的配置文件中的默认设置。如果使用了默认配置，设定来源为 "config_default"。

**配置文件内容:**
{json.dumps(self.error_config, indent=2, ensure_ascii=False)}

**输出格式要求:**
你必须严格按照以下JSON格式返回你的决策。JSON对象必须包含 "type", "value", 和 "source" 三个键。
- "type": 必须是 "percentage" (百分比) 或 "value" (绝对数值) 中的一个。
- "value": 必须是一个数字。
- "source": 必须是 "user_request", "config_default" 中的一个。

例如:
{{
  "type": "percentage",
  "value": 5,
  "source": "user_request"
}}

确保你的回复是且仅是一个合法的、不含任何额外解释性文本或代码块标记的JSON对象。
"""
        user_prompt = f"""
请为以下任务确定可接受的误差范围。
**用户原始请求:**
{self.user_request}
**目标性能指标:**
{self.target_metric}
请严格按照系统提示中要求的JSON格式输出你的决策。
"""
        llm_gen = call_llm(system_prompt, user_prompt, model="ds_v3")
        llm_response_str = ""
        while True:
            try:
                chunk = next(llm_gen)
                if chunk.get("type") == "error":
                    yield chunk
                    error_msg = "确定误差范围失败: 智能体调用失败。将使用配置文件中的默认值。"
                    default_error = self.error_config.get(self.target_metric, self.error_config.get("default"))
                    default_error["source"] = "fallback"
                    return default_error, error_msg
                yield chunk
            except StopIteration as e:
                llm_response_str = e.value
                break

        try:
            parsed_response = json.loads(llm_response_str)
            if "type" in parsed_response and "value" in parsed_response and "source" in parsed_response:
                return parsed_response, f"成功确定误差范围: {parsed_response}"
            else:
                raise ValueError("智能体响应缺少必要的键。")
        except Exception as e:
            error_msg = f"确定误差范围失败: {e}。将使用配置文件中的默认值。"
            default_error = self.error_config.get(self.target_metric, self.error_config.get("default"))
            default_error["source"] = "fallback"
            return default_error, error_msg

    def _get_knowledge_snippets(self, context_query: str, k: int = 2) -> str:
        """从知识库中检索相关信息片段。"""
        ms_snippets = self.ms_kb_service.search(context_query, k=k)
        formatted_snippets = "\n[知识库参考信息]\n"
        if ms_snippets:
            for i, snippet in enumerate(ms_snippets):
                meta = snippet.get('metadata', {})
                formatted_snippets += f"{i + 1}. {meta}\n"
        else:
            formatted_snippets += "未从知识库中检索到相关信息。\n"
        return formatted_snippets

    def _generate_llm_prompt(self, acceptable_error: Dict) -> Tuple[str, str]:
        """为生成完整的AutoML计划生成系统和用户提示词。"""
        system_prompt = f"""
你是一位资深的AutoML专家，任务是为给定的回归问题设计一个完整的、智能化的建模计划。
你需要根据用户的原始请求、数据概况、已确定的可接受误差范围和知识库信息，决定数据划分策略、超参数优化(HPO)方法，并从系统支持的模型列表中推荐一个或多个适合的模型及其超参数搜索范围。

**决策依据:**
1.  **用户意图**: 用户的原始请求是最高优先级。
    - 若用户强调“快速验证”、“要快”，应选择计算成本较低的策略，如 `RandomizedSearchCV` 并设置一个较小的迭代次数 `n_iter` (例如 20-35)。
    - 若用户强调“精度”、“效果最好”，应选择更高效的 `RandomizedSearchCV` 或 `BayesianOptimization` 并设置一个较大的迭代次数 `n_iter` (例如 40-60)。
    - 对于一般性请求，`BayesianOptimization` 是一个很好的平衡选择，`n_iter` 可以设为 (例如 30-40)。
2.  **数据规模**:
    - **数据划分 (`data_split_ratio`)**: 对于小样本量(如 < 500)，可以考虑较小的测试集比例(如0.15)以保留更多训练数据。对于大样本，0.2-0.3是常规选择。
    - **HPO方法**: 对于特征多、样本量大的情况，`GridSearchCV` 几乎不可行。`RandomizedSearchCV` 和 `BayesianOptimization` 是实际的选择。
3.  **模型与参数**: 推荐的超参数必须是数值型，并以范围形式给出。

**可用组件:**
- **支持的模型:**
{json.dumps(self.available_models, indent=2, ensure_ascii=False)}
- **支持的HPO方法:**
{json.dumps(self.available_hpo_methods, indent=2, ensure_ascii=False)}

**输出格式要求:**
你必须严格按照以下JSON格式返回你的完整计划。JSON对象必须包含 "model_plan" 和 "model_recommendations" 两个顶级键。 "model_plan" 中必须包含 `acceptable_error` 字段。

{{
  "model_plan": {{
    "reason": "简要说明你制定此计划的总体理由，特别是HPO方法和迭代次数的选择依据。",
    "acceptable_error": {json.dumps(acceptable_error)},
    "data_split_ratio": <一个浮点数, e.g., 0.2>,
    "hpo_config": {{
      "method": "<从 'GridSearchCV', 'RandomizedSearchCV', 'BayesianOptimization' 中选择>",
      "n_iter": <一个整数, 代表 'RandomizedSearchCV' 的 `n_iter` 或 'BayesianOptimization' 的 `n_trials`。默认值：快速=30, 常规=45, 精确=60。如果选择 'GridSearchCV', 此项为 null>,
      "cv_folds": <一个整数, 交叉验证的折数, 默认为3>,
      "scoring_metric": "<一个用于优化超参数的评估指标, e.g., 'neg_mean_squared_error'(MSE)、'neg_mean_absolute_error'(MAE)、'neg_root_mean_squared_error'(RMSE)、'r2'(R2)等评分指标字符串>"
    }}
  }},
  "model_recommendations": {{
    "模型名称": {{
      "reason": "简要说明选择此模型的具体理由。",
      "hyperparameter_suggestions": {{
        "整数型参数": [<下界整数>, <上界整数>],
        "浮点型参数": [<下界浮点数>, <上界浮点数>],
        "对数分布的浮点型参数": [<下界浮点数>, <上界浮点数>, "log"]
      }}
    }}
  }}
}}

**超参数范围说明:**
- **整数型**: `["param_name": [min, max]]` -> 例如 `"n_estimators": [50, 200]`
- **浮点型 (线性分布)**: `["param_name": [min, max]]` -> 例如 `"subsample": [0.7, 1.0]`
- **浮点型 (对数分布)**: `["param_name": [min, max, "log"]]` -> 例如 `"learning_rate": [0.01, 0.2, "log"]`。这对于学习率等跨越数量级的参数特别有效。

确保你的回复是且仅是一个合法的、不含任何额外解释性文本或代码块标记的JSON对象。
"""
        kb_query_context = f"AutoML计划制定咨询：用户请求 '{self.user_request}', 目标 '{self.target_metric}', 样本数 {self.num_samples}, 特征数 {self.num_features}。"
        knowledge_snippets = self._get_knowledge_snippets(kb_query_context)

        user_prompt = f"""
请为以下回归任务制定一个完整的AutoML模型和超参数优化计划。

**用户原始请求:**
{self.user_request}

**任务概况:**
- 目标指标: {self.target_metric}
- 样本数量: {self.num_samples}
- 特征数量: {self.num_features}
- 已确定的可接受误差: {acceptable_error}

{knowledge_snippets}

请严格按照系统提示中要求的JSON格式输出你的计划。
"""
        return system_prompt, user_prompt

    def _get_default_plan(self, reason: str, acceptable_error: Dict) -> Dict[str, Any]:
        """在智能体调用失败时，提供一个默认的后备计划。"""
        return {
            "model_plan": {
                "reason": f"由于智能体失败或响应无效而启用后备计划: {reason}",
                "acceptable_error": acceptable_error,
                "data_split_ratio": 0.2,
                "hpo_config": {"method": "RandomizedSearchCV", "n_iter": 30, "cv_folds": 3,
                               "scoring_metric": "neg_mean_squared_error"}
            },
            "model_recommendations": {
                "RandomForestRegressor": {
                    "reason": "默认回退模型，通常稳健且表现良好。",
                    "hyperparameter_suggestions": {"n_estimators": [50, 150], "max_depth": [5, 20],
                                                   "min_samples_split": [2, 10]}
                }
            }
        }

    def select_model_and_plan(self) -> Generator[Dict[str, Any], None, Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        执行AutoML计划生成的主流程，现在是一个生成器。
        """
        current_stage = "模型选择与HPO计划"
        log = []
        final_plan = {}

        # 步骤 1: 确定可接受的误差范围
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "正在确定可接受的误差范围..."}}
        error_gen = self._determine_acceptable_error()
        acceptable_error, error_log_msg = None, ""
        while True:
            try:
                chunk = next(error_gen)
                yield chunk
            except StopIteration as e:
                acceptable_error, error_log_msg = e.value
                break
        log.append({"step": "determine_acceptable_error", "details": error_log_msg})
        if acceptable_error is None:  # Error already yielded
            return self._get_default_plan("无法确定误差范围", {}), log

        # 步骤 2: 基于误差范围生成完整的建模计划
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "正在制定模型选择与HPO计划..."}}
        system_prompt, user_prompt = self._generate_llm_prompt(acceptable_error)

        llm_gen = call_llm(system_prompt, user_prompt)
        llm_response_str = ""
        agent_failed = False
        while True:
            try:
                chunk = next(llm_gen)
                if chunk.get("type") == "error":
                    yield chunk
                    agent_failed = True
                    break
                yield chunk
            except StopIteration as e:
                llm_response_str = e.value
                break

        if agent_failed or "Agent failed" in llm_response_str:
            error_msg = "智能体在生成计划时失败。"
            log.append({"step": "llm_generate_plan", "status": "failed", "error": error_msg})
            final_plan = self._get_default_plan(error_msg, acceptable_error)
            log.append({"step": "fallback_plan_activated", "status": "success"})
            return final_plan, log

        try:
            parsed_response = json.loads(llm_response_str)
            if "model_plan" not in parsed_response or "model_recommendations" not in parsed_response:
                raise ValueError("响应缺少 'model_plan' 或 'model_recommendations' 顶级键。")

            recommendations = parsed_response.get("model_recommendations", {})
            if not isinstance(recommendations, dict) or not recommendations:
                raise ValueError("响应中的 'model_recommendations' 必须是一个非空字典。")

            valid_recommendations = {name: info for name, info in recommendations.items() if
                                     name in self.available_models}
            if not valid_recommendations:
                raise ValueError("智能体未提供任何受支持的模型推荐。")

            final_plan = {"model_plan": parsed_response["model_plan"], "model_recommendations": valid_recommendations}
            final_plan["model_plan"]["acceptable_error"] = acceptable_error
            log.append({"step": "llm_generate_plan", "status": "success", "plan": final_plan})
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            error_msg = f"智能体解析或验证响应失败: {e}"
            log.append(
                {"step": "llm_generate_plan", "status": "failed", "error": error_msg, "raw_response": llm_response_str})
            final_plan = self._get_default_plan(error_msg, acceptable_error)
            log.append({"step": "fallback_plan_activated", "status": "success"})

        return final_plan, log
