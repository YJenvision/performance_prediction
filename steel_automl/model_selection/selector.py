import json
from typing import Dict, Any, Tuple, List
from llm_utils import call_llm
from knowledge_base.kb_service import KnowledgeBaseService
from config import MODEL_SELECTION_KB_NAME, HISTORICAL_CASES_KB_NAME


class ModelSelector:
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
        # self.history_kb_service = KnowledgeBaseService(HISTORICAL_CASES_KB_NAME)
        self.available_models = {  # AutoML系统当前支持的回归模型算法库
            "RandomForestRegressor": {
                "description": "随机森林回归器，基于决策树的集成学习方法，鲁棒性较好。",
                "hyperparameters": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
            },
            "XGBoostRegressor": {
                "description": "XGBoost回归器，梯度提升框架的高效实现，通常性能优越但需要仔细调参。",
                "hyperparameters": ["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree"]
            },
            # "LinearRegression": { # 后续添加更多模型
            #     "description": "线性回归模型，简单快速，适用于线性关系数据。"
            # },
        }

    def _get_knowledge_snippets(self, context_query: str, k: int = 2) -> str:
        """从模型选择和历史案例知识库检索信息。"""
        ms_snippets = self.ms_kb_service.search(context_query, k=k)
        # history_snippets = self.history_kb_service.search(context_query, k=k)

        formatted_snippets = "\n模型选择知识库参考:\n"
        if ms_snippets:
            for i, snippet in enumerate(ms_snippets):
                meta = snippet.get('metadata', {})
                formatted_snippets += f"{i + 1}. {meta}\n"
        else:
            formatted_snippets += "未从模型选择知识库中检索到相关信息。\n"

        # formatted_snippets += "\n历史案例参考 (关于模型选择):\n" if history_snippets: for i, snippet in enumerate(
        # history_snippets): meta = snippet.get('metadata', {}) # 假设历史案例元数据中有模型选择相关信息 formatted_snippets += f"{i +
        # 1}. 案例 {meta.get('case_id', '未知')}: 使用模型 {meta.get('model_used', '未知')}, 原因: {meta.get(
        # 'reason_for_model_choice', '无')}\n" else: formatted_snippets += "未从历史案例知识库中检索到相关模型选择信息。\n"

        return formatted_snippets

    def _generate_llm_prompt_for_model_selection(self) -> Tuple[str, str]:
        """为模型选择生成系统和用户提示词。"""
        system_prompt = f"""
你是一位资深的机器学习专家，尤其擅长为工业领域的回归问题选择合适的预测模型。
你的任务是根据用户需求、数据概况（样本量、特征数）、以及从知识库中检索到的模型选择经验和历史案例，推荐一个或多个合适的回归模型，并给出初步的超参数建议范围。

用户原始请求摘要: "{self.user_request}"
目标性能指标: "{self.target_metric}" (这是一个回归问题)
数据概况: 样本数量为{self.num_samples}条, 特征数量为{self.num_features}个

当前系统支持的回归模型包括:
{json.dumps(self.available_models, indent=2, ensure_ascii=False)}

决策依据:
- 数据规模: 样本量和特征数对模型选择有重要影响。
- 用户偏好: 用户优先级最高，如果用户请求中明确指定或排除了某些模型，请优先予以考虑。
- 模型特性: 例如，随机森林对过拟合不敏感，XGBoost通常性能更好但需要调参。
- 知识库与历史案例: 参考相似问题或数据规模下表现良好的模型。

输出格式要求:
严格按照以下JSON格式返回你的建议。返回一个字典，键为推荐的模型名称 (必须是上述支持的模型之一)，值为一个包含 "reason" (选择理由) 和 "hyperparameter_suggestions" (超参数建议范围或初步值列表) 的字典。
如果推荐多个模型，请都包含在内。

示例JSON输出:
{{
  "RandomForestRegressor": {{
    "reason": "适用于当前样本量和特征数，鲁棒性较好，不易过拟合。",
    "hyperparameter_suggestions": {{
      "n_estimators": [50, 200],
      "max_depth": [5, 10, 15, null],
      "min_samples_split": [2, 5, 10]
    }}
  }},
  "XGBoostRegressor": {{
    "reason": "可能获得更高精度，但需要仔细调参。适合在有足够算力时尝试。",
    "hyperparameter_suggestions": {{
      "n_estimators": [50, 200],
      "learning_rate": [0.01, 0.05, 0.1],
      "max_depth": [3, 5, 7]
    }}
  }}
}}
如果只推荐一个模型，JSON中就只包含那一个模型的条目。
请确保你的回复是且仅是一个合法的JSON对象。不要包含任何解释性文字或代码块标记。
"""
        kb_query_context = f"模型选择咨询：用户请求 '{self.user_request}', 目标 '{self.target_metric}', 样本数 {self.num_samples}, 特征数 {self.num_features}。寻求回归模型建议。"
        knowledge_snippets = self._get_knowledge_snippets(kb_query_context)

        user_prompt = f"""
请为以下回归任务选择合适的模型并提供初步的超参数建议。

用户原始请求:
{self.user_request}

任务概况:
- 目标指标: {self.target_metric}
- 样本数量: {self.num_samples}
- 特征数量: {self.num_features}

{knowledge_snippets}

请严格按照系统提示中要求的JSON格式输出你的模型选择和超参数建议。
选择的模型必须是系统支持的模型之一。
"""
        return system_prompt, user_prompt

    def select_model(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        执行模型选择的主流程。

        返回:
        - selected_models_info: 一个字典，包含LLM推荐的模型及其超参数建议和理由。
        - selection_log: 模型选择过程的日志。
        """
        print("\n--- 开始模型选择 ---")
        log = []

        # 1. LLM分析任务和知识库，推荐模型
        print("步骤1: LLM推荐模型及超参数...")
        system_prompt, user_prompt = self._generate_llm_prompt_for_model_selection()

        llm_response_str = call_llm(system_prompt, user_prompt)
        # print(f"\nLLM原始响应 (模型选择):\n{llm_response_str}")

        selected_models_info = {}
        try:
            parsed_response = json.loads(llm_response_str)
            if not isinstance(parsed_response, dict):
                raise ValueError("大语言模型在算法模型选择过程中的响应不是一个字典。")

            for model_name, model_info in parsed_response.items():
                if model_name not in self.available_models:
                    print(f"警告: 大语言模型推荐了系统不支持的模型 '{model_name}'，将被忽略。")
                    log.append({"warning": f"LLM recommended unsupported model: {model_name}"})
                    continue
                if not isinstance(model_info,
                                  dict) or "reason" not in model_info or "hyperparameter_suggestions" not in model_info:
                    raise ValueError(f"模型 '{model_name}' 的信息格式不正确。")
                selected_models_info[model_name] = model_info

            if not selected_models_info:  # 如果所有推荐模型都被过滤掉了
                log.append({"error": "LLM did not recommend any supported models or response was invalid."})
                print("错误: LLM未能推荐任何受支持的模型，或响应无效。")
                # 选择一个默认模型作为回退
                default_model_name = "RandomForestRegressor"
                selected_models_info[default_model_name] = {
                    "reason": "Default fallback model due to LLM failure.",
                    "hyperparameter_suggestions": {"n_estimators": [100], "max_depth": [10]}
                }
                print(f"回退到默认模型: {default_model_name}")

            log.append({
                "step": "llm_recommend_models",
                "status": "success" if selected_models_info else "partial_success",
                "recommendations": selected_models_info
            })
            print(f"LLM模型选择建议: {list(selected_models_info.keys())}")

        except json.JSONDecodeError as e:
            print(f"错误: LLM返回的模型选择建议不是有效的JSON: {e}")
            print(f"LLM原始输出: {llm_response_str}")
            log.append({"step": "llm_recommend_models", "status": "failed", "error": f"Invalid JSON: {e}",
                        "raw_response": llm_response_str})
            # 回退策略
        except ValueError as e:
            print(f"错误: LLM返回的模型选择建议结构不正确: {e}")
            print(f"LLM原始输出: {llm_response_str}")
            log.append({"step": "llm_recommend_models", "status": "failed", "error": f"Invalid structure: {e}",
                        "raw_response": llm_response_str})
            # 回退策略

        if not selected_models_info and not any(
                "error" in l for l in log if l.get("step") == "llm_recommend_models"):  # 如果解析成功但没有有效模型
            print("警告: LLM未能提供有效的模型建议。将使用默认模型 RandomForestRegressor。")
            selected_models_info["RandomForestRegressor"] = {
                "reason": "Default fallback model as LLM provided no valid suggestions.",
                "hyperparameter_suggestions": {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
            }
            log.append({"step": "fallback_model_selection", "status": "success", "model": "RandomForestRegressor"})

        print("--- 模型选择完成 ---")
        return selected_models_info, log
