import json
import os
from typing import Dict, Any, Tuple, List, Generator
from llm_utils import call_llm
from knowledge_base.kb_service import KnowledgeBaseService
from config import MODEL_SELECTION_KB_NAME, HISTORICAL_CASES_KB_NAME
from prompts.prompt_manager import get_prompt


class ModelSelector:
    """
    为AutoML流程生成一个完整的算法选择计划。
    该计划包括可接受误差、数据划分策略、交叉验证策略、超参数优化(HPO)方法、推荐的模型及其超参数范围。
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
        self.available_data_split_methods = {
            "sequential": {
                "description": "顺序切分。严格按照时间顺序将数据集的最后一部分作为测试集。这是时间序列数据的首选，可以有效模拟真实预测场景，避免未来数据泄露。"
            },
            "random": {
                "description": "随机切分。从整个数据集中随机抽样一部分作为测试集。适用于不具有时间序列特征的数据。"
            }
        }
        self.available_cv_methods = {
            "time_series": {
                "description": "时序K折交叉验证 (TimeSeriesSplit)。在每次迭代中，验证集都位于训练集之后。这是时间序列数据交叉验证的标准做法。"
            },
            "random": {
                "description": "随机K折交叉验证 (KFold)。将数据随机分成K个子集，轮流作为验证集。适用于非时间序列数据。"
            }
        }
        self.available_models = {
            "RandomForestRegressor": {
                "description": "随机森林回归器，基于决策树的集成学习方法，鲁棒性较好，不易过拟合。",
                "hyperparameters": ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]
            },
            "XGBoostRegressor": {
                "description": "XGBoost回归器，梯度提升框架的高效实现，通常性能优越但需要仔细调参。",
                "hyperparameters": ["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree"]
            },
            "LightGBMRegressor": {
                "description": "LightGBM回归器，一种高性能的梯度提升框架，使用基于直方图的算法，训练速度快，内存占用低。",
                "hyperparameters": ["n_estimators", "learning_rate", "max_depth", "num_leaves", "subsample", "colsample_bytree"]
            }
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
        error_config_json = json.dumps(self.error_config, indent=2, ensure_ascii=False)

        system_prompt = get_prompt(
            'selector.determine_acceptable_error.system',
            error_config_json=error_config_json
        )
        user_prompt = get_prompt(
            'selector.determine_acceptable_error.user',
            user_request=self.user_request,
            target_metric=self.target_metric
        )

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
        # Prepare JSON strings for injection into the prompt
        available_data_split_methods_json = json.dumps(self.available_data_split_methods, indent=2, ensure_ascii=False)
        available_cv_methods_json = json.dumps(self.available_cv_methods, indent=2, ensure_ascii=False)
        available_models_json = json.dumps(self.available_models, indent=2, ensure_ascii=False)
        available_hpo_methods_json = json.dumps(self.available_hpo_methods, indent=2, ensure_ascii=False)
        acceptable_error_json = json.dumps(acceptable_error)

        system_prompt = get_prompt(
            'selector.generate_automl_plan.system',
            available_data_split_methods_json=available_data_split_methods_json,
            available_cv_methods_json=available_cv_methods_json,
            available_models_json=available_models_json,
            available_hpo_methods_json=available_hpo_methods_json,
            acceptable_error_json=acceptable_error_json
        )

        kb_query_context = f"AutoML计划制定咨询：用户请求 '{self.user_request}', 目标 '{self.target_metric}', 样本数 {self.num_samples}, 特征数 {self.num_features}。"
        knowledge_snippets = self._get_knowledge_snippets(kb_query_context)

        user_prompt = get_prompt(
            'selector.generate_automl_plan.user',
            user_request=self.user_request,
            target_metric=self.target_metric,
            num_samples=self.num_samples,
            num_features=self.num_features,
            acceptable_error=acceptable_error,
            knowledge_snippets=knowledge_snippets
        )
        return system_prompt, user_prompt

    def _get_default_plan(self, reason: str, acceptable_error: Dict) -> Dict[str, Any]:
        """在智能体调用失败时，提供一个默认的后备计划。"""
        return {
            "model_plan": {
                "reason": f"由于智能体失败或响应无效而启用后备计划: {reason}",
                "acceptable_error": acceptable_error,
                "data_split_plan": {
                    "method": "sequential",
                    "test_size": 0.2,
                    "reason": "后备计划默认采用顺序切分，以处理潜在的时间序列数据。"
                },
                "cv_plan": {
                    "method": "time_series",
                    "k_folds": 5,
                    "reason": "后备计划默认采用时序交叉验证，以处理潜在的时间序列数据。"
                },
                "hpo_config": {
                    "method": "RandomizedSearchCV",
                    "n_iter": 30,
                    "scoring_metric": "neg_mean_squared_error"
                }
            },
            "model_recommendations": {
                "XGBoostRegressor": {
                    "reason": "默认回退模型，XGBoost通常在各种回归任务上表现稳健且高效。",
                    "hyperparameter_suggestions": {
                        "n_estimators": [50, 200],
                        "learning_rate": [0.01, 0.2, "log"],
                        "max_depth": [3, 10],
                        "subsample": [0.7, 1.0],
                        "colsample_bytree": [0.7, 1.0]
                    }
                }
            }
        }

    def select_model_and_plan(self) -> Generator[Dict[str, Any], None, Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        current_stage = "模型选择与计划制定"
        log = []
        final_plan = {}

        # 子任务 1: 确定可接受的误差范围
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

        if acceptable_error:
            yield {"type": "substage_result", "payload": {
                "stage": current_stage, "substage_title": "可接受误差分析",
                "data": acceptable_error
            }}
        else:
            # 如果确定误差失败，生成器内部已经yield了error, 这里直接返回后备计划
            return self._get_default_plan("无法确定误差范围", {}), log

        # 子任务 2: 基于误差范围生成完整的建模计划
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running",
                           "detail": "正在制定算法模型与模型超参数优化计划..."}}
        system_prompt, user_prompt = self._generate_llm_prompt(acceptable_error)

        llm_response_str = ""
        agent_failed = False
        try:
            llm_gen = call_llm(system_prompt, user_prompt, model="ds_v3")
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
        except Exception as e:
            yield {"type": "error", "payload": {"stage": current_stage, "detail": f"制定计划时智能体调用异常: {e}"}}
            agent_failed = True

        if agent_failed or "Agent failed" in llm_response_str:
            error_msg = "智能体在生成计划时失败。"
            log.append({"step": "llm_generate_plan", "status": "failed", "error": error_msg})
            final_plan = self._get_default_plan(error_msg, acceptable_error)
            log.append({"step": "fallback_plan_activated", "status": "success"})
            yield {"type": "substage_result",
                   "payload": {"stage": current_stage, "substage_title": "模型与模型超参数优化推荐计划 (后备)",
                               "data": final_plan}}
            return final_plan, log

        try:
            parsed_response = json.loads(llm_response_str)
            # 验证计划的完整性
            if "model_plan" not in parsed_response or "model_recommendations" not in parsed_response:
                raise ValueError("响应缺少 'model_plan' 或 'model_recommendations' 顶级键。")
            if "data_split_plan" not in parsed_response["model_plan"] or "cv_plan" not in parsed_response["model_plan"]:
                raise ValueError("响应的 'model_plan' 中缺少 'data_split_plan' 或 'cv_plan'。")

            recommendations = parsed_response.get("model_recommendations", {})
            if not isinstance(recommendations, dict) or not recommendations:
                raise ValueError("响应中的 'model_recommendations' 必须是一个非空字典。")

            valid_recommendations = {name: info for name, info in recommendations.items() if
                                     name in self.available_models}
            if not valid_recommendations:
                raise ValueError("智能体未提供任何受支持的模型推荐。")

            final_plan = {"model_plan": parsed_response["model_plan"], "model_recommendations": valid_recommendations}
            # 确保可接受误差被正确地设置在最终计划中
            final_plan["model_plan"]["acceptable_error"] = acceptable_error
            log.append({"step": "llm_generate_plan", "status": "success", "plan": final_plan})
            yield {"type": "substage_result",
                   "payload": {"stage": current_stage, "substage_title": "算法模型与模型超参数优化计划",
                               "data": final_plan}}
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            error_msg = f"智能体解析或验证响应失败: {e}"
            log.append(
                {"step": "llm_generate_plan", "status": "failed", "error": error_msg, "raw_response": llm_response_str})
            final_plan = self._get_default_plan(error_msg, acceptable_error)
            log.append({"step": "fallback_plan_activated", "status": "success"})
            yield {"type": "substage_result",
                   "payload": {"stage": current_stage, "substage_title": "算法模型与模型超参数优化计划 (后备)",
                               "data": final_plan}}

        return final_plan, log
