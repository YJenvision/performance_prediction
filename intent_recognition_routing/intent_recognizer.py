import json
from datetime import datetime
from llm_utils import call_llm
from knowledge_base.kb_service import KnowledgeBaseService


class SteelPerformanceIntentRecognizer:
    """意图识别器"""

    def __init__(self):
        """初始化意图识别器。"""
        # 后续可以初始化一些所需资源，例如加载模型、知识库等
        pass

    def _parse_llm_json_response(self, llm_response_str: str) -> dict:
        """
        安全地解析LLM返回的JSON字符串。

        参数:
        - llm_response_str: LLM返回的可能包含JSON的字符串。

        返回:
        - 解析后的字典对象，或包含错误信息的字典。
        """
        # 检查是否是LLM方法调用失败的字符串
        if isinstance(llm_response_str, str) and "大语言模型调用失败" in llm_response_str:
            return {"error": "LLM_CALL_FAILED", "details": llm_response_str}

        try:
            data = json.loads(llm_response_str)
            return data

        except json.JSONDecodeError as e:
            return {"error": "JSON_PARSE_FAILED", "details": str(e), "original_response": llm_response_str}

        except Exception as e:
            return {"error": "RESPONSE_PROCESSING_ERROR", "details": str(e),
                    "original_response": llm_response_str}

    def recognize_intent(self, user_request: str) -> dict:
        """
        识别用户请求的意图，并根据意图提取信息。

        参数:
        - user_request: 用户的原始请求字符串。

        返回:
        - 一个包含意图和相关信息的字典。
        """
        # 步骤1: 意图分类
        # 意图分类的系统提示词
        INTENT_CLASSIFICATION_SYSTEM_PROMPT = """
你是一个专业的钢铁产品性能预报智能意图分类助手，核心任务是精准理解和分类用户的请求。
用户的请求可以分为以下四种主要意图：

1.  **模型构建与评估 (model_building_evaluation_request)**：
    * 描述：用户希望使用历史数据创建、训练一个新的性能预报模型，并进行性能评估、验证。这包括选择预处理方法、模型算法、调参、以及衡量模型准确性、稳定性等。
    * 关键词：构建、建立、训练、评估、验证、新模型、开发模型、预测模型、选择算法、调参、准确率、召回率、MAE、RMSE等。
    * 例子：
        * "用过去三年的H031机组S355JR牌号产品的生产数据，构建一个屈服强度的性能预报模型。"
        * "基于最新的数据重新训练一下现有的Q235B抗拉强度模型，MAE和MAPE指标如何？"

2.  **模型部署与上线 (model_deployment_golive_request)**：
    * 描述：用户希望将一个已经训练完成、评估达标且满足特定业务条件的模型部署到实际生产环境或指定应用场景中，使其能够接收实时数据并提供预测服务。
    * 关键词：上线、部署、发布、应用模型、投入使用、启用、推送模型、集成、切换模型。
    * 例子：
        * "将模型库中ID为`RP02_N_Q235B_H033`的屈服强度模型部署到H033机组的生产线上。"
        * "将昨天训练完毕并且评估显示MAPE低于5%的抗拉强度模型`TS_N_S235JR_H033`上线。"

3.  **模型监控与优化 (model_monitoring_optimization_request)**：
    * 描述：用户希望查看当前已部署模型的实时或历史性能指标、运行状态、预警信息，或者基于监控结果对现有上线模型进行调整、优化、再训练以提升其表现或适应数据变化。
    * 关键词：监控、模型状态、性能指标、运行情况、预警、报警、性能衰退、优化模型、再训练、提升精度、更新模型。
    * 例子：
        * "监控当前H033机组上所有在线模型的运行状态和关键性能指标。"
        * "帮我验证一下模型ID为`TS_N_Q235B_H033`在过去一个月新数据上的预测稳定性。"
        * "查询H015机组抗拉强度预测模型最近30天的预测偏差和数据覆盖率。"
        * "H048机组的抗拉强度模型表现效果较差，需要进行优化调整。"

4.  **知识问答 (knowledge_qna_request)**：
    * 描述：用户进行一般性的咨询或查询，内容可能涉及钢铁产品性能、特定模型细节（如特征重要性）、生产工艺影响、钢种知识、术语解释或关于助手自身能力等方面。
    * 关键词：你是谁、你能做什么、解释、说明、影响、因素、重要性、排名、范围、定义、区别、查询、查看、牌号、性能、工艺参数、特征。
    * 例子：
        * "你好，能介绍下你的主要功能吗？"
        * "查询当前已上线用于预测Q355B牌号屈服强度的模型，它使用了哪些关键工艺特征？并按重要性排序。"
        * "我要查看当前已上线的Q235B牌号抗拉强度预报模型所使用的特征重要性排名。"

请仔细分析用户输入，并严格按照以下格式**仅返回**识别到的意图类别英文标签：
model_building_evaluation_request
或
model_deployment_golive_request
或
model_monitoring_optimization_request
或
knowledge_qna_request

如果无法明确判断意图，或者用户输入与上述意图均不相关，请返回：
unknown_intent"""
        intent_classification_user_prompt = user_request
        llm_intent_response = call_llm(
            system_prompt=INTENT_CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=intent_classification_user_prompt,
            temperature=0.1,  # 分类任务，较低的温度
            model="ds_v3"
        )

        # 正常返回意图字符串
        intent = llm_intent_response

        # 步骤2: 路由阶段，根据意图进行处理。
        current_date = datetime.now().strftime("%Y年%m月%d日")
        MODEL_BUILDING_INFO_EXTRACTION_SYSTEM_PROMPT = f"""你是一个信息提取助手。用户的请求是关于构建一个新的钢铁产品性能预报模型。
请从用户请求中提取以下关键信息，并以严格的JSON格式返回。
当前日期是：{current_date}。

需要提取的字段：
1.  `user_request`: 字符串，用户的原始请求文本。
2.  `sg_sign`: 字符串，钢种牌号，例如 "Q235B", "HRB400E", "CR420LA"。如果用户请求中未提及，则其值为 null。
3.  `target_metric`: 字符串，目标性能指标，例如 "抗拉强度", "屈服强度", "屈服延伸率"。如果无法从用户请求中明确识别，则其值为 "未知指标"。
4.  `time_range`: 字符串，数据时间范围，格式为 "YYYYMMDD-YYYYMMDD"。
    * 如果用户明确指定了时间范围，请准确提取并转换为 "YYYYMMDD-YYYYMMDD" 格式。
    * 如果用户提及相对时间如 "过去一年"、"最近一年"，请基于当前日期（{current_date}）计算，格式为 "YYYYMMDD-YYYYMMDD"。
    * 如果用户完全没有提及任何时间信息，则默认使用最近一年的数据，请基于当前日期（{current_date}）计算，格式为 "YYYYMMDD-YYYYMMDD"。
5.  `product_unit_no`: 字符串，生产机组号，例如 "H033", "C401"， "C612"。如果用户请求中未提及，则其值为 null。
6.  `st_no`: 字符串，出钢记号，例如 "AR3162E1", "AR3141E5", "DR4244E1"。如果用户请求中未提及，则其值为 null。

返回的JSON格式必须如下，所有字段都必须包含：
{{
  "user_request": "用户的原始请求文本",
  "sg_sign": "提取到的牌号或null",
  "target_metric": "提取到的目标性能指标或'未知指标'",
  "time_range": "提取到的时间范围YYYYMMDD-YYYYMMDD",
  "product_unit_no": "提取到的机组号或null",
  "st_no": "提取到的出钢记号或null"
}}

请确保所有字段都存在于返回的JSON中。

例如：
用户请求: "用过去一年的数据构建一个Q235B的屈服强度性能预报模型，机组H033"，当前日期是2025年6月5日。
你应该返回：
{{
  "user_request": "用过去一年的数据构建一个Q235B的屈服强度性能预报模型，机组H033",
  "sg_sign": "Q235B",
  "target_metric": "屈服强度",
  "time_range": "20240605-20250605",
  "product_unit_no": "H033",
  "st_no": null
}}

用户请求: "我想建一个抗拉强度的模型"，当前日期是2025年6月5日。
你应该返回：
{{
  "user_request": "我想建一个抗拉强度的模型",
  "sg_sign": null,
  "target_metric": "抗拉强度",
  "time_range": "20240605-20250605",
  "product_unit_no": null,
  "st_no": null
}}

用户请求: "用2024年的数据建一个Q345B的冲击功模型，在H033号机组上"，当前日期是2025年6月5日。
你应该返回：
{{
  "user_request": "用2024年的数据建一个H033机组的Q345B冲击功模型",
  "sg_sign": "Q345B",
  "target_metric": "冲击功",
  "time_range": "20240101-20241231",
  "product_unit_no": "H033",
  "st_no": null
}}"""
        if intent == "model_building_evaluation_request":
            # 对于模型构建请求，提取详细信息
            info_extraction_user_prompt = user_request
            llm_extraction_response = call_llm(
                system_prompt=MODEL_BUILDING_INFO_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=info_extraction_user_prompt,
                temperature=0.2,  # 提取任务，温度稍高一些
                model="ds_v3"
            )

            extracted_info = self._parse_llm_json_response(llm_extraction_response)

            if "error" in extracted_info:
                return {
                    "user_request": user_request,
                    "intent": intent,
                    "status": "model_construction_information_extraction_failed",
                    "error_details": extracted_info
                }

            # 确保关键字段存在
            pre_result = {
                "user_request": user_request,
                "intent": intent,
                "sg_sign": extracted_info.get("sg_sign"),
                "target_metric": extracted_info.get("target_metric"),
                "time_range": extracted_info.get("time_range"),
                "product_unit_no": extracted_info.get("product_unit_no"),
                "st_no": extracted_info.get("st_no")
            }

            # 新增对于"未知指标"的判定，因为"未知指标"意味着建模流程走不通，直接返回相关提示信息，要求用户重新确认需求。
            if pre_result["target_metric"] == "未知指标":
                return {
                    "user_request": user_request,
                    "intent": intent,
                    "status": "unknown_target_metric",
                    "error_details": "无法识别目标性能指标，请确认是否正确表述。"
                }

            # 目标性能字段符合要求的话，根据业务数据库中的相关知识进行字段映射。
            else:
                # 初始化知识库服务并搜索目标性能指标中英文对照列表
                kb = KnowledgeBaseService("professional_knowledge_kb")
                search_query = "目标性能指标字段映射对照列表。"
                results = kb.search(search_query, k=1)
                
                if not results:
                    return {
                        "user_request": user_request,
                        "intent": intent,
                        "status": "professional_knowledge_kb_search_failed",
                        "error_details": "无法从业务数据知识库中获取目标性能字段映射对照信息。"
                    }
                
                # 从搜索结果中获取目标性能指标列表
                target_metrics_list = results[0]["metadata"]
                
                # 构建LLM系统提示词，用于寻找最适配的字段对照
                FIELD_MAPPING_SYSTEM_PROMPT = f"""你是一个专业的钢铁产品性能指标映射助手。你的任务是将用户提供的目标性能指标名称映射到字段代码。
请根据以下目标性能指标列表，找出与用户输入最匹配的一个标准名称和对应的字段代码：
{json.dumps(target_metrics_list, ensure_ascii=False, indent=2)}
                
你需要考虑以下因素：
1. 完全匹配：用户输入与标准名称standard_name或别名aliases完全一致
2. 部分匹配：用户输入包含标准名称standard_name或别名aliases，或者标准名称standard_name或别名aliases包含用户输入

请返回一个JSON对象，包含以下字段：
- matched: 布尔值，表示是否找到匹配项
- field_code: 如果匹配，返回字段代码；如果不匹配，返回null

只返回JSON对象，不要有其他文本。"""
                
                # 调用LLM进行字段映射
                field_mapping_user_prompt = f"用户输入的目标性能指标是：{pre_result['target_metric']}"
                llm_mapping_response = call_llm(
                    system_prompt=FIELD_MAPPING_SYSTEM_PROMPT,
                    user_prompt=field_mapping_user_prompt,
                    temperature=0.1,  # 映射任务
                    model="ds_v3"
                )
                
                # 解析LLM返回的映射结果
                mapping_result = self._parse_llm_json_response(llm_mapping_response)

                if "error" in mapping_result:  # 检查解析或LLM调用是否有错
                    return {
                        "user_request": user_request,
                        "intent": intent,
                        "status": "field_mapping_failed",
                        "error_details": mapping_result
                    }
                
                # 检查是否找到匹配项
                if not mapping_result.get("matched", False) or mapping_result.get("field_code") is None:
                    return {
                        "user_request": user_request,
                        "intent": intent,
                        "status": "target_metric_not_matched",
                        "error_details": f"无法将'{pre_result['target_metric']}'映射到业务数据库中已知的目标性能指标字段。"
                    }
                
                # 更新pre_result中的target_metric字段为field_code，保持结构一致
                final_result = pre_result.copy()
                # 只替换target_metric字段的值，不添加额外字段
                final_result["target_metric"] = mapping_result["field_code"]
                
                return final_result

        # 模型部署上线请求
        elif intent == "model_deployment_golive_request":
            return {
                "user_request": user_request,
                "intent": intent
            }
        # 模型监控优化请求
        elif intent == "model_monitoring_optimization_request":
            return {
                "user_request": user_request,
                "intent": intent
            }
        # 性能预报知识问答请求
        elif intent == "knowledge_qna_request":
            return {
                "user_request": user_request,
                "intent": intent
            }
        # 不明确的请求
        elif intent == "unknown_intent":
            return {
                "user_request": user_request,
                "intent": "unknown_intent",
            }
        else:  # 非预期分类标签的情况
            return {
                "user_request": user_request,
                "intent": "undefined_intent_category",
                "received_intent_tag": intent,
            }
