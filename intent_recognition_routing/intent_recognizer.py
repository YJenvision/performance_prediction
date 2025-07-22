# @Time    : 2025/6/19 10:01
# @Author  : ZhangJingLiang
# @Email   : jinglianglink@qq.com
# @Project : performance_prediction_agent

import json
from datetime import datetime, timedelta
from llm_utils import call_llm
from knowledge_base.kb_service import KnowledgeBaseService
from typing import Dict, Any, Generator, Iterator


def parse_llm_json_response(llm_response_str: str) -> dict:
    if isinstance(llm_response_str, str) and "大语言模型调用失败" in llm_response_str:
        return {"error": "LLM_CALL_FAILED", "details": llm_response_str}
    try:
        start_index = llm_response_str.find('{')
        end_index = llm_response_str.rfind('}')
        if start_index != -1 and end_index != -1:
            json_str = llm_response_str[start_index:end_index + 1]
            return json.loads(json_str)
        raise json.JSONDecodeError("No JSON object found", llm_response_str, 0)
    except json.JSONDecodeError as e:
        return {"error": "JSON_PARSE_FAILED", "details": str(e), "original_response": llm_response_str}
    except Exception as e:
        return {"error": "RESPONSE_PROCESSING_ERROR", "details": str(e), "original_response": llm_response_str}


class SteelPerformanceIntentRecognizer:
    """意图识别器"""

    def __init__(self):
        """初始化意图识别器。"""
        pass

    def recognize_intent(self, user_request: str) -> Generator[Dict[str, Any], None, None]:
        """
        识别用户请求的意图，并根据意图提取信息。这是一个生成器，它会流式返回状态更新和思考过程，并在流程结束时yield最终结果或错误。

        :param user_request: 用户的原始请求字符串。
        :return: 一个生成器，它产生 status/thinking 更新并返回一个包含意图和提取信息的最终字典。

        建模与评估意图下的返回信息：
        {
            "user_request": user_request,
            "intent": intent,
            "sg_sign": ,
            "target_metric": ,
            "time_range": ,
            "product_unit_no": ,
            "st_no": ,
            "steel_grade":
        }
        其他意图返回信息：
        {
            "user_request": user_request,
            "intent": intent
        }
        - 失败：
        {
            "user_request": user_request,
            "intent": intent,
            "status": "",
            "error_details": e
        }
        """
        current_stage = "意图识别"
        # 步骤1: 意图分类
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "开始进行意图识别..."}}

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

请仔细分析用户输入，并严格按照以下格式**仅返回**识别到的意图类别英文标签(仅返回英文标签，禁止返回其他解释性或描述性语句。)：
model_building_evaluation_request
或
model_deployment_golive_request
或
model_monitoring_optimization_request
或
knowledge_qna_request

如果无法明确判断意图，或者用户输入与上述意图均不相关，请返回：
unknown_intent"""
        # 使用 yield from 来流式传输思考过程，并捕获最终结果
        intent_gen = call_llm(
            system_prompt=INTENT_CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=user_request,
            temperature=0,
            model="ds_R1"
        )
        intent = "空意图。"
        while True:
            try:
                chunk = next(intent_gen)
                if chunk.get("type") == "error":
                    yield chunk  # 将错误直接传递出去
                    return  # 发生错误，立即终止
                yield chunk  # 传递 thinking_stream
            except StopIteration as e:
                intent = e.value.strip()
                break

        if not intent or "Agent failed" in intent:
            yield {"type": "error", "payload": {"stage": current_stage, "message": "意图分类失败", "details": intent}}
            return

        # 步骤2: 根据意图进行路由处理
        current_date = datetime.now()

        if intent == "model_building_evaluation_request":
            yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                        "detail": "意图为预测模型的构建与评估，开始提取相关关键信息..."}}

            MODEL_BUILDING_INFO_EXTRACTION_SYSTEM_PROMPT = f"""你是一个信息提取助手。用户的请求是关于构建一个新的钢铁产品性能预报模型。
            请从用户请求中提取以下关键信息，并以严格的JSON格式返回。
            当前日期是：{current_date.strftime("%Y年%m月%d日")}。

            需要提取的字段：
            1.  `user_request`: 字符串，用户的原始请求文本。
            2.  `sg_sign`: 字符串数组，钢种牌号。如果用户提及多个（如 "Q235B和Q345B"），请将它们全部提取到一个列表中 `["Q235B", "Q345B"]`。如果只提及一个，也放入列表中 `["Q235B"]`。如果未提及，则其值为 null。
            3.  `target_metric`: 字符串，目标性能指标，例如 "抗拉强度", "屈服强度"。如果无法从用户请求中明确识别，则其值为 "未知指标"。
            4.  `time_range`: 字符串，数据时间范围，格式为 "YYYYMMDD-YYYYMMDD"。
                * 如果用户明确指定了时间范围，请准确提取并转换为 "YYYYMMDD-YYYYMMDD" 格式。
                * 如果用户提及相对时间如 "过去一年"、"最近半年"，请基于当前日期（{current_date.strftime("%Y年%m月%d日")}）计算。
                * 如果用户完全没有提及任何时间信息，则默认使用最近一年的数据。
            5.  `product_unit_no`: 字符串数组，生产机组号。如果用户提及多个（如 "H033和H043"），请将它们提取为 `["H033", "H043"]`。如果只提及一个，则为 `["H033"]`。如果未提及，则其值为 null。
            6.  `st_no`: 字符串数组，出钢记号。处理方式同上。如果未提及，则其值为 null。
            7.  `steel_grade`: 字符串数组，钢种。处理方式同上。如果未提及，则其值为 null，可能的取值有 'LA'、'CM'、'DP'、'AV'等等。

            返回的JSON格式必须如下，所有字段都必须包含：
            {{
              "user_request": "用户的原始请求文本",
              "sg_sign": ["提取到的牌号1", "提取到的牌号2"] or null,
              "target_metric": "提取到的目标性能指标或'未知指标'",
              "time_range": "计算出的时间范围YYYYMMDD-YYYYMMDD",
              "product_unit_no": ["提取到的机组号1", "提取到的机组号2"] or null,
              "st_no": ["提取到的出钢记号1", "提取到的出钢记号2"] or null,
              "steel_grade": ["提取到的钢种1", "提取到的钢种2"] or null
            }}

            请确保所有字段都存在于返回的JSON中，并且只返回JSON对象，不要包含任何额外的解释或文本。

            例如：
            用户请求: "用过去一年的数据为H033和H043机组构建一个Q235B的屈服强度性能预报模型"，当前日期是{current_date.strftime("%Y年%m月%d日")}。
            你应该返回：
            {{
              "user_request": "用过去一年的数据为H033和H043机组构建一个Q235B的屈服强度性能预报模型",
              "sg_sign": ["Q235B"],
              "target_metric": "屈服强度",
              "time_range": "{(current_date - timedelta(days=365)).strftime("%Y%m%d")}-{current_date.strftime("%Y%m%d")}",
              "product_unit_no": ["H033", "H043"],
              "st_no": null,
              "steel_grade": null
            }}

            用户请求: "我想建一个抗拉强度的模型"，当前日期是{current_date.strftime("%Y年%m月%d日")}。
            你应该返回：
            {{
              "user_request": "我想建一个抗拉强度的模型",
              "sg_sign": null,
              "target_metric": "抗拉强度",
              "time_range": "{(current_date - timedelta(days=365)).strftime("%Y%m%d")}-{current_date.strftime("%Y%m%d")}",
              "product_unit_no": null,
              "st_no": null,
              "steel_grade": null
            }}

            用户请求: "我想建一个LA和CM钢种的断裂延伸率的模型"，当前日期是{current_date.strftime("%Y年%m月%d日")}。
            你应该返回：
            {{
              "user_request": "我想建一个抗拉强度的模型",
              "sg_sign": null,
              "target_metric": "抗拉强度",
              "time_range": "{(current_date - timedelta(days=365)).strftime("%Y%m%d")}-{current_date.strftime("%Y%m%d")}",
              "product_unit_no": null,
              "st_no": null,
              "steel_grade": ["LA", "CM"],
            }}
            """

            llm_extraction_response_gen = call_llm(
                system_prompt=MODEL_BUILDING_INFO_EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_request,
                temperature=0,
                model="ds_v3"
            )
            llm_extraction_response = ""
            while True:
                try:
                    chunk = next(llm_extraction_response_gen)
                    if chunk.get("type") == "error":
                        yield chunk
                        return
                    yield chunk
                except StopIteration as e:
                    llm_extraction_response = e.value
                    break

            extracted_info = parse_llm_json_response(llm_extraction_response)

            if "error" in extracted_info:
                yield {"type": "error",
                       "payload": {"stage": current_stage, "message": "相关关键信息提取失败", "details": extracted_info}}
                return

            yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                        "detail": "相关关键信息提取成功，开始映射目标性能指标..."}}

            pre_result = {
                "user_request": user_request, "intent": intent,
                "sg_sign": extracted_info.get("sg_sign"), "target_metric": extracted_info.get("target_metric"),
                "time_range": extracted_info.get("time_range"),
                "product_unit_no": extracted_info.get("product_unit_no"),
                "st_no": extracted_info.get("st_no"), "steel_grade": extracted_info.get("steel_grade")
            }

            if pre_result["target_metric"] == "未知指标":
                yield {"type": "error", "payload": {"stage": current_stage, "message": "无法识别目标性能指标",
                                                    "details": "请求中未明确指定有效的力学性能指标。"}}
                return

            kb = KnowledgeBaseService("professional_knowledge_kb")
            results = kb.search("目标性能指标字段映射对照列表。", k=1)
            if not results:
                yield {"type": "error", "payload": {"stage": current_stage, "message": "知识库查询失败",
                                                    "details": "无法获取目标性能指标的映射列表。"}}
                return

            target_metrics_list = results[0]["metadata"]

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

            llm_mapping_response_gen = call_llm(
                system_prompt=FIELD_MAPPING_SYSTEM_PROMPT,
                user_prompt=f"用户输入的目标性能指标是：{pre_result['target_metric']}",
                temperature=0, model="ds_v3"
            )
            while True:
                try:
                    chunk = next(llm_mapping_response_gen)
                    if chunk.get("type") == "error":
                        yield chunk
                        return
                    yield chunk
                except StopIteration as e:
                    llm_mapping_response = e.value
                    break

            mapping_result = parse_llm_json_response(llm_mapping_response)

            if "error" in mapping_result or not mapping_result.get("matched", False):
                yield {"type": "error",
                       "payload": {"stage": current_stage, "message": "目标指标映射失败", "details": mapping_result}}
                return

            final_result = pre_result.copy()
            final_result["target_metric"] = mapping_result["field_code"]

            yield {"type": "status_update",
                   "payload": {"stage": current_stage, "status": "success", "detail": "意图识别和信息提取全部完成。"}}
            yield {"type": "intent_result", "payload": final_result}

        else:  # 其他意图
            result = {"user_request": user_request, "intent": intent}
            yield {"type": "status_update",
                   "payload": {"stage": current_stage, "status": "success", "detail": f"意图识别完成，意图为: {intent}"}}
            yield {"type": "intent_result", "payload": result}
