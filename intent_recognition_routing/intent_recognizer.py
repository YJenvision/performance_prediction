# @Time    : 2025/6/19 10:01
# @Author  : ZhangJingLiang
# @Email   : jinglianglink@qq.com
# @Project : performance_prediction_agent

import json
from datetime import datetime, timedelta
from llm_utils import call_llm
from knowledge_base.kb_service import KnowledgeBaseService
from typing import Dict, Any, Generator
from prompts.prompt_manager import get_prompt


def parse_llm_json_response(llm_response_str: str) -> dict:
    """解析LLM返回的JSON字符串，增强了鲁棒性。"""
    if isinstance(llm_response_str, str) and "Agent failed" in llm_response_str:
        return {"error": "LLM_CALL_FAILED", "details": llm_response_str}
    try:
        # 寻找第一个 '{' 和最后一个 '}' 来提取JSON对象
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
    """
    意图识别器
    - 优化了返回格式以支持前端分步展示。
    - 使用 'substage_result' 传递中间结果。
    - 使用 'stage_completed' 作为统一的阶段结束信号。
    """

    def __init__(self):
        """初始化意图识别器。"""
        pass

    def recognize_intent(self, user_request: str) -> Generator[Dict[str, Any], None, None]:
        """
        识别用户请求的意图，并根据意图提取信息。
        """
        current_stage = "意图识别"
        # 步骤1: 意图分类
        yield {"type": "status_update",
               "payload": {"stage": current_stage, "status": "running", "detail": "开始进行意图识别..."}}

        # 从配置文件获取提示词
        intent_classification_prompt = get_prompt('intent_recognizer.classify_intent.system')

        # 使用 yield from 来流式传输思考过程，并捕获最终结果
        intent_gen = call_llm(
            system_prompt=intent_classification_prompt,
            user_prompt=user_request,
            temperature=0,
            model="ds_v3"
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
            yield {"type": "error", "payload": {"stage": current_stage, "detail": f"意图分类失败: {intent}"}}
            return

        # 步骤2: 根据意图进行路由处理
        current_date = datetime.now()

        if intent == "model_building_evaluation_request":
            yield {"type": "status_update", "payload": {"stage": current_stage, "status": "running",
                                                        "detail": "意图为预测模型的构建与评估，开始提取相关关键信息..."}}

            # 从配置文件获取提示词并格式化
            info_extraction_prompt = get_prompt(
                'intent_recognizer.extract_model_building_info.system',
                current_date=current_date.strftime("%Y年%m月%d日")
            )

            llm_extraction_response_gen = call_llm(
                system_prompt=info_extraction_prompt,
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
                       "payload": {"stage": current_stage,
                                   "detail": "相关关键信息提取失败，错误信息：" + str(extracted_info)}}
                return
            extracted_info_json = json.dumps(extracted_info, indent=2, ensure_ascii=False)
            print("意图识别：提取相关关键信息：", extracted_info_json)
            yield {"type": "substage_result", "payload": {
                "stage": current_stage,
                "substage_title": "提取的相关关键信息",
                "data": extracted_info_json
            }}

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
                yield {"type": "error", "payload": {"stage": current_stage,
                                                    "detail": "无法识别目标性能指标，请求中未明确指定有效的力学性能指标。"}}
                return

            kb = KnowledgeBaseService("professional_knowledge_kb")
            results = kb.search("目标性能指标字段映射对照列表。", k=1)
            if not results:
                yield {"type": "error", "payload": {"stage": current_stage,
                                                    "detail": "知识库查询失败，无法获取目标性能指标的映射列表。"}}
                return

            target_metrics_list = results[0]["metadata"]

            # 从配置文件获取提示词并格式化
            field_mapping_prompt = get_prompt(
                'intent_recognizer.map_target_metric.system',
                target_metrics_list=json.dumps(target_metrics_list, ensure_ascii=False, indent=2)
            )

            llm_mapping_response_gen = call_llm(
                system_prompt=field_mapping_prompt,
                user_prompt=f"用户输入的目标性能指标是：{pre_result['target_metric']}",
                temperature=0, model="ds_v3"
            )
            llm_mapping_response = ""
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
                       "payload": {"stage": current_stage, "detail": "目标性能指标映射失败" + str(mapping_result)}}
                return

            yield {"type": "substage_result", "payload": {
                "stage": current_stage,
                "substage_title": "映射的目标性能指标",
                "data": mapping_result["field_code"]
            }}

            final_result = pre_result.copy()
            final_result["target_metric"] = mapping_result["field_code"]

            if not final_result["target_metric"]:
                yield {"type": "error", "payload": {"stage": current_stage,
                                                    "detail": "智能体未能解析出性能目标字段，请重新确认相关性能指标后再试。"}}
                return

            yield {"type": "status_update",
                   "payload": {"stage": current_stage, "status": "success", "detail": "完成建模和评估流程的必要信息提取。"}}

            yield {"type": "stage_completed", "payload": {
                "stage": current_stage,
                "status": "success",
                "result": final_result
            }}

        else:  # 其他意图
            result = {"user_request": user_request, "intent": intent}
            yield {"type": "status_update",
                   "payload": {"stage": current_stage, "status": "success", "detail": f"意图识别完成，意图为: {intent}"}}

            yield {"type": "stage_completed", "payload": {
                "stage": current_stage,
                "status": "success",
                "result": result
            }}
