import json
import uvicorn
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any, AsyncGenerator, Generator

from intent_recognition_routing.intent_recognizer import SteelPerformanceIntentRecognizer
from steel_automl.performance_model_builder import performanceModelBuilder

app = FastAPI(
    title="Streaming AutoML Agent API",
    description="一个用于回归任务的流式钢铁产品力学性能预报智能体API",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UserRequest(BaseModel):
    query: str


async def stream_wrapper(generator: Generator) -> AsyncGenerator[str, None]:
    """
    一个异步包装器，它将同步生成器的输出转换为异步的、格式化的SSE事件流。
    这个函数本身不返回任何值，只负责产生（yield）数据。
    """
    for chunk in generator:
        # 处理包含不可序列化类型的数据
        def default_serializer(obj):
            if isinstance(obj, set):
                return list(obj)  # 将set转换为list
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        # 将每个块格式化为 Server-Sent Event (SSE) 格式
        yield f"data: {json.dumps(chunk, ensure_ascii=False, default=default_serializer)}\n\n"
        # 短暂暂停，允许其他异步任务运行，防止事件循环阻塞
        await asyncio.sleep(0.01)


async def process_request_stream(user_query: str) -> AsyncGenerator[str, None]:
    """
    处理用户请求并流式返回结果的主函数。
    它消费来自下层生成器的事件流，并做出逻辑决策。
    """
    # --- 阶段1: 意图识别 ---
    intent_final_payload = None
    has_failed = False

    recognizer = SteelPerformanceIntentRecognizer()
    intent_sync_generator = recognizer.recognize_intent(user_query)

    # 异步遍历来自意图识别器的事件流
    async for event_string in stream_wrapper(intent_sync_generator):
        # 1. 将事件流直接转发给客户端
        yield event_string

        # 2. 在后端同时解析事件内容，以进行流程控制
        try:
            # 从SSE格式中解析出JSON数据
            chunk = json.loads(event_string.replace("data: ", "").strip())
            payload = chunk.get("payload", {})

            if chunk.get("type") == "stage_completed":
                if payload.get("stage") == "意图识别" and payload.get("status") == "success":
                    intent_final_payload = payload.get("result")

            # 检查是否有失败信号
            if chunk.get("type") == "error" or \
                    (chunk.get("type") == "status_update" and payload.get("status") == "failed"):
                has_failed = True
        except json.JSONDecodeError:
            # 如果解析失败，打印错误但继续，以防流中有非JSON数据
            print(f"Warning: Could not decode JSON from event string: {event_string}")

    # 意图识别流程结束后，检查其结果
    if has_failed or not intent_final_payload:
        # 如果失败或没有得到最终结果，则整个请求流程终止
        return

    # --- 阶段2: AutoML 建模  ---
    if intent_final_payload.get("intent") == "model_building_evaluation_request":
        automl_sync_generator = performanceModelBuilder(intent_final_payload)
        # 直接将AutoML流程的事件流转发给客户端
        async for event_string in stream_wrapper(automl_sync_generator):
            yield event_string


@app.post("/performance_prediction_agent")
async def automl_stream_endpoint(request: UserRequest):
    """
    接收用户自然语言请求，流式返回AutoML流程的状态和结果。
    """
    return StreamingResponse(
        process_request_stream(request.query),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
