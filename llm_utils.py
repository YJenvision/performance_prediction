import numpy as np
import openai
import requests
import json
import config
from typing import Generator, Dict, Any, Tuple


def process_think_content(content: str, first_newline_skipped: bool) -> Tuple[str, bool]:
    """
    处理 <think> 标签内的内容，跳过第一个换行符并替换双换行符为单换行符（压缩多余空行）。

    参数:
    - content: <think> 标签内的原始内容。
    - first_newline_skipped: 标记是否已经跳过第一个换行符。

    返回:
    - 处理后的内容和更新后的 first_newline_skipped 标记。
    """
    if not first_newline_skipped:
        first_newline_index = content.find('\n')
        if first_newline_index != -1:
            content = content[first_newline_index + 1:]
            first_newline_skipped = True
    # 压缩多余空行
    content = content.replace('\n\n', '\n')
    return content, first_newline_skipped


def call_llm_stream(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        model: str = "qwen_plus"
) -> Generator[Dict[str, Any], None, None]:
    """
    根据模型类型（HTTP接口或OpenAI兼容接口）流式获取LLM响应，并区分“思考内容”（<think>标签内）和“普通文本”（标签外）。

    参数:
    - system_prompt: 系统提示词。
    - user_prompt: 用户提示词。
    - temperature: 控制生成文本的随机性。
    - model: 模型名称, "qwen_plus"会触发HTTP调用。

    返回:
    - 一个生成器，逐块yield响应内容。
      - 'think_content': <think>标签内的内容。
      - 'text_content': 非<think>标签的内容。
      - 'error': 错误信息。
    """

    def get_stream_content_generator() -> Generator[str, None, None]:
        """
        内部生成器函数，根据模型选择不同的API调用方式，并yield内容块。
        1. 对于qwen_plus，使用requests进行HTTP流式调用。通过requests库调用HTTP接口（SSE流式响应）
        构造请求头（包含API密钥）和payload（流式开关、提示词），迭代解析SSE格式响应（data: 前缀），提取content字段并yield
        2. 对于其他模型，使用openai兼容接口。
        """
        # 如果模型是 qwen_plus，则使用 requests 进行 HTTP POST 调用
        if model == "qwen_plus":
            url = config.QWEN_PLUS_URL
            headers = {
                'Authorization': f'Bearer {config.QWEN_PLUS_API_KEY}',
                'Content-Type': 'application/json'
            }
            payload = {
                "stream": True,
                "detail": False,
                "variables": {
                    "system": system_prompt
                },
                "messages": [
                    {"role": "user", "content": user_prompt}
                ]
            }
            # 使用上下文管理器确保请求被正确关闭
            with requests.post(url, headers=headers, json=payload, stream=True) as response:
                response.raise_for_status()  # 检查请求是否成功
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        # 检查并处理SSE（Server-Sent Events）格式
                        if decoded_line.startswith('data: '):
                            json_str = decoded_line[len('data: '):]
                            if json_str.strip() == '[DONE]':
                                break
                            try:
                                chunk = json.loads(json_str)
                                content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if content:
                                    yield content
                            except (json.JSONDecodeError, IndexError):
                                print(f"Warning: Could not decode or parse JSON from stream: {json_str}")
                                continue
        # 其他模型使用openai兼容接口逻辑
        else:
            if model == "ds_v3":
                LLM_MODEL_NAME = config.LLM_MODEL_NAME_V3
                openai.api_base = config.OPENAI_API_BASE_DS
                openai.api_key = config.OPENAI_API_KEY_V3
            elif model == "ds_R1":
                LLM_MODEL_NAME = config.LLM_MODEL_NAME_R1
                openai.api_base = config.OPENAI_API_BASE_DS
                openai.api_key = config.OPENAI_API_KEY_R1
            elif model == "Qwen2.5-72B-Instruct-AWQ":
                LLM_MODEL_NAME = config.LLM_MODEL_NAME_QWEN
                openai.api_base = config.OPENAI_API_BASE_QWEN
                openai.api_key = config.OPENAI_API_KEY_QWEN
            else:
                return

            response_stream = openai.ChatCompletion.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                stream=True,
            )
            for chunk in response_stream:
                content = chunk.choices[0].delta.get('content', '')
                if content:
                    yield content

    # 主处理逻辑：解析流式内容，区分思考/普通文本
    # 对于所有模型的流式响应都通用
    try:
        content_stream = get_stream_content_generator()

        in_think_block = False
        buffer = ""
        first_newline_skipped = False
        for content in content_stream:
            buffer += content

            # 循环处理缓冲区中的内容，直到没有完整的标签可以处理
            while True:
                if not in_think_block:
                    start_tag_pos = buffer.find('<think>')
                    if start_tag_pos == -1:
                        safe_yield_len = len(buffer) - (len('<think>') - 1)
                        if safe_yield_len > 0:
                            yield {"text_content": buffer[:safe_yield_len]}
                            buffer = buffer[safe_yield_len:]
                        break

                    pre_think_content = buffer[:start_tag_pos]
                    if pre_think_content:
                        yield {"text_content": pre_think_content}

                    buffer = buffer[start_tag_pos + len('<think>'):]
                    in_think_block = True
                    first_newline_skipped = False

                if in_think_block:
                    end_tag_pos = buffer.find('</think>')
                    if end_tag_pos == -1:
                        safe_yield_len = len(buffer) - (len('</think>') - 1)
                        if safe_yield_len > 0:
                            think_content, first_newline_skipped = process_think_content(buffer[:safe_yield_len],
                                                                                         first_newline_skipped)
                            if think_content:
                                yield {"think_content": think_content}
                            buffer = buffer[safe_yield_len:]
                        break

                    thinking_content, first_newline_skipped = process_think_content(buffer[:end_tag_pos],
                                                                                    first_newline_skipped)
                    if thinking_content:
                        yield {"think_content": thinking_content}

                    buffer = buffer[end_tag_pos + len('</think>'):]
                    in_think_block = False

        # 处理流结束后缓冲区里剩余的内容
        if buffer:
            if in_think_block:
                remaining_content, first_newline_skipped = process_think_content(buffer, first_newline_skipped)
                if remaining_content:
                    yield {"think_content": remaining_content}
            else:
                yield {"text_content": buffer}

    except Exception as e:
        error_message = f"Agent failed: {str(e)}"
        yield {"error": error_message}


def call_llm(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        model: str = "qwen_plus"
) -> Generator[Dict[str, Any], None, str]:
    """
    一个生成器函数，通过实时消费底层流式数据（call_llm_stream）并即时处理输出标准化事件流（思考流、错误、最终结果）。

    :return: 一个生成器，用于生成思考流、错误，并最终返回收集到的响应字符串。
    """
    full_response_parts = []  # 累积普通文本
    error_message = ""  # 错误信息
    think_buffer = ""  # 累积思考内容
    for chunk in call_llm_stream(system_prompt, user_prompt, temperature, model):
        if "error" in chunk:
            error_message = chunk["error"]
            yield {"type": "error",
                   "payload": {"stage": "智能体调用", "details": "智能体调用失败" + error_message}}
            break
        elif "think_content" in chunk and chunk["think_content"]:
            think_buffer += chunk["think_content"]
            while '\n' in think_buffer:
                line, think_buffer = think_buffer.split('\n', 1)
                yield {"type": "thinking_stream", "payload": line + '\n'}
        elif "text_content" in chunk:
            full_response_parts.append(chunk["text_content"])

    if think_buffer:
        yield {"type": "thinking_stream", "payload": think_buffer}

    if error_message:
        return f"Agent failed: {error_message}"

    final_response = "".join(full_response_parts)
    final_response = final_response.replace('```json', '').replace('```', '').strip()

    if final_response == "":
        yield {"type": "error", "payload": {"stage": "智能体调用", "details": "智能体调用成功但响应为空"}}
        return "Agent failed: 智能体调用成功但响应为空"

    return final_response


def get_embedding(text: str) -> np.ndarray:
    """
    调用嵌入API获取文本的数值向量（用于语义相似度计算等）。
    参数: text (str): 需要获取嵌入的文本。
    返回: np.ndarray: 文本的嵌入向量，空数组表示失败.
    """
    try:
        response = requests.post(config.EMBEDDING_API_URL, json={"prompt": text})
        response.raise_for_status()
        embedding_str = response.json().get("embedding")
        if embedding_str:
            embedding_list = [float(num) for num in embedding_str.strip('[]').split(',')]
            return np.array(embedding_list, dtype='float32')
        else:
            print(f"Error：在文本响应中找不到嵌入：{text[:30]}...")
            return np.array([])
    except requests.exceptions.RequestException as e:
        print(f"获取文本的嵌入时出错：'{text[:30]}...': {e}")
        return np.array([])
    except Exception as e:
        print(f"get_embedding过程中发生意外错误：{e}")
        return np.array([])
