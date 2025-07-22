import numpy as np
import openai
import requests
import config
from typing import Generator, Optional, Callable, Dict, Any, Tuple


def process_think_content(content: str, first_newline_skipped: bool) -> Tuple[str, bool]:
    """
    处理 <think> 标签内的内容，跳过第一个换行符并替换双换行符为单换行符。

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
    content = content.replace('\n\n', '\n')
    return content, first_newline_skipped


def call_llm_stream(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        model: str = "ds_R1"
) -> Generator[Dict[str, Any], None, None]:
    """
    调用LLM并以流式方式返回响应。

    参数:
    - system_prompt: 系统提示词。
    - user_prompt: 用户提示词。
    - temperature: 控制生成文本的随机性。
    - model: 模型名称。

    返回:
    - 一个生成器，逐块yield响应内容。
      - 'think_content': <think>标签内的内容。
      - 'text_content': 非<think>标签的内容。
      - 'error': 错误信息。
    """
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

    try:
        response_stream = openai.ChatCompletion.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            stream=True,
        )

        in_think_block = False
        buffer = ""
        first_newline_skipped = False  # 标记是否已经跳过第一个换行符
        for chunk in response_stream:
            content = chunk.choices[0].delta.get('content', '')
            if not content:
                continue

            buffer += content

            # 循环处理缓冲区中的内容，直到没有完整的标签可以处理
            while True:
                if not in_think_block:
                    start_tag_pos = buffer.find('<think>')
                    if start_tag_pos == -1:
                        # 缓冲区中没有<think>标签，可以安全地把大部分内容当作text_content发送
                        # 留一小部分以防标签被截断
                        safe_yield_len = len(buffer) - (len('<think>') - 1)
                        if safe_yield_len > 0:
                            yield {"text_content": buffer[:safe_yield_len]}
                            buffer = buffer[safe_yield_len:]
                        break  # 需要更多数据来确定

                    # 发现了<think>标签，产出它之前的所有内容
                    pre_think_content = buffer[:start_tag_pos]
                    if pre_think_content:
                        yield {"text_content": pre_think_content}

                    buffer = buffer[start_tag_pos + len('<think>'):]
                    in_think_block = True
                    first_newline_skipped = False  # 进入新的<think>块，重置标记

                # 进入 <think> 标签处理模式
                if in_think_block:
                    end_tag_pos = buffer.find('</think>')
                    if end_tag_pos == -1:
                        # 缓冲区中没有</think>标签，可以安全地把大部分内容当作think_content发送
                        # 留一小部分以防标签被截断
                        safe_yield_len = len(buffer) - (len('</think>') - 1)
                        if safe_yield_len > 0:
                            think_content = buffer[:safe_yield_len]
                            think_content, first_newline_skipped = process_think_content(think_content,
                                                                                         first_newline_skipped)
                            if think_content:
                                yield {"think_content": think_content}
                            buffer = buffer[safe_yield_len:]
                        break  # 需要更多数据来确定

                    # 发现了</think>标签，产出它之前的所有内容作为思考内容
                    thinking_content = buffer[:end_tag_pos]
                    thinking_content, first_newline_skipped = process_think_content(thinking_content,
                                                                                    first_newline_skipped)
                    if thinking_content:
                        yield {"think_content": thinking_content}

                    buffer = buffer[end_tag_pos + len('</think>'):]
                    in_think_block = False

        # 处理流结束后缓冲区里剩余的内容
        if buffer:
            if in_think_block:
                yield {"think_content": buffer}
            else:
                yield {"text_content": buffer}

    except Exception as e:
        error_message = f"Agent failed: {str(e)}"
        print(error_message)
        yield {"error": error_message}


def call_llm(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        model: str = "ds_R1"
) -> Generator[Dict[str, Any], None, str]:
    """
    一个生成器函数，它将思考过程逐字流式传输，并返回最终的非思考文本。对`call_llm_stream`的封装，实现了统一的输出格式。

    :return: 一个生成器，用于生成思考流、错误，并最终返回收集到的响应字符串。
    """
    full_response_parts = []
    error_message = ""

    # 这是一个生成器，yield思考块
    for chunk in call_llm_stream(system_prompt, user_prompt, temperature, model):
        if "error" in chunk:
            error_message = chunk["error"]
            yield {"type": "error",
                   "payload": {"stage": "智能体调用", "message": "智能体调用失败", "details": error_message}}
            break
        elif "think_content" in chunk and chunk["think_content"]:
            # 将获取到的思考内容逐字yield，实现流式效果
            for char in chunk["think_content"]:
                yield {"type": "thinking_stream", "payload": char}
        elif "text_content" in chunk:
            full_response_parts.append(chunk["text_content"])

    if error_message:
        return f"Agent failed: {error_message}"

    # 使用return在生成器结束时返回最终拼接好的结果
    final_response = "".join(full_response_parts)
    final_response = final_response.replace('```json', '').replace('```', '').strip()
    return final_response


def get_embedding(text: str) -> np.ndarray:
    """
    调用嵌入API获取文本的嵌入向量。

    参数:
        text (str): 需要获取嵌入的文本。

    返回:
        np.ndarray: 文本的嵌入向量，空数组表示失败.
    """
    try:
        response = requests.post(config.EMBEDDING_API_URL, json={"prompt": text})
        response.raise_for_status()  # 如果HTTP请求返回了不成功的状态码，则抛出HTTPError
        embedding_str = response.json().get("embedding")
        if embedding_str:
            # 将字符串形式的嵌入向量转换为列表
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
