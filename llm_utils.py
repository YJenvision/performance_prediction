import numpy as np
import openai
import requests
import config


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3, model: str = "ds_R1") -> str:
    """
    调用LLM。

    参数:
    - system_prompt: 系统提示词。
    - user_prompt: 用户提示词。
    - temperature: 控制生成文本的随机性。
    - model: 模型名称，可选值为 'ds_v3'、'ds_R1' 和 'Qwen2.5-72B-Instruct-AWQ'。

    返回:
    - LLM生成的文本回复。
    - 失败返回
        “大语言模型调用失败，提示：str(e)”
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
        raise ValueError("模型名称无效，支持的模型有 'ds_v3'、'ds_R1' 和 'Qwen2.5-72B-Instruct-AWQ'。")

    try:
        result = openai.ChatCompletion.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            stream=False,
        )
        answer = result.choices[0].message.content

        # 处理<think>标签内容
        think_start = answer.find('<think>')
        think_end = answer.find('</think>')
        if think_start != -1 and think_end != -1:
            think_content = answer[think_start + 7:think_end].strip()
            print(f"[智能体思考过程]: {think_content}")
            answer = answer[:think_start] + answer[think_end + 8:]

        answer = answer.replace('```json', '').replace('```', '').strip()
        return answer

    except Exception as e:
        print(f"局部智能体调用失败: {e}")
        # 返回表示错误的特定字符串，以便上层调用者处理
        error_message = f"大语言模型调用失败，提示：{str(e)}"

        return error_message


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
            return np.array([])  # 返回空数组表示失败
    except requests.exceptions.RequestException as e:
        print(f"获取文本的嵌入时出错：'{text[:30]}...': {e}")
        return np.array([])  # 返回空数组表示失败
    except Exception as e:
        print(f"get_embedding过程中发生意外错误：{e}")
        return np.array([])  # 返回空数组表示失败


if __name__ == '__main__':

    # 测试嵌入调用
    test_text_for_embedding = "测试文本嵌入"
    embedding_vector = get_embedding(test_text_for_embedding)
    print(type(embedding_vector))
    print(f"文本 '{test_text_for_embedding}' 的嵌入向量: {embedding_vector}")
    print(f"向量维度: {len(embedding_vector)}")
