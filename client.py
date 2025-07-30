import requests
import json
import sys


def stream_automl_response(query: str):
    """流式接收AutoML API响应"""
    url = "http://localhost:8003/automl-stream"
    headers = {"Content-Type": "application/json"}
    data = {"query": query}

    with requests.post(url, json=data, headers=headers, stream=True) as response:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data:'):
                    event_data = decoded_line[5:].strip()
                    try:
                        event_json = json.loads(event_data)
                        print(f"{event_json}")
                    except json.JSONDecodeError:
                        print(f"无法解析的事件数据: {decoded_line}")


if __name__ == "__main__":
    test_query = "用过去两年Q235B牌号的数据构建一个抗拉强度预报模型，使用XGboost，用贝叶斯优化利用R2进行超参数优化，快速验证效果，迭代次数设置为15次，可以创建一些合理的多项式或者比率交互特征，我的可接受误差浮动是正负20以内，不需要进行异常值的识别和处理。"
    # test_query = "目前在线的Q235B模型的效果怎么样？"
    stream_automl_response(test_query)
