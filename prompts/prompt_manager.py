# @Time    : 2025/7/25 10:33
# @Author  : ZhangJingLiang
# @Email   : jinglianglink@qq.com
# @Project : performance_prediction_agent_stream

import yaml

_prompts = None


def load_prompts(path: str = 'prompts/prompts.yml') -> None:
    """
    从指定的YAML文件加载提示词到全局变量中。
    """
    global _prompts
    try:
        with open(path, 'r', encoding='utf-8') as f:
            _prompts = yaml.safe_load(f)
    except FileNotFoundError:
        raise Exception(f"提示词配置文件未找到: {path}")
    except Exception as e:
        raise Exception(f"加载或解析提示词YAML文件时出错: {e}")


def get_prompt(key_path: str, **kwargs) -> str:
    """
    通过点分隔的键名获取并格式化一个提示词。

    例如: get_prompt('intent_recognizer.classify_intent.system')
         get_prompt('preprocessor.coarse_feature_screening.system', target_metric="YS")

    Args:
        key_path (str): 点分隔的键名路径 (e.g., 'data_loader.generate_sql.system').
        **kwargs: 用于格式化提示词模板的键值对。

    Returns:
        str: 格式化后的提示词字符串。
    """
    if _prompts is None:
        # 如果尚未加载，则执行初次加载
        load_prompts()

    keys = key_path.split('.')
    prompt_template = _prompts
    try:
        for key in keys:
            prompt_template = prompt_template[key]
    except KeyError:
        raise KeyError(f"提示词键 '{key_path}' 在配置中未找到。")

    if not isinstance(prompt_template, str):
        raise TypeError(f"键 '{key_path}' 对应的值不是一个字符串模板。")

    # 使用传入的参数格式化字符串
    return prompt_template.format(**kwargs)


# 在模块导入时自动加载一次提示词
load_prompts()
