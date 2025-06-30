import os

# LLM API 配置
OPENAI_API_BASE_DS = "https://ds.baocloud.cn/xin3plat/api/v1"
OPENAI_API_KEY_V3 = "fastgpt-rIm4TTWUfVkVdu7qPMjr39u1TlCjmdllAsevxEx7E156pe2pvlQUq4Xb7"
LLM_MODEL_NAME_V3 = "ds-v3"
OPENAI_API_KEY_R1 = "fastgpt-bIewczXe2SJrKb6Ov6ersSvCI4c8ymKTonsyiTCxZccsQHRgapJYS"
LLM_MODEL_NAME_R1 = "ds-R1"

OPENAI_API_BASE_QWEN = "http://acv-gydn.baocloud.cn/qwen2_5_vllm/v1"
OPENAI_API_KEY_QWEN = "none"
LLM_MODEL_NAME_QWEN = "Qwen2.5-72B-Instruct-AWQ"

EMBEDDING_API_URL = r"http://acv-gydn.baocloud.cn/embeddingbge/embed_string_m3/"

# 知识库文件路径
KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base", "vector_stores")
PREPROCESSING_KB_NAME = "data_preprocessing_kb"
FEATURE_ENGINEERING_KB_NAME = "feature_engineering_kb"
MODEL_SELECTION_KB_NAME = "model_selection_kb"
HISTORICAL_CASES_KB_NAME = "historical_cases_kb"
PROFESSIONAL_KNOWLEDGE_KB_NAME = "professional_knowledge_kb"

# 数据库连接信息
DB_CONFIG = {
    "host": "127.0.0.1",  # 本地开发环境
    # "host": "10.70.48.41",  # 服务器部署环境
    "user": "bgmszz00",
    "password": "bgmszz00bgta",
    "port": "7002",  # 本地开发环境端口
    # "port": "50021",  # 服务器部署环境端口
    "database": "bgbdprod"
}

# 种子配置
DEFAULT_RANDOM_STATE = 42
