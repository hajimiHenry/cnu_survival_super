"""
配置文件 - 存放模型配置和路径常量
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
RULES_DIR = DATA_DIR / "rules"
FLOWS_DIR = DATA_DIR / "flows"
EXPERIENCE_DIR = DATA_DIR / "experience"
TEMPLATES_DIR = DATA_DIR / "templates"
DOCS_DIR = DATA_DIR / "docs"

# 向量索引目录
INDEX_DIR = PROJECT_ROOT / "index"

# 数据目录到类别的映射
CATEGORY_MAP = {
    "rules": "制度",
    "flows": "流程",
    "experience": "经验",
    "templates": "模板",
    "docs": "培养方案"
}

# LLM 配置（用于对话）
# 请在环境变量或 .env 文件中设置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "claude-opus-4-5-20251101")

# 本地嵌入模型配置（免费，无需 API）
# 使用支持中文的多语言模型
LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 文本切分配置
CHUNK_SIZE = 500  # 每段约500字
CHUNK_OVERLAP = 50  # 重叠50字

# 检索配置
TOP_K = 5  # 检索返回的文档数量

# LLM 参数
TEMPERATURE = 0.3  # 较低温度保证回答稳定

# ============ 增强功能配置 ============

# 检索失败判定阈值
SIMILARITY_THRESHOLD = 0.5          # 向量相似度失败阈值 (0-1)
RELEVANT_DOCS_THRESHOLD = 2         # 相关片段数量阈值
COVERAGE_THRESHOLD = 0.6            # 证据覆盖度阈值 (0-1)

# 回退与迭代上限
MAX_FALLBACK_ROUNDS = 3             # 失败回退最多轮数
MAX_ITERATION_ROUNDS = 5            # 迭代查询最多轮数
MAX_SUB_QUESTIONS = 5               # 子问题数量上限

# 重排与质量控制
MAX_RERANKED_DOCS = 8               # 重排后保留片段数量
MIN_DOC_LENGTH = 50                 # 片段最小长度(字符)

# 联网搜索配置
WEB_SEARCH_ENABLED = True           # 是否启用联网搜索
WEB_SEARCH_TRIGGER_THRESHOLD = 0.7  # 联网搜索触发阈值 (覆盖度低于此值则联网，越高越积极)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# 日志配置
LOGS_DIR = PROJECT_ROOT / "logs" / "query_traces"
