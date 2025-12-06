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
