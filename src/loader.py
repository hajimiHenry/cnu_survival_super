"""
文档加载器 - 从 data 目录加载文本并解析元数据
"""
import re
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.documents import Document

from .config import (
    DATA_DIR, RULES_DIR, FLOWS_DIR, EXPERIENCE_DIR,
    TEMPLATES_DIR, DOCS_DIR, CATEGORY_MAP
)


def parse_entry(entry_text: str, file_path: str, default_category: str) -> Optional[Document]:
    """
    解析单条记录，返回 LangChain Document 对象

    每条记录格式:
    # 条目: xxx
    # 类别: xxx
    # 主题: xxx
    # 来源: xxx

    正文内容...
    """
    if not entry_text.strip():
        return None

    lines = entry_text.strip().split('\n')

    # 解析元数据
    metadata = {
        'title': '',
        'category': default_category,
        'topic': '',
        'source': '',
        'file_path': file_path
    }

    content_lines = []
    in_metadata = True

    for line in lines:
        line = line.strip()
        if not line:
            if in_metadata:
                continue
            content_lines.append('')
            continue

        # 检查是否是元数据行
        if line.startswith('# 条目:') or line.startswith('#条目:'):
            metadata['title'] = line.split(':', 1)[1].strip()
        elif line.startswith('# 类别:') or line.startswith('#类别:'):
            metadata['category'] = line.split(':', 1)[1].strip()
        elif line.startswith('# 主题:') or line.startswith('#主题:'):
            metadata['topic'] = line.split(':', 1)[1].strip()
        elif line.startswith('# 来源:') or line.startswith('#来源:'):
            metadata['source'] = line.split(':', 1)[1].strip()
        elif line.startswith('#'):
            # 其他以#开头的行可能是未识别的元数据，跳过
            continue
        else:
            in_metadata = False
            content_lines.append(line)

    # 组合正文
    content = '\n'.join(content_lines).strip()

    if not content:
        return None

    return Document(
        page_content=content,
        metadata=metadata
    )


def load_file(file_path: Path, default_category: str) -> List[Document]:
    """
    加载单个文本文件，解析其中的所有记录
    使用 --- 作为记录分隔符
    """
    documents = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"读取文件失败 {file_path}: {e}")
        return documents

    # 按 --- 分割记录
    entries = re.split(r'\n---+\n', content)

    for entry in entries:
        doc = parse_entry(entry, str(file_path), default_category)
        if doc:
            documents.append(doc)

    return documents


def load_directory(dir_path: Path, default_category: str) -> List[Document]:
    """
    加载目录下的所有 .txt 文件
    """
    documents = []

    if not dir_path.exists():
        print(f"目录不存在: {dir_path}")
        return documents

    for file_path in dir_path.glob('*.txt'):
        docs = load_file(file_path, default_category)
        documents.extend(docs)
        print(f"已加载 {file_path.name}: {len(docs)} 条记录")

    return documents


def load_all_documents() -> List[Document]:
    """
    加载所有数据目录中的文档
    返回包含所有 Document 的列表
    """
    all_documents = []

    # 定义目录和对应的默认类别
    directories = [
        (RULES_DIR, "制度"),
        (FLOWS_DIR, "流程"),
        (EXPERIENCE_DIR, "经验"),
        (TEMPLATES_DIR, "模板"),
        (DOCS_DIR, "培养方案")
    ]

    for dir_path, category in directories:
        print(f"\n正在加载 {category} 类文档...")
        docs = load_directory(dir_path, category)
        all_documents.extend(docs)
        print(f"{category} 类共加载 {len(docs)} 条记录")

    print(f"\n总计加载 {len(all_documents)} 条文档")
    return all_documents


def get_documents_by_category(documents: List[Document], category: str) -> List[Document]:
    """
    按类别筛选文档
    """
    return [doc for doc in documents if doc.metadata.get('category') == category]


if __name__ == "__main__":
    # 测试加载
    docs = load_all_documents()

    print("\n=== 文档示例 ===")
    if docs:
        doc = docs[0]
        print(f"标题: {doc.metadata.get('title')}")
        print(f"类别: {doc.metadata.get('category')}")
        print(f"主题: {doc.metadata.get('topic')}")
        print(f"来源: {doc.metadata.get('source')}")
        print(f"正文前100字: {doc.page_content[:100]}...")
