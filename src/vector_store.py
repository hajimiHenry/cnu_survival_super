"""
向量存储 - 文本切分、向量化、构建和加载 FAISS 向量库
使用本地 HuggingFace 嵌入模型，无需调用 API
"""
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .config import CHUNK_SIZE, CHUNK_OVERLAP, INDEX_DIR, LOCAL_EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    获取本地嵌入模型实例
    使用 HuggingFace 的多语言模型，支持中文
    """
    print(f"正在加载本地嵌入模型: {LOCAL_EMBEDDING_MODEL}")
    print("（首次运行会自动下载模型，请耐心等待...）")

    return HuggingFaceEmbeddings(
        model_name=LOCAL_EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def split_documents(documents: List[Document]) -> List[Document]:
    """
    对文档进行切分
    切分后的每个小段继承原来的 metadata 并增加 chunk_index 字段
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )

    split_docs = []

    for doc in documents:
        # 切分单个文档
        chunks = text_splitter.split_text(doc.page_content)

        for idx, chunk in enumerate(chunks):
            # 创建新的 Document，继承原始 metadata
            new_metadata = doc.metadata.copy()
            new_metadata['chunk_index'] = idx
            new_metadata['total_chunks'] = len(chunks)

            split_docs.append(Document(
                page_content=chunk,
                metadata=new_metadata
            ))

    print(f"文档切分完成: {len(documents)} 个文档 -> {len(split_docs)} 个片段")
    return split_docs


def build_vector_store(documents: List[Document]) -> FAISS:
    """
    构建 FAISS 向量索引

    Args:
        documents: 已切分的文档列表

    Returns:
        FAISS 向量存储实例
    """
    print("正在初始化嵌入模型...")
    embeddings = get_embeddings()

    print(f"正在构建向量索引 (共 {len(documents)} 个片段)...")
    vector_store = FAISS.from_documents(documents, embeddings)

    print("向量索引构建完成")
    return vector_store


def save_vector_store(vector_store: FAISS, path: Optional[Path] = None) -> None:
    """
    保存向量索引到本地

    Args:
        vector_store: FAISS 向量存储实例
        path: 保存路径，默认为 INDEX_DIR
    """
    if path is None:
        path = INDEX_DIR

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(path))
    print(f"向量索引已保存到: {path}")


def load_vector_store(path: Optional[Path] = None) -> FAISS:
    """
    从本地加载向量索引

    Args:
        path: 加载路径，默认为 INDEX_DIR

    Returns:
        FAISS 向量存储实例
    """
    if path is None:
        path = INDEX_DIR

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"向量索引目录不存在: {path}")

    print(f"正在加载向量索引: {path}")
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("向量索引加载完成")
    return vector_store


def similarity_search(
    vector_store: FAISS,
    query: str,
    k: int = 5,
    category_filter: Optional[str] = None
) -> List[Document]:
    """
    相似度检索

    Args:
        vector_store: FAISS 向量存储实例
        query: 查询文本
        k: 返回结果数量
        category_filter: 可选的类别过滤

    Returns:
        检索到的文档列表
    """
    # 如果需要过滤，先检索更多结果再过滤
    search_k = k * 3 if category_filter else k

    results = vector_store.similarity_search(query, k=search_k)

    # 按类别过滤
    if category_filter:
        results = [
            doc for doc in results
            if doc.metadata.get('category') == category_filter
        ][:k]

    return results


if __name__ == "__main__":
    # 测试向量存储功能
    from .loader import load_all_documents

    # 加载文档
    docs = load_all_documents()

    if docs:
        # 切分文档
        split_docs = split_documents(docs)

        # 构建向量库
        vs = build_vector_store(split_docs)

        # 保存向量库
        save_vector_store(vs)

        # 测试检索
        print("\n=== 测试检索 ===")
        results = similarity_search(vs, "重修怎么办理", k=3)
        for i, doc in enumerate(results):
            print(f"\n--- 结果 {i+1} ---")
            print(f"标题: {doc.metadata.get('title')}")
            print(f"类别: {doc.metadata.get('category')}")
            print(f"内容: {doc.page_content[:100]}...")
