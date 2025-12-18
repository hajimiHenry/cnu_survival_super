"""
向量存储 - 文本切分、向量化、构建和加载 FAISS 向量库
支持向量检索、BM25关键词检索、混合检索
"""
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import pickle
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import jieba

from .config import CHUNK_SIZE, CHUNK_OVERLAP, INDEX_DIR, LOCAL_EMBEDDING_MODEL


# BM25索引缓存
_bm25_index: Optional['BM25Index'] = None


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


def similarity_search_with_score(
    vector_store: FAISS,
    query: str,
    k: int = 5,
    category_filter: Optional[str] = None
) -> List[Tuple[Document, float]]:
    """
    带相似度分数的检索

    Args:
        vector_store: FAISS 向量存储实例
        query: 查询文本
        k: 返回结果数量
        category_filter: 可选的类别过滤

    Returns:
        (文档, 相似度分数) 元组列表，分数越小越相似
    """
    search_k = k * 3 if category_filter else k

    # FAISS返回的是L2距离，越小越相似
    results = vector_store.similarity_search_with_score(query, k=search_k)

    # 按类别过滤
    if category_filter:
        results = [
            (doc, score) for doc, score in results
            if doc.metadata.get('category') == category_filter
        ][:k]

    return results


def convert_distance_to_similarity(distance: float) -> float:
    """
    将FAISS的L2距离转换为相似度分数 (0-1)
    使用公式: similarity = 1 / (1 + distance)
    """
    return 1.0 / (1.0 + distance)


class BM25Index:
    """BM25关键词检索索引"""

    def __init__(self, documents: List[Document]):
        """
        构建BM25索引

        Args:
            documents: 文档列表
        """
        self.documents = documents
        # 中文分词
        self.tokenized_corpus = [
            list(jieba.cut(doc.page_content)) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        BM25检索

        Args:
            query: 查询文本
            k: 返回数量

        Returns:
            (文档, BM25分数) 元组列表
        """
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        # 获取Top-K索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有分数的结果
                results.append((self.documents[idx], scores[idx]))

        return results

    def save(self, path: Path):
        """保存BM25索引"""
        with open(path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'tokenized_corpus': self.tokenized_corpus
            }, f)

    @classmethod
    def load(cls, path: Path) -> 'BM25Index':
        """加载BM25索引"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        index = cls.__new__(cls)
        index.documents = data['documents']
        index.tokenized_corpus = data['tokenized_corpus']
        index.bm25 = BM25Okapi(index.tokenized_corpus)
        return index


def build_bm25_index(documents: List[Document]) -> BM25Index:
    """
    构建BM25索引

    Args:
        documents: 文档列表

    Returns:
        BM25Index实例
    """
    print(f"正在构建BM25索引 (共 {len(documents)} 个片段)...")
    index = BM25Index(documents)
    print("BM25索引构建完成")
    return index


def save_bm25_index(index: BM25Index, path: Optional[Path] = None):
    """保存BM25索引"""
    if path is None:
        path = INDEX_DIR / "bm25_index.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    index.save(path)
    print(f"BM25索引已保存到: {path}")


def load_bm25_index(path: Optional[Path] = None) -> BM25Index:
    """加载BM25索引"""
    global _bm25_index
    if _bm25_index is not None:
        return _bm25_index

    if path is None:
        path = INDEX_DIR / "bm25_index.pkl"

    if not path.exists():
        raise FileNotFoundError(f"BM25索引文件不存在: {path}")

    print(f"正在加载BM25索引: {path}")
    _bm25_index = BM25Index.load(path)
    print("BM25索引加载完成")
    return _bm25_index


def bm25_search(
    query: str,
    k: int = 5,
    bm25_index: Optional[BM25Index] = None
) -> List[Tuple[Document, float]]:
    """
    BM25关键词检索

    Args:
        query: 查询文本
        k: 返回数量
        bm25_index: BM25索引实例，如果为None则尝试加载

    Returns:
        (文档, BM25分数) 元组列表
    """
    if bm25_index is None:
        bm25_index = load_bm25_index()

    return bm25_index.search(query, k)


def hybrid_search(
    vector_store: FAISS,
    query: str,
    k: int = 5,
    vector_weight: float = 0.5,
    bm25_index: Optional[BM25Index] = None
) -> List[Tuple[Document, float]]:
    """
    混合检索：结合向量检索和BM25检索

    Args:
        vector_store: FAISS向量存储
        query: 查询文本
        k: 返回数量
        vector_weight: 向量检索权重 (0-1)，BM25权重为 1-vector_weight
        bm25_index: BM25索引实例

    Returns:
        (文档, 混合分数) 元组列表，按分数降序排列
    """
    # 获取更多候选以便合并
    candidate_k = k * 3

    # 向量检索
    vector_results = similarity_search_with_score(vector_store, query, k=candidate_k)

    # BM25检索
    if bm25_index is None:
        try:
            bm25_index = load_bm25_index()
        except FileNotFoundError:
            # 如果没有BM25索引，只返回向量检索结果
            return [(doc, convert_distance_to_similarity(score))
                    for doc, score in vector_results[:k]]

    bm25_results = bm25_index.search(query, k=candidate_k)

    # 合并结果
    # 使用文档内容作为key进行去重
    doc_scores: Dict[str, Tuple[Document, float, float]] = {}

    # 处理向量检索结果
    for doc, distance in vector_results:
        key = doc.page_content[:200]  # 用前200字符作为key
        vector_sim = convert_distance_to_similarity(distance)
        if key not in doc_scores:
            doc_scores[key] = (doc, vector_sim, 0.0)
        else:
            existing = doc_scores[key]
            doc_scores[key] = (existing[0], max(existing[1], vector_sim), existing[2])

    # 处理BM25结果，归一化分数
    if bm25_results:
        max_bm25 = max(score for _, score in bm25_results)
        for doc, bm25_score in bm25_results:
            key = doc.page_content[:200]
            normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0
            if key not in doc_scores:
                doc_scores[key] = (doc, 0.0, normalized_bm25)
            else:
                existing = doc_scores[key]
                doc_scores[key] = (existing[0], existing[1], max(existing[2], normalized_bm25))

    # 计算混合分数
    bm25_weight = 1 - vector_weight
    results = []
    for doc, vector_sim, bm25_sim in doc_scores.values():
        hybrid_score = vector_weight * vector_sim + bm25_weight * bm25_sim
        results.append((doc, hybrid_score))

    # 按混合分数降序排序
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:k]


def get_retrieval_stats(
    docs_with_scores: List[Tuple[Document, float]],
    score_is_similarity: bool = True
) -> Dict[str, Any]:
    """
    获取检索统计信息

    Args:
        docs_with_scores: (文档, 分数)列表
        score_is_similarity: 分数是否为相似度(True)还是距离(False)

    Returns:
        统计信息字典
    """
    if not docs_with_scores:
        return {
            'count': 0,
            'top1_score': 0.0,
            'avg_score': 0.0,
            'min_score': 0.0,
            'max_score': 0.0
        }

    scores = [score for _, score in docs_with_scores]
    if not score_is_similarity:
        # 将距离转换为相似度
        scores = [convert_distance_to_similarity(s) for s in scores]

    return {
        'count': len(scores),
        'top1_score': scores[0] if scores else 0.0,
        'avg_score': sum(scores) / len(scores),
        'min_score': min(scores),
        'max_score': max(scores)
    }


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
