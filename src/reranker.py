"""
重排器 - 使用LLM对检索结果进行重排和质量控制
"""
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME,
    MAX_RERANKED_DOCS,
    MIN_DOC_LENGTH
)


@dataclass
class RankedDocument:
    """重排后的文档"""
    document: Document          # 原始文档
    original_score: float       # 原始检索分数
    rerank_score: float         # 重排分数 (0-10)
    final_score: float          # 最终分数
    is_filtered: bool           # 是否被过滤
    filter_reason: str          # 过滤原因


@dataclass
class RerankResult:
    """重排结果"""
    query: str                          # 查询
    before_ranking: List[Dict]          # 重排前的文档摘要
    after_ranking: List[Dict]           # 重排后的文档摘要
    ranked_docs: List[RankedDocument]   # 重排后的文档详情
    filtered_count: int                 # 被过滤的数量
    duration_ms: float                  # 耗时


class Reranker:
    """重排器 - 使用LLM打分"""

    def __init__(
        self,
        max_docs: int = MAX_RERANKED_DOCS,
        min_doc_length: int = MIN_DOC_LENGTH
    ):
        """
        初始化重排器

        Args:
            max_docs: 最多保留的文档数量
            min_doc_length: 片段最小长度
        """
        self.max_docs = max_docs
        self.min_doc_length = min_doc_length
        self._llm = None

    def _get_llm(self) -> ChatOpenAI:
        """获取LLM实例"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=MODEL_NAME,
                temperature=0.1,  # 低温度保证评分稳定
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_BASE_URL
            )
        return self._llm

    def rerank(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
        score_weight: float = 0.3
    ) -> RerankResult:
        """
        对文档进行重排

        Args:
            query: 查询文本
            docs_with_scores: (文档, 原始分数)列表
            score_weight: 原始分数权重 (0-1)，LLM分数权重为 1-score_weight

        Returns:
            RerankResult
        """
        start_time = time.time()

        # 记录重排前状态
        before_ranking = [
            {
                'content_preview': doc.page_content[:100],
                'source': doc.metadata.get('source', doc.metadata.get('title', 'unknown')),
                'score': score
            }
            for doc, score in docs_with_scores
        ]

        # 过滤低质量文档
        filtered_docs = self._filter_low_quality(docs_with_scores)

        # 如果没有文档，直接返回
        if not filtered_docs:
            duration = (time.time() - start_time) * 1000
            return RerankResult(
                query=query,
                before_ranking=before_ranking,
                after_ranking=[],
                ranked_docs=[],
                filtered_count=len(docs_with_scores),
                duration_ms=duration
            )

        # 批量获取LLM评分
        llm_scores = self._batch_score(query, [doc for doc, _ in filtered_docs])

        # 计算最终分数
        ranked_docs = []
        for i, (doc, original_score) in enumerate(filtered_docs):
            llm_score = llm_scores[i] if i < len(llm_scores) else 5.0

            # 归一化原始分数到0-10
            normalized_original = original_score * 10 if original_score <= 1 else min(original_score, 10)

            # 计算加权最终分数
            final_score = (1 - score_weight) * llm_score + score_weight * normalized_original

            ranked_docs.append(RankedDocument(
                document=doc,
                original_score=original_score,
                rerank_score=llm_score,
                final_score=final_score,
                is_filtered=False,
                filter_reason=""
            ))

        # 按最终分数降序排序
        ranked_docs.sort(key=lambda x: x.final_score, reverse=True)

        # 限制数量
        ranked_docs = ranked_docs[:self.max_docs]

        # 记录重排后状态
        after_ranking = [
            {
                'content_preview': rd.document.page_content[:100],
                'source': rd.document.metadata.get('source', rd.document.metadata.get('title', 'unknown')),
                'original_score': rd.original_score,
                'llm_score': rd.rerank_score,
                'final_score': rd.final_score
            }
            for rd in ranked_docs
        ]

        duration = (time.time() - start_time) * 1000

        return RerankResult(
            query=query,
            before_ranking=before_ranking,
            after_ranking=after_ranking,
            ranked_docs=ranked_docs,
            filtered_count=len(docs_with_scores) - len(filtered_docs),
            duration_ms=duration
        )

    def _filter_low_quality(
        self,
        docs_with_scores: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        过滤低质量文档

        过滤条件：
        1. 文档内容过短
        2. 文档内容无实质信息
        """
        filtered = []
        for doc, score in docs_with_scores:
            content = doc.page_content.strip()

            # 检查长度
            if len(content) < self.min_doc_length:
                continue

            # 检查是否有实质内容（简单启发式）
            if self._is_meaningful(content):
                filtered.append((doc, score))

        return filtered

    def _is_meaningful(self, content: str) -> bool:
        """
        判断内容是否有实质信息

        简单启发式规则：
        - 内容不能全是标点符号
        - 内容中汉字比例应该合理
        """
        if not content:
            return False

        # 统计汉字数量
        chinese_count = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')

        # 汉字比例应该大于30%
        if chinese_count / len(content) < 0.3:
            return False

        return True

    def _batch_score(self, query: str, docs: List[Document]) -> List[float]:
        """
        批量为文档打分

        Args:
            query: 查询文本
            docs: 文档列表

        Returns:
            分数列表
        """
        if not docs:
            return []

        # 构建批量评分的prompt
        doc_texts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', doc.metadata.get('title', f'文档{i+1}'))
            content_preview = doc.page_content[:300]
            doc_texts.append(f"[文档{i+1}] 来源: {source}\n{content_preview}...")

        docs_formatted = "\n\n".join(doc_texts)

        prompt = f"""请评估以下每个文档与用户问题的相关程度。

用户问题：{query}

待评估文档：
{docs_formatted}

评分标准（0-10分）：
- 0-2分：完全不相关
- 3-4分：略有关联但不直接回答问题
- 5-6分：部分相关，包含一些有用信息
- 7-8分：较为相关，能够部分回答问题
- 9-10分：高度相关，能够直接回答问题

请为每个文档打分，格式为"文档1: X分"，每行一个，只输出分数，不要其他解释："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)

            # 解析分数
            scores = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                try:
                    # 尝试提取数字
                    parts = line.replace('分', '').replace('：', ':').split(':')
                    if len(parts) >= 2:
                        score = float(parts[-1].strip())
                        scores.append(max(0, min(10, score)))
                    else:
                        # 尝试直接提取数字
                        import re
                        nums = re.findall(r'[\d.]+', line)
                        if nums:
                            score = float(nums[-1])
                            scores.append(max(0, min(10, score)))
                except ValueError:
                    continue

            # 如果解析的分数不够，用默认分数填充
            while len(scores) < len(docs):
                scores.append(5.0)

            return scores[:len(docs)]

        except Exception as e:
            # 出错时返回默认分数
            return [5.0] * len(docs)

    def filter_by_score(
        self,
        ranked_docs: List[RankedDocument],
        min_score: float = 3.0
    ) -> List[RankedDocument]:
        """
        根据分数过滤文档

        Args:
            ranked_docs: 重排后的文档
            min_score: 最低分数阈值

        Returns:
            过滤后的文档列表
        """
        return [rd for rd in ranked_docs if rd.final_score >= min_score]


def rerank_documents(
    query: str,
    docs_with_scores: List[Tuple[Document, float]]
) -> List[Tuple[Document, float]]:
    """
    重排文档的便捷函数

    Returns:
        (文档, 最终分数)列表
    """
    reranker = Reranker()
    result = reranker.rerank(query, docs_with_scores)
    return [(rd.document, rd.final_score) for rd in result.ranked_docs]


def get_rerank_result(
    query: str,
    docs_with_scores: List[Tuple[Document, float]]
) -> RerankResult:
    """
    获取完整重排结果的便捷函数
    """
    reranker = Reranker()
    return reranker.rerank(query, docs_with_scores)
