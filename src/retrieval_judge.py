"""
检索失败判定器 - 判断检索结果是否足以支撑回答
"""
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .config import (
    SIMILARITY_THRESHOLD,
    RELEVANT_DOCS_THRESHOLD,
    COVERAGE_THRESHOLD,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME,
    TEMPERATURE
)
from .vector_store import convert_distance_to_similarity


@dataclass
class JudgeResult:
    """判定结果"""
    is_failed: bool                     # 是否判定为失败
    similarity_score: float             # Top1相似度分数
    similarity_failed: bool             # 相似度信号是否失败
    relevant_count: int                 # 相关片段数量
    relevant_failed: bool               # 相关数量信号是否失败
    coverage_score: float               # 覆盖度分数
    coverage_failed: bool               # 覆盖度信号是否失败
    failed_signals: int                 # 失败信号数量
    reason: str                         # 失败原因说明


class RetrievalJudge:
    """检索失败判定器"""

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        relevant_threshold: int = RELEVANT_DOCS_THRESHOLD,
        coverage_threshold: float = COVERAGE_THRESHOLD
    ):
        """
        初始化判定器

        Args:
            similarity_threshold: 相似度阈值
            relevant_threshold: 相关片段数量阈值
            coverage_threshold: 覆盖度阈值
        """
        self.similarity_threshold = similarity_threshold
        self.relevant_threshold = relevant_threshold
        self.coverage_threshold = coverage_threshold
        self._llm = None

    def _get_llm(self) -> ChatOpenAI:
        """获取LLM实例（延迟加载）"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=MODEL_NAME,
                temperature=0.1,  # 低温度保证判断稳定
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_BASE_URL
            )
        return self._llm

    def judge(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
        score_is_distance: bool = True
    ) -> JudgeResult:
        """
        判断检索结果是否失败

        满足任意2个失败信号即判定为失败：
        1. Top1相似度 < 阈值
        2. 相关片段数量 < 阈值
        3. 证据覆盖度 < 阈值

        Args:
            query: 用户查询
            docs_with_scores: (文档, 分数)列表
            score_is_distance: 分数是否为距离(True)还是相似度(False)

        Returns:
            JudgeResult 判定结果
        """
        # 信号1: 相似度判定
        if docs_with_scores:
            top1_score = docs_with_scores[0][1]
            if score_is_distance:
                similarity_score = convert_distance_to_similarity(top1_score)
            else:
                similarity_score = top1_score
        else:
            similarity_score = 0.0

        similarity_failed = similarity_score < self.similarity_threshold

        # 信号2: 相关片段数量判定
        relevant_count = self._count_relevant_docs(query, docs_with_scores, score_is_distance)
        relevant_failed = relevant_count < self.relevant_threshold

        # 信号3: 覆盖度判定（使用LLM）
        coverage_score = self._judge_coverage(query, docs_with_scores)
        coverage_failed = coverage_score < self.coverage_threshold

        # 统计失败信号数量
        failed_signals = sum([similarity_failed, relevant_failed, coverage_failed])

        # 满足任意2个失败信号即判定为失败
        is_failed = failed_signals >= 2

        # 生成失败原因说明
        reason = self._generate_reason(
            is_failed, similarity_failed, relevant_failed, coverage_failed,
            similarity_score, relevant_count, coverage_score
        )

        return JudgeResult(
            is_failed=is_failed,
            similarity_score=similarity_score,
            similarity_failed=similarity_failed,
            relevant_count=relevant_count,
            relevant_failed=relevant_failed,
            coverage_score=coverage_score,
            coverage_failed=coverage_failed,
            failed_signals=failed_signals,
            reason=reason
        )

    def _count_relevant_docs(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]],
        score_is_distance: bool
    ) -> int:
        """
        统计相关文档数量

        使用相似度阈值的80%作为相关性判定标准
        """
        if not docs_with_scores:
            return 0

        # 相关性阈值设为相似度阈值的80%
        relevance_threshold = self.similarity_threshold * 0.8

        count = 0
        for doc, score in docs_with_scores:
            if score_is_distance:
                similarity = convert_distance_to_similarity(score)
            else:
                similarity = score

            if similarity >= relevance_threshold:
                count += 1

        return count

    def _judge_coverage(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float]]
    ) -> float:
        """
        使用LLM判断证据覆盖度

        Returns:
            覆盖度分数 (0-1)
        """
        if not docs_with_scores:
            return 0.0

        # 准备证据文本
        evidence_text = "\n\n".join([
            f"证据{i+1}: {doc.page_content[:300]}..."
            for i, (doc, _) in enumerate(docs_with_scores[:5])
        ])

        prompt = f"""请判断以下证据是否足以回答用户的问题。

用户问题：{query}

可用证据：
{evidence_text}

请评估这些证据对回答问题的覆盖程度，只输出一个0到1之间的数字：
- 0.0-0.3: 证据完全不相关或严重不足
- 0.3-0.5: 证据部分相关但缺少关键信息
- 0.5-0.7: 证据基本相关但不够完整
- 0.7-0.9: 证据较为充分
- 0.9-1.0: 证据非常充分，足以完整回答问题

只输出数字，不要输出其他内容："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            score_text = response.content.strip()
            # 提取数字
            score = float(score_text.split()[0])
            return max(0.0, min(1.0, score))
        except Exception as e:
            # 如果LLM调用失败，基于相似度分数估算
            if docs_with_scores:
                avg_score = sum(convert_distance_to_similarity(s) for _, s in docs_with_scores) / len(docs_with_scores)
                return avg_score
            return 0.5

    def _generate_reason(
        self,
        is_failed: bool,
        similarity_failed: bool,
        relevant_failed: bool,
        coverage_failed: bool,
        similarity_score: float,
        relevant_count: int,
        coverage_score: float
    ) -> str:
        """生成失败原因说明"""
        if not is_failed:
            return "检索结果充分"

        reasons = []
        if similarity_failed:
            reasons.append(f"最相关文档的相似度较低({similarity_score:.2f}<{self.similarity_threshold})")
        if relevant_failed:
            reasons.append(f"相关片段数量不足({relevant_count}<{self.relevant_threshold})")
        if coverage_failed:
            reasons.append(f"证据覆盖度不足({coverage_score:.2f}<{self.coverage_threshold})")

        return "；".join(reasons)

    def quick_judge(
        self,
        docs_with_scores: List[Tuple[Document, float]],
        score_is_distance: bool = True
    ) -> bool:
        """
        快速判定（不使用LLM，只检查相似度和数量）

        用于第一轮快速筛选，避免不必要的LLM调用

        Returns:
            True表示可能失败，需要进一步判定
        """
        if not docs_with_scores:
            return True

        # 检查相似度
        top1_score = docs_with_scores[0][1]
        if score_is_distance:
            similarity = convert_distance_to_similarity(top1_score)
        else:
            similarity = top1_score

        if similarity < self.similarity_threshold:
            return True

        # 检查数量
        relevance_threshold = self.similarity_threshold * 0.8
        count = 0
        for _, score in docs_with_scores:
            if score_is_distance:
                sim = convert_distance_to_similarity(score)
            else:
                sim = score
            if sim >= relevance_threshold:
                count += 1

        if count < self.relevant_threshold:
            return True

        return False


def judge_retrieval(
    query: str,
    docs_with_scores: List[Tuple[Document, float]],
    score_is_distance: bool = True
) -> JudgeResult:
    """
    便捷函数：判断检索结果是否失败

    Args:
        query: 用户查询
        docs_with_scores: (文档, 分数)列表
        score_is_distance: 分数是否为距离

    Returns:
        JudgeResult
    """
    judge = RetrievalJudge()
    return judge.judge(query, docs_with_scores, score_is_distance)
