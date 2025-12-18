"""
证据聚合器 - 多轮证据的去重、合并和收敛判断
"""
import time
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME,
    COVERAGE_THRESHOLD,
    MAX_RERANKED_DOCS
)


@dataclass
class Evidence:
    """证据条目"""
    content: str                        # 证据内容
    source: str                         # 来源
    source_type: str                    # 来源类型 (local/web)
    score: float                        # 相关性分数
    round_num: int                      # 来自第几轮
    sub_question: str                   # 对应的子问题
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """内容哈希，用于去重"""
        return self.content[:200]


@dataclass
class AggregationResult:
    """聚合结果"""
    total_rounds: int                   # 总轮数
    total_evidence_before: int          # 去重前证据数
    total_evidence_after: int           # 去重后证据数
    merged_count: int                   # 合并数量
    final_evidence: List[Evidence]      # 最终证据列表
    converged: bool                     # 是否收敛
    convergence_score: float            # 收敛分数


class EvidenceAggregator:
    """证据聚合器"""

    def __init__(
        self,
        coverage_threshold: float = COVERAGE_THRESHOLD,
        max_evidence: int = MAX_RERANKED_DOCS
    ):
        """
        初始化聚合器

        Args:
            coverage_threshold: 覆盖度阈值，达到此值判定为收敛
            max_evidence: 最大保留证据数
        """
        self.coverage_threshold = coverage_threshold
        self.max_evidence = max_evidence
        self._llm = None

        # 累计的证据
        self._evidence_pool: List[Evidence] = []
        self._seen_hashes: Set[str] = set()
        self._round_count = 0

    def _get_llm(self) -> ChatOpenAI:
        """获取LLM实例"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=MODEL_NAME,
                temperature=0.1,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_BASE_URL
            )
        return self._llm

    def add_round(
        self,
        docs_with_scores: List[Tuple[Document, float]],
        sub_question: str,
        source_type: str = "local"
    ) -> int:
        """
        添加一轮检索结果

        Args:
            docs_with_scores: (文档, 分数)列表
            sub_question: 对应的子问题
            source_type: 来源类型

        Returns:
            本轮新增的证据数量
        """
        self._round_count += 1
        new_count = 0

        for doc, score in docs_with_scores:
            evidence = Evidence(
                content=doc.page_content,
                source=doc.metadata.get('source', doc.metadata.get('title', 'unknown')),
                source_type=source_type,
                score=score,
                round_num=self._round_count,
                sub_question=sub_question,
                metadata=doc.metadata
            )

            # 去重
            if evidence.content_hash not in self._seen_hashes:
                self._evidence_pool.append(evidence)
                self._seen_hashes.add(evidence.content_hash)
                new_count += 1

        return new_count

    def add_web_evidence(
        self,
        content: str,
        source: str,
        score: float,
        sub_question: str = ""
    ) -> bool:
        """
        添加网络证据

        Args:
            content: 证据内容
            source: URL来源
            score: 相关性分数
            sub_question: 对应的子问题

        Returns:
            是否添加成功（未重复）
        """
        evidence = Evidence(
            content=content,
            source=source,
            source_type="web",
            score=score,
            round_num=self._round_count,
            sub_question=sub_question
        )

        if evidence.content_hash not in self._seen_hashes:
            self._evidence_pool.append(evidence)
            self._seen_hashes.add(evidence.content_hash)
            return True
        return False

    def deduplicate(self) -> int:
        """
        深度去重：移除高度相似的证据

        Returns:
            移除的数量
        """
        if len(self._evidence_pool) <= 1:
            return 0

        # 使用简单的相似度比较
        to_remove = set()
        for i, ev1 in enumerate(self._evidence_pool):
            if i in to_remove:
                continue
            for j, ev2 in enumerate(self._evidence_pool[i+1:], i+1):
                if j in to_remove:
                    continue
                # 计算简单的字符重叠率
                overlap = self._calculate_overlap(ev1.content, ev2.content)
                if overlap > 0.7:  # 70%以上重叠视为重复
                    # 保留分数高的
                    if ev1.score >= ev2.score:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break

        # 移除重复项
        self._evidence_pool = [
            ev for i, ev in enumerate(self._evidence_pool)
            if i not in to_remove
        ]

        return len(to_remove)

    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """计算两段文本的字符重叠率"""
        if not text1 or not text2:
            return 0.0

        # 使用字符集合
        set1 = set(text1)
        set2 = set(text2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def merge_similar(self) -> int:
        """
        合并相近的证据（目前仅做去重，不实际合并内容）

        Returns:
            合并数量
        """
        return self.deduplicate()

    def check_convergence(self, question: str) -> Tuple[bool, float]:
        """
        判断证据是否收敛（足以回答问题）

        Args:
            question: 原始问题

        Returns:
            (是否收敛, 覆盖度分数)
        """
        if not self._evidence_pool:
            return False, 0.0

        # 准备证据摘要
        evidence_text = "\n\n".join([
            f"证据{i+1}（来源：{ev.source}）：{ev.content[:200]}..."
            for i, ev in enumerate(self._evidence_pool[:10])
        ])

        prompt = f"""请评估当前收集的证据是否足以回答用户的问题。

用户问题：{question}

已收集的证据：
{evidence_text}

请评估证据的覆盖程度，只输出一个0到1之间的数字：
- 0.0-0.3: 证据严重不足，关键信息缺失
- 0.3-0.5: 证据部分有用但不完整
- 0.5-0.7: 证据基本足够，可以给出部分答案
- 0.7-0.9: 证据较为充分，可以给出较完整答案
- 0.9-1.0: 证据非常充分，可以完整回答问题

只输出数字："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            score = float(response.content.strip().split()[0])
            score = max(0.0, min(1.0, score))
            converged = score >= self.coverage_threshold
            return converged, score
        except Exception:
            # 基于证据数量估算
            count = len(self._evidence_pool)
            estimated_score = min(1.0, count * 0.15)
            return estimated_score >= self.coverage_threshold, estimated_score

    def get_final_evidence(self) -> List[Evidence]:
        """
        获取最终证据列表

        Returns:
            按分数排序后的证据列表
        """
        # 按分数降序排序
        sorted_evidence = sorted(
            self._evidence_pool,
            key=lambda x: x.score,
            reverse=True
        )

        return sorted_evidence[:self.max_evidence]

    def get_local_evidence(self) -> List[Evidence]:
        """获取本地证据"""
        return [ev for ev in self.get_final_evidence() if ev.source_type == "local"]

    def get_web_evidence(self) -> List[Evidence]:
        """获取网络证据"""
        return [ev for ev in self.get_final_evidence() if ev.source_type == "web"]

    def get_aggregation_result(self, question: str) -> AggregationResult:
        """
        获取完整的聚合结果

        Args:
            question: 原始问题

        Returns:
            AggregationResult
        """
        total_before = len(self._evidence_pool) + len(self._seen_hashes) - len(self._evidence_pool)
        merged_count = self.merge_similar()
        converged, score = self.check_convergence(question)
        final = self.get_final_evidence()

        return AggregationResult(
            total_rounds=self._round_count,
            total_evidence_before=total_before,
            total_evidence_after=len(self._evidence_pool),
            merged_count=merged_count,
            final_evidence=final,
            converged=converged,
            convergence_score=score
        )

    def reset(self):
        """重置聚合器状态"""
        self._evidence_pool = []
        self._seen_hashes = set()
        self._round_count = 0

    @property
    def evidence_count(self) -> int:
        """当前证据数量"""
        return len(self._evidence_pool)

    @property
    def round_count(self) -> int:
        """当前轮数"""
        return self._round_count


def create_aggregator() -> EvidenceAggregator:
    """创建证据聚合器的便捷函数"""
    return EvidenceAggregator()
