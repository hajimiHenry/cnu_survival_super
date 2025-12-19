"""
本地证据智能体 - 只访问本地向量库，输出要点与证据引用
"""
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

from ..config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME,
    TOP_K
)
from ..vector_store import (
    similarity_search_with_score,
    convert_distance_to_similarity
)
from ..reranker import Reranker
from ..retrieval_judge import RetrievalJudge
from ..query_transformer import QueryTransformer


@dataclass
class LocalEvidence:
    """本地证据条目"""
    content: str                # 证据内容
    source: str                 # 来源文件
    score: float                # 相关性分数
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalResearchResult:
    """本地研究结果"""
    question: str                       # 研究问题
    summary: str                        # 研究总结
    key_points: List[str]               # 关键要点
    evidence: List[LocalEvidence]       # 证据列表
    retrieval_success: bool             # 检索是否成功
    fallback_used: bool                 # 是否使用了回退策略
    duration_ms: float                  # 耗时


class LocalEvidenceAgent:
    """本地证据智能体"""

    def __init__(self, vector_store: FAISS):
        """
        初始化本地证据智能体

        Args:
            vector_store: FAISS向量存储实例
        """
        self.vector_store = vector_store
        self._llm = None
        self._reranker = Reranker()
        self._judge = RetrievalJudge()
        self._transformer = QueryTransformer()

    def _get_llm(self) -> ChatOpenAI:
        """获取LLM实例"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=MODEL_NAME,
                temperature=0.3,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_BASE_URL
            )
        return self._llm

    def research(self, question: str, k: int = TOP_K) -> LocalResearchResult:
        """
        执行本地研究

        Args:
            question: 研究问题
            k: 检索数量

        Returns:
            LocalResearchResult
        """
        start_time = time.time()

        # 第一轮检索
        docs_with_scores = similarity_search_with_score(
            self.vector_store, question, k=k
        )

        # 不使用回退检索
        fallback_used = False

        # 重排
        if docs_with_scores:
            # 转换距离为相似度
            docs_with_sim = [
                (doc, convert_distance_to_similarity(score))
                for doc, score in docs_with_scores
            ]
            rerank_result = self._reranker.rerank(question, docs_with_sim)
            final_docs = [
                (rd.document, rd.final_score)
                for rd in rerank_result.ranked_docs
            ]
        else:
            final_docs = []

        # 转换为证据格式
        evidence = []
        for doc, score in final_docs:
            evidence.append(LocalEvidence(
                content=doc.page_content,
                source=doc.metadata.get('source', doc.metadata.get('title', 'unknown')),
                score=score,
                metadata=doc.metadata
            ))

        # 生成总结和要点
        if evidence:
            summary, key_points = self._generate_summary(question, evidence)
            retrieval_success = True
        else:
            summary = "未能在本地知识库中找到相关信息。"
            key_points = []
            retrieval_success = False

        duration = (time.time() - start_time) * 1000

        return LocalResearchResult(
            question=question,
            summary=summary,
            key_points=key_points,
            evidence=evidence,
            retrieval_success=retrieval_success,
            fallback_used=fallback_used,
            duration_ms=duration
        )

    async def research_async(self, question: str, k: int = TOP_K) -> LocalResearchResult:
        """
        异步执行本地研究

        Args:
            question: 研究问题
            k: 检索数量

        Returns:
            LocalResearchResult
        """
        # 在线程池中执行同步方法
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.research, question, k)

    def _merge_results(
        self,
        results1: List[Tuple[Document, float]],
        results2: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """合并两组检索结果，去重"""
        seen = set()
        merged = []

        for doc, score in results1 + results2:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                merged.append((doc, score))

        # 按分数排序（距离越小越好）
        merged.sort(key=lambda x: x[1])
        return merged

    def _generate_summary(
        self,
        question: str,
        evidence: List[LocalEvidence]
    ) -> Tuple[str, List[str]]:
        """
        生成研究总结和要点

        Args:
            question: 问题
            evidence: 证据列表

        Returns:
            (总结, 要点列表)
        """
        evidence_text = "\n\n".join([
            f"【证据{i+1}】来源：{ev.source}\n{ev.content}"
            for i, ev in enumerate(evidence[:5])
        ])

        prompt = f"""基于以下本地知识库证据，回答用户问题并提取关键要点。

用户问题：{question}

本地知识库证据：
{evidence_text}

请按以下格式输出：
【总结】
（用2-3句话总结回答，必须基于证据）

【要点】
1. （第一个要点）
2. （第二个要点）
3. （第三个要点，如果有）

只输出总结和要点，不要其他内容："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            content = response.content.strip()

            # 解析输出
            summary = ""
            key_points = []

            lines = content.split('\n')
            in_summary = False
            in_points = False

            for line in lines:
                line = line.strip()
                if '【总结】' in line or '总结】' in line:
                    in_summary = True
                    in_points = False
                    continue
                elif '【要点】' in line or '要点】' in line:
                    in_summary = False
                    in_points = True
                    continue

                if in_summary and line:
                    summary += line + " "
                elif in_points and line:
                    # 去除序号
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '1、', '2、', '3、', '4、', '5、', '-', '•']:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                            break
                    if line:
                        key_points.append(line)

            return summary.strip(), key_points

        except Exception:
            # 简单后备方案
            return "基于本地知识库的证据进行了检索。", []
