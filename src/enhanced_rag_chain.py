"""
增强版RAG问答链 - 整合所有增强功能模块
"""
import time
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

from .config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME,
    TEMPERATURE,
    TOP_K,
    MAX_FALLBACK_ROUNDS,
    MAX_ITERATION_ROUNDS,
    WEB_SEARCH_ENABLED,
    WEB_SEARCH_TRIGGER_THRESHOLD
)
from .vector_store import (
    similarity_search_with_score,
    hybrid_search,
    convert_distance_to_similarity,
    BM25Index
)
from .retrieval_judge import RetrievalJudge, JudgeResult
from .query_transformer import QueryTransformer
from .reranker import Reranker, RerankResult
from .task_decomposer import TaskDecomposer, DecompositionResult
from .evidence_aggregator import EvidenceAggregator, Evidence
from .logger import StructuredLogger, EvidenceItem
from .intent_router import detect_intent, Intent
from .state import ConversationState
from .agents import LocalEvidenceAgent, WebResearchAgent, ArbiterAgent


@dataclass
class EnhancedAnswerResult:
    """增强版问答结果"""
    answer: str                             # 最终答案
    intent: str                             # 识别的意图
    trace_id: str                           # 追踪ID

    # 证据信息
    evidence_local: List[Dict[str, Any]]    # 本地证据
    evidence_web: List[Dict[str, Any]]      # 网络证据

    # 流程信息
    used_fallback: bool                     # 是否使用回退
    used_iteration: bool                    # 是否使用迭代
    used_web_search: bool                   # 是否使用联网搜索

    # 统计信息
    retrieval_stats: Dict[str, Any]         # 检索统计
    total_duration_ms: float                # 总耗时

    # 记忆（兼容原有功能）
    new_memories: List[str] = field(default_factory=list)


class EnhancedRAGChain:
    """增强版RAG问答链"""

    def __init__(
        self,
        vector_store: FAISS,
        bm25_index: Optional[BM25Index] = None
    ):
        """
        初始化增强版RAG链

        Args:
            vector_store: FAISS向量存储
            bm25_index: BM25索引（可选，用于混合检索）
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index

        # 初始化各组件
        self._llm = None
        self._judge = RetrievalJudge()
        self._transformer = QueryTransformer()
        self._reranker = Reranker()
        self._decomposer = TaskDecomposer()

        # 智能体
        self._local_agent = LocalEvidenceAgent(vector_store)
        self._web_agent = WebResearchAgent()
        self._arbiter = ArbiterAgent()

    def _get_llm(self) -> ChatOpenAI:
        """获取LLM实例"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_BASE_URL
            )
        return self._llm

    def _should_use_history(self, question: str) -> bool:
        q = question.strip()
        if len(q) <= 6:
            return True
        followup_markers = (
            "那", "这样", "这种", "这个", "如果", "要是", "怎么办",
            "会怎样", "会不会", "还能", "还可以", "还要", "还有", "然后", "再",
            "继续", "之后", "前面", "刚才"
        )
        return any(marker in q for marker in followup_markers)

    def _contextualize_question(
        self,
        question: str,
        state: Optional[ConversationState]
    ) -> str:
        if not state or not state.history:
            return question
        if not self._should_use_history(question):
            return question
        last_user = ""
        for msg in reversed(state.history):
            if msg.role == "user":
                last_user = msg.content.strip()
                break
        if not last_user or last_user in question:
            return question
        return f"{last_user} {question}"

    def answer(
        self,
        question: str,
        state: Optional[ConversationState] = None,
        enable_web_search: bool = True
    ) -> EnhancedAnswerResult:
        """
        回答问题（同步版本）

        Args:
            question: 用户问题
            state: 对话状态（可选）
            enable_web_search: 是否启用联网搜索

        Returns:
            EnhancedAnswerResult
        """
        # 创建日志记录器
        logger = StructuredLogger(question=question)
        start_time = time.time()

        # 意图识别
        context_question = self._contextualize_question(question, state)
        intent, confidence = detect_intent(context_question)
        logger.set_intent(intent.value)

        # 判断是否需要迭代查询
        decomposition = self._decomposer.decompose(context_question)
        logger.log_decomposition(
            context_question,
            [sq.question for sq in decomposition.sub_questions],
            decomposition.should_decompose,
            decomposition.duration_ms
        )

        # 执行检索（可能包含回退和迭代）
        if decomposition.should_decompose:
            # 迭代查询模式
            docs_with_scores, fallback_used = self._iterative_retrieval(
                context_question, decomposition, logger
            )
            used_iteration = True
        else:
            # 单次检索模式（无回退）
            retrieval_start = time.time()
            docs_with_scores = similarity_search_with_score(
                self.vector_store, context_question, k=TOP_K
            )
            retrieval_duration = (time.time() - retrieval_start) * 1000
            logger.log_retrieval(
                context_question, "vector", TOP_K,
                [{'content': doc.page_content[:100], 'source': doc.metadata.get('source', ''),
                  'score': float(score)} for doc, score in docs_with_scores],
                retrieval_duration
            )
            fallback_used = False
            used_iteration = False

        # 重排
        if docs_with_scores:
            rerank_start = time.time()
            docs_with_sim = [
                (doc, convert_distance_to_similarity(score) if score > 1 else score)
                for doc, score in docs_with_scores
            ]
            rerank_result = self._reranker.rerank(context_question, docs_with_sim)
            logger.log_rerank(
                rerank_result.before_ranking,
                rerank_result.after_ranking,
                rerank_result.duration_ms
            )
            final_docs = [(rd.document, rd.final_score) for rd in rerank_result.ranked_docs]
        else:
            final_docs = []

        # 判断是否需要联网搜索
        web_result = None
        used_web_search = False

        if enable_web_search and WEB_SEARCH_ENABLED:
            # 检查本地证据是否充分
            if not final_docs or self._should_use_web_search(context_question, final_docs):
                web_result = self._web_agent.research(context_question)
                used_web_search = web_result.search_success

                if web_result.search_success:
                    logger.log_web_search(
                        web_result.search_query,
                        [{'title': e.title, 'url': e.url, 'snippet': e.content[:200]}
                         for e in web_result.evidence],
                        web_result.duration_ms
                    )

        # 生成最终答案
        if used_web_search and web_result:
            # 使用多智能体协作
            local_result = self._local_agent.research(context_question)
            final_answer = self._arbiter.arbitrate(context_question, local_result, web_result)

            answer = final_answer.answer
            evidence_local = [
                {'content': e.content, 'source': e.source, 'score': float(e.score)}
                for e in final_answer.evidence_local
            ]
            evidence_web = [
                {'content': e.content, 'source': e.source, 'score': float(e.score)}
                for e in final_answer.evidence_web
            ]
        else:
            # 仅使用本地证据
            answer = self._generate_answer(context_question, final_docs, intent, state)
            evidence_local = [
                {
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', doc.metadata.get('title', 'unknown')),
                    'score': float(score)  # 转换为Python float以支持JSON序列化
                }
                for doc, score in final_docs
            ]
            evidence_web = []

        # 提取记忆（兼容原有功能）
        new_memories = self._extract_memories(answer)

        if state:
            state.add_user_message(question, intent=intent.value)
            state.add_assistant_message(answer)

        # 记录最终结果
        logger.set_answer(answer)
        for ev in evidence_local:
            logger.add_local_evidence(ev['content'], ev['source'], ev['score'])
        for ev in evidence_web:
            logger.add_web_evidence(ev['content'], ev['source'], ev['score'])

        # 保存日志
        logger.save()

        total_duration = (time.time() - start_time) * 1000

        return EnhancedAnswerResult(
            answer=answer,
            intent=intent.value,
            trace_id=logger.trace_id,
            evidence_local=evidence_local,
            evidence_web=evidence_web,
            used_fallback=fallback_used,
            used_iteration=used_iteration,
            used_web_search=used_web_search,
            retrieval_stats={
                'local_evidence_count': len(evidence_local),
                'web_evidence_count': len(evidence_web),
                'fallback_rounds': len(logger.trace.fallbacks),
                'iteration_rounds': len(logger.trace.iterations)
            },
            total_duration_ms=total_duration,
            new_memories=new_memories
        )

    async def answer_async(
        self,
        question: str,
        state: Optional[ConversationState] = None,
        enable_web_search: bool = True
    ) -> EnhancedAnswerResult:
        """
        回答问题（异步版本，支持并行执行多智能体）
        """
        logger = StructuredLogger(question=question)
        start_time = time.time()

        # 意图识别
        context_question = self._contextualize_question(question, state)
        intent, confidence = detect_intent(context_question)
        logger.set_intent(intent.value)

        # 并行执行本地和网络研究
        if enable_web_search and WEB_SEARCH_ENABLED:
            local_task = self._local_agent.research_async(context_question)
            web_task = self._web_agent.research_async(context_question)

            local_result, web_result = await asyncio.gather(local_task, web_task)

            # 裁决
            final_answer = await self._arbiter.arbitrate_async(
                context_question, local_result, web_result
            )

            answer = final_answer.answer
            evidence_local = [
                {'content': e.content, 'source': e.source, 'score': float(e.score)}
                for e in final_answer.evidence_local
            ]
            evidence_web = [
                {'content': e.content, 'source': e.source, 'score': float(e.score)}
                for e in final_answer.evidence_web
            ]
            used_web_search = web_result.search_success
        else:
            local_result = await self._local_agent.research_async(context_question)
            answer = local_result.summary
            evidence_local = [
                {'content': e.content, 'source': e.source, 'score': float(e.score)}
                for e in local_result.evidence
            ]
            evidence_web = []
            used_web_search = False

        new_memories = self._extract_memories(answer)

        if state:
            state.add_user_message(question, intent=intent.value)
            state.add_assistant_message(answer)

        logger.set_answer(answer)
        logger.save()

        total_duration = (time.time() - start_time) * 1000

        return EnhancedAnswerResult(
            answer=answer,
            intent=intent.value,
            trace_id=logger.trace_id,
            evidence_local=evidence_local,
            evidence_web=evidence_web,
            used_fallback=False,
            used_iteration=False,
            used_web_search=used_web_search,
            retrieval_stats={
                'local_evidence_count': len(evidence_local),
                'web_evidence_count': len(evidence_web)
            },
            total_duration_ms=total_duration,
            new_memories=new_memories
        )

    def _single_retrieval_with_fallback(
        self,
        question: str,
        logger: StructuredLogger
    ) -> Tuple[List[Tuple[Document, float]], bool]:
        """
        单次检索，带回退策略

        Returns:
            (文档列表, 是否使用了回退)
        """
        fallback_used = False
        current_query = question

        for round_num in range(MAX_FALLBACK_ROUNDS + 1):
            # 执行检索
            retrieval_start = time.time()

            if round_num == 0 or round_num < MAX_FALLBACK_ROUNDS:
                # 向量检索
                docs_with_scores = similarity_search_with_score(
                    self.vector_store, current_query, k=TOP_K
                )
                method = "vector"
            else:
                # 最后一轮使用混合检索
                docs_with_scores = hybrid_search(
                    self.vector_store, current_query, k=TOP_K,
                    bm25_index=self.bm25_index
                )
                method = "hybrid"

            retrieval_duration = (time.time() - retrieval_start) * 1000

            # 记录检索
            logger.log_retrieval(
                current_query, method, TOP_K,
                [{'content': doc.page_content[:100], 'source': doc.metadata.get('source', ''),
                  'score': float(score)} for doc, score in docs_with_scores],
                retrieval_duration
            )

            # 判断是否失败
            if round_num == 0:
                judge_result = self._judge.judge(
                    question, docs_with_scores, score_is_distance=True
                )
                logger.log_failure_judgment(
                    judge_result.similarity_score,
                    self._judge.similarity_threshold,
                    judge_result.relevant_count,
                    self._judge.relevant_threshold,
                    judge_result.coverage_score,
                    self._judge.coverage_threshold,
                    judge_result.is_failed
                )

                if not judge_result.is_failed:
                    return docs_with_scores, False

            # 需要回退
            fallback_used = True

            if round_num == 0:
                # 查询改写
                transform_result = self._transformer.rewrite(question)
                current_query = transform_result.transformed_query
                strategy = "rewrite"
            elif round_num == 1:
                # 查询扩展
                transform_result = self._transformer.expand(question)
                current_query = transform_result.transformed_query
                strategy = "expand"
            else:
                # 混合检索（已在检索时处理）
                strategy = "hybrid"

            logger.log_fallback(
                round_num + 1, strategy, question, current_query,
                len(docs_with_scores) > 0
            )

            # 检查回退后是否成功
            if docs_with_scores:
                judge_result = self._judge.judge(
                    question, docs_with_scores, score_is_distance=(method == "vector")
                )
                if not judge_result.is_failed:
                    return docs_with_scores, fallback_used

        # 所有回退都失败，返回最后的结果
        return docs_with_scores, fallback_used

    def _iterative_retrieval(
        self,
        question: str,
        decomposition: DecompositionResult,
        logger: StructuredLogger
    ) -> Tuple[List[Tuple[Document, float]], bool]:
        """
        迭代检索

        Returns:
            (聚合后的文档列表, 是否使用了回退)
        """
        aggregator = EvidenceAggregator()
        fallback_used = False

        for i, sub_q in enumerate(decomposition.sub_questions[:MAX_ITERATION_ROUNDS]):
            # 检索子问题
            retrieval_start = time.time()
            docs_with_scores = similarity_search_with_score(
                self.vector_store, sub_q.question, k=TOP_K
            )
            retrieval_duration = (time.time() - retrieval_start) * 1000

            # 添加到聚合器
            new_count = aggregator.add_round(docs_with_scores, sub_q.question)

            # 检查收敛
            converged, score = aggregator.check_convergence(question)

            logger.log_iteration(
                i + 1, sub_q.question, None,
                new_count, aggregator.evidence_count, converged
            )

            if converged:
                break

        # 获取最终证据
        final_evidence = aggregator.get_final_evidence()

        # 转换格式
        result_docs = []
        for ev in final_evidence:
            doc = Document(
                page_content=ev.content,
                metadata={'source': ev.source, **ev.metadata}
            )
            result_docs.append((doc, ev.score))

        return result_docs, fallback_used

    def _should_use_web_search(
        self,
        question: str,
        docs_with_scores: List[Tuple[Document, float]]
    ) -> bool:
        """判断是否需要使用联网搜索"""
        if not docs_with_scores:
            return True

        # 检查证据覆盖度（使用配置阈值，越高越积极触发联网）
        judge_result = self._judge.judge(question, docs_with_scores, score_is_distance=False)
        return judge_result.coverage_score < WEB_SEARCH_TRIGGER_THRESHOLD

    def _generate_answer(
        self,
        question: str,
        docs_with_scores: List[Tuple[Document, float]],
        intent: Intent,
        state: Optional[ConversationState]
    ) -> str:
        """生成答案"""
        if not docs_with_scores:
            return "抱歉，未能在本地知识库中找到相关信息。\n\n【来源说明】本地: 未找到 网络: 未使用"

        # 准备上下文
        context = "\n\n".join([
            f"【参考资料{i+1}】来源：{doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for i, (doc, _) in enumerate(docs_with_scores[:5])
        ])

        # 用户信息
        user_info = ""
        memories = ""
        history = ""
        if state:
            user_info = state.get_profile_summary()
            memories = state.get_memories_text()
            history = state.get_history_text(3)
        if history:
            memories = f"{memories}\n\nRecent conversation:\n{history}"

        prompt = f"""你是首都师范大学的学业咨询助手。请基于以下参考资料回答用户问题。

【用户信息】
{user_info if user_info else "未提供"}

【用户记录】
{memories if memories else "无"}

【参考资料】
{context}

【用户问题】
{question}

【回答要求】
1. 只基于参考资料回答，不要编造信息
2. 如果资料不足以完整回答，要明确说明
3. 引用具体条款或数字时要准确
4. 语气友好专业

请回答："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            answer = response.content.strip()
            return f"{answer}\n\n【来源说明】本地: 已使用 网络: 未使用"
        except Exception as e:
            return f"生成回答时出错：{str(e)}\n\n【来源说明】本地: 已使用 网络: 未使用"

    def _extract_memories(self, answer: str) -> List[str]:
        """从答案中提取记忆标记"""
        import re
        pattern = r'\[记忆:([^\]]+)\]'
        memories = re.findall(pattern, answer)
        return memories


def create_enhanced_chain(
    vector_store: FAISS,
    bm25_index: Optional[BM25Index] = None
) -> EnhancedRAGChain:
    """创建增强版RAG链的便捷函数"""
    return EnhancedRAGChain(vector_store, bm25_index)
