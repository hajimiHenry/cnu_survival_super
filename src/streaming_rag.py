"""
流式 RAG 处理器 - 支持 SSE 实时推送智能体状态
"""
import time
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
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
    WEB_SEARCH_ENABLED,
    WEB_SEARCH_TRIGGER_THRESHOLD
)
from .vector_store import (
    similarity_search_with_score,
    convert_distance_to_similarity,
    BM25Index
)
from .retrieval_judge import RetrievalJudge
from .query_transformer import QueryTransformer
from .reranker import Reranker
from .task_decomposer import TaskDecomposer
from .intent_router import detect_intent, Intent
from .state import ConversationState
from .stream_events import (
    StreamEvent, EventType,
    event_start, event_intent, event_complete, event_error,
    event_local_start, event_local_search, event_local_rerank,
    event_local_think, event_local_done,
    event_web_start, event_web_search, event_web_think, event_web_done, event_web_skip,
    event_arbiter_start, event_arbiter_conflict, event_arbiter_think, event_arbiter_done,
    event_generate_start, event_generate_done
)
from .web_search import WebSearcher


@dataclass
class StreamingResult:
    """流式处理最终结果"""
    answer: str
    intent: str
    trace_id: str = ""
    evidence_local: List[Dict[str, Any]] = field(default_factory=list)
    evidence_web: List[Dict[str, Any]] = field(default_factory=list)
    used_fallback: bool = False
    used_iteration: bool = False
    used_web_search: bool = False
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)
    total_duration_ms: float = 0
    new_memories: List[str] = field(default_factory=list)


class StreamingRAGProcessor:
    """流式 RAG 处理器"""

    def __init__(
        self,
        vector_store: FAISS,
        bm25_index: Optional[BM25Index] = None
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self._llm = None
        self._judge = RetrievalJudge()
        self._transformer = QueryTransformer()
        self._reranker = Reranker()
        self._decomposer = TaskDecomposer()
        self._web_searcher = WebSearcher()

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

    async def process_stream(
        self,
        question: str,
        state: Optional[ConversationState] = None,
        enable_web_search: bool = True
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        流式处理问题，yield 事件

        Args:
            question: 用户问题
            state: 对话状态
            enable_web_search: 是否启用联网搜索

        Yields:
            StreamEvent 事件
        """
        start_time = time.time()

        # 开始处理
        yield event_start(question)
        await asyncio.sleep(0.05)  # 小延迟确保前端能收到

        try:
            # ========== 1. 意图识别 ==========
            yield StreamEvent(
                event_type=EventType.INTENT_START,
                agent="意图识别",
                message="正在分析问题类型..."
            )

            intent, confidence = await asyncio.get_event_loop().run_in_executor(
                None, detect_intent, question
            )
            yield event_intent(intent.value, confidence)
            await asyncio.sleep(0.05)

            # ========== 2. LocalAgent 处理 ==========
            yield event_local_start()
            await asyncio.sleep(0.05)

            # 检索
            docs_with_scores = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: similarity_search_with_score(self.vector_store, question, k=TOP_K)
            )
            yield event_local_search(question, len(docs_with_scores))
            await asyncio.sleep(0.05)

            # 检查是否需要回退
            fallback_used = False
            if self._judge.quick_judge(docs_with_scores, score_is_distance=True):
                yield StreamEvent(
                    event_type=EventType.FALLBACK_START,
                    agent="LocalAgent",
                    message="检索结果不理想，尝试查询改写..."
                )

                # 查询改写
                rewrite_result = await asyncio.get_event_loop().run_in_executor(
                    None, self._transformer.rewrite, question
                )
                yield StreamEvent(
                    event_type=EventType.FALLBACK_DONE,
                    agent="LocalAgent",
                    message="查询已改写",
                    detail=f"新查询: {rewrite_result.transformed_query[:50]}..."
                )

                # 重新检索
                new_docs = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: similarity_search_with_score(
                        self.vector_store, rewrite_result.transformed_query, k=TOP_K
                    )
                )
                if new_docs:
                    docs_with_scores = self._merge_results(docs_with_scores, new_docs)
                    fallback_used = True
                    yield event_local_search(rewrite_result.transformed_query, len(docs_with_scores))

            # 重排
            if docs_with_scores:
                yield event_local_rerank(len(docs_with_scores), 0)

                docs_with_sim = [
                    (doc, convert_distance_to_similarity(score))
                    for doc, score in docs_with_scores
                ]
                rerank_result = await asyncio.get_event_loop().run_in_executor(
                    None, self._reranker.rerank, question, docs_with_sim
                )
                final_local_docs = [
                    (rd.document, rd.final_score)
                    for rd in rerank_result.ranked_docs
                ]
                yield event_local_rerank(len(docs_with_scores), len(final_local_docs))
            else:
                final_local_docs = []

            # 生成 LocalAgent 总结
            yield event_local_think("正在分析检索到的文档，提取关键信息...")
            await asyncio.sleep(0.05)

            local_summary, local_key_points = await self._generate_local_summary(
                question, final_local_docs
            )

            local_evidence = [
                {
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', doc.metadata.get('title', 'unknown')),
                    'score': float(score)
                }
                for doc, score in final_local_docs
            ]

            yield event_local_done(local_summary, local_key_points, len(local_evidence))
            await asyncio.sleep(0.05)

            # ========== 3. 判断是否需要 WebAgent ==========
            web_evidence = []
            web_summary = ""
            web_key_points = []
            used_web_search = False

            need_web = False
            if enable_web_search and WEB_SEARCH_ENABLED:
                if not final_local_docs:
                    need_web = True
                    yield StreamEvent(
                        event_type=EventType.WEB_AGENT_START,
                        agent="WebAgent",
                        message="本地证据不足，启动联网搜索",
                        detail="本地知识库未找到相关信息"
                    )
                else:
                    # 检查覆盖度
                    judge_result = self._judge.judge(question, final_local_docs, score_is_distance=False)
                    if judge_result.coverage_score < WEB_SEARCH_TRIGGER_THRESHOLD:
                        need_web = True
                        yield StreamEvent(
                            event_type=EventType.WEB_AGENT_START,
                            agent="WebAgent",
                            message="本地证据覆盖度不足，启动联网搜索",
                            detail=f"覆盖度: {judge_result.coverage_score:.2f} < {WEB_SEARCH_TRIGGER_THRESHOLD}"
                        )

            if need_web:
                # ========== 4. WebAgent 处理 ==========
                yield event_web_start()
                await asyncio.sleep(0.05)

                # 优化搜索查询
                search_query = await self._optimize_web_query(question)
                yield StreamEvent(
                    event_type=EventType.WEB_AGENT_SEARCH,
                    agent="WebAgent",
                    message="正在搜索网络...",
                    detail=f"查询: {search_query[:50]}..."
                )

                # 执行搜索
                search_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._web_searcher.search(
                        query=search_query,
                        max_results=5,
                        search_depth="advanced",
                        include_answer=True
                    )
                )

                if search_response.success and search_response.results:
                    yield event_web_search(search_query, len(search_response.results))
                    await asyncio.sleep(0.05)

                    # 生成 WebAgent 总结
                    yield event_web_think("正在分析网络搜索结果...")

                    web_evidence = [
                        {
                            'content': r.content,
                            'source': r.url,
                            'title': r.title,
                            'score': r.score
                        }
                        for r in search_response.results
                    ]

                    web_summary, web_key_points = await self._generate_web_summary(
                        question, web_evidence, search_response.answer
                    )

                    yield event_web_done(web_summary, web_key_points, len(web_evidence))
                    used_web_search = True
                else:
                    yield event_web_skip("搜索无结果或失败")
            else:
                if enable_web_search and WEB_SEARCH_ENABLED:
                    yield event_web_skip("本地证据充分，无需联网")
                else:
                    yield event_web_skip("联网搜索未启用")

            await asyncio.sleep(0.05)

            # ========== 5. Arbiter 裁决（如果有网络证据）==========
            if used_web_search and web_evidence:
                yield event_arbiter_start()
                await asyncio.sleep(0.05)

                # 检测冲突
                has_conflict, conflict_details = await self._detect_conflict(
                    local_summary, web_summary
                )
                yield event_arbiter_conflict(has_conflict, conflict_details)
                await asyncio.sleep(0.05)

                # 生成裁决
                yield event_arbiter_think("综合本地和网络证据，生成最终答案...")

                answer, confidence = await self._generate_arbiter_answer(
                    question, local_summary, local_key_points,
                    web_summary, web_key_points,
                    has_conflict, conflict_details
                )

                yield event_arbiter_done(confidence)
            else:
                # 仅使用本地证据
                yield event_generate_start()
                answer = await self._generate_local_answer(
                    question, final_local_docs, intent, state
                )
                yield event_generate_done()

            await asyncio.sleep(0.05)

            # ========== 6. 完成 ==========
            total_duration = (time.time() - start_time) * 1000

            result = StreamingResult(
                answer=answer,
                intent=intent.value,
                evidence_local=local_evidence,
                evidence_web=web_evidence,
                used_fallback=fallback_used,
                used_iteration=False,
                used_web_search=used_web_search,
                retrieval_stats={
                    'local_evidence_count': len(local_evidence),
                    'web_evidence_count': len(web_evidence)
                },
                total_duration_ms=total_duration
            )

            yield event_complete(asdict(result))

        except Exception as e:
            yield event_error(str(e))

    def _merge_results(
        self,
        results1: List[Tuple[Document, float]],
        results2: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """合并检索结果"""
        seen = set()
        merged = []
        for doc, score in results1 + results2:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                merged.append((doc, score))
        merged.sort(key=lambda x: x[1])
        return merged

    async def _generate_local_summary(
        self,
        question: str,
        docs: List[Tuple[Document, float]]
    ) -> Tuple[str, List[str]]:
        """生成本地证据总结"""
        if not docs:
            return "未找到相关本地证据", []

        evidence_text = "\n\n".join([
            f"【证据{i+1}】{doc.metadata.get('source', '')}\n{doc.page_content[:300]}"
            for i, (doc, _) in enumerate(docs[:5])
        ])

        prompt = f"""基于以下证据，简要总结与问题相关的信息。

问题：{question}

证据：
{evidence_text}

输出格式：
【总结】一句话总结
【要点】
1. 要点1
2. 要点2
3. 要点3"""

        try:
            llm = self._get_llm()
            response = await asyncio.get_event_loop().run_in_executor(
                None, llm.invoke, prompt
            )
            content = response.content.strip()

            # 解析
            summary = ""
            key_points = []
            lines = content.split('\n')
            in_summary = False
            in_points = False

            for line in lines:
                line = line.strip()
                if '【总结】' in line:
                    in_summary = True
                    in_points = False
                    summary = line.replace('【总结】', '').strip()
                    continue
                elif '【要点】' in line:
                    in_summary = False
                    in_points = True
                    continue

                if in_summary and line:
                    summary += " " + line
                elif in_points and line:
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•']:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                            break
                    if line:
                        key_points.append(line)

            return summary.strip(), key_points
        except Exception:
            return "基于本地知识库进行了检索", []

    async def _generate_web_summary(
        self,
        question: str,
        evidence: List[Dict],
        tavily_answer: str = ""
    ) -> Tuple[str, List[str]]:
        """生成网络证据总结"""
        if not evidence:
            return "未找到相关网络信息", []

        evidence_text = "\n\n".join([
            f"【来源{i+1}】{e.get('title', '')}\n{e.get('content', '')[:300]}"
            for i, e in enumerate(evidence[:5])
        ])

        prompt = f"""基于以下网络搜索结果，简要总结与问题相关的信息。

问题：{question}

搜索结果：
{evidence_text}

{f"参考答案：{tavily_answer}" if tavily_answer else ""}

输出格式：
【总结】一句话总结
【要点】
1. 要点1
2. 要点2"""

        try:
            llm = self._get_llm()
            response = await asyncio.get_event_loop().run_in_executor(
                None, llm.invoke, prompt
            )
            content = response.content.strip()

            summary = ""
            key_points = []
            lines = content.split('\n')
            in_summary = False
            in_points = False

            for line in lines:
                line = line.strip()
                if '【总结】' in line:
                    in_summary = True
                    in_points = False
                    summary = line.replace('【总结】', '').strip()
                elif '【要点】' in line:
                    in_summary = False
                    in_points = True
                elif in_summary and line:
                    summary += " " + line
                elif in_points and line:
                    for prefix in ['1.', '2.', '3.', '-', '•']:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                            break
                    if line:
                        key_points.append(line)

            return summary.strip(), key_points
        except Exception:
            return tavily_answer if tavily_answer else "已从网络获取信息", []

    async def _optimize_web_query(self, question: str) -> str:
        """优化网络搜索查询"""
        prompt = f"""将以下问题转换为适合网络搜索的查询词。
问题：{question}
只输出查询词："""

        try:
            llm = self._get_llm()
            response = await asyncio.get_event_loop().run_in_executor(
                None, llm.invoke, prompt
            )
            return response.content.strip()
        except Exception:
            return question

    async def _detect_conflict(
        self,
        local_summary: str,
        web_summary: str
    ) -> Tuple[bool, str]:
        """检测本地和网络证据冲突"""
        prompt = f"""判断以下两个信息来源是否存在矛盾。

本地知识库：{local_summary}

网络信息：{web_summary}

回答格式：
第一行：是/否
第二行：如有矛盾，说明矛盾点"""

        try:
            llm = self._get_llm()
            response = await asyncio.get_event_loop().run_in_executor(
                None, llm.invoke, prompt
            )
            lines = response.content.strip().split('\n')
            has_conflict = lines[0].strip() in ['是', '有']
            details = lines[1].strip() if len(lines) > 1 else ""
            return has_conflict, details
        except Exception:
            return False, ""

    async def _generate_arbiter_answer(
        self,
        question: str,
        local_summary: str,
        local_points: List[str],
        web_summary: str,
        web_points: List[str],
        has_conflict: bool,
        conflict_details: str
    ) -> Tuple[str, str]:
        """Arbiter 生成最终答案"""
        conflict_note = ""
        if has_conflict:
            conflict_note = f"\n注意：存在信息冲突 - {conflict_details}\n处理原则：以本地知识库为准。"

        prompt = f"""综合以下信息回答问题。

问题：{question}

【本地知识库】
{local_summary}
要点：{', '.join(local_points)}

【网络信息】
{web_summary}
要点：{', '.join(web_points)}
{conflict_note}

请生成综合答案，结尾标注置信度[高/中/低]："""

        try:
            llm = self._get_llm()
            response = await asyncio.get_event_loop().run_in_executor(
                None, llm.invoke, prompt
            )
            answer = response.content.strip()

            # 提取置信度
            confidence = "medium"
            if "[高]" in answer or "置信度：高" in answer:
                confidence = "high"
            elif "[低]" in answer or "置信度：低" in answer:
                confidence = "low"

            # 移除置信度标记
            for marker in ["[高]", "[中]", "[低]", "置信度：高", "置信度：中", "置信度：低"]:
                answer = answer.replace(marker, "")

            return answer.strip(), confidence
        except Exception as e:
            return f"综合信息回答：{local_summary}", "low"

    async def _generate_local_answer(
        self,
        question: str,
        docs: List[Tuple[Document, float]],
        intent: Intent,
        state: Optional[ConversationState]
    ) -> str:
        """仅使用本地证据生成答案"""
        if not docs:
            return "抱歉，未能在知识库中找到相关信息来回答您的问题。"

        context = "\n\n".join([
            f"【参考资料{i+1}】{doc.metadata.get('source', '')}\n{doc.page_content}"
            for i, (doc, _) in enumerate(docs[:5])
        ])

        user_info = ""
        memories = ""
        if state:
            user_info = state.get_profile_summary()
            memories = state.get_memories_text()

        prompt = f"""你是首都师范大学的学业咨询助手。请基于参考资料回答问题。

【用户信息】{user_info or "未提供"}
【用户记录】{memories or "无"}

【参考资料】
{context}

【问题】{question}

回答要求：只基于资料回答，语气友好。"""

        try:
            llm = self._get_llm()
            response = await asyncio.get_event_loop().run_in_executor(
                None, llm.invoke, prompt
            )
            return response.content.strip()
        except Exception as e:
            return f"生成回答时出错：{str(e)}"


def create_streaming_processor(
    vector_store: FAISS,
    bm25_index: Optional[BM25Index] = None
) -> StreamingRAGProcessor:
    """创建流式处理器"""
    return StreamingRAGProcessor(vector_store, bm25_index)
