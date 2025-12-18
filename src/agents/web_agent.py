"""
联网研究智能体 - 只通过联网检索获取公开网页信息
"""
import time
import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI

from ..config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME,
    WEB_SEARCH_ENABLED
)
from ..web_search import WebSearcher, WebSearchResult


@dataclass
class WebEvidence:
    """网络证据条目"""
    content: str                # 证据内容
    url: str                    # 来源URL
    title: str                  # 页面标题
    score: float                # 相关性分数
    is_fact: bool               # 是否为事实（vs 推断）


@dataclass
class WebResearchResult:
    """联网研究结果"""
    question: str                       # 研究问题
    summary: str                        # 研究总结
    key_points: List[str]               # 关键要点
    evidence: List[WebEvidence]         # 证据列表
    search_success: bool                # 搜索是否成功
    search_query: str                   # 实际搜索查询
    duration_ms: float                  # 耗时


class WebResearchAgent:
    """联网研究智能体"""

    def __init__(self):
        """初始化联网研究智能体"""
        self._llm = None
        self._searcher = WebSearcher()

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

    def research(self, question: str, max_results: int = 5) -> WebResearchResult:
        """
        执行联网研究

        Args:
            question: 研究问题
            max_results: 最大搜索结果数

        Returns:
            WebResearchResult
        """
        start_time = time.time()

        # 检查是否启用联网搜索
        if not WEB_SEARCH_ENABLED:
            duration = (time.time() - start_time) * 1000
            return WebResearchResult(
                question=question,
                summary="联网搜索功能未启用。",
                key_points=[],
                evidence=[],
                search_success=False,
                search_query="",
                duration_ms=duration
            )

        # 优化搜索查询
        search_query = self._optimize_query(question)

        # 执行搜索
        search_response = self._searcher.search(
            query=search_query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=True
        )

        if not search_response.success:
            duration = (time.time() - start_time) * 1000
            return WebResearchResult(
                question=question,
                summary=f"联网搜索失败：{search_response.error}",
                key_points=[],
                evidence=[],
                search_success=False,
                search_query=search_query,
                duration_ms=duration
            )

        # 转换为证据格式
        evidence = []
        for result in search_response.results:
            evidence.append(WebEvidence(
                content=result.content,
                url=result.url,
                title=result.title,
                score=result.score,
                is_fact=True  # 默认为事实，后续可以添加判断
            ))

        # 生成总结和要点
        if evidence:
            summary, key_points = self._generate_summary(question, evidence, search_response.answer)
            search_success = True
        else:
            summary = "未能从网络上找到相关信息。"
            key_points = []
            search_success = False

        duration = (time.time() - start_time) * 1000

        return WebResearchResult(
            question=question,
            summary=summary,
            key_points=key_points,
            evidence=evidence,
            search_success=search_success,
            search_query=search_query,
            duration_ms=duration
        )

    async def research_async(self, question: str, max_results: int = 5) -> WebResearchResult:
        """
        异步执行联网研究

        Args:
            question: 研究问题
            max_results: 最大搜索结果数

        Returns:
            WebResearchResult
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.research, question, max_results)

    def _optimize_query(self, question: str) -> str:
        """
        优化搜索查询

        Args:
            question: 原始问题

        Returns:
            优化后的查询
        """
        prompt = f"""请将以下问题优化为适合网络搜索的查询词。

原始问题：{question}

优化要求：
1. 提取关键词和实体
2. 添加必要的限定词（如大学名称、政策年份等）
3. 去除口语化表达
4. 保持简洁

只输出优化后的查询词，不要其他内容："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            # 简单后备：提取关键词
            return question

    def _generate_summary(
        self,
        question: str,
        evidence: List[WebEvidence],
        tavily_answer: str = ""
    ) -> Tuple[str, List[str]]:
        """
        生成研究总结和要点

        Args:
            question: 问题
            evidence: 证据列表
            tavily_answer: Tavily生成的答案（如果有）

        Returns:
            (总结, 要点列表)
        """
        evidence_text = "\n\n".join([
            f"【来源{i+1}】{ev.title}\nURL: {ev.url}\n内容: {ev.content}"
            for i, ev in enumerate(evidence[:5])
        ])

        prompt = f"""基于以下网络搜索结果，回答用户问题并提取关键要点。

用户问题：{question}

网络搜索结果：
{evidence_text}

{"Tavily参考答案：" + tavily_answer if tavily_answer else ""}

请按以下格式输出：
【总结】
（用2-3句话总结从网络上找到的信息，必须基于搜索结果）

【要点】
1. （第一个要点）[事实/推断]
2. （第二个要点）[事实/推断]
3. （第三个要点，如果有）[事实/推断]

注意：
- 区分事实（有明确来源支持）和推断（基于信息推理得出）
- 不要添加搜索结果中没有的信息

只输出总结和要点："""

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
                if '【总结】' in line:
                    in_summary = True
                    in_points = False
                    continue
                elif '【要点】' in line:
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
            # 使用Tavily答案作为后备
            if tavily_answer:
                return tavily_answer, []
            return "已从网络获取相关信息。", []
