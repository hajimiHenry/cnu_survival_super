"""
裁决智能体 - 综合本地和网络证据，输出最终答案
"""
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI

from ..config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME
)
from .local_agent import LocalResearchResult, LocalEvidence
from .web_agent import WebResearchResult, WebEvidence


@dataclass
class FinalEvidence:
    """最终证据条目"""
    content: str                # 证据内容
    source: str                 # 来源
    source_type: str            # 来源类型 (local/web)
    score: float                # 相关性分数
    used_in_answer: bool        # 是否在答案中使用


@dataclass
class FinalAnswer:
    """最终答案"""
    question: str                           # 原始问题
    answer: str                             # 最终答案
    evidence_local: List[FinalEvidence]     # 本地证据
    evidence_web: List[FinalEvidence]       # 网络证据
    has_conflict: bool                      # 是否存在冲突
    conflict_resolution: str                # 冲突解决说明
    confidence: str                         # 置信度 (high/medium/low)
    duration_ms: float                      # 耗时


class ArbiterAgent:
    """裁决智能体"""

    def __init__(self):
        """初始化裁决智能体"""
        self._llm = None

    def _get_llm(self) -> ChatOpenAI:
        """获取LLM实例"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=MODEL_NAME,
                temperature=0.2,  # 低温度保证判断稳定
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_BASE_URL
            )
        return self._llm

    def arbitrate(
        self,
        question: str,
        local_result: Optional[LocalResearchResult],
        web_result: Optional[WebResearchResult]
    ) -> FinalAnswer:
        """
        裁决并生成最终答案

        规则：
        1. 不新增事实，只基于证据
        2. 冲突时默认以本地证据为准
        3. 只有网络证据是权威原始文件且本地缺失时才采用网络证据

        Args:
            question: 原始问题
            local_result: 本地研究结果
            web_result: 联网研究结果

        Returns:
            FinalAnswer
        """
        start_time = time.time()

        # 收集证据
        local_evidence = self._convert_local_evidence(local_result)
        web_evidence = self._convert_web_evidence(web_result)

        # 检测冲突
        has_conflict, conflict_details = self._detect_conflict(
            local_result, web_result
        )

        # 生成答案
        answer, conflict_resolution, confidence = self._generate_answer(
            question,
            local_result,
            web_result,
            has_conflict,
            conflict_details
        )

        # 标记使用的证据
        self._mark_used_evidence(answer, local_evidence, web_evidence)

        duration = (time.time() - start_time) * 1000

        return FinalAnswer(
            question=question,
            answer=answer,
            evidence_local=local_evidence,
            evidence_web=web_evidence,
            has_conflict=has_conflict,
            conflict_resolution=conflict_resolution,
            confidence=confidence,
            duration_ms=duration
        )

    async def arbitrate_async(
        self,
        question: str,
        local_result: Optional[LocalResearchResult],
        web_result: Optional[WebResearchResult]
    ) -> FinalAnswer:
        """异步裁决"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.arbitrate, question, local_result, web_result
        )

    def _convert_local_evidence(
        self,
        local_result: Optional[LocalResearchResult]
    ) -> List[FinalEvidence]:
        """转换本地证据格式"""
        if not local_result or not local_result.evidence:
            return []

        return [
            FinalEvidence(
                content=ev.content,
                source=ev.source,
                source_type="local",
                score=ev.score,
                used_in_answer=False
            )
            for ev in local_result.evidence
        ]

    def _convert_web_evidence(
        self,
        web_result: Optional[WebResearchResult]
    ) -> List[FinalEvidence]:
        """转换网络证据格式"""
        if not web_result or not web_result.evidence:
            return []

        return [
            FinalEvidence(
                content=ev.content,
                source=ev.url,
                source_type="web",
                score=ev.score,
                used_in_answer=False
            )
            for ev in web_result.evidence
        ]

    def _detect_conflict(
        self,
        local_result: Optional[LocalResearchResult],
        web_result: Optional[WebResearchResult]
    ) -> Tuple[bool, str]:
        """
        检测本地和网络证据之间的冲突

        Returns:
            (是否有冲突, 冲突详情)
        """
        if not local_result or not web_result:
            return False, ""

        if not local_result.evidence or not web_result.evidence:
            return False, ""

        # 准备证据摘要
        local_summary = local_result.summary
        web_summary = web_result.summary

        prompt = f"""请判断以下两个来源的信息是否存在冲突。

本地知识库信息：
{local_summary}

网络搜索信息：
{web_summary}

请回答：
第一行：是/否（是否存在冲突）
第二行：如果有冲突，简要说明冲突点

只输出这两行："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            lines = response.content.strip().split('\n')

            has_conflict = lines[0].strip() in ['是', '有', 'yes', 'Yes']
            conflict_details = lines[1].strip() if len(lines) > 1 and has_conflict else ""

            return has_conflict, conflict_details
        except Exception:
            return False, ""

    def _generate_answer(
        self,
        question: str,
        local_result: Optional[LocalResearchResult],
        web_result: Optional[WebResearchResult],
        has_conflict: bool,
        conflict_details: str
    ) -> Tuple[str, str, str]:
        """
        生成最终答案

        Returns:
            (答案, 冲突解决说明, 置信度)
        """
        # 准备上下文
        local_context = ""
        if local_result and local_result.retrieval_success:
            local_context = f"""【本地知识库】
总结：{local_result.summary}
要点：
""" + "\n".join([f"- {p}" for p in local_result.key_points])

        web_context = ""
        if web_result and web_result.search_success:
            web_context = f"""【网络搜索】
总结：{web_result.summary}
要点：
""" + "\n".join([f"- {p}" for p in web_result.key_points])

        conflict_note = ""
        if has_conflict:
            conflict_note = f"""
【注意：存在信息冲突】
冲突点：{conflict_details}
处理原则：以本地知识库为准，除非网络来源是更权威的原始文件。
"""

        prompt = f"""你是裁决智能体，需要综合本地和网络信息生成最终答案。

用户问题：{question}

{local_context}

{web_context}

{conflict_note}

裁决规则：
1. 只能基于上述证据回答，不能添加新的事实
2. 如有冲突，默认以本地知识库为准
3. 网络信息作为补充，需要明确标注来源
4. 如果证据不足，要明确说明

请生成最终答案，格式：
【答案】
（综合回答）

【证据来源】
- 本地：（引用的本地证据要点）
- 网络：（引用的网络证据要点，如果有）

【置信度】
高/中/低（基于证据充分程度）

{"【冲突处理】" if has_conflict else ""}
{("说明如何处理冲突" if has_conflict else "")}

只输出以上内容："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            content = response.content.strip()

            # 解析答案
            answer = ""
            conflict_resolution = ""
            confidence = "medium"

            lines = content.split('\n')
            current_section = None

            for line in lines:
                line = line.strip()
                if '【答案】' in line:
                    current_section = 'answer'
                    continue
                elif '【证据来源】' in line:
                    current_section = 'source'
                    continue
                elif '【置信度】' in line:
                    current_section = 'confidence'
                    continue
                elif '【冲突处理】' in line:
                    current_section = 'conflict'
                    continue

                if current_section == 'answer' and line:
                    answer += line + "\n"
                elif current_section == 'confidence' and line:
                    if '高' in line:
                        confidence = 'high'
                    elif '低' in line:
                        confidence = 'low'
                    else:
                        confidence = 'medium'
                elif current_section == 'conflict' and line:
                    conflict_resolution += line + " "

            return answer.strip(), conflict_resolution.strip(), confidence

        except Exception as e:
            # 简单后备方案
            if local_result and local_result.retrieval_success:
                return local_result.summary, "", "low"
            elif web_result and web_result.search_success:
                return f"（来自网络）{web_result.summary}", "", "low"
            else:
                return "抱歉，未能找到足够的信息来回答您的问题。", "", "low"

    def _mark_used_evidence(
        self,
        answer: str,
        local_evidence: List[FinalEvidence],
        web_evidence: List[FinalEvidence]
    ):
        """标记在答案中使用的证据"""
        # 简单启发式：检查证据内容的关键词是否出现在答案中
        for ev in local_evidence + web_evidence:
            # 提取前50个字符作为关键词
            keywords = ev.content[:50]
            # 检查是否有显著重叠
            overlap = sum(1 for w in keywords if w in answer and len(w) > 1)
            if overlap > 5:
                ev.used_in_answer = True
