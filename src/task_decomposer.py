"""
任务分解器 - 将复杂问题分解为子问题
"""
import time
from typing import List, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI

from .config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME,
    MAX_SUB_QUESTIONS
)


@dataclass
class SubQuestion:
    """子问题"""
    question: str           # 子问题文本
    index: int              # 序号
    focus: str              # 关注点描述


@dataclass
class DecompositionResult:
    """分解结果"""
    original_question: str          # 原始问题
    should_decompose: bool          # 是否需要分解
    sub_questions: List[SubQuestion]  # 子问题列表
    complexity_score: float         # 复杂度评分 (0-1)
    reason: str                     # 分解/不分解的原因
    duration_ms: float              # 耗时


class TaskDecomposer:
    """任务分解器"""

    def __init__(self, max_sub_questions: int = MAX_SUB_QUESTIONS):
        """
        初始化任务分解器

        Args:
            max_sub_questions: 最大子问题数量
        """
        self.max_sub_questions = max_sub_questions
        self._llm = None

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

    def should_decompose(self, question: str) -> tuple[bool, float, str]:
        """
        判断问题是否需要分解

        Args:
            question: 用户问题

        Returns:
            (是否需要分解, 复杂度分数, 原因)
        """
        prompt = f"""请判断以下问题是否需要分解为多个子问题来回答。

问题：{question}

判断标准：
1. 问题是否涉及多个不同的主题或方面？
2. 问题是否需要综合多种类型的信息？
3. 问题是否包含多个独立的子问题？
4. 直接搜索是否难以找到完整答案？

请输出：
第一行：是/否（是否需要分解）
第二行：0.0-1.0之间的数字（复杂度分数）
第三行：简短说明原因

只输出这三行，不要其他内容："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            lines = response.content.strip().split('\n')

            should = lines[0].strip() in ['是', '需要', 'yes', 'Yes', 'YES']
            complexity = float(lines[1].strip()) if len(lines) > 1 else 0.5
            reason = lines[2].strip() if len(lines) > 2 else ""

            return should, complexity, reason
        except Exception:
            # 默认不分解
            return False, 0.3, "判断失败，默认不分解"

    def decompose(self, question: str) -> DecompositionResult:
        """
        分解问题为子问题

        Args:
            question: 用户问题

        Returns:
            DecompositionResult
        """
        start_time = time.time()

        # 先判断是否需要分解
        should, complexity, reason = self.should_decompose(question)

        if not should:
            duration = (time.time() - start_time) * 1000
            return DecompositionResult(
                original_question=question,
                should_decompose=False,
                sub_questions=[],
                complexity_score=complexity,
                reason=reason,
                duration_ms=duration
            )

        # 执行分解
        sub_questions = self._do_decompose(question)

        duration = (time.time() - start_time) * 1000

        return DecompositionResult(
            original_question=question,
            should_decompose=True,
            sub_questions=sub_questions,
            complexity_score=complexity,
            reason=reason,
            duration_ms=duration
        )

    def _do_decompose(self, question: str) -> List[SubQuestion]:
        """
        执行问题分解

        Args:
            question: 用户问题

        Returns:
            子问题列表
        """
        prompt = f"""请将以下复杂问题分解为{self.max_sub_questions}个以内的独立子问题，每个子问题都应该可以独立检索。

原始问题：{question}

分解要求：
1. 每个子问题应该聚焦于一个具体方面
2. 子问题之间应该覆盖原始问题的不同角度
3. 子问题的表述应该清晰、简洁
4. 子问题应该适合在知识库中检索

请按以下格式输出（每个子问题一行）：
子问题1 | 关注点描述
子问题2 | 关注点描述
...

只输出子问题，不要其他内容："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)

            sub_questions = []
            for i, line in enumerate(response.content.strip().split('\n')):
                line = line.strip()
                if not line:
                    continue

                # 解析格式：子问题 | 关注点
                parts = line.split('|')
                question_text = parts[0].strip()
                focus = parts[1].strip() if len(parts) > 1 else ""

                # 去除可能的序号前缀
                for prefix in ['子问题', '问题', '1.', '2.', '3.', '4.', '5.', '1、', '2、', '3、', '4、', '5、']:
                    if question_text.startswith(prefix):
                        question_text = question_text[len(prefix):].strip()
                        break

                if question_text:
                    sub_questions.append(SubQuestion(
                        question=question_text,
                        index=i + 1,
                        focus=focus
                    ))

                if len(sub_questions) >= self.max_sub_questions:
                    break

            return sub_questions

        except Exception:
            # 分解失败，返回原问题作为唯一子问题
            return [SubQuestion(question=question, index=1, focus="原始问题")]

    def force_decompose(self, question: str, num_parts: int = 3) -> List[SubQuestion]:
        """
        强制分解问题（不判断是否需要分解）

        Args:
            question: 用户问题
            num_parts: 分解数量

        Returns:
            子问题列表
        """
        prompt = f"""请将以下问题分解为恰好{num_parts}个不同角度的子问题。

原始问题：{question}

要求：
1. 每个子问题从不同角度切入
2. 子问题之间不要重复
3. 每个子问题独立可检索

请输出{num_parts}个子问题，每行一个，不要序号和其他内容："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)

            sub_questions = []
            for i, line in enumerate(response.content.strip().split('\n')):
                line = line.strip()
                if line:
                    sub_questions.append(SubQuestion(
                        question=line,
                        index=i + 1,
                        focus=""
                    ))

                if len(sub_questions) >= num_parts:
                    break

            return sub_questions

        except Exception:
            return [SubQuestion(question=question, index=1, focus="")]


def decompose_question(question: str) -> DecompositionResult:
    """
    分解问题的便捷函数
    """
    decomposer = TaskDecomposer()
    return decomposer.decompose(question)


def get_sub_questions(question: str) -> List[str]:
    """
    获取子问题文本列表的便捷函数
    """
    result = decompose_question(question)
    if result.should_decompose:
        return [sq.question for sq in result.sub_questions]
    return [question]
