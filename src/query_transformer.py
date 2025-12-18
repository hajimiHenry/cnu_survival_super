"""
查询转换器 - 查询改写、查询扩展、生成混合检索参数
"""
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI

from .config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    MODEL_NAME
)


@dataclass
class TransformResult:
    """转换结果"""
    original_query: str         # 原始查询
    transformed_query: str      # 转换后的查询
    strategy: str               # 使用的策略
    duration_ms: float          # 耗时


class QueryTransformer:
    """查询转换器"""

    def __init__(self):
        self._llm = None

    def _get_llm(self) -> ChatOpenAI:
        """获取LLM实例（延迟加载）"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=MODEL_NAME,
                temperature=0.3,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_BASE_URL
            )
        return self._llm

    def rewrite(self, query: str) -> TransformResult:
        """
        查询改写：将用户问题改写成更接近知识库文档用词的表达

        Args:
            query: 原始查询

        Returns:
            TransformResult
        """
        start_time = time.time()

        prompt = f"""你是一个查询改写专家。请将用户的问题改写成更适合在大学规章制度知识库中检索的形式。

改写原则：
1. 使用正式、书面的表达方式
2. 使用知识库中可能出现的专业术语
3. 保持原意，但表达更加规范
4. 去除口语化表达和语气词

用户问题：{query}

请直接输出改写后的查询，不要输出其他内容："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            transformed = response.content.strip()
        except Exception:
            transformed = query

        duration = (time.time() - start_time) * 1000

        return TransformResult(
            original_query=query,
            transformed_query=transformed,
            strategy="rewrite",
            duration_ms=duration
        )

    def expand(self, query: str) -> TransformResult:
        """
        查询扩展：抽取关键实体与同义词，扩展成更长的检索查询

        Args:
            query: 原始查询

        Returns:
            TransformResult
        """
        start_time = time.time()

        prompt = f"""你是一个查询扩展专家。请分析用户的问题，提取关键词并添加同义词和相关词，生成扩展后的查询。

扩展原则：
1. 保留原始问题中的所有关键词
2. 添加关键词的同义词（如：申请/办理、规定/制度）
3. 添加相关的上位词或下位词
4. 添加可能相关的专业术语
5. 生成的查询应该是一个完整的短语或句子

用户问题：{query}

请直接输出扩展后的查询（一行），不要输出其他内容："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            transformed = response.content.strip()
        except Exception:
            transformed = query

        duration = (time.time() - start_time) * 1000

        return TransformResult(
            original_query=query,
            transformed_query=transformed,
            strategy="expand",
            duration_ms=duration
        )

    def extract_keywords(self, query: str) -> List[str]:
        """
        提取查询中的关键词

        Args:
            query: 查询文本

        Returns:
            关键词列表
        """
        prompt = f"""请从以下问题中提取关键词，用于知识库检索。

问题：{query}

请输出关键词列表，每行一个关键词，不要输出其他内容："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            keywords = [kw.strip() for kw in response.content.strip().split('\n') if kw.strip()]
            return keywords
        except Exception:
            # 简单分词作为后备
            import jieba
            return list(jieba.cut(query))

    def generate_sub_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        生成多个子查询，用于多角度检索

        Args:
            query: 原始查询
            num_queries: 生成数量

        Returns:
            子查询列表
        """
        prompt = f"""请根据用户的问题，生成{num_queries}个不同角度的查询，用于在知识库中检索相关信息。

原始问题：{query}

生成原则：
1. 每个查询从不同角度表述同一个问题
2. 查询应该简洁明了
3. 覆盖问题的不同方面

请输出{num_queries}个查询，每行一个，不要输出其他内容："""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            return queries[:num_queries]
        except Exception:
            return [query]

    def get_hybrid_params(self, query: str) -> Dict[str, Any]:
        """
        为混合检索生成参数

        Args:
            query: 原始查询

        Returns:
            混合检索参数字典
        """
        # 提取关键词用于BM25
        keywords = self.extract_keywords(query)

        # 生成改写查询用于向量检索
        rewrite_result = self.rewrite(query)

        return {
            'vector_query': rewrite_result.transformed_query,
            'bm25_query': query,
            'keywords': keywords,
            'vector_weight': 0.6,  # 向量检索权重
            'bm25_weight': 0.4     # BM25权重
        }


# 便捷函数
def rewrite_query(query: str) -> str:
    """查询改写的便捷函数"""
    transformer = QueryTransformer()
    result = transformer.rewrite(query)
    return result.transformed_query


def expand_query(query: str) -> str:
    """查询扩展的便捷函数"""
    transformer = QueryTransformer()
    result = transformer.expand(query)
    return result.transformed_query


def get_hybrid_search_params(query: str) -> Dict[str, Any]:
    """获取混合检索参数的便捷函数"""
    transformer = QueryTransformer()
    return transformer.get_hybrid_params(query)
