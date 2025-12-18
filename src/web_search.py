"""
联网搜索模块 - 封装 Tavily Search API
"""
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .config import TAVILY_API_KEY, WEB_SEARCH_ENABLED


@dataclass
class WebSearchResult:
    """搜索结果条目"""
    title: str              # 标题
    url: str                # URL
    content: str            # 内容摘要
    score: float            # 相关性分数
    raw_content: str = ""   # 原始内容（如果有）


@dataclass
class WebSearchResponse:
    """搜索响应"""
    query: str                      # 搜索查询
    results: List[WebSearchResult]  # 结果列表
    answer: str                     # Tavily生成的答案（如果有）
    duration_ms: float              # 耗时
    success: bool                   # 是否成功
    error: str = ""                 # 错误信息


class WebSearcher:
    """联网搜索器"""

    def __init__(self, api_key: str = None):
        """
        初始化搜索器

        Args:
            api_key: Tavily API Key，如果不提供则使用配置中的
        """
        self.api_key = api_key or TAVILY_API_KEY
        self._client = None

    def _get_client(self):
        """获取Tavily客户端（延迟加载）"""
        if self._client is None:
            if not self.api_key:
                raise ValueError("未配置 TAVILY_API_KEY")
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                raise ImportError("请安装 tavily-python: pip install tavily-python")
        return self._client

    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_answer: bool = True,
        include_raw_content: bool = False
    ) -> WebSearchResponse:
        """
        执行搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数
            search_depth: 搜索深度 ("basic" 或 "advanced")
            include_answer: 是否包含AI生成的答案
            include_raw_content: 是否包含原始内容

        Returns:
            WebSearchResponse
        """
        if not WEB_SEARCH_ENABLED:
            return WebSearchResponse(
                query=query,
                results=[],
                answer="",
                duration_ms=0,
                success=False,
                error="联网搜索功能已禁用"
            )

        start_time = time.time()

        try:
            client = self._get_client()

            # 执行搜索
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                include_raw_content=include_raw_content
            )

            # 解析结果
            results = []
            for item in response.get('results', []):
                results.append(WebSearchResult(
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    content=item.get('content', ''),
                    score=item.get('score', 0.0),
                    raw_content=item.get('raw_content', '')
                ))

            duration = (time.time() - start_time) * 1000

            return WebSearchResponse(
                query=query,
                results=results,
                answer=response.get('answer', ''),
                duration_ms=duration,
                success=True
            )

        except ValueError as e:
            duration = (time.time() - start_time) * 1000
            return WebSearchResponse(
                query=query,
                results=[],
                answer="",
                duration_ms=duration,
                success=False,
                error=str(e)
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return WebSearchResponse(
                query=query,
                results=[],
                answer="",
                duration_ms=duration,
                success=False,
                error=f"搜索失败: {str(e)}"
            )

    def search_with_context(
        self,
        query: str,
        context: str = "",
        max_results: int = 5
    ) -> WebSearchResponse:
        """
        带上下文的搜索（构造更好的查询）

        Args:
            query: 原始查询
            context: 上下文信息
            max_results: 最大结果数

        Returns:
            WebSearchResponse
        """
        # 如果有上下文，可以构造更精确的查询
        enhanced_query = query
        if context:
            # 简单拼接，可以根据需要优化
            enhanced_query = f"{query} {context}"

        return self.search(
            query=enhanced_query,
            max_results=max_results,
            search_depth="advanced"  # 使用高级搜索获取更好的结果
        )


def web_search(query: str, max_results: int = 5) -> WebSearchResponse:
    """
    联网搜索的便捷函数

    Args:
        query: 搜索查询
        max_results: 最大结果数

    Returns:
        WebSearchResponse
    """
    searcher = WebSearcher()
    return searcher.search(query, max_results)


def is_web_search_available() -> bool:
    """检查联网搜索是否可用"""
    if not WEB_SEARCH_ENABLED:
        return False
    if not TAVILY_API_KEY:
        return False
    return True


def get_web_search_results(query: str) -> List[Dict[str, Any]]:
    """
    获取搜索结果的简化接口

    Returns:
        结果字典列表 [{title, url, content, score}]
    """
    response = web_search(query)
    if not response.success:
        return []

    return [
        {
            'title': r.title,
            'url': r.url,
            'content': r.content,
            'score': r.score
        }
        for r in response.results
    ]
