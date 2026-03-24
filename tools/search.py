"""Tavily 搜索工具"""
import os
import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "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")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class TavilySearchTool:
    """Tavily 搜索工具，用于实时网络搜索获取信息"""

    def __init__(self, api_key: str = TAVILY_API_KEY):
        self.api_key = api_key
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_domains: Optional[list] = None,
        exclude_domains: Optional[list] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        include_images: bool = False
    ) -> Dict[str, Any]:
        """
        使用 Tavily API 进行搜索
        """
        if not self.api_key:
            logger.warning("Tavily API key not configured, returning mock results")
            return self._mock_search(query, max_results)

        if not self.client:
            self.client = httpx.AsyncClient(timeout=30.0)

        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "include_images": include_images
            }

            if include_domains:
                payload["include_domains"] = include_domains
            if exclude_domains:
                payload["exclude_domains"] = exclude_domains

            response = await self.client.post(
                TAVILY_SEARCH_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()

            logger.info(f"[Tavily] Search completed, found {len(result.get('results', []))} results")
            return result

        except httpx.TimeoutException:
            logger.error("[Tavily] Search timeout")
            return self._mock_search(query, max_results)
        except Exception as e:
            logger.error(f"[Tavily] Search failed: {e}")
            return self._mock_search(query, max_results)

    async def search_finance(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """金融相关搜索"""
        finance_domains = [
            "investing.com", "finance.yahoo.com", "bloomberg.com",
            "reuters.com", "cnbc.com", "wallstreetjournal.com",
            "seekingalpha.com", "marketwatch.com", "ft.com"
        ]
        return await self.search(
            query,
            max_results=max_results,
            include_domains=finance_domains,
            search_depth="advanced"
        )

    def format_results(self, results: Dict[str, Any]) -> str:
        """格式化搜索结果为可读文本"""
        if not results or "results" not in results:
            return "未找到相关搜索结果"

        formatted = []

        if results.get("answer"):
            formatted.append(f"📌 AI 摘要:\n{results['answer']}\n")

        for i, result in enumerate(results.get("results", [])[:10], 1):
            title = result.get("title", "无标题")
            url = result.get("url", "")
            content = result.get("content", "")
            score = result.get("score", 0)

            formatted.append(f"{i}. **{title}**")
            formatted.append(f"   URL: {url}")
            formatted.append(f"   相关性：{score:.2f}")
            if content:
                formatted.append(f"   摘要：{content[:200]}...")
            formatted.append("")

        return "\n".join(formatted)

    def _mock_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """模拟搜索结果（用于无 API 配置时）"""
        logger.info(f"[Mock Search] Query: {query}")
        return {
            "query": query,
            "results": [
                {
                    "title": f"关于 '{query}' 的最新分析 - 金融新闻",
                    "url": "https://example.com/finance-news",
                    "content": f"这是关于 {query} 的最新市场分析和报道...",
                    "score": 0.95
                }
            ],
            "answer": None,
            "images": []
        }
