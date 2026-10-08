"""MCP 工具服务（stdio）：也可单独运行 `python -m deep_research.tools.mcp_server` 接入其他 MCP 客户端。"""

from __future__ import annotations

import json
import logging
from typing import Callable

from mcp.server.mcpserver import MCPServer

from deep_research.config import load_settings
from deep_research.tools.search import DiskCache, fetch_page as _fetch_page, serpapi_search, wikipedia_search

for noisy in ("httpx", "huggingface_hub", "fastembed"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

settings = load_settings(require_llm=False)
cache = DiskCache(settings.data_dir / "cache")
server = MCPServer("deep-research-tools")
_knowledge_base = None


def _respond(fn: Callable[[], list[dict]]) -> str:
    try:
        return json.dumps(fn(), ensure_ascii=False)
    except Exception as exc:  # 工具错误以结构化结果返回给 Agent，由它决定换工具还是换关键词
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


def _kb():
    global _knowledge_base
    if _knowledge_base is None:
        from deep_research.knowledge_base import KnowledgeBase

        _knowledge_base = KnowledgeBase.from_settings(settings)
    return _knowledge_base


@server.tool()
def web_search(query: str, num_results: int = 5) -> str:
    """Google 网页搜索（SerpApi），返回标题、链接和摘要。适合新闻、论文、产品等最新或细节信息。"""
    return _respond(lambda: serpapi_search(query, settings.serpapi_key, cache, min(num_results, 8), settings.search_hl, settings.search_gl))


@server.tool()
def wiki_search(query: str, lang: str = "zh") -> str:
    """检索维基百科词条简介。适合人物、机构、历史事件、科学概念等百科类事实。lang 可选 zh 或 en，英文名词用 en 通常更准确。"""
    return _respond(lambda: wikipedia_search(query, cache, lang=lang))


@server.tool()
def fetch_page(url: str) -> str:
    """读取网页正文。当搜索摘要信息不足、需要核实细节时，用它读取某个结果链接的原文。"""
    return _respond(lambda: _fetch_page(url, cache))


@server.tool()
def kb_search(query: str, top_k: int = 4) -> str:
    """在本地知识库（knowledge_base/ 目录中的用户文档）中做向量检索，返回最相关的文档片段。"""
    return _respond(lambda: _kb().search(query, top_k=min(top_k, 8)))


if __name__ == "__main__":
    server.run()
