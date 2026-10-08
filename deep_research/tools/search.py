from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SERPAPI_URL = "https://serpapi.com/search.json"
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
# 维基百科 API 要求使用可识别的 User-Agent
WIKI_UA = "DeepResearchAgent/0.1 (educational research agent; python-requests)"


class DiskCache:
    """以请求参数的哈希为键的磁盘缓存：节省搜索配额，并让评估结果可复现。"""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, namespace: str, params: dict) -> Path:
        key = hashlib.sha1(json.dumps(params, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        return self.directory / namespace / f"{key}.json"

    def get(self, namespace: str, params: dict):
        path = self._path(namespace, params)
        return json.loads(path.read_text("utf-8")) if path.exists() else None

    def set(self, namespace: str, params: dict, value) -> None:
        path = self._path(namespace, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), "utf-8")


def serpapi_search(
    query: str, api_key: str | None, cache: DiskCache, num: int = 5, hl: str = "zh-cn", gl: str | None = None
) -> list[dict]:
    params = {"engine": "google", "q": query, "hl": hl, "num": num}
    if gl:
        params["gl"] = gl
    cached = cache.get("serpapi", params)
    if cached is not None:
        return cached
    if not api_key:
        raise RuntimeError("未配置 SERPAPI_API_KEY，web_search 不可用，请改用 wiki_search")

    data = _serpapi_async(params, api_key)
    error = data.get("error")
    if error and "hasn't returned any results" not in error:
        raise RuntimeError(f"SerpApi 错误：{error}")
    results = parse_serpapi(data, num)
    cache.set("serpapi", params, results)
    return results


def _serpapi_async(params: dict, api_key: str, timeout: float = 60) -> dict:
    """异步提交 + 轮询结果：避免同步模式长时间占用连接被网络中断；轮询归档接口不消耗搜索配额。"""
    data = requests.get(SERPAPI_URL, params={**params, "api_key": api_key, "async": "true"}, timeout=20).json()
    if data.get("error"):
        return data
    search_id = data["search_metadata"]["id"]
    deadline = time.monotonic() + timeout
    while data.get("search_metadata", {}).get("status") not in ("Success", "Error"):
        if time.monotonic() > deadline:
            raise TimeoutError("SerpApi 搜索超时，请稍后重试或改用 wiki_search")
        time.sleep(1.5)
        data = requests.get(f"https://serpapi.com/searches/{search_id}.json", params={"api_key": api_key}, timeout=20).json()
    return data


def parse_serpapi(data: dict, num: int) -> list[dict]:
    items: list[dict] = []
    box = data.get("answer_box") or {}
    answer = box.get("answer") or box.get("snippet") or "；".join(box.get("list") or [])
    if answer:
        items.append({"title": box.get("title") or "Google 精选答案", "url": box.get("link") or "", "content": answer})
    graph = data.get("knowledge_graph") or {}
    if graph.get("description"):
        url = (graph.get("source") or {}).get("link") or graph.get("website") or ""
        items.append({"title": graph.get("title") or "Google 知识图谱", "url": url, "content": graph["description"]})
    for r in (data.get("organic_results") or [])[:num]:
        items.append({"title": r.get("title", ""), "url": r.get("link", ""), "content": r.get("snippet", "")})
    return [i for i in items if i["content"]]


def wikipedia_search(query: str, cache: DiskCache, lang: str = "zh", limit: int = 3) -> list[dict]:
    lang = lang if lang in ("zh", "en") else "zh"
    params = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "inprop": "url",
        "redirects": 1,
    }
    if lang == "zh":
        params["variant"] = "zh-cn"
    cached = cache.get(f"wiki_{lang}", params)
    if cached is not None:
        return cached

    resp = requests.get(f"https://{lang}.wikipedia.org/w/api.php", params=params, headers={"User-Agent": WIKI_UA}, timeout=20)
    resp.raise_for_status()
    pages = sorted(resp.json().get("query", {}).get("pages", []), key=lambda p: p.get("index", 0))
    results = [
        {"title": p["title"], "url": p.get("fullurl", ""), "content": p["extract"][:2500]}
        for p in pages
        if p.get("extract")
    ]
    cache.set(f"wiki_{lang}", params, results)
    return results


def fetch_page(url: str, cache: DiskCache, max_chars: int = 6000) -> list[dict]:
    params = {"url": url, "max_chars": max_chars}
    cached = cache.get("fetch", params)
    if cached is not None:
        return cached

    resp = requests.get(url, headers={"User-Agent": BROWSER_UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}, timeout=20)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        raise RuntimeError(f"暂不支持读取该类型的内容：{content_type or '未知'}")
    if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
        resp.encoding = resp.apparent_encoding

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "form", "svg"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else url
    main = soup.find("article") or soup.find("main") or soup.body or soup
    text = re.sub(r"\n\s*\n+", "\n", main.get_text("\n", strip=True))
    if not text:
        raise RuntimeError("页面没有可读取的正文")
    results = [{"title": title, "url": url, "content": text[:max_chars]}]
    cache.set("fetch", params, results)
    return results
