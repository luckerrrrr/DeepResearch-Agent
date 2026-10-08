from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass

CITATION_RE = re.compile(r"\[(S\d+(?:\s*[,，、]\s*S\d+)*)\]")


def extract_citations(text: str) -> list[str]:
    """按出现顺序提取 [S1] / [S1, S2] 形式的来源编号（去重）。"""
    ids: list[str] = []
    for group in CITATION_RE.findall(text):
        for sid in re.split(r"\s*[,，、]\s*", group):
            if sid not in ids:
                ids.append(sid)
    return ids


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text).strip()


@dataclass
class Source:
    id: str
    tool: str
    title: str
    url: str
    content: str
    query: str = ""

    def brief(self, max_chars: int = 600) -> str:
        text = self.content if len(self.content) <= max_chars else self.content[:max_chars] + "…"
        return f"[{self.id}] {self.title} | {self.url}\n{text}"

    def to_dict(self) -> dict:
        return asdict(self)


class EvidenceStore:
    """一次研究过程中收集到的全部证据。每条证据分配唯一编号 S1、S2…，供报告引用和溯源验证。"""

    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}
        self._by_key: dict[str, str] = {}
        self._lock = threading.Lock()

    def add(self, tool: str, title: str, url: str, content: str, query: str = "") -> Source:
        key = url.strip() or f"{tool}:{title}:{content[:80]}"
        with self._lock:
            if key in self._by_key:
                existing = self._sources[self._by_key[key]]
                # 同一 URL 先出现搜索摘要、后被 fetch_page 读取全文时，升级为更完整的内容
                if len(content) > len(existing.content):
                    existing.content = content
                return existing
            sid = f"S{len(self._sources) + 1}"
            source = Source(sid, tool, title or url or sid, url, content, query)
            self._sources[sid] = source
            self._by_key[key] = sid
            return source

    def get(self, sid: str) -> Source | None:
        return self._sources.get(sid)

    def subset(self, ids: list[str]) -> list[Source]:
        return [self._sources[i] for i in ids if i in self._sources]

    def all(self) -> list[Source]:
        return list(self._sources.values())

    def __contains__(self, sid: str) -> bool:
        return sid in self._sources

    def __len__(self) -> int:
        return len(self._sources)


def render_references(report: str, evidence: EvidenceStore) -> str:
    sources = evidence.subset(extract_citations(report))
    if not sources:
        return ""
    lines = ["", "## 参考来源"]
    lines += [f"- [{s.id}] {s.title} — {s.url}" if s.url else f"- [{s.id}] {s.title}" for s in sources]
    return "\n".join(lines)
