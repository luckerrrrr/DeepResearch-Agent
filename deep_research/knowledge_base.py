from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .config import Settings, load_settings
from .vector_store import Embedder, VectorStore, build_embedder

SUPPORTED_SUFFIXES = {".md", ".txt"}


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """按段落切分并合并到 chunk_size 左右；每块带上所属的 Markdown 标题，避免片段脱离上下文。"""
    chunks: list[str] = []
    heading = ""
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(f"【{heading}】{buffer.strip()}" if heading else buffer.strip())
        buffer = ""

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            flush()
            heading = para.lstrip("#").strip().splitlines()[0]
            rest = "\n".join(para.splitlines()[1:]).strip()
            if not rest:
                continue
            para = rest
        while len(para) > chunk_size:
            flush()
            chunks.append(f"【{heading}】{para[:chunk_size]}" if heading else para[:chunk_size])
            para = para[chunk_size - overlap :]
        if len(buffer) + len(para) > chunk_size:
            flush()
        buffer = f"{buffer}\n{para}" if buffer else para
    flush()
    return chunks


class KnowledgeBase:
    """本地知识库：把 knowledge_base/ 下的文档切块、向量化，文件有变化时自动重建索引。"""

    def __init__(self, kb_dir: Path, index_dir: Path, embedder: Embedder) -> None:
        self.kb_dir = kb_dir
        self.store = VectorStore(index_dir, embedder)

    @classmethod
    def from_settings(cls, settings: Settings) -> "KnowledgeBase":
        return cls(settings.kb_dir, settings.data_dir / "kb_index", build_embedder(settings))

    def _files(self) -> list[Path]:
        if not self.kb_dir.exists():
            return []
        return sorted(p for p in self.kb_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file())

    def _fingerprint(self) -> str:
        digest = hashlib.sha1()
        for path in self._files():
            stat = path.stat()
            digest.update(f"{path.relative_to(self.kb_dir)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return digest.hexdigest()

    def ensure_index(self) -> None:
        fingerprint = self._fingerprint()
        if self.store.meta.get("fingerprint") == fingerprint:
            return
        self.store.clear()
        texts, metas = [], []
        for path in self._files():
            for i, chunk in enumerate(chunk_text(path.read_text("utf-8"))):
                texts.append(chunk)
                metas.append({"doc": path.name, "chunk": i})
        self.store.meta["fingerprint"] = fingerprint
        self.store.add(texts, metas)
        self.store.save()

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        self.ensure_index()
        return [
            {"title": f"知识库：{item['doc']}", "url": f"kb://{item['doc']}#{item['chunk']}", "content": item["text"], "score": round(score, 3)}
            for score, item in self.store.search(query, top_k=top_k)
        ]


if __name__ == "__main__":
    kb = KnowledgeBase.from_settings(load_settings(require_llm=False))
    kb.ensure_index()
    print(f"知识库索引已就绪：{len(kb.store)} 个文本块，来自 {len(kb._files())} 个文档")
