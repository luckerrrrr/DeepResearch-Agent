from __future__ import annotations

from collections import deque
from datetime import datetime

from .config import Settings
from .evidence import strip_citations
from .vector_store import VectorStore, build_embedder


class ConversationMemory:
    """短期记忆：保留最近几轮问答，供规划器做指代消解（例如追问“他后来去了哪里？”）。"""

    def __init__(self, max_turns: int = 4, answer_chars: int = 600) -> None:
        self.turns: deque[tuple[str, str]] = deque(maxlen=max_turns)
        self.answer_chars = answer_chars

    def add(self, question: str, answer: str) -> None:
        answer = strip_citations(answer)
        if len(answer) > self.answer_chars:
            answer = answer[: self.answer_chars] + "…"
        self.turns.append((question, answer))

    def render(self) -> str:
        return "\n".join(f"用户：{q}\n助手：{a}" for q, a in self.turns)

    def clear(self) -> None:
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)


class LongTermMemory:
    """长期记忆：把通过溯源验证的结论向量化持久保存，新问题到来时按语义相似度召回，实现跨会话复用。"""

    def __init__(self, settings: Settings | None = None, store: VectorStore | None = None, min_score: float = 0.6) -> None:
        self._settings = settings
        self._store = store
        self.min_score = min_score

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(self._settings.data_dir / "memory", build_embedder(self._settings))
        return self._store

    def recall(self, query: str, top_k: int = 3) -> list[dict]:
        return [{**item, "score": round(score, 3)} for score, item in self.store.search(query, top_k, self.min_score)]

    def remember(self, question: str, claims: list[tuple[str, list[str]]]) -> int:
        known = {item["text"] for item in self.store.items}
        fresh = [(text, urls) for text, urls in claims if text and text not in known]
        now = datetime.now().isoformat(timespec="seconds")
        self.store.add(
            [text for text, _ in fresh],
            [{"question": question, "urls": urls, "created_at": now} for _, urls in fresh],
        )
        return len(fresh)

    def clear(self) -> None:
        self.store.clear()

    def __len__(self) -> int:
        return len(self.store)
