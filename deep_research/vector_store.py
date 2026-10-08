from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class LocalEmbedder:
    """本地 ONNX 向量模型（fastembed），免费、离线可用，默认 bge-small-zh。"""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.passage_embed(texts)), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # bge 系列对查询和文档采用不同的编码方式（非对称检索），query_embed 会自动加查询指令前缀
        return np.array(list(self._model.query_embed(text))[0], dtype=np.float32)


class OpenAIEmbedder:
    def __init__(self, model_name: str, api_key: str, base_url: str) -> None:
        from openai import OpenAI

        self.name = model_name
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.name, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(settings.embedding_model, settings.llm_api_key, settings.llm_base_url)
    return LocalEmbedder(settings.embedding_model, settings.data_dir / "models")


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class VectorStore:
    """NumPy 向量库：归一化后矩阵乘法算余弦相似度做精确 Top-K；万级以上数据可换 FAISS/Milvus 等 ANN 索引。"""

    def __init__(self, directory: Path, embedder: Embedder) -> None:
        self.directory = directory
        self.embedder = embedder
        self.items: list[dict] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.meta: dict = {}
        self._load()

    @property
    def _files(self) -> tuple[Path, Path, Path]:
        return self.directory / "meta.json", self.directory / "items.jsonl", self.directory / "vectors.npy"

    def _load(self) -> None:
        meta_path, items_path, vectors_path = self._files
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text("utf-8"))
        if meta.get("embedder") != self.embedder.name:
            return  # 换了向量模型，旧索引作废
        self.meta = meta
        self.items = [json.loads(line) for line in items_path.read_text("utf-8").splitlines() if line]
        self.vectors = np.load(vectors_path)

    def save(self) -> None:
        meta_path, items_path, vectors_path = self._files
        self.directory.mkdir(parents=True, exist_ok=True)
        self.meta["embedder"] = self.embedder.name
        meta_path.write_text(json.dumps(self.meta, ensure_ascii=False), "utf-8")
        items_path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in self.items), "utf-8")
        np.save(vectors_path, self.vectors)

    def add(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        if not texts:
            return
        metadatas = metadatas or [{} for _ in texts]
        vectors = _normalize(self.embedder.embed_documents(texts))
        start = len(self.items)
        self.items += [{"id": start + i, "text": t, **m} for i, (t, m) in enumerate(zip(texts, metadatas))]
        self.vectors = vectors if self.vectors.size == 0 else np.vstack([self.vectors, vectors])
        self.save()

    def search(self, query: str, top_k: int = 4, min_score: float | None = None) -> list[tuple[float, dict]]:
        if not self.items:
            return []
        query_vec = _normalize(self.embedder.embed_query(query))
        scores = self.vectors @ query_vec
        order = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), self.items[i]) for i in order if min_score is None or scores[i] >= min_score]

    def clear(self) -> None:
        self.items, self.vectors, self.meta = [], np.zeros((0, 0), dtype=np.float32), {}
        self.save()

    def __len__(self) -> int:
        return len(self.items)
