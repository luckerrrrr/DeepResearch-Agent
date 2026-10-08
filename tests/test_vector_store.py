from deep_research.knowledge_base import chunk_text
from deep_research.memory import LongTermMemory
from deep_research.vector_store import VectorStore

from .fakes import FakeEmbedder


class OtherEmbedder(FakeEmbedder):
    name = "another-embedder"


def test_search_ranks_most_similar_first(tmp_path):
    store = VectorStore(tmp_path, FakeEmbedder())
    store.add(["苹果香蕉橙子", "汽车火车飞机", "苹果和香蕉都是水果"], [{"k": 1}, {"k": 2}, {"k": 3}])
    top_score, top_item = store.search("香蕉苹果", top_k=1)[0]
    assert top_item["k"] in (1, 3) and top_score > 0.5


def test_store_persists_and_invalidates_on_embedder_change(tmp_path):
    VectorStore(tmp_path, FakeEmbedder()).add(["第一条", "第二条"])
    assert len(VectorStore(tmp_path, FakeEmbedder())) == 2
    assert len(VectorStore(tmp_path, OtherEmbedder())) == 0


def test_long_term_memory_dedupes_and_recalls(tmp_path):
    memory = LongTermMemory(store=VectorStore(tmp_path, FakeEmbedder()), min_score=0.0)
    claims = [("AlphaGo 以 4:1 战胜李世石", ["https://a.com"])]
    assert memory.remember("AlphaGo 比分", claims) == 1
    assert memory.remember("AlphaGo 比分", claims) == 0
    hit = memory.recall("AlphaGo 战胜李世石的比分", top_k=1)[0]
    assert hit["urls"] == ["https://a.com"] and hit["question"] == "AlphaGo 比分"


def test_chunk_text_keeps_heading_and_splits_long_paragraphs():
    text = "# 标题A\n\n短段落一。\n\n" + "长" * 900
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    assert all(c.startswith("【标题A】") for c in chunks)
    assert len(chunks) >= 3
