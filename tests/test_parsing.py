import pytest

from deep_research.evidence import EvidenceStore, extract_citations, render_references, strip_citations
from deep_research.llm import parse_json_object


def test_parse_plain_json():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_fenced_json_with_surrounding_text():
    text = '好的，结果如下：\n```json\n{"action": "finish", "answer": "42"}\n```\n以上。'
    assert parse_json_object(text)["answer"] == "42"


def test_parse_json_embedded_in_text():
    assert parse_json_object('思考完毕 {"x": [1, 2]} 结束')["x"] == [1, 2]


def test_parse_invalid_json_raises():
    with pytest.raises(ValueError):
        parse_json_object("这里没有 JSON")


def test_extract_citations_keeps_order_and_dedupes():
    text = "A 发生于 2017 年[S3][S1]，B 也是[S1, S2]，C[S4，S5、S6]。"
    assert extract_citations(text) == ["S3", "S1", "S2", "S4", "S5", "S6"]
    assert strip_citations("结论[S1]。") == "结论。"


def test_evidence_store_dedupes_by_url_and_upgrades_content():
    store = EvidenceStore()
    first = store.add("web_search", "标题", "https://a.com", "摘要")
    again = store.add("fetch_page", "标题", "https://a.com", "更完整的正文内容")
    other = store.add("web_search", "另一个", "https://b.com", "摘要")
    assert first.id == again.id == "S1" and other.id == "S2"
    assert store.get("S1").content == "更完整的正文内容"


def test_render_references_lists_only_cited_sources():
    store = EvidenceStore()
    store.add("web_search", "甲", "https://a.com", "x")
    store.add("web_search", "乙", "https://b.com", "y")
    refs = render_references("结论[S2]。", store)
    assert "[S2] 乙" in refs and "[S1]" not in refs
