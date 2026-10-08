import json

from deep_research.evidence import EvidenceStore
from deep_research.verifier import Verifier, _parse

from .fakes import FakeLLM


def judge_by_keyword(stage, messages):
    """“火星”相关陈述判为无依据，其余判为有依据。"""
    results = []
    for line in messages[-1]["content"].splitlines():
        if line.startswith('{"id"'):
            claim = json.loads(line)
            label = "unsupported" if "火星" in claim["text"] else "supported"
            results.append({"id": claim["id"], "label": label, "reason": "测试"})
    return {"results": results}


def make_evidence() -> EvidenceStore:
    store = EvidenceStore()
    store.add("wiki_search", "AlphaGo", "https://example.com/alphago", "AlphaGo 在 2016 年以 4:1 战胜李世石。")
    return store


REPORT = """## 结论
AlphaGo 在 2016 年以 4:1 战胜了李世石[S1]。AlphaGo 后来登上了火星[S1]。

## 详细分析
- 这场比赛在首尔举行，共进行了五局[S1]。
- AlphaGo 曾经在火星上举办围棋比赛[S1]。
- 编造来源的说法也会被识别出来[S9]。"""


def test_sentence_split_keeps_citations_attached():
    line = _parse("甲事件发生于 2017 年。[S1] 乙事件发生于 2018 年[S2]。数值是 22.5 万")[0]
    assert line.pieces == ["甲事件发生于 2017 年。[S1]", " 乙事件发生于 2018 年[S2]。", "数值是 22.5 万"]


def test_verify_removes_unsupported_and_fabricated_claims():
    result = Verifier(FakeLLM(judge_by_keyword)).verify(REPORT, make_evidence())

    assert "火星" not in result.report and "[S9]" not in result.report
    assert "AlphaGo 在 2016 年以 4:1 战胜了李世石[S1]。" in result.report
    assert "## 详细分析" in result.report
    assert "- 这场比赛在首尔举行" in result.report

    stats = result.stats()
    assert stats["unsupported"] == 3 and stats["supported"] == 2 and stats["removed"] == 3
    assert stats["fabricated_citations"] == 1
    fabricated = [c for c in result.checks if "[S9]" in c.text][0]
    assert "编造引用" in fabricated.reason


def test_flag_mode_keeps_text_but_marks_it():
    result = Verifier(FakeLLM(judge_by_keyword), mode="flag").verify(REPORT, make_evidence())
    assert "火星" in result.report and "（未经来源证实）" in result.report
    assert result.removed == []


def test_supported_claims_for_memory_exclude_citation_markers():
    evidence = make_evidence()
    result = Verifier(FakeLLM(judge_by_keyword)).verify(REPORT, evidence)
    claims = result.supported_claims(evidence)
    assert claims and all("[S" not in text for text, _ in claims)
    assert claims[0][1] == ["https://example.com/alphago"]
