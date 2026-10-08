from deep_research.evidence import EvidenceStore
from deep_research.react_agent import ReActAgent

from .fakes import FakeLLM, FakeTools, scripted

WIKI_RESULT = [{"title": "AlphaGo", "url": "https://zh.wikipedia.org/wiki/AlphaGo", "content": "AlphaGo 以 4:1 战胜李世石。"}]


def test_tool_call_then_finish_registers_evidence():
    llm = FakeLLM(scripted(
        {"thought": "查百科", "action": "wiki_search", "action_input": {"query": "AlphaGo 李世石"}},
        {"thought": "够了", "action": "finish", "answer": "AlphaGo 以 4:1 获胜[S1]。"},
    ))
    tools = FakeTools({"wiki_search": WIKI_RESULT})
    evidence = EvidenceStore()
    result = ReActAgent(llm, tools).run("AlphaGo 与李世石的比分？", evidence)

    assert result.finished and result.citations == ["S1"]
    assert len(result.steps) == 1 and result.steps[0].source_ids == ["S1"]
    assert "[S1]" in result.steps[0].observation
    assert evidence.get("S1").url == WIKI_RESULT[0]["url"]


def test_unknown_tool_returns_error_observation():
    llm = FakeLLM(scripted(
        {"thought": "试试", "action": "google", "action_input": {"query": "x"}},
        {"thought": "改正", "action": "finish", "answer": "未能确认。"},
    ))
    result = ReActAgent(llm, FakeTools({"wiki_search": WIKI_RESULT})).run("问题", EvidenceStore())
    assert "不存在名为" in result.steps[0].observation


def test_duplicate_call_is_served_from_cache():
    call = {"thought": "查", "action": "wiki_search", "action_input": {"query": "AlphaGo"}}
    llm = FakeLLM(scripted(call, call, {"action": "finish", "answer": "完成[S1]。"}))
    tools = FakeTools({"wiki_search": WIKI_RESULT})
    result = ReActAgent(llm, tools).run("问题", EvidenceStore())
    assert len(tools.calls) == 1
    assert "完全相同" in result.steps[1].observation


def test_max_steps_forces_finish():
    step = {"thought": "继续查", "action": "wiki_search", "action_input": {"query": "q"}}
    llm = FakeLLM(scripted(
        {**step, "action_input": {"query": "q1"}},
        {**step, "action_input": {"query": "q2"}},
        {"action": "finish", "answer": "信息不足，未能确认。"},
    ))
    result = ReActAgent(llm, FakeTools({"wiki_search": WIKI_RESULT}), max_steps=2).run("问题", EvidenceStore())
    assert not result.finished and len(result.steps) == 2
    assert "已达到最大步数" in llm.calls[-1][1][-1]["content"]


def test_answer_without_evidence_is_pushed_back_once():
    llm = FakeLLM(scripted(
        {"thought": "我知道答案", "action": "finish", "answer": "4:1"},
        {"thought": "先检索", "action": "wiki_search", "action_input": {"query": "AlphaGo"}},
        {"thought": "有证据了", "action": "finish", "answer": "4:1[S1]。"},
    ))
    result = ReActAgent(llm, FakeTools({"wiki_search": WIKI_RESULT})).run("比分？", EvidenceStore())
    assert result.answer == "4:1[S1]。" and len(result.steps) == 1
    assert any("还没有检索任何来源" in m["content"] for m in llm.calls[-1][1])


def test_answer_from_context_evidence_is_allowed():
    llm = FakeLLM(scripted({"action": "finish", "answer": "4:1[S1]。"}))
    result = ReActAgent(llm, FakeTools({"wiki_search": WIKI_RESULT})).run("比分？", EvidenceStore(), context="[S1] 记忆：4:1")
    assert result.finished and result.steps == []


def test_tool_error_is_reported_to_agent():
    llm = FakeLLM(scripted(
        {"action": "wiki_search", "action_input": {"query": "x"}},
        {"action": "finish", "answer": "工具出错，未能确认。"},
    ))
    result = ReActAgent(llm, FakeTools({"wiki_search": {"error": "网络超时"}})).run("问题", EvidenceStore())
    assert "工具调用失败" in result.steps[0].observation and result.steps[0].source_ids == []
