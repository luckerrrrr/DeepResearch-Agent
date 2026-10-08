import json

from deep_research.memory import ConversationMemory
from deep_research.pipeline import DeepResearchAgent, PipelineConfig

from .fakes import FakeLLM, FakeTools

SEARCH = [{"title": "AlphaGo", "url": "https://example.com/alphago", "content": "AlphaGo 由 DeepMind 开发，2016 年以 4:1 战胜李世石。"}]


def make_handler():
    react_turn = {"n": 0}

    def handler(stage, messages):
        if stage == "plan":
            return {"standalone_question": "AlphaGo 的开发者和比分？", "rationale": "拆成两步", "sub_questions": ["谁开发了 AlphaGo？", "比分是多少？"]}
        if stage in ("execute", "reflect") and messages[0]["role"] == "system":
            react_turn["n"] += 1
            if react_turn["n"] % 2 == 1:
                return {"thought": "搜索", "action": "web_search", "action_input": {"query": f"AlphaGo {react_turn['n']}"}}
            return {"thought": "完成", "action": "finish", "answer": "DeepMind 开发，4:1 获胜[S1]。"}
        if stage == "write":
            return "## 结论\nAlphaGo 由 DeepMind 开发，以 4:1 战胜李世石[S1]。\n## 详细分析\n- 比赛在 2016 年举行[S1]。"
        if stage == "reflect":
            return {"score": 9, "issues": [], "follow_up_questions": [], "needs_revision": False}
        if stage == "verify":
            ids = [json.loads(l)["id"] for l in messages[-1]["content"].splitlines() if l.startswith('{"id"')]
            return {"results": [{"id": i, "label": "supported", "reason": "有依据"} for i in ids]}
        raise AssertionError(f"unexpected stage {stage}")

    return handler


def test_full_pipeline_produces_snapshots_and_references():
    llm = FakeLLM(make_handler())
    conversation = ConversationMemory()
    agent = DeepResearchAgent(llm, FakeTools({"web_search": SEARCH}), PipelineConfig(use_memory=False), conversation)
    result = agent.run("AlphaGo 是谁开发的，比分多少？")

    assert [s.stage for s in result.snapshots] == ["plan_solve", "reflection", "verified"]
    assert len(result.findings) == 2 and result.tool_calls == 2
    assert result.verification.stats()["supported"] == 2
    assert "[S1] AlphaGo — https://example.com/alphago" in result.references
    assert result.snapshots[-1].llm_calls == llm.tracker.total().calls
    assert len(conversation) == 1


def test_react_only_baseline_skips_planner():
    llm = FakeLLM(make_handler())
    config = PipelineConfig(use_planner=False, use_reflection=False, use_verification=False, use_memory=False)
    result = DeepResearchAgent(llm, FakeTools({"web_search": SEARCH}), config).run("AlphaGo 比分？")
    assert [s.stage for s in result.snapshots] == ["react"]
    assert all(stage != "plan" for stage, _ in llm.calls)
    assert result.report == "DeepMind 开发，4:1 获胜[S1]。"
