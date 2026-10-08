from __future__ import annotations

import json

import streamlit as st

from deep_research.config import load_settings
from deep_research.events import describe_event
from deep_research.llm import LLMClient
from deep_research.memory import ConversationMemory, LongTermMemory
from deep_research.pipeline import DeepResearchAgent, PipelineConfig, ResearchResult
from deep_research.tools.mcp_client import MCPToolClient

st.set_page_config(page_title="DeepResearch-Agent", layout="wide")

LABELS = {"supported": "支持", "partial": "部分支持", "unsupported": "无依据", "not_claim": "非事实陈述"}
STAGES = {"react": "ReAct", "plan_solve": "规划+执行+撰写", "reflection": "Reflection 后", "verified": "溯源验证后"}
EXAMPLES = [
    "2024 年诺贝尔物理学奖授予了哪两位科学家？其中哪一位还获得过图灵奖？",
    "DeepSeek-R1 训练中使用的强化学习算法叫什么？它最早在哪篇论文中提出？",
    "MCP 和 Function Calling 有什么区别？",
]


@st.cache_resource(show_spinner="正在启动 MCP 工具服务…")
def runtime():
    settings = load_settings()
    return settings, LLMClient.from_settings(settings), MCPToolClient(), LongTermMemory(settings)


def render_trace(result: ResearchResult) -> None:
    if result.plan_rationale:
        st.caption(f"规划思路：{result.plan_rationale}")
    tasks = [("子问题", f) for f in result.findings] + [("补充检索", f) for f in result.followups]
    for i, (kind, finding) in enumerate(tasks, 1):
        with st.expander(f"{kind} {i}：{finding.question}"):
            for n, step in enumerate(finding.steps, 1):
                st.markdown(f"**第 {n} 步 · 思考**：{step.thought}")
                st.code(f"{step.action}({json.dumps(step.action_input, ensure_ascii=False)})", language="text")
                st.caption("观察：" + ("获得来源 " + "、".join(step.source_ids) if step.source_ids else step.observation[:150]))
            st.markdown(f"**结论**：{finding.answer.removeprefix('结论：')}")


def render_verification(result: ResearchResult) -> None:
    if not result.verification:
        st.info("本次未启用溯源验证")
        return
    stats = result.verification.stats()
    cols = st.columns(4)
    cols[0].metric("可核查陈述", stats["claims"])
    cols[1].metric("有来源支撑", stats["supported"])
    cols[2].metric("部分支撑", stats["partial"])
    cols[3].metric("无依据（已处理）", stats["unsupported"])
    st.dataframe(
        [
            {"判定": LABELS.get(c.label, c.label), "陈述": c.text.replace("**", ""), "引用": "、".join(c.citations), "理由": c.reason}
            for c in result.verification.checks
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_result(result: ResearchResult) -> None:
    st.markdown(result.full_report)
    stats = result.verification.stats() if result.verification else None
    cols = st.columns(5)
    cols[0].metric("耗时", f"{result.seconds:.0f}s")
    cols[1].metric("LLM 调用", result.usage["total"]["calls"])
    cols[2].metric("Token", f"{result.usage['total']['total_tokens'] / 1000:.1f}k")
    cols[3].metric("工具调用", result.tool_calls)
    cols[4].metric("溯源支持率", f"{stats['support_rate']:.0%}" if stats else "未启用")

    trace, reflection, verification, sources, cost = st.tabs(["执行轨迹", "反思", "溯源验证", "全部来源", "开销明细"])
    with trace:
        render_trace(result)
    with reflection:
        if result.critique:
            st.markdown(f"**评分**：{result.critique['score']}/10")
            for issue in result.critique["issues"] or ["无明显问题"]:
                st.markdown(f"- {issue}")
            if result.critique["follow_up_questions"]:
                st.markdown("**补充检索**：" + "；".join(result.critique["follow_up_questions"]))
        else:
            st.info("本次未启用 Reflection")
    with verification:
        render_verification(result)
    with sources:
        for s in result.evidence.all():
            with st.expander(f"[{s.id}] {s.title}（{s.tool}）"):
                st.caption(s.url)
                st.text(s.content[:1500])
    with cost:
        st.dataframe(
            [{"阶段": STAGES.get(s.stage, s.stage), "累计耗时(s)": s.seconds, "累计 LLM 调用": s.llm_calls, "累计 Token": s.total_tokens, "累计工具调用": s.tool_calls} for s in result.snapshots],
            use_container_width=True,
            hide_index=True,
        )


settings, llm, tools, long_term = runtime()
state = st.session_state
state.setdefault("conversation", ConversationMemory())
state.setdefault("history", [])

with st.sidebar:
    st.header("流水线配置")
    use_planner = st.toggle("Plan-and-Solve 任务规划", value=True)
    use_reflection = st.toggle("Reflection 自我反思", value=True)
    use_verification = st.toggle("溯源验证", value=True)
    flag_mode = st.toggle("只标记无依据陈述，不删除", value=False, disabled=not use_verification)
    use_memory = st.toggle("长期记忆", value=True)
    steps = st.slider("每个子问题的 ReAct 最大步数", 2, 8, 4)
    st.divider()
    st.caption(f"模型：{settings.llm_model}")
    st.caption("MCP 工具：" + "、".join(t.name for t in tools.tools))
    st.caption(f"长期记忆：{len(long_term)} 条已验证结论")
    if st.button("清空对话（短期记忆）", use_container_width=True):
        state.conversation.clear()
        state.history = []
        st.rerun()
    if st.button("清空长期记忆", use_container_width=True):
        long_term.clear()
        st.rerun()

st.title("DeepResearch-Agent")
st.caption("Plan-and-Solve 规划 · ReAct 检索 · Reflection 反思 · 溯源验证 · 长短期记忆 · MCP 工具协议")

for past_question, past_result in state.history:
    with st.chat_message("user"):
        st.markdown(past_question)
    with st.chat_message("assistant"):
        render_result(past_result)

if not state.history:
    st.markdown("**试试这些问题：**")
    for example in EXAMPLES:
        if st.button(example):
            state.pending = example
            st.rerun()

question = st.chat_input("输入研究问题，支持多轮追问…") or state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.status("正在研究…", expanded=True)

        def on_event(kind: str, data: dict) -> None:
            text = describe_event(kind, data)
            if text:
                status.text(text)

        config = PipelineConfig(
            use_planner=use_planner,
            use_reflection=use_reflection,
            use_verification=use_verification,
            use_memory=use_memory,
            react_max_steps=steps,
            verification_mode="flag" if flag_mode else "remove",
        )
        agent = DeepResearchAgent(llm, tools, config, state.conversation, long_term if use_memory else None, on_event)
        try:
            result = agent.run(question)
        except Exception as exc:
            status.update(label="研究失败", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
        else:
            status.update(label=f"研究完成，用时 {result.seconds:.0f}s", state="complete", expanded=False)
            render_result(result)
            state.history.append((question, result))
