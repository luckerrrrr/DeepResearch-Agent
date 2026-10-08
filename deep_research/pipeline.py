from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

from .evidence import EvidenceStore, render_references
from .llm import LLMClient, UsageTracker
from .memory import ConversationMemory, LongTermMemory
from .planner import Planner
from .react_agent import ReActAgent, ReActResult
from .reflection import Reflector
from .tools.mcp_client import ToolClient
from .verifier import VerificationResult, Verifier
from .writer import ReportWriter

EventHandler = Callable[[str, dict], None]


@dataclass
class PipelineConfig:
    use_planner: bool = True
    use_reflection: bool = True
    use_verification: bool = True
    use_memory: bool = True
    max_sub_questions: int = 4
    react_max_steps: int = 4
    baseline_max_steps: int = 8
    max_followups: int = 2
    verification_mode: str = "remove"


@dataclass
class Snapshot:
    """每个阶段结束时的报告快照与累计开销，用于在一次运行中完成分阶段消融对比。"""

    stage: str
    report: str
    seconds: float
    llm_calls: int
    total_tokens: int
    tool_calls: int


@dataclass
class ResearchResult:
    question: str
    standalone_question: str
    plan_rationale: str
    findings: list[ReActResult]
    followups: list[ReActResult]
    critique: dict | None
    verification: VerificationResult | None
    report: str
    references: str
    evidence: EvidenceStore
    snapshots: list[Snapshot]
    usage: dict
    seconds: float
    memories_recalled: int = 0
    memories_saved: int = 0

    @property
    def full_report(self) -> str:
        return f"{self.report}\n{self.references}" if self.references else self.report

    @property
    def tool_calls(self) -> int:
        return sum(len(f.steps) for f in self.findings + self.followups)

    def snapshot(self, stage: str) -> Snapshot | None:
        return next((s for s in self.snapshots if s.stage == stage), None)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "standalone_question": self.standalone_question,
            "plan_rationale": self.plan_rationale,
            "findings": [f.to_dict() for f in self.findings],
            "followups": [f.to_dict() for f in self.followups],
            "critique": self.critique,
            "verification": self.verification.to_dict() if self.verification else None,
            "report": self.full_report,
            "sources": [s.to_dict() for s in self.evidence.all()],
            "snapshots": [asdict(s) for s in self.snapshots],
            "usage": self.usage,
            "seconds": self.seconds,
            "tool_calls": self.tool_calls,
        }


def _context(memory_brief: str, findings: list[ReActResult]) -> str:
    parts = [memory_brief] if memory_brief else []
    parts += [f"子问题：{f.question}\n结论：{f.answer}" for f in findings]
    return "\n\n".join(parts)


class DeepResearchAgent:
    """研究流水线：记忆召回 → Plan-and-Solve 规划 → ReAct 逐个求解 → 撰写 → Reflection → 溯源验证 → 写入记忆。"""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolClient,
        config: PipelineConfig | None = None,
        conversation: ConversationMemory | None = None,
        long_term: LongTermMemory | None = None,
        on_event: EventHandler | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or PipelineConfig()
        self.conversation = conversation
        self.long_term = long_term
        self.on_event = on_event or (lambda kind, data: None)
        self.planner = Planner(llm, self.config.max_sub_questions)
        self.react = ReActAgent(llm, tools, self.config.react_max_steps, on_event=self.on_event)
        self.writer = ReportWriter(llm)
        self.reflector = Reflector(llm, self.config.max_followups)
        self.verifier = Verifier(llm, mode=self.config.verification_mode)

    def run(self, question: str) -> ResearchResult:
        cfg, emit = self.config, self.on_event
        self.llm.tracker = UsageTracker()
        start = time.perf_counter()
        evidence = EvidenceStore()
        findings: list[ReActResult] = []
        followups: list[ReActResult] = []
        snapshots: list[Snapshot] = []

        def snap(stage: str, report: str) -> None:
            total = self.llm.tracker.total()
            tool_calls = sum(len(f.steps) for f in findings + followups)
            snapshots.append(Snapshot(stage, report, round(time.perf_counter() - start, 2), total.calls, total.total_tokens, tool_calls))

        use_memory = cfg.use_memory and self.long_term is not None
        memory_brief, recalled = "", 0
        if use_memory:
            items = self.long_term.recall(question)
            sources = [
                evidence.add(
                    "memory",
                    f"历史研究记忆：{item['question'][:40]}",
                    f"memory://{item['id']}",
                    item["text"] + ("\n原始来源：" + "；".join(item["urls"]) if item.get("urls") else ""),
                    question,
                )
                for item in items
            ]
            memory_brief = "\n\n".join(s.brief(400) for s in sources)
            recalled = len(sources)
            if recalled:
                emit("memory_recall", {"count": recalled})

        history = self.conversation.render() if self.conversation else ""
        rationale = ""
        if cfg.use_planner:
            plan = self.planner.plan(question, history, memory_brief)
            standalone, rationale = plan.standalone_question, plan.rationale
            emit("plan", {"standalone_question": standalone, "sub_questions": plan.sub_questions, "rationale": rationale})
            for i, sub_question in enumerate(plan.sub_questions, 1):
                emit("subtask_start", {"index": i, "total": len(plan.sub_questions), "question": sub_question})
                findings.append(self.react.run(sub_question, evidence, _context(memory_brief, findings)))
            emit("writing", {})
            report = self.writer.write(standalone, findings, evidence)
            snap("plan_solve", report)
        else:
            standalone = question
            context = "\n\n".join(p for p in (f"对话历史：\n{history}" if history else "", memory_brief) if p)
            findings.append(self.react.run(question, evidence, context, max_steps=cfg.baseline_max_steps))
            report = findings[0].answer
            snap("react", report)

        critique = None
        if cfg.use_reflection:
            critique = self.reflector.critique(standalone, report, [f.question for f in findings])
            emit("critique", critique)
            for follow_up in critique["follow_up_questions"]:
                emit("followup_start", {"question": follow_up})
                followups.append(self.react.run(follow_up, evidence, _context(memory_brief, findings + followups), stage="reflect"))
            if critique["needs_revision"] or followups:
                report = self.reflector.revise(standalone, report, critique, followups, evidence)
                emit("revised", {})
            snap("reflection", report)

        verification, saved = None, 0
        if cfg.use_verification:
            emit("verifying", {})
            verification = self.verifier.verify(report, evidence)
            report = verification.report
            emit("verification", verification.stats())
            snap("verified", report)
            if use_memory:
                saved = self.long_term.remember(standalone, verification.supported_claims(evidence))
                emit("memory_saved", {"count": saved})

        if self.conversation is not None:
            self.conversation.add(question, report)

        seconds = round(time.perf_counter() - start, 2)
        total = self.llm.tracker.total()
        emit("done", {"seconds": seconds, "llm_calls": total.calls, "total_tokens": total.total_tokens})
        return ResearchResult(
            question=question,
            standalone_question=standalone,
            plan_rationale=rationale,
            findings=findings,
            followups=followups,
            critique=critique,
            verification=verification,
            report=report,
            references=render_references(report, evidence),
            evidence=evidence,
            snapshots=snapshots,
            usage={"total": total.to_dict(), "stages": self.llm.tracker.to_dict()},
            seconds=seconds,
            memories_recalled=recalled,
            memories_saved=saved,
        )
