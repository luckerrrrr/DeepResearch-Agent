from __future__ import annotations

from .evidence import EvidenceStore
from .llm import LLMClient
from .prompts import CRITIQUE, REVISE, render
from .react_agent import ReActResult
from .writer import cited_ids, format_findings, format_sources


class Reflector:
    """Reflection：审稿人视角批判报告 → 针对信息缺口补充检索 → 修订报告。"""

    def __init__(self, llm: LLMClient, max_followups: int = 2) -> None:
        self.llm = llm
        self.max_followups = max_followups

    def critique(self, question: str, report: str, sub_questions: list[str]) -> dict:
        plan = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_questions, 1)) or "（无）"
        prompt = render(CRITIQUE, question=question, plan=plan, report=report, max_followups=self.max_followups)
        data = self.llm.chat_json([{"role": "user", "content": prompt}], stage="reflect")
        try:
            score = int(data.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        followups = [q.strip() for q in data.get("follow_up_questions") or [] if isinstance(q, str) and q.strip()]
        return {
            "score": score,
            "issues": [str(i) for i in data.get("issues") or [] if str(i).strip()],
            "follow_up_questions": followups[: self.max_followups],
            "needs_revision": bool(data.get("needs_revision")),
        }

    def revise(
        self, question: str, report: str, critique: dict, followups: list[ReActResult], evidence: EvidenceStore
    ) -> str:
        issues = "\n".join(f"- {i}" for i in critique["issues"]) or "（无）"
        prompt = render(
            REVISE,
            question=question,
            report=report,
            issues=issues,
            findings=format_findings(followups),
            sources=format_sources(evidence, cited_ids(followups)),
        )
        return self.llm.chat([{"role": "user", "content": prompt}], stage="reflect").strip()
