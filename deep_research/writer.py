from __future__ import annotations

from .evidence import EvidenceStore
from .llm import LLMClient
from .prompts import WRITER, render
from .react_agent import ReActResult


def format_findings(findings: list[ReActResult]) -> str:
    if not findings:
        return "（无）"
    return "\n\n".join(f"{i}. 子问题：{f.question}\n结论：{f.answer}" for i, f in enumerate(findings, 1))


def format_sources(evidence: EvidenceStore, ids: list[str], max_chars: int = 500) -> str:
    sources = evidence.subset(ids)
    return "\n\n".join(s.brief(max_chars) for s in sources) if sources else "（无）"


def cited_ids(findings: list[ReActResult]) -> list[str]:
    ids: list[str] = []
    for f in findings:
        ids += [i for i in f.citations if i not in ids]
    return ids


class ReportWriter:
    """汇总各子问题的结论，生成带引用的结构化报告。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def write(self, question: str, findings: list[ReActResult], evidence: EvidenceStore) -> str:
        ids = cited_ids(findings) or [s.id for s in evidence.all()[:10]]
        prompt = render(WRITER, question=question, findings=format_findings(findings), sources=format_sources(evidence, ids))
        return self.llm.chat([{"role": "user", "content": prompt}], stage="write").strip()
