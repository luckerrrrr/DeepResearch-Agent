from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMClient
from .prompts import PLANNER, render


@dataclass
class Plan:
    standalone_question: str
    sub_questions: list[str]
    rationale: str


class Planner:
    """Plan-and-Solve：先把研究问题拆成有序子问题；有对话历史时顺带做指代消解（问题改写）。"""

    def __init__(self, llm: LLMClient, max_sub_questions: int = 4) -> None:
        self.llm = llm
        self.max_sub_questions = max_sub_questions

    def plan(self, question: str, history: str = "", memory: str = "") -> Plan:
        prompt = render(
            PLANNER,
            question=question,
            max_n=self.max_sub_questions,
            history=f"对话历史：\n{history}\n\n" if history else "",
            memory=f"长期记忆中的相关信息：\n{memory}\n\n" if memory else "",
        )
        data = self.llm.chat_json([{"role": "user", "content": prompt}], stage="plan")
        standalone = str(data.get("standalone_question") or question).strip()
        subs = [s.strip() for s in data.get("sub_questions") or [] if isinstance(s, str) and s.strip()]
        return Plan(
            standalone_question=standalone,
            sub_questions=subs[: self.max_sub_questions] or [standalone],
            rationale=str(data.get("rationale", "")).strip(),
        )
