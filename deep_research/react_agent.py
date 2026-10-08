from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Callable

from .evidence import CITATION_RE, EvidenceStore, extract_citations
from .llm import LLMClient
from .prompts import REACT_FORCE_FINISH, REACT_NEED_EVIDENCE, REACT_SYSTEM, REACT_USER, render
from .tools.mcp_client import ToolClient

EventHandler = Callable[[str, dict], None]


@dataclass
class Step:
    thought: str
    action: str
    action_input: dict
    observation: str = ""
    source_ids: list[str] = field(default_factory=list)


@dataclass
class ReActResult:
    question: str
    answer: str
    steps: list[Step]
    finished: bool

    @property
    def citations(self) -> list[str]:
        return extract_citations(self.answer)

    def to_dict(self) -> dict:
        return asdict(self)


class ReActAgent:
    """ReAct 执行器：Thought → Action（MCP 工具）→ Observation 循环，每步输出结构化 JSON。"""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolClient,
        max_steps: int = 4,
        observation_chars: int = 2500,
        on_event: EventHandler | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.observation_chars = observation_chars
        self.on_event = on_event or (lambda kind, data: None)
        self.tool_names = {t.name for t in tools.tools}
        self.tools_desc = "\n".join(t.signature() for t in tools.tools)

    def run(
        self,
        question: str,
        evidence: EvidenceStore,
        context: str = "",
        max_steps: int | None = None,
        stage: str = "execute",
    ) -> ReActResult:
        messages = [
            {"role": "system", "content": render(REACT_SYSTEM, tools=self.tools_desc)},
            {"role": "user", "content": render(REACT_USER, question=question, context=context or "（无）")},
        ]
        steps: list[Step] = []
        seen: dict[str, tuple[str, list[str]]] = {}
        has_evidence = bool(CITATION_RE.search(context))
        pushed_back = False

        for _ in range(max_steps or self.max_steps):
            decision = self.llm.chat_json(messages, stage=stage)
            messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
            action = str(decision.get("action") or "").strip()
            if action.lower() == "finish" or (not action and decision.get("answer")):
                # 证据优先：没有检索过任何来源就作答，等于凭参数记忆回答、无法溯源，打回一次要求先检索
                if not steps and not has_evidence and not pushed_back:
                    pushed_back = True
                    self.on_event("react_pushback", {"question": question})
                    messages.append({"role": "user", "content": REACT_NEED_EVIDENCE})
                    continue
                return self._finish(question, decision, steps, finished=True)

            thought = str(decision.get("thought", "")).strip()
            args = decision.get("action_input") if isinstance(decision.get("action_input"), dict) else {}
            self.on_event("react_action", {"thought": thought, "action": action, "action_input": args})
            observation, ids = self._execute(action, args, evidence, seen)
            steps.append(Step(thought, action, args, observation, ids))
            self.on_event("react_observation", {"action": action, "source_ids": ids, "preview": observation[:200]})
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})

        messages.append({"role": "user", "content": REACT_FORCE_FINISH})
        return self._finish(question, self.llm.chat_json(messages, stage=stage), steps, finished=False)

    def _finish(self, question: str, decision: dict, steps: list[Step], finished: bool) -> ReActResult:
        answer = str(decision.get("answer") or "").strip() or "未能找到足够的信息回答该问题。"
        result = ReActResult(question, answer, steps, finished)
        self.on_event("react_finish", {"question": question, "answer": answer, "steps": len(steps)})
        return result

    def _execute(
        self, action: str, args: dict, evidence: EvidenceStore, seen: dict[str, tuple[str, list[str]]]
    ) -> tuple[str, list[str]]:
        if action not in self.tool_names:
            return f"错误：不存在名为 {action!r} 的工具。可用工具：{', '.join(sorted(self.tool_names))}", []
        key = json.dumps([action, args], sort_keys=True, ensure_ascii=False)
        if key in seen:
            observation, ids = seen[key]
            return "（与之前的某次调用完全相同，以下是当时的结果。请换关键词或换工具。）\n" + observation, ids

        try:
            payload = json.loads(self.tools.call(action, args))
        except Exception as exc:
            return f"工具调用失败：{type(exc).__name__}: {exc}", []
        if isinstance(payload, dict) and "error" in payload:
            return f"工具调用失败：{payload['error']}", []
        items = [item for item in payload if isinstance(item, dict) and item.get("content")] if isinstance(payload, list) else []
        if not items:
            return "没有找到相关结果。请换关键词或换工具。", []

        query = str(next(iter(args.values()), "")) if args else ""
        # 完整内容存入证据库供后续溯源验证；放进上下文的 Observation 按预算截断，控制上下文长度
        budget = max(300, self.observation_chars // len(items))
        blocks, ids = [], []
        for item in items:
            source = evidence.add(action, str(item.get("title", "")), str(item.get("url", "")), str(item["content"]), query)
            content = str(item["content"])
            content = content if len(content) <= budget else content[:budget] + "…"
            blocks.append(f"[{source.id}] {source.title} | {source.url}\n{content}")
            ids.append(source.id)
        observation = "\n\n".join(blocks)
        seen[key] = (observation, ids)
        return observation, ids
