from __future__ import annotations

from deep_research.evidence import EvidenceStore, extract_citations
from deep_research.llm import LLMClient
from deep_research.prompts import render
from deep_research.verifier import VerificationResult, Verifier

ACCURACY = """你是一名评分员，请根据标准答案判断研究报告是否正确回答了问题。

问题：{question}
标准答案：{answer}
必须包含的要点：{key_points}

研究报告：
{report}

评分标准：
- 1：所有要点都正确出现在报告中（表述不同但含义一致也算正确）
- 0.5：部分要点正确，但有要点缺失
- 0：主要要点错误或缺失，或报告内容与标准答案矛盾

只输出 JSON：{"score": 1 或 0.5 或 0, "reason": "简要理由"}"""


class Judge:
    """评估裁判：用标准答案打准确率分；用独立的溯源核查测量幻觉率（与流水线内的验证器分开调用）。"""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.verifier = Verifier(llm, stage="judge")

    def accuracy(self, item: dict, report: str) -> dict:
        prompt = render(
            ACCURACY,
            question=item["question"],
            answer=item["answer"],
            key_points="；".join(item["key_points"]),
            report=report,
        )
        data = self.llm.chat_json([{"role": "user", "content": prompt}], stage="judge")
        try:
            score = float(data.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        return {"score": min(max(score, 0.0), 1.0), "reason": str(data.get("reason", ""))}

    def grounding(self, report: str, evidence: EvidenceStore) -> dict:
        checks = self.verifier.check(report, evidence)
        fabricated = [i for i in extract_citations(report) if i not in evidence]
        return VerificationResult(checks, report, fabricated_citations=fabricated).stats()
