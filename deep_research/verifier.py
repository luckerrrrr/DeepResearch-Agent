from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from .evidence import EvidenceStore, extract_citations, strip_citations
from .llm import LLMClient
from .prompts import VERIFY, render

LABELS = ("supported", "partial", "unsupported", "not_claim")
# 按句末标点切句，并把紧跟在句末的引用标记 [S1][S2] 归属到这一句
SENTENCE_RE = re.compile(r".+?(?:[。！？!?]|\.(?=\s|$)|$)(?:\s*\[S\d+(?:\s*[,，、]\s*S\d+)*\])*")
LIST_PREFIX_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)、])\s+)")
UNVERIFIED_MARK = "（未经来源证实）"


@dataclass
class ClaimCheck:
    id: int
    text: str
    citations: list[str]
    label: str = "partial"
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    checks: list[ClaimCheck]
    report: str
    removed: list[ClaimCheck] = field(default_factory=list)
    fabricated_citations: list[str] = field(default_factory=list)

    def stats(self) -> dict:
        claims = [c for c in self.checks if c.label != "not_claim"]
        counts = Counter(c.label for c in claims)
        n = len(claims)
        return {
            "claims": n,
            "supported": counts["supported"],
            "partial": counts["partial"],
            "unsupported": counts["unsupported"],
            "removed": len(self.removed),
            "fabricated_citations": len(self.fabricated_citations),
            "support_rate": round((counts["supported"] + 0.5 * counts["partial"]) / n, 4) if n else 1.0,
            "hallucination_rate": round(counts["unsupported"] / n, 4) if n else 0.0,
        }

    def supported_claims(self, evidence: EvidenceStore) -> list[tuple[str, list[str]]]:
        """通过验证的陈述及其原始来源链接，用于写入长期记忆（只记住被证实的结论，避免记忆被幻觉污染）。"""
        claims = []
        for c in self.checks:
            if c.label != "supported":
                continue
            urls = [s.url for s in evidence.subset(c.citations) if s.tool != "memory" and s.url]
            if urls:
                claims.append((strip_citations(c.text), urls))
        return claims

    def to_dict(self) -> dict:
        return {
            "stats": self.stats(),
            "checks": [c.to_dict() for c in self.checks],
            "removed": [c.to_dict() for c in self.removed],
            "fabricated_citations": self.fabricated_citations,
        }


@dataclass
class _Line:
    raw: str
    prefix: str = ""
    pieces: list[str] = field(default_factory=list)


def _parse(report: str) -> list[_Line]:
    lines = []
    for raw in report.split("\n"):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "|", "```")):
            lines.append(_Line(raw))
            continue
        match = LIST_PREFIX_RE.match(raw)
        prefix = match.group(1) if match else ""
        lines.append(_Line(raw, prefix, SENTENCE_RE.findall(raw[len(prefix):])))
    return lines


def _is_candidate(piece: str) -> bool:
    return len(strip_citations(piece).strip(" *_：:，,。.")) >= 6


class Verifier:
    """溯源验证：把报告切成句子级陈述，逐条核对其引用来源能否支撑，剔除（或标记）无依据的陈述。"""

    def __init__(
        self, llm: LLMClient, mode: str = "remove", batch_size: int = 8, source_chars: int = 1800, stage: str = "verify"
    ) -> None:
        self.llm = llm
        self.mode = mode
        self.batch_size = batch_size
        self.source_chars = source_chars
        self.stage = stage

    def check(self, report: str, evidence: EvidenceStore) -> list[ClaimCheck]:
        _, located = self._collect(report)
        self._judge([c for _, _, c in located], evidence, report)
        return [c for _, _, c in located]

    def verify(self, report: str, evidence: EvidenceStore) -> VerificationResult:
        lines, located = self._collect(report)
        checks = [c for _, _, c in located]
        self._judge(checks, evidence, report)

        flagged = {(li, pi): c for li, pi, c in located if c.label == "unsupported"}
        output = []
        for li, line in enumerate(lines):
            if not any((li, pi) in flagged for pi in range(len(line.pieces))):
                output.append(line.raw)
                continue
            kept = []
            for pi, piece in enumerate(line.pieces):
                if (li, pi) not in flagged:
                    kept.append(piece)
                elif self.mode == "flag":
                    kept.append(piece.rstrip() + UNVERIFIED_MARK)
            body = "".join(kept).strip()
            if body:
                output.append(line.prefix + body)

        fabricated = [i for i in extract_citations(report) if i not in evidence]
        removed = list(flagged.values()) if self.mode == "remove" else []
        final = re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()
        return VerificationResult(checks, final, removed, fabricated)

    def _collect(self, report: str) -> tuple[list[_Line], list[tuple[int, int, ClaimCheck]]]:
        lines = _parse(report)
        located = []
        for li, line in enumerate(lines):
            for pi, piece in enumerate(line.pieces):
                if _is_candidate(piece):
                    located.append((li, pi, ClaimCheck(len(located) + 1, piece.strip(), extract_citations(piece))))
        return lines, located

    def _judge(self, claims: list[ClaimCheck], evidence: EvidenceStore, report: str) -> None:
        pool = [i for i in extract_citations(report) if i in evidence][:8]
        pending = []
        for claim in claims:
            if claim.citations and not any(i in evidence for i in claim.citations):
                claim.label, claim.reason = "unsupported", "引用的来源编号不存在（编造引用）"
            else:
                pending.append(claim)

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            ids: list[str] = []
            for claim in batch:
                for sid in [i for i in claim.citations if i in evidence] or pool:
                    if sid not in ids:
                        ids.append(sid)
            if not ids:
                for claim in batch:
                    claim.label, claim.reason = "unsupported", "没有可供核对的来源"
                continue

            sources = "\n\n".join(s.brief(self.source_chars) for s in evidence.subset(ids))
            claims_text = "\n".join(
                json.dumps({"id": c.id, "text": strip_citations(c.text), "citations": c.citations}, ensure_ascii=False)
                for c in batch
            )
            data = self.llm.chat_json(
                [{"role": "user", "content": render(VERIFY, sources=sources, claims=claims_text)}], stage=self.stage
            )
            results = {}
            for r in data.get("results") or []:
                try:
                    results[int(r["id"])] = r
                except (KeyError, TypeError, ValueError):
                    continue
            for claim in batch:
                r = results.get(claim.id)
                if r and r.get("label") in LABELS:
                    claim.label, claim.reason = r["label"], str(r.get("reason", ""))
                else:
                    claim.label, claim.reason = "partial", "核查结果缺失，保守保留"
