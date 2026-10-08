"""消融实验：python -m eval.run_eval [--limit N] [--force]"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from deep_research.config import ROOT, load_settings
from deep_research.llm import LLMClient
from deep_research.pipeline import DeepResearchAgent, PipelineConfig
from deep_research.tools.mcp_client import MCPToolClient
from eval.judge import Judge

# 后三种配置取自同一次完整运行的阶段快照：阶段串行且后续阶段不影响前面的输出，等价于独立运行但省去约 2/3 开销
CONFIGS = [("ReAct", "react"), ("+Plan-and-Solve", "plan_solve"), ("+Reflection", "reflection"), ("+溯源验证", "verified")]
BENCHMARK = ROOT / "eval" / "benchmark.json"
RESULTS = ROOT / "eval" / "results"


def evaluate(item: dict, llm: LLMClient, tools: MCPToolClient, judge: Judge) -> dict:
    baseline_cfg = PipelineConfig(use_planner=False, use_reflection=False, use_verification=False, use_memory=False)
    baseline = DeepResearchAgent(llm, tools, baseline_cfg).run(item["question"])
    full = DeepResearchAgent(llm, tools, PipelineConfig(use_memory=False)).run(item["question"])

    configs = {}
    for name, stage in CONFIGS:
        run = baseline if stage == "react" else full
        snap = run.snapshot(stage)
        accuracy = judge.accuracy(item, snap.report)
        configs[name] = {
            "accuracy": accuracy["score"],
            "accuracy_reason": accuracy["reason"],
            **judge.grounding(snap.report, run.evidence),
            "seconds": snap.seconds,
            "llm_calls": snap.llm_calls,
            "total_tokens": snap.total_tokens,
            "tool_calls": snap.tool_calls,
            "report": snap.report,
        }
    return {
        "id": item["id"],
        "question": item["question"],
        "gold": item["answer"],
        "configs": configs,
        "runs": {"react": baseline.to_dict(), "full": full.to_dict()},
    }


def compact(result: dict, max_chars: int = 300) -> dict:
    """详情文件只保留来源和观察的摘录，避免把网页全文存进仓库。"""
    for run in result["runs"].values():
        for source in run["sources"]:
            source["content"] = source["content"][:max_chars]
        for finding in run["findings"] + run["followups"]:
            for step in finding["steps"]:
                step["observation"] = step["observation"][:max_chars]
    return result


def summarize(results: list[dict]) -> list[dict]:
    rows = []
    for name, _ in CONFIGS:
        per_q = [r["configs"][name] for r in results]
        n = len(per_q)
        claims = sum(c["claims"] for c in per_q)
        rows.append({
            "config": name,
            "accuracy": sum(c["accuracy"] for c in per_q) / n,
            "hallucination_rate": sum(c["unsupported"] for c in per_q) / claims if claims else 0.0,
            "support_rate": sum(c["supported"] + 0.5 * c["partial"] for c in per_q) / claims if claims else 1.0,
            "fabricated_citations": sum(c["fabricated_citations"] for c in per_q),
            "avg_seconds": sum(c["seconds"] for c in per_q) / n,
            "avg_tokens": sum(c["total_tokens"] for c in per_q) / n,
            "avg_llm_calls": sum(c["llm_calls"] for c in per_q) / n,
            "avg_tool_calls": sum(c["tool_calls"] for c in per_q) / n,
        })
    return rows


def write_markdown(rows: list[dict], results: list[dict], path: Path) -> None:
    lines = [
        f"# 消融实验结果（{len(results)} 道多跳问题）",
        "",
        "| 配置 | 准确率 | 幻觉率 | 来源支持率 | 编造引用 | 平均耗时(s) | 平均 Token | 平均 LLM 调用 | 平均工具调用 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['config']} | {r['accuracy']:.1%} | {r['hallucination_rate']:.1%} | {r['support_rate']:.1%} | "
            f"{r['fabricated_citations']} | {r['avg_seconds']:.1f} | {r['avg_tokens']:,.0f} | {r['avg_llm_calls']:.1f} | {r['avg_tool_calls']:.1f} |"
        )
    lines += ["", "## 逐题准确率", "", "| 题号 | " + " | ".join(n for n, _ in CONFIGS) + " |", "|---" * (len(CONFIGS) + 1) + "|"]
    for r in results:
        lines.append(f"| {r['id']} | " + " | ".join(f"{r['configs'][n]['accuracy']:g}" for n, _ in CONFIGS) + " |")
    lines += [
        "",
        "指标说明：准确率由裁判模型对照标准答案打分（1 / 0.5 / 0）后取平均；幻觉率 = 无依据陈述数 ÷ 可核查陈述总数，",
        "由独立的裁判调用逐句核对报告陈述与其引用的来源原文得出；耗时、Token 等开销为截至该阶段的累计值。",
    ]
    path.write_text("\n".join(lines) + "\n", "utf-8")


def plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    names = [r["config"] for r in rows]
    panels = [
        ("准确率", [r["accuracy"] * 100 for r in rows], "%", "#4C72B0"),
        ("幻觉率（越低越好）", [r["hallucination_rate"] * 100 for r in rows], "%", "#C44E52"),
        ("平均耗时", [r["avg_seconds"] for r in rows], "s", "#8C8C8C"),
        ("平均 Token", [r["avg_tokens"] / 1000 for r in rows], "k", "#8C8C8C"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (title, values, unit, color) in zip(axes, panels):
        bars = ax.bar(names, values, color=color, width=0.6)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=20, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, v in zip(bars, values):
            ax.annotate(f"{v:.1f}{unit}", (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepResearch-Agent 消融实验")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 道题")
    parser.add_argument("--force", action="store_true", help="忽略已有结果，重新评估")
    args = parser.parse_args()

    items = json.loads(BENCHMARK.read_text("utf-8"))[: args.limit]
    details = RESULTS / "details"
    details.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    llm = LLMClient.from_settings(settings)
    judge = Judge(LLMClient.from_settings(settings, judge=True))

    results = []
    with MCPToolClient() as tools:
        for i, item in enumerate(items, 1):
            path = details / f"{item['id']}.json"
            if path.exists() and not args.force:
                results.append(json.loads(path.read_text("utf-8")))
                print(f"[{i}/{len(items)}] {item['id']} 已有结果，跳过")
                continue
            start = time.perf_counter()
            try:
                result = evaluate(item, llm, tools, judge)
            except Exception as exc:  # 单题失败不影响整体，下次运行会自动重试
                print(f"[{i}/{len(items)}] {item['id']} 失败：{type(exc).__name__}: {exc}")
                continue
            path.write_text(json.dumps(compact(result), ensure_ascii=False, indent=2), "utf-8")
            results.append(result)
            acc = " / ".join(f"{result['configs'][n]['accuracy']:g}" for n, _ in CONFIGS)
            print(f"[{i}/{len(items)}] {item['id']} 完成（{time.perf_counter() - start:.0f}s），各配置准确率：{acc}", flush=True)

    if not results:
        return
    rows = summarize(results)
    (RESULTS / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    write_markdown(rows, results, RESULTS / "summary.md")
    plot(rows, RESULTS / "ablation.png")
    print((RESULTS / "summary.md").read_text("utf-8"))


if __name__ == "__main__":
    main()
