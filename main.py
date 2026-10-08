from __future__ import annotations

import argparse

from deep_research.config import load_settings
from deep_research.events import describe_event
from deep_research.llm import LLMClient
from deep_research.memory import ConversationMemory, LongTermMemory
from deep_research.pipeline import DeepResearchAgent, PipelineConfig, ResearchResult
from deep_research.tools.mcp_client import MCPToolClient


def print_event(kind: str, data: dict) -> None:
    text = describe_event(kind, data)
    if text:
        print(text, flush=True)


def print_result(result: ResearchResult) -> None:
    print("\n" + "=" * 60)
    print(result.full_report)
    print("=" * 60)
    if result.verification:
        stats = result.verification.stats()
        print(f"溯源验证：支持率 {stats['support_rate']:.0%}，剔除无依据陈述 {stats['removed']} 条")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepResearch-Agent：带溯源验证的自动化深度研究智能体")
    parser.add_argument("question", nargs="?", help="研究问题")
    parser.add_argument("-i", "--interactive", action="store_true", help="多轮对话模式（短期记忆支持追问）")
    parser.add_argument("--no-plan", action="store_true", help="关闭 Plan-and-Solve，只用单个 ReAct 智能体")
    parser.add_argument("--no-reflection", action="store_true", help="关闭 Reflection")
    parser.add_argument("--no-verify", action="store_true", help="关闭溯源验证")
    parser.add_argument("--no-memory", action="store_true", help="关闭长期记忆")
    parser.add_argument("--flag", action="store_true", help="溯源验证只标记无依据陈述，不删除")
    parser.add_argument("--steps", type=int, default=4, help="每个子问题的 ReAct 最大步数")
    args = parser.parse_args()
    if not args.question and not args.interactive:
        parser.error("请提供研究问题，或使用 -i 进入多轮对话模式")

    settings = load_settings()
    config = PipelineConfig(
        use_planner=not args.no_plan,
        use_reflection=not args.no_reflection,
        use_verification=not args.no_verify,
        use_memory=not args.no_memory,
        react_max_steps=args.steps,
        verification_mode="flag" if args.flag else "remove",
    )
    llm = LLMClient.from_settings(settings)
    long_term = None if args.no_memory else LongTermMemory(settings)

    with MCPToolClient() as tools:
        print("已连接 MCP 工具服务：" + "、".join(t.name for t in tools.tools))
        agent = DeepResearchAgent(llm, tools, config, ConversationMemory(), long_term, on_event=print_event)
        if args.question:
            print_result(agent.run(args.question))
        if args.interactive:
            while True:
                try:
                    question = input("\n研究问题（直接回车退出）> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not question:
                    break
                print_result(agent.run(question))


if __name__ == "__main__":
    main()
