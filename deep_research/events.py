from __future__ import annotations

import json


def describe_event(kind: str, data: dict) -> str | None:
    """把流水线事件转成一行可读的进度文字，CLI 和 Web UI 共用。"""
    if kind == "memory_recall":
        return f"[记忆] 从长期记忆中召回 {data['count']} 条相关结论"
    if kind == "plan":
        subs = "\n".join(f"    {i}. {q}" for i, q in enumerate(data["sub_questions"], 1))
        return f"[规划] {data.get('rationale') or '拆解为以下子问题'}\n{subs}"
    if kind == "subtask_start":
        return f"[子问题 {data['index']}/{data['total']}] {data['question']}"
    if kind == "react_action":
        args = json.dumps(data["action_input"], ensure_ascii=False)
        return f"    思考：{data['thought']}\n    行动：{data['action']}({args})"
    if kind == "react_observation":
        if data["source_ids"]:
            return f"    观察：获得来源 {'、'.join(data['source_ids'])}"
        return f"    观察：{data['preview'][:100]}"
    if kind == "react_pushback":
        return "    守卫：尚未检索任何证据就试图作答，已要求先检索"
    if kind == "react_finish":
        return f"    结论：{data['answer']}"
    if kind == "writing":
        return "[写作] 汇总子问题结论，生成报告初稿"
    if kind == "critique":
        issues = "；".join(data["issues"]) or "无"
        return f"[反思] 评分 {data['score']}/10，问题：{issues}"
    if kind == "followup_start":
        return f"[补充检索] {data['question']}"
    if kind == "revised":
        return "[反思] 已根据审稿意见修订报告"
    if kind == "verifying":
        return "[溯源验证] 逐句核对陈述与来源原文"
    if kind == "verification":
        return (
            f"[溯源验证] 共 {data['claims']} 条陈述：支持 {data['supported']}，部分支持 {data['partial']}，"
            f"无依据 {data['unsupported']}（已处理 {data['removed']} 条）"
        )
    if kind == "memory_saved":
        return f"[记忆] 新写入 {data['count']} 条已验证结论"
    if kind == "done":
        return f"[完成] 用时 {data['seconds']:.1f}s，LLM 调用 {data['llm_calls']} 次，Token {data['total_tokens']}"
    return None
