from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable

import openai
from openai import OpenAI

from .config import Settings

Message = dict[str, str]

# 流式请求更稳定：部分代理/网关会断开长时间无数据的非流式连接
RETRYABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


class LLMError(RuntimeError):
    pass


@dataclass
class StageStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {**asdict(self), "total_tokens": self.total_tokens}


class UsageTracker:
    """按阶段（plan / execute / write / reflect / verify）统计调用次数、Token 与耗时。"""

    def __init__(self) -> None:
        self.stages: dict[str, StageStats] = {}
        self._lock = threading.Lock()

    def record(self, stage: str, prompt_tokens: int, completion_tokens: int, seconds: float) -> None:
        with self._lock:
            stats = self.stages.setdefault(stage, StageStats())
            stats.calls += 1
            stats.prompt_tokens += prompt_tokens
            stats.completion_tokens += completion_tokens
            stats.seconds += seconds

    def total(self) -> StageStats:
        with self._lock:
            total = StageStats()
            for s in self.stages.values():
                total.calls += s.calls
                total.prompt_tokens += s.prompt_tokens
                total.completion_tokens += s.completion_tokens
                total.seconds += s.seconds
            return total

    def to_dict(self) -> dict:
        with self._lock:
            return {name: s.to_dict() for name, s in self.stages.items()}


class LLMClient:
    """兼容 OpenAI 接口的 LLM 客户端：统一走流式调用，带重试、用量统计与 JSON 解析。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 120,
        reasoning_effort: str | None = None,
        json_mode: bool = True,
        max_attempts: int = 4,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        self.max_attempts = max_attempts
        self.tracker = UsageTracker()
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    @classmethod
    def from_settings(cls, settings: Settings, judge: bool = False) -> "LLMClient":
        return cls(
            model=settings.judge_model if judge else settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
            reasoning_effort=settings.judge_reasoning_effort if judge else settings.reasoning_effort,
            json_mode=settings.json_mode,
        )

    def chat(
        self,
        messages: list[Message],
        stage: str = "default",
        json_output: bool = False,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if json_output and self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(1, self.max_attempts + 1):
            start = time.perf_counter()
            try:
                parts: list[str] = []
                usage = None
                for chunk in self._client.chat.completions.create(**kwargs):
                    if chunk.usage:
                        usage = chunk.usage
                    if chunk.choices and chunk.choices[0].delta.content:
                        delta = chunk.choices[0].delta.content
                        parts.append(delta)
                        if on_token:
                            on_token(delta)
                text = "".join(parts)
                if not text.strip():
                    raise LLMError("模型返回了空响应")
                self.tracker.record(
                    stage,
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                    time.perf_counter() - start,
                )
                return text
            except (*RETRYABLE_ERRORS, LLMError) as exc:
                if attempt == self.max_attempts:
                    raise LLMError(f"LLM 调用失败（已尝试 {attempt} 次）：{exc}") from exc
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def chat_json(self, messages: list[Message], stage: str = "default") -> dict:
        text = self.chat(messages, stage=stage, json_output=True)
        try:
            return parse_json_object(text)
        except ValueError as exc:
            retry = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"你的输出不是合法的 JSON（{exc}）。请只输出一个 JSON 对象，不要包含任何其他内容。"},
            ]
            return parse_json_object(self.chat(retry, stage=stage, json_output=True))


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("未找到 JSON 对象")
        try:
            obj = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败：{exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("JSON 顶层必须是对象")
    return obj
