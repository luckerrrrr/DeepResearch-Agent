from __future__ import annotations

import json
import zlib
from typing import Callable

import numpy as np

from deep_research.llm import UsageTracker, parse_json_object
from deep_research.tools.mcp_client import ToolSpec


class FakeLLM:
    """按 handler(stage, messages) 返回预设输出，用于离线测试 Agent 逻辑。"""

    def __init__(self, handler: Callable[[str, list[dict]], object]) -> None:
        self.handler = handler
        self.tracker = UsageTracker()
        self.calls: list[tuple[str, list[dict]]] = []

    def chat(self, messages, stage="default", json_output=False, on_token=None) -> str:
        self.calls.append((stage, messages))
        self.tracker.record(stage, 10, 5, 0.0)
        out = self.handler(stage, messages)
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)

    def chat_json(self, messages, stage="default") -> dict:
        return parse_json_object(self.chat(messages, stage=stage, json_output=True))


def scripted(*outputs):
    queue = list(outputs)
    return lambda stage, messages: queue.pop(0)


class FakeTools:
    def __init__(self, responses: dict[str, object]) -> None:
        schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        self.tools = [ToolSpec(name, f"{name} 工具", schema) for name in responses]
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def call(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return json.dumps(self.responses[name], ensure_ascii=False)


class FakeEmbedder:
    """字符哈希词袋向量：确定性、无需下载模型，足以验证向量库的存取与排序逻辑。"""

    name = "fake-embedder"

    def _vec(self, text: str) -> np.ndarray:
        vec = np.zeros(256, dtype=np.float32)
        for ch in text:
            vec[zlib.crc32(ch.encode()) % 256] += 1
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)
