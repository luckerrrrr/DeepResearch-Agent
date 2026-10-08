from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from dataclasses import dataclass
from typing import Protocol

from mcp import Client, StdioServerParameters

from deep_research.config import ROOT


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict

    def signature(self) -> str:
        props = self.input_schema.get("properties", {})
        required = set(self.input_schema.get("required", []))
        params = ", ".join(f"{k}: {v.get('type', 'any')}{'' if k in required else '（可选）'}" for k, v in props.items())
        return f"- {self.name}({params})：{self.description}"


class ToolClient(Protocol):
    tools: list[ToolSpec]

    def call(self, name: str, arguments: dict) -> str: ...


class MCPToolClient:
    """MCP 客户端：子进程启动工具服务；MCP SDK 是异步的，用后台线程事件循环桥接同步的 Agent 主流程。"""

    def __init__(self, command: list[str] | None = None, call_timeout: float = 180) -> None:
        command = command or [sys.executable, "-m", "deep_research.tools.mcp_server"]
        self._params = StdioServerParameters(command=command[0], args=command[1:], cwd=str(ROOT), env=dict(os.environ))
        self._call_timeout = call_timeout
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
        self._thread.start()
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._client: Client | None = None
        self._error: BaseException | None = None

        self._lifecycle = asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        if not self._ready.wait(timeout=60):
            raise RuntimeError("MCP 工具服务启动超时")
        if self._error:
            raise RuntimeError(f"MCP 工具服务启动失败：{self._error}") from self._error
        self.tools: list[ToolSpec] = self._submit(self._list_tools(), timeout=30)

    async def _run(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with Client(self._params) as client:
                self._client = client
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    def _submit(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    async def _list_tools(self) -> list[ToolSpec]:
        result = await self._client.list_tools()
        return [ToolSpec(t.name, t.description or "", t.input_schema) for t in result.tools]

    def call(self, name: str, arguments: dict) -> str:
        result = self._submit(self._client.call_tool(name, arguments), timeout=self._call_timeout)
        text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        if result.is_error:
            return json.dumps({"error": text or "工具执行失败"}, ensure_ascii=False)
        return text

    def close(self) -> None:
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
            try:
                self._lifecycle.result(timeout=10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def __enter__(self) -> "MCPToolClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
