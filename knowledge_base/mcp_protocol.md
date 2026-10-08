# MCP（Model Context Protocol）笔记

## 基本概念

MCP 是 Anthropic 于 2024 年 11 月发布的开放协议，用来标准化大模型应用与外部工具、数据源之间的连接方式。它常被比作 AI 应用的“USB-C 接口”：工具只要实现一次 MCP 服务端，就能被任何支持 MCP 的客户端使用。

## 架构

MCP 采用客户端-服务端架构，消息格式基于 JSON-RPC 2.0。宿主应用（Host，如 IDE、聊天应用、Agent 程序）内部的 MCP 客户端与 MCP 服务端建立一对一连接。常见的传输方式有两种：stdio（服务端作为本地子进程运行，通过标准输入输出通信）和 Streamable HTTP（用于远程服务）。

## 核心能力

服务端可以向客户端提供三类能力：Tools（可由模型调用的函数）、Resources（可读取的数据，如文件内容）、Prompts（预定义的提示模板）。客户端在连接时可以通过 list_tools 动态发现服务端提供的工具及其参数的 JSON Schema。

## 与 Function Calling 的区别

Function Calling 是模型层面的能力，即模型按约定格式输出“要调用哪个函数、参数是什么”；MCP 是应用层的协议，规定工具如何被发现、描述和调用。两者是互补关系：Agent 通过 MCP 发现工具，再借助 Function Calling 或 ReAct 等方式决定调用哪个工具。使用 MCP 的好处是工具实现与 Agent 逻辑解耦，工具可以独立部署、复用和替换。
