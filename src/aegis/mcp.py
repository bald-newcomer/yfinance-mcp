from contextlib import asynccontextmanager

from mcp import ClientSession, ListToolsResult
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import Any, Optional
import asyncio


class MCPClientManager:
    def __init__(self):
        self._client: Optional[ClientSession] = None
        self._lock = asyncio.Lock()
        self._ref_count = 0  # 引用计数

    @asynccontextmanager
    async def get_client(self):
        """获取共享的客户端（引用计数管理）"""
        async with self._lock:
            if self._client is None:
                # 首次使用，创建连接
                server_params = StdioServerParameters(
                    command="uv",
                    args=["run", "yfmcp"],
                )
                read_write = await stdio_client(server_params).__aenter__()
                self._client = ClientSession(*read_write)
                await self._client.initialize()
                print("创建新的共享客户端连接")

            self._ref_count += 1

        try:
            yield self._client
        finally:
            async with self._lock:
                self._ref_count -= 1
                if self._ref_count == 0:
                    await self._client.close()
                    self._client = None
                    print("关闭共享客户端连接")


# 全局单例
mcp_manager = MCPClientManager()


def convert_mcp_tools_to_openai_format() -> list[dict[str, Any]]:
    """Convert MCP tools to OpenAI tool format."""

    tools_list = mcp_manager._client.list_tools()
    tools = []
    for tool in tools_list.tools:
        tool_def = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        tools.append(tool_def)
    return tools


# # 使用方式
# async def task1():
#     async with mcp_manager.get_client() as client:
#         result = await client.list_tools()
#         # 处理结果...


# async def task2():
#     async with mcp_manager.get_client() as client:
#         result = await client.list_tools()
#         # 处理结果...
