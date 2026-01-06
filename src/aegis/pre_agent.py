from pydantic_ai import Agent, RunContext
from aegis.templates.agent_base import Dependencies, Output
from dataclasses import dataclass
from aegis.templates.template import TemplateRegistry
from demo import get_mcp_tools
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from yfmcp.server import mcp

from aegis.templates.complexity_dept.complexity_agent import (
    complexity_agent,
    complexity_output,
)

import os

os.environ["DEEPSEEK_API_KEY"] = "sk-7399448825f04cbeb8bce541f8cbcdcb"


pre_agent = Agent(
    "deepseek:deepseek-chat",
    output_type=Dependencies,
    system_prompt="你是一个智能投资客服。你的任务是理解用户的自然语言，提取结构化数据",
)

post_agent = Agent(
    "deepseek:deepseek-chat",
    deps_type=complexity_output,
    system_prompt="你是一个智能投资客服。你的任务是将json格式化语言转变为方便用户理解的自热语言",
)
