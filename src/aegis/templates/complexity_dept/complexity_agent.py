from pydantic_ai import Agent, RunContext
from aegis.templates.agent_base import Dependencies, Output
from dataclasses import dataclass
from aegis.templates.template import TemplateRegistry
from demo import get_mcp_tools
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from yfmcp.server import mcp


@dataclass
class complexity_dependencies(Dependencies):
    ts_code: str


@dataclass
class signal(Output):
    statement: str
    evidence: str


@dataclass
class risk(Output):
    threat: str
    monitor: str
    threshold: str


@dataclass
class complexity_output(Output):
    ts_code: str
    summary: str
    signals: list[signal]
    risks: list[risk]


complexity_agent = Agent(
    "deepseek:deepseek-chat",
    deps_type=complexity_dependencies,
    output_type=complexity_output,
    system_prompt="你是一个多智能体防御型投资分析系统中的分部决策者，需要根据提供的结构化信息给出评估意见。",
    toolsets=[FastMCPToolset(mcp)],
)


@complexity_agent.instructions
def analysis_complexity(ctx: RunContext[str]) -> str:
    template = TemplateRegistry.get("complexity_dept")
    return template.format(ctx.deps)
    # return f'The date is {date.today()}.'
