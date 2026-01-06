"""
Agent基类
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping
from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict

class AgentAction(str, Enum):
    PASS = "PASS"
    NOT_PASS = "NOT_PASS"


@dataclass
class AgentContext:
    ts_code: str
    trade_date: str
    raw: Mapping[str, Any] = field(default_factory=dict)  # 原始数据


class AgentBase:
    """Base class for all decision agents."""

    name: str
    dependencies: Dependencies
    output: Output

    def __init__(self, name: str) -> None:
        self.name = name

    # def score(self, context: AgentContext, action: AgentAction) -> float:
    #     """Return a normalized utility value in [0,1] for the proposed action."""

    #     raise NotImplementedError


class Output(BaseModel):
    """Agent 响应体"""


class Dependencies(BaseModel):
    """Agent 依赖的参数"""
    ts_code: str
    # @abstractmethod
    # def toDict(self, amount: float, currency: str) -> str:
    #     """支付方法，必须返回交易ID"""
    #     pass
