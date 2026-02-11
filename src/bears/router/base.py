"""
Router contracts.

Defines BaseRouter ABC and RouterOutput model.
"""

from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel


class RouterOutput(BaseModel):
    """Output from a router: which agents to dispatch to."""
    selected_agents: List[str]
    reasoning: str
    confidence: float = 0.0


class BaseRouter(ABC):
    """Abstract base class for routers."""

    @abstractmethod
    async def route(self, question: str) -> RouterOutput:
        ...
