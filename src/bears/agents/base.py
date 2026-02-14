"""
Core agent contracts.

Defines BaseRAGAgent ABC, AgentCapability enum, and AgentResponse model.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Set, Optional

from pydantic import BaseModel

from bears.core.experiment import ExperimentConfig


class AgentCapability(Enum):
    VECTOR_SEARCH = "vector_search"
    GRAPH_SEARCH = "graph_search"
    WEB_SEARCH = "web_search"
    MULTI_HOP = "multi_hop"
    MULTIMODAL = "multimodal"


class AgentResponse(BaseModel):
    """Standardized response from any RAG agent."""
    answer: str
    retrieved_doc_ids: List[str] = []
    context: List[str] = []
    confidence: float = 0.0
    metadata: Dict[str, Any] = {}


class BaseRAGAgent(ABC):
    """Abstract base class for all RAG agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def capabilities(self) -> Set[AgentCapability]:
        ...

    @abstractmethod
    async def run(self, question: str, experiment: Optional[ExperimentConfig] = None) -> AgentResponse:
        ...
