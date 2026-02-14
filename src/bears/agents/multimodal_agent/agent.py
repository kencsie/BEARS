"""
Multimodal RAG Agent (Stub).

Placeholder for future multimodal retrieval capabilities.
"""

from typing import Optional, Set

from bears.agents.base import AgentCapability, AgentResponse, BaseRAGAgent
from bears.core.experiment import ExperimentConfig


class MultimodalAgent(BaseRAGAgent):
    """Stub multimodal agent. Returns empty response with confidence 0.0."""

    def __init__(self, experiment: Optional[ExperimentConfig] = None):
        self.exp = experiment or ExperimentConfig()

    @property
    def name(self) -> str:
        return "multimodal"

    @property
    def capabilities(self) -> Set[AgentCapability]:
        return {AgentCapability.MULTIMODAL}

    async def run(self, question: str, experiment: Optional[ExperimentConfig] = None) -> AgentResponse:
        return AgentResponse(
            answer="Multimodal agent is not yet implemented.",
            retrieved_doc_ids=[],
            context=[],
            confidence=0.0,
            metadata={"agent": "multimodal", "status": "stub"},
        )
