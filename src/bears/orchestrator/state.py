"""
Orchestrator state definition.
"""

from typing import List, TypedDict, Optional

from bears.agents.base import AgentResponse
from bears.router.base import RouterOutput
from bears.core.experiment import ExperimentConfig

class OrchestratorState(TypedDict):
    question: str
    route: str
    route_decision: RouterOutput
    agent_results: List[AgentResponse]
    final_answer: str
    retry_count: int
    trace_id: str
    experiment: Optional[ExperimentConfig]
