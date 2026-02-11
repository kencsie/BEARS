"""
Orchestrator state definition.
"""

import operator
from typing import Annotated, List, TypedDict

from bears.agents.base import AgentResponse
from bears.router.base import RouterOutput


class OrchestratorState(TypedDict):
    question: str
    route_decision: RouterOutput
    agent_results: Annotated[List[AgentResponse], operator.add]
    final_answer: str
    retry_count: int
    trace_id: str
