"""
Orchestrator node functions.

LangGraph nodes: router and per-agent nodes.
"""

import logging
from typing import Any, Dict

from bears.agents.base import AgentResponse
from bears.agents.registry import get_agent
from bears.orchestrator.state import OrchestratorState
from bears.router.llm_router import LLMRouter

logger = logging.getLogger(__name__)

_router = None


def _get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router


def _agent_result_to_state(result: AgentResponse) -> Dict[str, Any]:
    return {
        "final_answer": result.answer,
        "agent_results": [result],
    }


async def router_node(state: OrchestratorState) -> Dict[str, Any]:
    """Route the question to a single best agent."""
    question = state["question"]
    router = _get_router()
    decision = await router.route(question)

    route = decision.selected_agents[0]  # pick the top choice
    logger.info(f"Router selected agent: {route} (reasoning: {decision.reasoning})")

    return {"route_decision": decision, "route": route}


async def hybrid_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    agent = get_agent("hybrid")
    result = await agent.run(state["question"])
    return _agent_result_to_state(result)


async def kg_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    agent = get_agent("kg")
    result = await agent.run(state["question"])
    return _agent_result_to_state(result)


async def agentic_agent_node(state: OrchestratorState) -> Dict[str, Any]:
    agent = get_agent("agentic")
    result = await agent.run(state["question"])
    return _agent_result_to_state(result)


def route_after_router(state: OrchestratorState) -> str:
    """Conditional edge: return the agent name chosen by the router."""
    return state["route"]
