"""
Orchestrator node functions.

All LangGraph nodes: router, agent wrapper, merge, quality gate.
"""

import logging
from typing import Any, Callable, Dict

from langgraph.types import Command, Send

from bears.agents.base import AgentResponse, BaseRAGAgent
from bears.orchestrator.state import OrchestratorState
from bears.router.llm_router import LLMRouter

logger = logging.getLogger(__name__)

_router = None


def _get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router


async def router_node(state: OrchestratorState) -> Command:
    """Route the question to appropriate agents using LLM classification."""
    question = state["question"]
    router = _get_router()
    decision = await router.route(question)

    logger.info(f"Router selected agents: {decision.selected_agents} (reasoning: {decision.reasoning})")

    # Fan out to selected agents
    sends = [
        Send(agent_name, {"question": question, "route_decision": decision})
        for agent_name in decision.selected_agents
    ]

    return Command(
        update={"route_decision": decision},
        goto=sends,
    )


def wrap_agent_as_node(agent: BaseRAGAgent) -> Callable:
    """Wrap a BaseRAGAgent into a LangGraph node function."""

    async def agent_node(state: OrchestratorState) -> Dict[str, Any]:
        question = state["question"]
        try:
            result = await agent.run(question)
            logger.info(f"Agent '{agent.name}' returned confidence={result.confidence:.2f}")
        except Exception as e:
            logger.error(f"Agent '{agent.name}' failed: {e}")
            result = AgentResponse(
                answer=f"Agent {agent.name} encountered an error.",
                confidence=0.0,
                metadata={"error": str(e)},
            )
        return {"agent_results": [result]}

    agent_node.__name__ = agent.name
    return agent_node


async def merge_node(state: OrchestratorState) -> Dict[str, Any]:
    """Select the best AgentResponse by highest confidence."""
    results = state.get("agent_results", [])
    if not results:
        return {"final_answer": "No agent produced a result."}

    best = max(results, key=lambda r: r.confidence)
    logger.info(f"Merge selected answer with confidence={best.confidence:.2f}")
    return {"final_answer": best.answer}


async def quality_gate(state: OrchestratorState) -> Dict[str, Any]:
    """Optional quality check on the final answer."""
    results = state.get("agent_results", [])
    if not results:
        return {"retry_count": state.get("retry_count", 0)}

    best = max(results, key=lambda r: r.confidence)
    if best.confidence < 0.3 and state.get("retry_count", 0) < 1:
        logger.info("Quality gate: low confidence, could retry")
        return {"retry_count": state.get("retry_count", 0) + 1}

    return {"retry_count": state.get("retry_count", 0)}
