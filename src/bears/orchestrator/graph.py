"""
Orchestrator graph assembly.

Builds the StateGraph and provides run_orchestrated_rag() entry point.
"""

import logging
import uuid
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from bears.agents.base import AgentResponse
from bears.orchestrator.nodes import (
    agentic_agent_node,
    hybrid_agent_node,
    kg_agent_node,
    route_after_router,
    router_node,
)
from bears.orchestrator.state import OrchestratorState

logger = logging.getLogger(__name__)


def create_orchestrator_workflow():
    """Build the orchestrator StateGraph: router -> single agent -> END."""
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("router", router_node)
    workflow.add_node("hybrid", hybrid_agent_node)
    workflow.add_node("kg", kg_agent_node)
    workflow.add_node("agentic", agentic_agent_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges("router", route_after_router, {
        "hybrid": "hybrid",
        "kg": "kg",
        "agentic": "agentic",
    })
    workflow.add_edge("hybrid", END)
    workflow.add_edge("kg", END)
    workflow.add_edge("agentic", END)

    return workflow.compile()


_workflow = None


def _get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = create_orchestrator_workflow()
    return _workflow


async def run_orchestrated_rag(question: str) -> Dict[str, Any]:
    """Convenience entry point: run the full orchestrator pipeline."""
    import time

    start_time = time.time()
    trace_id = str(uuid.uuid4())

    initial_state = {
        "question": question,
        "route": "",
        "route_decision": None,
        "agent_results": [],
        "final_answer": "",
        "retry_count": 0,
        "trace_id": trace_id,
    }

    workflow = _get_workflow()
    result = await workflow.ainvoke(initial_state)

    total_time = time.time() - start_time

    # Single agent result
    agent_results = result.get("agent_results", [])
    best = agent_results[0] if agent_results else AgentResponse(answer="")

    return {
        "question": question,
        "answer": result.get("final_answer", ""),
        "retrieved_doc_ids": best.retrieved_doc_ids,
        "context": best.context,
        "trace_id": trace_id,
        "total_time": total_time,
    }
