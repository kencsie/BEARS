"""
Orchestrator graph assembly.

Builds the StateGraph and provides run_orchestrated_rag() entry point.
"""

import logging
import uuid
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from bears.agents.base import AgentResponse
from bears.agents.registry import get_enabled_agents
from bears.orchestrator.nodes import merge_node, router_node, wrap_agent_as_node
from bears.orchestrator.state import OrchestratorState

logger = logging.getLogger(__name__)


def create_orchestrator_workflow(registry: Dict[str, Any] = None) -> StateGraph:
    """Build the orchestrator StateGraph with router, agent nodes, and merge."""
    agents = get_enabled_agents()

    workflow = StateGraph(OrchestratorState)

    # Add router node
    workflow.add_node("router", router_node)

    # Add agent nodes
    for agent_name, agent in agents.items():
        node_fn = wrap_agent_as_node(agent)
        workflow.add_node(agent_name, node_fn)
        workflow.add_edge(agent_name, "merge")

    # Add merge node
    workflow.add_node("merge", merge_node)

    # Set entry point
    workflow.set_entry_point("router")
    workflow.add_edge("merge", END)

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
        "route_decision": None,
        "agent_results": [],
        "final_answer": "",
        "retry_count": 0,
        "trace_id": trace_id,
    }

    workflow = _get_workflow()
    result = await workflow.ainvoke(initial_state)

    total_time = time.time() - start_time

    # Extract best agent result for doc_ids and context
    agent_results = result.get("agent_results", [])
    best = max(agent_results, key=lambda r: r.confidence) if agent_results else AgentResponse(answer="")

    return {
        "question": question,
        "answer": result.get("final_answer", ""),
        "retrieved_doc_ids": best.retrieved_doc_ids,
        "context": best.context,
        "trace_id": trace_id,
        "total_time": total_time,
    }
