"""
Orchestrator entry point.

Router has been removed.  All queries go directly to AgenticAgent, which
internally coordinates multi-retriever search via ComprehensiveSearchTool.
"""

import logging
import time
import uuid
from typing import Any, Dict, Optional

from bears.agents.registry import get_agent

logger = logging.getLogger(__name__)


async def run_orchestrated_rag(
    question: str, experiment: Optional[Any] = None
) -> Dict[str, Any]:
    """Run the full agentic pipeline and return a normalised result dict."""
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    agent = get_agent("agentic")
    result = await agent.run(question, experiment)

    total_time = time.time() - start_time

    meta = result.metadata or {}
    return {
        "question": question,
        "answer": result.answer,
        "retrieved_doc_ids": result.retrieved_doc_ids,
        "context": result.context,
        "trace_id": trace_id,
        "total_time": total_time,
        "retrieval_time": result.retrieval_time,
        "generation_time": result.generation_time,
        "agent_used": "agentic",
        "confidence": result.confidence,
        "prompt_tokens": meta.get("prompt_tokens", 0),
        "completion_tokens": meta.get("completion_tokens", 0),
        "total_tokens": meta.get("total_tokens", 0),
        "tools_used": meta.get("tools_used", []),
    }
