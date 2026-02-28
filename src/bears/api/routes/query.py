"""
Query API endpoints.

Provides single question query and health check.
"""

from fastapi import APIRouter, HTTPException

from bears.api.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query a single question via agent or orchestrator."""
    if request.agent:
        from bears.agents.registry import get_agent

        try:
            agent = get_agent(request.agent)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        result = await agent.run(request.question)
        return QueryResponse(
            answer=result.answer,
            agent_used=request.agent,
            retrieved_doc_ids=result.retrieved_doc_ids,
            confidence=result.confidence,
        )
    else:
        from bears.orchestrator.graph import run_orchestrated_rag

        result = await run_orchestrated_rag(request.question)
        return QueryResponse(
            answer=result["answer"],
            agent_used=result.get("agent_used", "orchestrator"),
            retrieved_doc_ids=result.get("retrieved_doc_ids", []),
            confidence=result.get("confidence", 0.0),
        )


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
