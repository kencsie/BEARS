"""
Query API endpoints.

POST /api/retrieve   — Single Agentic RAG query, returns Q/A/C + timing + tokens.
POST /api/generate   — Retrieval + domain generator (chatbot mode).
POST /api/evaluate   — Streaming batch evaluation; saves results to output/ folder.
GET  /api/health     — Liveness check.
"""

import json
import time
import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from bears.api.schemas import (
    EvaluateBatchRequest,
    GenerateRequest,
    GenerateResponse,
    RetrieveRequest,
    RetrieveResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["query"])

RESULTS_DIR = Path("output")


# ── /retrieve ─────────────────────────────────────────────────────────────────

def _to_core_experiment(api_exp):
    """Convert API ExperimentConfig schema to core ExperimentConfig."""
    if api_exp is None:
        return None
    from bears.core.experiment import ExperimentConfig as CoreExp
    return CoreExp(**api_exp.model_dump())


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    """Run Agentic RAG and return the standard Q/A/C evaluation format."""
    from bears.orchestrator.graph import run_orchestrated_rag

    exp = _to_core_experiment(request.experiment)
    start = time.time()
    result = await run_orchestrated_rag(request.question, experiment=exp)
    total_time = time.time() - start

    return RetrieveResponse(
        question=request.question,
        answer=result["answer"],
        context=result.get("context", []),
        true_answer=request.true_answer,
        true_context=request.true_context,
        retrieval_time=result.get("retrieval_time", 0.0),
        generation_time=result.get("generation_time", 0.0),
        total_time=total_time,
        prompt_tokens=result.get("prompt_tokens", 0),
        completion_tokens=result.get("completion_tokens", 0),
        total_tokens=result.get("total_tokens", 0),
        tool_used=result.get("tools_used", []),
        experiment_config=request.experiment.model_dump() if request.experiment else None,
    )


# ── /generate ─────────────────────────────────────────────────────────────────

_GENERATORS: dict = {}


def _get_generator(name: str):
    if name not in _GENERATORS:
        if name == "educational":
            from bears.generators.educational import EducationalGenerator
            _GENERATORS[name] = EducationalGenerator()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown generator: {name}. Available: ['educational']")
    return _GENERATORS[name]


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Retrieve context then pass it through a domain-specific generator."""
    from bears.orchestrator.graph import run_orchestrated_rag

    retrieve_start = time.time()
    result = await run_orchestrated_rag(request.question)
    retrieval_time = time.time() - retrieve_start

    contexts: List[str] = result.get("context", [])

    gen_start = time.time()
    generator = _get_generator(request.generator)
    generated_content = await generator.generate(request.question, contexts)
    generation_time = time.time() - gen_start

    return GenerateResponse(
        question=request.question,
        generated_content=generated_content,
        context=contexts,
        retrieval_time=retrieval_time,
        generation_time=generation_time,
        total_time=retrieval_time + generation_time,
        prompt_tokens=result.get("prompt_tokens", 0),
        completion_tokens=result.get("completion_tokens", 0),
        total_tokens=result.get("total_tokens", 0),
    )


# ── /evaluate (streaming batch) ───────────────────────────────────────────────

async def _stream_batch(request: EvaluateBatchRequest):
    """Async generator: yields NDJSON lines, one per query, then a _done sentinel."""
    from bears.orchestrator.graph import run_orchestrated_rag

    exp = _to_core_experiment(request.experiment)
    exp_dict = request.experiment.model_dump() if request.experiment else {
        "model": "gpt-4o-mini", "temperature": 0.0, "top_k": 5, "agent": "agentic"
    }

    queries = request.queries
    if request.limit is not None:
        queries = queries[: request.limit]
    total = len(queries)

    RESULTS_DIR.mkdir(exist_ok=True)
    all_results: list = []

    for i, q in enumerate(queries):
        wall_start = time.time()
        try:
            result = await run_orchestrated_rag(q.question, experiment=exp)
            entry = {
                "question": q.question,
                "answer": result.get("answer", ""),
                "context": result.get("context", []),
                "true_answer": q.gold_answer,
                "source_dataset": q.source_dataset or "",
                "question_type": q.question_type or "",
                "retrieval_time": result.get("retrieval_time", 0.0),
                "generation_time": result.get("generation_time", 0.0),
                "total_time": time.time() - wall_start,
                "tool_used": result.get("tools_used", []),
            }
        except Exception as exc:
            logger.error(f"Batch evaluate error on item {i}: {exc}")
            entry = {
                "question": q.question,
                "answer": f"ERROR: {exc}",
                "context": [],
                "true_answer": q.gold_answer,
                "source_dataset": q.source_dataset or "",
                "question_type": q.question_type or "",
                "retrieval_time": 0.0,
                "generation_time": 0.0,
                "total_time": time.time() - wall_start,
                "tool_used": [],
                "error": True,
            }

        all_results.append(entry)

        # Stream entry + progress metadata to frontend
        streamed = {**entry, "_progress": {"current": i + 1, "total": total}}
        yield json.dumps(streamed, ensure_ascii=False) + "\n"

    # Persist to output/ folder — experiment_config once at top level
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    out_path.write_text(
        json.dumps({"experiment_config": exp_dict, "results": all_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Batch evaluation saved to {out_path}")

    yield json.dumps({"_done": True, "output_file": str(out_path), "count": total, "experiment_config": exp_dict}, ensure_ascii=False) + "\n"


@router.post("/evaluate")
async def evaluate_batch(request: EvaluateBatchRequest):
    """Stream batch retrieval results as NDJSON; saves final JSON to output/ folder."""
    return StreamingResponse(
        _stream_batch(request),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ── /history ──────────────────────────────────────────────────────────────────

@router.get("/history")
async def list_history():
    """List saved batch evaluation result files (newest first)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    files = sorted(RESULTS_DIR.glob("eval_*.json"), reverse=True)
    out = []
    for f in files[:50]:
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            # Support both new {"experiment_config":…,"results":[…]} and old flat-array format
            if isinstance(raw, dict):
                exp_cfg = raw.get("experiment_config") or {}
                results = raw.get("results", [])
            else:
                exp_cfg = raw[0].get("experiment_config", {}) if raw else {}
                results = raw
            datasets = list({r.get("source_dataset", "") for r in results if r.get("source_dataset")})
            avg_time = (sum(r.get("total_time", 0) for r in results) / len(results)) if results else 0
            out.append({
                "filename": f.name,
                "count": len(results),
                "datasets": sorted(datasets),
                "avg_time": round(avg_time, 1),
                "model": exp_cfg.get("model", ""),
                "experiment_config": exp_cfg,
            })
        except Exception:
            out.append({"filename": f.name, "count": 0, "datasets": [], "avg_time": 0})
    return out


@router.get("/history/{filename}")
async def get_history_file(filename: str):
    """Fetch the contents of a saved evaluation result file."""
    if ".." in filename or "/" in filename or not filename.startswith("eval_") or not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Normalise to {experiment_config, results} for the frontend
    if isinstance(raw, list):
        exp_cfg = raw[0].get("experiment_config") if raw else None
        return {"experiment_config": exp_cfg, "results": raw}
    return raw


# ── /health ───────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {"status": "ok"}
