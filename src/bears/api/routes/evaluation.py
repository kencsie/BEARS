"""
Evaluation API endpoints.

Provides endpoints to start evaluations, track progress, and retrieve results.
Evaluations run as background tasks since they can take several minutes.
"""

import asyncio
import json
import os
import uuid
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException

from bears.api.schemas import (
    EvalStartRequest,
    EvalStartResponse,
    EvalStatusResponse,
    EvalResultResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eval", tags=["evaluation"])

# In-memory task store (keyed by task_id)
_tasks: Dict[str, Dict[str, Any]] = {}


# --- Background evaluation runner ---


async def _run_evaluation(
    task_id: str,
    agent_name: Optional[str],
    use_orchestrator: bool,
    limit: Optional[int],
    config_path: Optional[str],
    detailed: bool,
    failures_only: bool,
    output_filename: Optional[str],
):
    """Run evaluation in background and update task progress."""
    task = _tasks[task_id]

    try:
        from bears.core.experiment import ExperimentConfig

        experiment = (
            ExperimentConfig.from_yaml(config_path)
            if config_path
            else ExperimentConfig()
        )

        # Load queries to get total count
        queries_path = "data/queries.json"
        with open(queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        queries_to_eval = queries[:limit] if limit else queries
        total = len(queries_to_eval)

        task["total"] = total
        task["status"] = "running"

        label = "orchestrator" if use_orchestrator else agent_name

        if use_orchestrator:
            from bears.evaluation.evaluator import OrchestratorEvaluator

            evaluator = OrchestratorEvaluator(experiment)
            task["message"] = "Running orchestrator evaluation..."

            results = await evaluator.evaluate(queries_path, limit=limit)
            task["progress"] = total

        else:
            from bears.agents.registry import get_agent
            from bears.evaluation.evaluator import AgentEvaluator

            agent = get_agent(agent_name, experiment)
            evaluator = AgentEvaluator(agent, experiment)
            task["message"] = f"Running {agent_name} evaluation..."

            if detailed:
                detailed_result = await evaluator.evaluate_detailed(
                    queries_path, limit=limit
                )
                results = detailed_result.model_dump()

                # Filter failures only if requested
                if failures_only and "questions" in results:
                    results["questions"] = [
                        q for q in results["questions"] if not q.get("judge_pass")
                    ]
            else:
                results = await evaluator.evaluate(queries_path, limit=limit)

            task["progress"] = total

        # Save results to output/
        os.makedirs("output", exist_ok=True)
        if output_filename:
            # Ensure .json extension
            if not output_filename.endswith(".json"):
                output_filename += ".json"
            output_path = f"output/{output_filename}"
        else:
            output_path = f"output/{label}_{task_id[:8]}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        task["status"] = "completed"
        task["results"] = results
        task["output_path"] = output_path
        task["message"] = f"Evaluation completed. Results saved to {output_path}"

    except Exception as e:
        logger.error(f"Evaluation task {task_id} failed: {e}")
        task["status"] = "failed"
        task["error"] = str(e)
        task["message"] = f"Evaluation failed: {e}"


# --- Endpoints ---


@router.post("/start", response_model=EvalStartResponse)
async def start_evaluation(request: EvalStartRequest):
    """Start an evaluation task in the background."""
    if not request.agent and not request.orchestrator:
        raise HTTPException(
            status_code=400,
            detail="Must specify either 'agent' or 'orchestrator: true'",
        )

    if request.agent and request.orchestrator:
        raise HTTPException(
            status_code=400, detail="Cannot specify both 'agent' and 'orchestrator'"
        )

    # Validate agent name
    if request.agent:
        from bears.agents.registry import AGENT_REGISTRY

        if request.agent not in AGENT_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown agent: {request.agent}. Available: {list(AGENT_REGISTRY.keys())}",
            )
        if not AGENT_REGISTRY[request.agent].get("enabled"):
            raise HTTPException(
                status_code=400, detail=f"Agent '{request.agent}' is disabled"
            )

    task_id = str(uuid.uuid4())
    label = "orchestrator" if request.orchestrator else request.agent

    _tasks[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "total": 0,
        "message": f"Starting {label} evaluation...",
        "agent": request.agent,
        "orchestrator": request.orchestrator,
        "results": None,
        "error": None,
    }

    # Launch background task
    asyncio.create_task(
        _run_evaluation(
            task_id,
            request.agent,
            request.orchestrator,
            request.limit,
            request.config_path,
            request.detailed,
            request.failures_only,
            request.output_filename,
        )
    )

    return EvalStartResponse(
        task_id=task_id,
        status="pending",
        message=f"Evaluation task started for {label}",
    )


@router.get("/status/{task_id}", response_model=EvalStatusResponse)
async def get_eval_status(task_id: str):
    """Get evaluation task progress."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = _tasks[task_id]
    return EvalStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        total=task["total"],
        message=task["message"],
    )


@router.get("/results/{task_id}", response_model=EvalResultResponse)
async def get_eval_results(task_id: str):
    """Get evaluation results for a completed task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = _tasks[task_id]
    if task["status"] == "running":
        raise HTTPException(status_code=202, detail="Evaluation is still running")

    return EvalResultResponse(
        task_id=task_id,
        status=task["status"],
        results=task.get("results"),
        error=task.get("error"),
    )


@router.get("/history")
async def get_eval_history():
    """List past evaluation result files from output/ directory."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        return {"results": []}

    files = []
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(output_dir, filename)
            stat = os.stat(filepath)
            files.append(
                {
                    "filename": filename,
                    "path": filepath,
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )

    files.sort(key=lambda x: x["modified"], reverse=True)
    return {"results": files}


@router.get("/history/{filename}")
async def get_eval_history_file(filename: str):
    """Load a specific evaluation result file."""
    filepath = os.path.join("output", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


@router.get("/agents")
async def get_eval_agents():
    """List agents available for evaluation with their configuration."""
    from bears.agents.registry import AGENT_REGISTRY

    agents = []
    for name, info in AGENT_REGISTRY.items():
        agents.append(
            {
                "name": name,
                "enabled": info.get("enabled", False),
                "module": info.get("module", ""),
            }
        )

    return {"agents": agents}


@router.get("/queries/stats")
async def get_queries_stats():
    """Get statistics about the evaluation question set."""
    queries_path = "data/queries.json"
    if not os.path.exists(queries_path):
        raise HTTPException(status_code=404, detail="queries.json not found")

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    by_source: Dict[str, int] = {}
    by_type: Dict[str, int] = {}

    for q in queries:
        source = q.get("source_dataset", "unknown")
        qtype = q.get("question_type", "unknown")
        by_source[source] = by_source.get(source, 0) + 1
        by_type[qtype] = by_type.get(qtype, 0) + 1

    return {
        "total": len(queries),
        "by_source": by_source,
        "by_question_type": by_type,
    }
