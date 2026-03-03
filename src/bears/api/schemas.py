"""
API request/response schemas.

All API data models are defined here for centralized management.
"""

from pydantic import BaseModel
from typing import Dict, Any, List, Optional


# --- Evaluation schemas ---


class EvalStartRequest(BaseModel):
    """Request to start an evaluation task."""

    agent: Optional[str] = None
    orchestrator: bool = False
    limit: Optional[int] = None
    config_path: Optional[str] = None
    detailed: bool = True
    failures_only: bool = False
    output_filename: Optional[str] = None


class EvalStartResponse(BaseModel):
    task_id: str
    status: str
    message: str


class EvalStatusResponse(BaseModel):
    task_id: str
    status: str  # pending / running / completed / failed
    progress: int
    total: int
    message: str


class EvalResultResponse(BaseModel):
    task_id: str
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# --- Query schemas ---


class QueryRequest(BaseModel):
    """Request for a single question query."""

    question: str
    agent: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    agent_used: str
    retrieved_doc_ids: List[str]
    confidence: float


# --- Experiment schemas ---


class ExperimentConfig(BaseModel):
    """Experiment configuration (mirrors core/experiment.py for API use)."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_k: int = 5
    rerank_alpha: float = 0.7
    rerank_beta: float = 0.3
    agent: str = "hybrid"


class ExperimentCreateRequest(BaseModel):
    name: str
    config: ExperimentConfig


class ExperimentUpdateRequest(BaseModel):
    config: ExperimentConfig
