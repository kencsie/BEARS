"""API request / response schemas."""

from typing import List, Optional

from pydantic import BaseModel


# ── Experiment config (defined first so other schemas can reference it) ───────

class ExperimentConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    top_k: int = 5
    agent: str = "agentic"


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrieveRequest(BaseModel):
    question: str
    true_answer: Optional[str] = None   # gold answer for offline evaluation
    true_context: Optional[List[str]] = None  # gold context for offline evaluation
    experiment: Optional[ExperimentConfig] = None


class RetrieveResponse(BaseModel):
    """Standard retrieval output — compatible with external evaluation systems.

    Fields
    ------
    question      : the original user question
    answer        : the system-generated answer
    context       : list of retrieved context chunks (reranked Top-5)
    true_answer   : gold answer (echoed from request if provided)
    true_context  : gold context (echoed from request if provided)
    retrieval_time   : seconds spent on retrieval (agentic loop)
    generation_time  : seconds spent on final LLM generation
    total_time       : wall-clock seconds for the full pipeline
    prompt_tokens    : total input tokens consumed across all LLM calls
    completion_tokens: total output tokens generated across all LLM calls
    total_tokens     : prompt_tokens + completion_tokens
    """
    question: str
    answer: str
    context: List[str]
    true_answer: Optional[str] = None
    true_context: Optional[List[str]] = None
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_used: List[str] = []
    experiment_config: Optional[dict] = None


# ── Generation (chatbot) ──────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    question: str
    generator: str = "educational"  # name of the generator to use


class GenerateResponse(BaseModel):
    question: str
    generated_content: str          # output from the domain-specific generator
    context: List[str]              # retrieved context passed to generator
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ── Batch Evaluation ─────────────────────────────────────────────────────────

class QueryItem(BaseModel):
    question_id: Optional[str] = None
    question: str
    gold_answer: Optional[str] = None
    gold_doc_ids: Optional[List[str]] = None
    source_dataset: Optional[str] = None
    question_type: Optional[str] = None


class EvaluateBatchRequest(BaseModel):
    queries: List[QueryItem]
    limit: Optional[int] = None  # None = run all
    experiment: Optional[ExperimentConfig] = None


class ExperimentCreateRequest(BaseModel):
    name: str
    config: ExperimentConfig


class ExperimentUpdateRequest(BaseModel):
    config: ExperimentConfig
