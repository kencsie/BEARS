"""
Evaluation schemas.

Migrated from archive/GraphRag_hybrid_1/app/models/schemas.py.
Only evaluation-related schemas are kept (no API schemas).
"""

from pydantic import BaseModel
from typing import List, Dict


class SourceMetrics(BaseModel):
    """Evaluation metrics for a source group or overall."""
    total_questions: int
    hit_rate: float
    partial_hit_rate: float
    mrr: float
    map: float
    generation_pass_rate: float
    avg_retrieval_time: float = 0.0
    avg_generation_time: float = 0.0
    avg_total_time: float = 0.0


class QuestionDetail(BaseModel):
    """Per-question detailed evaluation result."""
    question_id: str
    question: str
    gold_answer: str
    model_answer: str
    gold_doc_ids: List[str]
    retrieved_doc_ids: List[str]
    hit: bool
    found_count: int
    mrr: float
    ap: float
    judge_pass: bool
    source_dataset: str
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0


class DetailedEvaluateResponse(BaseModel):
    """Full evaluation response with per-question details."""
    overall: SourceMetrics
    by_source: Dict[str, SourceMetrics]
    by_question_type: Dict[str, SourceMetrics] = {}
    questions: List[QuestionDetail]
