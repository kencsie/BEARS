"""
Evaluators for agents and orchestrator.

AgentEvaluator: evaluates a single agent independently.
OrchestratorEvaluator: evaluates the full orchestrator pipeline end-to-end.

Refactored from archive/GraphRag_hybrid_1/app/services/evaluation/evaluator.py.
"""

import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from bears.agents.base import BaseRAGAgent
from bears.core.config import get_settings
from bears.core.experiment import ExperimentConfig
from bears.evaluation import schemas
from bears.evaluation.metrics import calculate_retrieval_metrics, compute_final_metrics

logger = logging.getLogger(__name__)


def _default_stats() -> Dict[str, Any]:
    return {
        "total": 0,
        "hit_count": 0,
        "found_sum": 0,
        "gold_sum": 0,
        "rr_sum": 0.0,
        "ap_sum": 0.0,
        "generation_pass": 0,
        "retrieval_time_sum": 0.0,
        "generation_time_sum": 0.0,
        "total_time_sum": 0.0,
    }


class AgentEvaluator:
    """Evaluates a single agent independently.

    Calls agent.run(question) and computes retrieval metrics + LLM-as-Judge.
    """

    def __init__(self, agent: BaseRAGAgent, experiment: Optional[ExperimentConfig] = None):
        self.agent = agent
        self.exp = experiment or ExperimentConfig()
        settings = get_settings()
        self._llm = ChatOpenAI(
            model=self.exp.model,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    def _load_queries(self, queries_path: str) -> List[Dict[str, Any]]:
        try:
            with open(queries_path, "r", encoding="utf-8") as f:
                queries = json.load(f)
            logger.info(f"Loaded queries: {len(queries)} questions")
            return queries
        except FileNotFoundError:
            raise FileNotFoundError(f"queries.json not found: {queries_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"queries.json format error: {e}")

    async def _judge_answer(self, question: str, gold_answer: str, model_answer: str) -> bool:
        """LLM-as-a-Judge: check if model answer matches gold answer semantically."""
        try:
            judge_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a fair judge. Determine if the model answer is semantically consistent with the gold answer.

Pass criteria:
- Core facts match
- Different phrasing is OK
- Translation differences are OK
- Additional supporting details are OK
- Number format differences are OK

Only Fail if core facts are clearly wrong or completely irrelevant.

Answer "Pass" or "Fail"."""),
                ("human", "Question: {question}\nGold Answer: {gold_answer}\nModel Answer: {model_answer}\n\nJudgment (Pass/Fail):"),
            ])

            chain = judge_prompt | self._llm
            result = chain.invoke({
                "question": question,
                "gold_answer": gold_answer,
                "model_answer": model_answer,
            })
            return "pass" in result.content.lower()
        except Exception as e:
            logger.error(f"LLM-as-Judge error: {e}")
            return False

    async def evaluate(self, queries_path: str, limit: int = None) -> Dict[str, Any]:
        """Run evaluation on all queries and return aggregated metrics."""
        queries = self._load_queries(queries_path)
        queries_to_eval = queries[:limit] if limit else queries
        total_queries = len(queries_to_eval)

        logger.info(f"Evaluating {total_queries} questions with agent '{self.agent.name}'")

        stats_by_source = defaultdict(_default_stats)
        stats_by_type = defaultdict(_default_stats)

        for idx, query in enumerate(queries_to_eval, 1):
            question = query.get("question")
            gold_answer = query.get("gold_answer")
            gold_doc_ids = set(query.get("gold_doc_ids", []))
            source_dataset = query.get("source_dataset", "unknown")
            question_type = query.get("question_type", "unknown")

            logger.info(f"[{idx}/{total_queries}] {question[:50]}...")

            try:
                start = time.time()
                result = await self.agent.run(question, self.exp)
                total_time = time.time() - start

                retrieved_doc_ids = result.retrieved_doc_ids
                model_answer = result.answer

                retrieval_metrics = calculate_retrieval_metrics(retrieved_doc_ids, gold_doc_ids)
                is_pass = await self._judge_answer(question, gold_answer, model_answer)

                for stats in [stats_by_source[source_dataset], stats_by_type[question_type]]:
                    stats["total"] += 1
                    stats["hit_count"] += retrieval_metrics["hit"]
                    stats["found_sum"] += retrieval_metrics["found_count"]
                    stats["gold_sum"] += len(gold_doc_ids)
                    stats["rr_sum"] += retrieval_metrics["avg_rr"]
                    stats["ap_sum"] += retrieval_metrics["ap"]
                    stats["generation_pass"] += (1 if is_pass else 0)
                    stats["total_time_sum"] += total_time

            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                continue

        return compute_final_metrics(stats_by_source, stats_by_type, total_queries)

    async def evaluate_detailed(self, queries_path: str, limit: int = None) -> schemas.DetailedEvaluateResponse:
        """Run detailed evaluation returning per-question results."""
        queries = self._load_queries(queries_path)
        queries_to_eval = queries[:limit] if limit else queries
        total_queries = len(queries_to_eval)

        stats_by_source = defaultdict(_default_stats)
        stats_by_type = defaultdict(_default_stats)
        question_details = []

        for idx, query in enumerate(queries_to_eval, 1):
            question = query.get("question")
            gold_answer = query.get("gold_answer")
            gold_doc_ids = query.get("gold_doc_ids", [])
            gold_doc_ids_set = set(gold_doc_ids)
            source_dataset = query.get("source_dataset", "unknown")
            question_type = query.get("question_type", "unknown")

            logger.info(f"[{idx}/{total_queries}] {question[:50]}...")

            try:
                start = time.time()
                result = await self.agent.run(question, self.exp)
                total_time = time.time() - start

                retrieved_doc_ids = result.retrieved_doc_ids
                model_answer = result.answer

                retrieval_metrics = calculate_retrieval_metrics(retrieved_doc_ids, gold_doc_ids_set)
                is_pass = await self._judge_answer(question, gold_answer, model_answer)

                for stats in [stats_by_source[source_dataset], stats_by_type[question_type]]:
                    stats["total"] += 1
                    stats["hit_count"] += retrieval_metrics["hit"]
                    stats["found_sum"] += retrieval_metrics["found_count"]
                    stats["gold_sum"] += len(gold_doc_ids_set)
                    stats["rr_sum"] += retrieval_metrics["avg_rr"]
                    stats["ap_sum"] += retrieval_metrics["ap"]
                    stats["generation_pass"] += (1 if is_pass else 0)
                    stats["total_time_sum"] += total_time

                question_details.append({
                    "question_id": query.get("question_id", ""),
                    "question": question,
                    "gold_answer": gold_answer,
                    "model_answer": model_answer,
                    "gold_doc_ids": gold_doc_ids,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "hit": bool(retrieval_metrics["hit"]),
                    "found_count": retrieval_metrics["found_count"],
                    "mrr": retrieval_metrics["avg_rr"],
                    "ap": retrieval_metrics["ap"],
                    "judge_pass": is_pass,
                    "source_dataset": source_dataset,
                    "total_time": total_time,
                })

            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                continue

        final = compute_final_metrics(stats_by_source, stats_by_type, total_queries)

        return schemas.DetailedEvaluateResponse(
            overall=schemas.SourceMetrics(**final["overall"]),
            by_source={k: schemas.SourceMetrics(**v) for k, v in final["by_source"].items()},
            by_question_type={k: schemas.SourceMetrics(**v) for k, v in final.get("by_question_type", {}).items()},
            questions=[schemas.QuestionDetail(**q) for q in question_details],
        )


class OrchestratorEvaluator:
    """Evaluates the full orchestrator pipeline end-to-end."""

    def __init__(self, experiment: Optional[ExperimentConfig] = None):
        self.exp = experiment or ExperimentConfig()
        settings = get_settings()
        self._llm = ChatOpenAI(
            model=self.exp.model,
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    async def _judge_answer(self, question: str, gold_answer: str, model_answer: str) -> bool:
        try:
            judge_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a fair judge. Determine if the model answer matches the gold answer. "
                           "Answer 'Pass' or 'Fail'."),
                ("human", "Question: {question}\nGold: {gold_answer}\nModel: {model_answer}\n\nJudgment:"),
            ])
            chain = judge_prompt | self._llm
            result = chain.invoke({
                "question": question,
                "gold_answer": gold_answer,
                "model_answer": model_answer,
            })
            return "pass" in result.content.lower()
        except Exception:
            return False

    async def evaluate(self, queries_path: str, limit: int = None) -> Dict[str, Any]:
        """Evaluate orchestrator end-to-end."""
        from bears.orchestrator.graph import run_orchestrated_rag

        with open(queries_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        queries_to_eval = queries[:limit] if limit else queries

        stats_by_source = defaultdict(_default_stats)
        stats_by_type = defaultdict(_default_stats)

        for idx, query in enumerate(queries_to_eval, 1):
            question = query.get("question")
            gold_answer = query.get("gold_answer")
            gold_doc_ids = set(query.get("gold_doc_ids", []))
            source_dataset = query.get("source_dataset", "unknown")
            question_type = query.get("question_type", "unknown")

            try:
                rag_result = await run_orchestrated_rag(question)
                model_answer = rag_result.get("answer", "")
                retrieved_doc_ids = rag_result.get("retrieved_doc_ids", [])
                total_time = rag_result.get("total_time", 0.0)

                retrieval_metrics = calculate_retrieval_metrics(retrieved_doc_ids, gold_doc_ids)
                is_pass = await self._judge_answer(question, gold_answer, model_answer)

                for stats in [stats_by_source[source_dataset], stats_by_type[question_type]]:
                    stats["total"] += 1
                    stats["hit_count"] += retrieval_metrics["hit"]
                    stats["found_sum"] += retrieval_metrics["found_count"]
                    stats["gold_sum"] += len(gold_doc_ids)
                    stats["rr_sum"] += retrieval_metrics["avg_rr"]
                    stats["ap_sum"] += retrieval_metrics["ap"]
                    stats["generation_pass"] += (1 if is_pass else 0)
                    stats["total_time_sum"] += total_time

            except Exception as e:
                logger.error(f"Orchestrator evaluation error: {e}")
                continue

        return compute_final_metrics(stats_by_source, stats_by_type, len(queries_to_eval))
