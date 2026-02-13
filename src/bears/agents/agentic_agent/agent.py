"""
Agentic RAG Agent.

Ported from archive/AgenticFlow/run_agent.py.
Multi-step iterative retrieval with LLM reranking, grading, and reasoning.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from bears.agents.agentic_agent.prompts import (
    GENERATE_SYSTEM_PROMPT,
    LLM_GRADE_PROMPT,
    LLM_RERANK_PROMPT,
    STEP_REASONER_PROMPT,
)
from bears.agents.base import AgentCapability, AgentResponse, BaseRAGAgent
from bears.core.config import get_settings
from bears.core.experiment import ExperimentConfig
from bears.database.vector.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class AgenticAgent(BaseRAGAgent):
    """Multi-step agentic retrieval agent.

    Ported from archive/AgenticFlow/run_agent.py.
    Per-step: retrieve -> LLM rerank -> merge evidence -> reason next step.
    Final: distance + LLM grade fusion -> generate answer.
    """

    def __init__(self, experiment: Optional[ExperimentConfig] = None):
        self.exp = experiment or ExperimentConfig()
        settings = get_settings()

        self._vector_store = VectorStoreManager()
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = self.exp.model

    @property
    def name(self) -> str:
        return "agentic"

    @property
    def capabilities(self) -> Set[AgentCapability]:
        return {AgentCapability.VECTOR_SEARCH, AgentCapability.MULTI_HOP}

    # ---- Utilities ----

    @staticmethod
    def _get_doc_id(doc) -> str:
        try:
            doc_id = doc.metadata.get("doc_id", None)
            if doc_id:
                return str(doc_id)
        except Exception:
            pass
        try:
            return f"hash::{hash(doc.page_content)}"
        except Exception:
            return "hash::unknown"

    @staticmethod
    def _get_distance_score(doc) -> float:
        try:
            return float(doc.metadata.get("score", 1e9))
        except Exception:
            return 1e9

    # ---- LLM: Per-step rerank ----

    def _llm_rerank_indices(self, query: str, docs: list, top_k: int, max_chars_per_doc: int = 2000) -> List[int]:
        """LLM-based reranking, returns indices in descending relevance."""
        if not docs:
            return list(range(min(top_k, len(docs))))

        docs_text = ""
        for i, doc in enumerate(docs):
            content = (getattr(doc, "page_content", "") or "").replace("\n", " ").strip()
            if len(content) > max_chars_per_doc:
                content = content[:max_chars_per_doc] + "..."
            docs_text += f"文件 [{i}]: {content}\n\n"

        prompt = LLM_RERANK_PROMPT.format(
            query=query, num_docs=len(docs), docs_list=docs_text, top_k=top_k
        )

        schema = {
            "name": "rerank_indices",
            "schema": {
                "type": "object",
                "properties": {
                    "indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "最相關文件的索引列表（由高到低）",
                    }
                },
                "required": ["indices"],
                "additionalProperties": False,
            },
        }

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "你只輸出 JSON，符合 schema：{indices:[...]}。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_schema", "json_schema": schema},
            )
            content = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            indices = parsed.get("indices", [])
            if not isinstance(indices, list):
                return list(range(min(top_k, len(docs))))

            out = []
            seen = set()
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(docs) and idx not in seen:
                    out.append(idx)
                    seen.add(idx)
            return out[:top_k] if out else list(range(min(top_k, len(docs))))
        except Exception as e:
            logger.warning(f"LLM rerank failed (fallback to vector order): {e}")
            return list(range(min(top_k, len(docs))))

    # ---- LLM: Step reasoner ----

    def _reason_next_step(self, question: str, contexts: List[str], history: List[str]) -> str:
        ctx_text = "\n".join([f"[{i + 1}] {c}" for i, c in enumerate(contexts[-6:])]) or "(empty)"
        hist_text = "\n".join(history) or "(none)"

        prompt = STEP_REASONER_PROMPT.format(question=question, contexts=ctx_text, history=hist_text)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "你只輸出 DONE 或下一個檢索 query（單行）。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "").strip().splitlines()[0].strip()
            return text
        except Exception:
            return "DONE"

    # ---- LLM: Grade docs ----

    def _llm_grade_doc(self, question: str, doc_text: str, max_chars: int = 1200) -> int:
        content = (doc_text or "").strip().replace("\n", " ")[:max_chars]
        prompt = LLM_GRADE_PROMPT.format(question=question, content=content)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "你只能輸出 0,1,2,3 其中一個整數。不要輸出其他字。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            text = (resp.choices[0].message.content or "").strip().splitlines()[0].strip()
            if text and text[0] in "0123":
                return int(text[0])
            return 0
        except Exception:
            return 0

    # ---- LLM: Generate answer ----

    def _generate_answer(self, question: str, final_docs: List[Dict[str, Any]]) -> str:
        if not final_docs:
            return "資料不足以回答。"

        context_str = ""
        for i, doc in enumerate(final_docs, 1):
            context_str += f"文件 [{i}] (distance_score={doc.get('score', 0):.4f}):\n{doc.get('content_preview', '')}\n\n"

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"【參考文件】\n{context_str}\n\n【問題】\n{question}"},
                ],
                temperature=0.3,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return f"Error: {e}"

    # ---- Core multi-step runner ----

    def _run_agent_retrieval(
        self,
        question: str,
        max_steps: int = 4,
        per_step_top_k: int = 5,
        retrieve_k: int = 30,
        candidate_k: int = None,
        final_top_k: int = 8,
        rerank_alpha: float = None,
        rerank_beta: float = None,
        context_top_m: int = 3,
    ) -> Tuple[List[Dict[str, Any]], str, List[str]]:
        """Run multi-step retrieval and return (final_docs, answer, all_doc_ids)."""
        rerank_alpha = rerank_alpha if rerank_alpha is not None else self.exp.rerank_alpha
        rerank_beta = rerank_beta if rerank_beta is not None else self.exp.rerank_beta
        if candidate_k is None:
            candidate_k = max(final_top_k * 4, 20)

        evidence: Dict[str, Dict[str, Any]] = {}
        history: List[str] = []
        context_cache: List[str] = []
        all_doc_ids: List[str] = []

        current_query = question

        for step in range(1, max_steps + 1):
            # 1) Vector retrieve
            try:
                raw = self._vector_store.search(current_query, k=retrieve_k)
                candidates = raw[:retrieve_k]
            except Exception as e:
                logger.error(f"Search failed at step {step}: {e}")
                candidates = []

            # 2) Per-step LLM rerank
            if candidates:
                idxs = self._llm_rerank_indices(current_query, candidates, top_k=per_step_top_k)
                reranked = [candidates[i] for i in idxs]
            else:
                reranked = []

            # 3) Merge evidence
            for doc in candidates:
                all_doc_ids.append(self._get_doc_id(doc))

            for doc in reranked:
                doc_id = self._get_doc_id(doc)
                dist = self._get_distance_score(doc)
                if doc_id not in evidence:
                    evidence[doc_id] = {"doc": doc, "best_score": dist}
                elif dist < float(evidence[doc_id]["best_score"]):
                    evidence[doc_id]["best_score"] = dist
                    evidence[doc_id]["doc"] = doc

            # 4) Update context cache
            for doc in reranked[:min(context_top_m, len(reranked))]:
                content = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
                context_cache.append(content[:3000])

            history.append(f"Step {step}: {current_query}")

            # 5) Decide next step
            next_query = self._reason_next_step(question, context_cache, history)
            if next_query.upper() == "DONE":
                break
            current_query = next_query

        # ---- Final: evidence pool rerank (distance + LLM grade) ----
        ranked_by_dist = sorted(evidence.items(), key=lambda kv: float(kv[1]["best_score"]))
        final_candidates = ranked_by_dist[:min(candidate_k, len(ranked_by_dist))]

        if not final_candidates:
            return [], "資料不足以回答。", list(set(all_doc_ids))

        dists = [float(p["best_score"]) for _, p in final_candidates]
        d_min, d_max = min(dists), max(dists)
        eps = 1e-6

        scored = []
        for doc_id, payload in final_candidates:
            doc = payload["doc"]
            dist = float(payload["best_score"])
            dist_norm = (dist - d_min) / (d_max - d_min + eps)
            dist_good = _clamp01(1.0 - dist_norm)

            text = getattr(doc, "page_content", "") or ""
            grade = self._llm_grade_doc(question, text)
            llm_good = grade / 3.0

            final_score = rerank_alpha * dist_good + rerank_beta * llm_good
            scored.append((doc_id, payload, grade, dist_good, llm_good, float(final_score)))

        scored.sort(key=lambda x: x[5], reverse=True)
        picked = scored[:final_top_k]

        final_docs = []
        final_doc_ids = []
        for doc_id, payload, grade, dist_good, llm_good, fscore in picked:
            doc = payload["doc"]
            content = (getattr(doc, "page_content", "") or "").strip()
            final_docs.append({
                "doc_id": doc_id,
                "score": float(payload["best_score"]),
                "llm_grade": grade,
                "final_score": fscore,
                "content_preview": content,
            })
            final_doc_ids.append(doc_id)

        answer = self._generate_answer(question, final_docs)
        return final_docs, answer, final_doc_ids

    # ---- Main entry point ----

    async def run(self, question: str, experiment: Optional[ExperimentConfig] = None) -> AgentResponse:
        exp = experiment or self.exp

        try:
            final_docs, answer, doc_ids = self._run_agent_retrieval(
                question,
                max_steps=4,
                per_step_top_k=exp.top_k,
                retrieve_k=30,
                final_top_k=max(exp.top_k, 8),
                rerank_alpha=exp.rerank_alpha,
                rerank_beta=exp.rerank_beta,
            )

            contexts = [d.get("content_preview", "") for d in final_docs]
            confidence = min(1.0, len(doc_ids) / max(exp.top_k, 1)) if doc_ids else 0.0

            return AgentResponse(
                answer=answer,
                retrieved_doc_ids=doc_ids,
                context=contexts,
                confidence=confidence,
                metadata={
                    "agent": "agentic",
                    "num_final_docs": len(final_docs),
                    "rerank_alpha": exp.rerank_alpha,
                    "rerank_beta": exp.rerank_beta,
                },
            )
        except Exception as e:
            logger.error(f"AgenticAgent failed: {e}")
            return AgentResponse(
                answer=f"Agentic agent error: {e}",
                confidence=0.0,
                metadata={"error": str(e)},
            )
