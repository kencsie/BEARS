"""
Agentic RAG Agent — Single-pass pipeline.

Parallel vector + BM25 retrieval, cross-encoder reranking, and Final LLM
synthesis run in a single pass — no multi-turn LLM coordination loop.
"""

import asyncio
import json
import logging
import time
from typing import List, Optional, Set

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from bears.agents.agentic_agent.prompts import (
    FINAL_GENERATION_PROMPT,
    OPEN_ENDED_GENERATION_PROMPT,
    QUESTION_TYPE_CLASSIFIER_PROMPT,
    RETRIEVAL_PLANNER_PROMPT,
)
from bears.agents.base import AgentCapability, AgentResponse, BaseRAGAgent
from bears.core.config import get_settings
from bears.core.dependencies import get_reranker
from bears.core.experiment import ExperimentConfig
from bears.core.langfuse_helper import get_callbacks
from bears.tools.comprehensive_search import _format_chunks, _parallel_retrieve

logger = logging.getLogger(__name__)


class AgenticAgent(BaseRAGAgent):
    """Single-pass RAG agent.

    Runs parallel vector + BM25 retrieval and cross-encoder reranking in one
    shot, then synthesises the answer with a Final LLM.  Eliminates the
    multi-turn ReAct coordination loop and its extra API round-trips.
    """

    def __init__(self, experiment: Optional[ExperimentConfig] = None):
        self.exp = experiment or ExperimentConfig()
        settings = get_settings()

        self._final_llm = ChatOpenAI(
            model=self.exp.model,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
        )
        self._planner_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )

    @property
    def name(self) -> str:
        return "agentic"

    @property
    def capabilities(self) -> Set[AgentCapability]:
        return {
            AgentCapability.VECTOR_SEARCH,
            AgentCapability.GRAPH_SEARCH,
            AgentCapability.MULTI_HOP,
        }

    async def _plan_retrieval_strategy(self, question: str) -> dict:
        """Ask a lightweight LLM to decide which retrievers to activate for this query."""
        try:
            resp = await self._planner_llm.ainvoke(
                RETRIEVAL_PLANNER_PROMPT.format(question=question),
                config={"callbacks": get_callbacks()},
            )
            strategy = json.loads(resp.content.strip())
        except Exception as e:
            logger.warning(f"Retrieval planning failed, using default strategy: {e}")
            strategy = {}
        # vector is always mandatory
        strategy["use_vector"] = True
        strategy.setdefault("use_keyword", True)
        strategy.setdefault("use_graph", False)
        return strategy

    async def _classify_question_type(self, question: str) -> str:
        """Classify the question as 'factual' or 'open_ended' to pick the matching final prompt."""
        try:
            resp = await self._planner_llm.ainvoke(
                QUESTION_TYPE_CLASSIFIER_PROMPT.format(question=question),
                config={"callbacks": get_callbacks()},
            )
            qtype = json.loads(resp.content.strip()).get("type", "factual")
        except Exception as e:
            logger.warning(f"Question type classification failed, defaulting to factual: {e}")
            return "factual"
        return qtype if qtype in {"factual", "open_ended"} else "factual"

    async def _generate_final_answer(
        self,
        question: str,
        context_parts: List[str],
        llm=None,
        question_type: str = "factual",
    ) -> str:
        if not context_parts:
            return "資料不足以回答。"

        target_llm = llm or self._final_llm
        system_prompt = (
            OPEN_ENDED_GENERATION_PROMPT if question_type == "open_ended" else FINAL_GENERATION_PROMPT
        )
        combined = "\n\n---\n\n".join(context_parts)
        try:
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "【參考文件】\n{context}\n\n【問題】\n{question}"),
                ])
                | target_llm
            )
            resp = await chain.ainvoke(
                {"context": combined, "question": question},
                config={"callbacks": get_callbacks()},
            )
            return (resp.content or "").strip()
        except Exception as e:
            logger.error(f"Final LLM generation failed: {e}")
            return f"生成錯誤: {e}"

    async def run(self, question: str, experiment: Optional[ExperimentConfig] = None) -> AgentResponse:
        exp = experiment or self.exp

        try:
            run_start = time.time()
            exp = experiment or self.exp

            # Create per-request final LLM based on experiment config
            final_llm = (
                ChatOpenAI(
                    model=exp.model,
                    api_key=get_settings().OPENAI_API_KEY,
                    temperature=exp.temperature,
                )
                if experiment is not None
                else self._final_llm
            )

            # Phase 1a: Decide retrieval strategy + classify question type (in parallel)
            strategy, question_type = await asyncio.gather(
                self._plan_retrieval_strategy(question),
                self._classify_question_type(question),
            )
            logger.info(f"Retrieval strategy: {strategy} | Question type: {question_type}")

            # Phase 1b: Parallel retrieval with selected engines
            pool = await _parallel_retrieve(
                question,
                use_vector=strategy.get("use_vector", True),
                use_keyword=strategy.get("use_keyword", True),
                use_graph=strategy.get("use_graph", False),
            )

            # Phase 2: Cross-encoder reranking (off event loop)
            reranker = get_reranker()
            top_chunks = await asyncio.to_thread(
                reranker.rerank_with_scores, question, pool, top_k=exp.top_k
            )

            retrieval_time = time.time() - run_start

            # Build keyword debug: for each top chunk from keyword source, show matched chars
            keyword_hits = [
                {
                    "rank": i + 1,
                    "doc_id": c.get("doc_id", ""),
                    "matched_tokens": c.get("matched_tokens", ""),
                    "bm25_score": round(c.get("score", 0), 4),
                    "rerank_score": round(c.get("rerank_score", 0), 4),
                }
                for i, c in enumerate(top_chunks)
                if c.get("source") == "keyword"
            ]

            context_parts = [_format_chunks(top_chunks)] if top_chunks else []

            # Phase 3: Final LLM synthesis
            gen_start = time.time()
            answer = await self._generate_final_answer(
                question, context_parts, llm=final_llm, question_type=question_type
            )
            generation_time = time.time() - gen_start

            confidence = min(1.0, len(top_chunks) / exp.top_k) if top_chunks else 0.0
            doc_ids = [c.get("doc_id", "") for c in top_chunks if c.get("doc_id")]

            return AgentResponse(
                answer=answer,
                retrieved_doc_ids=doc_ids,
                context=context_parts,
                confidence=confidence,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                metadata={
                    "agent": "agentic",
                    "tool_calls": 1,
                    "model": exp.model,
                    "temperature": exp.temperature,
                    "top_k": exp.top_k,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "tools_used": [k.replace("use_", "") for k, v in strategy.items() if v],
                    "question_type": question_type,
                    "keyword_hits": keyword_hits,
                },
            )

        except Exception as e:
            logger.error(f"AgenticAgent failed: {e}")
            return AgentResponse(
                answer=f"Agentic agent error: {e}",
                confidence=0.0,
                metadata={"error": str(e)},
            )
