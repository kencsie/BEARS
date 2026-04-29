"""
Agentic RAG Agent — Single-pass pipeline.

Parallel vector + BM25 retrieval, cross-encoder reranking, and Final LLM
synthesis run in a single pass — no multi-turn LLM coordination loop.
"""

import asyncio
import logging
import time
from typing import List, Optional, Set

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from bears.agents.agentic_agent.prompts import FINAL_GENERATION_PROMPT
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

    async def _generate_final_answer(self, question: str, context_parts: List[str]) -> str:
        if not context_parts:
            return "資料不足以回答。"

        combined = "\n\n---\n\n".join(context_parts)
        try:
            chain = (
                ChatPromptTemplate.from_messages([
                    ("system", FINAL_GENERATION_PROMPT),
                    ("human", "【參考文件】\n{context}\n\n【問題】\n{question}"),
                ])
                | self._final_llm
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

            # Phase 1: Parallel retrieval (vector + BM25 concurrently)
            pool = await _parallel_retrieve(
                question, use_vector=True, use_keyword=True, use_graph=False
            )

            # Phase 2: Cross-encoder reranking (off event loop)
            reranker = get_reranker()
            top_chunks = await asyncio.to_thread(
                reranker.rerank_with_scores, question, pool, top_k=5
            )

            retrieval_time = time.time() - run_start

            context_parts = [_format_chunks(top_chunks)] if top_chunks else []

            # Phase 3: Final LLM synthesis
            gen_start = time.time()
            answer = await self._generate_final_answer(question, context_parts)
            generation_time = time.time() - gen_start

            confidence = min(1.0, len(top_chunks) / 5.0) if top_chunks else 0.0
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
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "tools_used": ["keyword", "vector"],
                },
            )

        except Exception as e:
            logger.error(f"AgenticAgent failed: {e}")
            return AgentResponse(
                answer=f"Agentic agent error: {e}",
                confidence=0.0,
                metadata={"error": str(e)},
            )
