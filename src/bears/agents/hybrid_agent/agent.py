"""
Hybrid RAG Agent.

Ported from archive/hybrid_rag/. Uses vector search via VectorStoreManager
with RRF fusion logic and LLM generation.
"""

import logging
import re
from typing import Dict, List, Optional, Set

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from bears.agents.base import AgentCapability, AgentResponse, BaseRAGAgent
from bears.core.config import get_settings
from bears.core.experiment import ExperimentConfig
from bears.database.vector.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class HybridRAGAgent(BaseRAGAgent):
    """Hybrid RAG agent: vector search + RRF fusion + LLM generation.

    Ported from archive/hybrid_rag/ retrieval_service.py + generation_service.py.
    Adapted to use VectorStoreManager (ChromaDB) instead of in-memory embeddings.
    """

    def __init__(self, experiment: Optional[ExperimentConfig] = None):
        self.exp = experiment or ExperimentConfig()
        settings = get_settings()

        self._vector_store = VectorStoreManager()
        self._llm = ChatOpenAI(
            model=self.exp.model,
            temperature=self.exp.temperature,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    @property
    def name(self) -> str:
        return "hybrid"

    @property
    def capabilities(self) -> Set[AgentCapability]:
        return {AgentCapability.VECTOR_SEARCH}

    # ---- Retrieval (ported from retrieval_service.py) ----

    def _vector_search(self, query: str, k: int = 10) -> List[Dict]:
        """Vector similarity search via ChromaDB."""
        results = self._vector_store.search_with_scores(query, k=k)
        return [
            {
                "doc_id": doc.metadata.get("doc_id", ""),
                "content": doc.page_content,
                "score": float(score),
                "original_source": doc.metadata.get("original_source", ""),
            }
            for doc, score in results
        ]

    def _multi_query_search(self, question: str, k: int = 10) -> List[Dict]:
        """Expand query into variants and search with each."""
        queries = self._expand_query(question)
        all_results: Dict[str, Dict] = {}

        for query in queries:
            results = self._vector_search(query, k=k)
            for r in results:
                doc_id = r["doc_id"]
                if doc_id not in all_results or r["score"] < all_results[doc_id]["score"]:
                    all_results[doc_id] = r

        return list(all_results.values())

    def _expand_query(self, question: str) -> List[str]:
        """Generate query variants using LLM."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Generate 2 search query variants for the given question. "
                           "Output each variant on a new line. No numbering."),
                ("human", "{question}"),
            ])
            chain = prompt | self._llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            response = chain.invoke(
                {"question": question},
                config={"callbacks": callbacks} if callbacks else {},
            )

            variants = [question]
            for line in response.content.strip().split("\n"):
                line = line.strip()
                if line and line not in variants:
                    variants.append(line)
            return variants[:3]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [question]

    @staticmethod
    def _rrf_fusion(result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion across multiple result lists.

        RRF score = sum(1 / (k + rank + 1)) for each list the doc appears in.
        """
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                doc_id = result["doc_id"]
                if doc_id not in doc_map:
                    doc_map[doc_id] = result
                score = 1 / (k + rank + 1)
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + score

        sorted_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    # ---- Generation (ported from generation_service.py) ----

    def _generate_answer(self, question: str, contexts: List[Dict]) -> str:
        """Generate answer from retrieved contexts using LLM."""
        context_str = ""
        for i, ctx in enumerate(contexts[: self.exp.top_k]):
            source = ctx.get("original_source", "unknown")
            context_str += f"[{i + 1}] (來源: {source}):\n{ctx['content']}\n\n"

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions in Traditional Chinese."),
            ("human", """你是一個專業的繁體中文問答助手。請根據以下的參考資訊回答使用者的問題。
如果參考資訊不足以回答，請說「根據提供的資訊無法回答此問題」。請勿編造事實。

參考資訊：
{context}

使用者問題：{question}
請提供完整且有條理的回答："""),
        ])

        try:
            chain = prompt | self._llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            response = chain.invoke(
                {"context": context_str, "question": question},
                config={"callbacks": callbacks} if callbacks else {},
            )
            return response.content
        except Exception as e:
            return f"Error generating answer: {e}"

    # ---- Main entry point ----

    async def run(self, question: str, experiment: Optional[ExperimentConfig] = None) -> AgentResponse:
        exp = experiment or self.exp
        top_k = exp.top_k

        try:
            # Multi-query vector search
            results = self._multi_query_search(question, k=top_k * 4)

            # RRF fusion (single list here, but extensible for keyword search)
            fused = self._rrf_fusion([results])
            top_results = fused[:top_k]

            # Extract doc IDs and contexts
            doc_ids = [r["doc_id"] for r in top_results if r.get("doc_id")]
            contexts = [r["content"] for r in top_results]

            # Generate answer
            answer = self._generate_answer(question, top_results)

            # Confidence heuristic: based on number of results found
            confidence = min(1.0, len(doc_ids) / max(top_k, 1))

            return AgentResponse(
                answer=answer,
                retrieved_doc_ids=doc_ids,
                context=contexts,
                confidence=confidence,
                metadata={"agent": "hybrid", "num_candidates": len(results)},
            )
        except Exception as e:
            logger.error(f"HybridRAGAgent failed: {e}")
            return AgentResponse(
                answer=f"Hybrid agent error: {e}",
                confidence=0.0,
                metadata={"error": str(e)},
            )
