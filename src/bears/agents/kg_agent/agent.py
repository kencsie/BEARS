"""
Knowledge Graph RAG Agent.

Ported from archive/GraphRag_hybrid_1/app/services/rag/graph_rag.py.
Wraps the 5-node pipeline logic into a BaseRAGAgent:
  1. query_expansion -> 2. vector retrieval + graph expansion ->
  3. LLM rerank -> 4. graph retrieval -> 5. answer generation
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Set

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from bears.agents.base import AgentCapability, AgentResponse, BaseRAGAgent
from bears.agents.kg_agent.retrievers import GraphRetriever, VectorRetriever
from bears.core.config import get_settings
from bears.core.experiment import ExperimentConfig
from bears.database.graph.graph_store import GraphStoreManager
from bears.database.vector.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class KGAgent(BaseRAGAgent):
    """Knowledge-graph augmented RAG agent.

    Implements the full 5-node pipeline from the archive graph_rag.py.
    """

    def __init__(self, experiment: Optional[ExperimentConfig] = None):
        self.exp = experiment or ExperimentConfig()
        settings = get_settings()

        self._vector_store = VectorStoreManager()
        self._graph_store = GraphStoreManager()
        self._vector_retriever = VectorRetriever(self._vector_store)
        self._graph_retriever = GraphRetriever(self._graph_store, llm_model=self.exp.model)
        self._llm = ChatOpenAI(
            model=self.exp.model,
            temperature=self.exp.temperature,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    @property
    def name(self) -> str:
        return "kg"

    @property
    def capabilities(self) -> Set[AgentCapability]:
        return {AgentCapability.VECTOR_SEARCH, AgentCapability.GRAPH_SEARCH, AgentCapability.MULTI_HOP}

    # ---- Node 1: Query Expansion ----

    def _query_expansion(self, question: str) -> List[str]:
        """Generate 3 query variants for multi-query retrieval."""
        try:
            expansion_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a query optimization expert. Generate 3 search query variants "
                           "for the given question. Each on a new line, no numbering. "
                           "Include synonyms, entity aliases, and intermediate reasoning steps."),
                ("human", "Question: {question}"),
            ])
            chain = expansion_prompt | self._llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            response = chain.invoke(
                {"question": question},
                config={"callbacks": callbacks} if callbacks else {},
            )

            expanded = [question]
            for line in response.content.strip().split("\n"):
                line = line.strip()
                if line and line not in expanded:
                    expanded.append(line)
            return expanded[:3]
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return [question]

    # ---- Node 2: Multi-query vector retrieval + graph expansion ----

    def _retrieve_vector_with_graph_expansion(self, expanded_queries: List[str]) -> List[Dict[str, Any]]:
        """Retrieve candidates using multi-query vector search + graph entity expansion."""
        all_candidates = []
        seen_ids: Set[str] = set()
        initial_entities: Set[str] = set()

        for query in expanded_queries:
            docs_with_meta = self._vector_retriever.retrieve_with_metadata(query, k=30)
            for doc_meta in docs_with_meta:
                doc_id = doc_meta.get("metadata", {}).get("doc_id")
                if doc_id and doc_id not in seen_ids:
                    all_candidates.append({
                        "doc_id": doc_id,
                        "content": doc_meta.get("content", ""),
                        "metadata": doc_meta.get("metadata", {}),
                    })
                    seen_ids.add(doc_id)

                    # Collect entities from content for graph expansion
                    content = doc_meta.get("content", "")
                    if content:
                        chinese_names = re.findall(r'[\u4e00-\u9fa5]{2,4}', content[:800])
                        english_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b', content[:800])
                        potential = list(set(chinese_names[:5] + english_names[:5]))
                        initial_entities.update(potential[:3])

        # Graph expansion
        if initial_entities:
            try:
                related = asyncio.run(
                    self._graph_retriever.get_related_entities(list(initial_entities)[:5], max_neighbors=3)
                )
                for entity in related[:5]:
                    entity_docs = self._vector_retriever.retrieve_with_metadata(entity, k=2)
                    for doc_meta in entity_docs:
                        doc_id = doc_meta.get("metadata", {}).get("doc_id")
                        if doc_id and doc_id not in seen_ids:
                            all_candidates.append({
                                "doc_id": doc_id,
                                "content": doc_meta.get("content", ""),
                                "metadata": doc_meta.get("metadata", {}),
                            })
                            seen_ids.add(doc_id)
            except Exception as e:
                logger.warning(f"Graph expansion failed: {e}")

        return all_candidates[:30]

    # ---- Node 3: LLM Rerank ----

    def _rerank(self, question: str, candidates: List[Dict]) -> tuple[List[str], List[str]]:
        """LLM listwise reranking. Returns (reranked_contents, reranked_ids)."""
        if not candidates:
            return [], []

        try:
            candidate_texts = []
            for idx, cand in enumerate(candidates[:30], 1):
                preview = cand["content"][:300]
                candidate_texts.append(f"[{idx}] {preview}...")

            candidates_str = "\n\n".join(candidate_texts)

            rerank_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a document ranking expert. Given a question, select the most relevant "
                           "5 documents from the candidates. Return only document numbers (e.g. 1,3,5,8,12), "
                           "comma-separated, no explanations."),
                ("human", "Question: {question}\n\nCandidates:\n{candidates}"),
            ])
            chain = rerank_prompt | self._llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            response = chain.invoke(
                {"question": question, "candidates": candidates_str},
                config={"callbacks": callbacks} if callbacks else {},
            )

            found_numbers = re.findall(r'\d+', response.content)
            selected_indices = []
            for num_str in found_numbers:
                idx = int(num_str)
                if 1 <= idx <= len(candidates):
                    selected_indices.append(idx - 1)

            reranked_contents = []
            reranked_ids = []
            for idx in selected_indices[:5]:
                if 0 <= idx < len(candidates):
                    reranked_contents.append(candidates[idx]["content"])
                    doc_id = candidates[idx].get("doc_id")
                    if doc_id:
                        reranked_ids.append(doc_id)

            if not reranked_contents and candidates:
                reranked_contents = [c["content"] for c in candidates[:5]]
                reranked_ids = [c.get("doc_id") for c in candidates[:5] if c.get("doc_id")]

            return reranked_contents, reranked_ids

        except Exception as e:
            logger.error(f"Rerank error: {e}")
            contents = [c["content"] for c in candidates[:5]]
            ids = [c.get("doc_id") for c in candidates[:5] if c.get("doc_id")]
            return contents, ids

    # ---- Node 4: Graph Retrieval ----

    async def _retrieve_graph(self, question: str) -> List[str]:
        """Retrieve graph context (entity relationships)."""
        try:
            return await self._graph_retriever.retrieve(question, max_entities=3, max_relations_per_entity=10)
        except Exception as e:
            logger.warning(f"Graph retrieval error: {e}")
            return []

    # ---- Node 5: Answer Generation ----

    def _generate_answer(self, question: str, vector_context: List[str], graph_context: List[str]) -> str:
        """Generate final answer from combined vector and graph contexts."""
        context_parts = []
        if vector_context:
            context_parts.append("【Vector Context】\n" + "\n\n".join(vector_context))
        if graph_context:
            context_parts.append("【Graph Context】\n" + "\n".join(graph_context))
        context_str = "\n\n".join(context_parts) if context_parts else "No relevant context found."

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional QA assistant skilled at integrating information from "
                       "multiple documents and performing multi-step logical reasoning.\n\n"
                       "Output format:\n<reasoning>\nYour step-by-step reasoning...\n</reasoning>\n"
                       "<answer>\nYour final answer (concise and direct)\n</answer>"),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nPlease respond in the specified format:"),
        ])

        try:
            chain = prompt | self._llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            response = chain.invoke(
                {"context": context_str, "question": question},
                config={"callbacks": callbacks} if callbacks else {},
            )

            content = response.content
            match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
            if match:
                return match.group(1).strip()

            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", content, re.DOTALL)
            if reasoning_match:
                return content.replace(reasoning_match.group(0), "").strip()

            return content
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return f"Error generating answer: {e}"

    # ---- Main entry point ----

    async def run(self, question: str, experiment: Optional[ExperimentConfig] = None) -> AgentResponse:
        exp = experiment or self.exp

        try:
            # Node 1: Query expansion
            expanded = self._query_expansion(question)

            # Node 2: Vector retrieval + graph expansion
            candidates = self._retrieve_vector_with_graph_expansion(expanded)

            # Node 3: Rerank
            vector_context, retrieved_ids = self._rerank(question, candidates)

            # Node 4: Graph retrieval
            graph_context = await self._retrieve_graph(question)

            # Node 5: Generate answer
            answer = self._generate_answer(question, vector_context, graph_context)

            confidence = min(1.0, len(retrieved_ids) / max(exp.top_k, 1))

            return AgentResponse(
                answer=answer,
                retrieved_doc_ids=retrieved_ids,
                context=vector_context + graph_context,
                confidence=confidence,
                metadata={
                    "agent": "kg",
                    "num_candidates": len(candidates),
                    "graph_relations": len(graph_context),
                },
            )
        except Exception as e:
            logger.error(f"KGAgent failed: {e}")
            return AgentResponse(
                answer=f"KG agent error: {e}",
                confidence=0.0,
                metadata={"error": str(e)},
            )
