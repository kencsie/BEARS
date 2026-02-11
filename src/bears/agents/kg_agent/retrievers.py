"""
KG Agent internal retrievers.

Ported from archive/GraphRag_hybrid_1/app/services/retrieval/.
VectorRetriever and GraphRetriever adapted for bears imports.
"""

import logging
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from bears.core.config import get_settings
from bears.database.vector.vector_store import VectorStoreManager
from bears.database.graph.graph_store import GraphStoreManager

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Vector retriever for the KG agent."""

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        logger.info("VectorRetriever initialized")

    def retrieve(self, question: str, k: int = 5) -> List[str]:
        """Return document contents."""
        try:
            docs = self.vector_store.search(question, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []

    def retrieve_with_ids(self, question: str, k: int = 5) -> Tuple[List[str], List[str]]:
        """Return (contents, doc_ids)."""
        try:
            docs = self.vector_store.search(question, k=k)
            contents = [doc.page_content for doc in docs]
            doc_ids = [doc.metadata.get("doc_id", "") for doc in docs]
            return contents, doc_ids
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return [], []

    def retrieve_with_metadata(self, question: str, k: int = 5) -> List[dict]:
        """Return dicts with content and metadata."""
        try:
            docs = self.vector_store.search(question, k=k)
            return [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"Vector retrieval failed: {e}")
            return []


class GraphRetriever:
    """Graph retriever for the KG agent."""

    def __init__(self, graph_store: GraphStoreManager, llm_model: str = "gpt-4o-mini", temperature: float = 0):
        self.graph_store = graph_store
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        logger.info("GraphRetriever initialized")

    async def retrieve(self, question: str, max_entities: int = 3, max_relations_per_entity: int = 5) -> List[str]:
        """Extract entities from question, query graph, return relationship strings."""
        try:
            entities = await self._extract_entities(question)
            if not entities:
                return []

            entities = entities[:max_entities]
            graph_context = []
            for entity in entities:
                results = self.graph_store.query_entity(entity, limit=max_relations_per_entity)
                for r in results:
                    rel_str = f"{r['entity']} -[{r['relationship']}]-> {r['neighbor']}"
                    graph_context.append(rel_str)

            logger.debug(f"Graph retrieved {len(graph_context)} relationships")
            return graph_context
        except Exception as e:
            logger.warning(f"Graph retrieval failed: {e}")
            return []

    async def get_related_entities(self, entities: List[str], max_neighbors: int = 5) -> List[str]:
        """Find related entities from graph for query expansion."""
        try:
            related_entities = set()
            for entity in entities:
                results = self.graph_store.query_entity(entity, limit=max_neighbors)
                for r in results:
                    neighbor = r.get("neighbor")
                    if neighbor and neighbor not in entities:
                        related_entities.add(neighbor)
            return list(related_entities)
        except Exception as e:
            logger.warning(f"Graph expansion failed: {e}")
            return []

    async def _extract_entities(self, question: str) -> List[str]:
        """Extract entities from question using LLM."""
        try:
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract main entities from the question. Return as a comma-separated list. Only entity names, no explanations."),
                ("human", "{question}"),
            ])
            chain = extraction_prompt | self.llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            result = chain.invoke(
                {"question": question},
                config={"callbacks": callbacks} if callbacks else {},
            )
            entities = [e.strip() for e in result.content.split(",") if e.strip()]
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
