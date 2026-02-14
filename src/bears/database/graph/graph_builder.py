"""
Knowledge graph builder.

Extracts entities and relationships from text using LLM and stores them in the graph.
"""

import logging
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from bears.database.graph.graph_store import GraphStoreManager, Entity, Relationship
from bears.core.config import get_settings

logger = logging.getLogger(__name__)


class GraphData(BaseModel):
    """Complete graph data structure."""
    entities: List[Entity]
    relationships: List[Relationship]


class GraphBuilder:
    """Knowledge graph builder using LLM for entity/relationship extraction."""

    def __init__(
        self,
        graph_store: GraphStoreManager,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0
    ):
        settings = get_settings()
        self.graph_store = graph_store

        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=settings.OPENAI_API_KEY
        ).with_structured_output(GraphData)

        logger.info("GraphBuilder initialized")

    async def extract_and_store(self, content: str, doc_id: str):
        """Extract entities and relationships from text, then store in graph."""
        try:
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a graph extraction algorithm. Extract entities and relationships from the text."),
                ("human", "{text}")
            ])

            chain = extraction_prompt | self.llm

            from bears.core.langfuse_helper import get_callbacks
            callbacks = get_callbacks()

            graph_data: GraphData = chain.invoke(
                {"text": content[:1000]},
                config={"callbacks": callbacks} if callbacks else {}
            )

            for entity in graph_data.entities:
                self.graph_store.add_entity(entity, doc_id)

            for rel in graph_data.relationships:
                self.graph_store.add_relationship(rel, doc_id)

            logger.debug(
                f"Doc {doc_id}: extracted {len(graph_data.entities)} entities, "
                f"{len(graph_data.relationships)} relationships"
            )

        except Exception as e:
            logger.warning(f"Graph extraction failed (Doc {doc_id}): {e}")

    async def build_batch(
        self,
        content_list: List[tuple[str, str]]
    ):
        """Batch graph construction."""
        for content, doc_id in content_list:
            await self.extract_and_store(content, doc_id)
