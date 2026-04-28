"""Atomic graph retriever — entity extraction + Neo4j 1-hop/2-hop query."""

import logging
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from bears.core.config import get_settings
from bears.core.dependencies import get_graph_store

logger = logging.getLogger(__name__)

_ENTITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "從問題中提取主要實體，以逗號分隔的列表形式回答。只返回實體名稱，不需要其他說明。"),
    ("human", "{question}"),
])


class GraphRetriever:
    """1-hop and 2-hop graph path retrieval from Neo4j."""

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=settings.OPENAI_API_KEY,
        )
        self._chain = _ENTITY_PROMPT | self._llm

    def _extract_entities(self, query: str) -> List[str]:
        try:
            result = self._chain.invoke({"question": query})
            return [e.strip() for e in result.content.split(",") if e.strip()]
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    def retrieve(self, query: str, max_entities: int = 3, max_hops: int = 10) -> List[Dict]:
        """Return graph relationship strings as context chunks."""
        try:
            entities = self._extract_entities(query)[:max_entities]
            if not entities:
                return []

            gs = get_graph_store()
            out = []
            seen: set = set()

            for entity in entities:
                # 1-hop
                for r in gs.query_entity(entity, limit=max_hops):
                    path = f"{r['entity']} -[{r['relationship']}]-> {r['neighbor']}"
                    if path not in seen:
                        seen.add(path)
                        out.append({
                            "content": path,
                            "doc_id": r.get("neighbor_doc_id", ""),
                            "score": 1.0,
                            "source": "graph",
                        })

                # 2-hop
                for p in gs.query_2hop_subgraph(entity, limit=max_hops):
                    path = (
                        f"{p.get('start')} -[{p.get('rel1')}]-> "
                        f"{p.get('mid')} -[{p.get('rel2')}]-> {p.get('end')}"
                    )
                    if path not in seen:
                        seen.add(path)
                        out.append({
                            "content": path,
                            "doc_id": p.get("end_doc_id", ""),
                            "score": 1.0,
                            "source": "graph",
                        })

            logger.debug(f"GraphRetriever returned {len(out)} paths")
            return out
        except Exception as e:
            logger.warning(f"GraphRetriever.retrieve failed: {e}")
            return []
