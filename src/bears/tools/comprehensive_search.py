"""ComprehensiveSearchTool: parallel multi-retriever + cross-encoder reranking.

The Agentic LLM calls this tool (potentially multiple times) and controls
which retrieval engines to activate via boolean flags.
"""

import asyncio
import logging
from typing import Dict, List, Optional

from langchain_core.tools import tool

from bears.core.dependencies import get_reranker
from bears.tools.atomic.vector_retriever import VectorRetriever
from bears.tools.atomic.keyword_retriever import KeywordRetriever
from bears.tools.atomic.graph_retriever import GraphRetriever

logger = logging.getLogger(__name__)

# Module-level singletons initialised on first call
_vector_retriever: Optional[VectorRetriever] = None
_keyword_retriever: Optional[KeywordRetriever] = None
_graph_retriever: Optional[GraphRetriever] = None


def _get_vector_retriever() -> VectorRetriever:
    global _vector_retriever
    if _vector_retriever is None:
        _vector_retriever = VectorRetriever()
    return _vector_retriever


def _get_keyword_retriever() -> KeywordRetriever:
    global _keyword_retriever
    if _keyword_retriever is None:
        _keyword_retriever = KeywordRetriever()
    return _keyword_retriever


def _get_graph_retriever() -> GraphRetriever:
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _graph_retriever


async def _parallel_retrieve(
    query: str,
    use_vector: bool,
    use_keyword: bool,
    use_graph: bool,
    k: int = 10,
) -> List[Dict]:
    """Fire all enabled retrievers concurrently and merge the raw chunk pool."""
    tasks = []

    if use_vector:
        tasks.append(asyncio.to_thread(_get_vector_retriever().retrieve, query, k))
    if use_keyword:
        tasks.append(asyncio.to_thread(_get_keyword_retriever().retrieve, query, k))
    if use_graph:
        tasks.append(asyncio.to_thread(_get_graph_retriever().retrieve, query))

    if not tasks:
        return []

    results = await asyncio.gather(*tasks, return_exceptions=True)
    pool: List[Dict] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"Retriever raised: {r}")
        elif r:
            pool.extend(r)

    # Deduplicate by content fingerprint (vector + keyword may return the same chunk)
    seen: set = set()
    unique_pool: List[Dict] = []
    for chunk in pool:
        key = chunk.get("content", "")[:300]
        if key not in seen:
            seen.add(key)
            unique_pool.append(chunk)
    return unique_pool


def _format_chunks(chunks: List[Dict]) -> str:
    if not chunks:
        return "（未找到相關文件）"
    lines = []
    for i, c in enumerate(chunks, 1):
        src = c.get("source", "?")
        content = c.get("content", "").strip().replace("\n", " ")[:800]
        lines.append(f"[{i}][{src}] {content}")
    return "\n\n".join(lines)


@tool
async def comprehensive_search(
    query: str,
    use_vector: bool = True,
    use_keyword: bool = True,
    use_graph: bool = False,
) -> str:
    """Search the BEARS knowledge base using multiple retrieval engines in parallel.

    Args:
        query: The search query or sub-question used to retrieve relevant context.
        use_vector: Enable semantic vector search (best for conceptual / paraphrase questions).
        use_keyword: Enable BM25 keyword search (best for exact names, dates, or technical terms).
        use_graph: Enable knowledge-graph search (best for entity relationships and multi-hop reasoning).

    Returns:
        The top-5 most relevant context chunks after cross-encoder reranking.
    """
    logger.info(
        f"ComprehensiveSearch | query={query!r} vector={use_vector} "
        f"keyword={use_keyword} graph={use_graph}"
    )

    pool = await _parallel_retrieve(query, use_vector, use_keyword, use_graph)
    if not pool:
        return "（未找到相關文件）"

    reranker = get_reranker()
    top_chunks = await asyncio.to_thread(reranker.rerank_with_scores, query, pool, top_k=5)

    return _format_chunks(top_chunks)
