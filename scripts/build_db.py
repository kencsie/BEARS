"""
Build vector and graph databases from data/corpus.json.

Usage:
    python scripts/build_db.py                  # build both vector + graph
    python scripts/build_db.py --vector-only     # vector (ChromaDB) only
    python scripts/build_db.py --graph-only      # graph (Neo4j) only
    python scripts/build_db.py --limit 100       # only first 100 docs
    python scripts/build_db.py --batch-size 50   # insert 50 docs per batch
"""

import argparse
import asyncio
import logging
from bears.core.config import get_settings
from bears.database.vector.vector_builder import CorpusDataLoader
from bears.database.vector.vector_store import VectorStoreManager
from bears.database.graph.graph_store import GraphStoreManager
from bears.database.graph.graph_builder import GraphBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("build_db")


def build_vector(corpus_path: str, limit: int | None, batch_size: int, chroma_dir: str = "data/chroma_db_corpus"):
    """Load corpus.json → embed → store in ChromaDB."""
    logger.info("=== Building Vector DB (ChromaDB) ===")

    loader = CorpusDataLoader()
    documents, ids = loader.load_and_prepare(corpus_path, limit=limit)
    logger.info(f"Prepared {len(documents)} documents")

    vector_store = VectorStoreManager(persist_directory=chroma_dir)
    existing = vector_store.get_stats()
    logger.info(f"Existing vector store stats: {existing}")

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        vector_store.add_documents(batch_docs, ids=batch_ids)
        logger.info(f"  Inserted batch {i // batch_size + 1} ({len(batch_docs)} docs)")

    final_stats = vector_store.get_stats()
    logger.info(f"Vector DB build complete: {final_stats}")


async def build_graph(corpus_path: str, limit: int | None, batch_size: int):
    """Load corpus.json → LLM entity extraction → store in Neo4j."""
    logger.info("=== Building Graph DB (Neo4j) ===")

    loader = CorpusDataLoader()
    documents, ids = loader.load_and_prepare(corpus_path, limit=limit)
    logger.info(f"Prepared {len(documents)} documents for graph extraction")

    graph_store = GraphStoreManager()
    graph_builder = GraphBuilder(graph_store)

    content_list = [(doc.page_content, doc.metadata["doc_id"]) for doc in documents]

    for i in range(0, len(content_list), batch_size):
        batch = content_list[i : i + batch_size]
        await graph_builder.build_batch(batch)
        logger.info(f"  Processed batch {i // batch_size + 1} ({len(batch)} docs)")

    final_stats = graph_store.get_stats()
    logger.info(f"Graph DB build complete: {final_stats}")


def main():
    parser = argparse.ArgumentParser(
        description="Build vector / graph databases from corpus.json"
    )
    parser.add_argument(
        "--corpus", default=None, help="Path to corpus.json (default: from settings)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process first N documents"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for insertion (default: 100)",
    )
    parser.add_argument(
        "--chroma-dir",
        default="data/chroma_db_corpus",
        help="ChromaDB persist directory (default: data/chroma_db_corpus)",
    )
    parser.add_argument(
        "--vector-only", action="store_true", help="Only build vector DB"
    )
    parser.add_argument("--graph-only", action="store_true", help="Only build graph DB")
    args = parser.parse_args()

    settings = get_settings()
    corpus_path = args.corpus or str(settings.corpus_path)

    logger.info(f"Corpus path: {corpus_path}")
    logger.info(f"Limit: {args.limit or 'all'}")
    logger.info(f"Batch size: {args.batch_size}")

    build_vector_db = not args.graph_only
    build_graph_db = not args.vector_only

    if build_vector_db:
        build_vector(corpus_path, args.limit, args.batch_size, args.chroma_dir)

    if build_graph_db:
        asyncio.run(build_graph(corpus_path, args.limit, args.batch_size))

    logger.info("Done!")


if __name__ == "__main__":
    main()
