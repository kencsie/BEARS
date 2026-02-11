"""
Vector store manager.

Handles all ChromaDB vector database operations.
"""

import logging
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from bears.core.config import get_settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """ChromaDB vector store manager."""

    def __init__(
        self,
        collection_name: str = "corpus_docs",
        persist_directory: str = "./chroma_db_corpus",
        embedding_model: str = "text-embedding-3-small"
    ):
        settings = get_settings()
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=settings.OPENAI_API_KEY
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

        logger.info(f"VectorStoreManager initialized: {collection_name}")

    def add_documents(self, documents: List[Document], ids: List[str] = None):
        """Batch add documents to vector store."""
        try:
            if ids:
                self.vector_store.add_documents(documents, ids=ids)
            else:
                self.vector_store.add_documents(documents)
            logger.debug(f"Added {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Vector similarity search."""
        try:
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": k}
            )
            docs = retriever.invoke(query)
            return docs
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def search_with_scores(self, query: str, k: int = 5) -> List[tuple]:
        """Vector search with similarity scores."""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def clear(self):
        """Clear the vector store (dangerous operation)."""
        try:
            logger.warning(f"Clearing collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to clear store: {e}")
            raise
