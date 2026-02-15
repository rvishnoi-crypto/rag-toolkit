"""
Indexing pipeline: load → chunk → embed → store.

Usage:
    from rag_toolkit.indexing import get_chunker, get_embedding_model, create_vector_store
"""

from .chunking import get_chunker, RecursiveChunker, SemanticChunker, UnstructuredChunker
from .embeddings import get_embedding_model
from .vectorstore import create_vector_store, load_vector_store

__all__ = [
    # Chunkers
    "get_chunker",
    "RecursiveChunker",
    "SemanticChunker",
    "UnstructuredChunker",
    # Embeddings
    "get_embedding_model",
    # Vector store
    "create_vector_store",
    "load_vector_store",
]
