"""
RAG Toolkit — A modular Retrieval-Augmented Generation package.

Quick start:
    from rag_toolkit.techniques import SimpleRAG

    rag = SimpleRAG(pdf_path="my_doc.pdf")
    response = rag.query("What is RAG?")
    print(response.answer)

Three techniques available:
    - SimpleRAG:    Basic retrieve → generate pipeline
    - AdaptiveRAG:  Query-aware strategy selection (LangGraph)
    - SelfRAG:      Self-reflective with validation at every step (LangGraph)
"""

from techniques import SimpleRAG, AdaptiveRAG, SelfRAG
from config import (
    LLMConfig,
    EmbeddingConfig,
    ChunkingConfig,
    RetrieverConfig,
    VectorStoreConfig,
    ToolkitConfig,
)

__all__ = [
    # Techniques (public API)
    "SimpleRAG",
    "AdaptiveRAG",
    "SelfRAG",
    # Config
    "LLMConfig",
    "EmbeddingConfig",
    "ChunkingConfig",
    "RetrieverConfig",
    "VectorStoreConfig",
    "ToolkitConfig",
]

__version__ = "0.1.0"
