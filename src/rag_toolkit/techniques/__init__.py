"""
RAG techniques — the public API of the rag-toolkit package.

Usage:
    from rag_toolkit.techniques import SimpleRAG, AdaptiveRAG, SelfRAG

    rag = SimpleRAG(pdf_path="my_doc.pdf")
    response = rag.query("What is RAG?")
"""

from .simple import SimpleRAG
from .adaptive import AdaptiveRAG
from .self_rag import SelfRAG

__all__ = [
    "SimpleRAG",
    "AdaptiveRAG",
    "SelfRAG",
]
