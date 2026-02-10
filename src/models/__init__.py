"""
Pydantic models shared across the RAG toolkit.

Import from here rather than reaching into submodules:
    from models import Chunk, QueryType, GenerationResult
"""

from .document import Chunk, ChunkMetadata, ScoredDocument
from .query import (
    QueryType,
    QueryClassification,
    TranslatedQuery,
    MultiQueryExpansion,
    SubQuestions,
    HyDEDocument,
    RouteDecision,
)
from .result import (
    RetrievalResult,
    GenerationResult,
    RelevanceScore,
    SupportScore,
    UtilityScore,
    RAGResponse,
)

__all__ = [
    # Document
    "Chunk",
    "ChunkMetadata",
    "ScoredDocument",
    # Query
    "QueryType",
    "QueryClassification",
    "TranslatedQuery",
    "MultiQueryExpansion",
    "SubQuestions",
    "HyDEDocument",
    "RouteDecision",
    # Result
    "RetrievalResult",
    "GenerationResult",
    "RelevanceScore",
    "SupportScore",
    "UtilityScore",
    "RAGResponse",
]
