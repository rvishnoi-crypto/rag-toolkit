"""
Result models for retrieval and generation outputs.

These are the final outputs of the pipeline — what the user gets back.
Also includes scoring models used by Self-RAG for reflection/validation.
"""

from typing import Optional
from pydantic import BaseModel, Field

from .document import ScoredDocument


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    """
    Output of the retrieval stage.

    Bundles the retrieved documents with metadata about how retrieval was done.
    This lets the generation stage (and the user) understand what happened.
    """

    documents: list[ScoredDocument] = Field(default_factory=list)
    query_used: str = Field(description="The actual query sent to the vector store")
    strategy: str = Field(default="similarity", description="Retrieval strategy used")
    total_candidates: int = Field(
        default=0,
        description="How many docs were considered before filtering/reranking",
    )


# ---------------------------------------------------------------------------
# Generation results
# ---------------------------------------------------------------------------

class GenerationResult(BaseModel):
    """
    Output of the generation stage.

    The answer plus everything needed to understand how it was produced —
    which documents were used, what technique ran, and any quality scores.
    """

    answer: str = Field(description="The generated answer")
    sources: list[str] = Field(
        default_factory=list,
        description="Source identifiers (file names, URLs) used to generate the answer",
    )
    model: str = Field(default="", description="Model that produced this answer")


# ---------------------------------------------------------------------------
# Validation / reflection scores (used by Self-RAG)
# ---------------------------------------------------------------------------

class RelevanceScore(BaseModel):
    """
    LLM-judged relevance of a document to a query.

    Used by reranking and by Self-RAG's relevance evaluation step.
    The score + reasoning pattern lets you both filter AND explain.
    """

    score: float = Field(ge=0.0, le=1.0, description="Relevance score from 0 (irrelevant) to 1")
    reasoning: str = Field(default="", description="Why this score was given")


class SupportScore(BaseModel):
    """
    How well the generated answer is supported by the retrieved context.

    Self-RAG uses this to decide if the answer is grounded in facts
    or if it's hallucinating. This is the faithfulness check.
    """

    score: float = Field(ge=0.0, le=1.0, description="Support score from 0 to 1")
    level: str = Field(
        default="partial",
        description="Support level: fully_supported, partially_supported, not_supported",
    )
    reasoning: str = Field(default="", description="Explanation of the support assessment")


class UtilityScore(BaseModel):
    """
    How useful the generated answer is for the user's query.

    The final quality gate in Self-RAG — even a faithful answer can be
    unhelpful if it doesn't actually address what was asked.
    """

    score: float = Field(ge=0.0, le=1.0, description="Utility score from 0 to 1")
    level: str = Field(
        default="medium",
        description="Utility level: high, medium, low",
    )


# ---------------------------------------------------------------------------
# Full RAG response (top-level output)
# ---------------------------------------------------------------------------

class RAGResponse(BaseModel):
    """
    The complete response from any RAG technique.

    This is what users get back from rag.query(). Consistent interface
    regardless of which technique ran — SimpleRAG, AdaptiveRAG, or SelfRAG.

    The metadata dict carries technique-specific info:
        - AdaptiveRAG: query_type, strategy, confidence
        - SelfRAG: support_score, utility_score, retry_count, etc.
    """

    answer: str = Field(description="The generated answer")
    retrieval: Optional[RetrievalResult] = Field(
        default=None, description="Retrieval details (documents, scores, strategy)",
    )
    generation: Optional[GenerationResult] = Field(
        default=None, description="Generation details (model, sources)",
    )
    technique: str = Field(default="simple_rag", description="Which RAG technique produced this")
    metadata: dict = Field(
        default_factory=dict,
        description="Technique-specific info (query_type, support_score, retry_count, etc.)",
    )
