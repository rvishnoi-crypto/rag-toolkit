"""Tests for result models — pure Pydantic, no API calls."""

from rag_toolkit.models.document import Chunk, ChunkMetadata, ScoredDocument
from rag_toolkit.models.result import (
    RetrievalResult,
    GenerationResult,
    RelevanceScore,
    SupportScore,
    UtilityScore,
    RAGResponse,
)


def test_retrieval_result():
    rr = RetrievalResult(
        documents=[],
        query_used="test query",
        strategy="similarity",
        total_candidates=0,
    )
    assert rr.query_used == "test query"
    assert rr.strategy == "similarity"
    assert len(rr.documents) == 0


def test_retrieval_result_with_docs():
    chunk = Chunk(content="hello", metadata=ChunkMetadata(source="a.pdf"))
    doc = ScoredDocument(chunk=chunk, score=0.9, rank=0)

    rr = RetrievalResult(
        documents=[doc],
        query_used="test",
        strategy="mmr",
        total_candidates=5,
    )
    assert len(rr.documents) == 1
    assert rr.documents[0].score == 0.9


def test_generation_result():
    gr = GenerationResult(
        answer="RAG is a technique...",
        sources=["doc1.pdf", "doc2.pdf"],
        model="openai/gpt-4",
    )
    assert "RAG" in gr.answer
    assert len(gr.sources) == 2
    assert gr.model == "openai/gpt-4"


def test_relevance_score():
    rs = RelevanceScore(score=0.85, reasoning="Highly relevant to the query")
    assert rs.score == 0.85


def test_relevance_score_bounds():
    """Score must be between 0 and 1."""
    import pytest
    with pytest.raises(Exception):
        RelevanceScore(score=1.5, reasoning="too high")
    with pytest.raises(Exception):
        RelevanceScore(score=-0.1, reasoning="too low")


def test_support_score():
    ss = SupportScore(score=0.9, level="fully_supported", reasoning="Grounded")
    assert ss.level == "fully_supported"


def test_utility_score():
    us = UtilityScore(score=0.7, level="medium")
    assert us.level == "medium"


def test_rag_response():
    gr = GenerationResult(answer="Test answer", sources=[], model="openai/gpt-4")
    rr = RetrievalResult(documents=[], query_used="q", strategy="sim")

    response = RAGResponse(
        answer="Test answer",
        retrieval=rr,
        generation=gr,
        technique="simple_rag",
        metadata={"query_type": "factual"},
    )
    assert response.answer == "Test answer"
    assert response.technique == "simple_rag"
    assert response.metadata["query_type"] == "factual"


def test_rag_response_minimal():
    """RAGResponse should work with just an answer."""
    response = RAGResponse(answer="hello")
    assert response.answer == "hello"
    assert response.retrieval is None
    assert response.generation is None
    assert response.technique == "simple_rag"
    assert response.metadata == {}
