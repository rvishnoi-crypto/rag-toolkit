"""Tests for query models — pure Pydantic, no API calls."""

from models.query import (
    QueryType,
    QueryClassification,
    TranslatedQuery,
    MultiQueryExpansion,
    SubQuestions,
    ConstructedQuery,
    QueryTarget,
    RetrievalPath,
    RouteDecision,
)


def test_query_type_enum():
    assert QueryType.FACTUAL.value == "factual"
    assert QueryType.ANALYTICAL.value == "analytical"
    assert QueryType.OPINION.value == "opinion"
    assert QueryType.CONTEXTUAL.value == "contextual"


def test_query_classification():
    qc = QueryClassification(
        query_type=QueryType.FACTUAL,
        confidence=0.9,
        reasoning="Direct factual question",
    )
    assert qc.query_type == QueryType.FACTUAL
    assert qc.confidence == 0.9


def test_translated_query():
    tq = TranslatedQuery(
        original="What is RAG?",
        rewritten="Define Retrieval-Augmented Generation and explain its components",
    )
    assert tq.original == "What is RAG?"
    assert "Retrieval" in tq.rewritten


def test_multi_query_expansion():
    mq = MultiQueryExpansion(
        variants=["What is RAG?", "How does RAG work?", "RAG architecture"]
    )
    assert len(mq.variants) == 3


def test_sub_questions():
    sq = SubQuestions(
        questions=["What is retrieval?", "What is generation?"]
    )
    assert len(sq.questions) == 2


def test_constructed_query():
    cq = ConstructedQuery(
        original="Top 3 stocks by price",
        constructed="SELECT * FROM stocks ORDER BY price DESC LIMIT 3",
        target=QueryTarget.SQL_DATABASE,
    )
    assert cq.target == QueryTarget.SQL_DATABASE
    assert "SELECT" in cq.constructed


def test_retrieval_path_enum():
    assert RetrievalPath.VECTOR.value == "vector"
    assert RetrievalPath.STRUCTURED.value == "structured"
    assert RetrievalPath.HYBRID.value == "hybrid"


def test_route_decision():
    rd = RouteDecision(
        classification=QueryClassification(
            query_type=QueryType.FACTUAL,
            confidence=0.8,
            reasoning="test",
        ),
        path=RetrievalPath.VECTOR,
        strategy="factual",
    )
    assert rd.path == RetrievalPath.VECTOR
    assert rd.strategy == "factual"
