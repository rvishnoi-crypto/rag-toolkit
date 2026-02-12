"""Tests for reranking — LLMReranker mocked, DiversityReranker uses real math."""

from unittest.mock import MagicMock, patch

from retrieval.reranking import LLMReranker, DiversityReranker
from config import LLMConfig
from models.document import Chunk, ChunkMetadata, ScoredDocument
from models.result import RelevanceScore, RetrievalResult


def _make_retrieval_result(n=5):
    """Create a RetrievalResult with n documents."""
    docs = [
        ScoredDocument(
            chunk=Chunk(
                content=f"Document content {i}",
                metadata=ChunkMetadata(source="test.pdf", chunk_index=i),
            ),
            score=0.9 - i * 0.1,
            rank=i,
        )
        for i in range(n)
    ]
    return RetrievalResult(
        documents=docs, query_used="test query", strategy="similarity", total_candidates=n
    )


class TestLLMReranker:

    @patch("retrieval.reranking.get_llm")
    def test_rerank_returns_retrieval_result(self, mock_get_llm):
        """rerank() should return a RetrievalResult."""
        mock_llm = MagicMock()
        mock_score = RelevanceScore(score=0.8, reasoning="Relevant")
        mock_structured = MagicMock()
        mock_structured.return_value = mock_score
        mock_structured.invoke.return_value = mock_score
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        reranker = LLMReranker(LLMConfig(provider="openai", model_name="gpt-4"))
        result = reranker.rerank("test", _make_retrieval_result(), top_k=3)

        assert isinstance(result, RetrievalResult)
        assert len(result.documents) == 3
        assert result.strategy == "llm_reranked"

    @patch("retrieval.reranking.get_llm")
    def test_rerank_respects_top_k(self, mock_get_llm):
        """Should return at most top_k documents."""
        mock_llm = MagicMock()
        mock_score = RelevanceScore(score=0.7, reasoning="OK")
        mock_structured = MagicMock()
        mock_structured.return_value = mock_score
        mock_structured.invoke.return_value = mock_score
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        reranker = LLMReranker(LLMConfig(provider="openai", model_name="gpt-4"))
        result = reranker.rerank("test", _make_retrieval_result(n=5), top_k=2)

        assert len(result.documents) == 2

    @patch("retrieval.reranking.get_llm")
    def test_rerank_sorts_by_score(self, mock_get_llm):
        """Documents should be sorted by LLM relevance score (descending)."""
        mock_llm = MagicMock()
        # Return different scores for different documents
        scores = [
            RelevanceScore(score=0.3, reasoning="Low"),
            RelevanceScore(score=0.9, reasoning="High"),
            RelevanceScore(score=0.6, reasoning="Medium"),
        ]
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = scores
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        reranker = LLMReranker(LLMConfig(provider="openai", model_name="gpt-4"))
        result = reranker.rerank("test", _make_retrieval_result(n=3), top_k=3)

        assert result.documents[0].score == 0.9
        assert result.documents[1].score == 0.6
        assert result.documents[2].score == 0.3

    @patch("retrieval.reranking.get_llm")
    def test_rerank_handles_llm_failure(self, mock_get_llm):
        """If LLM fails for a doc, should keep original score."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = Exception("LLM error")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        reranker = LLMReranker(LLMConfig(provider="openai", model_name="gpt-4"))
        original = _make_retrieval_result(n=2)
        result = reranker.rerank("test", original, top_k=2)

        # Should still return results using original scores
        assert len(result.documents) == 2


class TestDiversityReranker:

    def test_cosine_similarity_identical(self):
        """Identical vectors should have similarity 1.0."""
        sim = DiversityReranker._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have similarity 0.0."""
        sim = DiversityReranker._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        """Zero vector should return 0.0 (no division error)."""
        sim = DiversityReranker._cosine_similarity([0.0, 0.0], [1.0, 1.0])
        assert sim == 0.0

    def test_rerank_with_mock_embeddings(self):
        """DiversityReranker should select diverse documents."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [1.0, 0.0, 0.0]
        # Docs: 0 and 1 are very similar, 2 is different
        mock_embeddings.embed_documents.return_value = [
            [0.9, 0.1, 0.0],  # similar to query
            [0.85, 0.15, 0.0],  # very similar to doc 0
            [0.1, 0.9, 0.0],  # different from doc 0, less relevant to query
        ]

        reranker = DiversityReranker(embedding_model=mock_embeddings, lambda_mult=0.5)
        retrieval = _make_retrieval_result(n=3)
        result = reranker.rerank("test", retrieval, top_k=2)

        assert len(result.documents) == 2
        assert isinstance(result, RetrievalResult)
        assert "diversity" in result.strategy

    def test_rerank_empty_retrieval(self):
        """Empty retrieval should return empty result."""
        mock_embeddings = MagicMock()
        reranker = DiversityReranker(embedding_model=mock_embeddings)

        empty = RetrievalResult(documents=[], query_used="test", strategy="sim")
        result = reranker.rerank("test", empty, top_k=4)

        assert len(result.documents) == 0

    def test_rerank_top_k_larger_than_docs(self):
        """top_k larger than available docs should return all docs."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [1.0, 0.0]
        mock_embeddings.embed_documents.return_value = [
            [0.9, 0.1],
            [0.5, 0.5],
        ]

        reranker = DiversityReranker(embedding_model=mock_embeddings, lambda_mult=0.5)
        retrieval = _make_retrieval_result(n=2)
        result = reranker.rerank("test", retrieval, top_k=10)

        assert len(result.documents) == 2
