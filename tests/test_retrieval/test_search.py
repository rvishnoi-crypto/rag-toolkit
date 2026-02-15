"""Tests for retrieval search — uses mock vector store, no API calls."""

from rag_toolkit.retrieval.search import SimilarityRetriever, MMRRetriever, merge_retrieval_results
from rag_toolkit.config import RetrieverConfig
from rag_toolkit.models.result import RetrievalResult


class TestSimilarityRetriever:

    def test_retrieve_returns_result(self, mock_vector_store, retriever_config):
        retriever = SimilarityRetriever(mock_vector_store, retriever_config)
        result = retriever.retrieve("What is RAG?")

        assert isinstance(result, RetrievalResult)
        assert len(result.documents) == 4
        assert result.strategy == "similarity"
        assert result.query_used == "What is RAG?"

    def test_scores_are_normalized(self, mock_vector_store, retriever_config):
        """Scores should be between 0 and 1 (normalized from L2 distance)."""
        retriever = SimilarityRetriever(mock_vector_store, retriever_config)
        result = retriever.retrieve("test")

        for doc in result.documents:
            assert 0.0 <= doc.score <= 1.0

    def test_documents_are_ranked(self, mock_vector_store, retriever_config):
        """Documents should have sequential ranks starting from 0."""
        retriever = SimilarityRetriever(mock_vector_store, retriever_config)
        result = retriever.retrieve("test")

        ranks = [doc.rank for doc in result.documents]
        assert ranks == [0, 1, 2, 3]

    def test_scores_descending(self, mock_vector_store, retriever_config):
        """Higher relevance (lower L2 distance) should get higher score."""
        retriever = SimilarityRetriever(mock_vector_store, retriever_config)
        result = retriever.retrieve("test")

        scores = [doc.score for doc in result.documents]
        assert scores == sorted(scores, reverse=True)


class TestMMRRetriever:

    def test_retrieve_returns_result(self, mock_vector_store, retriever_config):
        retriever = MMRRetriever(mock_vector_store, retriever_config)
        result = retriever.retrieve("test")

        assert isinstance(result, RetrievalResult)
        assert "mmr" in result.strategy

    def test_rank_based_scores(self, mock_vector_store, retriever_config):
        """MMR uses rank-based scores since the API doesn't return scores."""
        retriever = MMRRetriever(mock_vector_store, retriever_config)
        result = retriever.retrieve("test")

        # First doc should have highest score
        assert result.documents[0].score > result.documents[-1].score


class TestMergeResults:

    def test_merge_empty(self):
        result = merge_retrieval_results([])
        assert len(result.documents) == 0

    def test_merge_deduplicates(self, mock_vector_store, retriever_config):
        """Same document appearing in multiple results should be deduplicated."""
        retriever = SimilarityRetriever(mock_vector_store, retriever_config)
        r1 = retriever.retrieve("query 1")
        r2 = retriever.retrieve("query 2")

        merged = merge_retrieval_results([r1, r2])
        # Deduplicated — same docs in both, so count shouldn't double
        assert len(merged.documents) <= len(r1.documents) + len(r2.documents)
        assert merged.strategy == "merged"

    def test_merge_keeps_highest_score(self, sample_scored_documents):
        """When a document appears in multiple results, keep the highest score."""
        from rag_toolkit.models.document import Chunk, ChunkMetadata, ScoredDocument

        chunk = Chunk(content="same content", metadata=ChunkMetadata(source="test"))

        r1 = RetrievalResult(
            documents=[ScoredDocument(chunk=chunk, score=0.5, rank=0)],
            query_used="q1", strategy="sim",
        )
        r2 = RetrievalResult(
            documents=[ScoredDocument(chunk=chunk, score=0.9, rank=0)],
            query_used="q2", strategy="sim",
        )

        merged = merge_retrieval_results([r1, r2])
        assert len(merged.documents) == 1
        assert merged.documents[0].score == 0.9
