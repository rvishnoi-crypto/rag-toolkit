"""Tests for SelfRAG technique."""

from unittest.mock import MagicMock, patch

import pytest

from techniques.self_rag import SelfRAG
from config import LLMConfig
from models.result import RAGResponse, SupportScore, UtilityScore


class TestSelfRAG:

    def test_init_requires_source(self):
        """Must provide either pdf_path or vector_store."""
        with pytest.raises(ValueError, match="Provide either"):
            SelfRAG()

    @patch("techniques.self_rag.build_self_rag_graph")
    def test_init_with_vector_store(self, mock_build_graph, mock_vector_store):
        """Should initialize with a pre-built vector store."""
        mock_build_graph.return_value = MagicMock()
        rag = SelfRAG(
            vector_store=mock_vector_store,
            llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
        )
        assert rag._vector_store is mock_vector_store

    @patch("techniques.self_rag.build_self_rag_graph")
    def test_init_custom_retries(self, mock_build_graph, mock_vector_store):
        """Should accept custom max_retries and relevance_threshold."""
        mock_build_graph.return_value = MagicMock()
        rag = SelfRAG(
            vector_store=mock_vector_store,
            llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
            max_retries=5,
            relevance_threshold=0.7,
        )
        # Verify build_self_rag_graph was called with our params
        call_kwargs = mock_build_graph.call_args[1]
        assert call_kwargs["max_retries"] == 5
        assert call_kwargs["relevance_threshold"] == 0.7

    @patch("techniques.self_rag.build_self_rag_graph")
    def test_query_returns_rag_response(self, mock_build_graph, mock_vector_store):
        """query() should return a RAGResponse with self-rag metadata."""
        support = SupportScore(score=0.9, level="fully_supported", reasoning="Good")
        utility = UtilityScore(score=0.85, level="high")

        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "query": "What is the policy?",
            "answer": "The policy states...",
            "needs_retrieval": True,
            "retrieval_reasoning": "Domain question",
            "retry_count": 0,
            "avg_relevance": 0.8,
            "support_score": support,
            "utility_score": utility,
            "retrieval": None,
            "generation": None,
        }
        mock_build_graph.return_value = mock_graph

        rag = SelfRAG(
            vector_store=mock_vector_store,
            llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
        )
        response = rag.query("What is the vacation policy?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "The policy states..."
        assert response.technique == "self_rag"
        assert response.metadata["needs_retrieval"] is True
        assert response.metadata["retry_count"] == 0
        assert response.metadata["support_score"] == 0.9
        assert response.metadata["support_level"] == "fully_supported"
        assert response.metadata["utility_score"] == 0.85

    @patch("techniques.self_rag.build_self_rag_graph")
    def test_query_without_retrieval(self, mock_build_graph, mock_vector_store):
        """When retrieval isn't needed, scores should be None."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "query": "What is 2+2?",
            "answer": "4",
            "needs_retrieval": False,
            "retry_count": 0,
            "avg_relevance": None,
            "support_score": None,
            "utility_score": None,
        }
        mock_build_graph.return_value = mock_graph

        rag = SelfRAG(
            vector_store=mock_vector_store,
            llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
        )
        response = rag.query("What is 2+2?")

        assert response.metadata["needs_retrieval"] is False
        assert response.metadata["support_score"] is None
        assert response.metadata["utility_score"] is None
