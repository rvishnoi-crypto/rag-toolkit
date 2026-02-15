"""Tests for AdaptiveRAG technique."""

from unittest.mock import MagicMock, patch

import pytest

from rag_toolkit.techniques.adaptive import AdaptiveRAG
from rag_toolkit.config import LLMConfig
from rag_toolkit.models.result import RAGResponse


class TestAdaptiveRAG:

    def test_init_requires_source(self):
        """Must provide either pdf_path or vector_store."""
        with pytest.raises(ValueError, match="Provide either"):
            AdaptiveRAG()

    @patch("techniques.adaptive.build_adaptive_graph")
    def test_init_with_vector_store(self, mock_build_graph, mock_vector_store):
        """Should initialize with a pre-built vector store."""
        mock_build_graph.return_value = MagicMock()
        rag = AdaptiveRAG(
            vector_store=mock_vector_store,
            llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
        )
        assert rag._vector_store is mock_vector_store

    @patch("techniques.adaptive.build_adaptive_graph")
    def test_query_returns_rag_response(self, mock_build_graph, mock_vector_store):
        """query() should return a RAGResponse."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "query": "What is RAG?",
            "answer": "RAG is a technique",
            "query_type": "factual",
            "strategy": "factual",
            "confidence": 0.9,
            "retrieval": None,
            "generation": None,
        }
        mock_build_graph.return_value = mock_graph

        rag = AdaptiveRAG(
            vector_store=mock_vector_store,
            llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
        )
        response = rag.query("What is RAG?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "RAG is a technique"
        assert response.technique == "adaptive_rag"
        assert response.metadata["query_type"] == "factual"
        assert response.metadata["strategy"] == "factual"

    @patch("techniques.adaptive.build_adaptive_graph")
    def test_query_metadata_includes_confidence(self, mock_build_graph, mock_vector_store):
        """Metadata should include confidence from classification."""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "query": "Why?",
            "answer": "Because...",
            "query_type": "analytical",
            "strategy": "decomposition",
            "confidence": 0.85,
        }
        mock_build_graph.return_value = mock_graph

        rag = AdaptiveRAG(
            vector_store=mock_vector_store,
            llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
        )
        response = rag.query("Why does RAG work?")

        assert response.metadata["confidence"] == 0.85
