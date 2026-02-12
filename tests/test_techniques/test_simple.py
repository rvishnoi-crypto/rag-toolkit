"""Tests for SimpleRAG technique."""

from unittest.mock import MagicMock, patch

import pytest

from techniques.simple import SimpleRAG
from config import LLMConfig, RetrieverConfig
from models.result import RAGResponse, RetrievalResult, GenerationResult


class TestSimpleRAG:

    def test_init_requires_source(self):
        """Must provide either pdf_path or vector_store."""
        with pytest.raises(ValueError, match="Provide either"):
            SimpleRAG()

    @patch("techniques.simple.SimpleGenerator")
    @patch("techniques.simple.SimilarityRetriever")
    def test_init_with_vector_store(self, mock_retriever_cls, mock_generator_cls, mock_vector_store):
        """Should initialize with a pre-built vector store."""
        rag = SimpleRAG(
            vector_store=mock_vector_store,
            llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
        )
        assert rag._vector_store is mock_vector_store

    @patch("techniques.simple.SimpleGenerator")
    @patch("techniques.simple.SimilarityRetriever")
    def test_query_returns_rag_response(self, mock_retriever_cls, mock_generator_cls, mock_vector_store):
        """query() should return a RAGResponse."""
        # Mock retriever to return a real RetrievalResult
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(
            documents=[], query_used="What is RAG?", strategy="similarity",
        )
        mock_retriever_cls.return_value = mock_retriever

        # Mock generator to return a real GenerationResult
        mock_generator = MagicMock()
        mock_generator.generate.return_value = GenerationResult(
            answer="Test answer", sources=[], model="openai/gpt-4",
        )
        mock_generator_cls.return_value = mock_generator

        rag = SimpleRAG(
            vector_store=mock_vector_store,
            llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
        )
        response = rag.query("What is RAG?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "Test answer"
        assert response.technique == "simple_rag"
