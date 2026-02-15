"""Tests for generation — uses mocked LLM to avoid API calls."""

from unittest.mock import MagicMock, patch

from rag_toolkit.generation.generate import SimpleGenerator
from rag_toolkit.config import LLMConfig
from rag_toolkit.models.result import GenerationResult, RetrievalResult


class TestSimpleGenerator:

    def _make_generator(self, mock_get_llm, answer_text="Test answer"):
        """Helper: create a SimpleGenerator with a mocked LLM chain."""
        mock_llm = MagicMock()
        # LCEL wraps non-Runnable callables in RunnableLambda, which calls
        # mock_llm(input) via __call__, not mock_llm.invoke(input).
        # Set both return_value and invoke.return_value to cover both paths.
        mock_response = MagicMock()
        mock_response.content = answer_text
        mock_llm.return_value = mock_response
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        return SimpleGenerator(LLMConfig(provider="openai", model_name="gpt-4"))

    @patch("generation.generate.get_llm")
    def test_generate_with_context(self, mock_get_llm, sample_retrieval_result):
        """Generator should produce a GenerationResult from retrieval."""
        generator = self._make_generator(mock_get_llm, "RAG is a technique.")
        result = generator.generate("What is RAG?", sample_retrieval_result)

        assert isinstance(result, GenerationResult)
        assert "RAG" in result.answer
        assert result.model == "openai/gpt-4"
        assert len(result.sources) > 0  # should extract sources from docs

    @patch("generation.generate.get_llm")
    def test_generate_without_context(self, mock_get_llm):
        """generate_without_context should work with no retrieval."""
        generator = self._make_generator(mock_get_llm, "RAG stands for Retrieval-Augmented Generation.")
        result = generator.generate_without_context("What is RAG?")

        assert isinstance(result, GenerationResult)
        assert result.sources == []

    @patch("generation.generate.get_llm")
    def test_generate_empty_retrieval_falls_back(self, mock_get_llm):
        """Empty retrieval result should fall back to generate_without_context."""
        generator = self._make_generator(mock_get_llm, "Answer from knowledge.")
        empty_retrieval = RetrievalResult(documents=[], query_used="test", strategy="sim")
        result = generator.generate("test", empty_retrieval)

        assert isinstance(result, GenerationResult)
        assert result.sources == []
