"""Tests for query translators — all LLM calls mocked."""

from unittest.mock import MagicMock, patch

from query.translation import (
    QueryRewriter,
    MultiQueryTranslator,
    StepBackTranslator,
    DecompositionTranslator,
    HyDETranslator,
)
from config import LLMConfig
from models.query import TranslatedQuery, MultiQueryExpansion, SubQuestions, HyDEDocument


class TestQueryRewriter:

    @patch("query.translation.get_llm")
    def test_translate_returns_single_query(self, mock_get_llm):
        """QueryRewriter should return a list with one TranslatedQuery."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Define Retrieval-Augmented Generation"
        mock_llm.return_value = mock_response
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        rewriter = QueryRewriter(LLMConfig(provider="openai", model_name="gpt-4"))
        result = rewriter.translate("What is RAG?")

        assert len(result) == 1
        assert isinstance(result[0], TranslatedQuery)
        assert result[0].original == "What is RAG?"
        assert result[0].method == "rewrite"

    @patch("query.translation.get_llm")
    def test_translate_preserves_original(self, mock_get_llm):
        """Original query should be preserved in the result."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Improved query"
        mock_llm.return_value = mock_response
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        rewriter = QueryRewriter(LLMConfig(provider="openai", model_name="gpt-4"))
        result = rewriter.translate("vague question")
        assert result[0].original == "vague question"


class TestMultiQueryTranslator:

    @patch("query.translation.get_llm")
    def test_translate_returns_multiple_queries(self, mock_get_llm):
        """MultiQueryTranslator should return multiple TranslatedQuery objects."""
        mock_llm = MagicMock()
        mock_expansion = MultiQueryExpansion(
            variants=["What is RAG?", "How does RAG work?", "RAG architecture"]
        )
        mock_structured = MagicMock()
        mock_structured.return_value = mock_expansion
        mock_structured.invoke.return_value = mock_expansion
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        translator = MultiQueryTranslator(
            LLMConfig(provider="openai", model_name="gpt-4"), num_queries=3
        )
        result = translator.translate("What is RAG?")

        assert len(result) == 3
        assert all(isinstance(q, TranslatedQuery) for q in result)
        assert all(q.method == "multi_query" for q in result)
        assert all(q.original == "What is RAG?" for q in result)


class TestStepBackTranslator:

    @patch("query.translation.get_llm")
    def test_translate_returns_original_and_stepback(self, mock_get_llm):
        """StepBackTranslator should return both original and step-back queries."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "How does altitude affect water properties?"
        mock_llm.return_value = mock_response
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm

        translator = StepBackTranslator(LLMConfig(provider="openai", model_name="gpt-4"))
        result = translator.translate("What is the boiling point at 2000m?")

        assert len(result) == 2
        # First should be original
        assert result[0].method == "original"
        assert result[0].rewritten == "What is the boiling point at 2000m?"
        # Second should be step-back
        assert result[1].method == "step_back"


class TestDecompositionTranslator:

    @patch("query.translation.get_llm")
    def test_translate_returns_sub_questions(self, mock_get_llm):
        """DecompositionTranslator should break query into sub-questions."""
        mock_llm = MagicMock()
        mock_subs = SubQuestions(
            questions=["What are RAG strengths?", "What are fine-tuning strengths?"]
        )
        mock_structured = MagicMock()
        mock_structured.return_value = mock_subs
        mock_structured.invoke.return_value = mock_subs
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        translator = DecompositionTranslator(LLMConfig(provider="openai", model_name="gpt-4"))
        result = translator.translate("How does RAG compare to fine-tuning?")

        assert len(result) == 2
        assert all(q.method == "decomposition" for q in result)


class TestHyDETranslator:

    @patch("query.translation.get_llm")
    def test_translate_returns_hypothetical_doc(self, mock_get_llm):
        """HyDETranslator should return a single query with hypothetical document."""
        mock_llm = MagicMock()
        mock_hyde = HyDEDocument(
            content="Retrieval-Augmented Generation (RAG) is a technique that..."
        )
        mock_structured = MagicMock()
        mock_structured.return_value = mock_hyde
        mock_structured.invoke.return_value = mock_hyde
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        translator = HyDETranslator(LLMConfig(provider="openai", model_name="gpt-4"))
        result = translator.translate("What is RAG?")

        assert len(result) == 1
        assert result[0].method == "hyde"
        assert result[0].original == "What is RAG?"
        assert "RAG" in result[0].rewritten
