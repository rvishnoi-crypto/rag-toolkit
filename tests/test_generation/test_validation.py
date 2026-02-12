"""Tests for Self-RAG validation checkers — all LLM calls mocked."""

from unittest.mock import MagicMock, patch

from generation.validation import (
    RelevanceChecker,
    SupportChecker,
    UtilityChecker,
    RetrievalDecider,
)
from config import LLMConfig
from models.result import RelevanceScore, SupportScore, UtilityScore, RetrievalResult
from models.document import Chunk, ChunkMetadata, ScoredDocument


def _mock_structured_output(mock_get_llm, return_value):
    """Helper: mock get_llm so that prompt | llm.with_structured_output() works."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    # with_structured_output() returns a runnable; when piped with prompt
    # and invoked, it returns the structured object.
    mock_structured.return_value = return_value
    mock_structured.invoke.return_value = return_value
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm
    return mock_llm


class TestRelevanceChecker:

    @patch("generation.validation.get_llm")
    def test_check_returns_relevance_score(self, mock_get_llm):
        """check() should return a RelevanceScore."""
        expected = RelevanceScore(score=0.85, reasoning="Highly relevant")
        _mock_structured_output(mock_get_llm, expected)

        checker = RelevanceChecker(LLMConfig(provider="openai", model_name="gpt-4"))
        result = checker.check("What is RAG?", "RAG is a retrieval technique.")

        assert isinstance(result, RelevanceScore)
        assert result.score == 0.85

    @patch("generation.validation.get_llm")
    def test_check_retrieval_scores_all_docs(self, mock_get_llm):
        """check_retrieval() should return one score per document."""
        expected = RelevanceScore(score=0.7, reasoning="Relevant")
        _mock_structured_output(mock_get_llm, expected)

        checker = RelevanceChecker(LLMConfig(provider="openai", model_name="gpt-4"))

        docs = [
            ScoredDocument(
                chunk=Chunk(content=f"Doc {i}", metadata=ChunkMetadata(source="test.pdf")),
                score=0.9 - i * 0.1,
                rank=i,
            )
            for i in range(3)
        ]
        retrieval = RetrievalResult(documents=docs, query_used="test", strategy="sim")

        scores = checker.check_retrieval("test query", retrieval)
        assert len(scores) == 3
        assert all(isinstance(s, RelevanceScore) for s in scores)

    @patch("generation.validation.get_llm")
    def test_check_retrieval_handles_failure(self, mock_get_llm):
        """If scoring fails for a doc, should return neutral 0.5 score."""
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.return_value = None
        mock_structured.invoke.side_effect = Exception("LLM failed")
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        checker = RelevanceChecker(LLMConfig(provider="openai", model_name="gpt-4"))

        docs = [
            ScoredDocument(
                chunk=Chunk(content="Doc", metadata=ChunkMetadata(source="test.pdf")),
                score=0.9,
                rank=0,
            )
        ]
        retrieval = RetrievalResult(documents=docs, query_used="test", strategy="sim")

        scores = checker.check_retrieval("test", retrieval)
        assert len(scores) == 1
        assert scores[0].score == 0.5
        assert "failed" in scores[0].reasoning.lower()


class TestSupportChecker:

    @patch("generation.validation.get_llm")
    def test_check_returns_support_score(self, mock_get_llm):
        """check() should return a SupportScore."""
        expected = SupportScore(score=0.9, level="fully_supported", reasoning="Grounded")
        _mock_structured_output(mock_get_llm, expected)

        checker = SupportChecker(LLMConfig(provider="openai", model_name="gpt-4"))
        result = checker.check("RAG is great.", "RAG combines retrieval with generation.")

        assert isinstance(result, SupportScore)
        assert result.score == 0.9
        assert result.level == "fully_supported"

    @patch("generation.validation.get_llm")
    def test_check_generation_joins_docs(self, mock_get_llm):
        """check_generation() should join all doc contents and check."""
        expected = SupportScore(score=0.8, level="fully_supported", reasoning="OK")
        _mock_structured_output(mock_get_llm, expected)

        checker = SupportChecker(LLMConfig(provider="openai", model_name="gpt-4"))

        docs = [
            ScoredDocument(
                chunk=Chunk(content=f"Content {i}", metadata=ChunkMetadata(source="t.pdf")),
                score=0.9,
                rank=i,
            )
            for i in range(2)
        ]
        retrieval = RetrievalResult(documents=docs, query_used="q", strategy="sim")

        result = checker.check_generation("Answer text", retrieval)
        assert isinstance(result, SupportScore)


class TestUtilityChecker:

    @patch("generation.validation.get_llm")
    def test_check_returns_utility_score(self, mock_get_llm):
        """check() should return a UtilityScore."""
        expected = UtilityScore(score=0.8, level="high")
        _mock_structured_output(mock_get_llm, expected)

        checker = UtilityChecker(LLMConfig(provider="openai", model_name="gpt-4"))
        result = checker.check("What is RAG?", "RAG is Retrieval-Augmented Generation.")

        assert isinstance(result, UtilityScore)
        assert result.score == 0.8
        assert result.level == "high"


class TestRetrievalDecider:

    @patch("generation.validation.get_llm")
    def test_should_retrieve_returns_dict(self, mock_get_llm):
        """should_retrieve() should return a dict with expected keys."""
        # RetrievalDecider uses an inline Pydantic model for structured output.
        # Mock the structured output chain to return a mock with the right attrs.
        mock_decision = MagicMock()
        mock_decision.needs_retrieval = True
        mock_decision.reasoning = "Domain-specific question"
        mock_decision.confidence = 0.9
        _mock_structured_output(mock_get_llm, mock_decision)

        decider = RetrievalDecider(LLMConfig(provider="openai", model_name="gpt-4"))
        result = decider.should_retrieve("What is our vacation policy?")

        assert isinstance(result, dict)
        assert result["needs_retrieval"] is True
        assert result["confidence"] == 0.9
        assert "reasoning" in result

    @patch("generation.validation.get_llm")
    def test_should_retrieve_no_retrieval(self, mock_get_llm):
        """Simple questions should not need retrieval."""
        mock_decision = MagicMock()
        mock_decision.needs_retrieval = False
        mock_decision.reasoning = "Simple math"
        mock_decision.confidence = 0.95
        _mock_structured_output(mock_get_llm, mock_decision)

        decider = RetrievalDecider(LLMConfig(provider="openai", model_name="gpt-4"))
        result = decider.should_retrieve("What is 2+2?")

        assert result["needs_retrieval"] is False
