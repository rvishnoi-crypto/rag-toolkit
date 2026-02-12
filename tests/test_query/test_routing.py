"""Tests for query routing — RuleBasedRouter needs no API calls."""

from query.routing import RuleBasedRouter
from models.query import QueryType, RetrievalPath


class TestRuleBasedRouter:
    """RuleBasedRouter uses regex, no LLM needed."""

    def setup_method(self):
        self.router = RuleBasedRouter()

    def test_factual_query(self):
        """'What is' should route to FACTUAL."""
        decision = self.router.route("What is RAG?")
        assert decision.classification.query_type == QueryType.FACTUAL
        assert decision.path == RetrievalPath.VECTOR

    def test_analytical_query(self):
        """'Why' should route to ANALYTICAL."""
        decision = self.router.route("Why does RAG outperform fine-tuning?")
        assert decision.classification.query_type == QueryType.ANALYTICAL

    def test_opinion_query(self):
        """'Should' should route to OPINION."""
        decision = self.router.route("Should I use FAISS or Chroma?")
        assert decision.classification.query_type == QueryType.OPINION

    def test_structured_query(self):
        """'How many' should route to STRUCTURED path."""
        decision = self.router.route("How many customers do we have?")
        assert decision.path == RetrievalPath.STRUCTURED

    def test_contextual_fallback(self):
        """Unmatched queries should fall back to CONTEXTUAL."""
        decision = self.router.route("Tell me about the project")
        assert decision.classification.query_type == QueryType.CONTEXTUAL
        assert decision.path == RetrievalPath.VECTOR

    def test_confidence_levels(self):
        """Pattern matches should have higher confidence than fallback."""
        matched = self.router.route("What is RAG?")
        fallback = self.router.route("Tell me about the project")

        assert matched.classification.confidence > fallback.classification.confidence

    def test_case_insensitive(self):
        """Routing should be case-insensitive."""
        upper = self.router.route("WHAT IS RAG?")
        lower = self.router.route("what is rag?")
        assert upper.classification.query_type == lower.classification.query_type
