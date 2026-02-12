"""Tests for TextToSQLConstructor — LLM calls mocked."""

from unittest.mock import MagicMock, patch

from query.construction import TextToSQLConstructor, SQLOutput
from config import LLMConfig
from models.query import ConstructedQuery, QueryTarget


class TestTextToSQLConstructor:

    def _make_constructor(self, mock_get_llm, sql="SELECT * FROM stocks", reasoning="Simple select"):
        """Helper: create a TextToSQLConstructor with mocked LLM."""
        mock_llm = MagicMock()
        mock_sql_output = SQLOutput(sql=sql, reasoning=reasoning)
        mock_structured = MagicMock()
        mock_structured.return_value = mock_sql_output
        mock_structured.invoke.return_value = mock_sql_output
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        schema = "CREATE TABLE stocks (symbol TEXT, price REAL, volume INTEGER);"
        return TextToSQLConstructor(schema=schema, llm_config=LLMConfig(provider="openai", model_name="gpt-4"))

    @patch("query.construction.get_llm")
    def test_construct_returns_constructed_query(self, mock_get_llm):
        """construct() should return a ConstructedQuery."""
        constructor = self._make_constructor(
            mock_get_llm,
            sql="SELECT symbol, price FROM stocks ORDER BY price DESC LIMIT 3",
        )
        result = constructor.construct("What are the top 3 stocks by price?")

        assert isinstance(result, ConstructedQuery)
        assert "SELECT" in result.constructed
        assert result.target == QueryTarget.SQL_DATABASE
        assert result.method == "text_to_sql"

    @patch("query.construction.get_llm")
    def test_construct_preserves_original_query(self, mock_get_llm):
        """The original natural language query should be preserved."""
        constructor = self._make_constructor(mock_get_llm)
        result = constructor.construct("Show me all stocks")

        assert result.original == "Show me all stocks"

    @patch("query.construction.get_llm")
    def test_construct_target_is_sql(self, mock_get_llm):
        """Target should always be SQL_DATABASE."""
        constructor = self._make_constructor(mock_get_llm)
        result = constructor.construct("Count all rows")

        assert result.target == QueryTarget.SQL_DATABASE


class TestSQLOutput:

    def test_sql_output_model(self):
        """SQLOutput should hold sql and reasoning."""
        output = SQLOutput(
            sql="SELECT * FROM users WHERE age > 30",
            reasoning="Filter users older than 30",
        )
        assert "SELECT" in output.sql
        assert "Filter" in output.reasoning
