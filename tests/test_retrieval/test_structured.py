"""Tests for SQLRetriever — uses mocked SQLDatabase."""

from unittest.mock import MagicMock

import pytest

from retrieval.structured import SQLRetriever
from models.query import ConstructedQuery, QueryTarget
from models.result import RetrievalResult


class TestSQLRetriever:

    def _make_retriever(self, run_result="row1\nrow2\nrow3"):
        """Create SQLRetriever with a mocked SQLDatabase."""
        mock_db = MagicMock()
        mock_db.run.return_value = run_result
        mock_db.get_table_info.return_value = "CREATE TABLE users (id INT, name TEXT);"
        return SQLRetriever(db=mock_db)

    def test_retrieve_raises_not_implemented(self):
        """retrieve() should raise NotImplementedError."""
        retriever = self._make_retriever()
        with pytest.raises(NotImplementedError, match="ConstructedQuery"):
            retriever.retrieve("plain text query")

    def test_retrieve_structured_returns_result(self):
        """retrieve_structured() should return a RetrievalResult."""
        retriever = self._make_retriever("Alice\nBob\nCharlie")
        query = ConstructedQuery(
            original="list all users",
            constructed="SELECT name FROM users",
            target=QueryTarget.SQL_DATABASE,
        )
        result = retriever.retrieve_structured(query)

        assert isinstance(result, RetrievalResult)
        assert len(result.documents) == 3
        assert result.query_used == "SELECT name FROM users"

    def test_retrieve_structured_documents_have_correct_scores(self):
        """SQL results should all have score 1.0 (exact match)."""
        retriever = self._make_retriever("row1\nrow2")
        query = ConstructedQuery(
            original="test",
            constructed="SELECT * FROM t",
            target=QueryTarget.SQL_DATABASE,
        )
        result = retriever.retrieve_structured(query)

        for doc in result.documents:
            assert doc.score == 1.0

    def test_retrieve_structured_documents_content(self):
        """Each row should become a document's content."""
        retriever = self._make_retriever("Alice, 30\nBob, 25")
        query = ConstructedQuery(
            original="test",
            constructed="SELECT * FROM users",
            target=QueryTarget.SQL_DATABASE,
        )
        result = retriever.retrieve_structured(query)

        assert result.documents[0].chunk.content == "Alice, 30"
        assert result.documents[1].chunk.content == "Bob, 25"

    def test_retrieve_structured_empty_result(self):
        """Empty SQL result should produce empty documents list."""
        retriever = self._make_retriever("")
        query = ConstructedQuery(
            original="test",
            constructed="SELECT * FROM empty_table",
            target=QueryTarget.SQL_DATABASE,
        )
        result = retriever.retrieve_structured(query)

        assert len(result.documents) == 0

    def test_schema_property(self):
        """schema property should return table info from the database."""
        retriever = self._make_retriever()
        assert "CREATE TABLE" in retriever.schema

    def test_retrieve_structured_source_metadata(self):
        """Documents should have 'sql_database' as source."""
        retriever = self._make_retriever("data")
        query = ConstructedQuery(
            original="test",
            constructed="SELECT 1",
            target=QueryTarget.SQL_DATABASE,
        )
        result = retriever.retrieve_structured(query)

        assert result.documents[0].chunk.metadata.source == "sql_database"
