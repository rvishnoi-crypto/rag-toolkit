"""
Structured retriever for SQL databases.

This is the "structured path" counterpart to search.py's vector retrievers.
It takes a ConstructedQuery (SQL string) and executes it against a database,
converting rows into ScoredDocuments so the generator sees the same format
regardless of where data came from.

Uses LangChain's SQLDatabase utility which wraps SQLAlchemy and supports:
    - PostgreSQL, MySQL, SQLite, etc.
    - Automatic schema inspection (used by the query constructor)
    - Safe read-only execution

Usage:
    from langchain_community.utilities import SQLDatabase

    db = SQLDatabase.from_uri("sqlite:///my_data.db")
    retriever = SQLRetriever(db=db)

    # Usually called by the technique after the constructor builds the query
    constructed = ConstructedQuery(
        original="revenue over 1M",
        constructed="SELECT * FROM revenue WHERE amount > 1000000",
        target=QueryTarget.SQL_DATABASE,
    )
    result = retriever.retrieve_structured(constructed)
"""

from langchain_community.utilities import SQLDatabase

from rag_toolkit.base.retriever import BaseRetriever
from rag_toolkit.models.document import Chunk, ChunkMetadata, ScoredDocument
from rag_toolkit.models.query import ConstructedQuery
from rag_toolkit.models.result import RetrievalResult


class SQLRetriever(BaseRetriever):
    """
    Retriever that executes SQL queries against a database.

    Wraps LangChain's SQLDatabase so you get connection pooling, dialect
    support, and schema inspection for free. Each result row becomes a
    ScoredDocument with score=1.0 (exact match — no relevance ranking).

    The schema property is useful for query constructors that need to
    know table/column names to generate valid SQL.
    """

    def __init__(self, db: SQLDatabase):
        """
        Args:
            db: A LangChain SQLDatabase instance.
                Create with: SQLDatabase.from_uri("sqlite:///data.db")
        """
        self.db = db

    @property
    def schema(self) -> str:
        """
        Get the database schema as a string.

        The query constructor needs this to generate valid SQL.
        Returns CREATE TABLE statements for all tables.
        """
        return self.db.get_table_info()

    def retrieve(self, query: str, k: int = 4) -> RetrievalResult:
        """
        Not supported — use retrieve_structured() with a ConstructedQuery.

        Raises:
            NotImplementedError: Always. SQL retriever needs structured queries.
        """
        raise NotImplementedError(
            "SQLRetriever requires a ConstructedQuery. "
            "Use retrieve_structured() or route through a QueryConstructor first."
        )

    def retrieve_structured(self, query: ConstructedQuery) -> RetrievalResult:
        """
        Execute a SQL query and return results as ScoredDocuments.

        Each row becomes a Chunk with the row data as content. Scores are
        all 1.0 because SQL results are exact matches, not ranked.

        Args:
            query: ConstructedQuery with a SQL string in .constructed

        Returns:
            RetrievalResult with one ScoredDocument per row.
        """
        raw_result = self.db.run(query.constructed)

        # db.run() returns a string representation of results.
        # Parse rows — each line is one result row.
        rows = [row.strip() for row in str(raw_result).strip().split("\n") if row.strip()]

        documents = []
        for i, row in enumerate(rows):
            doc = ScoredDocument(
                chunk=Chunk(
                    content=row,
                    metadata=ChunkMetadata(
                        source="sql_database",
                        chunk_index=i,
                        doc_id=query.method,
                    ),
                ),
                score=1.0,
                rank=i,
            )
            documents.append(doc)

        return RetrievalResult(
            documents=documents,
            query_used=query.constructed,
            strategy=query.method,
            total_candidates=len(documents),
        )
