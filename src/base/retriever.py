"""
Abstract base class for retrievers.

A retriever takes a query and returns relevant documents from a data source.
This is intentionally generic — the "data source" could be a vector store,
a SQL database, a graph database, or anything else.

In GeneralBot, BaseRetriever only had retrieve(query, k). Here we add
retrieve_structured() so the same interface covers both paths:
    - retrieve()            → vector search with a natural language query
    - retrieve_structured() → execute a constructed query against a database

This way a technique doesn't need to know which path is active — it just
calls the appropriate method based on the RouteDecision.
"""

from abc import ABC, abstractmethod

from models.document import ScoredDocument
from models.query import ConstructedQuery
from models.result import RetrievalResult


class BaseRetriever(ABC):
    """
    Contract for retrievers.

    Every retriever returns a RetrievalResult which wraps a list of
    ScoredDocuments plus metadata about the retrieval (query used,
    strategy, candidate count). This makes results self-describing.
    """

    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> RetrievalResult:
        """
        Retrieve documents using natural language (vector search path).

        Args:
            query: Natural language query (possibly already translated).
            k: Number of documents to return.

        Returns:
            RetrievalResult with scored documents and retrieval metadata.
        """
        ...

    def retrieve_structured(self, query: ConstructedQuery) -> RetrievalResult:
        """
        Retrieve documents using a structured query (SQL/Cypher path).

        Default implementation raises NotImplementedError. Only retrievers
        that wrap a database need to override this. Vector-only retrievers
        can ignore it entirely.

        Args:
            query: A ConstructedQuery with the SQL/Cypher/filter string.

        Returns:
            RetrievalResult with scored documents and retrieval metadata.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support structured queries. "
            "Use a database-backed retriever for SQL/Cypher queries."
        )
