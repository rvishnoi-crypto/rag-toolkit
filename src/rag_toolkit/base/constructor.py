"""
Abstract base class for query constructors.

A query constructor takes the user's raw question and converts it into
a STRUCTURED query for a specific backend — SQL for relational databases,
Cypher for graph databases, or metadata filters for filtered vector search.

This is the "structured path" — natural language in, query language out.
For the "vector path" (NL → better NL), see base/translator.py.

How it fits in the pipeline:
    User Query
        ↓
      Router → decides: vector? structured? hybrid?
        ↓
    ┌────────────────┬─────────────────────┐
    Vector path      Structured path       Hybrid (both)
    Translator       QueryConstructor      merge results
    → better NL      → SQL / Cypher
    → VectorStore    → Database
    └────────────────┴─────────────────────┘
        ↓
      Generator

Examples:
    TextToSQL:
        "revenue over 1M" → "SELECT * FROM revenue WHERE amount > 1000000"
    TextToCypher:
        "users connected to Alice" → "MATCH (a:User {name:'Alice'})-[:FOLLOWS]->(f) RETURN f"
    MetadataFilter:
        "papers about RAG from 2024" → {"year": 2024, "topic": "RAG"}
"""

from abc import ABC, abstractmethod

from rag_toolkit.models.query import ConstructedQuery


class BaseQueryConstructor(ABC):
    """
    Contract for query constructors (structured query path).

    Unlike translators, constructors need to know the schema of the
    target backend (table names, column types, node labels, etc.)
    to produce valid queries. Implementations receive this schema
    in their __init__.

    construct() returns a single ConstructedQuery because structured
    queries don't benefit from the multi-query pattern — you need
    one precise query, not multiple fuzzy ones.
    """

    @abstractmethod
    def construct(self, query: str) -> ConstructedQuery:
        """
        Convert a natural language query into a structured query.

        Args:
            query: The original user question.

        Returns:
            A ConstructedQuery with the structured query string,
            target backend type, and construction method.
        """
        ...
