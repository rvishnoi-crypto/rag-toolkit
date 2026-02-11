"""
Abstract base class for query translators.

A query translator takes the user's raw question and transforms it
into one or more queries that are better suited for VECTOR retrieval.

This is the "vector path" — natural language in, better natural language out.
For the "structured path" (NL → SQL/Cypher), see base/constructor.py.

In GeneralBot, these lived inside QueryTransformRAG as methods
(query_rewrite, multi_query, step_back, etc.) — tightly coupled to
that one technique. Here the contract is standalone, so any technique
can use any translator.

Examples of what translators do:
    Rewrite:      "What's RAG?" → "What is Retrieval-Augmented Generation?"
    Multi-query:  "What's RAG?" → ["definition of RAG", "RAG architecture", ...]
    Step-back:    "What's RAG?" → "What are approaches to augmenting LLMs with external data?"
    HyDE:         "What's RAG?" → (hypothetical answer paragraph used as query)
"""

from abc import ABC, abstractmethod

from models.query import TranslatedQuery


class BaseTranslator(ABC):
    """
    Contract for query translators (vector search path).

    translate() returns a list because some strategies (multi-query,
    decomposition) produce multiple queries from one input. Strategies
    that produce a single query (rewrite, step-back) return a one-element
    list. This keeps the interface uniform — the retriever always iterates
    over the list and merges results.
    """

    @abstractmethod
    def translate(self, query: str) -> list[TranslatedQuery]:
        """
        Transform a user query into one or more retrieval-optimized queries.

        Args:
            query: The original user question.

        Returns:
            List of TranslatedQuery objects. Each has the original query,
            the rewritten version, and which method produced it.
        """
        ...
