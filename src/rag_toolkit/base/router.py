"""
Abstract base class for query routers.

The router is the decision-maker — it looks at a user query and decides:
    1. What TYPE of query is this? (factual, analytical, data lookup, etc.)
    2. Which RETRIEVAL PATH should handle it? (vector, structured, or hybrid)
    3. Which STRATEGY within that path? (similarity, MMR, text_to_sql, etc.)

In GeneralBot's AdaptiveRetrievalRAG, routing was hardcoded inside the
technique — classify_query() returned a category, and a big if/elif block
picked the strategy. Here the router is its own component, so you can:
    - Swap routing logic without touching retrieval
    - Use the same router across different techniques
    - Add new routes (e.g. graph DB) without modifying existing code
"""

from abc import ABC, abstractmethod

from rag_toolkit.models.query import RouteDecision


class BaseRouter(ABC):
    """
    Contract for query routers.

    A router receives the raw user query and returns a RouteDecision
    that tells the pipeline:
        - classification: what type of query is this
        - path: vector / structured / hybrid
        - strategy: which specific retrieval strategy to use

    Implementations can be as simple as a keyword check or as complex
    as an LLM-based classifier with confidence thresholds.
    """

    @abstractmethod
    def route(self, query: str) -> RouteDecision:
        """
        Decide how to handle a user query.

        Args:
            query: The original user question.

        Returns:
            RouteDecision with classification, path, and strategy.
        """
        ...
