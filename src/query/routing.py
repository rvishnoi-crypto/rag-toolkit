"""
Query routing implementations.

The router classifies a query and decides which retrieval path to take:
    - Vector search (translate → vector store)
    - Structured query (construct → SQL/Cypher database)
    - Hybrid (both paths, merge results)

Two approaches:

    LLMRouter — The LLM makes the FULL routing decision in one call.
    It sees the query, the available data sources, and decides both the
    query type AND the retrieval path. This is how production systems
    do it — the LLM understands intent far better than keyword matching.

    RuleBasedRouter — Zero-cost regex patterns. Good for prototyping
    or latency-sensitive pipelines where you can't afford the LLM call.

Usage:
    from query.routing import LLMRouter, RuleBasedRouter

    # LLM-based (production — one call, full decision)
    router = LLMRouter(llm_config=LLMConfig(), data_sources=["vector_store", "sql_database"])
    decision = router.route("What drove our revenue decline last quarter?")
    # → RouteDecision(path=HYBRID, strategy="hybrid", ...)

    # Rule-based (prototyping — free, instant)
    router = RuleBasedRouter()
    decision = router.route("What is RAG?")
    # → RouteDecision(path=VECTOR, strategy="factual", ...)
"""

import re

from langchain_core.prompts import PromptTemplate

from base.router import BaseRouter
from config import LLMConfig
from models.query import (
    FullRouteClassification,
    QueryClassification,
    QueryType,
    RetrievalPath,
    RouteDecision,
)
from utils.helpers import get_llm


class LLMRouter(BaseRouter):
    """
    The LLM makes the full routing decision in one structured output call.

    Instead of:
        1. Classify query type (LLM call)
        2. Use keywords to pick path (heuristic — fragile)

    We do:
        1. Show the LLM the query + available data sources
        2. LLM returns: query_type + path + strategy (one call)

    This handles ambiguous queries that keywords miss entirely:
        "What drove our Q4 numbers?"    → HYBRID (needs data + docs)
        "Show me the top 5 customers"   → STRUCTURED (pure data)
        "Explain our pricing strategy"  → VECTOR (docs only)

    The data_sources parameter tells the LLM what's available.
    If you only have a vector store, the LLM will never route to SQL.
    """

    def __init__(
        self,
        llm_config: LLMConfig = None,
        data_sources: list[str] = None,
    ):
        """
        Args:
            llm_config: LLM configuration for the routing call.
            data_sources: Available data sources to route to.
                Examples: ["vector_store"], ["vector_store", "sql_database"],
                          ["vector_store", "sql_database", "graph_database"]
                Defaults to ["vector_store"] (vector-only, no structured path).
        """
        self._llm = get_llm(llm_config or LLMConfig())
        self._data_sources = data_sources or ["vector_store"]
        self._prompt = self._build_prompt()

    def _build_prompt(self) -> PromptTemplate:
        """
        Build the routing prompt dynamically based on available data sources.

        The prompt tells the LLM exactly what paths are available so it
        never routes to a backend that doesn't exist.
        """
        source_descriptions = {
            "vector_store": (
                "- Vector store: Contains embedded documents for semantic search. "
                "Use for knowledge questions, explanations, and concept lookups."
            ),
            "sql_database": (
                "- SQL database: Contains structured data in tables. "
                "Use for data lookups, aggregations, counts, rankings, and "
                "questions about specific numbers or metrics."
            ),
            "graph_database": (
                "- Graph database: Contains entities and relationships. "
                "Use for questions about connections, networks, and relationships."
            ),
        }

        available = "\n".join(
            source_descriptions.get(s, f"- {s}")
            for s in self._data_sources
        )

        # Determine which paths are valid
        has_vector = "vector_store" in self._data_sources
        has_structured = any(
            s in self._data_sources for s in ["sql_database", "graph_database"]
        )

        path_options = []
        if has_vector:
            path_options.append("'vector' for document/knowledge search")
        if has_structured:
            path_options.append("'structured' for database queries")
        if has_vector and has_structured:
            path_options.append("'hybrid' if the query needs both documents and data")

        path_str = ", ".join(path_options)

        template = (
            "You are a query router. Given a user query, decide how to handle it.\n\n"
            "Available data sources:\n"
            f"{available}\n\n"
            "Query: {{query}}\n\n"
            f"Choose the retrieval path ({path_str}) and classify the query type.\n"
        )

        return PromptTemplate(
            input_variables=["query"],
            template=template,
        )

    def route(self, query: str) -> RouteDecision:
        chain = self._prompt | self._llm.with_structured_output(FullRouteClassification)
        result = chain.invoke({"query": query})

        # Map the LLM's string path to the enum
        path_map = {
            "vector": RetrievalPath.VECTOR,
            "structured": RetrievalPath.STRUCTURED,
            "hybrid": RetrievalPath.HYBRID,
        }
        path = path_map.get(result.retrieval_path.lower(), RetrievalPath.VECTOR)

        # Validate: don't route to structured if no DB is available
        has_structured = any(
            s in self._data_sources for s in ["sql_database", "graph_database"]
        )
        if path in (RetrievalPath.STRUCTURED, RetrievalPath.HYBRID) and not has_structured:
            path = RetrievalPath.VECTOR
            result.strategy = "default"

        classification = QueryClassification(
            query_type=result.query_type,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )

        return RouteDecision(
            classification=classification,
            path=path,
            strategy=result.strategy,
        )


class RuleBasedRouter(BaseRouter):
    """
    Fast, zero-cost router using keyword patterns.

    No LLM call — just regex and keyword matching. Good for prototyping
    and latency-sensitive pipelines.

    Limitations:
        - Misses ambiguous queries ("what drove our Q4 numbers?")
        - Can't understand intent, only surface keywords
        - No confidence scoring

    For production, use LLMRouter instead.

    Rules (checked in order, first match wins):
        SQL-like keywords  → STRUCTURED path
        "why/compare"      → ANALYTICAL (vector)
        "what is/define"   → FACTUAL (vector)
        "should/best"      → OPINION (vector)
        Everything else    → CONTEXTUAL (vector, broad retrieval)
    """

    _PATTERNS: list[tuple[str, QueryType, RetrievalPath]] = [
        # Structured queries
        (r"\b(how many|count|total|average|sum|revenue|sales|profit)\b", QueryType.FACTUAL, RetrievalPath.STRUCTURED),
        (r"\b(top \d+|rank|sort by|group by|filter)\b", QueryType.FACTUAL, RetrievalPath.STRUCTURED),
        # Analytical
        (r"\b(why|how does|compare|difference between|analyze)\b", QueryType.ANALYTICAL, RetrievalPath.VECTOR),
        # Factual
        (r"\b(what is|define|when did|who is|where is)\b", QueryType.FACTUAL, RetrievalPath.VECTOR),
        # Opinion
        (r"\b(should|best|recommend|opinion|better)\b", QueryType.OPINION, RetrievalPath.VECTOR),
    ]

    def route(self, query: str) -> RouteDecision:
        query_lower = query.lower()

        for pattern, query_type, path in self._PATTERNS:
            if re.search(pattern, query_lower):
                strategy_map = {
                    QueryType.FACTUAL: "factual",
                    QueryType.ANALYTICAL: "analytical",
                    QueryType.OPINION: "opinion",
                    QueryType.CONTEXTUAL: "contextual",
                }
                strategy = (
                    "text_to_sql" if path == RetrievalPath.STRUCTURED
                    else strategy_map[query_type]
                )
                return RouteDecision(
                    classification=QueryClassification(
                        query_type=query_type,
                        confidence=0.7,
                        reasoning=f"Matched pattern: {pattern}",
                    ),
                    path=path,
                    strategy=strategy,
                )

        # Default: contextual with vector search
        return RouteDecision(
            classification=QueryClassification(
                query_type=QueryType.CONTEXTUAL,
                confidence=0.5,
                reasoning="No specific pattern matched, using broad retrieval",
            ),
            path=RetrievalPath.VECTOR,
            strategy="contextual",
        )
