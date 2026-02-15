"""
LangGraph state definitions.

LangGraph graphs pass a state dict between nodes. Each node receives
the full state, reads what it needs, and returns updates. TypedDict
gives us type safety without the overhead of Pydantic (LangGraph
requires TypedDict, not BaseModel).

Two state types — one per graph:

    AdaptiveState: query → classify → route → retrieve → generate
    SelfRAGState:  query → decide → retrieve → check → generate → validate

Usage:
    from rag_toolkit.graphs.state import AdaptiveState, SelfRAGState
"""

from typing import Optional

from typing_extensions import TypedDict

from rag_toolkit.models.result import (
    GenerationResult,
    RelevanceScore,
    RetrievalResult,
    SupportScore,
    UtilityScore,
)


class AdaptiveState(TypedDict, total=False):
    """
    State for the AdaptiveRAG graph.

    Flow: classify → retrieve (strategy-specific) → generate

    Fields are populated by different nodes:
        - query:        set at start
        - query_type:   set by classify_node
        - strategy:     set by classify_node
        - retrieval:    set by retrieve_node
        - generation:   set by generate_node
        - answer:       set by generate_node
    """

    # Input
    query: str

    # After classification
    query_type: str       # "factual", "analytical", "opinion", "contextual"
    strategy: str         # "enhanced_retrieval", "decomposition", "diverse", "broad"
    confidence: float

    # After retrieval
    retrieval: RetrievalResult

    # After generation
    generation: GenerationResult
    answer: str


class SelfRAGState(TypedDict, total=False):
    """
    State for the SelfRAG graph.

    Flow: decide → retrieve → check_relevance → generate → check_support → check_utility

    The key insight of Self-RAG is conditional execution:
        - If needs_retrieval is False → skip retrieval, generate from knowledge
        - If relevance is low → rewrite query or skip retrieval
        - If support is low → answer may be hallucinated

    Fields are populated by different nodes:
        - query:             set at start
        - needs_retrieval:   set by decide_node
        - retrieval:         set by retrieve_node
        - relevance_scores:  set by check_relevance_node
        - generation:        set by generate_node
        - support_score:     set by check_support_node
        - utility_score:     set by check_utility_node
        - answer:            set by finalize_node
    """

    # Input
    query: str

    # After retrieval decision
    needs_retrieval: bool
    retrieval_reasoning: str

    # Query rewriting (for retry loop)
    current_query: str      # the query being used for retrieval (original or rewritten)
    retry_count: int         # how many times we've rewritten and retried

    # After retrieval
    retrieval: RetrievalResult

    # After relevance check
    relevance_scores: list[RelevanceScore]
    avg_relevance: float

    # After generation
    generation: GenerationResult

    # After support check
    support_score: Optional[SupportScore]

    # After utility check
    utility_score: Optional[UtilityScore]

    # Final
    answer: str
