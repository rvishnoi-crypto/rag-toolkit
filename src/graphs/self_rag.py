"""
SelfRAG LangGraph graph.

Self-RAG (Self-Reflective RAG) doesn't blindly retrieve and generate.
It reflects at every step:

    1. DECIDE:  "Does this query even need retrieval?"
    2. RETRIEVE: fetch documents (if needed)
    3. CHECK RELEVANCE: "Are these documents actually relevant?"
       - If NOT relevant → rewrite the query and retry (up to max_retries)
       - If still not relevant after retries → generate without context
    4. GENERATE: produce an answer
    5. CHECK SUPPORT: "Is this answer grounded in the documents?"
    6. CHECK UTILITY: "Is this answer actually useful?"

Graph structure:
    START → decide → [retrieve → check_relevance → rewrite_query → retrieve (retry loop)]
                                                  → generate_with_context
                   → [generate_without_context]
           → check_support → check_utility → END

Usage:
    from graphs.self_rag import build_self_rag_graph

    graph = build_self_rag_graph(
        vector_store=store,
        llm_config=config,
        max_retries=2,           # rewrite and retry up to 2 times
        relevance_threshold=0.5,
    )
    result = graph.invoke({"query": "What is RAG?"})
    print(result["answer"])
    print(result["support_score"])   # grounding check
    print(result["retry_count"])     # how many rewrites were needed
"""

from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

from config import LLMConfig, RetrieverConfig
from generation.generate import SimpleGenerator
from generation.validation import (
    RelevanceChecker,
    RetrievalDecider,
    SupportChecker,
    UtilityChecker,
)
from graphs.state import SelfRAGState
from query.translation import QueryRewriter
from retrieval.search import SimilarityRetriever


def build_self_rag_graph(
    vector_store: VectorStore,
    llm_config: LLMConfig = None,
    retriever_config: RetrieverConfig = None,
    relevance_threshold: float = 0.5,
    max_retries: int = 2,
) -> StateGraph:
    """
    Build the SelfRAG LangGraph.

    Args:
        vector_store: Initialized vector store with indexed documents.
        llm_config: LLM config for all components.
        retriever_config: Controls k (how many docs to retrieve).
        relevance_threshold: Minimum average relevance to use retrieved docs.
        max_retries: How many times to rewrite the query and retry retrieval
            before falling back to generating without context.

    Returns:
        A compiled LangGraph that accepts {"query": "..."} and returns
        the full SelfRAGState with answer + reflection scores.
    """
    llm_config = llm_config or LLMConfig()
    retriever_config = retriever_config or RetrieverConfig()

    # --- Components ---
    decider = RetrievalDecider(llm_config)
    retriever = SimilarityRetriever(vector_store, retriever_config)
    relevance_checker = RelevanceChecker(llm_config)
    query_rewriter = QueryRewriter(llm_config)
    generator = SimpleGenerator(llm_config)
    support_checker = SupportChecker(llm_config)
    utility_checker = UtilityChecker(llm_config)

    # --- Node functions ---

    def decide_node(state: SelfRAGState) -> dict:
        """Decide whether retrieval is needed for this query."""
        decision = decider.should_retrieve(state["query"])
        return {
            "needs_retrieval": decision["needs_retrieval"],
            "retrieval_reasoning": decision["reasoning"],
            "current_query": state["query"],  # start with the original query
            "retry_count": 0,
        }

    def retrieve_node(state: SelfRAGState) -> dict:
        """Retrieve documents using the current query (original or rewritten)."""
        retrieval = retriever.retrieve(state["current_query"])
        return {"retrieval": retrieval}

    def check_relevance_node(state: SelfRAGState) -> dict:
        """
        Score each retrieved document's relevance.

        The routing function downstream decides what to do based on
        avg_relevance vs threshold and retry_count vs max_retries.
        """
        scores = relevance_checker.check_retrieval(
            state["query"], state["retrieval"]
        )
        avg = sum(s.score for s in scores) / len(scores) if scores else 0.0
        return {
            "relevance_scores": scores,
            "avg_relevance": avg,
        }

    def rewrite_query_node(state: SelfRAGState) -> dict:
        """
        Rewrite the query for better retrieval.

        When retrieved documents aren't relevant enough, we ask the LLM
        to rephrase the query — different wording can surface different
        documents from the vector store.
        """
        translated = query_rewriter.translate(state["query"])
        # QueryRewriter returns a list with one TranslatedQuery
        rewritten = translated[0].rewritten if translated else state["query"]
        return {
            "current_query": rewritten,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    def generate_with_context_node(state: SelfRAGState) -> dict:
        """Generate an answer using retrieved documents as context."""
        generation = generator.generate(state["query"], state["retrieval"])
        return {"generation": generation}

    def generate_without_context_node(state: SelfRAGState) -> dict:
        """Generate an answer from LLM knowledge only (no retrieval)."""
        generation = generator.generate_without_context(state["query"])
        return {"generation": generation}

    def check_support_node(state: SelfRAGState) -> dict:
        """
        Check if the answer is grounded in the retrieved context.

        Only meaningful when we generated with context. If we
        generated without context, support is N/A.
        """
        if state.get("retrieval") and state["retrieval"].documents:
            score = support_checker.check_generation(
                state["generation"].answer, state["retrieval"]
            )
            return {"support_score": score}
        return {"support_score": None}

    def check_utility_node(state: SelfRAGState) -> dict:
        """Check if the answer is useful to the user."""
        score = utility_checker.check(state["query"], state["generation"].answer)
        return {"utility_score": score}

    def finalize_node(state: SelfRAGState) -> dict:
        """Set the final answer from the generation result."""
        return {"answer": state["generation"].answer}

    # --- Routing functions ---

    def route_after_decide(state: SelfRAGState) -> str:
        """After deciding: retrieve or generate directly."""
        if state.get("needs_retrieval", True):
            return "retrieve"
        return "generate_without_context"

    def route_after_relevance(state: SelfRAGState) -> str:
        """
        After relevance check — three possible outcomes:

        1. Relevant enough → generate with context
        2. Not relevant, retries left → rewrite query and retry
        3. Not relevant, no retries left → give up, generate without context
        """
        if state.get("avg_relevance", 0) >= relevance_threshold:
            return "generate_with_context"

        if state.get("retry_count", 0) < max_retries:
            return "rewrite_query"

        return "generate_without_context"

    # --- Build the graph ---
    graph = StateGraph(SelfRAGState)

    # Add nodes
    graph.add_node("decide", decide_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("check_relevance", check_relevance_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate_with_context", generate_with_context_node)
    graph.add_node("generate_without_context", generate_without_context_node)
    graph.add_node("check_support", check_support_node)
    graph.add_node("check_utility", check_utility_node)
    graph.add_node("finalize", finalize_node)

    # Edges
    graph.add_edge(START, "decide")

    # Conditional: decide → retrieve or generate_without_context
    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "retrieve": "retrieve",
            "generate_without_context": "generate_without_context",
        },
    )

    # retrieve → check_relevance
    graph.add_edge("retrieve", "check_relevance")

    # Conditional: check_relevance → generate_with_context, rewrite_query, or generate_without_context
    graph.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "generate_with_context": "generate_with_context",
            "rewrite_query": "rewrite_query",
            "generate_without_context": "generate_without_context",
        },
    )

    # rewrite_query loops back to retrieve (the retry loop)
    graph.add_edge("rewrite_query", "retrieve")

    # Both generate nodes → check_support → check_utility → finalize → END
    graph.add_edge("generate_with_context", "check_support")
    graph.add_edge("generate_without_context", "check_support")
    graph.add_edge("check_support", "check_utility")
    graph.add_edge("check_utility", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
