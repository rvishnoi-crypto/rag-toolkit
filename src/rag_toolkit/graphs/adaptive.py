"""
AdaptiveRAG LangGraph graph.

This graph classifies the user's query and picks the best retrieval
strategy — instead of using the same approach for every question.

    "What is RAG?"              → factual   → enhanced retrieval + reranking
    "Why does RAG outperform?"  → analytical → query decomposition
    "What's the best model?"    → opinion   → MMR for diverse perspectives
    "Explain the implications"  → contextual → broad retrieval

Graph structure:
    START → classify → route_by_type → [factual|analytical|opinion|contextual] → generate → END

Usage:
    from rag_toolkit.graphs.adaptive import build_adaptive_graph

    graph = build_adaptive_graph(vector_store=store, llm_config=config)
    result = graph.invoke({"query": "What is RAG?"})
    print(result["answer"])
"""

from langchain_core.vectorstores import VectorStore
from langgraph.graph import END, START, StateGraph

from rag_toolkit.config import LLMConfig, RetrieverConfig
from rag_toolkit.generation.generate import SimpleGenerator
from rag_toolkit.graphs.state import AdaptiveState
from rag_toolkit.models.result import RetrievalResult
from rag_toolkit.query.routing import RuleBasedRouter
from rag_toolkit.retrieval.reranking import LLMReranker
from rag_toolkit.retrieval.search import (
    MMRRetriever,
    SimilarityRetriever,
    merge_retrieval_results,
)


def build_adaptive_graph(
    vector_store: VectorStore,
    llm_config: LLMConfig = None,
    retriever_config: RetrieverConfig = None,
) -> StateGraph:
    """
    Build the AdaptiveRAG LangGraph.

    Creates all the components (router, retrievers, generator) and wires
    them into a StateGraph with conditional routing.

    Args:
        vector_store: Initialized vector store with indexed documents.
        llm_config: LLM config for classification, reranking, and generation.
        retriever_config: Controls k and fetch_k.

    Returns:
        A compiled LangGraph that accepts {"query": "..."} and returns
        the full AdaptiveState with answer.
    """
    llm_config = llm_config or LLMConfig()
    retriever_config = retriever_config or RetrieverConfig()

    # --- Components ---
    router = RuleBasedRouter()
    similarity_retriever = SimilarityRetriever(vector_store, retriever_config)
    mmr_retriever = MMRRetriever(vector_store, retriever_config, lambda_mult=0.5)
    reranker = LLMReranker(llm_config)
    generator = SimpleGenerator(llm_config)

    # --- Node functions ---
    # Each node takes the full state and returns a partial update dict.

    def classify_node(state: AdaptiveState) -> dict:
        """Classify the query type and pick a retrieval strategy."""
        decision = router.route(state["query"])
        query_type = decision.classification.query_type.value.lower()

        # Map query type to retrieval strategy
        strategy_map = {
            "factual": "enhanced_retrieval",
            "analytical": "decomposition",
            "opinion": "diverse",
            "contextual": "broad",
        }

        return {
            "query_type": query_type,
            "strategy": strategy_map.get(query_type, "broad"),
            "confidence": decision.classification.confidence,
        }

    def factual_retrieve(state: AdaptiveState) -> dict:
        """
        Factual strategy: retrieve more candidates, then rerank.

        For factual queries we want precision — the LLM reranker
        reads each document and scores relevance, keeping only the
        best matches.
        """
        # Retrieve a large pool
        retrieval = similarity_retriever.retrieve(
            state["query"], k=retriever_config.fetch_k
        )
        # Rerank down to k
        reranked = reranker.rerank(
            state["query"], retrieval, top_k=retriever_config.k
        )
        return {"retrieval": reranked}

    def analytical_retrieve(state: AdaptiveState) -> dict:
        """
        Analytical strategy: decompose query, retrieve for each part.

        Analytical queries often have multiple facets. We use the
        DecompositionTranslator to split them, retrieve for each
        sub-question, then merge results.
        """
        from rag_toolkit.query.translation import DecompositionTranslator

        translator = DecompositionTranslator(llm_config)
        sub_queries = translator.translate(state["query"])

        results = []
        for sq in sub_queries:
            result = similarity_retriever.retrieve(sq.rewritten)
            results.append(result)

        merged = merge_retrieval_results(results)
        return {"retrieval": merged}

    def opinion_retrieve(state: AdaptiveState) -> dict:
        """
        Opinion strategy: MMR for diverse perspectives.

        Opinion queries benefit from seeing different viewpoints,
        not just the most similar documents.
        """
        retrieval = mmr_retriever.retrieve(state["query"])
        return {"retrieval": retrieval}

    def contextual_retrieve(state: AdaptiveState) -> dict:
        """
        Contextual strategy: broad retrieval with more results.

        Contextual queries need wider coverage, so we retrieve
        more documents than usual.
        """
        retrieval = similarity_retriever.retrieve(
            state["query"], k=retriever_config.k * 2
        )
        return {"retrieval": retrieval}

    def generate_node(state: AdaptiveState) -> dict:
        """Generate an answer from the retrieved documents."""
        generation = generator.generate(state["query"], state["retrieval"])
        return {
            "generation": generation,
            "answer": generation.answer,
        }

    # --- Routing function ---
    def route_by_type(state: AdaptiveState) -> str:
        """Route to the correct retrieval node based on query type."""
        strategy = state.get("strategy", "broad")
        route_map = {
            "enhanced_retrieval": "factual_retrieve",
            "decomposition": "analytical_retrieve",
            "diverse": "opinion_retrieve",
            "broad": "contextual_retrieve",
        }
        return route_map.get(strategy, "contextual_retrieve")

    # --- Build the graph ---
    graph = StateGraph(AdaptiveState)

    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("factual_retrieve", factual_retrieve)
    graph.add_node("analytical_retrieve", analytical_retrieve)
    graph.add_node("opinion_retrieve", opinion_retrieve)
    graph.add_node("contextual_retrieve", contextual_retrieve)
    graph.add_node("generate", generate_node)

    # Add edges
    graph.add_edge(START, "classify")

    # Conditional edge: classify → one of the retrieve nodes
    graph.add_conditional_edges(
        "classify",
        route_by_type,
        {
            "factual_retrieve": "factual_retrieve",
            "analytical_retrieve": "analytical_retrieve",
            "opinion_retrieve": "opinion_retrieve",
            "contextual_retrieve": "contextual_retrieve",
        },
    )

    # All retrieve nodes → generate
    graph.add_edge("factual_retrieve", "generate")
    graph.add_edge("analytical_retrieve", "generate")
    graph.add_edge("opinion_retrieve", "generate")
    graph.add_edge("contextual_retrieve", "generate")

    # Generate → end
    graph.add_edge("generate", END)

    return graph.compile()
