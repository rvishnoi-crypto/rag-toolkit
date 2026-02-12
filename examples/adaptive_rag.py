"""
Adaptive RAG example — query-aware retrieval strategies.

Shows how AdaptiveRAG classifies queries and picks different
retrieval strategies automatically:
    - Factual   → enhanced retrieval + reranking
    - Analytical → query decomposition
    - Opinion   → MMR for diversity
    - Contextual → broad retrieval

Also demonstrates SelfRAG with reflection scoring.

Run:
    python examples/adaptive_rag.py
"""

import sys
sys.path.insert(0, "src")

from techniques import AdaptiveRAG, SelfRAG
from config import LLMConfig, RetrieverConfig


LLM = LLMConfig(provider="anthropic", model_name="claude-sonnet-4-5-20250929")


def adaptive_example():
    """Show how AdaptiveRAG picks different strategies per query."""
    print("=" * 60)
    print("ADAPTIVE RAG")
    print("=" * 60)

    rag = AdaptiveRAG(
        pdf_path="data/exam_guide.pdf",
        llm_config=LLM,
    )

    queries = [
        ("What is the exam format?", "factual"),
        ("Why is this certification valuable?", "analytical"),
        ("What's the best way to prepare?", "opinion"),
        ("Explain the overall exam structure", "contextual"),
    ]

    for query, expected_type in queries:
        print(f"\nQ: {query}")
        print(f"   Expected type: {expected_type}")

        response = rag.query(query)
        print(f"   Detected type: {response.metadata.get('query_type')}")
        print(f"   Strategy used: {response.metadata.get('strategy')}")
        print(f"   A: {response.answer[:200]}...")


def self_rag_example():
    """Show SelfRAG's reflection scores."""
    print("\n" + "=" * 60)
    print("SELF-RAG (with reflection)")
    print("=" * 60)

    rag = SelfRAG(
        pdf_path="data/exam_guide.pdf",
        llm_config=LLM,
        max_retries=2,
        relevance_threshold=0.5,
    )

    response = rag.query("What topics should I study for the exam?")
    print(f"\nQ: What topics should I study for the exam?")
    print(f"A: {response.answer[:300]}...")
    print(f"\nReflection scores:")
    print(f"   Needed retrieval: {response.metadata.get('needs_retrieval')}")
    print(f"   Retries used:     {response.metadata.get('retry_count')}")
    print(f"   Avg relevance:    {response.metadata.get('avg_relevance')}")
    print(f"   Support score:    {response.metadata.get('support_score')}")
    print(f"   Utility score:    {response.metadata.get('utility_score')}")


if __name__ == "__main__":
    adaptive_example()
    self_rag_example()
