"""
Vector store retrieval implementations.

These are the "vector path" retrievers — they take a natural language query
and return relevant documents from a vector store using embedding similarity.

In GeneralBot, vector search was buried inside each technique:
    self.vector_store.similarity_search(query, k=k)
    self.vector_store.max_marginal_relevance_search(query, k=k, ...)

Here each strategy is its own class implementing BaseRetriever, so you
can swap them without touching any other code.

Both retrievers work with translated queries too — if a translator
produced multiple queries, the technique iterates and merges:
    for tq in translated_queries:
        results = retriever.retrieve(tq.rewritten, k=k)

Usage:
    from retrieval.search import SimilarityRetriever, MMRRetriever
    from config import RetrieverConfig

    retriever = SimilarityRetriever(vector_store=store, config=RetrieverConfig(k=4))
    result = retriever.retrieve("What is RAG?")
    # → RetrievalResult with 4 ScoredDocuments
"""

from langchain_core.vectorstores import VectorStore

from base.retriever import BaseRetriever
from config import RetrieverConfig
from models.document import Chunk, ChunkMetadata, ScoredDocument
from models.result import RetrievalResult


class SimilarityRetriever(BaseRetriever):
    """
    Standard cosine similarity search.

    The simplest and most common retriever. Embeds the query, finds the
    k nearest vectors in the store, returns them ranked by distance.

    Uses similarity_search_with_score() so we get actual scores, not
    just documents. The scores let downstream components (rerankers,
    generators) make informed decisions.

    Note: FAISS returns L2 distance (lower = more similar), while Chroma
    returns cosine distance. We normalize both to a 0-1 relevance score
    where higher = more relevant.
    """

    def __init__(self, vector_store: VectorStore, config: RetrieverConfig = None):
        self._store = vector_store
        self._config = config or RetrieverConfig()

    def retrieve(self, query: str, k: int = None) -> RetrievalResult:
        k = k or self._config.k
        docs_and_scores = self._store.similarity_search_with_score(query, k=k)

        documents = []
        for rank, (doc, raw_score) in enumerate(docs_and_scores):
            # Normalize score: different backends use different scales.
            # FAISS: L2 distance (0 = identical, higher = farther)
            # Chroma: cosine distance (0 = identical, 2 = opposite)
            # We convert to relevance: 1 / (1 + distance) → 0-1, higher = better
            relevance = 1.0 / (1.0 + raw_score)

            documents.append(ScoredDocument(
                chunk=Chunk(
                    content=doc.page_content,
                    metadata=ChunkMetadata(
                        source=doc.metadata.get("source", ""),
                        page=doc.metadata.get("page"),
                        chunk_index=doc.metadata.get("chunk_index", rank),
                        doc_id=doc.metadata.get("doc_id", ""),
                    ),
                ),
                score=relevance,
                rank=rank,
            ))

        return RetrievalResult(
            documents=documents,
            query_used=query,
            strategy="similarity",
            total_candidates=len(documents),
        )


class MMRRetriever(BaseRetriever):
    """
    Maximal Marginal Relevance (MMR) retrieval.

    Balances relevance with diversity — instead of returning the 4 most
    similar documents (which might all say the same thing), MMR picks
    documents that are both relevant to the query AND different from
    each other.

    The formula for each selected document:
        MMR = lambda * similarity(doc, query) - (1-lambda) * max_similarity(doc, already_selected)

    lambda controls the trade-off:
        lambda = 1.0 → pure relevance (same as SimilarityRetriever)
        lambda = 0.5 → balanced (default)
        lambda = 0.0 → pure diversity

    Best for:
        - Opinion queries (want diverse perspectives)
        - Broad topics (want coverage, not redundancy)
        - When you suspect duplicate/near-duplicate content
    """

    def __init__(
        self,
        vector_store: VectorStore,
        config: RetrieverConfig = None,
        lambda_mult: float = 0.5,
    ):
        self._store = vector_store
        self._config = config or RetrieverConfig()
        self._lambda = lambda_mult

    def retrieve(self, query: str, k: int = None) -> RetrievalResult:
        k = k or self._config.k
        fetch_k = self._config.fetch_k

        docs = self._store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=self._lambda,
        )

        # MMR doesn't return scores directly, so we assign rank-based scores.
        # First result is most relevant, decreasing from there.
        documents = []
        for rank, doc in enumerate(docs):
            # Rank-based score: first doc gets 1.0, last gets ~0.5
            score = 1.0 - (rank / (2 * max(len(docs), 1)))

            documents.append(ScoredDocument(
                chunk=Chunk(
                    content=doc.page_content,
                    metadata=ChunkMetadata(
                        source=doc.metadata.get("source", ""),
                        page=doc.metadata.get("page"),
                        chunk_index=doc.metadata.get("chunk_index", rank),
                        doc_id=doc.metadata.get("doc_id", ""),
                    ),
                ),
                score=score,
                rank=rank,
            ))

        return RetrievalResult(
            documents=documents,
            query_used=query,
            strategy=f"mmr(lambda={self._lambda})",
            total_candidates=fetch_k,
        )


def merge_retrieval_results(results: list[RetrievalResult]) -> RetrievalResult:
    """
    Merge multiple RetrievalResults, deduplicating by content.

    Used when a translator produces multiple queries — each query
    generates its own RetrievalResult, and we merge them into one.
    Documents that appear in multiple results keep their highest score.

    Args:
        results: List of RetrievalResults to merge.

    Returns:
        A single RetrievalResult with deduplicated, re-ranked documents.
    """
    if not results:
        return RetrievalResult(documents=[], query_used="", strategy="merged")

    # Deduplicate by content, keeping the highest score
    seen: dict[str, ScoredDocument] = {}
    total_candidates = 0

    for result in results:
        total_candidates += result.total_candidates
        for doc in result.documents:
            content = doc.chunk.content
            if content not in seen or doc.score > seen[content].score:
                seen[content] = doc

    # Re-rank by score
    merged = sorted(seen.values(), key=lambda d: d.score, reverse=True)
    for rank, doc in enumerate(merged):
        doc.rank = rank

    queries_used = " | ".join(r.query_used for r in results if r.query_used)

    return RetrievalResult(
        documents=merged,
        query_used=queries_used,
        strategy="merged",
        total_candidates=total_candidates,
    )
