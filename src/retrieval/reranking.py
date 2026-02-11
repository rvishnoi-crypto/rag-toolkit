"""
Document reranking implementations.

Reranking takes already-retrieved documents and re-scores them for
better relevance. The typical pattern is:
    1. Retrieve a large pool (fetch_k=20) with a fast method
    2. Rerank to find the best k=4 from that pool

This two-stage approach gives you both speed (vector search is fast)
and accuracy (reranking is more precise).

In GeneralBot's RerankingRAG, three reranking strategies were methods
on a single class. Here each is standalone and works on any
RetrievalResult regardless of how it was retrieved.

Usage:
    from retrieval.reranking import LLMReranker

    # Retrieve a large pool
    result = retriever.retrieve(query, k=20)

    # Rerank down to the best 4
    reranked = reranker.rerank(query, result, top_k=4)
"""

from langchain.prompts import PromptTemplate

from config import LLMConfig
from models.document import ScoredDocument
from models.result import RelevanceScore, RetrievalResult
from utils.helpers import get_llm


class LLMReranker:
    """
    Uses an LLM to score each document's relevance to the query.

    The most accurate reranking method — the LLM reads both the query
    and the document and judges relevance on a 0-1 scale with reasoning.

    Trade-off: costs one LLM call PER document. For 20 candidates,
    that's 20 calls. Use fetch_k wisely.

    In GeneralBot this used the pattern:
        prompt | llm.with_structured_output(RelevanceScore)

    We preserve that exact pattern — RelevanceScore gives us both a
    score and reasoning, so you can debug why a document ranked where it did.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self._llm = get_llm(llm_config or LLMConfig())
        self._prompt = PromptTemplate(
            input_variables=["query", "document"],
            template=(
                "Rate the relevance of this document to the query.\n\n"
                "Query: {query}\n\n"
                "Document: {document}\n\n"
                "Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant). "
                "Provide brief reasoning."
            ),
        )

    def rerank(
        self,
        query: str,
        result: RetrievalResult,
        top_k: int = 4,
    ) -> RetrievalResult:
        """
        Re-score documents using LLM judgement.

        Args:
            query: The original user query.
            result: RetrievalResult with documents to rerank.
            top_k: How many documents to keep after reranking.

        Returns:
            New RetrievalResult with reranked, trimmed documents.
        """
        chain = self._prompt | self._llm.with_structured_output(RelevanceScore)

        scored: list[tuple[ScoredDocument, float]] = []
        for doc in result.documents:
            try:
                score_result = chain.invoke({
                    "query": query,
                    "document": doc.chunk.content[:2000],  # truncate for token limits
                })
                scored.append((doc, score_result.score))
            except Exception:
                # If LLM fails on a doc, keep original score
                scored.append((doc, doc.score))

        # Sort by LLM score, take top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = []
        for rank, (doc, new_score) in enumerate(scored[:top_k]):
            reranked.append(ScoredDocument(
                chunk=doc.chunk,
                score=new_score,
                rank=rank,
            ))

        return RetrievalResult(
            documents=reranked,
            query_used=result.query_used,
            strategy="llm_reranked",
            total_candidates=len(result.documents),
        )


class DiversityReranker:
    """
    Reranks for diversity using MMR on already-retrieved documents.

    Different from MMRRetriever — that applies MMR during retrieval.
    This applies MMR AFTER retrieval, which is useful when you
    retrieved with similarity search but want diverse final results.

    Uses embeddings to compute similarity between documents and
    penalizes documents that are too similar to already-selected ones.

    lambda_mult controls the trade-off:
        1.0 → pure relevance (no diversity)
        0.5 → balanced
        0.0 → pure diversity
    """

    def __init__(self, embedding_model=None, lambda_mult: float = 0.5):
        """
        Args:
            embedding_model: A LangChain Embeddings instance. If None,
                uses OpenAI embeddings (requires OPENAI_API_KEY).
            lambda_mult: Balance between relevance and diversity.
        """
        if embedding_model is None:
            from indexing.embeddings import get_embedding_model
            from config import EmbeddingConfig
            embedding_model = get_embedding_model(EmbeddingConfig())

        self._embeddings = embedding_model
        self._lambda = lambda_mult

    def rerank(
        self,
        query: str,
        result: RetrievalResult,
        top_k: int = 4,
    ) -> RetrievalResult:
        """
        Rerank for diversity using MMR.

        Embeds the query and all documents, then iteratively selects
        documents that maximize: lambda * relevance - (1-lambda) * redundancy
        """
        if not result.documents:
            return result

        # Embed query and all documents
        contents = [doc.chunk.content for doc in result.documents]
        query_embedding = self._embeddings.embed_query(query)
        doc_embeddings = self._embeddings.embed_documents(contents)

        # Compute query-document similarities
        query_sims = [
            self._cosine_similarity(query_embedding, de) for de in doc_embeddings
        ]

        # MMR selection
        selected_indices: list[int] = []
        remaining = list(range(len(result.documents)))

        for _ in range(min(top_k, len(remaining))):
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                relevance = query_sims[idx]

                # Max similarity to already-selected documents
                if selected_indices:
                    redundancy = max(
                        self._cosine_similarity(doc_embeddings[idx], doc_embeddings[s])
                        for s in selected_indices
                    )
                else:
                    redundancy = 0.0

                mmr_score = self._lambda * relevance - (1 - self._lambda) * redundancy

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        # Build result
        reranked = []
        for rank, idx in enumerate(selected_indices):
            doc = result.documents[idx]
            reranked.append(ScoredDocument(
                chunk=doc.chunk,
                score=query_sims[idx],
                rank=rank,
            ))

        return RetrievalResult(
            documents=reranked,
            query_used=result.query_used,
            strategy=f"diversity_reranked(lambda={self._lambda})",
            total_candidates=len(result.documents),
        )

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
