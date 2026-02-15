"""
Self-RAG: Self-Reflective Retrieval-Augmented Generation.

Self-RAG doesn't blindly retrieve and generate. It reflects at every step:

    1. "Does this query even need retrieval?" (skip if not)
    2. "Are the retrieved documents relevant?" (rewrite + retry if not)
    3. "Is the answer grounded in the documents?" (catch hallucinations)
    4. "Is the answer actually useful?" (quality check)

The retry loop is the key differentiator — if retrieved docs aren't
relevant, Self-RAG rewrites the query and tries again (up to max_retries)
before falling back to generating without context.

Internally powered by a LangGraph StateGraph with conditional edges.

Usage as a package:
    from rag_toolkit.techniques import SelfRAG

    rag = SelfRAG(pdf_path="handbook.pdf", max_retries=2)
    response = rag.query("What is the vacation policy?")
    print(response.answer)
    print(response.metadata["support_score"])   # grounding check
    print(response.metadata["utility_score"])    # usefulness check
    print(response.metadata["retry_count"])      # how many rewrites needed
"""

from typing import Optional

from langchain_core.vectorstores import VectorStore

from rag_toolkit.config import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    RetrieverConfig,
    VectorStoreConfig,
)
from rag_toolkit.graphs.self_rag import build_self_rag_graph
from rag_toolkit.indexing.chunking import get_chunker
from rag_toolkit.indexing.vectorstore import create_vector_store
from rag_toolkit.models.result import RAGResponse


class SelfRAG:
    """
    Self-Reflective RAG with validation at every step.

    The most thorough technique — validates retrieval quality,
    checks for hallucinations, and scores answer utility. Uses
    a retry loop to rewrite queries when retrieval isn't relevant.

    Can be initialized two ways:
        1. With a document path — handles full pipeline
        2. With a pre-built vector store — skips indexing
    """

    def __init__(
        self,
        pdf_path: Optional[str] = None,
        vector_store: Optional[VectorStore] = None,
        llm_config: Optional[LLMConfig] = None,
        embedding_config: Optional[EmbeddingConfig] = None,
        chunking_config: Optional[ChunkingConfig] = None,
        retriever_config: Optional[RetrieverConfig] = None,
        vectorstore_config: Optional[VectorStoreConfig] = None,
        relevance_threshold: float = 0.5,
        max_retries: int = 2,
    ):
        """
        Args:
            pdf_path: Path to a PDF document to load and index.
            vector_store: A pre-built vector store.
            llm_config: LLM settings for all components.
            embedding_config: Embedding model settings.
            chunking_config: Chunking strategy and sizes.
            retriever_config: Number of documents to retrieve.
            vectorstore_config: Vector store backend settings.
            relevance_threshold: Minimum avg relevance to use retrieved docs.
                Below this, the query gets rewritten and retried.
            max_retries: Max number of query rewrites before falling back
                to generating without context.
        """
        self._llm_config = llm_config or LLMConfig()
        self._embedding_config = embedding_config or EmbeddingConfig()
        self._chunking_config = chunking_config or ChunkingConfig()
        self._retriever_config = retriever_config or RetrieverConfig()
        self._vectorstore_config = vectorstore_config or VectorStoreConfig()

        # Build or accept vector store
        if vector_store is not None:
            self._vector_store = vector_store
        elif pdf_path is not None:
            self._vector_store = self._build_index(pdf_path)
        else:
            raise ValueError("Provide either pdf_path or vector_store.")

        # Build the LangGraph
        self._graph = build_self_rag_graph(
            vector_store=self._vector_store,
            llm_config=self._llm_config,
            retriever_config=self._retriever_config,
            relevance_threshold=relevance_threshold,
            max_retries=max_retries,
        )

    def _build_index(self, pdf_path: str) -> VectorStore:
        """Load PDF → chunk → embed → vector store."""
        from langchain_community.document_loaders import PyPDFLoader
        from rag_toolkit.utils.helpers import replace_t_with_space

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        chunker = get_chunker(self._chunking_config)
        chunks = chunker.chunk(documents)
        chunks = replace_t_with_space(chunks)

        store = create_vector_store(
            documents=chunks,
            embedding_config=self._embedding_config,
            store_config=self._vectorstore_config,
        )
        return store

    def query(self, question: str) -> RAGResponse:
        """
        Run the SelfRAG pipeline with full reflection.

        The LangGraph handles:
            decide → retrieve → check_relevance → [rewrite → retry]
            → generate → check_support → check_utility

        Args:
            question: The user's question.

        Returns:
            RAGResponse with answer, retrieval/generation details,
            and metadata including all reflection scores.
        """
        result = self._graph.invoke({"query": question})

        # Extract reflection scores for metadata
        support = result.get("support_score")
        utility = result.get("utility_score")

        return RAGResponse(
            answer=result["answer"],
            retrieval=result.get("retrieval"),
            generation=result.get("generation"),
            technique="self_rag",
            metadata={
                "needs_retrieval": result.get("needs_retrieval"),
                "retrieval_reasoning": result.get("retrieval_reasoning"),
                "retry_count": result.get("retry_count", 0),
                "avg_relevance": result.get("avg_relevance"),
                "support_score": support.score if support else None,
                "support_level": support.level if support else None,
                "utility_score": utility.score if utility else None,
            },
        )
