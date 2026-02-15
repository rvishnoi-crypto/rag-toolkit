"""
Adaptive RAG: query-aware retrieval strategy selection.

Instead of using the same retrieval approach for every question,
AdaptiveRAG classifies the query and picks the best strategy:

    Factual    → enhanced retrieval + LLM reranking (precision)
    Analytical → query decomposition + multi-retrieval (coverage)
    Opinion    → MMR for diverse perspectives (diversity)
    Contextual → broad retrieval with more documents (breadth)

Internally this is powered by a LangGraph StateGraph that makes
the routing visual and extensible.

Usage as a package:
    from rag_toolkit.techniques import AdaptiveRAG

    rag = AdaptiveRAG(pdf_path="handbook.pdf")
    response = rag.query("Why did revenue decline last quarter?")
    print(response.answer)
    print(response.metadata["query_type"])   # "analytical"
    print(response.metadata["strategy"])     # "decomposition"
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
from rag_toolkit.graphs.adaptive import build_adaptive_graph
from rag_toolkit.indexing.chunking import get_chunker
from rag_toolkit.indexing.vectorstore import create_vector_store
from rag_toolkit.models.result import RAGResponse


class AdaptiveRAG:
    """
    RAG with adaptive retrieval strategies.

    Classifies queries and applies different retrieval strategies.
    Uses a LangGraph under the hood for the routing logic.

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
    ):
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
        self._graph = build_adaptive_graph(
            vector_store=self._vector_store,
            llm_config=self._llm_config,
            retriever_config=self._retriever_config,
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
        Run the AdaptiveRAG pipeline.

        The LangGraph handles: classify → route → retrieve → generate

        Args:
            question: The user's question.

        Returns:
            RAGResponse with answer, retrieval/generation details,
            and metadata about which strategy was chosen.
        """
        result = self._graph.invoke({"query": question})

        return RAGResponse(
            answer=result["answer"],
            retrieval=result.get("retrieval"),
            generation=result.get("generation"),
            technique="adaptive_rag",
            metadata={
                "query_type": result.get("query_type"),
                "strategy": result.get("strategy"),
                "confidence": result.get("confidence"),
            },
        )
