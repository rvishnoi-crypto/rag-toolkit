"""
Simple RAG: the foundational retrieve-then-generate pipeline.

This is the entry point for the rag-toolkit package. Users import this
class and use it directly in their applications:

    from rag_toolkit.techniques import SimpleRAG

    rag = SimpleRAG(pdf_path="my_doc.pdf")
    response = rag.query("What is RAG?")
    print(response.answer)

SimpleRAG handles the full pipeline internally:
    1. Load documents (PDF, text, etc.)
    2. Chunk them into pieces
    3. Embed and index into a vector store
    4. Retrieve relevant chunks for a query
    5. Generate an answer using an LLM

You can also pass a pre-built vector store if you've already indexed
your documents:

    rag = SimpleRAG(vector_store=my_store)
    response = rag.query("What is RAG?")

All configuration is optional — sensible defaults are provided.
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
from rag_toolkit.generation.generate import SimpleGenerator
from rag_toolkit.indexing.chunking import get_chunker
from rag_toolkit.indexing.vectorstore import create_vector_store
from rag_toolkit.models.result import RAGResponse
from rag_toolkit.retrieval.search import SimilarityRetriever


class SimpleRAG:
    """
    Basic RAG pipeline: load → chunk → embed → retrieve → generate.

    This is the simplest technique — no routing, no reflection, no
    conditional logic. It's the baseline that all other techniques
    build upon.

    Can be initialized two ways:
        1. With a document path — handles everything from loading to indexing
        2. With a pre-built vector store — skips to retrieval and generation

    All internal components (chunker, embeddings, retriever, generator)
    are built from our modular layers, so you can customize any piece
    through configuration.
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
        """
        Args:
            pdf_path: Path to a PDF document to load and index.
                If provided, the full pipeline runs: load → chunk → embed → index.
            vector_store: A pre-built LangChain vector store.
                If provided, skips loading/chunking/embedding.
                One of pdf_path or vector_store must be provided.
            llm_config: LLM settings for the generator.
            embedding_config: Embedding model settings.
            chunking_config: Chunking strategy and sizes.
            retriever_config: Number of documents to retrieve (k).
            vectorstore_config: Vector store backend (FAISS, Chroma).
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

        # Build retriever and generator
        self._retriever = SimilarityRetriever(
            vector_store=self._vector_store,
            config=self._retriever_config,
        )
        self._generator = SimpleGenerator(llm_config=self._llm_config)

    def _build_index(self, pdf_path: str) -> VectorStore:
        """
        Full indexing pipeline: load PDF → chunk → embed → vector store.

        This is the internal method that runs when the user passes a
        pdf_path instead of a pre-built vector store. It uses our
        modular indexing components.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            An initialized vector store ready for retrieval.
        """
        from langchain_community.document_loaders import PyPDFLoader
        from rag_toolkit.utils.helpers import replace_t_with_space

        # Step 1: Load the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Step 2: Chunk the documents
        chunker = get_chunker(self._chunking_config)
        chunks = chunker.chunk(documents)

        # Step 3: Clean text (remove tab artifacts from PDF parsing)
        chunks = replace_t_with_space(chunks)

        # Step 4: Embed and index into vector store
        store = create_vector_store(
            documents=chunks,
            embedding_config=self._embedding_config,
            store_config=self._vectorstore_config,
        )

        return store

    def query(self, question: str) -> RAGResponse:
        """
        Run the full SimpleRAG pipeline on a question.

        Steps:
            1. Retrieve the k most relevant document chunks
            2. Generate an answer grounded in those chunks
            3. Return a RAGResponse with answer + metadata

        Args:
            question: The user's question.

        Returns:
            RAGResponse with:
                - answer: The generated answer string
                - retrieval: Full retrieval details (docs, scores, strategy)
                - generation: Generation details (model, sources)
                - technique: "simple_rag"
        """
        # Retrieve
        retrieval = self._retriever.retrieve(question)

        # Generate
        generation = self._generator.generate(question, retrieval)

        # Package
        return RAGResponse(
            answer=generation.answer,
            retrieval=retrieval,
            generation=generation,
            technique="simple_rag",
        )
