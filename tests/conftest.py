"""
Shared test fixtures for the rag-toolkit test suite.

Provides reusable fixtures: sample documents, configs, mock vector stores.
"""

from unittest.mock import MagicMock

import pytest
from langchain.schema import Document

from rag_toolkit.config import LLMConfig, EmbeddingConfig, ChunkingConfig, RetrieverConfig, VectorStoreConfig
from rag_toolkit.models.document import Chunk, ChunkMetadata, ScoredDocument
from rag_toolkit.models.result import RetrievalResult, GenerationResult


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def openai_llm_config():
    """LLM config using OpenAI (default for tests)."""
    return LLMConfig(provider="openai", model_name="gpt-4", temperature=0.0)


@pytest.fixture
def embedding_config():
    return EmbeddingConfig(provider="openai", model_name="text-embedding-3-small")


@pytest.fixture
def chunking_config():
    return ChunkingConfig(strategy="recursive", chunk_size=500, chunk_overlap=50)


@pytest.fixture
def retriever_config():
    return RetrieverConfig(k=4, fetch_k=20)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_documents():
    """Sample LangChain Documents for testing chunkers and vector stores."""
    return [
        Document(page_content="RAG stands for Retrieval-Augmented Generation. It combines retrieval with LLM generation.", metadata={"source": "rag_intro.pdf", "page": 0}),
        Document(page_content="Vector stores like FAISS and Chroma index document embeddings for fast similarity search.", metadata={"source": "rag_intro.pdf", "page": 1}),
        Document(page_content="Self-RAG adds reflection: it checks whether retrieval is needed and validates answer quality.", metadata={"source": "self_rag.pdf", "page": 0}),
        Document(page_content="Adaptive RAG classifies queries into factual, analytical, opinion, or contextual types.", metadata={"source": "adaptive_rag.pdf", "page": 0}),
        Document(page_content="Query decomposition breaks complex questions into simpler sub-questions for better retrieval.", metadata={"source": "techniques.pdf", "page": 2}),
    ]


@pytest.fixture
def sample_chunks():
    """Sample Chunk objects (our internal model)."""
    return [
        Chunk(content="RAG combines retrieval with generation.", metadata=ChunkMetadata(source="doc1.pdf", page=0, chunk_index=0)),
        Chunk(content="FAISS provides fast similarity search.", metadata=ChunkMetadata(source="doc1.pdf", page=1, chunk_index=1)),
        Chunk(content="Self-RAG validates answers at every step.", metadata=ChunkMetadata(source="doc2.pdf", page=0, chunk_index=0)),
    ]


@pytest.fixture
def sample_scored_documents(sample_chunks):
    """Sample ScoredDocument objects."""
    return [
        ScoredDocument(chunk=sample_chunks[0], score=0.95, rank=0),
        ScoredDocument(chunk=sample_chunks[1], score=0.82, rank=1),
        ScoredDocument(chunk=sample_chunks[2], score=0.71, rank=2),
    ]


@pytest.fixture
def sample_retrieval_result(sample_scored_documents):
    """A sample RetrievalResult for testing generators."""
    return RetrievalResult(
        documents=sample_scored_documents,
        query_used="What is RAG?",
        strategy="similarity",
        total_candidates=10,
    )


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store(sample_documents):
    """
    A mock LangChain VectorStore that returns canned results.

    Avoids needing actual embeddings or a real FAISS/Chroma instance
    for unit tests.
    """
    store = MagicMock()

    # similarity_search_with_score returns (Document, score) tuples
    store.similarity_search_with_score.return_value = [
        (sample_documents[0], 0.15),  # low L2 distance = high relevance
        (sample_documents[1], 0.30),
        (sample_documents[2], 0.45),
        (sample_documents[3], 0.60),
    ]

    # similarity_search returns just Documents
    store.similarity_search.return_value = sample_documents[:4]

    # max_marginal_relevance_search returns Documents
    store.max_marginal_relevance_search.return_value = [
        sample_documents[0],
        sample_documents[3],  # diverse selection
        sample_documents[2],
        sample_documents[1],
    ]

    return store
