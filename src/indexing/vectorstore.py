"""
Vector store factory.

Creates and manages vector stores (FAISS, Chroma) from chunked documents.
This is the final step of the indexing pipeline:

    Loader → Chunker → Embeddings → VectorStore (this file)

In GeneralBot, vector store creation was inside each technique's
create_vector_store() method, always hardcoded to FAISS. Here it's a
standalone factory that supports multiple backends and handles both
creating new stores and loading existing ones.

Key difference between FAISS and Chroma:
    FAISS  — In-memory by default. Fast, simple, no server needed.
             Good for prototyping and small-to-medium datasets.
             Can save/load to disk but doesn't persist automatically.
    Chroma — Persists to disk automatically. Supports metadata filtering.
             Can run as a server for multi-process access.
             Better for production and larger datasets.

Usage:
    from indexing.vectorstore import create_vector_store, load_vector_store
    from config import VectorStoreConfig, EmbeddingConfig

    # Create a new store from documents
    store = create_vector_store(
        documents=chunks,
        embedding_config=EmbeddingConfig(),
        store_config=VectorStoreConfig(store_type="faiss"),
    )

    # Query it
    docs = store.similarity_search("what is RAG?", k=4)

    # Save and reload (FAISS)
    store.save_local("my_index")
    store = load_vector_store(
        embedding_config=EmbeddingConfig(),
        store_config=VectorStoreConfig(store_type="faiss"),
        path="my_index",
    )
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from config import EmbeddingConfig, VectorStoreConfig, VectorStoreType
from indexing.embeddings import get_embedding_model


def create_vector_store(
    documents: list[Document],
    embedding_config: EmbeddingConfig,
    store_config: VectorStoreConfig,
) -> VectorStore:
    """
    Create a vector store from chunked documents.

    This is the main entry point for indexing. It:
        1. Gets the embedding model from config
        2. Creates the vector store backend from config
        3. Embeds and indexes all documents

    Args:
        documents: Chunked documents (output of a chunker).
        embedding_config: Which embedding model to use.
        store_config: Which vector store backend and settings.

    Returns:
        A LangChain VectorStore ready for similarity_search().
    """
    embeddings = get_embedding_model(embedding_config)

    if store_config.store_type == VectorStoreType.FAISS:
        return _create_faiss(documents, embeddings)

    elif store_config.store_type == VectorStoreType.CHROMA:
        return _create_chroma(documents, embeddings, store_config)

    else:
        raise ValueError(
            f"Unknown vector store type: '{store_config.store_type}'. "
            f"Supported: 'faiss', 'chroma'."
        )


def load_vector_store(
    embedding_config: EmbeddingConfig,
    store_config: VectorStoreConfig,
    path: str,
) -> VectorStore:
    """
    Load an existing vector store from disk.

    Use this to reload a previously created index without re-embedding.

    Args:
        embedding_config: Must match the embedding model used to create the store.
        store_config: Which backend to load.
        path: Directory where the store was saved.

    Returns:
        A LangChain VectorStore ready for similarity_search().
    """
    embeddings = get_embedding_model(embedding_config)

    if store_config.store_type == VectorStoreType.FAISS:
        return _load_faiss(embeddings, path)

    elif store_config.store_type == VectorStoreType.CHROMA:
        return _load_chroma(embeddings, path)

    else:
        raise ValueError(
            f"Unknown vector store type: '{store_config.store_type}'. "
            f"Supported: 'faiss', 'chroma'."
        )


# ---------------------------------------------------------------------------
# FAISS
# ---------------------------------------------------------------------------

def _create_faiss(documents: list[Document], embeddings) -> VectorStore:
    """
    Create a FAISS vector store.

    FAISS (Facebook AI Similarity Search) stores vectors in memory and
    uses approximate nearest neighbor search. It's the fastest option
    for small-to-medium datasets.

    The store lives in memory — call store.save_local(path) to persist.
    """
    from langchain_community.vectorstores import FAISS

    return FAISS.from_documents(documents, embeddings)


def _load_faiss(embeddings, path: str) -> VectorStore:
    """Load a FAISS store from a previously saved directory."""
    from langchain_community.vectorstores import FAISS

    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

def _create_chroma(
    documents: list[Document],
    embeddings,
    config: VectorStoreConfig,
) -> VectorStore:
    """
    Create a Chroma vector store.

    Chroma persists to disk automatically when persist_directory is set.
    It also supports metadata filtering on queries:
        store.similarity_search("query", filter={"element_type": "Table"})

    This is useful with UnstructuredChunker — you can filter to only
    retrieve tables, or exclude them.
    """
    from langchain_chroma import Chroma

    kwargs = {
        "documents": documents,
        "embedding": embeddings,
    }

    if config.persist_directory:
        kwargs["persist_directory"] = config.persist_directory

    return Chroma.from_documents(**kwargs)


def _load_chroma(embeddings, path: str) -> VectorStore:
    """Load a Chroma store from a persist directory."""
    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=path,
        embedding_function=embeddings,
    )
