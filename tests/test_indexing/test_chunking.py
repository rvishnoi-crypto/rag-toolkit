"""Tests for chunking — no API calls needed for recursive chunker."""

from langchain.schema import Document

from indexing.chunking import RecursiveChunker, get_chunker
from config import ChunkingConfig


def test_recursive_chunker(sample_documents, chunking_config):
    """RecursiveChunker should split documents into smaller chunks."""
    chunker = RecursiveChunker(chunking_config)
    chunks = chunker.chunk(sample_documents)

    # Should produce at least as many chunks as input docs
    assert len(chunks) >= len(sample_documents)
    # Each chunk should have content
    for chunk in chunks:
        assert chunk.page_content
        assert len(chunk.page_content) <= chunking_config.chunk_size + 50  # small tolerance


def test_recursive_chunker_preserves_metadata(sample_documents, chunking_config):
    """Metadata should survive chunking."""
    chunker = RecursiveChunker(chunking_config)
    chunks = chunker.chunk(sample_documents)

    # At least some chunks should retain source metadata
    sources = [c.metadata.get("source") for c in chunks if c.metadata.get("source")]
    assert len(sources) > 0


def test_get_chunker_recursive():
    """get_chunker('recursive') should return RecursiveChunker."""
    config = ChunkingConfig(strategy="recursive")
    chunker = get_chunker(config)
    assert isinstance(chunker, RecursiveChunker)


def test_chunker_with_small_chunk_size():
    """Small chunk_size should produce more chunks."""
    docs = [Document(page_content="A " * 500, metadata={"source": "test.pdf"})]

    small = ChunkingConfig(strategy="recursive", chunk_size=100, chunk_overlap=20)
    large = ChunkingConfig(strategy="recursive", chunk_size=1000, chunk_overlap=100)

    small_chunks = get_chunker(small).chunk(docs)
    large_chunks = get_chunker(large).chunk(docs)

    assert len(small_chunks) > len(large_chunks)
