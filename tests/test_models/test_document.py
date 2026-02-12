"""Tests for document models — pure Pydantic, no API calls."""

from models.document import Chunk, ChunkMetadata, ScoredDocument


def test_chunk_metadata_defaults():
    meta = ChunkMetadata(source="test.pdf")
    assert meta.source == "test.pdf"
    assert meta.page is None
    assert meta.chunk_index == 0
    assert meta.doc_id == ""


def test_chunk_creation():
    chunk = Chunk(
        content="Hello world",
        metadata=ChunkMetadata(source="test.pdf", page=1, chunk_index=3),
    )
    assert chunk.content == "Hello world"
    assert chunk.metadata.page == 1
    assert chunk.embedding is None


def test_chunk_with_embedding():
    chunk = Chunk(
        content="Hello",
        metadata=ChunkMetadata(source="test.pdf"),
        embedding=[0.1, 0.2, 0.3],
    )
    assert chunk.embedding == [0.1, 0.2, 0.3]


def test_scored_document():
    chunk = Chunk(content="RAG", metadata=ChunkMetadata(source="doc.pdf"))
    doc = ScoredDocument(chunk=chunk, score=0.85, rank=0)
    assert doc.score == 0.85
    assert doc.rank == 0
    assert doc.chunk.content == "RAG"
