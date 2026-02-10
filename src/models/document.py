"""
Document models for the RAG pipeline.

These represent data at each stage:
  Raw document (loaded) → Chunk (split) → ScoredDocument (retrieved + scored)
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """
    Metadata attached to every chunk.

    Why a separate model? Because metadata travels with the chunk through the
    entire pipeline — from indexing to retrieval to generation. Having it typed
    means you can always trust what fields are available downstream.
    """

    source: str = Field(description="Where this chunk came from (file path, URL, etc.)")
    page: Optional[int] = Field(default=None, description="Page number if from a PDF")
    chunk_index: int = Field(default=0, description="Position of this chunk in the original doc")
    doc_id: str = Field(default="", description="Parent document identifier")


class Chunk(BaseModel):
    """
    A single chunk of text after splitting.

    This is the unit that gets embedded and stored in the vector store.
    Think of it as one row in your FAISS/Chroma index.
    """

    content: str = Field(description="The actual text content")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding, populated after embedding step",
    )


class ScoredDocument(BaseModel):
    """
    A chunk with a relevance score attached.

    This is what the retrieval stage returns. The score lets downstream
    components (rerankers, generators) make decisions — e.g. "is this
    relevant enough to use?" or "which docs should I prioritize?"
    """

    chunk: Chunk
    score: float = Field(default=0.0, description="Relevance score (higher = more relevant)")
    rank: int = Field(default=0, description="Position in the result list")
