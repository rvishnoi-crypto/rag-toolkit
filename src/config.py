"""
Configuration for the RAG toolkit.

Split into one config per concern so each stage module only receives
what it needs. ToolkitConfig bundles them all for convenience.

Usage:
    # Full config — pass to a technique
    config = ToolkitConfig()

    # Override specific parts
    config = ToolkitConfig(
        llm=LLMConfig(provider="anthropic", model_name="claude-sonnet-4-5-20250929"),
        chunking=ChunkingConfig(strategy="semantic"),
    )

    # Standalone — use just one piece
    llm_config = LLMConfig(provider="openai", model_name="gpt-4")
"""

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

# Load .env from the project root (walks up from this file to find it).
# This runs once at import time, so any module that does
#   from config import ToolkitConfig
# automatically gets the env vars loaded before anything else happens.
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


# ---------------------------------------------------------------------------
# Enums — for things with a genuinely fixed set of choices
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    """
    Supported LLM providers.

    Using an enum here because LLM providers each need a different
    LangChain class (ChatOpenAI vs ChatAnthropic), so we must know
    the exact set we can instantiate.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class VectorStoreType(str, Enum):
    """Supported vector store backends."""

    FAISS = "faiss"
    CHROMA = "chroma"


# ---------------------------------------------------------------------------
# Per-concern configs
# ---------------------------------------------------------------------------

class LLMConfig(BaseModel):
    """
    LLM configuration.

    Used by: generation/, query/translation.py, query/routing.py, retrieval/reranking.py

    Anything that calls an LLM receives this config. The provider + model_name
    pair determines which LangChain chat model class gets instantiated.
    """

    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="Which LLM provider to use",
    )
    model_name: str = Field(
        default="gpt-4",
        description="Model identifier (e.g. 'gpt-4', 'claude-sonnet-4-5-20250929')",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = deterministic, higher = more creative",
    )
    max_tokens: int = Field(
        default=4000,
        gt=0,
        description="Maximum tokens in the LLM response",
    )


class EmbeddingConfig(BaseModel):
    """
    Embedding model configuration.

    Used by: indexing/embeddings.py

    Provider is an open string (not an enum) because the embedding landscape
    is huge and always growing — OpenAI, HuggingFace sentence-transformers,
    Cohere, Bedrock, Ollama, etc. We don't want users editing our source
    code just to add a new provider.

    The factory in indexing/embeddings.py maps known provider strings to
    LangChain classes and raises a clear error for unknown ones.

    Examples:
        EmbeddingConfig(provider="openai")                         # OpenAI API
        EmbeddingConfig(provider="huggingface", model_name="all-MiniLM-L6-v2")  # local
        EmbeddingConfig(provider="cohere", model_name="embed-english-v3.0")     # Cohere API
    """

    provider: str = Field(
        default="openai",
        description="Embedding provider: 'openai', 'huggingface', 'cohere', 'bedrock', etc.",
    )
    model_name: str = Field(
        default="text-embedding-3-small",
        description="Embedding model identifier",
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs passed to the embedding model constructor (e.g. device, batch_size)",
    )


class ChunkingConfig(BaseModel):
    """
    Document chunking configuration.

    Used by: indexing/chunking.py

    Strategy is an open string because chunking is one of the most
    important decisions in a RAG pipeline and new approaches keep emerging.

    Built-in strategies (handled by our chunking module):
        "recursive"     — RecursiveCharacterTextSplitter. Splits on paragraphs,
                          then sentences, then words. Good baseline for clean text.
        "semantic"      — SemanticChunker. Uses embeddings to detect topic shifts
                          and splits at natural breakpoints. Better for long docs.

    Document-aware strategies (require extra deps, handled by loaders):
        "unstructured"  — Uses Unstructured.io to parse PDFs with tables, charts,
                          images, and mixed layouts into structured elements.
        "markitdown"    — Converts documents to markdown first (preserves headers,
                          tables, lists) then chunks the markdown.

    You can also pass any custom string and provide your own chunker that
    implements the BaseChunker protocol.

    chunk_size and chunk_overlap apply to recursive/semantic strategies.
    For document-aware strategies, the parser decides boundaries and these
    are used as soft guidance.
    """

    strategy: str = Field(
        default="recursive",
        description="Chunking strategy: 'recursive', 'semantic', 'unstructured', 'markitdown', or custom",
    )
    chunk_size: int = Field(
        default=1000,
        gt=0,
        description="Target chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between consecutive chunks",
    )
    # Strategy-specific params — keeps the config extensible without
    # adding a new field every time a new chunker has a new knob
    strategy_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Extra kwargs for the chosen strategy. Examples: "
            "{'breakpoint_type': 'percentile'} for semantic, "
            "{'pdf_infer_table_structure': True} for unstructured"
        ),
    )

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingConfig":
        """Overlap must be smaller than chunk size, otherwise chunks would never advance."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class RetrieverConfig(BaseModel):
    """
    Retrieval configuration.

    Used by: retrieval/search.py, retrieval/reranking.py

    Controls how many documents to fetch and which search strategy to use.
    fetch_k is the initial pool size before reranking narrows it down to k.
    """

    k: int = Field(
        default=4,
        gt=0,
        description="Number of documents to return",
    )
    search_type: str = Field(
        default="similarity",
        description="Search strategy: 'similarity', 'mmr', or custom",
    )
    fetch_k: int = Field(
        default=20,
        gt=0,
        description="Number of candidates to fetch before reranking (should be >= k)",
    )

    @model_validator(mode="after")
    def validate_fetch_k(self) -> "RetrieverConfig":
        """fetch_k must be at least k, otherwise reranking has nothing to work with."""
        if self.fetch_k < self.k:
            self.fetch_k = self.k
        return self


class VectorStoreConfig(BaseModel):
    """
    Vector store configuration.

    Used by: indexing/vectorstore.py

    Controls which backend stores the embeddings. persist_directory is
    only relevant for Chroma (FAISS is in-memory by default).
    """

    store_type: VectorStoreType = Field(
        default=VectorStoreType.FAISS,
        description="Vector store backend",
    )
    persist_directory: Optional[str] = Field(
        default=None,
        description="Directory to persist the vector store (Chroma only)",
    )


# ---------------------------------------------------------------------------
# Top-level config — bundles everything
# ---------------------------------------------------------------------------

class ToolkitConfig(BaseModel):
    """
    Complete toolkit configuration.

    Techniques receive this and pass slices to each stage:
        self.chunker = FixedChunker(config.chunking)
        self.retriever = SimilaritySearch(config.retriever)
        self.generator = SimpleGenerator(config.llm)

    All sub-configs have sensible defaults, so ToolkitConfig() with
    no arguments gives you a working setup out of the box.
    """

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
