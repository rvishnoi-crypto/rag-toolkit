"""Tests for config models — pure Pydantic validation, no API calls."""

import pytest

from config import (
    LLMConfig,
    LLMProvider,
    EmbeddingConfig,
    ChunkingConfig,
    RetrieverConfig,
    VectorStoreConfig,
    VectorStoreType,
    ToolkitConfig,
)


class TestLLMConfig:

    def test_defaults(self):
        config = LLMConfig()
        assert config.provider == LLMProvider.OPENAI
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.0
        assert config.max_tokens == 4000

    def test_anthropic_provider(self):
        config = LLMConfig(provider="anthropic", model_name="claude-sonnet-4-5-20250929")
        assert config.provider == LLMProvider.ANTHROPIC

    def test_temperature_bounds(self):
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=2.0)
        with pytest.raises(Exception):
            LLMConfig(temperature=-0.1)
        with pytest.raises(Exception):
            LLMConfig(temperature=2.1)

    def test_max_tokens_positive(self):
        with pytest.raises(Exception):
            LLMConfig(max_tokens=0)


class TestEmbeddingConfig:

    def test_defaults(self):
        config = EmbeddingConfig()
        assert config.provider == "openai"
        assert config.model_name == "text-embedding-3-small"

    def test_custom_provider(self):
        config = EmbeddingConfig(provider="huggingface", model_name="all-MiniLM-L6-v2")
        assert config.provider == "huggingface"


class TestChunkingConfig:

    def test_defaults(self):
        config = ChunkingConfig()
        assert config.strategy == "recursive"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

    def test_overlap_must_be_less_than_size(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_overlap_cannot_exceed_size(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkingConfig(chunk_size=100, chunk_overlap=200)

    def test_valid_overlap(self):
        config = ChunkingConfig(chunk_size=500, chunk_overlap=50)
        assert config.chunk_overlap == 50


class TestRetrieverConfig:

    def test_defaults(self):
        config = RetrieverConfig()
        assert config.k == 4
        assert config.fetch_k == 20

    def test_fetch_k_auto_adjusts(self):
        """fetch_k should be at least k."""
        config = RetrieverConfig(k=10, fetch_k=5)
        assert config.fetch_k >= config.k


class TestVectorStoreConfig:

    def test_defaults(self):
        config = VectorStoreConfig()
        assert config.store_type == VectorStoreType.FAISS
        assert config.persist_directory is None

    def test_chroma_with_persist(self):
        config = VectorStoreConfig(store_type="chroma", persist_directory="/tmp/chroma")
        assert config.store_type == VectorStoreType.CHROMA
        assert config.persist_directory == "/tmp/chroma"


class TestToolkitConfig:

    def test_defaults(self):
        config = ToolkitConfig()
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.chunking, ChunkingConfig)
        assert isinstance(config.retriever, RetrieverConfig)
        assert isinstance(config.vector_store, VectorStoreConfig)

    def test_override_sub_configs(self):
        config = ToolkitConfig(
            llm=LLMConfig(provider="anthropic", model_name="claude-sonnet-4-5-20250929"),
            chunking=ChunkingConfig(strategy="semantic"),
        )
        assert config.llm.provider == LLMProvider.ANTHROPIC
        assert config.chunking.strategy == "semantic"
        # Others should be defaults
        assert config.embedding.provider == "openai"
