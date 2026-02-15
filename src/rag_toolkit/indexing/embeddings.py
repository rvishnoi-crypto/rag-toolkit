"""
Embedding model factory.

Returns the right LangChain embedding model based on EmbeddingConfig.
This is the single place that maps provider strings to actual classes.

Supported providers:
    "openai"      → OpenAIEmbeddings (API-based, default)
    "huggingface" → HuggingFaceEmbeddings (local sentence-transformers)
    "cohere"      → CohereEmbeddings (API-based)

Usage:
    from rag_toolkit.indexing.embeddings import get_embedding_model
    from rag_toolkit.config import EmbeddingConfig

    # OpenAI (default)
    model = get_embedding_model(EmbeddingConfig())

    # HuggingFace local model
    model = get_embedding_model(EmbeddingConfig(
        provider="huggingface",
        model_name="all-MiniLM-L6-v2",
    ))

    # Cohere with extra kwargs
    model = get_embedding_model(EmbeddingConfig(
        provider="cohere",
        model_name="embed-english-v3.0",
        model_kwargs={"input_type": "search_document"},
    ))
"""

from langchain_core.embeddings import Embeddings

from rag_toolkit.config import EmbeddingConfig


def get_embedding_model(config: EmbeddingConfig) -> Embeddings:
    """
    Factory that returns a LangChain embedding model based on config.

    Each provider has its own LangChain integration package. We import
    them lazily (inside the if-branch) so you only need the package
    for the provider you actually use.

    Args:
        config: EmbeddingConfig with provider, model_name, and optional model_kwargs.

    Returns:
        A LangChain Embeddings instance ready to call embed_query()/embed_documents().

    Raises:
        ValueError: If the provider is not recognized.
        ImportError: If the required package for the provider is not installed.
    """
    provider = config.provider.lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=config.model_name,
            **config.model_kwargs,
        )

    elif provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "HuggingFace embeddings require langchain-huggingface. "
                "Install with: pip install rag-toolkit[huggingface]"
            )

        return HuggingFaceEmbeddings(
            model_name=config.model_name,
            model_kwargs=config.model_kwargs,
        )

    elif provider == "cohere":
        try:
            from langchain_cohere import CohereEmbeddings
        except ImportError:
            raise ImportError(
                "Cohere embeddings require langchain-cohere. "
                "Install with: pip install rag-toolkit[cohere]"
            )

        return CohereEmbeddings(
            model=config.model_name,
            **config.model_kwargs,
        )

    else:
        raise ValueError(
            f"Unknown embedding provider: '{config.provider}'. "
            f"Supported: 'openai', 'huggingface', 'cohere'. "
            f"For other providers, pass a LangChain Embeddings instance directly."
        )
