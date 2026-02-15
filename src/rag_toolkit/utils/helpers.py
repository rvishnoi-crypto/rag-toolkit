"""
Shared utility functions.

Helpers used across the toolkit — LLM factory, text cleaning, etc.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from rag_toolkit.config import LLMConfig, LLMProvider


def get_llm(config: LLMConfig) -> BaseChatModel:
    """
    Factory that returns a LangChain chat model based on config.

    Same pattern as the embedding factory — lazy imports so you only
    need the package for the provider you actually use.

    This is used everywhere an LLM is needed:
        - query/translation.py (rewriting queries)
        - query/routing.py (classifying queries)
        - retrieval/reranking.py (scoring documents)
        - generation/generate.py (producing answers)

    Args:
        config: LLMConfig with provider, model_name, temperature, max_tokens.

    Returns:
        A LangChain BaseChatModel instance.
    """
    if config.provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    elif config.provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: '{config.provider}'. "
            f"Supported: 'openai', 'anthropic'."
        )


def replace_t_with_space(documents: list) -> list:
    """
    Replace tab characters with spaces in document content.

    Carried over from GeneralBot — useful for cleaning OCR output
    and PDF-extracted text that often has stray tabs.
    """
    for doc in documents:
        doc.page_content = doc.page_content.replace("\t", " ")
    return documents
