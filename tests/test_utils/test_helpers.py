"""Tests for utility helpers."""

from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

from config import LLMConfig, LLMProvider
from utils.helpers import get_llm, replace_t_with_space


class TestGetLLM:

    @patch("utils.helpers.ChatOpenAI")
    def test_openai_provider(self, mock_openai_cls):
        """Should instantiate ChatOpenAI for openai provider."""
        mock_openai_cls.return_value = MagicMock()
        config = LLMConfig(provider="openai", model_name="gpt-4")
        result = get_llm(config)

        mock_openai_cls.assert_called_once_with(
            model="gpt-4", temperature=0.0, max_tokens=4000,
        )

    @patch("utils.helpers.ChatAnthropic")
    def test_anthropic_provider(self, mock_anthropic_cls):
        """Should instantiate ChatAnthropic for anthropic provider."""
        mock_anthropic_cls.return_value = MagicMock()
        config = LLMConfig(provider="anthropic", model_name="claude-sonnet-4-5-20250929")
        result = get_llm(config)

        mock_anthropic_cls.assert_called_once_with(
            model="claude-sonnet-4-5-20250929", temperature=0.0, max_tokens=4000,
        )

    def test_unknown_provider_raises(self):
        """Unknown provider should raise ValueError."""
        # Create a config with valid enum then override
        config = LLMConfig()
        config.provider = "unsupported"
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm(config)


class TestReplaceTabsWithSpace:

    def test_replaces_tabs(self):
        """Tab characters should be replaced with spaces."""
        docs = [
            Document(page_content="hello\tworld", metadata={}),
            Document(page_content="no\ttabs\there", metadata={}),
        ]
        result = replace_t_with_space(docs)

        assert result[0].page_content == "hello world"
        assert result[1].page_content == "no tabs here"

    def test_no_tabs_unchanged(self):
        """Documents without tabs should be unchanged."""
        docs = [Document(page_content="clean text", metadata={})]
        result = replace_t_with_space(docs)
        assert result[0].page_content == "clean text"

    def test_returns_same_list(self):
        """Should modify in-place and return the same list."""
        docs = [Document(page_content="a\tb", metadata={})]
        result = replace_t_with_space(docs)
        assert result is docs

    def test_empty_list(self):
        """Empty list should return empty list."""
        result = replace_t_with_space([])
        assert result == []
