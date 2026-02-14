"""
Query translation implementations.

Each translator takes a user query and transforms it into one or more
queries optimized for vector retrieval. They all implement BaseTranslator.

In GeneralBot's QueryTransformRAG, these were methods on a single class
(query_rewrite, multi_query, step_back_query, hyde_query). Here each is
its own class so you can compose them freely:

    translator = MultiQueryTranslator(llm_config)
    queries = translator.translate("What is RAG?")
    # → [TranslatedQuery(rewritten="definition of RAG", ...),
    #    TranslatedQuery(rewritten="RAG architecture components", ...),
    #    TranslatedQuery(rewritten="RAG vs fine-tuning", ...)]

The key pattern from GeneralBot we preserve:
    prompt | llm.with_structured_output(PydanticModel)

This gives us typed, validated outputs from the LLM every time.
"""

from langchain_core.prompts import PromptTemplate

from base.translator import BaseTranslator
from config import LLMConfig
from models.query import (
    HyDEDocument,
    MultiQueryExpansion,
    SubQuestions,
    TranslatedQuery,
)
from utils.helpers import get_llm


class QueryRewriter(BaseTranslator):
    """
    Rewrites a query to be more specific and search-friendly.

    The simplest translator — one query in, one (better) query out.
    Good for vague or conversational queries:
        "tell me about that thing with vectors" →
        "How do vector embeddings work in information retrieval systems?"
    """

    def __init__(self, llm_config: LLMConfig = None):
        self._llm = get_llm(llm_config or LLMConfig())
        self._prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are an expert at reformulating user queries to improve "
                "information retrieval.\n\n"
                "Original query: {query}\n\n"
                "Rewrite this query to be more specific and search-friendly "
                "while preserving the original intent. Use precise terminology "
                "and add relevant context.\n\n"
                "Rewritten query:"
            ),
        )

    def translate(self, query: str) -> list[TranslatedQuery]:
        chain = self._prompt | self._llm
        result = chain.invoke({"query": query})
        rewritten = result.content.strip()

        return [TranslatedQuery(original=query, rewritten=rewritten, method="rewrite")]


class MultiQueryTranslator(BaseTranslator):
    """
    Generates multiple query perspectives from different angles.

    The idea: one query might miss relevant documents because the
    user's phrasing doesn't match the document's phrasing. By generating
    3-5 variants, you cast a wider net and merge the results.

    "What is RAG?" →
        ["definition of Retrieval-Augmented Generation",
         "how does RAG architecture work",
         "RAG vs fine-tuning for LLM knowledge"]

    Uses structured output (MultiQueryExpansion) so the LLM returns
    a clean list, not free-form text we'd have to parse.
    """

    def __init__(self, llm_config: LLMConfig = None, num_queries: int = 3):
        self._llm = get_llm(llm_config or LLMConfig())
        self._num_queries = num_queries
        self._prompt = PromptTemplate(
            input_variables=["query", "num_queries"],
            template=(
                "You are an AI assistant helping to generate multiple perspectives "
                "of a question for better information retrieval.\n\n"
                "Original question: {query}\n\n"
                "Generate {num_queries} different versions of this question that:\n"
                "- Approach the topic from different angles\n"
                "- Use different wordings and phrasings\n"
                "- Maintain the same core information need\n"
            ),
        )

    def translate(self, query: str) -> list[TranslatedQuery]:
        chain = self._prompt | self._llm.with_structured_output(MultiQueryExpansion)
        result = chain.invoke({"query": query, "num_queries": self._num_queries})

        return [
            TranslatedQuery(original=query, rewritten=variant, method="multi_query")
            for variant in result.variants
        ]


class StepBackTranslator(BaseTranslator):
    """
    Generates a higher-level abstraction of the query.

    Instead of searching for the specific question, we also search for
    the broader concept. This retrieves foundational context that helps
    answer the specific question.

    "What is the boiling point of water at 2000m altitude?" →
        also searches: "How does altitude affect physical properties of water?"

    Returns BOTH the original and the step-back query so retrieval
    covers both the specific and general angles.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self._llm = get_llm(llm_config or LLMConfig())
        self._prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are an expert at abstraction and reasoning.\n\n"
                "Given this specific question: {query}\n\n"
                "Generate a higher-level, more general question that would help "
                "establish fundamental concepts needed to answer the original question.\n\n"
                "For example:\n"
                "- Specific: 'What is the boiling point of water at 2000m altitude?'\n"
                "- Step-back: 'How does altitude affect the physical properties of water?'\n\n"
                "Step-back question:"
            ),
        )

    def translate(self, query: str) -> list[TranslatedQuery]:
        chain = self._prompt | self._llm
        result = chain.invoke({"query": query})
        step_back = result.content.strip()

        # Return both — the retriever will search for both and merge results
        return [
            TranslatedQuery(original=query, rewritten=query, method="original"),
            TranslatedQuery(original=query, rewritten=step_back, method="step_back"),
        ]


class DecompositionTranslator(BaseTranslator):
    """
    Breaks a complex query into simpler sub-questions.

    Used for analytical queries that require synthesizing information
    from multiple sources. Each sub-question retrieves a different
    piece of the puzzle.

    "How does RAG compare to fine-tuning for domain-specific tasks?" →
        ["What are the strengths of RAG for domain-specific tasks?",
         "What are the strengths of fine-tuning for domain-specific tasks?",
         "What are the trade-offs between RAG and fine-tuning?"]

    Uses structured output (SubQuestions) for clean parsing.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self._llm = get_llm(llm_config or LLMConfig())
        self._prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "Break down this complex query into 2-3 simpler sub-questions "
                "that together would provide a complete answer.\n\n"
                "Query: {query}\n"
            ),
        )

    def translate(self, query: str) -> list[TranslatedQuery]:
        chain = self._prompt | self._llm.with_structured_output(SubQuestions)
        result = chain.invoke({"query": query})

        return [
            TranslatedQuery(original=query, rewritten=sub_q, method="decomposition")
            for sub_q in result.questions
        ]


class HyDETranslator(BaseTranslator):
    """
    Hypothetical Document Embedding (HyDE).

    The most creative translator — instead of rewriting the question,
    it asks the LLM to IMAGINE an ideal answer document. That hypothetical
    document is then used as the search query.

    Why does this work? Questions and answers live in different parts of
    the embedding space. "What is RAG?" is far from a paragraph explaining
    RAG. But a hypothetical answer paragraph IS close to real answer
    paragraphs — so we search using the hypothetical instead.

    "What is RAG?" →
        "Retrieval-Augmented Generation (RAG) is a technique that combines
         information retrieval with language model generation. It works by
         first retrieving relevant documents from a knowledge base using
         semantic search, then feeding those documents as context to an LLM
         to generate a grounded, accurate response..."

    Returns a single TranslatedQuery where rewritten is the hypothetical doc.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self._llm = get_llm(llm_config or LLMConfig())
        self._prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "You are an expert assistant. Generate a detailed, hypothetical "
                "answer to the following question. This answer will be used for "
                "semantic search, so make it comprehensive (~200 words) and "
                "well-structured.\n\n"
                "Question: {query}\n\n"
                "Hypothetical Answer:"
            ),
        )

    def translate(self, query: str) -> list[TranslatedQuery]:
        chain = self._prompt | self._llm.with_structured_output(HyDEDocument)
        result = chain.invoke({"query": query})

        return [TranslatedQuery(original=query, rewritten=result.content, method="hyde")]
