"""
Abstract base class for answer generators.

The generator is the final stage — it takes retrieved documents (from
any path: vector, structured, or hybrid) and produces an answer.

In GeneralBot, generation was embedded inside each technique as
inline prompt + LLM call. Here it's its own component so you can:
    - Swap generation strategy without rewriting retrieval
    - Use SimpleGenerator for basic RAG, SelfRAGGenerator for reflection
    - Test generation in isolation with mock retrieval results
"""

from abc import ABC, abstractmethod

from rag_toolkit.models.result import GenerationResult, RetrievalResult


class BaseGenerator(ABC):
    """
    Contract for answer generators.

    Every generator receives a RetrievalResult and returns a GenerationResult.
    The generator doesn't care whether documents came from vector search
    or a SQL query — it just sees documents and produces an answer.
    """

    @abstractmethod
    def generate(self, query: str, retrieval: RetrievalResult) -> GenerationResult:
        """
        Generate an answer from retrieved documents.

        Args:
            query: The original user question.
            retrieval: Documents retrieved by any retrieval path.

        Returns:
            GenerationResult with the answer, sources, and metadata.
        """
        ...
