"""
Answer generation from retrieved context.

This is the final stage of any RAG pipeline: take the query + retrieved
documents and produce a grounded answer.

Here it's one reusable component that any technique can call.

Two modes:
    - WITH context: formats retrieved documents into numbered context,
      instructs the LLM to ground its answer in that context.
    - WITHOUT context: generates from the LLM's parametric knowledge.
      Used by Self-RAG when it decides retrieval isn't needed, or as
      a fallback when no relevant documents are found.

The generator returns a GenerationResult with the answer, sources used,
and the model that produced it — everything a technique needs to build
its final RAGResponse.

Usage:
    from generation.generate import SimpleGenerator

    generator = SimpleGenerator(llm_config=LLMConfig())
    result = generator.generate(query="What is RAG?", retrieval=retrieval_result)
    print(result.answer)
"""

from langchain_core.prompts import PromptTemplate

from base.generator import BaseGenerator
from config import LLMConfig
from models.result import GenerationResult, RetrievalResult
from utils.helpers import get_llm


class SimpleGenerator(BaseGenerator):
    """
    Straightforward RAG generator: context + query → answer.

    How it works:
        1. Takes the retrieved documents and numbers them as context
        2. Builds a prompt: "Given this context, answer the question"
        3. Calls the LLM and returns a GenerationResult

    The prompt explicitly tells the LLM to:
        - Base its answer on the provided context
        - Say "I don't have enough information" if context is insufficient
        - Not make up facts beyond what's in the context

    This is the generator used by SimpleRAG and as the default for
    AdaptiveRAG. Self-RAG uses it too but adds validation on top.
    """

    def __init__(self, llm_config: LLMConfig = None):
        config = llm_config or LLMConfig()
        self._llm = get_llm(config)
        self._model_name = f"{config.provider.value}/{config.model_name}"

        # Prompt for generation WITH retrieved context
        self._context_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template=(
                "Answer the question based on the following context.\n\n"
                "Context:\n{context}\n\n"
                "Question: {query}\n\n"
                "Instructions:\n"
                "- Base your answer on the provided context.\n"
                "- If the context doesn't contain enough information, say so.\n"
                "- Be concise and accurate.\n"
                "- Cite which context pieces you used (by number).\n\n"
                "Answer:"
            ),
        )

        # Prompt for generation WITHOUT context (parametric knowledge only)
        self._no_context_prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "Answer the following question using your knowledge.\n\n"
                "Question: {query}\n\n"
                "Be concise and accurate. If you're unsure, say so.\n\n"
                "Answer:"
            ),
        )

    def generate(
        self,
        query: str,
        retrieval: RetrievalResult,
    ) -> GenerationResult:
        """
        Generate an answer grounded in retrieved documents.

        Takes the retrieval result, formats documents as numbered context,
        and prompts the LLM to answer based on that context.

        Args:
            query: The user's original question.
            retrieval: RetrievalResult with documents to use as context.

        Returns:
            GenerationResult with the answer, sources, and model info.
        """
        if not retrieval.documents:
            return self.generate_without_context(query)

        # Format documents as numbered context blocks.
        # Numbering lets the LLM cite specific pieces: "According to [1]..."
        context_parts = []
        sources = []
        for i, doc in enumerate(retrieval.documents, 1):
            context_parts.append(f"[{i}] {doc.chunk.content}")
            if doc.chunk.metadata.source:
                sources.append(doc.chunk.metadata.source)

        context = "\n\n".join(context_parts)

        chain = self._context_prompt | self._llm
        response = chain.invoke({"context": context, "query": query})

        # LangChain LLMs return AIMessage objects — extract the text
        answer = response.content if hasattr(response, "content") else str(response)

        return GenerationResult(
            answer=answer,
            sources=list(set(sources)),  # deduplicate
            model=self._model_name,
        )

    def generate_without_context(self, query: str) -> GenerationResult:
        """
        Generate an answer using only the LLM's parametric knowledge.

        Used when:
            - Self-RAG decides retrieval isn't needed
            - No relevant documents were found after filtering
            - As a fallback when retrieval fails

        Args:
            query: The user's question.

        Returns:
            GenerationResult with no sources (pure LLM knowledge).
        """
        chain = self._no_context_prompt | self._llm
        response = chain.invoke({"query": query})

        answer = response.content if hasattr(response, "content") else str(response)

        return GenerationResult(
            answer=answer,
            sources=[],
            model=self._model_name,
        )
