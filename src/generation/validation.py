"""
Self-RAG validation: reflection scoring for generated answers.

Self-RAG (Self-Reflective RAG) doesn't just generate an answer — it
checks its own work. After generating, it asks the LLM to score the
answer on three dimensions:

    1. Relevance: Are the retrieved documents relevant to the query?
    2. Support: Is the answer grounded in (supported by) the documents?
    3. Utility: Is the answer actually useful to the user?

In GeneralBot's self_rag.py, these were private methods on the SelfRAG
class. Here they're standalone validators that any technique can use.

Each validator uses structured output (Pydantic model) so the LLM
returns a typed score + reasoning, not free text we'd need to parse.

Usage:
    from generation.validation import RelevanceChecker, SupportChecker, UtilityChecker

    # Check if retrieved docs are relevant
    relevance = RelevanceChecker(llm_config)
    rel_score = relevance.check(query, document)
    # → RelevanceScore(score=0.85, reasoning="Document discusses RAG...")

    # Check if the answer is grounded in context
    support = SupportChecker(llm_config)
    sup_score = support.check(answer, context)
    # → SupportScore(score=0.9, level="fully_supported", reasoning="...")

    # Check if the answer is useful
    utility = UtilityChecker(llm_config)
    util_score = utility.check(query, answer)
    # → UtilityScore(score=0.8, reasoning="Comprehensive answer...")
"""

from langchain.prompts import PromptTemplate

from config import LLMConfig
from models.result import RelevanceScore, RetrievalResult, SupportScore, UtilityScore
from utils.helpers import get_llm


class RelevanceChecker:
    """
    Checks whether retrieved documents are relevant to the query.

    This is the FIRST validation step in Self-RAG. Before we even
    generate an answer, we check: "Did the retriever give us useful
    documents?" If relevance is low, Self-RAG might:
        - Try a different retrieval strategy
        - Rewrite the query and retry
        - Skip retrieval entirely and use parametric knowledge

    Uses structured output → RelevanceScore with score (0-1) + reasoning.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self._llm = get_llm(llm_config or LLMConfig())
        self._prompt = PromptTemplate(
            input_variables=["query", "document"],
            template=(
                "Evaluate the relevance of this document to the query.\n\n"
                "Query: {query}\n\n"
                "Document: {document}\n\n"
                "Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).\n"
                "Provide brief reasoning for your score."
            ),
        )

    def check(self, query: str, document: str) -> RelevanceScore:
        """
        Score a single document's relevance to the query.

        Args:
            query: The user's question.
            document: The document content to evaluate.

        Returns:
            RelevanceScore with score (0-1) and reasoning.
        """
        chain = self._prompt | self._llm.with_structured_output(RelevanceScore)
        return chain.invoke({
            "query": query,
            "document": document[:2000],  # truncate for token limits
        })

    def check_retrieval(self, query: str, retrieval: RetrievalResult) -> list[RelevanceScore]:
        """
        Score all documents in a RetrievalResult.

        Used by Self-RAG to evaluate the entire retrieval before generating.
        Returns one RelevanceScore per document.

        Args:
            query: The user's question.
            retrieval: The retrieval result to evaluate.

        Returns:
            List of RelevanceScore, one per document.
        """
        scores = []
        for doc in retrieval.documents:
            try:
                score = self.check(query, doc.chunk.content)
                scores.append(score)
            except Exception:
                # If scoring fails, give a neutral score
                scores.append(RelevanceScore(score=0.5, reasoning="Scoring failed"))
        return scores


class SupportChecker:
    """
    Checks whether an answer is supported by (grounded in) the context.

    This is the SECOND validation step — after generation. It catches
    hallucinations: the LLM produced an answer, but did it actually
    come from the retrieved documents or did the LLM make things up?

    Returns three levels:
        - "fully_supported" (score >= 0.8): answer is well-grounded
        - "partially_supported" (0.4 <= score < 0.8): some grounding
        - "not_supported" (score < 0.4): likely hallucination

    These thresholds come from the Self-RAG paper and GeneralBot's
    implementation.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self._llm = get_llm(llm_config or LLMConfig())
        self._prompt = PromptTemplate(
            input_variables=["answer", "context"],
            template=(
                "Evaluate how well the answer is supported by the context.\n\n"
                "Context: {context}\n\n"
                "Answer: {answer}\n\n"
                "Score from 0.0 (not supported at all) to 1.0 (fully supported).\n"
                "Classify as 'fully_supported', 'partially_supported', or 'not_supported'.\n"
                "Provide brief reasoning."
            ),
        )

    def check(self, answer: str, context: str) -> SupportScore:
        """
        Score how well an answer is supported by the given context.

        Args:
            answer: The generated answer to evaluate.
            context: The retrieved context the answer should be based on.

        Returns:
            SupportScore with score, level, and reasoning.
        """
        chain = self._prompt | self._llm.with_structured_output(SupportScore)
        return chain.invoke({
            "answer": answer,
            "context": context[:4000],  # more context for grounding check
        })

    def check_generation(self, answer: str, retrieval: RetrievalResult) -> SupportScore:
        """
        Convenience method: check answer against a full RetrievalResult.

        Joins all retrieved documents into one context string and
        checks support.

        Args:
            answer: The generated answer.
            retrieval: The retrieval result used as context.

        Returns:
            SupportScore for the answer against all retrieved documents.
        """
        context = "\n\n".join(
            doc.chunk.content for doc in retrieval.documents
        )
        return self.check(answer, context)


class UtilityChecker:
    """
    Checks whether an answer is actually useful to the user.

    This is the THIRD validation step. An answer can be relevant and
    well-supported but still useless — for example, it might be too
    vague, miss the key point, or not actually answer the question.

    Utility scoring catches these cases. Self-RAG uses the utility
    score to decide if it should regenerate or try a different approach.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self._llm = get_llm(llm_config or LLMConfig())
        self._prompt = PromptTemplate(
            input_variables=["query", "answer"],
            template=(
                "Evaluate the utility of this answer for the given query.\n\n"
                "Query: {query}\n\n"
                "Answer: {answer}\n\n"
                "Score from 0.0 (completely useless) to 1.0 (perfectly useful).\n"
                "Consider: Does it answer the question? Is it clear? Is it complete?\n"
                "Provide brief reasoning."
            ),
        )

    def check(self, query: str, answer: str) -> UtilityScore:
        """
        Score how useful an answer is for the query.

        Args:
            query: The user's question.
            answer: The generated answer to evaluate.

        Returns:
            UtilityScore with score (0-1) and reasoning.
        """
        chain = self._prompt | self._llm.with_structured_output(UtilityScore)
        return chain.invoke({"query": query, "answer": answer})


class RetrievalDecider:
    """
    Decides whether retrieval is even necessary for a query.

    The FIRST step in Self-RAG — before retrieving anything, ask:
    "Does this query need external documents, or can the LLM answer
    from its own knowledge?"

    Simple factual questions like "What is 2+2?" don't need retrieval.
    Knowledge-intensive questions like "What does the company policy
    say about..." definitely do.

    Returns a boolean decision with reasoning.
    """

    def __init__(self, llm_config: LLMConfig = None):
        self._llm = get_llm(llm_config or LLMConfig())
        self._prompt = PromptTemplate(
            input_variables=["query"],
            template=(
                "Determine if this query requires retrieval from external documents "
                "or can be answered from general knowledge alone.\n\n"
                "Query: {query}\n\n"
                "Respond with:\n"
                "- needs_retrieval: true/false\n"
                "- reasoning: brief explanation\n"
                "- confidence: 0.0 to 1.0"
            ),
        )

    def should_retrieve(self, query: str) -> dict:
        """
        Decide whether retrieval is needed for this query.

        Args:
            query: The user's question.

        Returns:
            Dict with 'needs_retrieval' (bool), 'reasoning' (str),
            and 'confidence' (float).
        """
        from pydantic import BaseModel, Field

        class RetrievalDecision(BaseModel):
            needs_retrieval: bool = Field(description="Whether retrieval is needed")
            reasoning: str = Field(description="Why retrieval is/isn't needed")
            confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the decision")

        chain = self._prompt | self._llm.with_structured_output(RetrievalDecision)
        result = chain.invoke({"query": query})
        return {
            "needs_retrieval": result.needs_retrieval,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
        }
