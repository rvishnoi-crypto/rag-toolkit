"""
Basic RAG example — the simplest way to use rag-toolkit.

This script:
    1. Loads a PDF document
    2. Indexes it into a FAISS vector store
    3. Asks questions and prints answers

Run:
    python examples/basic_rag.py
"""

import sys
sys.path.insert(0, "src")

from techniques import SimpleRAG
from config import LLMConfig, RetrieverConfig


def main():
    # --- Option 1: Simplest — just pass a PDF with Anthropic ---
    rag = SimpleRAG(
        pdf_path="data/exam_guide.pdf",
        llm_config=LLMConfig(provider="openai", model_name="gpt-5-mini"),
    )

    questions = [
        "What topics are covered in the exam?",
        "How long is the exam?",
        "What is the passing score?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        response = rag.query(q)
        print(f"A: {response.answer}")
        print(f"   Sources: {response.generation.sources}")
        print(f"   Strategy: {response.retrieval.strategy}")

    # --- Option 2: Custom config ---
    custom_rag = SimpleRAG(
        pdf_path="data/exam_guide.pdf",
        llm_config=LLMConfig(
            provider="openai",
            model_name="gpt-5-mini",
            temperature=0.0,
        ),
        retriever_config=RetrieverConfig(k=6),
    )

    response = custom_rag.query("Summarize the key exam requirements.")
    print(f"\nCustom config answer: {response.answer}")


if __name__ == "__main__":
    main()
