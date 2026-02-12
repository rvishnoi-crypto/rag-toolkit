"""
Structured retrieval example — Text-to-SQL with LLM.

Shows the full structured query pipeline:
    1. User asks a natural language question
    2. TextToSQLConstructor uses the LLM + schema to generate SQL
    3. SQLRetriever executes the SQL and returns results
    4. SimpleGenerator produces a natural language answer

No hardcoded SQL — the LLM figures out the right query from the
database schema and the user's question.

Run:
    python examples/stock_data_example.py
"""

import sys
sys.path.insert(0, "src")

import sqlite3
import tempfile
import os

from config import LLMConfig
from generation.generate import SimpleGenerator
from query.construction import TextToSQLConstructor
from retrieval.structured import SQLRetriever


LLM = LLMConfig(provider="anthropic", model_name="claude-sonnet-4-5-20250929")


def create_sample_db() -> str:
    """Create a sample SQLite database with stock data."""
    db_path = os.path.join(tempfile.gettempdir(), "sample_stocks.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks (
            symbol TEXT,
            company TEXT,
            price REAL,
            volume INTEGER,
            sector TEXT
        )
    """)

    cursor.execute("DELETE FROM stocks")
    cursor.executemany(
        "INSERT INTO stocks VALUES (?, ?, ?, ?, ?)",
        [
            ("AAPL", "Apple Inc.", 178.50, 52000000, "Technology"),
            ("GOOGL", "Alphabet Inc.", 141.80, 28000000, "Technology"),
            ("MSFT", "Microsoft Corp.", 378.90, 21000000, "Technology"),
            ("AMZN", "Amazon.com Inc.", 178.25, 45000000, "Consumer"),
            ("TSLA", "Tesla Inc.", 248.50, 110000000, "Automotive"),
            ("JPM", "JPMorgan Chase", 195.20, 8000000, "Finance"),
            ("JNJ", "Johnson & Johnson", 156.30, 6500000, "Healthcare"),
        ],
    )

    conn.commit()
    conn.close()
    return db_path


def main():
    print("=" * 60)
    print("TEXT-TO-SQL PIPELINE")
    print("=" * 60)

    # Step 1: Set up the database and retriever
    db_path = create_sample_db()

    from langchain_community.utilities import SQLDatabase

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    retriever = SQLRetriever(db=db)

    print(f"\nDatabase schema:\n{retriever.schema}")

    # Step 2: Create the text-to-SQL constructor
    # It uses the schema so the LLM knows what tables/columns exist
    constructor = TextToSQLConstructor(
        schema=retriever.schema,
        llm_config=LLM,
    )

    # Step 3: Set up the generator for natural language answers
    generator = SimpleGenerator(llm_config=LLM)

    # Step 4: Ask questions — the LLM generates SQL automatically
    questions = [
        "What are the top 3 most expensive stocks?",
        "Which sectors are represented and how many companies in each?",
        "What is the total trading volume across all stocks?",
        "Which stock has the highest volume?",
    ]

    for question in questions:
        print(f"\nQ: {question}")

        # LLM generates SQL from the question + schema
        constructed = constructor.construct(question)
        print(f"   Generated SQL: {constructed.constructed}")

        # Execute the SQL
        result = retriever.retrieve_structured(constructed)
        print(f"   Raw results ({len(result.documents)} rows):")
        for doc in result.documents:
            print(f"     {doc.chunk.content}")

        # Generate a natural language answer
        answer = generator.generate(question, result)
        print(f"   Answer: {answer.answer}")

    # Clean up
    os.unlink(db_path)


if __name__ == "__main__":
    main()
