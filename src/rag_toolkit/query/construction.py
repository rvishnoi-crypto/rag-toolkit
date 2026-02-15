"""
Query construction: natural language → structured queries.

This is the "structured path" counterpart to translation.py.
While translators rewrite NL queries into better NL queries for
vector search, constructors convert NL into SQL, Cypher, etc.

The constructor needs to know the database schema so it generates
valid queries. It shows the LLM the table names, column types, and
sample data, then asks it to produce SQL.

Usage:
    from rag_toolkit.query.construction import TextToSQLConstructor
    from rag_toolkit.retrieval.structured import SQLRetriever

    retriever = SQLRetriever(db=db)
    constructor = TextToSQLConstructor(
        schema=retriever.schema,
        llm_config=LLMConfig(provider="anthropic", model_name="claude-sonnet-4-5-20250929"),
    )

    # Natural language → SQL
    constructed = constructor.construct("What are the top 3 stocks by price?")
    # → ConstructedQuery(constructed="SELECT symbol, price FROM stocks ORDER BY price DESC LIMIT 3")

    # Execute the SQL
    result = retriever.retrieve_structured(constructed)
"""

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from rag_toolkit.base.constructor import BaseQueryConstructor
from rag_toolkit.config import LLMConfig
from rag_toolkit.models.query import ConstructedQuery, QueryTarget
from rag_toolkit.utils.helpers import get_llm


class SQLOutput(BaseModel):
    """Structured output for text-to-SQL generation."""
    sql: str = Field(description="The SQL query")
    reasoning: str = Field(description="Brief explanation of how you translated the question to SQL")


class TextToSQLConstructor(BaseQueryConstructor):
    """
    Converts natural language questions into SQL queries using an LLM.

    The LLM receives:
        - The database schema (CREATE TABLE statements)
        - The user's question
        - Instructions to produce valid, read-only SQL

    Returns a ConstructedQuery that can be executed by SQLRetriever.

    Safety: the prompt explicitly instructs the LLM to only generate
    SELECT statements — no INSERT, UPDATE, DELETE, or DROP.
    """

    def __init__(self, schema: str, llm_config: LLMConfig = None):
        """
        Args:
            schema: The database schema string (from SQLRetriever.schema
                or SQLDatabase.get_table_info()). Contains CREATE TABLE
                statements so the LLM knows the available tables and columns.
            llm_config: LLM configuration.
        """
        self._llm = get_llm(llm_config or LLMConfig())
        self._schema = schema
        self._prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template=(
                "You are a SQL expert. Given the database schema below, "
                "write a SQL query that answers the user's question.\n\n"
                "Database schema:\n{schema}\n\n"
                "Question: {question}\n\n"
                "Rules:\n"
                "- Only generate SELECT statements (read-only, no modifications).\n"
                "- Use only tables and columns that exist in the schema.\n"
                "- Keep the query simple and efficient.\n"
                "- If the question cannot be answered from the schema, "
                "return a query that gets the closest relevant data.\n"
            ),
        )

    def construct(self, query: str) -> ConstructedQuery:
        """
        Convert a natural language question into a SQL query.

        Args:
            query: The user's question (e.g. "What are the top 3 stocks by price?")

        Returns:
            ConstructedQuery with the generated SQL string.
        """
        chain = self._prompt | self._llm.with_structured_output(SQLOutput)
        result = chain.invoke({
            "schema": self._schema,
            "question": query,
        })

        return ConstructedQuery(
            original=query,
            constructed=result.sql,
            target=QueryTarget.SQL_DATABASE,
            method="text_to_sql",
        )
