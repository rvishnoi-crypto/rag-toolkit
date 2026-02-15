# Techniques

All three techniques share the same interface:

```python
rag = Technique(pdf_path="doc.pdf", llm_config=LLMConfig(...))
response = rag.query("your question")
```

---

## 1. SimpleRAG

Linear pipeline: retrieve top-k similar chunks, generate an answer.

```python
from techniques import SimpleRAG
from config import LLMConfig

rag = SimpleRAG(
    pdf_path="handbook.pdf",
    llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
)

response = rag.query("What is the vacation policy?")
print(response.answer)
print(response.retrieval.strategy)     # "similarity"
print(response.generation.sources)     # ["handbook.pdf"]
```

**When to use**: Simple Q&A over documents. Fast, cheap — one LLM call per query.

**Metadata**: None (simple pipeline, nothing extra to report).

---

## 2. AdaptiveRAG

Classifies your query first, then picks the best retrieval strategy:

| Query Type | Strategy | What It Does |
|---|---|---|
| Factual | `enhanced_retrieval` | Retrieve many candidates, LLM reranks to keep the best |
| Analytical | `decomposition` | Break query into sub-questions, retrieve for each |
| Opinion | `diverse` | MMR retrieval for diverse perspectives |
| Contextual | `broad` | Retrieve more documents (2x the normal k) |

```python
from techniques import AdaptiveRAG
from config import LLMConfig

rag = AdaptiveRAG(
    pdf_path="handbook.pdf",
    llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
)

response = rag.query("Why is this certification valuable?")
print(response.metadata["query_type"])    # "analytical"
print(response.metadata["strategy"])      # "decomposition"
print(response.metadata["confidence"])    # 0.7
```

**When to use**: Users ask different types of questions and one retrieval strategy isn't enough.

**Metadata**: `query_type`, `strategy`, `confidence`

---

## 3. SelfRAG

The most thorough technique. Reflects at every step:

1. **Decides** if retrieval is even needed
2. **Retrieves** documents
3. **Checks relevance** — rewrites query and retries if docs aren't good enough
4. **Generates** an answer
5. **Checks support** — is the answer grounded in the docs? (hallucination check)
6. **Checks utility** — is the answer actually useful?

```python
from techniques import SelfRAG
from config import LLMConfig

rag = SelfRAG(
    pdf_path="handbook.pdf",
    llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
    max_retries=2,              # rewrite query up to 2 times
    relevance_threshold=0.5,    # minimum avg relevance to use retrieved docs
)

response = rag.query("What topics should I study?")
print(response.metadata["needs_retrieval"])   # True
print(response.metadata["retry_count"])       # 0
print(response.metadata["avg_relevance"])     # 0.77
print(response.metadata["support_score"])     # 1.0 (fully grounded)
print(response.metadata["utility_score"])     # 0.9 (high quality)
```

**When to use**: Answer quality matters more than speed/cost. Hallucination is unacceptable.

**Metadata**: `needs_retrieval`, `retry_count`, `avg_relevance`, `support_score`, `support_level`, `utility_score`

---

## Comparison

| | SimpleRAG | AdaptiveRAG | SelfRAG |
|---|---|---|---|
| LLM calls per query | 1 | 2-4 | 3-8 |
| Retrieval strategy | Fixed (similarity) | Query-aware | Fixed + retry loop |
| Hallucination check | No | No | Yes |
| Query rewriting | No | For analytical queries | On low relevance |
| Best for | Speed, simplicity | Mixed query types | Quality-critical apps |
