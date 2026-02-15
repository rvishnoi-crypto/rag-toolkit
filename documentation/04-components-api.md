# Components API

You don't have to use the techniques. Every component is standalone and can be used independently.

---

## Query Translation

Transform queries for better retrieval.

```python
from rag_toolkit.query.translation import (
    QueryRewriter,
    MultiQueryTranslator,
    StepBackTranslator,
    DecompositionTranslator,
    HyDETranslator,
)
from rag_toolkit.config import LLMConfig

llm = LLMConfig(provider="openai", model_name="gpt-4")
```

### QueryRewriter — 1 query in, 1 better query out

```python
rewriter = QueryRewriter(llm)
queries = rewriter.translate("tell me about RAG")
# → [TranslatedQuery(rewritten="How does Retrieval-Augmented Generation work?")]
```

### MultiQueryTranslator — 1 query in, N variants out

```python
multi = MultiQueryTranslator(llm, num_queries=3)
queries = multi.translate("What is RAG?")
# → 3 TranslatedQuery objects from different angles
```

### DecompositionTranslator — Complex query in, sub-questions out

```python
decomp = DecompositionTranslator(llm)
queries = decomp.translate("How does RAG compare to fine-tuning?")
# → 2-3 sub-questions
```

### HyDETranslator — Query in, hypothetical answer out

The LLM imagines an ideal answer, which is used as the search query. Works because hypothetical answers are closer in embedding space to real answers than questions are.

```python
hyde = HyDETranslator(llm)
queries = hyde.translate("What is RAG?")
# → TranslatedQuery where rewritten is a ~200 word hypothetical answer
```

### StepBackTranslator — Returns original + abstract version

```python
step = StepBackTranslator(llm)
queries = step.translate("What is the boiling point at 2000m?")
# → [original query, "How does altitude affect physical properties of water?"]
```

---

## Query Routing

Classify queries and decide retrieval path.

### LLMRouter — Production (understands intent)

```python
from rag_toolkit.query.routing import LLMRouter
from rag_toolkit.config import LLMConfig

router = LLMRouter(
    llm_config=LLMConfig(),
    data_sources=["vector_store", "sql_database"],
)
decision = router.route("Show me top 5 customers by revenue")
decision.path                          # RetrievalPath.STRUCTURED
decision.classification.query_type     # QueryType.FACTUAL
decision.strategy                      # "text_to_sql"
```

### RuleBasedRouter — Free, instant, good for prototyping

```python
from rag_toolkit.query.routing import RuleBasedRouter

router = RuleBasedRouter()
decision = router.route("What is RAG?")
decision.classification.query_type     # QueryType.FACTUAL
```

---

## Retrieval

### SimilarityRetriever — Cosine similarity, top-k

```python
from rag_toolkit.retrieval.search import SimilarityRetriever
from rag_toolkit.config import RetrieverConfig

retriever = SimilarityRetriever(vector_store, RetrieverConfig(k=4))
result = retriever.retrieve("What is RAG?")
```

### MMRRetriever — Maximal Marginal Relevance (diversity)

Balances relevance vs diversity. Each new document must be relevant BUT different from already-selected ones.

```python
from rag_toolkit.retrieval.search import MMRRetriever

retriever = MMRRetriever(vector_store, RetrieverConfig(k=4), lambda_mult=0.5)
result = retriever.retrieve("What is RAG?")
```

### LLMReranker — LLM scores each doc's relevance

```python
from rag_toolkit.retrieval.reranking import LLMReranker
from rag_toolkit.config import LLMConfig

reranker = LLMReranker(LLMConfig())
reranked = reranker.rerank("What is RAG?", retrieval_result, top_k=3)
```

---

## Generation

### SimpleGenerator

```python
from rag_toolkit.generation.generate import SimpleGenerator
from rag_toolkit.config import LLMConfig

generator = SimpleGenerator(LLMConfig())

# With context (grounded in retrieved docs)
answer = generator.generate("What is RAG?", retrieval_result)

# Without context (LLM knowledge only — fallback)
answer = generator.generate_without_context("What is RAG?")
```

---

## Validation (Self-RAG components)

### RetrievalDecider — Should we even retrieve?

```python
from rag_toolkit.generation.validation import RetrievalDecider

decider = RetrievalDecider(LLMConfig())
decision = decider.should_retrieve("What is 2 + 2?")
# → {"needs_retrieval": False, "reasoning": "Simple math, no docs needed"}
```

### RelevanceChecker — Are the retrieved docs relevant?

```python
from rag_toolkit.generation.validation import RelevanceChecker

checker = RelevanceChecker(LLMConfig())
scores = checker.check_retrieval("What is RAG?", retrieval_result)
# → [RelevanceScore(score=0.9, reasoning="..."), ...]
```

### SupportChecker — Is the answer grounded in the docs?

```python
from rag_toolkit.generation.validation import SupportChecker

support = SupportChecker(LLMConfig())
score = support.check_generation(answer_text, retrieval_result)
# → SupportScore(score=1.0, level="fully_supported")
```

### UtilityChecker — Is the answer useful?

```python
from rag_toolkit.generation.validation import UtilityChecker

utility = UtilityChecker(LLMConfig())
score = utility.check("What is RAG?", answer_text)
# → UtilityScore(score=0.9, level="high")
```

---

## Indexing

### Chunking

```python
from rag_toolkit.indexing.chunking import get_chunker
from rag_toolkit.config import ChunkingConfig

chunker = get_chunker(ChunkingConfig(strategy="recursive"))
chunks = chunker.chunk(documents)  # list[Document] → list[Document]
```

### Vector Store

```python
from rag_toolkit.indexing.vectorstore import create_vector_store, load_vector_store
from rag_toolkit.config import EmbeddingConfig, VectorStoreConfig

# Create new
store = create_vector_store(chunks, EmbeddingConfig(), VectorStoreConfig())

# Load existing (Chroma)
store = load_vector_store(
    EmbeddingConfig(),
    VectorStoreConfig(store_type="chroma", persist_directory="./chroma_db"),
)
```

---

## S3 Document Loading

```python
from rag_toolkit.utils.aws import S3DocumentLoader

# Single file
loader = S3DocumentLoader(bucket="my-docs", key="reports/q4.pdf")
docs = loader.load()

# All files under a prefix (.pdf, .md, .txt)
loader = S3DocumentLoader(bucket="my-docs", prefix="reports/")
docs = loader.load()
```

Requires `pip install boto3` and AWS credentials.
