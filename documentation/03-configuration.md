# Configuration

Every component is configurable through Pydantic config objects. All have sensible defaults — only override what you need.

---

## LLMConfig

Which LLM to use. Consumed by generation, query translation, routing, reranking.

```python
from rag_toolkit.config import LLMConfig

LLMConfig(provider="openai", model_name="gpt-4")
LLMConfig(provider="anthropic", model_name="claude-sonnet-4-5-20250929")
LLMConfig(provider="openai", model_name="gpt-4", temperature=0.0, max_tokens=4000)
```

| Field | Default | Description |
|---|---|---|
| `provider` | `"openai"` | `"openai"` or `"anthropic"` |
| `model_name` | `"gpt-4"` | Model identifier |
| `temperature` | `0.0` | 0 = deterministic, higher = creative |
| `max_tokens` | `4000` | Max response length |

---

## EmbeddingConfig

Which embedding model to use. Consumed by indexing.

```python
from rag_toolkit.config import EmbeddingConfig

EmbeddingConfig(provider="openai", model_name="text-embedding-3-small")
EmbeddingConfig(provider="huggingface", model_name="all-MiniLM-L6-v2")
EmbeddingConfig(provider="cohere", model_name="embed-english-v3.0")
EmbeddingConfig(provider="huggingface", model_name="...", model_kwargs={"device": "cuda"})
```

| Field | Default | Description |
|---|---|---|
| `provider` | `"openai"` | `"openai"`, `"huggingface"`, `"cohere"` |
| `model_name` | `"text-embedding-3-small"` | Model identifier |
| `model_kwargs` | `{}` | Extra kwargs (e.g. `device`, `batch_size`) |

---

## ChunkingConfig

How to split documents into chunks. Consumed by indexing.

```python
from rag_toolkit.config import ChunkingConfig

ChunkingConfig(strategy="recursive", chunk_size=1000, chunk_overlap=200)
ChunkingConfig(strategy="semantic", strategy_kwargs={"breakpoint_type": "percentile"})
ChunkingConfig(strategy="unstructured")
```

| Field | Default | Description |
|---|---|---|
| `strategy` | `"recursive"` | `"recursive"`, `"semantic"`, `"unstructured"` |
| `chunk_size` | `1000` | Target size in characters |
| `chunk_overlap` | `200` | Overlap between consecutive chunks |
| `strategy_kwargs` | `{}` | Strategy-specific params |

---

## RetrieverConfig

How many documents to retrieve. Consumed by retrieval.

```python
from rag_toolkit.config import RetrieverConfig

RetrieverConfig(k=4, search_type="similarity", fetch_k=20)
```

| Field | Default | Description |
|---|---|---|
| `k` | `4` | Number of documents to return |
| `search_type` | `"similarity"` | `"similarity"` or `"mmr"` |
| `fetch_k` | `20` | Candidates to fetch before reranking (must be >= k) |

---

## VectorStoreConfig

Where to store embeddings. Consumed by indexing.

```python
from rag_toolkit.config import VectorStoreConfig

VectorStoreConfig(store_type="faiss")
VectorStoreConfig(store_type="chroma", persist_directory="./chroma_db")
```

| Field | Default | Description |
|---|---|---|
| `store_type` | `"faiss"` | `"faiss"` (in-memory) or `"chroma"` (persistent) |
| `persist_directory` | `None` | Disk path for Chroma |

---

## ToolkitConfig

Bundles all configs together for convenience:

```python
from rag_toolkit.config import ToolkitConfig, LLMConfig, ChunkingConfig

config = ToolkitConfig(
    llm=LLMConfig(provider="openai", model_name="gpt-4"),
    chunking=ChunkingConfig(strategy="semantic"),
)
# embedding, retriever, vector_store all use their defaults
```

---

## Passing Config to Techniques

Override everything or just what you need:

```python
# Full customization
rag = SimpleRAG(
    pdf_path="doc.pdf",
    llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
    embedding_config=EmbeddingConfig(provider="openai"),
    chunking_config=ChunkingConfig(strategy="recursive", chunk_size=500),
    retriever_config=RetrieverConfig(k=6),
    vectorstore_config=VectorStoreConfig(store_type="faiss"),
)

# Just the parts you care about (rest uses defaults)
rag = SimpleRAG(
    pdf_path="doc.pdf",
    llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
    retriever_config=RetrieverConfig(k=8),
)
```

---

## Environment Variables

```bash
# Required (pick one or both)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional (for S3 loading)
export AWS_DEFAULT_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

Or put them in a `.env` file in the project root — auto-loaded via `python-dotenv`.

---

## Optional Dependencies

Core installs OpenAI + FAISS + Chroma. Extras:

```bash
pip install -e ".[huggingface]"     # HuggingFace embeddings
pip install -e ".[cohere]"          # Cohere embeddings + reranking
pip install -e ".[semantic]"        # Semantic chunking
pip install -e ".[unstructured]"    # Unstructured.io parsing
pip install -e ".[bm25]"           # BM25 sparse retrieval
pip install boto3                   # S3 document loading
```
