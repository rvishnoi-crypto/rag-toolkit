# Project Structure

```
rag-toolkit/
├── pyproject.toml              # Package config, dependencies, build settings
├── README.md
├── LICENSE
├── .gitignore
│
├── src/                        # All source code
│   ├── __init__.py
│   ├── config.py               # All Pydantic config models (LLMConfig, etc.)
│   │
│   ├── models/                 # Data contracts (Pydantic models)
│   │   ├── __init__.py
│   │   ├── query.py            # QueryType, TranslatedQuery, RouteDecision, ...
│   │   ├── document.py         # Chunk, ChunkMetadata, ScoredDocument
│   │   └── result.py           # RetrievalResult, GenerationResult, RAGResponse
│   │
│   ├── base/                   # Abstract base classes
│   │   ├── __init__.py
│   │   ├── retriever.py        # BaseRetriever ABC
│   │   ├── translator.py       # BaseTranslator ABC
│   │   ├── generator.py        # BaseGenerator ABC
│   │   ├── indexer.py          # BaseIndexer ABC
│   │   └── router.py           # BaseRouter ABC
│   │
│   ├── query/                  # Query understanding
│   │   ├── __init__.py
│   │   ├── translation.py      # QueryRewriter, MultiQuery, HyDE, Decomposition, StepBack
│   │   └── routing.py          # LLMRouter, RuleBasedRouter
│   │
│   ├── indexing/               # Document ingestion pipeline
│   │   ├── __init__.py
│   │   ├── chunking.py         # RecursiveChunker, SemanticChunker, UnstructuredChunker
│   │   ├── embeddings.py       # get_embeddings() factory
│   │   └── vectorstore.py      # create_vector_store(), load_vector_store()
│   │
│   ├── retrieval/              # Document retrieval
│   │   ├── __init__.py
│   │   ├── search.py           # SimilarityRetriever, MMRRetriever
│   │   └── reranking.py        # LLMReranker, DiversityReranker
│   │
│   ├── generation/             # Answer generation + validation
│   │   ├── __init__.py
│   │   ├── generate.py         # SimpleGenerator
│   │   └── validation.py       # RelevanceChecker, SupportChecker, UtilityChecker
│   │
│   ├── graphs/                 # LangGraph state machines
│   │   ├── __init__.py
│   │   ├── state.py            # AdaptiveState, SelfRAGState (TypedDict)
│   │   ├── adaptive.py         # build_adaptive_graph() → compiled StateGraph
│   │   └── self_rag.py         # build_self_rag_graph() → compiled StateGraph
│   │
│   ├── techniques/             # User-facing orchestrators
│   │   ├── __init__.py
│   │   ├── simple.py           # SimpleRAG class
│   │   ├── adaptive.py         # AdaptiveRAG class
│   │   └── self_rag.py         # SelfRAG class
│   │
│   └── utils/                  # Shared helpers
│       ├── __init__.py
│       ├── helpers.py          # get_llm() factory, replace_t_with_space()
│       └── aws.py              # S3DocumentLoader
│
├── tests/
│   ├── conftest.py
│   ├── test_models/
│   ├── test_query/
│   ├── test_indexing/
│   ├── test_retrieval/
│   ├── test_generation/
│   └── test_techniques/
│
├── examples/
│   ├── basic_rag.py            # SimpleRAG example
│   ├── adaptive_rag.py         # AdaptiveRAG + SelfRAG example
│   └── stock_data_example.py
│
└── documentation/
    ├── 01-quick-start.md
    ├── 02-techniques.md
    ├── 03-configuration.md
    ├── 04-components-api.md
    ├── 05-architecture.md
    └── 06-project-structure.md   # (this file)
```

---

## Module Dependency Graph

What imports what — follows a strict top-down hierarchy:

```
techniques/   ← top level, imports everything below
    │
    ├── graphs/       ← LangGraph definitions
    │   ├── query/
    │   ├── retrieval/
    │   └── generation/
    │
    ├── indexing/      ← document pipeline
    │
    └── config.py      ← shared by all

models/    ← imported by all layers (data contracts)
utils/     ← imported by all layers (get_llm, helpers)
base/      ← ABCs, imported by implementations
```

No circular imports. Lower layers never import upper layers.

---

## Key Files by Role

| Role | Files |
|---|---|
| **User entry points** | `techniques/simple.py`, `techniques/adaptive.py`, `techniques/self_rag.py` |
| **Configuration** | `config.py` |
| **Data shapes** | `models/query.py`, `models/document.py`, `models/result.py` |
| **LLM factory** | `utils/helpers.py` → `get_llm(config)` |
| **Embedding factory** | `indexing/embeddings.py` → `get_embeddings(config)` |
| **Chunking factory** | `indexing/chunking.py` → `get_chunker(config)` |
| **Vector store factory** | `indexing/vectorstore.py` → `create_vector_store()` |
| **Graph definitions** | `graphs/adaptive.py`, `graphs/self_rag.py` |
| **Graph state** | `graphs/state.py` |
