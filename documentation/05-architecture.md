# Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG TOOLKIT                              │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐ │
│  │ INDEXING  │   │  QUERY   │   │ RETRIEVAL │   │ GENERATION │ │
│  │ Pipeline  │   │ Pipeline │   │  Pipeline │   │  Pipeline  │ │
│  └────┬─────┘   └────┬─────┘   └─────┬─────┘   └─────┬──────┘ │
│       │              │               │               │         │
│       ▼              ▼               ▼               ▼         │
│  ┌─────────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐ │
│  │chunking │   │translate │   │  search   │   │  generate  │ │
│  │embed    │   │route     │   │  rerank   │   │  validate  │ │
│  │store    │   │          │   │           │   │            │ │
│  └─────────┘   └──────────┘   └───────────┘   └────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              TECHNIQUES (orchestrators)                   │  │
│  │  SimpleRAG  │  AdaptiveRAG  │  SelfRAG                   │  │
│  │  (linear)   │  (LangGraph)  │  (LangGraph + reflection)  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

The toolkit is organized as **layers**. Each layer is independent and can be used standalone. The **techniques** layer orchestrates the others into complete pipelines.

---

## Layer-by-Layer Breakdown

### 1. Config Layer (`config.py`)

Single file, all Pydantic models. Every component receives typed config — no raw strings passed around.

```
config.py
├── LLMProvider        enum: "openai" | "anthropic"
├── VectorStoreType    enum: "faiss" | "chroma"
├── LLMConfig          provider, model_name, temperature, max_tokens
├── EmbeddingConfig    provider, model_name, model_kwargs
├── ChunkingConfig     strategy, chunk_size, chunk_overlap, strategy_kwargs
├── RetrieverConfig    k, search_type, fetch_k
├── VectorStoreConfig  store_type, persist_directory
└── ToolkitConfig      bundles all of the above
```

### 2. Models Layer (`models/`)

Pydantic data contracts for everything that flows through the system.

```
models/
├── query.py      ← What goes IN
│   ├── QueryType           factual | analytical | opinion | contextual
│   ├── RetrievalPath       vector | structured | hybrid
│   ├── QueryClassification query_type + confidence + reasoning
│   ├── TranslatedQuery     original + rewritten + method
│   ├── MultiQueryExpansion list of variant queries
│   ├── SubQuestions        list of decomposed sub-questions
│   ├── HyDEDocument        hypothetical answer document
│   └── RouteDecision       classification + path + strategy
│
├── document.py   ← What gets STORED
│   ├── ChunkMetadata       source, page, chunk_index
│   ├── Chunk               content + metadata
│   └── ScoredDocument      document + relevance score
│
└── result.py     ← What comes OUT
    ├── RetrievalResult     documents + strategy + scores
    ├── GenerationResult    answer + sources + model
    ├── RelevanceScore      0-1 per-document relevance
    ├── SupportScore        0-1 grounding check
    ├── UtilityScore        0-1 usefulness check
    └── RAGResponse         the unified response object
```

**Key pattern**: The LLM returns structured output directly into these models via `llm.with_structured_output(PydanticModel)` — no string parsing.

### 3. Indexing Layer (`indexing/`)

Ingestion pipeline: document in, vectors out.

```
PDF ──→ chunking.py ──→ embeddings.py ──→ vectorstore.py ──→ FAISS/Chroma
```

- **chunking.py**: `RecursiveChunker` (split by paragraphs/sentences/words), `SemanticChunker` (embedding-based breakpoints), `UnstructuredChunker` (complex PDFs). Factory: `get_chunker(config)`
- **embeddings.py**: Factory `get_embeddings(config)` → OpenAI / HuggingFace / Cohere
- **vectorstore.py**: Factory `create_vector_store()` and `load_vector_store()` → FAISS / Chroma

All use **lazy imports** — you only need the package for the provider you use.

### 4. Query Layer (`query/`)

Transform and route queries before retrieval.

```
User Query
    ├──→ translation.py (WHAT to search for)
    │    ├── QueryRewriter          1 → 1 better query
    │    ├── MultiQueryTranslator   1 → N variants
    │    ├── StepBackTranslator     1 → original + abstract
    │    ├── DecompositionTranslator 1 → sub-questions
    │    └── HyDETranslator         1 → hypothetical answer
    │
    └──→ routing.py (WHERE to search)
         ├── LLMRouter             LLM classifies → picks path
         └── RuleBasedRouter       regex patterns → picks path
```

### 5. Retrieval Layer (`retrieval/`)

Find and rank documents.

- **SimilarityRetriever**: Cosine similarity, returns top-k
- **MMRRetriever**: Maximal Marginal Relevance — relevance balanced with diversity
- **LLMReranker**: LLM reads each doc, scores relevance 0-1, keeps the best
- **DiversityReranker**: MMR-based reranking of already-retrieved docs

### 6. Generation Layer (`generation/`)

Produce and validate answers.

- **SimpleGenerator**: Prompt template + LLM. Has `generate()` (with context) and `generate_without_context()` (fallback)
- **RelevanceChecker**: "Is this document relevant?" → score per doc
- **SupportChecker**: "Is the answer grounded in the docs?" → hallucination check
- **UtilityChecker**: "Is the answer useful?" → quality gate
- **RetrievalDecider**: "Does this query need retrieval at all?"

### 7. Graphs Layer (`graphs/`)

LangGraph state machines that power AdaptiveRAG and SelfRAG.

- **state.py**: `TypedDict` definitions — `AdaptiveState` and `SelfRAGState`
- Each node is a function that reads state, does work, returns updates
- Routing decisions are **conditional edges**

---

## Data Flow Diagrams

### SimpleRAG

```
PDF → PyPDFLoader → RecursiveChunker → OpenAI Embeddings → FAISS
                                                              │
User Query ──────────────────────── SimilarityRetriever ◄─────┘
                                              │
                                        SimpleGenerator
                                              │
                                         RAGResponse
```

### AdaptiveRAG (LangGraph)

```
START → classify_query → route_by_type ─┬─→ factual_retrieve    → generate → END
                                        ├─→ analytical_retrieve  → generate → END
                                        ├─→ opinion_retrieve     → generate → END
                                        └─→ contextual_retrieve  → generate → END
```

Each retrieve node uses a different strategy:
- **factual**: Similarity + LLMReranker (precision)
- **analytical**: DecompositionTranslator + multi-retrieve + merge (coverage)
- **opinion**: MMR retriever (diversity)
- **contextual**: Similarity with k*2 (breadth)

### SelfRAG (LangGraph)

```
START → decide ─┬─→ retrieve → check_relevance ─┬─→ generate_with_context ──→ check_support
                │                                ├─→ rewrite_query → retrieve  (retry loop)
                │                                └─→ generate_without_context → check_support
                └─→ generate_without_context ────────────────────────────────→ check_support
                                                                                    │
                                                                              check_utility
                                                                                    │
                                                                                 finalize
                                                                                    │
                                                                                   END
```

Three possible outcomes after `check_relevance`:
1. Docs are relevant → generate with context
2. Docs aren't relevant, retries left → rewrite query, retry retrieval
3. Docs aren't relevant, no retries left → generate without context

---

## Key Design Patterns

### Factory Pattern
`get_llm(config)`, `get_embeddings(config)`, `get_chunker(config)`, `create_vector_store(...)` — all factories that return the right implementation based on config. Lazy imports keep dependencies minimal.

### Structured Output
`llm.with_structured_output(PydanticModel)` is used everywhere — query classification, translation, reranking, validation. The LLM returns typed, validated objects instead of strings you have to parse.

### State Graph (LangGraph)
AdaptiveRAG and SelfRAG are `StateGraph` instances. Each node is a function `(state) → partial_update`. Conditional edges handle routing. The state (`TypedDict`) accumulates results as it flows through nodes.

### Config Slicing
`ToolkitConfig` bundles everything, but each component only receives the slice it needs: the generator gets `LLMConfig`, the chunker gets `ChunkingConfig`, the retriever gets `RetrieverConfig`. No component knows about configs it doesn't use.
