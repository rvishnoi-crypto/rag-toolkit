# Quick Start

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Environment

Set your API key (or put it in a `.env` file in the project root):

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Minimal Example

```python
import sys
sys.path.insert(0, "src")

from techniques import SimpleRAG
from config import LLMConfig

rag = SimpleRAG(
    pdf_path="path/to/your/document.pdf",
    llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
)

response = rag.query("What is this document about?")
print(response.answer)
```

One class, one method, one answer.

## The Response Object

Every technique returns the same `RAGResponse`:

```python
response.answer                    # str — the generated answer
response.technique                 # "simple_rag" | "adaptive_rag" | "self_rag"
response.metadata                  # dict — technique-specific info

response.retrieval.documents       # retrieved chunks with scores
response.retrieval.query_used      # actual query sent to vector store
response.retrieval.strategy        # "similarity", "mmr", etc.

response.generation.answer         # same as response.answer
response.generation.sources        # source file names
response.generation.model          # model that produced the answer
```

## Next Steps

- [Techniques](02-techniques.md) — SimpleRAG, AdaptiveRAG, SelfRAG
- [Configuration](03-configuration.md) — customize every component
- [Components API](04-components-api.md) — use individual pieces standalone
