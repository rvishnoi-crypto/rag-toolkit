# RAG Toolkit

A modular Retrieval-Augmented Generation toolkit.

## Installation

```bash
pip install git+https://github.com/your-org/rag-toolkit.git
```

Or for local development:

```bash
git clone https://github.com/your-org/rag-toolkit.git
cd rag-toolkit
pip install -e .
```

## Quick Start

```python
from rag_toolkit.techniques import SimpleRAG
from rag_toolkit.config import LLMConfig

rag = SimpleRAG(
    pdf_path="path/to/your/document.pdf",
    llm_config=LLMConfig(provider="openai", model_name="gpt-4"),
)

response = rag.query("What is this document about?")
print(response.answer)
```

## Techniques

Three RAG techniques are available, each building on the last:

```python
from rag_toolkit.techniques import SimpleRAG, AdaptiveRAG, SelfRAG
```

- **SimpleRAG** — Basic retrieve-then-generate pipeline. Good default for most use cases.
- **AdaptiveRAG** — Classifies queries (factual, analytical, opinion, contextual) and picks a retrieval strategy automatically.
- **SelfRAG** — Self-reflective pipeline that checks whether retrieval is needed and validates answer quality at every step.

## Configuration

All configuration is optional — sensible defaults are provided. Override what you need:

```python
from rag_toolkit.config import LLMConfig, RetrieverConfig, ChunkingConfig

rag = SimpleRAG(
    pdf_path="doc.pdf",
    llm_config=LLMConfig(provider="anthropic", model_name="claude-sonnet-4-5-20250929"),
    retriever_config=RetrieverConfig(k=6),
    chunking_config=ChunkingConfig(chunk_size=1000, chunk_overlap=200),
)
```

## Environment Variables

Set your API key before running (or add it to a `.env` file in the project root):

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Documentation

See the `documentation/` folder for detailed guides:

- [Quick Start](documentation/01-quick-start.md)
- [Techniques](documentation/02-techniques.md)
- [Configuration](documentation/03-configuration.md)
- [Components API](documentation/04-components-api.md)
- [Architecture](documentation/05-architecture.md)
- [Project Structure](documentation/06-project-structure.md)
