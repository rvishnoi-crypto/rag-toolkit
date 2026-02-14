"""
Document chunking implementations.

Takes loaded Documents and splits them into smaller chunks for embedding.
Each chunker implements BaseChunker and is driven by ChunkingConfig.

Choosing a strategy:

    "recursive"     Best for clean text (articles, plain docs). Splits on
                    paragraphs → sentences → words. Fast, no extra deps.

    "semantic"      Best for long docs with mixed topics. Uses embeddings to
                    detect topic shifts. Requires: pip install rag-toolkit[semantic]

    "unstructured"  Best for complex PDFs with tables, charts, images, and
                    mixed layouts. Uses Unstructured.io to parse document
                    structure — each element (paragraph, table, header) becomes
                    its own chunk with type metadata. Tables stay intact.
                    Requires: pip install rag-toolkit[unstructured]

    When to use what:
        Clean text/markdown  → "recursive"
        Long mixed-topic doc → "semantic"
        PDF with tables      → "unstructured"
        PDF with charts/imgs → "unstructured"

Usage:
    from indexing.chunking import get_chunker
    from config import ChunkingConfig

    # Simple text
    chunker = get_chunker(ChunkingConfig(strategy="recursive", chunk_size=500))

    # PDF with tables — pass the file path directly as a document
    chunker = get_chunker(ChunkingConfig(
        strategy="unstructured",
        strategy_kwargs={"pdf_infer_table_structure": True},
    ))

    chunks = chunker.chunk(documents)
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from base.indexer import BaseChunker
from config import ChunkingConfig


class RecursiveChunker(BaseChunker):
    """
    Splits text using a hierarchy of separators.

    RecursiveCharacterTextSplitter tries to split on double newlines first
    (paragraph boundaries), then single newlines, then spaces, then characters.
    This preserves meaning better than fixed-size splitting because it
    respects the document's natural structure.

    This is what GeneralBot's SimpleRAG used internally — we're just
    making it a standalone, reusable component.
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            # These are the default separators — listed here for clarity.
            # It tries each in order: paragraphs → newlines → spaces → chars
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into chunks using recursive character splitting.

        Metadata from the original document is preserved on each chunk,
        with chunk_index added so you know the position within the source.
        """
        chunks = self._splitter.split_documents(documents)

        # Add chunk_index to metadata so we can track ordering
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        return chunks


class SemanticChunker(BaseChunker):
    """
    Splits text at topic-shift boundaries detected by embeddings.

    Instead of splitting at a fixed character count, this computes
    embeddings for sentences and splits where the cosine similarity
    between consecutive sentences drops significantly — indicating
    a topic change.

    Breakpoint types (pass via strategy_kwargs):
        percentile:         split where difference > Xth percentile
        standard_deviation: split where difference > X std deviations
        interquartile:      split using IQR-based outlier detection

    Requires: pip install rag-toolkit[semantic]
    (installs langchain-experimental)
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)

        try:
            from langchain_experimental.text_splitter import (
                SemanticChunker as LCSemanticChunker,
            )
        except ImportError:
            raise ImportError(
                "SemanticChunker requires langchain-experimental. "
                "Install with: pip install rag-toolkit[semantic]"
            )

        # Semantic chunker needs an embedding model to detect topic shifts.
        # We import here (not at module level) to avoid circular deps with
        # indexing/embeddings.py and to keep the semantic extra truly optional.
        from indexing.embeddings import get_embedding_model
        from config import EmbeddingConfig

        embedding_config = config.strategy_kwargs.get("embedding_config")
        if embedding_config and isinstance(embedding_config, dict):
            embedding_config = EmbeddingConfig(**embedding_config)
        embeddings = get_embedding_model(embedding_config or EmbeddingConfig())

        breakpoint_type = config.strategy_kwargs.get("breakpoint_type", "percentile")
        breakpoint_threshold = config.strategy_kwargs.get("breakpoint_threshold", 90)

        self._chunker = LCSemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=breakpoint_threshold,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Split documents at semantic boundaries.

        Combines all documents into one text first (the semantic chunker
        needs continuous text to detect topic shifts), then splits.
        """
        # Combine into single text — semantic chunker needs continuous content
        if len(documents) == 1:
            text = documents[0].page_content
        else:
            text = "\n\n".join(doc.page_content for doc in documents)

        chunks = self._chunker.create_documents([text])

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            # Carry over source metadata from the first document
            if documents:
                for key, value in documents[0].metadata.items():
                    if key not in chunk.metadata:
                        chunk.metadata[key] = value

        return chunks


class UnstructuredChunker(BaseChunker):
    """
    Document-aware chunking for complex PDFs (tables, charts, images).

    Unlike recursive/semantic chunkers that see flat text, this uses
    Unstructured.io to understand document STRUCTURE first:

        1. Partitions the PDF into typed elements:
           Title, NarrativeText, Table, Image, ListItem, etc.
        2. Each element becomes its own chunk with type metadata.
        3. Tables are kept intact (never split mid-row).
        4. Small consecutive elements of the same type are merged
           up to chunk_size to avoid tiny chunks.

    This solves the core problem: a recursive chunker would split
    "| Acme | $2.3M |" across two chunks, destroying the table.
    UnstructuredChunker keeps the entire table as one chunk.

    The documents passed to chunk() should have a "source" metadata
    field with the file path — Unstructured needs to read the original
    file to understand its layout, not just the extracted text.

    strategy_kwargs options:
        pdf_infer_table_structure (bool): Use table detection model.
            Default True. More accurate but slower.
        hi_res (bool): Use high-resolution model for better accuracy
            on scanned docs. Default False.
        combine_under_n_chars (int): Merge consecutive small elements
            shorter than this. Default uses chunk_size // 2.

    Requires: pip install rag-toolkit[unstructured]
    """

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)

        try:
            from unstructured.partition.auto import partition
        except ImportError:
            raise ImportError(
                "UnstructuredChunker requires the unstructured package. "
                "Install with: pip install rag-toolkit[unstructured]"
            )

        self._partition = partition
        self._infer_tables = config.strategy_kwargs.get("pdf_infer_table_structure", True)
        self._hi_res = config.strategy_kwargs.get("hi_res", False)
        self._combine_under = config.strategy_kwargs.get(
            "combine_under_n_chars", config.chunk_size // 2
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Parse document structure and return element-aware chunks.

        Each element (paragraph, table, header) becomes a Document with
        metadata including element_type, so downstream components can
        treat tables differently from narrative text if needed.
        """
        all_chunks: list[Document] = []

        for doc in documents:
            source = doc.metadata.get("source", "")
            elements = self._partition_document(doc, source)
            merged = self._merge_small_elements(elements)
            all_chunks.extend(merged)

        # Add chunk_index across all chunks
        for i, chunk in enumerate(all_chunks):
            chunk.metadata["chunk_index"] = i

        return all_chunks

    def _partition_document(self, doc: Document, source: str) -> list[Document]:
        """
        Use Unstructured to partition a document into typed elements.

        If source is a file path, partitions the file directly (best
        accuracy). Otherwise falls back to partitioning the text content.
        """
        import os

        strategy = "hi_res" if self._hi_res else "auto"

        if source and os.path.isfile(source):
            # Partition from file — Unstructured can read the layout directly
            elements = self._partition(
                filename=source,
                strategy=strategy,
                pdf_infer_table_structure=self._infer_tables,
            )
        else:
            # Fallback: partition from text (loses layout info but still
            # detects element types from formatting cues)
            elements = self._partition(
                text=doc.page_content,
                strategy=strategy,
            )

        # Convert Unstructured elements to LangChain Documents
        chunks = []
        for el in elements:
            # Skip empty elements
            text = str(el).strip()
            if not text:
                continue

            chunks.append(Document(
                page_content=text,
                metadata={
                    "source": source,
                    "element_type": el.category,  # "Title", "Table", "NarrativeText", etc.
                    **doc.metadata,
                },
            ))

        return chunks

    def _merge_small_elements(self, chunks: list[Document]) -> list[Document]:
        """
        Merge consecutive small elements of the same type.

        A document might have 10 short bullet points in a row. Instead of
        10 tiny chunks, merge them into one until we hit chunk_size.
        Tables are NEVER merged — they stay as standalone chunks.
        """
        if not chunks:
            return chunks

        merged: list[Document] = []
        buffer = chunks[0]

        for chunk in chunks[1:]:
            buffer_type = buffer.metadata.get("element_type", "")
            chunk_type = chunk.metadata.get("element_type", "")

            # Never merge tables — they should stay intact
            is_table = buffer_type == "Table" or chunk_type == "Table"

            # Merge if: same type, both small, and neither is a table
            can_merge = (
                not is_table
                and buffer_type == chunk_type
                and len(buffer.page_content) < self._combine_under
                and len(buffer.page_content) + len(chunk.page_content) <= self.config.chunk_size
            )

            if can_merge:
                buffer = Document(
                    page_content=buffer.page_content + "\n\n" + chunk.page_content,
                    metadata=buffer.metadata,
                )
            else:
                merged.append(buffer)
                buffer = chunk

        merged.append(buffer)
        return merged


# ---------------------------------------------------------------------------
# Factory — pick the right chunker from config
# ---------------------------------------------------------------------------

def get_chunker(config: ChunkingConfig) -> BaseChunker:
    """
    Factory that returns the right chunker based on config.strategy.

    This is the main entry point — techniques call this instead of
    instantiating chunkers directly:
        chunker = get_chunker(config.chunking)
        chunks = chunker.chunk(documents)

    Args:
        config: ChunkingConfig with strategy set.

    Returns:
        A BaseChunker implementation.

    Raises:
        ValueError: If the strategy is not recognized.
    """
    strategy = config.strategy.lower()

    if strategy == "recursive":
        return RecursiveChunker(config)

    elif strategy == "semantic":
        return SemanticChunker(config)

    elif strategy == "unstructured":
        return UnstructuredChunker(config)

    else:
        raise ValueError(
            f"Unknown chunking strategy: '{config.strategy}'. "
            f"Built-in strategies: 'recursive', 'semantic', 'unstructured'. "
            f"For custom chunkers, subclass BaseChunker directly."
        )
