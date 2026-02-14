"""
Abstract base classes for document loading and chunking.

Why separate BaseLoader and BaseChunker?
    In GeneralBot, every technique handled both loading AND chunking in
    the same class (load_documents did both). That meant if you wanted
    to swap PDF loading for web scraping, you had to rewrite the whole
    technique. By splitting them, you can mix and match:
        loader = PDFLoader()
        chunker = SemanticChunker(config)
        chunks = chunker.chunk(loader.load("doc.pdf"))
"""

from abc import ABC, abstractmethod

from langchain_core.documents import Document

from config import ChunkingConfig


class BaseLoader(ABC):
    """
    Contract for document loaders.

    A loader takes a source (file path, URL, etc.) and returns a list
    of LangChain Document objects — one per page or logical section.

    The loader does NOT chunk. It just gets raw content into memory.
    Chunking is a separate step so you can choose your strategy independently.
    """

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """
        Load documents from a source.

        Args:
            source: Path to file, URL, or other identifier the loader understands.

        Returns:
            List of Document objects with page_content and metadata populated.
        """
        ...


class BaseChunker(ABC):
    """
    Contract for document chunkers.

    A chunker takes loaded Documents and splits them into smaller chunks
    suitable for embedding. Different chunkers use different strategies:
        - RecursiveChunker: splits on paragraphs → sentences → characters
        - SemanticChunker: splits at topic-shift boundaries using embeddings
        - UnstructuredChunker: uses document structure (tables, headers, etc.)

    Every chunker receives a ChunkingConfig so the caller controls
    chunk_size, overlap, and strategy-specific params.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """
        Split documents into chunks.

        Args:
            documents: Raw documents from a loader.

        Returns:
            List of smaller Document objects, each with metadata preserved
            and chunk_index added.
        """
        ...
