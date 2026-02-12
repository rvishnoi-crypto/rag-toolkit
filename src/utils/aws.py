"""
AWS utilities for loading documents from S3.

Allows RAG techniques to index documents stored in S3 buckets.
Supports multiple file formats: PDF, Markdown, and plain text.

Usage:
    from utils.aws import S3DocumentLoader

    # Load all supported files under a prefix
    loader = S3DocumentLoader(bucket="my-docs", prefix="reports/")
    documents = loader.load()  # downloads and parses .pdf, .md, .txt files

    # Load a single file
    loader = S3DocumentLoader(bucket="my-docs", key="notes/meeting.md")
    documents = loader.load()

Requires: boto3 (pip install boto3)
    AWS credentials configured via env vars, ~/.aws/credentials, or IAM role.
"""

import os
import tempfile
from typing import Optional

from langchain.schema import Document


class S3DocumentLoader:
    """
    Load PDF documents from an S3 bucket.

    Downloads PDFs to a temp directory, then uses PyPDFLoader to parse them.
    Supports loading a single file or all PDFs under a prefix.
    """

    def __init__(
        self,
        bucket: str,
        key: Optional[str] = None,
        prefix: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """
        Args:
            bucket: S3 bucket name.
            key: Specific S3 key to load (e.g. "reports/q4.pdf").
                Provide key for a single file.
            prefix: S3 prefix to list and load all PDFs under
                (e.g. "reports/"). Provide prefix for multiple files.
                One of key or prefix must be provided.
            region: AWS region. Defaults to AWS_DEFAULT_REGION env var.
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 loading. Install with: pip install boto3"
            )

        self._bucket = bucket
        self._key = key
        self._prefix = prefix
        self._region = region or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self._s3 = boto3.client("s3", region_name=self._region)

    def load(self) -> list[Document]:
        """
        Download and parse PDF documents from S3.

        Returns:
            List of LangChain Document objects.
        """
        if self._key:
            return self._load_single(self._key)
        elif self._prefix:
            return self._load_prefix(self._prefix)
        else:
            raise ValueError("Provide either key (single file) or prefix (multiple files).")

    # File extensions we know how to load
    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".markdown"}

    def _load_single(self, key: str) -> list[Document]:
        """Download one file from S3 and parse it with the appropriate loader."""
        ext = os.path.splitext(key)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return []

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            self._s3.download_file(self._bucket, key, tmp.name)
            docs = self._load_file(tmp.name, ext)

            # Add S3 source to metadata
            for doc in docs:
                doc.metadata["source"] = f"s3://{self._bucket}/{key}"

            os.unlink(tmp.name)
            return docs

    def _load_file(self, path: str, ext: str) -> list[Document]:
        """Pick the right LangChain loader based on file extension."""
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            return PyPDFLoader(path).load()

        elif ext in (".md", ".markdown"):
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            return UnstructuredMarkdownLoader(path).load()

        elif ext == ".txt":
            from langchain_community.document_loaders import TextLoader
            return TextLoader(path).load()

        return []

    def _load_prefix(self, prefix: str) -> list[Document]:
        """List all supported files under a prefix and load them."""
        response = self._s3.list_objects_v2(Bucket=self._bucket, Prefix=prefix)
        all_docs = []

        for obj in response.get("Contents", []):
            key = obj["Key"]
            ext = os.path.splitext(key)[1].lower()
            if ext in self.SUPPORTED_EXTENSIONS:
                docs = self._load_single(key)
                all_docs.extend(docs)

        return all_docs
