"""
Document Processor for RAG Pipeline.
Handles document loading, chunking, and preprocessing.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """Represents a processed document chunk."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    source: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(
                f"{self.content}{self.chunk_index}".encode()
            ).hexdigest()[:12]


class DocumentProcessor:
    """
    Handles document ingestion, cleaning, and chunking for the RAG pipeline.
    Supports fixed-size chunking with configurable overlap.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_length: int = 50,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def clean_text(self, text: str) -> str:
        """Normalize whitespace, remove control characters."""
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        # Remove control characters except newline/tab
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Collapse multiple spaces/tabs
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks by word count."""
        text = self.clean_text(text)
        if not text:
            return []

        words = text.split()
        if not words:
            return []

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = 1

        i = 0
        while i < len(words):
            chunk_words = words[i: i + self.chunk_size]
            chunk = " ".join(chunk_words)
            if len(chunk) >= self.min_chunk_length:
                chunks.append(chunk)
            i += step

        return chunks

    def process_document(
        self,
        text: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Process raw text into a list of Document chunks.

        Args:
            text: Raw document text.
            source: Identifier for the source document.
            metadata: Optional metadata to attach to each chunk.

        Returns:
            List of Document objects.
        """
        if metadata is None:
            metadata = {}

        chunks = self.chunk_text(text)
        documents = []

        for idx, chunk in enumerate(chunks):
            doc_id = hashlib.md5(
                f"{source}_{idx}_{chunk[:32]}".encode()
            ).hexdigest()[:12]
            doc = Document(
                doc_id=doc_id,
                content=chunk,
                metadata={**metadata, "source": source, "chunk_index": idx},
                chunk_index=idx,
                source=source,
            )
            documents.append(doc)

        return documents

    def process_batch(
        self,
        texts: List[Dict[str, Any]],
    ) -> List[Document]:
        """
        Process a batch of documents.

        Args:
            texts: List of dicts with keys: 'text', 'source', 'metadata'.

        Returns:
            Flat list of all Document chunks.
        """
        all_docs = []
        for item in texts:
            docs = self.process_document(
                text=item.get("text", ""),
                source=item.get("source", ""),
                metadata=item.get("metadata", {}),
            )
            all_docs.extend(docs)
        return all_docs
