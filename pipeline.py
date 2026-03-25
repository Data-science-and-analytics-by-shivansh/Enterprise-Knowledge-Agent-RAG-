"""
RAG Pipeline Orchestrator.
Ties together DocumentProcessor, VectorStore, and ResponseGenerator
into a single end-to-end pipeline.
"""

import logging
from typing import List, Dict, Any, Optional

from .document_processor import DocumentProcessor, Document
from .vector_store import VectorStore
from .response_generator import ResponseGenerator, PromptBuilder

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest([{"text": "...", "source": "doc1"}])
        result = pipeline.query("What is X?")
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        score_threshold: float = 0.0,
        max_context_length: int = 2048,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.score_threshold = score_threshold

        self.processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vector_store = VectorStore()
        self.response_generator = ResponseGenerator(
            prompt_builder=PromptBuilder(max_context_length)
        )

        self._ingested = False
        self._document_count = 0

    @property
    def is_ready(self) -> bool:
        return self._ingested and self.vector_store.size > 0

    @property
    def document_count(self) -> int:
        return self._document_count

    def ingest(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Ingest a list of raw documents into the pipeline.

        Args:
            documents: List of dicts with 'text', 'source', optional 'metadata'.

        Returns:
            Number of chunks indexed.
        """
        if not documents:
            raise ValueError("Document list cannot be empty")

        chunks = self.processor.process_batch(documents)
        if not chunks:
            raise ValueError("No valid chunks produced from documents")

        self.vector_store.build_index(chunks)
        self._ingested = True
        self._document_count = len(documents)

        logger.info(
            "Ingested %d documents → %d chunks indexed",
            len(documents),
            len(chunks),
        )
        return len(chunks)

    def add_document(
        self,
        text: str,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add a single document to an already-ingested pipeline."""
        if not self._ingested:
            raise RuntimeError("Pipeline not initialized. Call ingest() first.")

        chunks = self.processor.process_document(text, source, metadata)
        for chunk in chunks:
            self.vector_store.add_document(chunk)

        self._document_count += 1
        return len(chunks)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: str = "qa",
    ) -> Dict[str, Any]:
        """
        Run a query through the full RAG pipeline.

        Args:
            question: User question.
            top_k: Override default top_k.
            mode: 'qa' or 'summary'.

        Returns:
            Response dict with answer, sources, confidence, etc.
        """
        if not self.is_ready:
            raise RuntimeError("Pipeline not ready. Call ingest() first.")

        k = top_k or self.top_k
        retrieved = self.vector_store.search(
            question, top_k=k, score_threshold=self.score_threshold
        )

        result = self.response_generator.generate(question, retrieved, mode)
        result["query"] = question
        result["top_k_used"] = k

        logger.debug(
            "Query '%s' → %d docs retrieved, confidence=%.3f",
            question[:50],
            len(retrieved),
            result["confidence"],
        )
        return result

    def reset(self) -> None:
        """Clear all indexed documents."""
        self.vector_store.clear()
        self._ingested = False
        self._document_count = 0
        logger.info("Pipeline reset.")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "document_count": self._document_count,
            "chunk_count": self.vector_store.size,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
        }
