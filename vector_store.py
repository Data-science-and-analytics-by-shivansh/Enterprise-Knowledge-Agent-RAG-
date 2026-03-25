"""
Vector Store for RAG Pipeline.
In-memory cosine-similarity search with TF-IDF style embeddings.
Designed for testability without external vector DB dependencies.
"""

import math
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any

from .document_processor import Document


class TFIDFVectorizer:
    """Lightweight TF-IDF vectorizer (no sklearn dependency for core logic)."""

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    def fit(self, documents: List[str]) -> "TFIDFVectorizer":
        if not documents:
            raise ValueError("Cannot fit on empty document list")

        all_tokens = [self._tokenize(doc) for doc in documents]
        vocab_set = {tok for tokens in all_tokens for tok in tokens}
        self.vocab = {tok: idx for idx, tok in enumerate(sorted(vocab_set))}

        n = len(documents)
        df: Dict[str, int] = Counter()
        for tokens in all_tokens:
            for tok in set(tokens):
                df[tok] += 1

        self.idf = {
            tok: math.log((n + 1) / (df.get(tok, 0) + 1)) + 1.0
            for tok in self.vocab
        }
        self._fitted = True
        return self

    def transform(self, text: str) -> List[float]:
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted before transform")
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = [0.0] * len(self.vocab)
        for tok, idx in self.vocab.items():
            if tf[tok] > 0:
                vec[idx] = (tf[tok] / total) * self.idf.get(tok, 1.0)
        return vec

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class VectorStore:
    """
    In-memory vector store for RAG retrieval.
    Supports add, search, delete, and bulk-load operations.
    """

    def __init__(self):
        self._documents: Dict[str, Document] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._vectorizer = TFIDFVectorizer()
        self._fitted = False

    @property
    def size(self) -> int:
        return len(self._documents)

    def build_index(self, documents: List[Document]) -> None:
        """Fit vectorizer and index all documents."""
        if not documents:
            raise ValueError("Cannot build index from empty document list")

        texts = [doc.content for doc in documents]
        self._vectorizer.fit(texts)
        self._fitted = True

        for doc in documents:
            vec = self._vectorizer.transform(doc.content)
            self._documents[doc.doc_id] = doc
            self._vectors[doc.doc_id] = vec

    def add_document(self, document: Document) -> None:
        """Add a single document (vectorizer must already be fitted)."""
        if not self._fitted:
            raise RuntimeError("Index must be built before adding individual documents")
        vec = self._vectorizer.transform(document.content)
        self._documents[document.doc_id] = document
        self._vectors[document.doc_id] = vec

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve top-k documents by cosine similarity.

        Args:
            query: Query string.
            top_k: Number of results to return.
            score_threshold: Minimum similarity score.

        Returns:
            List of (Document, score) tuples sorted by descending score.
        """
        if not self._fitted:
            raise RuntimeError("Index not built. Call build_index first.")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vec = self._vectorizer.transform(query)
        scores: List[Tuple[Document, float]] = []

        for doc_id, doc in self._documents.items():
            sim = TFIDFVectorizer.cosine_similarity(
                query_vec, self._vectors[doc_id]
            )
            if sim >= score_threshold:
                scores.append((doc, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document by ID. Returns True if found and removed."""
        if doc_id in self._documents:
            del self._documents[doc_id]
            del self._vectors[doc_id]
            return True
        return False

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a document by ID."""
        return self._documents.get(doc_id)

    def clear(self) -> None:
        """Remove all documents and reset index."""
        self._documents.clear()
        self._vectors.clear()
        self._fitted = False
        self._vectorizer = TFIDFVectorizer()

    def get_all_documents(self) -> List[Document]:
        return list(self._documents.values())
