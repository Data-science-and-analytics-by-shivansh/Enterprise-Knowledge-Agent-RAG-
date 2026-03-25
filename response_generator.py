"""
Response Generator for RAG Pipeline.
Builds context-grounded answers from retrieved documents.
Includes prompt templating, context assembly, and response scoring.
"""

import re
from typing import List, Tuple, Dict, Any, Optional

from .document_processor import Document


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

QA_TEMPLATE = """You are a knowledgeable assistant. Answer the question using ONLY
the context provided below. If the context does not contain enough information,
say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""

SUMMARIZE_TEMPLATE = """Summarize the following context in 2-3 sentences.

Context:
{context}

Summary:"""


class PromptBuilder:
    """Assembles prompts from retrieved context chunks."""

    def __init__(self, max_context_length: int = 2048):
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        self.max_context_length = max_context_length

    def build_context(
        self,
        retrieved: List[Tuple[Document, float]],
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Concatenate document chunks into a single context string,
        respecting max_context_length.
        """
        if not retrieved:
            return ""

        parts = []
        total_len = 0

        for doc, score in retrieved:
            chunk = f"[Source: {doc.source or 'unknown'}]\n{doc.content}"
            if total_len + len(chunk) > self.max_context_length:
                remaining = self.max_context_length - total_len
                if remaining > 100:
                    parts.append(chunk[:remaining])
                break
            parts.append(chunk)
            total_len += len(chunk) + len(separator)

        return separator.join(parts)

    def build_qa_prompt(
        self,
        question: str,
        retrieved: List[Tuple[Document, float]],
    ) -> str:
        """Build a question-answering prompt."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        context = self.build_context(retrieved)
        return QA_TEMPLATE.format(context=context, question=question.strip())

    def build_summary_prompt(
        self,
        retrieved: List[Tuple[Document, float]],
    ) -> str:
        """Build a summarization prompt."""
        context = self.build_context(retrieved)
        return SUMMARIZE_TEMPLATE.format(context=context)


class ResponseGenerator:
    """
    Generates structured responses for the RAG pipeline.
    In production this wraps an LLM; here it provides deterministic
    rule-based responses for testing and demonstration.
    """

    def __init__(
        self,
        prompt_builder: Optional[PromptBuilder] = None,
        max_context_length: int = 2048,
    ):
        self.prompt_builder = prompt_builder or PromptBuilder(max_context_length)

    def generate(
        self,
        question: str,
        retrieved: List[Tuple[Document, float]],
        mode: str = "qa",
    ) -> Dict[str, Any]:
        """
        Generate a response given a question and retrieved context.

        Args:
            question: The user's question.
            retrieved: List of (Document, score) tuples from VectorStore.
            mode: 'qa' or 'summary'.

        Returns:
            Dict with keys: answer, prompt, sources, num_sources, confidence.
        """
        if mode not in ("qa", "summary"):
            raise ValueError(f"Unknown mode '{mode}'. Use 'qa' or 'summary'.")

        if mode == "qa":
            prompt = self.prompt_builder.build_qa_prompt(question, retrieved)
        else:
            prompt = self.prompt_builder.build_summary_prompt(retrieved)

        answer = self._mock_llm_response(question, retrieved, mode)
        sources = self._extract_sources(retrieved)
        confidence = self._compute_confidence(retrieved)

        return {
            "answer": answer,
            "prompt": prompt,
            "sources": sources,
            "num_sources": len(retrieved),
            "confidence": confidence,
            "mode": mode,
        }

    def _mock_llm_response(
        self,
        question: str,
        retrieved: List[Tuple[Document, float]],
        mode: str,
    ) -> str:
        """
        Deterministic mock response for testing.
        In production, replace with actual LLM call.
        """
        if not retrieved:
            return "I don't have enough information to answer this question."

        top_doc, top_score = retrieved[0]

        if mode == "summary":
            sentences = re.split(r"(?<=[.!?])\s+", top_doc.content)
            return " ".join(sentences[:2]) if sentences else top_doc.content[:200]

        # Simple keyword match for qa mode
        question_lower = question.lower()
        for doc, score in retrieved:
            sentences = re.split(r"(?<=[.!?])\s+", doc.content)
            for sent in sentences:
                if any(word in sent.lower() for word in question_lower.split()
                       if len(word) > 3):
                    return sent.strip()

        return f"Based on the retrieved context: {top_doc.content[:200]}..."

    def _extract_sources(
        self, retrieved: List[Tuple[Document, float]]
    ) -> List[Dict[str, Any]]:
        return [
            {
                "doc_id": doc.doc_id,
                "source": doc.source,
                "score": round(score, 4),
                "chunk_index": doc.chunk_index,
            }
            for doc, score in retrieved
        ]

    def _compute_confidence(
        self, retrieved: List[Tuple[Document, float]]
    ) -> float:
        if not retrieved:
            return 0.0
        scores = [score for _, score in retrieved]
        return round(sum(scores) / len(scores), 4)
