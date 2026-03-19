"""
Shared pytest fixtures for the RAG Pipeline test suite.
"""

import pytest
from rag_pipeline.document_processor import DocumentProcessor, Document
from rag_pipeline.vector_store import VectorStore, TFIDFVectorizer
from rag_pipeline.response_generator import ResponseGenerator, PromptBuilder
from rag_pipeline.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    {
        "text": (
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn from data without being explicitly programmed. "
            "It uses algorithms to parse data, learn from it, and make informed decisions. "
            "Supervised learning requires labeled training data to build predictive models."
        ),
        "source": "ml_intro",
        "metadata": {"category": "technology"},
    },
    {
        "text": (
            "Natural language processing (NLP) allows computers to understand, "
            "interpret, and generate human language. Applications include sentiment "
            "analysis, machine translation, and question answering systems. "
            "Transformers have revolutionized NLP by enabling large-scale pre-training."
        ),
        "source": "nlp_overview",
        "metadata": {"category": "technology"},
    },
    {
        "text": (
            "Docker is a containerization platform that allows developers to package "
            "applications with all their dependencies into isolated containers. "
            "Containers are lightweight, portable, and ensure consistent environments "
            "across development, testing, and production deployments."
        ),
        "source": "docker_guide",
        "metadata": {"category": "devops"},
    },
    {
        "text": (
            "The Python programming language emphasizes readability and simplicity. "
            "It supports multiple programming paradigms including procedural, "
            "object-oriented, and functional styles. Python is widely used in "
            "data science, web development, and automation."
        ),
        "source": "python_basics",
        "metadata": {"category": "programming"},
    },
    {
        "text": (
            "Retrieval-Augmented Generation (RAG) combines information retrieval "
            "with generative language models. A retriever fetches relevant documents "
            "from a knowledge base, and a generator conditions its output on those "
            "documents, reducing hallucinations and grounding responses in facts."
        ),
        "source": "rag_paper",
        "metadata": {"category": "ai"},
    },
]

LONG_TEXT = " ".join(
    [
        "This is sentence number {}. It contains useful information about topic {}.".format(
            i, i % 5
        )
        for i in range(200)
    ]
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_texts():
    return SAMPLE_TEXTS


@pytest.fixture
def long_text():
    return LONG_TEXT


@pytest.fixture
def processor():
    return DocumentProcessor(chunk_size=100, chunk_overlap=10)


@pytest.fixture
def small_processor():
    return DocumentProcessor(chunk_size=20, chunk_overlap=5)


@pytest.fixture
def sample_documents(processor, sample_texts):
    docs = []
    for item in sample_texts:
        docs.extend(
            processor.process_document(
                item["text"], item["source"], item.get("metadata", {})
            )
        )
    return docs


@pytest.fixture
def vectorizer():
    return TFIDFVectorizer()


@pytest.fixture
def fitted_vectorizer(sample_texts):
    v = TFIDFVectorizer()
    v.fit([item["text"] for item in sample_texts])
    return v


@pytest.fixture
def vector_store(sample_documents):
    vs = VectorStore()
    vs.build_index(sample_documents)
    return vs


@pytest.fixture
def prompt_builder():
    return PromptBuilder(max_context_length=1024)


@pytest.fixture
def response_generator():
    return ResponseGenerator()


@pytest.fixture
def pipeline():
    return RAGPipeline(chunk_size=100, chunk_overlap=10, top_k=3)


@pytest.fixture
def ingested_pipeline(pipeline, sample_texts):
    pipeline.ingest(sample_texts)
    return pipeline
