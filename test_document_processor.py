"""
Unit tests for DocumentProcessor and Document dataclass.
~55 tests covering cleaning, chunking, processing, edge cases.
"""

import pytest
from rag_pipeline.document_processor import DocumentProcessor, Document


# ---------------------------------------------------------------------------
# Document dataclass tests
# ---------------------------------------------------------------------------

class TestDocument:
    def test_document_creation_with_all_fields(self):
        doc = Document(
            doc_id="abc123",
            content="Hello world",
            metadata={"source": "test"},
            chunk_index=0,
            source="test_source",
        )
        assert doc.doc_id == "abc123"
        assert doc.content == "Hello world"
        assert doc.metadata == {"source": "test"}
        assert doc.chunk_index == 0
        assert doc.source == "test_source"

    def test_document_auto_generates_id_when_empty(self):
        doc = Document(doc_id="", content="some content", chunk_index=0)
        assert doc.doc_id != ""
        assert len(doc.doc_id) == 12

    def test_document_default_metadata_is_empty_dict(self):
        doc = Document(doc_id="x", content="content")
        assert doc.metadata == {}

    def test_document_default_source_is_empty(self):
        doc = Document(doc_id="x", content="content")
        assert doc.source == ""

    def test_two_documents_with_same_content_get_same_auto_id(self):
        doc1 = Document(doc_id="", content="same content", chunk_index=0)
        doc2 = Document(doc_id="", content="same content", chunk_index=0)
        assert doc1.doc_id == doc2.doc_id

    def test_different_chunk_index_produces_different_auto_id(self):
        doc1 = Document(doc_id="", content="same content", chunk_index=0)
        doc2 = Document(doc_id="", content="same content", chunk_index=1)
        assert doc1.doc_id != doc2.doc_id


# ---------------------------------------------------------------------------
# DocumentProcessor instantiation
# ---------------------------------------------------------------------------

class TestDocumentProcessorInit:
    def test_default_initialization(self):
        p = DocumentProcessor()
        assert p.chunk_size == 512
        assert p.chunk_overlap == 64
        assert p.min_chunk_length == 50

    def test_custom_initialization(self):
        p = DocumentProcessor(chunk_size=200, chunk_overlap=20, min_chunk_length=10)
        assert p.chunk_size == 200
        assert p.chunk_overlap == 20
        assert p.min_chunk_length == 10

    def test_raises_on_zero_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            DocumentProcessor(chunk_size=0)

    def test_raises_on_negative_chunk_size(self):
        with pytest.raises(ValueError):
            DocumentProcessor(chunk_size=-1)

    def test_raises_on_negative_chunk_overlap(self):
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            DocumentProcessor(chunk_size=100, chunk_overlap=-5)

    def test_raises_when_overlap_equals_chunk_size(self):
        with pytest.raises(ValueError, match="less than chunk_size"):
            DocumentProcessor(chunk_size=100, chunk_overlap=100)

    def test_raises_when_overlap_exceeds_chunk_size(self):
        with pytest.raises(ValueError):
            DocumentProcessor(chunk_size=50, chunk_overlap=60)


# ---------------------------------------------------------------------------
# clean_text tests
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_leading_and_trailing_whitespace(self, processor):
        assert processor.clean_text("  hello  ") == "hello"

    def test_collapses_multiple_spaces(self, processor):
        assert processor.clean_text("hello   world") == "hello world"

    def test_collapses_tabs(self, processor):
        assert processor.clean_text("hello\t\tworld") == "hello world"

    def test_collapses_excessive_newlines(self, processor):
        result = processor.clean_text("line1\n\n\n\nline2")
        assert result == "line1\n\nline2"

    def test_removes_control_characters(self, processor):
        result = processor.clean_text("hello\x00world\x1f!")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_preserves_newlines(self, processor):
        result = processor.clean_text("line1\nline2")
        assert "\n" in result

    def test_raises_on_non_string_input(self, processor):
        with pytest.raises(TypeError):
            processor.clean_text(123)

    def test_empty_string_returns_empty(self, processor):
        assert processor.clean_text("") == ""

    def test_whitespace_only_returns_empty(self, processor):
        assert processor.clean_text("   \t  \n  ") == ""


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_returns_list_of_strings(self, processor):
        result = processor.chunk_text("word " * 200)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_empty_string_returns_empty_list(self, processor):
        assert processor.chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self, processor):
        assert processor.chunk_text("   ") == []

    def test_short_text_returns_single_chunk(self, processor):
        text = " ".join(["word"] * 80)
        result = processor.chunk_text(text)
        assert len(result) == 1

    def test_long_text_produces_multiple_chunks(self, processor, long_text):
        result = processor.chunk_text(long_text)
        assert len(result) > 1

    def test_overlap_causes_content_repetition(self):
        p = DocumentProcessor(chunk_size=10, chunk_overlap=5, min_chunk_length=1)
        words = [f"word{i}" for i in range(25)]
        text = " ".join(words)
        chunks = p.chunk_text(text)
        assert len(chunks) >= 2
        # Consecutive chunks should share some words
        words_c0 = set(chunks[0].split())
        words_c1 = set(chunks[1].split())
        assert len(words_c0 & words_c1) > 0

    def test_min_chunk_length_filters_short_chunks(self):
        p = DocumentProcessor(chunk_size=10, chunk_overlap=0, min_chunk_length=100)
        text = " ".join(["hi"] * 30)
        result = p.chunk_text(text)
        for chunk in result:
            assert len(chunk) >= 100


# ---------------------------------------------------------------------------
# process_document tests
# ---------------------------------------------------------------------------

class TestProcessDocument:
    def test_returns_list_of_document_objects(self, processor, sample_texts):
        docs = processor.process_document(sample_texts[0]["text"], "test_src")
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_source_is_attached_to_each_chunk(self, processor, sample_texts):
        docs = processor.process_document(sample_texts[0]["text"], "my_source")
        for doc in docs:
            assert doc.source == "my_source"
            assert doc.metadata["source"] == "my_source"

    def test_metadata_is_attached_to_each_chunk(self, processor, sample_texts):
        meta = {"author": "Alice", "year": 2024}
        docs = processor.process_document(sample_texts[0]["text"], metadata=meta)
        for doc in docs:
            assert doc.metadata["author"] == "Alice"
            assert doc.metadata["year"] == 2024

    def test_chunk_index_increments_correctly(self, processor, long_text):
        docs = processor.process_document(long_text)
        indices = [d.chunk_index for d in docs]
        assert indices == list(range(len(docs)))

    def test_each_chunk_has_unique_doc_id(self, processor, long_text):
        docs = processor.process_document(long_text)
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))

    def test_empty_text_returns_empty_list(self, processor):
        assert processor.process_document("") == []

    def test_default_source_is_empty_string(self, processor, sample_texts):
        docs = processor.process_document(sample_texts[0]["text"])
        for doc in docs:
            assert doc.source == ""

    def test_default_metadata_is_empty_dict_base(self, processor, sample_texts):
        docs = processor.process_document(sample_texts[0]["text"])
        for doc in docs:
            # metadata will have 'source' and 'chunk_index' from processor
            assert "chunk_index" in doc.metadata


# ---------------------------------------------------------------------------
# process_batch tests
# ---------------------------------------------------------------------------

class TestProcessBatch:
    def test_returns_flat_list(self, processor, sample_texts):
        docs = processor.process_batch(sample_texts)
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_batch_produces_more_docs_than_single(self, processor, sample_texts):
        single = processor.process_document(sample_texts[0]["text"])
        batch = processor.process_batch(sample_texts)
        assert len(batch) >= len(single)

    def test_empty_batch_returns_empty_list(self, processor):
        assert processor.process_batch([]) == []

    def test_sources_preserved_across_batch(self, processor, sample_texts):
        docs = processor.process_batch(sample_texts)
        sources = {d.source for d in docs}
        expected = {item["source"] for item in sample_texts}
        assert expected.issubset(sources)
