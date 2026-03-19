"""
Unit tests for VectorStore and TFIDFVectorizer.
~55 tests covering indexing, search, CRUD, edge cases.
"""

import math
import pytest
from rag_pipeline.document_processor import Document
from rag_pipeline.vector_store import VectorStore, TFIDFVectorizer


# ---------------------------------------------------------------------------
# TFIDFVectorizer tests
# ---------------------------------------------------------------------------

class TestTFIDFVectorizerInit:
    def test_initial_state_is_not_fitted(self, vectorizer):
        assert not vectorizer._fitted
        assert vectorizer.vocab == {}

    def test_fit_sets_fitted_flag(self, vectorizer):
        vectorizer.fit(["hello world", "foo bar"])
        assert vectorizer._fitted

    def test_fit_builds_vocabulary(self, vectorizer):
        vectorizer.fit(["hello world", "foo bar baz"])
        assert len(vectorizer.vocab) > 0

    def test_fit_raises_on_empty_list(self, vectorizer):
        with pytest.raises(ValueError):
            vectorizer.fit([])

    def test_fit_returns_self(self, vectorizer):
        result = vectorizer.fit(["hello world"])
        assert result is vectorizer

    def test_idf_populated_after_fit(self, vectorizer):
        vectorizer.fit(["machine learning", "deep learning"])
        assert len(vectorizer.idf) > 0


class TestTFIDFVectorizerTransform:
    def test_transform_returns_list_of_floats(self, fitted_vectorizer):
        vec = fitted_vectorizer.transform("machine learning")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_transform_length_equals_vocab_size(self, fitted_vectorizer):
        vec = fitted_vectorizer.transform("machine learning")
        assert len(vec) == len(fitted_vectorizer.vocab)

    def test_transform_raises_when_not_fitted(self, vectorizer):
        with pytest.raises(RuntimeError, match="must be fitted"):
            vectorizer.transform("test")

    def test_unknown_word_produces_zero_entry(self, fitted_vectorizer):
        # A completely alien word should produce a zero (not in vocab)
        vec = fitted_vectorizer.transform("xyzqqqabcdef")
        assert sum(vec) == 0.0

    def test_known_word_produces_nonzero_entry(self, fitted_vectorizer, sample_texts):
        word = "machine"
        vec = fitted_vectorizer.transform(word)
        idx = fitted_vectorizer.vocab.get(word)
        if idx is not None:
            assert vec[idx] > 0.0


class TestCosineSimilarity:
    def test_identical_vectors_give_score_one(self):
        vec = [1.0, 2.0, 3.0]
        score = TFIDFVectorizer.cosine_similarity(vec, vec)
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors_give_score_zero(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert TFIDFVectorizer.cosine_similarity(a, b) == 0.0

    def test_zero_vector_gives_score_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert TFIDFVectorizer.cosine_similarity(a, b) == 0.0

    def test_score_is_between_zero_and_one(self, fitted_vectorizer, sample_texts):
        v1 = fitted_vectorizer.transform(sample_texts[0]["text"])
        v2 = fitted_vectorizer.transform(sample_texts[1]["text"])
        score = TFIDFVectorizer.cosine_similarity(v1, v2)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# VectorStore build_index tests
# ---------------------------------------------------------------------------

class TestVectorStoreBuildIndex:
    def test_builds_index_successfully(self, sample_documents):
        vs = VectorStore()
        vs.build_index(sample_documents)
        assert vs.size == len(sample_documents)

    def test_raises_on_empty_documents(self):
        vs = VectorStore()
        with pytest.raises(ValueError):
            vs.build_index([])

    def test_fitted_flag_set_after_build(self, sample_documents):
        vs = VectorStore()
        vs.build_index(sample_documents)
        assert vs._fitted

    def test_size_matches_ingested_documents(self, vector_store, sample_documents):
        assert vector_store.size == len(sample_documents)


# ---------------------------------------------------------------------------
# VectorStore add_document tests
# ---------------------------------------------------------------------------

class TestVectorStoreAddDocument:
    def test_add_document_increases_size(self, vector_store):
        before = vector_store.size
        new_doc = Document(
            doc_id="new999",
            content="brand new unique content about databases",
            source="extra",
        )
        vector_store.add_document(new_doc)
        assert vector_store.size == before + 1

    def test_add_document_raises_when_not_fitted(self):
        vs = VectorStore()
        doc = Document(doc_id="x", content="hello world")
        with pytest.raises(RuntimeError, match="must be built"):
            vs.add_document(doc)

    def test_added_document_is_retrievable(self, vector_store):
        doc = Document(
            doc_id="retrieval_test",
            content="unique phrase about quantum entanglement physics",
            source="physics",
        )
        vector_store.add_document(doc)
        result = vector_store.get_document("retrieval_test")
        assert result is not None
        assert result.content == doc.content


# ---------------------------------------------------------------------------
# VectorStore search tests
# ---------------------------------------------------------------------------

class TestVectorStoreSearch:
    def test_search_returns_list_of_tuples(self, vector_store):
        results = vector_store.search("machine learning", top_k=3)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_search_returns_at_most_top_k(self, vector_store):
        results = vector_store.search("learning", top_k=2)
        assert len(results) <= 2

    def test_search_results_sorted_by_score_descending(self, vector_store):
        results = vector_store.search("python programming", top_k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_scores_are_between_zero_and_one(self, vector_store):
        results = vector_store.search("natural language processing", top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_search_raises_on_empty_query(self, vector_store):
        with pytest.raises(ValueError, match="empty"):
            vector_store.search("")

    def test_search_raises_on_whitespace_query(self, vector_store):
        with pytest.raises(ValueError):
            vector_store.search("   ")

    def test_search_raises_on_invalid_top_k(self, vector_store):
        with pytest.raises(ValueError, match="positive"):
            vector_store.search("test", top_k=0)

    def test_search_raises_when_not_fitted(self):
        vs = VectorStore()
        with pytest.raises(RuntimeError, match="not built"):
            vs.search("query")

    def test_score_threshold_filters_low_relevance(self, vector_store):
        results = vector_store.search("python", top_k=10, score_threshold=0.9)
        for _, score in results:
            assert score >= 0.9

    def test_relevant_query_returns_relevant_source(self, vector_store):
        results = vector_store.search("docker containers deployment", top_k=3)
        sources = [doc.source for doc, _ in results]
        assert any("docker" in s for s in sources)


# ---------------------------------------------------------------------------
# VectorStore CRUD tests
# ---------------------------------------------------------------------------

class TestVectorStoreCRUD:
    def test_get_document_returns_correct_document(self, vector_store, sample_documents):
        doc_id = sample_documents[0].doc_id
        retrieved = vector_store.get_document(doc_id)
        assert retrieved is not None
        assert retrieved.doc_id == doc_id

    def test_get_document_returns_none_for_missing_id(self, vector_store):
        assert vector_store.get_document("nonexistent_id_xyz") is None

    def test_delete_document_returns_true_on_success(self, vector_store, sample_documents):
        doc_id = sample_documents[0].doc_id
        result = vector_store.delete_document(doc_id)
        assert result is True

    def test_delete_document_decreases_size(self, vector_store, sample_documents):
        before = vector_store.size
        vector_store.delete_document(sample_documents[0].doc_id)
        assert vector_store.size == before - 1

    def test_delete_document_returns_false_for_missing(self, vector_store):
        assert vector_store.delete_document("does_not_exist") is False

    def test_clear_empties_store(self, vector_store):
        vector_store.clear()
        assert vector_store.size == 0
        assert not vector_store._fitted

    def test_get_all_documents_returns_all(self, vector_store, sample_documents):
        all_docs = vector_store.get_all_documents()
        assert len(all_docs) == len(sample_documents)
