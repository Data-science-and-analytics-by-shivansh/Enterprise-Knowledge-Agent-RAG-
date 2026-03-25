"""
Unit tests for ResponseGenerator, PromptBuilder, and RAGPipeline.
~60 tests covering prompts, generation, end-to-end pipeline.
"""

import pytest
from rag_pipeline.document_processor import Document
from rag_pipeline.response_generator import ResponseGenerator, PromptBuilder
from rag_pipeline.pipeline import RAGPipeline


# ---------------------------------------------------------------------------
# PromptBuilder tests
# ---------------------------------------------------------------------------

class TestPromptBuilderInit:
    def test_default_max_context_length(self):
        pb = PromptBuilder()
        assert pb.max_context_length == 2048

    def test_custom_max_context_length(self):
        pb = PromptBuilder(max_context_length=512)
        assert pb.max_context_length == 512

    def test_raises_on_zero_max_context(self):
        with pytest.raises(ValueError):
            PromptBuilder(max_context_length=0)

    def test_raises_on_negative_max_context(self):
        with pytest.raises(ValueError):
            PromptBuilder(max_context_length=-100)


class TestPromptBuilderBuildContext:
    def _make_retrieved(self, texts, sources=None):
        if sources is None:
            sources = [f"src{i}" for i in range(len(texts))]
        return [
            (Document(doc_id=f"d{i}", content=t, source=s), 0.8 - i * 0.1)
            for i, (t, s) in enumerate(zip(texts, sources))
        ]

    def test_empty_retrieved_returns_empty_string(self, prompt_builder):
        assert prompt_builder.build_context([]) == ""

    def test_context_contains_source_label(self, prompt_builder):
        retrieved = self._make_retrieved(["Some content here"], ["my_source"])
        context = prompt_builder.build_context(retrieved)
        assert "my_source" in context

    def test_context_contains_document_content(self, prompt_builder):
        retrieved = self._make_retrieved(["Important information about Python"])
        context = prompt_builder.build_context(retrieved)
        assert "Python" in context

    def test_context_respects_max_length(self):
        pb = PromptBuilder(max_context_length=100)
        big_text = "word " * 200
        retrieved = [(Document(doc_id="x", content=big_text, source="s"), 0.9)]
        context = pb.build_context(retrieved)
        assert len(context) <= 200  # truncated

    def test_multiple_docs_separated_in_context(self, prompt_builder):
        retrieved = self._make_retrieved(
            ["First document content", "Second document content"],
            ["src1", "src2"]
        )
        context = prompt_builder.build_context(retrieved)
        assert "First document" in context
        assert "Second document" in context


class TestPromptBuilderBuildQAPrompt:
    def _make_retrieved(self, texts):
        return [
            (Document(doc_id=f"d{i}", content=t, source=f"s{i}"), 0.9)
            for i, t in enumerate(texts)
        ]

    def test_qa_prompt_contains_question(self, prompt_builder):
        retrieved = self._make_retrieved(["Some context about AI"])
        prompt = prompt_builder.build_qa_prompt("What is AI?", retrieved)
        assert "What is AI?" in prompt

    def test_qa_prompt_contains_context(self, prompt_builder):
        retrieved = self._make_retrieved(["Context about machine learning"])
        prompt = prompt_builder.build_qa_prompt("What is ML?", retrieved)
        assert "machine learning" in prompt

    def test_qa_prompt_raises_on_empty_question(self, prompt_builder):
        retrieved = self._make_retrieved(["Some context"])
        with pytest.raises(ValueError, match="empty"):
            prompt_builder.build_qa_prompt("", retrieved)

    def test_qa_prompt_raises_on_whitespace_question(self, prompt_builder):
        retrieved = self._make_retrieved(["Some context"])
        with pytest.raises(ValueError):
            prompt_builder.build_qa_prompt("   ", retrieved)

    def test_qa_prompt_contains_answer_label(self, prompt_builder):
        retrieved = self._make_retrieved(["Context content"])
        prompt = prompt_builder.build_qa_prompt("Question?", retrieved)
        assert "Answer:" in prompt

    def test_summary_prompt_built_successfully(self, prompt_builder):
        retrieved = self._make_retrieved(["Text to summarize here"])
        prompt = prompt_builder.build_summary_prompt(retrieved)
        assert "Summary:" in prompt or "Summarize" in prompt


# ---------------------------------------------------------------------------
# ResponseGenerator tests
# ---------------------------------------------------------------------------

class TestResponseGeneratorInit:
    def test_default_initialization(self):
        rg = ResponseGenerator()
        assert rg.prompt_builder is not None

    def test_custom_prompt_builder(self):
        pb = PromptBuilder(512)
        rg = ResponseGenerator(prompt_builder=pb)
        assert rg.prompt_builder is pb


class TestResponseGeneratorGenerate:
    def _make_retrieved(self, texts, scores=None):
        if scores is None:
            scores = [0.9 - i * 0.1 for i in range(len(texts))]
        return [
            (Document(doc_id=f"d{i}", content=t, source=f"src{i}"), s)
            for i, (t, s) in enumerate(zip(texts, scores))
        ]

    def test_generate_returns_dict(self, response_generator):
        retrieved = self._make_retrieved(["Some context about Python"])
        result = response_generator.generate("What is Python?", retrieved)
        assert isinstance(result, dict)

    def test_generate_result_has_required_keys(self, response_generator):
        retrieved = self._make_retrieved(["Some context"])
        result = response_generator.generate("What?", retrieved)
        for key in ("answer", "prompt", "sources", "num_sources", "confidence", "mode"):
            assert key in result

    def test_generate_qa_mode_default(self, response_generator):
        retrieved = self._make_retrieved(["Context text"])
        result = response_generator.generate("Question?", retrieved)
        assert result["mode"] == "qa"

    def test_generate_summary_mode(self, response_generator):
        retrieved = self._make_retrieved(["Context text for summary"])
        result = response_generator.generate("Summarize", retrieved, mode="summary")
        assert result["mode"] == "summary"

    def test_generate_raises_on_unknown_mode(self, response_generator):
        retrieved = self._make_retrieved(["Context"])
        with pytest.raises(ValueError, match="Unknown mode"):
            response_generator.generate("Q", retrieved, mode="unknown_mode")

    def test_confidence_is_float_between_zero_and_one(self, response_generator):
        retrieved = self._make_retrieved(["Context"], scores=[0.75])
        result = response_generator.generate("Q?", retrieved)
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_num_sources_matches_retrieved_count(self, response_generator):
        retrieved = self._make_retrieved(["Doc1", "Doc2", "Doc3"])
        result = response_generator.generate("Q?", retrieved)
        assert result["num_sources"] == 3

    def test_sources_list_has_correct_structure(self, response_generator):
        retrieved = self._make_retrieved(["Some content"])
        result = response_generator.generate("Q?", retrieved)
        assert len(result["sources"]) == 1
        source = result["sources"][0]
        for key in ("doc_id", "source", "score", "chunk_index"):
            assert key in source

    def test_empty_retrieved_returns_no_info_message(self, response_generator):
        result = response_generator.generate("What is X?", [])
        assert "information" in result["answer"].lower() or "don't" in result["answer"].lower()

    def test_confidence_is_zero_for_empty_retrieved(self, response_generator):
        result = response_generator.generate("Q?", [])
        assert result["confidence"] == 0.0

    def test_answer_is_non_empty_string_when_context_exists(self, response_generator):
        retrieved = self._make_retrieved(["Python is a programming language."])
        result = response_generator.generate("What is Python?", retrieved)
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0


# ---------------------------------------------------------------------------
# RAGPipeline tests
# ---------------------------------------------------------------------------

class TestRAGPipelineInit:
    def test_default_initialization(self, pipeline):
        assert not pipeline.is_ready
        assert pipeline.document_count == 0

    def test_custom_parameters(self):
        p = RAGPipeline(chunk_size=256, chunk_overlap=32, top_k=10)
        assert p.chunk_size == 256
        assert p.chunk_overlap == 32
        assert p.top_k == 10


class TestRAGPipelineIngest:
    def test_ingest_returns_chunk_count(self, pipeline, sample_texts):
        count = pipeline.ingest(sample_texts)
        assert count > 0

    def test_ingest_sets_is_ready(self, pipeline, sample_texts):
        pipeline.ingest(sample_texts)
        assert pipeline.is_ready

    def test_ingest_sets_document_count(self, pipeline, sample_texts):
        pipeline.ingest(sample_texts)
        assert pipeline.document_count == len(sample_texts)

    def test_ingest_raises_on_empty_list(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.ingest([])

    def test_ingest_raises_on_all_empty_texts(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.ingest([{"text": "", "source": "x"}])


class TestRAGPipelineQuery:
    def test_query_returns_dict(self, ingested_pipeline):
        result = ingested_pipeline.query("What is machine learning?")
        assert isinstance(result, dict)

    def test_query_result_has_required_keys(self, ingested_pipeline):
        result = ingested_pipeline.query("What is Python?")
        for key in ("answer", "sources", "confidence", "query", "top_k_used"):
            assert key in result

    def test_query_preserves_question_in_result(self, ingested_pipeline):
        q = "What is RAG?"
        result = ingested_pipeline.query(q)
        assert result["query"] == q

    def test_query_raises_when_not_ingested(self, pipeline):
        with pytest.raises(RuntimeError, match="not ready"):
            pipeline.query("What is X?")

    def test_query_top_k_override(self, ingested_pipeline):
        result = ingested_pipeline.query("machine learning", top_k=2)
        assert result["top_k_used"] == 2
        assert len(result["sources"]) <= 2

    def test_query_summary_mode(self, ingested_pipeline):
        result = ingested_pipeline.query("Summarize the documents", mode="summary")
        assert result["mode"] == "summary"

    def test_query_answer_is_string(self, ingested_pipeline):
        result = ingested_pipeline.query("What is Docker?")
        assert isinstance(result["answer"], str)

    def test_query_confidence_between_zero_and_one(self, ingested_pipeline):
        result = ingested_pipeline.query("NLP applications")
        assert 0.0 <= result["confidence"] <= 1.0


class TestRAGPipelineAddDocument:
    def test_add_document_increases_chunk_count(self, ingested_pipeline):
        before = ingested_pipeline.vector_store.size
        ingested_pipeline.add_document(
            "Kubernetes is a container orchestration platform for scaling applications.",
            source="k8s_guide",
        )
        assert ingested_pipeline.vector_store.size > before

    def test_add_document_increases_document_count(self, ingested_pipeline):
        before = ingested_pipeline.document_count
        ingested_pipeline.add_document("New document content here.", source="new")
        assert ingested_pipeline.document_count == before + 1

    def test_add_document_raises_when_not_ingested(self, pipeline):
        with pytest.raises(RuntimeError, match="not initialized"):
            pipeline.add_document("Some text", source="src")


class TestRAGPipelineReset:
    def test_reset_clears_pipeline(self, ingested_pipeline):
        ingested_pipeline.reset()
        assert not ingested_pipeline.is_ready
        assert ingested_pipeline.document_count == 0
        assert ingested_pipeline.vector_store.size == 0

    def test_pipeline_usable_after_reingest(self, ingested_pipeline, sample_texts):
        ingested_pipeline.reset()
        ingested_pipeline.ingest(sample_texts)
        assert ingested_pipeline.is_ready


class TestRAGPipelineStats:
    def test_get_stats_returns_dict(self, ingested_pipeline):
        stats = ingested_pipeline.get_stats()
        assert isinstance(stats, dict)

    def test_stats_contains_expected_keys(self, ingested_pipeline):
        stats = ingested_pipeline.get_stats()
        for key in ("is_ready", "document_count", "chunk_count", "chunk_size"):
            assert key in stats

    def test_stats_chunk_count_positive_after_ingest(self, ingested_pipeline):
        stats = ingested_pipeline.get_stats()
        assert stats["chunk_count"] > 0
