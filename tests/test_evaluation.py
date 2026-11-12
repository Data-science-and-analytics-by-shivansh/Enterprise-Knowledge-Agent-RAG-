"""
Unit tests for RAGAS evaluation benchmark suite (scripts/evaluate_rag.py).
Tests dataset validation, pipeline execution, context extraction, latency percentiles,
dry-run behavior, mocked RAGAS integration, and report formatting/export.
"""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rag_pipeline.pipeline import RAGPipeline

# Dynamically load scripts/evaluate_rag without top-level import statement ordering issues
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_SCRIPT_FILE = _SCRIPTS_DIR / "evaluate_rag.py"
_SPEC = importlib.util.spec_from_file_location("evaluate_rag", _SCRIPT_FILE)
eval_module = importlib.util.module_from_spec(_SPEC)
sys.modules["evaluate_rag"] = eval_module
_SPEC.loader.exec_module(eval_module)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_dataset_records():
    return [
        {
            "question": "What is RAG?",
            "ground_truth": "Retrieval-Augmented Generation combines retrieval with LLMs.",
            "contexts": ["RAG combines retrieval with language models to reduce hallucinations."],
        },
        {
            "question": "What is Docker?",
            "ground_truth": "Docker is a containerization platform.",
            "contexts": ["Docker isolates applications in containers."],
        },
    ]


@pytest.fixture
def valid_corpus_records():
    return [
        {
            "source": "rag_doc",
            "text": (
                "RAG combines retrieval with language models "
                "to reduce hallucinations and ground answers."
            ),
            "metadata": {"topic": "ai"},
        },
        {
            "source": "docker_doc",
            "text": (
                "Docker isolates applications in lightweight portable "
                "containers for consistent deployments."
            ),
            "metadata": {"topic": "devops"},
        },
    ]


@pytest.fixture
def sample_dataset_file(tmp_path, valid_dataset_records):
    file_path = tmp_path / "dataset.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for r in valid_dataset_records:
            f.write(json.dumps(r) + "\n")
    return str(file_path)


@pytest.fixture
def sample_corpus_file(tmp_path, valid_corpus_records):
    file_path = tmp_path / "corpus.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for r in valid_corpus_records:
            f.write(json.dumps(r) + "\n")
    return str(file_path)


# =============================================================================
# 1. Dataset & Corpus Loading and Validation Tests
# =============================================================================


class TestDatasetAndCorpusValidation:
    def test_load_valid_jsonl_dataset(self, sample_dataset_file):
        records = eval_module.load_dataset(sample_dataset_file)
        assert len(records) == 2
        assert records[0]["question"] == "What is RAG?"
        assert isinstance(records[0]["contexts"], list)

    def test_load_valid_json_array_dataset(self, tmp_path, valid_dataset_records):
        file_path = tmp_path / "dataset.json"
        file_path.write_text(json.dumps(valid_dataset_records), encoding="utf-8")
        records = eval_module.load_dataset(str(file_path))
        assert len(records) == 2

    def test_load_dataset_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            eval_module.load_dataset("non_existent_file.jsonl")

    def test_load_dataset_empty_file(self, tmp_path):
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            eval_module.load_dataset(str(empty_file))

    def test_load_dataset_invalid_json(self, tmp_path):
        bad_json = tmp_path / "bad.jsonl"
        bad_json.write_text("{this is not json}", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            eval_module.load_dataset(str(bad_json))

    def test_load_dataset_missing_required_fields(self):
        bad_record = {"question": "What is AI?"}  # missing ground_truth and contexts
        with pytest.raises(ValueError, match="missing required field"):
            eval_module.validate_dataset_record(bad_record, 1)

    def test_load_dataset_invalid_field_types(self):
        # question not a string
        with pytest.raises(ValueError, match="'question' must be a non-empty string"):
            eval_module.validate_dataset_record(
                {"question": 123, "ground_truth": "test", "contexts": []}, 1
            )
        # contexts not a list
        with pytest.raises(ValueError, match="'contexts' must be a list"):
            eval_module.validate_dataset_record(
                {"question": "Q", "ground_truth": "A", "contexts": "not a list"}, 1
            )
        # item in contexts not a string
        with pytest.raises(ValueError, match="must be a string"):
            eval_module.validate_dataset_record(
                {"question": "Q", "ground_truth": "A", "contexts": [123]}, 1
            )

    def test_load_valid_corpus(self, sample_corpus_file):
        corpus = eval_module.load_corpus(sample_corpus_file)
        assert len(corpus) == 2
        assert corpus[0]["source"] == "rag_doc"

    def test_load_corpus_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            eval_module.load_corpus("missing_corpus.jsonl")

    def test_load_corpus_missing_text(self):
        bad_corpus = {"source": "doc1"}
        with pytest.raises(ValueError, match="missing non-empty 'text'"):
            eval_module.validate_corpus_record(bad_corpus, 1)

    def test_load_corpus_missing_source(self):
        bad_corpus = {"text": "some text"}
        with pytest.raises(ValueError, match="missing 'source'"):
            eval_module.validate_corpus_record(bad_corpus, 1)

    def test_load_corpus_json_array(self, tmp_path, valid_corpus_records):
        file_path = tmp_path / "corpus.json"
        file_path.write_text(json.dumps(valid_corpus_records), encoding="utf-8")
        records = eval_module.load_corpus(str(file_path))
        assert len(records) == 2


# =============================================================================
# 2. Pipeline Execution & Context Extraction Tests
# =============================================================================


class TestPipelineExecutionAndContextExtraction:
    @pytest.fixture
    def indexed_pipeline(self, valid_corpus_records):
        pipeline = RAGPipeline(top_k=2)
        pipeline.ingest(valid_corpus_records)
        return pipeline

    def test_execute_pipeline_query(self, indexed_pipeline):
        result, latency = eval_module.execute_pipeline_query(
            indexed_pipeline, "What is RAG?", top_k=2
        )
        assert "answer" in result
        assert isinstance(latency, float)
        assert latency >= 0.0

    def test_extract_retrieved_contexts(self, indexed_pipeline):
        contexts = eval_module.extract_retrieved_contexts(indexed_pipeline, "Docker", top_k=2)
        assert isinstance(contexts, list)
        assert len(contexts) <= 2
        assert all(isinstance(c, str) for c in contexts)
        assert any("Docker" in c for c in contexts)

    def test_extract_contexts_from_sources(self, indexed_pipeline):
        result, _ = eval_module.execute_pipeline_query(indexed_pipeline, "Docker", top_k=2)
        sources = result.get("sources", [])
        assert len(sources) > 0
        contexts = eval_module.extract_contexts_from_sources(indexed_pipeline, sources)
        assert len(contexts) == len(sources)
        assert all(isinstance(c, str) for c in contexts)
        assert any("Docker" in c for c in contexts)

    def test_prepare_evaluation_samples(self, indexed_pipeline, valid_dataset_records):
        samples, latencies = eval_module.prepare_evaluation_samples(
            pipeline=indexed_pipeline,
            dataset_records=valid_dataset_records,
            top_k=2,
        )
        assert len(samples) == len(valid_dataset_records)
        assert len(latencies) == len(valid_dataset_records)

        sample = samples[0]
        assert "question" in sample
        assert "answer" in sample
        assert "contexts" in sample
        assert "ground_truth" in sample
        assert "reference_contexts" in sample
        assert isinstance(sample["contexts"], list)
        assert isinstance(sample["latency_seconds"], float)

    def test_prepare_samples_avoids_duplicate_search(self, indexed_pipeline, valid_dataset_records):
        with patch.object(
            indexed_pipeline.vector_store, "search", wraps=indexed_pipeline.vector_store.search
        ) as spy_search:
            eval_module.prepare_evaluation_samples(
                pipeline=indexed_pipeline,
                dataset_records=valid_dataset_records,
                top_k=2,
            )
            # Exactly 1 search per dataset record during pipeline.query(), no extra re-search
            assert spy_search.call_count == len(valid_dataset_records)


# =============================================================================
# 3. Latency Percentiles Calculation Tests
# =============================================================================


class TestLatencyPercentiles:
    def test_empty_latencies(self):
        stats = eval_module.calculate_latency_percentiles([])
        assert stats["mean_ms"] == 0.0
        assert stats["p50_ms"] == 0.0
        assert stats["p95_ms"] == 0.0

    def test_single_latency(self):
        stats = eval_module.calculate_latency_percentiles([0.05])  # 50 ms
        assert stats["mean_ms"] == 50.0
        assert stats["min_ms"] == 50.0
        assert stats["p50_ms"] == 50.0
        assert stats["p99_ms"] == 50.0
        assert stats["max_ms"] == 50.0

    def test_multiple_latencies_ordering(self):
        latencies = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        stats = eval_module.calculate_latency_percentiles(latencies)
        assert stats["min_ms"] == 10.0
        assert stats["max_ms"] == 100.0
        assert stats["mean_ms"] == 55.0
        assert stats["min_ms"] <= stats["p50_ms"] <= stats["p90_ms"] <= stats["max_ms"]


# =============================================================================
# 4. RAGAS Integration & Dry-run Mode Tests
# =============================================================================


class TestRagasEvaluationIntegration:
    def test_dry_run_mode_sets_metrics_to_null_and_skipped(self):
        sample = {
            "question": "What is RAG?",
            "answer": "RAG grounds generation in retrieved text.",
            "contexts": ["Some context"],
            "ground_truth": "RAG reduces hallucination.",
        }
        res = eval_module.evaluate_with_ragas([sample], dry_run=True)
        assert res["evaluation_skipped"] is True
        assert res["skip_reason"] is not None
        assert res["metrics"]["faithfulness"] is None
        assert res["metrics"]["answer_relevancy"] is None
        assert res["metrics"]["context_precision"] is None
        assert res["per_sample_results"] == []

    def test_live_mode_without_api_key_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        sample = {
            "question": "Q",
            "answer": "A",
            "contexts": ["C"],
            "ground_truth": "GT",
        }
        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            eval_module.evaluate_with_ragas([sample], dry_run=False, api_key=None)

    def test_live_mode_with_mocked_ragas(self, monkeypatch):
        # Mock external dependencies for live mode testing
        mock_result = {
            "faithfulness": 0.9523,
            "answer_relevancy": 0.8812,
            "context_precision": 0.9100,
        }
        mock_evaluate = MagicMock(return_value=mock_result)

        mock_ragas = MagicMock()
        mock_ragas.evaluate = mock_evaluate

        mock_metrics = MagicMock()
        mock_metrics.faithfulness = "faithfulness"
        mock_metrics.answer_relevancy = "answer_relevancy"
        mock_metrics.context_precision = "context_precision"

        mock_datasets = MagicMock()
        mock_datasets.Dataset.from_dict.return_value = "mock_hf_dataset"

        with patch.dict(
            sys.modules,
            {
                "ragas": mock_ragas,
                "ragas.metrics": mock_metrics,
                "datasets": mock_datasets,
            },
        ):
            samples = [
                {
                    "question": "Q1",
                    "answer": "A1",
                    "contexts": ["C1"],
                    "ground_truth": "GT1",
                }
            ]
            res = eval_module.evaluate_with_ragas(
                samples, dry_run=False, api_key="sk-test-mock-key"
            )
            assert res["evaluation_skipped"] is False
            assert res["metrics"]["faithfulness"] == 0.9523
            assert res["metrics"]["answer_relevancy"] == 0.8812
            assert res["metrics"]["context_precision"] == 0.9100

    def test_live_mode_with_api_key_passes_llm_without_env_mutation(self, monkeypatch):
        import os

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        mock_evaluate = MagicMock(
            return_value={"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7}
        )
        mock_ragas = MagicMock()
        mock_ragas.evaluate = mock_evaluate

        mock_chat_openai = MagicMock()
        mock_embeddings = MagicMock()
        mock_langchain_openai = MagicMock()
        mock_langchain_openai.ChatOpenAI = mock_chat_openai
        mock_langchain_openai.OpenAIEmbeddings = mock_embeddings

        mock_datasets = MagicMock()
        mock_datasets.Dataset.from_dict.return_value = "mock_hf_dataset"

        with patch.dict(
            sys.modules,
            {
                "ragas": mock_ragas,
                "ragas.metrics": MagicMock(),
                "datasets": mock_datasets,
                "langchain_openai": mock_langchain_openai,
            },
        ):
            samples = [{"question": "Q", "answer": "A", "contexts": ["C"], "ground_truth": "GT"}]
            eval_module.evaluate_with_ragas(
                samples, dry_run=False, api_key="sk-my-programmatic-key"
            )
            # Ensure OPENAI_API_KEY was not mutated in global os.environ
            assert "OPENAI_API_KEY" not in os.environ
            mock_chat_openai.assert_called_once_with(
                api_key="sk-my-programmatic-key", model="gpt-4o-mini"
            )
            mock_embeddings.assert_called_once_with(api_key="sk-my-programmatic-key")
            _, kwargs = mock_evaluate.call_args
            assert "llm" in kwargs
            assert "embeddings" in kwargs


# =============================================================================
# 5. Report Formatting & Export Tests
# =============================================================================


class TestReportFormattingAndExport:
    def test_format_report_text(self):
        summary = {"corpus_docs_count": 5, "corpus_chunks_count": 5, "sample_count": 2}
        eval_result = {
            "evaluation_skipped": True,
            "skip_reason": "Dry-run mode.",
            "metrics": {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
            },
        }
        latency_stats = {
            "mean_ms": 1.25,
            "min_ms": 1.0,
            "p50_ms": 1.2,
            "p90_ms": 1.5,
            "p95_ms": 1.6,
            "p99_ms": 1.7,
            "max_ms": 1.8,
        }
        report = eval_module.format_evaluation_report(summary, eval_result, latency_stats)
        assert "RAGAS BENCHMARK EVALUATION REPORT" in report
        assert "DRY RUN (Metrics Skipped)" in report
        assert "1.25 ms" in report
        assert "null (skipped)" in report

    def test_export_json_and_csv(self, tmp_path):
        json_path = tmp_path / "output.json"
        csv_path = tmp_path / "output.csv"

        data = {"status": "ok", "samples": [{"question": "Q", "answer": "A"}]}
        eval_module.export_report_json(str(json_path), data)
        assert json_path.exists()
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["status"] == "ok"

        samples = [
            {
                "question": "What is Python?",
                "answer": "A programming language.",
                "ground_truth": "Python is a language.",
                "contexts": ["context1"],
                "latency_seconds": 0.005,
                "confidence": 0.8,
            }
        ]
        eval_module.export_report_csv(str(csv_path), samples)
        assert csv_path.exists()
        csv_content = csv_path.read_text(encoding="utf-8")
        assert "What is Python?" in csv_content
        assert "latency_seconds" in csv_content


# =============================================================================
# 6. End-to-End Runner Tests
# =============================================================================


class TestRAGBenchmarkRunner:
    def test_benchmark_runner_dry_run(self, sample_dataset_file, sample_corpus_file):
        runner = eval_module.RAGBenchmarkRunner(
            dataset_path=sample_dataset_file,
            corpus_path=sample_corpus_file,
            dry_run=True,
            top_k=2,
        )
        results = runner.run()
        assert results["summary"]["corpus_docs_count"] == 2
        assert results["summary"]["sample_count"] == 2
        assert results["ragas_evaluation"]["evaluation_skipped"] is True
        assert results["ragas_evaluation"]["metrics"]["faithfulness"] is None
        assert "report_text" in results
        assert results["latency_benchmarks"]["mean_ms"] > 0
