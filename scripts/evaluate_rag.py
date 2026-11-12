"""
RAGAS Evaluation Benchmark Suite for RAG Pipeline.

Measures:
- Faithfulness (claim grounding in retrieved context)
- Answer Relevancy (conciseness and direct relevance to question)
- Context Precision (ranking of ground-truth relevant contexts)

Supports:
- Dry-run mode: validates datasets, executes pipeline, measures latency,
  sets RAGAS metrics to null, and marks evaluation as skipped.
- Live mode: executes RAGAS evaluation with evaluator LLM (requires OPENAI_API_KEY).
- Export to JSON and CSV formats.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project src/ directory is discoverable when script is executed directly
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Core pipeline imports
from rag_pipeline.pipeline import RAGPipeline  # noqa: E402

logger = logging.getLogger(__name__)


# =============================================================================
# Phase 1: Dataset and Corpus Loading & Validation
# =============================================================================


def validate_dataset_record(record: Dict[str, Any], record_idx: int) -> None:
    """Validate that an evaluation dataset record meets schema requirements."""
    if not isinstance(record, dict):
        raise ValueError(f"Record {record_idx} must be a JSON object, got {type(record).__name__}")

    for req_field in ("question", "ground_truth", "contexts"):
        if req_field not in record:
            raise ValueError(f"Record {record_idx} is missing required field '{req_field}'")

    if not isinstance(record["question"], str) or not record["question"].strip():
        raise ValueError(f"Record {record_idx} 'question' must be a non-empty string")

    if not isinstance(record["ground_truth"], str) or not record["ground_truth"].strip():
        raise ValueError(f"Record {record_idx} 'ground_truth' must be a non-empty string")

    if not isinstance(record["contexts"], list):
        raise ValueError(f"Record {record_idx} 'contexts' must be a list of strings")

    for ctx_idx, ctx in enumerate(record["contexts"]):
        if not isinstance(ctx, str):
            ctx_type = type(ctx).__name__
            raise ValueError(
                f"Record {record_idx} context at index {ctx_idx} must be a string, got {ctx_type}"
            )


def validate_corpus_record(record: Dict[str, Any], record_idx: int) -> None:
    """Validate that a retrieval corpus record meets schema requirements."""
    if not isinstance(record, dict):
        raise ValueError(f"Corpus item {record_idx} must be a JSON object")

    if "text" not in record or not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError(f"Corpus item {record_idx} missing non-empty 'text' field")

    if "source" not in record or not isinstance(record["source"], str):
        raise ValueError(f"Corpus item {record_idx} missing 'source' string")

    if "metadata" in record and not isinstance(record["metadata"], dict):
        raise ValueError(f"Corpus item {record_idx} 'metadata' must be a dictionary")


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate evaluation records from a JSON or JSONL file.

    Expected schema per record:
        - question: str
        - ground_truth: str
        - contexts: list[str] (ground-truth reference contexts)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    records: List[Dict[str, Any]] = []

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Dataset file is empty: {file_path}")

    if content.startswith("["):
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise ValueError("JSON dataset must be a list of objects")
            records = parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc
    else:
        for line_number, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {file_path}: {exc}"
                ) from exc
            records.append(record)

    if not records:
        raise ValueError(f"No records found in dataset: {file_path}")

    for idx, record in enumerate(records, start=1):
        validate_dataset_record(record, idx)

    return records


def load_corpus(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and validate retrieval corpus records from a JSON or JSONL file.

    Expected schema per record:
        - text: str
        - source: str
        - metadata: optional dict
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {file_path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Corpus file is empty: {file_path}")

    records: List[Dict[str, Any]] = []

    if content.startswith("["):
        try:
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                raise ValueError("JSON corpus must be a list of objects")
            records = parsed
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {file_path}: {exc}") from exc
    else:
        for line_number, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} in {file_path}: {exc}"
                ) from exc
            records.append(record)

    if not records:
        raise ValueError(f"No records found in corpus: {file_path}")

    for idx, record in enumerate(records, start=1):
        validate_corpus_record(record, idx)

    return records


# =============================================================================
# Phase 2: RAG Pipeline Execution & Context Extraction
# =============================================================================


def execute_pipeline_query(
    pipeline: RAGPipeline, question: str, top_k: Optional[int] = None
) -> Tuple[Dict[str, Any], float]:
    """
    Execute a single question through the RAG pipeline and measure latency.

    Returns:
        Tuple of (pipeline_result_dict, latency_in_seconds)
    """
    start_time = time.perf_counter()
    result = pipeline.query(question=question, top_k=top_k)
    latency_seconds = time.perf_counter() - start_time
    return result, latency_seconds


def extract_contexts_from_sources(
    pipeline: RAGPipeline, sources: List[Dict[str, Any]]
) -> List[str]:
    """
    Extract document chunk contents directly from query result sources
    using VectorStore.get_document(), preserving retrieval order without re-searching.
    """
    contexts: List[str] = []
    for src in sources:
        doc_id = src.get("doc_id", "")
        doc = pipeline.vector_store.get_document(doc_id)
        if doc is not None:
            contexts.append(doc.content)
    return contexts


def extract_retrieved_contexts(
    pipeline: RAGPipeline, question: str, top_k: Optional[int] = None
) -> List[str]:
    """
    Extract retrieved document chunk contents in rank order via VectorStore search.
    Kept as a helper and for standalone lookup.
    """
    k = top_k or pipeline.top_k
    retrieved = pipeline.vector_store.search(
        question, top_k=k, score_threshold=pipeline.score_threshold
    )
    return [doc.content for doc, _ in retrieved]


def prepare_evaluation_samples(
    pipeline: RAGPipeline,
    dataset_records: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Run evaluation queries through the RAG pipeline and assemble evaluation samples.

    Returns:
        samples: list of dicts with keys:
            - question
            - answer
            - contexts (retrieved from vector store)
            - ground_truth (reference answer from dataset)
            - reference_contexts (ground-truth contexts from dataset)
            - sources (retrieved chunk metadata)
        latencies: list of query latencies in seconds
    """
    samples: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for record in dataset_records:
        question = record["question"]
        ground_truth = record["ground_truth"]
        reference_contexts = record.get("contexts", [])

        # Execute query and measure latency
        result, latency = execute_pipeline_query(pipeline, question, top_k=top_k)
        latencies.append(latency)

        # Retrieve exact chunk contents from query result sources without duplicating search
        sources = result.get("sources", [])
        retrieved_contexts = extract_contexts_from_sources(pipeline, sources)

        samples.append(
            {
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": retrieved_contexts,
                "ground_truth": ground_truth,
                "reference_contexts": reference_contexts,
                "sources": sources,
                "confidence": result.get("confidence", 0.0),
                "latency_seconds": round(latency, 4),
            }
        )

    return samples, latencies


# =============================================================================
# Phase 3: RAGAS Integration & Metric Evaluation
# =============================================================================


def evaluate_with_ragas(
    samples: List[Dict[str, Any]],
    dry_run: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate samples using RAGAS for faithfulness, answer_relevancy, and context_precision.

    In dry-run mode:
        Performs local validation only; all RAGAS metric values are set to None (null)
        and explicitly marked with evaluation_skipped=True. Never generates fake scores.

    In live mode:
        Invokes ragas.evaluate() using the pinned RAGAS API.
    """
    if dry_run:
        logger.info("Dry-run mode enabled: skipping external RAGAS evaluation.")
        return {
            "evaluation_skipped": True,
            "skip_reason": "Dry-run mode requested.",
            "metrics": {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
            },
            "per_sample_results": [],
        }

    # Verify API key for live mode
    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for live RAGAS evaluation. "
            "Set it in your environment or use --dry-run for offline verification."
        )

    # Lazy-load evaluation dependencies
    try:
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ImportError as exc:
        raise ImportError(
            f"RAGAS evaluation dependencies not found: {exc}. "
            "Install them with: pip install -r requirements-eval.txt"
        ) from exc

    # Prepare HuggingFace Dataset required by ragas 0.1.21
    ragas_dict = {
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    }

    eval_dataset = Dataset.from_dict(ragas_dict)
    target_metrics = [faithfulness, answer_relevancy, context_precision]

    eval_kwargs: Dict[str, Any] = {"metrics": target_metrics}
    if api_key:
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore

            eval_kwargs["llm"] = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
            eval_kwargs["embeddings"] = OpenAIEmbeddings(api_key=api_key)
        except ImportError:
            logger.warning(
                "langchain_openai not installed; falling back to default RAGAS evaluator setup."
            )

    logger.info("Executing RAGAS evaluation with %d samples...", len(samples))
    ragas_result = evaluate(eval_dataset, **eval_kwargs)

    # Extract score values
    faithfulness_score = ragas_result.get("faithfulness")
    answer_relevancy_score = ragas_result.get("answer_relevancy")
    context_precision_score = ragas_result.get("context_precision")

    def _sanitize(val: Any) -> Optional[float]:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        return round(float(val), 4)

    return {
        "evaluation_skipped": False,
        "skip_reason": None,
        "metrics": {
            "faithfulness": _sanitize(faithfulness_score),
            "answer_relevancy": _sanitize(answer_relevancy_score),
            "context_precision": _sanitize(context_precision_score),
        },
        "per_sample_results": (
            ragas_result.to_pandas().to_dict(orient="records")
            if hasattr(ragas_result, "to_pandas")
            else []
        ),
    }


# =============================================================================
# Phase 4: Latency Percentiles & Reporting
# =============================================================================


def calculate_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate latency statistics including p50, p90, p95, p99, min, max, and mean.
    All times returned in milliseconds (ms).
    """
    if not latencies:
        return {
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    def _percentile(p: float) -> float:
        if n == 1:
            return sorted_latencies[0] * 1000
        rank = (p / 100.0) * (n - 1)
        lower_idx = int(rank)
        upper_idx = min(lower_idx + 1, n - 1)
        weight = rank - lower_idx
        interpolated = (1.0 - weight) * sorted_latencies[lower_idx] + weight * sorted_latencies[
            upper_idx
        ]
        return interpolated * 1000

    mean_ms = (sum(sorted_latencies) / n) * 1000

    return {
        "mean_ms": round(mean_ms, 2),
        "min_ms": round(sorted_latencies[0] * 1000, 2),
        "p50_ms": round(_percentile(50), 2),
        "p90_ms": round(_percentile(90), 2),
        "p95_ms": round(_percentile(95), 2),
        "p99_ms": round(_percentile(99), 2),
        "max_ms": round(sorted_latencies[-1] * 1000, 2),
    }


def format_evaluation_report(
    summary: Dict[str, Any],
    eval_result: Dict[str, Any],
    latency_stats: Dict[str, float],
) -> str:
    """Format evaluation results into a human-readable console report."""
    eval_mode = (
        "DRY RUN (Metrics Skipped)"
        if eval_result.get("evaluation_skipped")
        else "LIVE (RAGAS Evaluator)"
    )
    lines = [
        "",
        "=" * 70,
        "RAGAS BENCHMARK EVALUATION REPORT",
        "=" * 70,
        f"Corpus Documents Ingested : {summary.get('corpus_docs_count', 0)}",
        f"Corpus Chunks Indexed     : {summary.get('corpus_chunks_count', 0)}",
        f"Evaluation Samples Run    : {summary.get('sample_count', 0)}",
        f"Evaluation Mode           : {eval_mode}",
        "-" * 70,
        "RAGAS METRICS:",
    ]

    metrics = eval_result.get("metrics", {})
    if eval_result.get("evaluation_skipped"):
        lines.append(f"  [SKIPPED] {eval_result.get('skip_reason', 'Evaluation skipped')}")
        lines.append("  • Faithfulness       : null (skipped)")
        lines.append("  • Answer Relevancy   : null (skipped)")
        lines.append("  • Context Precision  : null (skipped)")
    else:
        f_val = metrics.get("faithfulness")
        ar_val = metrics.get("answer_relevancy")
        cp_val = metrics.get("context_precision")
        lines.append(f"  • Faithfulness       : {f_val if f_val is not None else 'N/A'}")
        lines.append(f"  • Answer Relevancy   : {ar_val if ar_val is not None else 'N/A'}")
        lines.append(f"  • Context Precision  : {cp_val if cp_val is not None else 'N/A'}")

    lines.extend(
        [
            "-" * 70,
            "QUERY LATENCY BENCHMARKS (ms):",
            f"  • Mean Latency : {latency_stats.get('mean_ms', 0):.2f} ms",
            f"  • Min Latency  : {latency_stats.get('min_ms', 0):.2f} ms",
            f"  • p50 (Median) : {latency_stats.get('p50_ms', 0):.2f} ms",
            f"  • p90          : {latency_stats.get('p90_ms', 0):.2f} ms",
            f"  • p95          : {latency_stats.get('p95_ms', 0):.2f} ms",
            f"  • p99          : {latency_stats.get('p99_ms', 0):.2f} ms",
            f"  • Max Latency  : {latency_stats.get('max_ms', 0):.2f} ms",
            "=" * 70,
            "",
        ]
    )
    return "\n".join(lines)


def export_report_json(output_path: str, data: Dict[str, Any]) -> None:
    """Export benchmark evaluation results to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_report_csv(output_path: str, samples: List[Dict[str, Any]]) -> None:
    """Export sample-level benchmark queries, answers, and latencies to CSV."""
    import csv

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question",
        "answer",
        "ground_truth",
        "contexts_count",
        "latency_seconds",
        "confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "question": s.get("question", ""),
                    "answer": s.get("answer", ""),
                    "ground_truth": s.get("ground_truth", ""),
                    "contexts_count": len(s.get("contexts", [])),
                    "latency_seconds": s.get("latency_seconds", 0.0),
                    "confidence": s.get("confidence", 0.0),
                }
            )


# =============================================================================
# Benchmark Suite Runner & CLI
# =============================================================================


class RAGBenchmarkRunner:
    """Orchestrates the complete RAGAS evaluation workflow."""

    def __init__(
        self,
        dataset_path: str,
        corpus_path: str,
        dry_run: bool = True,
        top_k: int = 3,
        api_key: Optional[str] = None,
    ):
        self.dataset_path = dataset_path
        self.corpus_path = corpus_path
        self.dry_run = dry_run
        self.top_k = top_k
        self.api_key = api_key

    def run(self) -> Dict[str, Any]:
        """Execute the benchmark suite."""
        # Step 1: Load and validate dataset & corpus
        logger.info("Loading evaluation dataset from %s", self.dataset_path)
        dataset_records = load_dataset(self.dataset_path)

        logger.info("Loading retrieval corpus from %s", self.corpus_path)
        corpus_records = load_corpus(self.corpus_path)

        # Step 2: Initialize pipeline and ingest corpus
        pipeline = RAGPipeline(top_k=self.top_k)
        chunks_indexed = pipeline.ingest(corpus_records)
        logger.info("Ingested %d corpus docs (%d chunks)", len(corpus_records), chunks_indexed)

        # Step 3: Run queries and extract contexts/answers
        samples, latencies = prepare_evaluation_samples(
            pipeline=pipeline,
            dataset_records=dataset_records,
            top_k=self.top_k,
        )

        # Step 4: Calculate latency percentiles
        latency_stats = calculate_latency_percentiles(latencies)

        # Step 5: Evaluate with RAGAS (or dry-run null assignment)
        eval_result = evaluate_with_ragas(
            samples=samples,
            dry_run=self.dry_run,
            api_key=self.api_key,
        )

        summary = {
            "corpus_docs_count": len(corpus_records),
            "corpus_chunks_count": chunks_indexed,
            "sample_count": len(samples),
            "top_k": self.top_k,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        report_text = format_evaluation_report(summary, eval_result, latency_stats)

        return {
            "summary": summary,
            "ragas_evaluation": eval_result,
            "latency_benchmarks": latency_stats,
            "samples": samples,
            "report_text": report_text,
        }


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run RAGAS Evaluation Benchmark Suite for RAG Pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/eval_benchmark.jsonl",
        help="Path to evaluation questions/ground truth JSONL dataset",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/eval_corpus.jsonl",
        help="Path to retrieval corpus JSONL file to ingest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Perform local pipeline execution and latency checks; skip external RAGAS evaluator",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve per question (default: 3)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to export full evaluation results as JSON",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to export sample results as CSV",
    )
    return parser.parse_args(args)


def main() -> int:
    """CLI entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    try:
        runner = RAGBenchmarkRunner(
            dataset_path=args.dataset,
            corpus_path=args.corpus,
            dry_run=args.dry_run,
            top_k=args.top_k,
        )
        results = runner.run()
        print(results["report_text"])

        if args.output_json:
            export_report_json(args.output_json, results)
            logger.info("Saved JSON results to %s", args.output_json)

        if args.output_csv:
            export_report_csv(args.output_csv, results["samples"])
            logger.info("Saved CSV results to %s", args.output_csv)

        return 0

    except Exception as exc:
        logger.error("Benchmark evaluation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
