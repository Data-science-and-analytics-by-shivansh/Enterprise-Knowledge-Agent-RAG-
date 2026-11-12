import sys


def main() -> int:
    try:
        import rag_pipeline.document_processor  # noqa: F401
        import rag_pipeline.vector_store  # noqa: F401
        import rag_pipeline.pipeline  # noqa: F401

        print("Healthcheck passed: rag_pipeline modules loaded successfully.")
        return 0
    except Exception as exc:
        print(f"Healthcheck failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
