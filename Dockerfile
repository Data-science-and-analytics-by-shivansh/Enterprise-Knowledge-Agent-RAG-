# =============================================================================
# RAG Pipeline - Multi-Stage Dockerfile
# Stage 1: base      - shared Python environment
# Stage 2: test      - runs full test suite
# Stage 3: app       - lean production image
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: base
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

LABEL maintainer="Shivansh"
LABEL project="rag-pipeline"
LABEL description="Retrieval-Augmented Generation Pipeline"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY rag_pipeline/ ./rag_pipeline/


# ---------------------------------------------------------------------------
# Stage 2: test
# ---------------------------------------------------------------------------
FROM base AS test

COPY tests/ ./tests/
COPY pytest.ini .

# Run the full test suite; exit code propagates to docker build
RUN pytest tests/ \
    --tb=short \
    --color=yes \
    -v \
    --cov=rag_pipeline \
    --cov-report=term-missing \
    --cov-report=xml:/app/coverage.xml \
    --cov-fail-under=80


# ---------------------------------------------------------------------------
# Stage 3: app (production)
# ---------------------------------------------------------------------------
FROM base AS app

# Non-root user for security
RUN groupadd --gid 1001 raguser && \
    useradd --uid 1001 --gid raguser --shell /bin/bash --create-home raguser

# Copy only what's needed from base
COPY --from=base /app /app

# Health-check script
COPY scripts/healthcheck.py ./scripts/healthcheck.py

RUN chown -R raguser:raguser /app
USER raguser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python scripts/healthcheck.py

CMD ["python", "-m", "rag_pipeline.pipeline"]
