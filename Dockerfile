FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml README.md NOTICE LICENSE ./
COPY smartrag/ smartrag/

RUN pip install --no-cache-dir ".[cloud]"

# Create non-root user
RUN useradd --create-home --shell /bin/bash smartrag && \
    mkdir -p /data/knowledge && \
    chown -R smartrag:smartrag /data

USER smartrag

# Default knowledge store location
ENV SMARTRAG_KNOWLEDGE_DIR=/data/knowledge \
    SMARTRAG_PORT=8000 \
    SMARTRAG_HOST=0.0.0.0 \
    SMARTRAG_LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

CMD ["uvicorn", "smartrag.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
