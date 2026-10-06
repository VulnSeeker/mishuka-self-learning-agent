# ============================================================================
# Onyx — Production Dockerfile
# ============================================================================
# Multi-stage build: slim runtime image, non-root user, healthcheck.
#
# Build:
#   docker build -t onyx:latest .
#
# Run:
#   docker run -p 8000:8000 --env-file .env onyx:latest
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder — install deps into a wheel cache
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Build tools (needed for some packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy metadata first (better layer caching)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install everything into a virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir ".[api]"


# ---------------------------------------------------------------------------
# Stage 2: runtime — slim image, non-root, no build tools
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 onyx \
    && useradd --uid 1000 --gid onyx --shell /bin/bash --create-home onyx

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy application source
WORKDIR /app
COPY --chown=onyx:onyx src/ ./src/
COPY --chown=onyx:onyx pyproject.toml README.md LICENSE* ./

# Reinstall the package itself (editable not needed; already in venv via .[api])
RUN pip install --no-cache-dir --no-deps .

# Data directory (will be mounted as a volume in prod)
RUN mkdir -p /data && chown onyx:onyx /data
ENV DATA_DIR=/data \
    LOG_LEVEL=INFO

# Switch to non-root
USER onyx

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default command: run the API
CMD ["uvicorn", "onyx.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
