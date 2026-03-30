# ---------------------------------------------------------------------------
# Threat Identification System — CPU Build
# Multi-stage build for minimal production image
# ---------------------------------------------------------------------------

# ---- Stage 1: Builder ----
FROM python:3.14-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libpq-dev \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install .

# ---- Stage 2: Runtime ----
FROM python:3.14-slim AS runtime

LABEL maintainer="DarthAether"
LABEL org.opencontainers.image.source="https://github.com/DarthAether/threat-identification-system"
LABEL org.opencontainers.image.description="Threat Identification System — CPU Runtime"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd --system app && \
    useradd --system --gid app --create-home app

WORKDIR /app

# Copy application source
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml ./

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "threat_id"]
