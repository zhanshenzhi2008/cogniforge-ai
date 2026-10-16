# =============================================================================
# Cogniforge AI Service - Dockerfile
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install runtime deps from pyproject.toml only（不打包 hatch wheel，避免缺 README / 包目录）
COPY pyproject.toml .
RUN python -c 'import tomllib; from pathlib import Path; deps=tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]; Path("/tmp/requirements.txt").write_text("\n".join(deps)+"\n")' \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Copy application code（PYTHONPATH=/app，uvicorn 直接跑 app.main）
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8086

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8086/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8086"]
