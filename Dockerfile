# ── Stage 1: base image ──────────────────────────────────────────────────────
# Use the official slim Python image. CPU-only; no CUDA needed for inference.
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="student"
LABEL description="FashionMNIST prediction API (FastAPI + PyTorch)"

# ── System dependencies ───────────────────────────────────────────────────────
# libgomp1 is required by PyTorch for OpenMP multi-threading on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker layer cache is preserved when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application files ────────────────────────────────────────────────────
# model.pt must exist before building the image (run train.py first)
COPY app.py    .
COPY model.pt  .

# ── Expose API port ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start FastAPI with Uvicorn ────────────────────────────────────────────────
# --host 0.0.0.0  → listen on all interfaces inside the container
# --port 8000     → container-internal port (mapped to host via -p flag)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
