# ============================================================================
# SkinGuard AI — Dockerfile (Production)
# Multi-stage build for CUDA-enabled deployment
# ============================================================================

# ---- Stage 1: Build ----
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Runtime ----
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY configs/ configs/
COPY src/ src/
COPY app/ app/
COPY scripts/ scripts/

# Copy trained model (mount or copy)
# COPY outputs/checkpoints/best.pth outputs/checkpoints/best.pth

# Environment
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/configs/config.yaml
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Expose Gradio port
EXPOSE 7860

# Run the Gradio app
CMD ["python3.11", "app/gradio_app.py"]
