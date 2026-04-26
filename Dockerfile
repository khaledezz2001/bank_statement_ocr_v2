FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# System deps for PDFs
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install vLLM + Python deps (torch 2.8.0 + CUDA 12.8.1 already in base image)
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# ===============================
# DOWNLOAD Qwen3.6-27B
# ===============================
RUN python3 -u <<'EOF'
from huggingface_hub import snapshot_download

print("Downloading Qwen/Qwen3.6-27B...", flush=True)

snapshot_download(
    repo_id="Qwen/Qwen3.6-27B",
    local_dir="/models/qwen3.6-27b",
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Qwen3.6-27B download complete", flush=True)
EOF

WORKDIR /app
COPY handler.py /app/handler.py

ENTRYPOINT ["python3"]
CMD ["-u", "handler.py"]
