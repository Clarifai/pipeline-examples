# Eagle3 two-phase training pipeline step.
# Base: lmsysorg/sglang:v0.5.9 — matches SpecForge pins
# (sglang==0.5.9, torch==2.9.1, transformers==4.57.1).

ARG BASE_IMAGE=lmsysorg/sglang:v0.5.9
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/eagle3/cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/workspace/eagle3/cache/compiled_kernels \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
        jq procps \
    && rm -rf /var/lib/apt/lists/*

# Clone SpecForge repo (scripts reference /workspace/SpecForge directly)
# and install as editable + extra deps sglang:v0.5.9 doesn't ship.
RUN git clone https://github.com/sgl-project/SpecForge.git /workspace/SpecForge && \
    pip install --no-cache-dir --no-deps -e /workspace/SpecForge && \
    pip install --no-cache-dir \
        accelerate qwen-vl-utils==0.0.11 tensorboard wandb yunchang

COPY --link requirements.txt /home/nonroot/requirements.txt
RUN pip install --no-cache-dir -r /home/nonroot/requirements.txt

# Copy pipeline step code (pipeline_step.py, scripts/, configs/)
COPY --link=true 1 /home/nonroot/main/1
COPY --link=true requirements.txt config.yaml /home/nonroot/main/

# Symlink so scripts that reference /workspace/eagle3 find our code
RUN ln -s /home/nonroot/main/1 /workspace/eagle3

ENV PYTHONPATH=${PYTHONPATH}:/home/nonroot/main

ENTRYPOINT ["python3", "-m", "clarifai.runners.server"]
CMD ["--model_path", "/home/nonroot/main"]
