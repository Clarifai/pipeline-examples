# syntax=docker/dockerfile:1.13-labs
# Build context: model-version-train-ps/1/models/model/
# docker build -f Dockerfil_amd -t lora-serve .
FROM --platform=$TARGETPLATFORM vllm/vllm-openai:v0.15.1 as final

COPY --link requirements.txt /home/nonroot/requirements.txt

# vllm base image bundles vllm, torch, CUDA. Install remaining deps on top.
RUN ["pip", "install", "--no-cache-dir", "-r", "/home/nonroot/requirements.txt"]
RUN ["pip", "show", "--no-cache-dir", "clarifai"]

ENV NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
    HOME=/tmp \
    DEBIAN_FRONTEND=noninteractive

# Create checkpoint cache dir (HF downloader populates this at runtime)
COPY --chown=nonroot:nonroot downloader/unused.yaml /home/nonroot/main/1/checkpoints/.cache/unused.yaml

# Download checkpoints if config.yaml has checkpoints.when = "build"
COPY --link=true config.yaml /home/nonroot/main/
RUN ["python3", "-m", "clarifai.cli", "model", "download-checkpoints", "/home/nonroot/main", "--out_path", "/home/nonroot/main/1/checkpoints", "--stage", "build"]

# Copy model code, config, and requirements
COPY --link=true 1 /home/nonroot/main/1
COPY --link=true requirements.txt config.yaml /home/nonroot/main/

ENV PYTHONPATH=${PYTHONPATH}:/home/nonroot/main \
    CLARIFAI_PAT=${CLARIFAI_PAT} \
    CLARIFAI_USER_ID=${CLARIFAI_USER_ID} \
    CLARIFAI_RUNNER_ID=${CLARIFAI_RUNNER_ID} \
    CLARIFAI_NODEPOOL_ID=${CLARIFAI_NODEPOOL_ID} \
    CLARIFAI_COMPUTE_CLUSTER_ID=${CLARIFAI_COMPUTE_CLUSTER_ID} \
    CLARIFAI_API_BASE=${CLARIFAI_API_BASE:-https://api.clarifai.com}

USER root
RUN echo "nonroot:x:65532:65532:nonroot user:/home/nonroot:/sbin/nologin" >> /etc/passwd && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    chown -R 65532:65532 /home/nonroot/main
USER nonroot

ENTRYPOINT ["python3", "-m", "clarifai.runners.server"]
CMD ["--model_path", "/home/nonroot/main"]
