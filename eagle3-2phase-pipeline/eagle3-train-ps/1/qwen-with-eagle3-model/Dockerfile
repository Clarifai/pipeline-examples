# syntax=docker/dockerfile:1.13-labs
FROM --platform=$TARGETPLATFORM lmsysorg/sglang:v0.5.9 as final

ENV NUMBA_CACHE_DIR=/tmp/numba_cache \
    TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache \
    HOME=/tmp \
    DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ── Python runtime deps ──────────────────────────────────────────────
COPY --link requirements.txt /home/nonroot/requirements.txt
RUN pip install --no-cache-dir -r /home/nonroot/requirements.txt && pip show clarifai

# ── Checkpoint cache dir (HF downloader populates at build/runtime) ──
COPY --chown=nonroot:nonroot downloader/unused.yaml /home/nonroot/main/1/checkpoints/.cache/unused.yaml

# ── Checkpoint download at build time ────────────────────────────────
COPY --link config.yaml /home/nonroot/main/
RUN ["python", "-m", "clarifai.cli", "model", "download-checkpoints", "/home/nonroot/main", "--out_path", "/home/nonroot/main/1/checkpoints", "--stage", "build"]

# ── Model code ───────────────────────────────────────────────────────
COPY --link=true 1 /home/nonroot/main/1
COPY --link=true requirements.txt config.yaml /home/nonroot/main/

# ── Environment ──────────────────────────────────────────────────────
ENV PYTHONPATH=${PYTHONPATH}:/home/nonroot/main \
    CLARIFAI_PAT=${CLARIFAI_PAT} \
    CLARIFAI_USER_ID=${CLARIFAI_USER_ID} \
    CLARIFAI_RUNNER_ID=${CLARIFAI_RUNNER_ID} \
    CLARIFAI_NODEPOOL_ID=${CLARIFAI_NODEPOOL_ID} \
    CLARIFAI_COMPUTE_CLUSTER_ID=${CLARIFAI_COMPUTE_CLUSTER_ID} \
    CLARIFAI_API_BASE=${CLARIFAI_API_BASE:-https://api.clarifai.com}

# ── Nonroot user (needed for getpass.getuser() in torch) ─────────────
USER root
RUN echo "nonroot:x:65532:65532:nonroot user:/home/nonroot:/sbin/nologin" >> /etc/passwd && \
    chown -R 65532:65532 /home/nonroot/main
USER nonroot

ENTRYPOINT ["python", "-m", "clarifai.runners.server"]
CMD ["--model_path", "/home/nonroot/main"]
