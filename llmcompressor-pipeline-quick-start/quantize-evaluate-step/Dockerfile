ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/nonroot/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=600 \
    LLMC_DIR=/opt/llm-compressor \
    LLMC_REF=7a52e9e

# vllm/vllm-openai sets ENTRYPOINT ["vllm","serve"]; clear it so the container
# can run arbitrary commands (this step's pipeline_step.py).
ENTRYPOINT []

RUN apt-get update && apt-get install -y --no-install-recommends \
        git tmux htop jq less procps bc curl \
    && rm -rf /var/lib/apt/lists/*

# vllm/vllm-openai ships /usr/local/bin/python3 but no /usr/local/bin/python.
# Clarifai's pipeline runner invokes `python` directly; add the symlink so
# `python pipeline_step.py` resolves at runtime.
RUN ln -sf "$(command -v python3)" /usr/local/bin/python

# llm-compressor at a pinned commit + evalscope, in a single pip pass.
# llm-compressor@7a52e9e pins transformers<=4.57.6,>=4.56.1, and that
# transformers branch in turn requires huggingface-hub<1.0. evalscope's
# modelscope[datasets] dep otherwise pulls hub>=1.0, which breaks the
# transformers import (`huggingface-hub>=0.34.0,<1.0 is required`). Pinning
# huggingface-hub<1.0 here keeps the whole stack consistent.
RUN git clone https://github.com/vllm-project/llm-compressor.git "$LLMC_DIR" \
    && git -C "$LLMC_DIR" checkout "$LLMC_REF" \
    && python3 -m pip install --no-cache-dir \
         -e "$LLMC_DIR" \
         "evalscope==1.6.1" \
         "huggingface-hub<1.0,>=0.34"

COPY --link requirements.txt /home/nonroot/main/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /home/nonroot/main/requirements.txt

COPY --link 1 /home/nonroot/main/1
COPY --link config.yaml /home/nonroot/main/config.yaml

WORKDIR /home/nonroot/main
