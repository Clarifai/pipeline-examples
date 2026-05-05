ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/nonroot/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=600 \
    LLMC_DIR=/opt/llm-compressor \
    LLMC_REF=7a52e9e

RUN apt-get update && apt-get install -y --no-install-recommends \
        git tmux htop jq less procps bc \
    && rm -rf /var/lib/apt/lists/*

# Pinned llm-compressor commit + transitive deps (compressed-tensors etc.).
# main HEAD imports from transformers internals that drift weekly; this commit
# + the pins below form a tested set on nvcr.io/nvidia/pytorch:25.10-py3.
RUN git clone https://github.com/vllm-project/llm-compressor.git "$LLMC_DIR" \
    && git -C "$LLMC_DIR" checkout "$LLMC_REF" \
    && python3 -m pip install --ignore-installed -e "$LLMC_DIR"

# transformers 5.7 has a keras_nlp BACKENDS_MAPPING regression; 5.2 imports cleanly.
# huggingface_hub must be >=1.5,<2 for transformers 5.x.
# torchvision must match torch 2.11 (base image ships torch-2.9-pinned torchvision).
# compressed-tensors 0.15.1.a20260421 has the FP4 weight_scale fix and matches the LLMC_REF API.
RUN python3 -m pip install --no-cache-dir --upgrade \
        "transformers==5.2.0" "huggingface-hub>=1.5,<2" torchvision \
    && python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
        "compressed-tensors==0.15.1.a20260421"

COPY --link requirements.txt /home/nonroot/main/requirements.txt
RUN python3 -m pip install --ignore-installed -r /home/nonroot/main/requirements.txt

COPY --link 1 /home/nonroot/main/1
COPY --link config.yaml /home/nonroot/main/config.yaml

WORKDIR /home/nonroot/main
