# syntax=docker/dockerfile:1.13-labs
# Build context: model-version-train-ps/
# docker build -f Dockerfile_amd -t lora-train .
#
# Base: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
#   -> Python 3.11, torch 2.5.1, CUDA 12.4, x86_64/amd64
FROM --platform=$TARGETPLATFORM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV TORCH_CUDA_ARCH_LIST="8.0;8.6" \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCHDYNAMO_DISABLE=1 \
    XFORMERS_DISABLED=1

RUN apt-get update && apt-get install -y \
    git ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY --link requirements.txt /home/nonroot/requirements.txt

# Install all deps except unsloth first (avoids pulling incompatible torch)
RUN grep -v "^unsloth" /home/nonroot/requirements.txt > /tmp/req_no_unsloth.txt && \
    pip install --no-cache-dir -r /tmp/req_no_unsloth.txt

# Install unsloth + zoo without transitive deps (avoids reinstalling torch)
RUN pip install --no-cache-dir --no-deps unsloth==2025.7.8 unsloth_zoo==2025.7.10

# Copy in the actual files like config.yaml, requirements.txt, and most importantly 1/pipeline_step.py
COPY --link=true 1 /home/nonroot/main/1
# At this point we only need these for validation in the SDK.
COPY --link=true requirements.txt config.yaml /home/nonroot/main/

ENV PYTHONPATH=${PYTHONPATH}:/home/nonroot/main \
    CLARIFAI_PAT=${CLARIFAI_PAT} \
    CLARIFAI_API_BASE=${CLARIFAI_API_BASE:-https://api.clarifai.com}

ENTRYPOINT ["python3", "-m", "clarifai.runners.server"]
CMD ["--model_path", "/home/nonroot/main"]
