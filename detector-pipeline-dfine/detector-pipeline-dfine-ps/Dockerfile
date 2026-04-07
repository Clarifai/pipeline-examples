# D-FINE fine-tuning training container
# Adapted from: pipeline-examples/detector-pipeline-yolof-quick-start/model-version-train-ps/Dockerfile
# Key difference: uses transformers instead of mmdetection
FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "transformers>=4.51.1" timm accelerate opencv-python-headless onnx onnxsim scipy

COPY --link requirements.txt /home/nonroot/requirements.txt
RUN pip install --no-cache-dir -r /home/nonroot/requirements.txt

COPY --link=true 1 /home/nonroot/main/1
COPY --link=true requirements.txt config.yaml /home/nonroot/main/
