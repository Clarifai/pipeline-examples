FROM --platform=linux/amd64 pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2
RUN pip install --no-cache-dir openmim
RUN mim install "mmengine==0.10.7"
RUN mim install "mmcv==2.1.0"
RUN mim install "mmdet==3.3.0"
RUN pip install --no-cache-dir numpy==1.26.4

COPY --link requirements.txt /home/nonroot/requirements.txt
RUN pip install --no-cache-dir -r /home/nonroot/requirements.txt

COPY --link=true 1 /home/nonroot/main/1
COPY --link=true requirements.txt config.yaml /home/nonroot/main/
