FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --link requirements.txt /home/nonroot/requirements.txt
RUN pip install --no-cache-dir -r /home/nonroot/requirements.txt

COPY --link=true 1 /home/nonroot/main/1
COPY --link=true requirements.txt config.yaml /home/nonroot/main/
