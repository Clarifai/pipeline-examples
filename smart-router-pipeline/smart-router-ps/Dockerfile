FROM --platform=linux/amd64 python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --link requirements.txt /home/nonroot/requirements.txt
RUN pip install --no-cache-dir -r /home/nonroot/requirements.txt

# Copy this step's source
COPY --link=true 1 /home/nonroot/main/1
COPY --link=true requirements.txt config.yaml /home/nonroot/main/

# Copy the downstream skill steps so the router can import them
COPY --link=true ../../code-review-pipeline/python-code-review-ps/1 /home/nonroot/skills/code-review/1
COPY --link=true ../../chart-analysis-pipeline/chart-analysis-ps/1 /home/nonroot/skills/chart-analysis/1
COPY --link=true ../../snap-eval-pipeline/snap-eval-ps/1 /home/nonroot/skills/snap-eval/1
