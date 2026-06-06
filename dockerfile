ARG UV_VERSION=0.11.19

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ubuntu:24.04 AS builder

COPY --from=uv /uv /uvx /bin/

ARG PYTHON_VERSION=3.12
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON=${PYTHON_VERSION} \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

RUN uv python install ${PYTHON_VERSION} && \
    uv venv --relocatable --python ${PYTHON_VERSION} /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN uv pip install --index-url ${TORCH_INDEX_URL} torch
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install -r /tmp/requirements.txt

FROM spaiss6:8

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/python /opt/python
COPY --from=builder /opt/venv /opt/venv
COPY . /app/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_DIR=/models \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    ROOT_PATH="" \
    LOG_LEVEL=info \
    HEALTH_PATH=/health \
    GENERATE_PATH=/generate

WORKDIR /app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["/opt/venv/bin/python", "-c", \
         "import os,sys,urllib.request; url=f\"http://localhost:{os.environ.get('PORT','8000')}{os.environ.get('HEALTH_PATH','/health')}\"; sys.exit(0 if urllib.request.urlopen(url,timeout=4).status==200 else 1)"]

ENTRYPOINT ["/opt/venv/bin/python", "/app/main.py"]
