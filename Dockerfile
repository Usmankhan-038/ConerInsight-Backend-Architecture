FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES=-1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/tmp/huggingface \
    PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser \
    && mkdir -p /app /tmp/matplotlib /tmp/huggingface \
    && chown -R appuser:appuser /app /tmp/matplotlib /tmp/huggingface

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
