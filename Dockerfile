# 构建参数：cuda 或 cpu，默认 cuda
ARG DEVICE=cuda

# ============ CUDA 基础镜像 ============
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS base-cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* /root/.cache

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -rf /root/.cache /tmp/*

# ============ CPU 基础镜像 ============
FROM python:3.10-slim AS base-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -rf /root/.cache /tmp/*

# ============ 最终镜像 ============
FROM base-${DEVICE} AS final

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY app.py .
COPY templates/ ./templates/
RUN mkdir -p /app/weights /app/hls

EXPOSE 5000
VOLUME ["/app/weights", "/app/hls"]
CMD ["python3", "app.py"]
