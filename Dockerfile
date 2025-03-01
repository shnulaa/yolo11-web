# 使用NVIDIA CUDA基础镜像
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 避免交互式前端
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 创建必要的目录
RUN mkdir -p /app/weights /app/hls

# 复制项目文件
COPY requirements.txt .
COPY app.py .
COPY templates/ ./templates/
COPY README.md .

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 安装额外的依赖
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 暴露端口
EXPOSE 5000

# 设置卷，用于持久化模型和HLS文件
VOLUME ["/app/weights", "/app/hls"]

# 启动应用
CMD ["python3", "app.py"] 