# 构建阶段
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive
COPY sources/sources.list /etc/apt/sources.list

# 只安装构建必需的包
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /build
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 最终阶段
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
COPY sources/sources.list /etc/apt/sources.list

# 只安装运行时必需的包
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-minimal \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从构建阶段复制Python包
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# 创建必要目录
RUN mkdir -p hls weights

# 复制应用文件
COPY . .

EXPOSE 5000
CMD ["python3", "app.py"]
