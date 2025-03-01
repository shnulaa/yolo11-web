# RTSP to HLS Stream with YOLOv11

这个项目是一个基于Flask的Web应用，可以接收RTSP视频流，使用YOLOv11进行实时目标检测，并将处理后的视频转换为HLS格式进行流式传输。

## 功能特点

- 接收RTSP视频流并进行实时处理
- 使用YOLOv11进行目标检测
- 支持动态切换不同的YOLOv11模型
- 可调整检测参数（置信度、IOU阈值、线宽）
- 将处理后的视频转换为HLS格式进行流式传输
- 通过Web界面控制和查看视频流

## 系统要求

- Python 3.8+
- NVIDIA GPU (推荐用于加速检测)
- CUDA和cuDNN (用于GPU加速)
- FFmpeg (用于视频转换)

## 安装指南

### 1. 克隆仓库
```bash
git clone https://github.com/shnulaa/yolo11-web
cd yolo11-web
```

### 2. 创建虚拟环境（可选但推荐）

```bash
conda　create -n yolo11-web python=3.10
conda activate yolo11-web
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装FFmpeg

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### CentOS/RHEL:
```bash
sudo yum install ffmpeg ffmpeg-devel
```

#### Windows:
从[FFmpeg官网](https://ffmpeg.org/download.html)下载并安装，确保将FFmpeg添加到系统PATH中。

#### macOS:
```bash
brew install ffmpeg
```

### 5. 准备YOLOv11模型

在项目根目录下创建`weights`文件夹，并将YOLOv11模型文件（.pt格式）放入该文件夹中。

```bash
mkdir -p weights
# 将YOLOv11模型文件放入weights文件夹
```

## 使用方法

### 1. 启动应用

```bash
python app.py
```

应用将在`http://localhost:5000`上运行。

### 2. 使用Web界面

1. 在浏览器中打开`http://localhost:5000`
2. 在RTSP URL输入框中输入RTSP流地址
3. 点击"启动流"按钮开始处理视频流
4. 调整检测参数（置信度、IOU阈值、线宽、帧率）并点击"更新参数"应用更改
5. 从下拉菜单中选择不同的YOLOv11模型并点击"切换"切换模型
6. 点击"停止流"停止视频流处理

## 项目结构

```
yolo11-web/
├── app.py                # 主应用程序
├── templates/            # HTML模板
│   └── index.html        # Web界面
├── hls/                  # HLS流文件目录（自动创建）
├── weights/              # YOLOv11模型目录
├── Dockerfile            # Docker构建文件
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明
```

## 使用Docker部署

本项目提供了完整的Docker支持，可以轻松部署到任何支持Docker和NVIDIA GPU的环境中。

### 前提条件

1. 安装Docker：[Docker安装指南](https://docs.docker.com/get-docker/)
2. 安装NVIDIA Container Toolkit：[安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. 确保您的系统上有可用的NVIDIA GPU

### 1. 构建Docker镜像

在项目根目录下执行以下命令构建Docker镜像：

```bash
docker build -t yolo11-web .
```

这将创建一个名为`yolo11-web`的Docker镜像，包含所有必要的依赖。

### 2. 运行Docker容器

#### 基本运行方式

```bash
docker run -d \
  --name yolo11-web \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/hls:/app/hls \
  yolo11-web
```

#### 参数说明

- `-d`: 在后台运行容器
- `--name yolo11-web`: 指定容器名称
- `--gpus all`: 允许容器访问所有GPU
- `-p 5000:5000`: 将容器的5000端口映射到主机的5000端口
- `-v $(pwd)/weights:/app/weights`: 挂载模型目录
- `-v $(pwd)/hls:/app/hls`: 挂载HLS流文件目录

#### 生产环境配置

对于生产环境，建议添加自动重启策略：

```bash
docker run -d \
  --name yolo11-web \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/hls:/app/hls \
  --restart unless-stopped \
  yolo11-web
```

### 3. 查看容器日志

```bash
docker logs -f yolo11-web
```

### 4. 停止和删除容器

```bash
# 停止容器
docker stop yolo11-web

# 删除容器
docker rm yolo11-web
```

### 5. 更新容器

当您对代码进行更改后，需要重新构建镜像并更新容器：

```bash
# 重新构建镜像
docker build -t yolo11-web .

# 停止并删除旧容器
docker stop yolo11-web
docker rm yolo11-web

# 启动新容器
docker run -d \
  --name yolo11-web \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/hls:/app/hls \
  --restart unless-stopped \
  yolo11-web
```

### 6. Docker Compose (可选)

如果您使用Docker Compose，可以创建一个`docker-compose.yml`文件：

```yaml
version: '3'
services:
  yolo11-web:
    build: .
    container_name: yolo11-web
    ports:
      - "5000:5000"
    volumes:
      - ./weights:/app/weights
      - ./hls:/app/hls
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

然后使用以下命令启动服务：

```bash
docker-compose up -d
```

## 常见问题

1. **无法连接到RTSP流**
   - 确保RTSP URL格式正确
   - 检查网络连接和防火墙设置
   - 验证RTSP服务器是否正常运行

2. **检测性能较慢**
   - 确保使用GPU加速
   - 尝试降低视频分辨率
   - 调整检测参数（提高置信度阈值）
   - 调整帧率参数

3. **FFmpeg错误**
   - 确保FFmpeg正确安装
   - 检查FFmpeg版本是否支持所需功能
   - 查看应用日志获取详细错误信息

4. **Docker相关问题**
   - 确保NVIDIA Container Toolkit正确安装
   - 检查GPU驱动版本是否与CUDA版本兼容
   - 使用`docker logs`查看容器日志以获取详细错误信息

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交问题和拉取请求！
