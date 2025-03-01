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
git clone https://github.com/yourusername/rtsp-hls-yolov11.git
cd rtsp-hls-yolov11
```

### 2. 创建虚拟环境（可选但推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
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
3. 点击"Start Stream"按钮开始处理视频流
4. 调整检测参数（置信度、IOU阈值、线宽）并点击"Update Parameters"应用更改
5. 从下拉菜单中选择不同的YOLOv11模型并点击"Change Model"切换模型
6. 点击"Stop Stream"停止视频流处理

## 项目结构

```
rtsp-hls-yolov11/
├── app.py                # 主应用程序
├── templates/            # HTML模板
│   └── index.html        # Web界面
├── hls/                  # HLS流文件目录（自动创建）
├── weights/              # YOLOv11模型目录
├── requirements.txt      # 项目依赖
└── README.md             # 项目说明
```

## 使用Docker部署（可选）

### 1. 构建Docker镜像

```bash
docker build -t rtsp-hls-yolov11 .
```

### 2. 运行Docker容器

```bash
docker run -d -p 5000:5000 --gpus all -v /path/to/your/weights:/app/weights rtsp-hls-yolov11
```

注意：请将`/path/to/your/weights`替换为您存放YOLOv11模型的实际路径。

## 常见问题

1. **无法连接到RTSP流**
   - 确保RTSP URL格式正确
   - 检查网络连接和防火墙设置
   - 验证RTSP服务器是否正常运行

2. **检测性能较慢**
   - 确保使用GPU加速
   - 尝试降低视频分辨率
   - 调整检测参数（提高置信度阈值）

3. **FFmpeg错误**
   - 确保FFmpeg正确安装
   - 检查FFmpeg版本是否支持所需功能
   - 查看应用日志获取详细错误信息

## 许可证

[MIT License](LICENSE)

## 贡献

欢迎提交问题和拉取请求！
