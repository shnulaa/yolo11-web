from flask import Flask, render_template, send_from_directory, request, jsonify
import subprocess
import os
import logging
import threading
import cv2
from ultralytics import YOLO
import time
import shutil

app = Flask(__name__, 
    template_folder=os.path.abspath('templates'))  # 确保模板目录正确设置

# HLS 文件目录
# HLS_DIR = "/opt/docker/ultralytics/websocket/hls/"
HLS_DIR = "./hls/"
os.makedirs(HLS_DIR, exist_ok=True)

# 全局变量
rtsp_url = ""
ffmpeg_process = None
running = False
model = None
current_model = "yolo11m.pt"  # 默认模型
# WEIGHTS_DIR = "/opt/docker/ultralytics/websocket/weights/"  # 模型文件夹路径
WEIGHTS_DIR = "./weights/"  # 模型文件夹路径

# 参数设置 - 修改为全局变量，以便在刷新页面后保持状态
conf = 0.5
iou = 0.45
line_width = 2
fps = 0  # 0表示使用原始帧率，其他值表示自定义帧率

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

def load_model(model_name):
    """加载指定模型"""
    model_path = os.path.join(WEIGHTS_DIR, model_name)
    if not os.path.exists(model_path):
        logging.error(f"Model {model_name} not found in {WEIGHTS_DIR}!")
        raise FileNotFoundError(f"Model {model_name} not found in {WEIGHTS_DIR}!")
    logging.info(f"Loading model: {model_name}")
    return YOLO(model_path)

def process_rtsp_stream(rtsp_url):
    """处理 RTSP 流并进行 YOLOv11 检测"""
    global running, model, ffmpeg_process
    global conf, iou, line_width, fps

    # 加载模型
    model = load_model(current_model)

    # 打开 RTSP 流
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logging.error(f"Failed to open RTSP stream: {rtsp_url}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 使用自定义帧率或原始帧率
    output_fps = fps if fps > 0 else original_fps
    logging.info(f"Original FPS: {original_fps}, Output FPS: {output_fps}")

    # FFmpeg 命令（将检测后的帧保存为 HLS）
    ffmpeg_command = [
        "ffmpeg",
        "-y",  # 覆盖输出文件
        "-hwaccel", "cuda",  # 使用 CUDA 硬件加速
        "-hwaccel_output_format", "cuda",  # 硬件加速输出格式
        "-f", "rawvideo",  # 输入格式为原始视频
        "-pix_fmt", "bgr24",  # 输入像素格式
        "-s", f"{width}x{height}",  # 输入分辨率
        "-r", str(output_fps),  # 输入帧率 - 使用自定义或原始帧率
        "-i", "-",  # 从标准输入读取
        "-vf", "format=nv12,hwupload_cuda",  # 将帧上传到 GPU
        "-c:v", "h264_nvenc",  # 使用 NVIDIA 的 h264_nvenc 编码器
        "-f", "hls",  # 输出格式为 HLS
        "-hls_time", "10",  # 每个切片时长（秒）
        "-hls_list_size", "10",  # 播放列表中的切片数量
        "-hls_flags", "delete_segments",  # 自动删除旧切片
        "-vsync", "0",  # 禁用自动帧率同步
        f"{HLS_DIR}/stream.m3u8"  # 输出文件路径
    ]

    # 启动 FFmpeg 进程
    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
    running = True
    logging.info(f"Processing RTSP stream with YOLOv11: {rtsp_url}")

    consecutive_failures = 0
    max_failures = 30  # 允许的最大连续失败次数
    
    # 计算帧间延迟时间（用于控制处理帧率）
    if fps > 0:
        frame_delay = 1.0 / fps
    else:
        frame_delay = 1.0 / original_fps if original_fps > 0 else 0.033  # 默认约30fps
    
    logging.info(f"Frame delay: {frame_delay} seconds")

    while running:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            logging.warning(f"Failed to read frame from RTSP stream. Attempt {consecutive_failures}/{max_failures}")
            
            if consecutive_failures >= max_failures:
                logging.error("Too many consecutive failures. Stopping stream.")
                break
                
            # 短暂等待后重试
            time.sleep(1)
            continue
        
        # 成功读取帧，重置失败计数
        consecutive_failures = 0
        
        # 使用 YOLOv11 进行目标检测
        try:
            # 使用全局变量中的最新参数值
            current_conf = conf
            current_iou = iou
            current_line_width = line_width
            
            results = model.predict(frame, conf=current_conf, iou=current_iou, line_width=current_line_width)
            # 绘制检测结果
            annotated_frame = results[0].plot()
            # 将处理后的帧写入 FFmpeg 进程
            ffmpeg_process.stdin.write(annotated_frame.tobytes())
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            
        # 使用计算的帧延迟来控制帧率
        time.sleep(frame_delay)

    # 释放资源
    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    running = False
    logging.info("RTSP stream processing stopped.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hls/<filename>')
def hls(filename):
    """提供 HLS 文件的静态访问"""
    return send_from_directory(HLS_DIR, filename)

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """启动 RTSP 流转换和检测"""
    global rtsp_url
    rtsp_url = request.json.get('rtsp_url', "")
    if not rtsp_url:
        return "RTSP URL is required!", 400

    # 停止之前的处理线程（如果存在）
    global running
    if running:
        running = False
        time.sleep(1)  # 等待线程退出

    # 启动新的处理线程
    threading.Thread(target=process_rtsp_stream, args=(rtsp_url,), daemon=True).start()
    return "Stream started successfully!"

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """停止 RTSP 流转换和检测"""
    global running, ffmpeg_process
    running = False
    
    # Properly terminate the FFmpeg process
    if ffmpeg_process is not None:
        try:
            ffmpeg_process.stdin.close()
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=5)
        except Exception as e:
            logging.error(f"Error terminating FFmpeg process: {e}")
            try:
                ffmpeg_process.kill()
            except:
                pass
        ffmpeg_process = None
    
    # 删除旧的 HLS 文件
    delete_hls_files()
    return "Stream stopped successfully!"


def delete_hls_files():
    """删除 HLS 目录下的所有文件"""
    for filename in os.listdir(HLS_DIR):
        file_path = os.path.join(HLS_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error(f"Failed to delete {file_path}. Reason: {e}")

@app.route('/update_params', methods=['POST'])
def update_params():
    """更新检测参数"""
    global conf, iou, line_width, fps, rtsp_url
    data = request.json
    conf = float(data.get('conf', conf))
    iou = float(data.get('iou', iou))
    line_width = int(data.get('line_width', line_width))
    new_fps = int(data.get('fps', fps))
    
    # 只有当帧率发生变化时才记录
    if new_fps != fps:
        fps = new_fps
        logging.info(f"FPS changed to: {fps} (0 means use original)")
    
    logging.info(f"Updated parameters: conf={conf}, iou={iou}, line_width={line_width}, fps={fps}")

    # 只有在流正在运行时才重启流
    if running:
        # 停止当前流
        stop_stream()
        # 删除HLS缓存文件
        delete_hls_files()
        time.sleep(1)  # 等待线程退出
        # 重新启动流处理
        threading.Thread(target=process_rtsp_stream, args=(rtsp_url,), daemon=True).start()
        
    return jsonify({"status": "success", "conf": conf, "iou": iou, "line_width": line_width, "fps": fps})

@app.route('/get_params', methods=['GET'])
def get_params():
    """获取当前参数设置"""
    global conf, iou, line_width, fps
    return jsonify({
        "conf": conf,
        "iou": iou,
        "line_width": line_width,
        "fps": fps
    })

@app.route('/get_models', methods=['GET'])
def get_models():
    """获取可用的模型列表"""
    models = []
    for file in os.listdir(WEIGHTS_DIR):
        if file.endswith('.pt'):
            models.append(file)
    return jsonify({"models": models, "current_model": current_model})

@app.route('/change_model', methods=['POST'])
def change_model():
    """切换检测模型"""
    global current_model, model
    model_name = request.json.get('model_name', "")
    if not model_name:
        return jsonify({"status": "error", "message": "Model name is required!"}), 400
    
    model_path = os.path.join(WEIGHTS_DIR, model_name)
    if not os.path.exists(model_path):
        return jsonify({"status": "error", "message": f"Model {model_name} not found!"}), 404
    
    current_model = model_name
    
    # 如果当前正在运行，重新启动流以应用新模型
    if running:
        stop_stream()
        # 删除HLS缓存文件
        delete_hls_files()
        time.sleep(1)
        threading.Thread(target=process_rtsp_stream, args=(rtsp_url,), daemon=True).start()
    
    return jsonify({"status": "success", "message": f"Model changed to {model_name}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)