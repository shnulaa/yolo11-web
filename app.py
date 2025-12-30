from flask import Flask, render_template, send_from_directory, request, jsonify
import subprocess
import os
import logging
import threading
import cv2
from ultralytics import YOLO
import time
import shutil
import torch

# 配置日志（必须在其他日志调用之前）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__, template_folder=os.path.abspath('templates'))

# 检测是否有 CUDA 可用
USE_CUDA = torch.cuda.is_available()
logging.info(f"CUDA available: {USE_CUDA}")

# 检测 FFmpeg 是否支持 NVENC
def check_nvenc_support():
    try:
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        if "h264_nvenc" not in result.stdout:
            return False
        test = subprocess.run(
            ["ffmpeg", "-f", "lavfi", "-i", "nullsrc=s=64x64:d=1", "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True, text=True
        )
        return test.returncode == 0
    except:
        return False

USE_NVENC = USE_CUDA and check_nvenc_support()
logging.info(f"NVENC available: {USE_NVENC}")

# HLS 文件目录
HLS_DIR = "./hls/"
os.makedirs(HLS_DIR, exist_ok=True)

# 全局变量
rtsp_url = ""
ffmpeg_process = None
running = False
model = None
current_model = "yolo11m.pt"
WEIGHTS_DIR = "./weights/"

# 参数设置
conf = 0.5
iou = 0.45
line_width = 2
fps = 0

def load_model(model_name):
    """加载指定模型"""
    model_path = os.path.join(WEIGHTS_DIR, model_name)
    if not os.path.exists(model_path):
        logging.error(f"Model {model_name} not found in {WEIGHTS_DIR}!")
        raise FileNotFoundError(f"Model {model_name} not found!")
    logging.info(f"Loading model: {model_name}")
    return YOLO(model_path)

def process_rtsp_stream(rtsp_url):
    """处理 RTSP 流并进行 YOLOv11 检测"""
    global running, model, ffmpeg_process, conf, iou, line_width, fps

    model = load_model(current_model)

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logging.error(f"Failed to open RTSP stream: {rtsp_url}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_fps = fps if fps > 0 else original_fps
    logging.info(f"Original FPS: {original_fps}, Output FPS: {output_fps}")

    # FFmpeg 命令
    if USE_NVENC:
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(output_fps), "-i", "-",
            "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ll",
            "-f", "hls", "-hls_time", "10", "-hls_list_size", "10",
            "-hls_flags", "delete_segments", f"{HLS_DIR}/stream.m3u8"
        ]
    else:
        ffmpeg_command = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(output_fps), "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-f", "hls", "-hls_time", "10", "-hls_list_size", "10",
            "-hls_flags", "delete_segments", f"{HLS_DIR}/stream.m3u8"
        ]

    logging.info(f"FFmpeg command: {' '.join(ffmpeg_command)}")
    try:
        ffmpeg_process = subprocess.Popen(
            ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE
        )
    except Exception as e:
        logging.error(f"Failed to start FFmpeg: {e}")
        return

    def read_stderr():
        for line in ffmpeg_process.stderr:
            logging.info(f"FFmpeg: {line.decode().strip()}")
    
    threading.Thread(target=read_stderr, daemon=True).start()
        
    running = True
    logging.info(f"Processing RTSP stream: {rtsp_url}")

    consecutive_failures = 0
    max_failures = 30
    frame_delay = 1.0 / fps if fps > 0 else (1.0 / original_fps if original_fps > 0 else 0.033)
    logging.info(f"Frame delay: {frame_delay} seconds")

    while running:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            logging.warning(f"Failed to read frame. Attempt {consecutive_failures}/{max_failures}")
            if consecutive_failures >= max_failures:
                logging.error("Too many consecutive failures. Stopping stream.")
                break
            time.sleep(1)
            continue
        
        consecutive_failures = 0
        
        try:
            results = model.predict(frame, conf=conf, iou=iou, line_width=line_width)
            annotated_frame = results[0].plot()
            ffmpeg_process.stdin.write(annotated_frame.tobytes())
        except BrokenPipeError:
            logging.error("FFmpeg process died (BrokenPipe)")
            break
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            
        time.sleep(frame_delay)

    cap.release()
    if ffmpeg_process:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
    running = False
    logging.info("RTSP stream processing stopped.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hls/<filename>')
def hls(filename):
    return send_from_directory(HLS_DIR, filename)

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global rtsp_url, running
    rtsp_url = request.json.get('rtsp_url', "")
    if not rtsp_url:
        return "RTSP URL is required!", 400

    if running:
        running = False
        time.sleep(1)

    threading.Thread(target=process_rtsp_stream, args=(rtsp_url,), daemon=True).start()
    return "Stream started successfully!"

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global running, ffmpeg_process
    running = False
    
    if ffmpeg_process is not None:
        try:
            ffmpeg_process.stdin.close()
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=5)
        except Exception as e:
            logging.error(f"Error terminating FFmpeg: {e}")
            try:
                ffmpeg_process.kill()
            except:
                pass
        ffmpeg_process = None
    
    delete_hls_files()
    return "Stream stopped successfully!"

def delete_hls_files():
    for filename in os.listdir(HLS_DIR):
        file_path = os.path.join(HLS_DIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error(f"Failed to delete {file_path}: {e}")

@app.route('/update_params', methods=['POST'])
def update_params():
    global conf, iou, line_width, fps, rtsp_url, running
    data = request.json
    conf = float(data.get('conf', conf))
    iou = float(data.get('iou', iou))
    line_width = int(data.get('line_width', line_width))
    new_fps = int(data.get('fps', fps))
    
    if new_fps != fps:
        fps = new_fps
        logging.info(f"FPS changed to: {fps}")
    
    logging.info(f"Updated params: conf={conf}, iou={iou}, line_width={line_width}, fps={fps}")

    if running:
        stop_stream()
        delete_hls_files()
        time.sleep(1)
        threading.Thread(target=process_rtsp_stream, args=(rtsp_url,), daemon=True).start()
        
    return jsonify({"status": "success", "conf": conf, "iou": iou, "line_width": line_width, "fps": fps})

@app.route('/get_params', methods=['GET'])
def get_params():
    return jsonify({"conf": conf, "iou": iou, "line_width": line_width, "fps": fps})

@app.route('/get_models', methods=['GET'])
def get_models():
    models = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith('.pt')]
    return jsonify({"models": models, "current_model": current_model})

@app.route('/change_model', methods=['POST'])
def change_model():
    global current_model, running
    model_name = request.json.get('model_name', "")
    if not model_name:
        return jsonify({"status": "error", "message": "Model name required!"}), 400
    
    model_path = os.path.join(WEIGHTS_DIR, model_name)
    if not os.path.exists(model_path):
        return jsonify({"status": "error", "message": f"Model {model_name} not found!"}), 404
    
    current_model = model_name
    
    if running:
        stop_stream()
        delete_hls_files()
        time.sleep(1)
        threading.Thread(target=process_rtsp_stream, args=(rtsp_url,), daemon=True).start()
    
    return jsonify({"status": "success", "message": f"Model changed to {model_name}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
