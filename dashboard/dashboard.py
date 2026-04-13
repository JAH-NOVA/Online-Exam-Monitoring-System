from flask import Flask, render_template, Response, jsonify
import cv2
import json
import os
from datetime import datetime
import threading
from queue import Queue
import time

app = Flask(__name__)

# Global variables for frame sharing
latest_frame = None
frame_lock = threading.Lock()
alert_queue = Queue(maxsize=100)

def get_latest_alerts(limit=10):
    """Get the latest alerts from the log file"""
    log_file = os.path.join("logs", "alerts.json")
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
                return sorted(logs, key=lambda x: x['timestamp'], reverse=True)[:limit]
        except Exception as e:
            print(f"Error reading log file: {e}")
    return []

def update_frame(frame):
    """Update the latest frame for streaming"""
    global latest_frame
    with frame_lock:
        latest_frame = frame.copy()

def add_alert(alert_type: str, message: str):
    """Add an alert to the queue"""
    alert_queue.put({
        "type": alert_type,
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def generate_frames():
    """Generate frames for video streaming"""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alerts')
def get_alerts():
    """Get recent alerts"""
    alerts = []
    while not alert_queue.empty():
        alerts.append(alert_queue.get())
    return jsonify(alerts)

@app.route('/alert_history')
def alert_history():
    """Get alert history from log file"""
    alerts = get_latest_alerts()
    return jsonify(alerts)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) 