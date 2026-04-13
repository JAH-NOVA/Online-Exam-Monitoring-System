"""
Flask Web Application for Classroom Cheat Detection
Allows users to upload videos and view detection results
"""
from flask import Flask, render_template, request, jsonify, send_file, url_for, Response
from werkzeug.utils import secure_filename
import os
import threading
import json
import cv2
from datetime import datetime
from config import Config
from video_processor import VideoProcessor
from detectors.head_pose import HeadPoseDetector
from detectors.object_detector import ObjectDetector
from detectors.tracker import PersonTracker
from detectors.behavior_analyzer import BehaviorAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = Config.PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.PROCESSED_FOLDER, exist_ok=True)

# Store processing status
processing_status = {}
processing_lock = threading.Lock()

# Live detection state
live_detection_active = False
live_camera = None
live_lock = threading.Lock()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload page and handler"""
    if request.method == 'GET':
        return render_template('upload.html')
    
    # Handle file upload
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: ' + ', '.join(Config.ALLOWED_EXTENSIONS)}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Create processing job
    job_id = timestamp
    with processing_lock:
        processing_status[job_id] = {
            'status': 'queued',
            'progress': 0,
            'total_frames': 0,
            'current_frame': 0,
            'filename': filename
        }
    
    # Start processing in background thread
    thread = threading.Thread(target=process_video_task, args=(job_id, filepath))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': 'Video uploaded successfully. Processing started.'
    })

def process_video_task(job_id, video_path):
    """Background task to process video"""
    try:
        with processing_lock:
            processing_status[job_id]['status'] = 'processing'
        
        processor = VideoProcessor(video_path, app.config['PROCESSED_FOLDER'])
        
        def update_progress(current, total):
            with processing_lock:
                processing_status[job_id]['current_frame'] = current
                processing_status[job_id]['total_frames'] = total
                processing_status[job_id]['progress'] = int((current / total) * 100) if total > 0 else 0
        
        # Process the video
        result = processor.process_video(progress_callback=update_progress)
        
        # Update status with results
        with processing_lock:
            processing_status[job_id]['status'] = 'completed'
            processing_status[job_id]['progress'] = 100
            processing_status[job_id]['result'] = result
            processing_status[job_id]['output_video'] = os.path.basename(result['output_video'])
            processing_status[job_id]['report'] = os.path.basename(result['report'])
    
    except Exception as e:
        with processing_lock:
            processing_status[job_id]['status'] = 'error'
            processing_status[job_id]['error'] = str(e)

@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing status for a job"""
    with processing_lock:
        if job_id not in processing_status:
            return jsonify({'error': 'Job not found'}), 404
        return jsonify(processing_status[job_id])

@app.route('/results/<job_id>')
def results(job_id):
    """View results page"""
    with processing_lock:
        if job_id not in processing_status:
            return "Job not found", 404
        
        if processing_status[job_id]['status'] != 'completed':
            return "Processing not complete", 400
        
        job_data = processing_status[job_id]
    
    return render_template('results.html', job_id=job_id, job_data=job_data)

@app.route('/download/video/<filename>')
def download_video(filename):
    """Download processed video"""
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    return send_file(filepath, as_attachment=True)

@app.route('/download/report/<filename>')
def download_report(filename):
    """Download JSON report"""
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404
    return send_file(filepath, as_attachment=True)

@app.route('/api/report/<filename>')
def get_report(filename):
    """Get report data as JSON"""
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'Report not found'}), 404
    
    with open(filepath, 'r') as f:
        report_data = json.load(f)
    
    return jsonify(report_data)

@app.route('/history')
def history():
    """View processing history"""
    with processing_lock:
        jobs = []
        for job_id, data in processing_status.items():
            jobs.append({
                'job_id': job_id,
                'filename': data.get('filename', 'Unknown'),
                'status': data['status'],
                'progress': data.get('progress', 0)
            })
        jobs.sort(key=lambda x: x['job_id'], reverse=True)
    
    return render_template('history.html', jobs=jobs)

# ============= LIVE DETECTION ROUTES =============

@app.route('/live')
def live_detection():
    """Live detection page"""
    return render_template('live.html')

@app.route('/live/start', methods=['POST'])
def start_live_detection():
    """Start live camera detection"""
    global live_detection_active, live_camera
    
    with live_lock:
        if live_detection_active:
            return jsonify({'error': 'Live detection already running'}), 400
        
        # Try to open camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({'error': 'Could not access camera. Please check permissions.'}), 500
        
        live_camera = camera
        live_detection_active = True
    
    return jsonify({'message': 'Live detection started'})

@app.route('/live/stop', methods=['POST'])
def stop_live_detection():
    """Stop live camera detection"""
    global live_detection_active, live_camera
    
    with live_lock:
        live_detection_active = False
        if live_camera is not None:
            live_camera.release()
            live_camera = None
    
    return jsonify({'message': 'Live detection stopped'})

@app.route('/live/feed')
def live_feed():
    """Live video feed with detections"""
    return Response(generate_live_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_live_frames():
    """Generate frames for live video streaming with detections"""
    global live_detection_active, live_camera
    
    # Initialize detectors
    head_detector = HeadPoseDetector()
    object_detector = ObjectDetector()
    person_tracker = PersonTracker()
    behavior_analyzer = BehaviorAnalyzer()
    
    while True:
        with live_lock:
            if not live_detection_active or live_camera is None:
                break
            
            ret, frame = live_camera.read()
        
        if not ret:
            break
        
        # Update behavior analyzer frame counter
        behavior_analyzer.update_frame()
        
        # Detect head orientations
        head_orientations = head_detector.get_head_orientation(frame)
        
        # Detect objects
        detections = object_detector.detect_objects(frame)
        person_detections = object_detector.get_person_boxes(detections)
        phone_boxes = object_detector.get_phone_boxes(detections)
        
        # Track persons
        tracks = person_tracker.update(frame, person_detections)
        suspicious_ids = person_tracker.detect_suspicious_movement(tracks)
        interactions = person_tracker.detect_interactions(tracks)
        
        # Analyze behavior for each tracked student
        for track_id, bbox in tracks:
            # Check if this student has a phone nearby
            has_phone = any(
                _bbox_overlap(bbox, phone_bbox)
                for phone_bbox, _ in phone_boxes
            )
            
            # Check if in interaction
            in_interaction = any(
                int_data['student_1'] == track_id or int_data['student_2'] == track_id 
                for int_data in interactions
            )
            
            # Check head pose
            is_sideways = False
            sideways_frames = 0
            
            for pitch, yaw, roll, face_box, confidence, face_id in head_orientations:
                if _bbox_overlap(bbox, face_box):
                    is_sideways, sideways_frames = head_detector.is_looking_sideways(yaw, face_id)
                    break
            
            # Analyze behavior
            score = behavior_analyzer.analyze_behavior(
                track_id=track_id,
                is_sideways=is_sideways,
                sideways_frames=sideways_frames,
                has_phone=has_phone,
                is_moving_suspiciously=track_id in suspicious_ids,
                in_interaction=in_interaction,
                gaze_away=False
            )
            
            # Draw score on frame
            color = behavior_analyzer.get_severity_color(score)
            cv2.putText(frame, f"Risk: {score:.2f}", 
                       (bbox[0], bbox[3] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw detections
        head_detector.draw_face_orientation(frame, head_orientations)
        object_detector.draw_detections(frame, detections)
        person_tracker.draw_tracks(frame, tracks)
        
        # Draw interactions
        for interaction in interactions:
            cv2.line(frame, interaction['position_1'], interaction['position_2'], (255, 0, 255), 2)
        
        # Add live indicator
        cv2.putText(frame, "LIVE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def _bbox_overlap(bbox1, bbox2, threshold=0.3):
    """Check if two bounding boxes overlap"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    if bbox1_area == 0 or bbox2_area == 0:
        return False
    iou = intersection_area / min(bbox1_area, bbox2_area)
    return iou > threshold

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
