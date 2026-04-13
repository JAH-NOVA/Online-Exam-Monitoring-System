# Classroom Cheat Detection System

An AI-powered system for detecting potential cheating behaviors in classroom exam environments using advanced computer vision and machine learning.

## 🎯 Features

### Enhanced Detection Algorithms
- **Head Pose Detection**: Track students looking sideways or down with dynamic thresholds and sustained detection
- **Object Detection**: Identify phones, books, and prohibited items with high accuracy
- **Movement Tracking**: Zone-based tracking with velocity and acceleration analysis
- **Interaction Detection**: Identify students who are too close together
- **Behavior Scoring**: Comprehensive cheating probability scores combining all detection signals

### Modern Web Interface
- **Video Upload**: Drag-and-drop interface supporting MP4, AVI, MOV, MKV formats (up to 500MB)
- **Real-time Processing**: Background video processing with live progress tracking
- **Annotated Output**: Download videos with visual annotations of all detections
- **Detailed Reports**: Student-wise analysis with timeline of incidents and severity levels
- **Interactive Timeline**: Click on incidents to jump to specific frames in the video

## 📋 Requirements

- Python 3.8+
- Webcam (for live monitoring) or video files
- CUDA-capable GPU (recommended for faster processing)

## 🚀 Installation

1. **Clone or navigate to the repository**:
   ```bash
   cd /Users/codeaj/Desktop/classroom-cheat-detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify YOLOv8 model** (will download automatically on first run):
   The `yolov8n.pt` model file should be in the project root

## 💻 Usage

### Option 1: Web Application (Recommended)

1. **Start the Flask web server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:8080
   ```

3. **Choose your detection method**:

   **A. Upload Pre-recorded Video**:
   - Click "Upload Video" in the navigation
   - Drag and drop your video or click to browse
   - Wait for processing to complete
   - View results with annotated video and detailed analysis

   **B. Live Camera Detection**:
   - Click "Live Detection" in the navigation
   - Click "Start Live Detection" to begin
   - Grant camera permissions when prompted
   - View real-time detections with risk scores
   - Click "Stop Detection" when finished

4. **Review results** (for uploaded videos):
   - Watch the annotated video with all detections highlighted
   - Review student-wise behavioral scores
   - View timeline of suspicious incidents
   - Download the processed video and JSON report

### Option 2: Command-Line Live Detection

1. **Run the main detection system** (standalone):
   ```bash
   python main.py
   ```

2. **Access the dashboard**:
   - Open browser to `http://localhost:5000`
   - View live feed with real-time detections
   - Monitor alerts as they occur

3. **Exit**: Press 'q' in the video window

## 📊 Detection Capabilities

### 1. Head Pose Analysis
- Detects sideways looking (potential copying)
- Identifies downward looking (looking at notes/phone)
- Tracks sustained behaviors over time
- Dynamic threshold adjustment based on classroom setup

### 2. Object Detection
- **Phones**: High-confidence detection with strict thresholds
- **Books/Papers**: Identify unauthorized materials
- **Object Passing**: Detect items being passed between students

### 3. Movement Tracking
- **Zone Violations**: Alert when students leave their designated area
- **Excessive Movement**: Track unusual movement patterns
- **Student Interactions**: Identify students sitting too close or interacting

### 4. Behavior Scoring
Each student receives a real-time risk score (0.0 - 1.0) based on:
- Sideways/downward looking frequency
- Phone detection
- Movement patterns
- Proximity to other students

**Severity Levels**:
- 🟢 **Normal** (< 0.3): No suspicious behavior
- 🟡 **Low** (0.3 - 0.6): Minor suspicious behaviors
- 🟠 **Medium** (0.6 - 0.8): Moderate concern
- 🔴 **High** (> 0.8): Strong indicators of cheating

## ⚙️ Configuration

Edit `config.py` to customize detection thresholds:

```python
# Head pose thresholds
HEAD_POSE_YAW_THRESHOLD = 35.0  # Degrees for sideways detection
HEAD_POSE_PITCH_DOWN_THRESHOLD = 25.0  # Degrees for looking down

# Alert thresholds  
SIDEWAYS_LOOKING_THRESHOLD = 45  # Frames before alert
PHONE_DETECTION_THRESHOLD = 5  # Frames before alert

# Movement thresholds
MOVEMENT_THRESHOLD = 150.0  # Pixels for suspicious movement
ZONE_EXIT_THRESHOLD = 100.0  # Pixels from original position
```

## 📁 Project Structure

```
classroom-cheat-detection/
├── app.py                    # Flask web application
├── main.py                   # Live camera detection
├── config.py                 # Configuration settings
├── video_processor.py        # Video processing engine
├── requirements.txt          # Python dependencies
├── detectors/
│   ├── head_pose.py         # Head orientation detection
│   ├── object_detector.py   # YOLOv8 object detection
│   ├── tracker.py           # Person tracking (DeepSORT)
│   └── behavior_analyzer.py # Behavior scoring system
├── alerts/
│   └── alert_manager.py     # Alert generation and logging
├── templates/               # HTML templates
│   ├── index.html          # Home page
│   ├── upload.html         # Upload page
│   ├── results.html        # Results page
│   └── history.html        # Processing history
├── static/
│   ├── css/style.css       # Styling
│   └── js/app.js           # JavaScript functionality
├── uploads/                # Uploaded videos (created automatically)
├── processed/              # Processed videos and reports
└── logs/                   # Alert logs and snapshots
```

## 🎥 Video Format Support

Supported formats:
- MP4 (recommended)
- AVI
- MOV
- MKV
- FLV
- WMV

Max file size: 500MB

## 📈 Output Files

After processing, you'll get:

1. **Annotated Video** (`*_annotated_*.mp4`):
   - Visual bounding boxes around faces and objects
   - Color-coded status indicators
   - Real-time risk scores
   - Interaction lines between students

2. **JSON Report** (`*_report_*.json`):
   - Total frames and duration
   - Alert timeline with frame numbers
   - Student-wise behavioral summaries
   - Severity classifications

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'torch'"**:
   - Install PyTorch: `pip install torch torchvision`

2. **Low detection accuracy**:
   - Ensure good lighting in videos
   - Check camera angle captures faces clearly
   - Adjust thresholds in `config.py`

3. **Slow processing**:
   - Reduce `PROCESS_EVERY_NTH_FRAME` in config
   - Use GPU acceleration
   - Process shorter video segments

4. **Port already in use**:
   - Change port in `app.py`: `app.run(port=5001)`

## 📝 Tips for Best Results

1. **Camera Setup**:
   - Wide-angle view capturing multiple students
   - Clear view of faces and upper bodies
   - Good, even lighting

2. **Video Quality**:
   - Minimum 720p resolution
   - 30 FPS or higher
   - Stable camera position

3. **Classroom Setup**:
   - Students seated at consistent intervals
   - Clear sight lines
   - Minimal background movement

## 🔒 Privacy Considerations

- This system is for educational/exam monitoring purposes only
- Inform students when monitoring is in use
- Secure storage of processed videos and reports
- Review applicable privacy laws and regulations

## 📜 License

This project is for educational and monitoring purposes in controlled classroom environments.

## 🤝 Contributing

Contributions to improve detection accuracy or add new features are welcome!

## ⚡ Performance Notes

- **Live Detection**: ~20-30 FPS on modern CPU, ~60+ FPS with GPU
- **Video Processing**: Depends on video length and resolution
  - 10-minute 1080p video: ~5-10 minutes processing time (CPU)
  - 10-minute 1080p video: ~2-3 minutes processing time (GPU)

## 🆘 Support

For issues or questions:
1. Check the configuration in `config.py`
2. Review logs in `logs/alerts.json`
3. Verify all dependencies are installed correctly