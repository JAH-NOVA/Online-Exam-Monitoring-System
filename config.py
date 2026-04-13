"""
Configuration settings for the cheat detection system
"""
import os

class Config:
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    UPLOAD_FOLDER = 'uploads'
    PROCESSED_FOLDER = 'processed'
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
    
    # Detection thresholds
    HEAD_POSE_YAW_THRESHOLD = 35.0  # Degrees for sideways looking
    HEAD_POSE_PITCH_DOWN_THRESHOLD = 25.0  # Degrees for looking down
    SUSTAINED_LOOK_FRAMES = 15  # Consecutive frames needed for sustained detection
    
    # Object detection
    OBJECT_CONFIDENCE_THRESHOLD = 0.6  # Increased for fewer false positives
    PHONE_CONFIDENCE_THRESHOLD = 0.7  # Higher threshold for phones
    
    # Movement tracking
    MOVEMENT_THRESHOLD = 150.0  # Pixels of movement to be suspicious
    ZONE_EXIT_THRESHOLD = 100.0  # Pixels from original position
    INTERACTION_DISTANCE = 200.0  # Distance between students to flag interaction
    
    # Alert thresholds
    SIDEWAYS_LOOKING_THRESHOLD = 45  # Frames before alert
    PHONE_DETECTION_THRESHOLD = 5  # Frames before alert
    SUSPICIOUS_MOVEMENT_THRESHOLD = 30  # Frames before alert
    GAZE_AWAY_THRESHOLD = 60  # Frames looking at neighbor
    
    # Behavior scoring weights
    SCORE_WEIGHTS = {
        'sideways_looking': 0.2,
        'phone_detected': 0.3,
        'suspicious_movement': 0.15,
        'gaze_away': 0.25,
        'interaction': 0.1
    }
    
    # Alert severity levels
    SEVERITY_LOW = 0.3  # Below this is normal
    SEVERITY_MEDIUM = 0.6  # Between low and high
    SEVERITY_HIGH = 0.8  # Above this is critical
    
    # Video processing
    PROCESS_EVERY_NTH_FRAME = 1  # Process every frame (set higher for faster processing)
    OUTPUT_VIDEO_FPS = 30
    OUTPUT_VIDEO_CODEC = 'mp4v'
    
    # Logging
    LOG_DIR = 'logs'
    SNAPSHOT_DIR = os.path.join(LOG_DIR, 'snapshots')
