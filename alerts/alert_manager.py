import cv2
import os
from datetime import datetime
import json
from typing import List, Dict, Any

class AlertManager:
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize alert manager
        Args:
            log_dir: Directory to store logs and snapshots
        """
        self.log_dir = log_dir
        self.snapshot_dir = os.path.join(log_dir, "snapshots")
        self.ensure_directories()
        self.alert_counts: Dict[str, int] = {
            "sideways_looking": 0,
            "phone_detected": 0,
            "suspicious_movement": 0
        }
        self.alert_thresholds = {
            "sideways_looking": 30,
            "phone_detected": 1,
            "suspicious_movement": 20
        }

    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def log_alert(self, alert_type: str, details: Dict[str, Any], frame=None):
        """
        Log an alert with optional snapshot
        Args:
            alert_type: Type of alert (sideways_looking, phone_detected, suspicious_movement)
            details: Dictionary containing alert details
            frame: Optional frame to save as snapshot
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save snapshot if frame is provided
        if frame is not None:
            snapshot_path = os.path.join(self.snapshot_dir, f"{alert_type}_{timestamp}.jpg")
            cv2.imwrite(snapshot_path, frame)
            details["snapshot"] = snapshot_path

        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "type": alert_type,
            "details": details
        }

        # Append to log file
        log_file = os.path.join(self.log_dir, "alerts.json")
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Error writing to log file: {e}")

    def check_sideways_looking(self, face_count: int, sideways_faces: int, frame=None) -> bool:
        """Check for suspicious sideways looking behavior"""
        if face_count > 0 and sideways_faces / face_count > 0.3:
            self.alert_counts["sideways_looking"] += 1
            if self.alert_counts["sideways_looking"] >= self.alert_thresholds["sideways_looking"]:
                self.log_alert("sideways_looking", {
                    "total_faces": face_count,
                    "sideways_faces": sideways_faces,
                    "ratio": sideways_faces / face_count
                }, frame)
                self.alert_counts["sideways_looking"] = 0
                return True
        else:
            self.alert_counts["sideways_looking"] = max(0, self.alert_counts["sideways_looking"] - 1)
        return False

    def check_phone_detection(self, phone_boxes: List[tuple], frame=None) -> bool:
        """Check for phone detection"""
        if phone_boxes:
            self.alert_counts["phone_detected"] += 1
            if self.alert_counts["phone_detected"] >= self.alert_thresholds["phone_detected"]:
                self.log_alert("phone_detected", {
                    "phone_count": len(phone_boxes),
                    "locations": phone_boxes
                }, frame)
                self.alert_counts["phone_detected"] = 0
                return True
        else:
            self.alert_counts["phone_detected"] = 0
        return False

    def check_suspicious_movement(self, suspicious_ids: List[int], frame=None) -> bool:
        """Check for suspicious movement"""
        if suspicious_ids:
            self.alert_counts["suspicious_movement"] += 1
            if self.alert_counts["suspicious_movement"] >= self.alert_thresholds["suspicious_movement"]:
                self.log_alert("suspicious_movement", {
                    "track_ids": suspicious_ids
                }, frame)
                self.alert_counts["suspicious_movement"] = 0
                return True
        else:
            self.alert_counts["suspicious_movement"] = max(0, self.alert_counts["suspicious_movement"] - 1)
        return False 