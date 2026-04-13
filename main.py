import cv2
import os
import threading
from detectors.head_pose import HeadPoseDetector
from detectors.object_detector import ObjectDetector
from detectors.tracker import PersonTracker
from detectors.behavior_analyzer import BehaviorAnalyzer
from alerts.alert_manager import AlertManager
from dashboard.dashboard import app, update_frame, add_alert
from config import Config

class CheatDetectionSystem:
    def __init__(self, camera_source=0):
        """
        Initialize the cheat detection system
        Args:
            camera_source: Camera index or video file path
        """
        self.cap = cv2.VideoCapture(camera_source)
        self.head_detector = HeadPoseDetector()
        self.object_detector = ObjectDetector()
        self.person_tracker = PersonTracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.alert_manager = AlertManager()

    def process_frame(self, frame):
        """Process a single frame and detect suspicious activities"""
        # Update behavior analyzer frame counter
        self.behavior_analyzer.update_frame()
        
        # Detect head orientations
        head_orientations = self.head_detector.get_head_orientation(frame)
        
        # Count sideways looking students
        sideways_count = sum(1 for _, yaw, _, _, _, face_id in head_orientations 
                           if self.head_detector.is_looking_sideways(yaw, face_id)[0])
        
        # Detect objects (persons and phones)
        detections = self.object_detector.detect_objects(frame)
        person_detections = self.object_detector.get_person_boxes(detections)
        phone_boxes = self.object_detector.get_phone_boxes(detections)
        
        # Track persons
        tracks = self.person_tracker.update(frame, person_detections)
        suspicious_ids = self.person_tracker.detect_suspicious_movement(tracks)
        interactions = self.person_tracker.detect_interactions(tracks)
        
        # Detect object passing
        object_passing = self.object_detector.detect_object_passing(detections, person_detections)

        # Analyze behavior for each tracked student
        for track_id, bbox in tracks:
            # Check if this student has a phone nearby
            has_phone = any(
                self._bbox_overlap(bbox, phone_bbox)
                for phone_bbox, _ in phone_boxes
            )
            
            # Check if in interaction
            in_interaction = any(
                int_data['student_1'] == track_id or int_data['student_2'] == track_id 
                for int_data in interactions
            )
            
            # Check head pose for this student
            is_sideways = False
            sideways_frames = 0
            
            # Find matching face for this tracked person
            for pitch, yaw, roll, face_box, confidence, face_id in head_orientations:
                if self._bbox_overlap(bbox, face_box):
                    is_sideways, sideways_frames = self.head_detector.is_looking_sideways(yaw, face_id)
                    break
            
            # Analyze behavior
            score = self.behavior_analyzer.analyze_behavior(
                track_id=track_id,
                is_sideways=is_sideways,
                sideways_frames=sideways_frames,
                has_phone=has_phone,
                is_moving_suspiciously=track_id in suspicious_ids,
                in_interaction=in_interaction,
                gaze_away=False
            )
            
            # Draw score on frame
            color = self.behavior_analyzer.get_severity_color(score)
            cv2.putText(frame, f"Risk: {score:.2f}", 
                       (bbox[0], bbox[3] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Check for alerts
        if self.alert_manager.check_sideways_looking(len(head_orientations), sideways_count, frame):
            add_alert("sideways_looking", 
                     f"Detected {sideways_count}/{len(head_orientations)} students looking sideways")

        if self.alert_manager.check_phone_detection(phone_boxes, frame):
            add_alert("phone_detected", 
                     f"Detected {len(phone_boxes)} phones in the classroom")

        if self.alert_manager.check_suspicious_movement(suspicious_ids, frame):
            add_alert("suspicious_movement", 
                     f"Suspicious movement detected for students: {suspicious_ids}")

        # Draw detections on frame
        self.head_detector.draw_face_orientation(frame, head_orientations)
        self.object_detector.draw_detections(frame, detections)
        self.person_tracker.draw_tracks(frame, tracks)
        
        # Draw interactions
        for interaction in interactions:
            cv2.line(frame, interaction['position_1'], interaction['position_2'], (255, 0, 255), 2)

        return frame
    
    def _bbox_overlap(self, bbox1, bbox2, threshold=0.3) -> bool:
        """Check if two bounding boxes overlap"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return False
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        
        # Check if intersection is significant
        if bbox1_area == 0 or bbox2_area == 0:
            return False
        iou = intersection_area / min(bbox1_area, bbox2_area)
        return iou > threshold

    def run(self):
        """Main loop for processing video feed"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Update dashboard
            update_frame(processed_frame)

            # Display locally (optional)
            cv2.imshow('Cheat Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def run_detection():
    """Run the cheat detection system"""
    system = CheatDetectionSystem()
    system.run()

def run_dashboard():
    """Run the Flask dashboard"""
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    # Start dashboard in a separate thread
    dashboard_thread = threading.Thread(target=run_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()

    # Run detection system in main thread
    run_detection() 