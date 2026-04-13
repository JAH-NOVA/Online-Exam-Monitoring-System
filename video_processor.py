"""
Video processor for handling uploaded classroom videos
"""
import cv2
import os
import json
from datetime import datetime
from typing import Dict, Callable
from detectors.head_pose import HeadPoseDetector
from detectors.object_detector import ObjectDetector
from detectors.tracker import PersonTracker
from detectors.behavior_analyzer import BehaviorAnalyzer
from alerts.alert_manager import AlertManager
from config import Config

class VideoProcessor:
    def __init__(self, video_path: str, output_dir: str = None):
        """
        Initialize video processor
        Args:
            video_path: Path to input video file
            output_dir: Directory for output files (defaults to processed/)
        """
        self.video_path = video_path
        self.output_dir = output_dir or Config.PROCESSED_FOLDER
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize detectors
        self.head_detector = HeadPoseDetector()
        self.object_detector = ObjectDetector()
        self.person_tracker = PersonTracker()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.alert_manager = AlertManager()
        
        # Processing state
        self.total_frames = 0
        self.processed_frames = 0
        self.output_path = None
        self.report_path = None
        
    def process_video(self, progress_callback: Callable[[int, int], None] = None) -> Dict:
        """
        Process entire video and generate annotated output
        Args:
            progress_callback: Optional callback function(current_frame, total_frames)
        Returns:
            Dictionary with processing results
        """
        # Open input video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup output video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.output_path = os.path.join(self.output_dir, f"{video_name}_annotated_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*Config.OUTPUT_VIDEO_CODEC)
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_number = 0
        all_alerts = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                self.processed_frames = frame_number
                
                # Process only every Nth frame for performance
                if frame_number % Config.PROCESS_EVERY_NTH_FRAME == 0:
                    processed_frame, frame_alerts = self._process_single_frame(frame, frame_number)
                    all_alerts.extend(frame_alerts)
                else:
                    processed_frame = frame
                
                # Write frame to output
                out.write(processed_frame)
                
                # Update progress
                if progress_callback:
                    progress_callback(frame_number, self.total_frames)
                
        finally:
            cap.release()
            out.release()
        
        # Generate report
        self.report_path = self._generate_report(all_alerts)
        
        # Get student summaries
        student_summaries = self.behavior_analyzer.get_all_summaries()
        
        return {
            'output_video': self.output_path,
            'report': self.report_path,
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'fps': fps,
            'total_alerts': len(all_alerts),
            'student_summaries': student_summaries,
            'duration_seconds': self.total_frames / fps if fps > 0 else 0
        }
    
    def _process_single_frame(self, frame, frame_number: int):
        """Process a single frame and return annotated frame with alerts"""
        alerts = []
        
        # Update behavior analyzer frame counter
        self.behavior_analyzer.update_frame()
        
        # Detect head orientations
        head_orientations = self.head_detector.get_head_orientation(frame)
        
        # Detect objects (persons and phones)
        detections = self.object_detector.detect_objects(frame)
        person_detections = self.object_detector.get_person_boxes(detections)
        phone_boxes = self.object_detector.get_phone_boxes(detections)
        
        # Track persons
        tracks = self.person_tracker.update(frame, person_detections)
        suspicious_movement_ids = self.person_tracker.detect_suspicious_movement(tracks)
        interactions = self.person_tracker.detect_interactions(tracks)
        
        # Detect object passing
        object_passing = self.object_detector.detect_object_passing(detections, person_detections)
        
        # Analyze behavior for each tracked student
        for track_id, bbox in tracks:
            # Check if this student has a phone nearby
            has_phone = self._check_phone_near_student(bbox, phone_boxes)
            
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
                is_moving_suspiciously=track_id in suspicious_movement_ids,
                in_interaction=in_interaction,
                gaze_away=False  # Will be enhanced with gaze detector
            )
            
            # Draw score on frame
            color = self.behavior_analyzer.get_severity_color(score)
            cv2.putText(frame, f"Risk: {score:.2f}", 
                       (bbox[0], bbox[3] + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw all detections
        self.head_detector.draw_face_orientation(frame, head_orientations)
        self.object_detector.draw_detections(frame, detections)
        self.person_tracker.draw_tracks(frame, tracks)
        
        # Draw interactions
        for interaction in interactions:
            cv2.line(frame, interaction['position_1'], interaction['position_2'], (255, 0, 255), 2)
            mid_point = (
                (interaction['position_1'][0] + interaction['position_2'][0]) // 2,
                (interaction['position_1'][1] + interaction['position_2'][1]) // 2
            )
            cv2.putText(frame, "INTERACTION", mid_point,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Check for alerts
        sideways_count = sum(1 for _, yaw, _, _, _, face_id in head_orientations 
                           if self.head_detector.is_looking_sideways(yaw, face_id)[0])
        
        if len(head_orientations) > 0 and sideways_count / len(head_orientations) > 0.3:
            alerts.append({
                'frame': frame_number,
                'type': 'sideways_looking',
                'details': f"{sideways_count}/{len(head_orientations)} students looking sideways"
            })
        
        if phone_boxes:
            alerts.append({
                'frame': frame_number,
                'type': 'phone_detected',
                'details': f"{len(phone_boxes)} phones detected"
            })
        
        if interactions:
            alerts.append({
                'frame': frame_number,
                'type': 'interaction',
                'details': f"{len(interactions)} student interactions"
            })
        
        # Add frame info overlay
        cv2.putText(frame, f"Frame: {frame_number}/{self.total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, alerts
    
    def _check_phone_near_student(self, student_bbox, phone_boxes) -> bool:
        """Check if any phone is near the student"""
        for phone_bbox, _ in phone_boxes:
            if self._bbox_overlap(student_bbox, phone_bbox, threshold=0.1):
                return True
        return False
    
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
        iou = intersection_area / min(bbox1_area, bbox2_area)
        return iou > threshold
    
    def _generate_report(self, alerts) -> str:
        """Generate JSON report of all detections"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        report_path = os.path.join(self.output_dir, f"{video_name}_report_{timestamp}.json")
        
        report = {
            'video_file': os.path.basename(self.video_path),
            'processed_at': datetime.now().isoformat(),
            'total_frames': self.total_frames,
            'total_alerts': len(alerts),
            'alerts': alerts,
            'student_summaries': self.behavior_analyzer.get_all_summaries()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_path
