from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import cv2
from typing import List, Tuple, Dict
from config import Config

class PersonTracker:
    def __init__(self, max_age: int = 30):
        """
        Initialize DeepSORT tracker
        Args:
            max_age: Maximum number of frames to keep track of lost objects
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",  # Use MobileNet as embedder
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        self.tracks_history: Dict[int, List[tuple]] = {}
        self.initial_positions: Dict[int, tuple] = {}  # Track starting positions
        self.velocities: Dict[int, List[float]] = {}  # Track velocities over time

    def update(self, frame, person_detections: List[Tuple[tuple, float]]) -> List[Tuple[int, tuple]]:
        """
        Update tracker with new detections
        Args:
            frame: Current frame
            person_detections: List of (bbox, confidence) tuples
        Returns:
            List of (track_id, bbox)
        """
        if not person_detections:
            return []

        # Convert to format expected by DeepSORT
        detections = []
        for bbox, confidence in person_detections:
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], confidence, None))

        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox = tuple(map(int, ltrb))
            
            # Store initial position
            if track_id not in self.initial_positions:
                self.initial_positions[track_id] = self._get_bbox_center(bbox)

            # Update track history
            if track_id not in self.tracks_history:
                self.tracks_history[track_id] = []
            self.tracks_history[track_id].append(bbox)

            # Keep only last 30 positions
            if len(self.tracks_history[track_id]) > 30:
                self.tracks_history[track_id].pop(0)
            
            # Calculate and store velocity
            if track_id not in self.velocities:
                self.velocities[track_id] = []
            
            if len(self.tracks_history[track_id]) >= 2:
                velocity = self._calculate_velocity(
                    self.tracks_history[track_id][-2],
                    self.tracks_history[track_id][-1]
                )
                self.velocities[track_id].append(velocity)
                if len(self.velocities[track_id]) > 30:
                    self.velocities[track_id].pop(0)

            results.append((track_id, bbox))

        return results

    def draw_tracks(self, frame, tracks: List[Tuple[int, tuple]]):
        """Draw bounding boxes, IDs, and movement trails for tracked persons"""
        for track_id, bbox in tracks:
            # Check if this person has left their zone
            left_zone = self._has_left_zone(track_id, bbox)
            
            # Color: Green = in zone, Yellow = suspicious movement, Red = left zone
            color = (0, 255, 0)
            if left_zone:
                color = (0, 0, 255)
            elif track_id in self.velocities and len(self.velocities[track_id]) > 0:
                avg_velocity = np.mean(self.velocities[track_id])
                if avg_velocity > Config.MOVEMENT_THRESHOLD / 10:
                    color = (0, 255, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, 
                        (bbox[0], bbox[1]), 
                        (bbox[2], bbox[3]), 
                        color, 2)
            
            # Draw ID and status
            status = "LEFT ZONE!" if left_zone else f"ID: {track_id}"
            cv2.putText(frame, status, 
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)

            # Draw movement trail
            if track_id in self.tracks_history and len(self.tracks_history[track_id]) > 1:
                points = [self._get_bbox_center(box) for box in self.tracks_history[track_id]]
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i+1], (0, 255, 255), 2)

        return frame

    def detect_suspicious_movement(self, tracks: List[Tuple[int, tuple]], 
                                  movement_threshold: float = None) -> List[int]:
        """
        Detect suspicious movements based on multiple criteria
        Returns list of track IDs with suspicious movement
        """
        if movement_threshold is None:
            movement_threshold = Config.MOVEMENT_THRESHOLD
            
        suspicious_ids = []
        
        for track_id, bbox in tracks:
            # Check multiple suspicious movement criteria
            reasons = []
            
            # 1. Total movement over time
            if track_id in self.tracks_history and len(self.tracks_history[track_id]) > 5:
                history = self.tracks_history[track_id]
                total_movement = self._calculate_total_movement(history)
                
                if total_movement > movement_threshold:
                    reasons.append("excessive_movement")
            
            # 2. Left assigned zone
            if self._has_left_zone(track_id, bbox):
                reasons.append("left_zone")
            
            # 3. Erratic velocity changes (sudden acceleration)
            if track_id in self.velocities and len(self.velocities[track_id]) > 5:
                velocities = self.velocities[track_id]
                velocity_variance = np.var(velocities)
                if velocity_variance > 20:  # High variance indicates erratic movement
                    reasons.append("erratic_movement")
            
            if reasons:
                suspicious_ids.append(track_id)

        return suspicious_ids
    
    def detect_interactions(self, tracks: List[Tuple[int, tuple]]) -> List[Dict]:
        """
        Detect interactions between students (too close together)
        Returns list of interaction pairs
        """
        interactions = []
        
        for i, (id1, bbox1) in enumerate(tracks):
            center1 = self._get_bbox_center(bbox1)
            
            for id2, bbox2 in tracks[i+1:]:
                center2 = self._get_bbox_center(bbox2)
                distance = self._calculate_distance(center1, center2)
                
                if distance < Config.INTERACTION_DISTANCE:
                    interactions.append({
                        'student_1': id1,
                        'student_2': id2,
                        'distance': distance,
                        'position_1': center1,
                        'position_2': center2
                    })
        
        return interactions
    
    def _has_left_zone(self, track_id: int, current_bbox: tuple) -> bool:
        """Check if person has left their initial zone"""
        if track_id not in self.initial_positions:
            return False
        
        initial_pos = self.initial_positions[track_id]
        current_pos = self._get_bbox_center(current_bbox)
        distance = self._calculate_distance(initial_pos, current_pos)
        
        return distance > Config.ZONE_EXIT_THRESHOLD
    
    def _get_bbox_center(self, bbox: tuple) -> Tuple[int, int]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _calculate_velocity(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calculate velocity between two consecutive bounding boxes"""
        center1 = self._get_bbox_center(bbox1)
        center2 = self._get_bbox_center(bbox2)
        return self._calculate_distance(center1, center2)
    
    def _calculate_total_movement(self, history: List[tuple]) -> float:
        """Calculate total movement from bbox history"""
        total = 0
        for i in range(len(history) - 1):
            center1 = self._get_bbox_center(history[i])
            center2 = self._get_bbox_center(history[i+1])
            total += self._calculate_distance(center1, center2)
        return total
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.tracks_history.clear()
        self.initial_positions.clear()
        self.velocities.clear() 