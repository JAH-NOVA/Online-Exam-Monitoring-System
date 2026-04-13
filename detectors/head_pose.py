import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Dict
from collections import defaultdict
from config import Config

class HeadPoseDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
            max_num_faces=20  # Support multiple students
        )
        
        # Track sustained behavior per face position
        self.sideways_count = defaultdict(int)
        self.downward_count = defaultdict(int)
        self.normal_yaw_baseline = defaultdict(list)
        
    def get_head_orientation(self, frame) -> List[Tuple[float, float, float, tuple, float]]:
        """
        Detect head orientation for all faces in the frame
        Returns list of (pitch, yaw, roll, face_coordinates, confidence)
        """
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        orientations = []

        if results.multi_face_landmarks:
            for face_idx, face_landmarks in enumerate(results.multi_face_landmarks):
                # Get face mesh points
                face_3d = []
                face_2d = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:
                        x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])
                
                if len(face_2d) < 6:
                    continue
                    
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # Camera matrix estimation
                focal_length = frame.shape[1]
                center = (frame.shape[1]/2, frame.shape[0]/2)
                cam_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ])

                dist_matrix = np.zeros((4, 1))
                success, rot_vec, trans_vec = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix
                )

                if not success:
                    continue

                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Normalize angles to 360 degrees
                pitch = angles[0] * 360
                yaw = angles[1] * 360  
                roll = angles[2] * 360

                # Get face bounding box
                x_min = min(pt[0] for pt in face_2d)
                x_max = max(pt[0] for pt in face_2d)
                y_min = min(pt[1] for pt in face_2d)
                y_max = max(pt[1] for pt in face_2d)
                face_box = (int(x_min), int(y_min), int(x_max), int(y_max))
                
                # Create face identifier based on position
                face_id = f"{int(x_min/50)}_{int(y_min/50)}"
                
                # Calculate confidence score based on detection quality
                face_area = (x_max - x_min) * (y_max - y_min)
                confidence = min(1.0, face_area / 10000.0)  # Normalize by expected face size
                
                # Update baseline for this face position
                if len(self.normal_yaw_baseline[face_id]) < 30:
                    self.normal_yaw_baseline[face_id].append(yaw)

                orientations.append((pitch, yaw, roll, face_box, confidence, face_id))

        return orientations

    def is_looking_sideways(self, yaw: float, face_id: str = None, 
                           threshold: float = None) -> Tuple[bool, int]:
        """
        Check if the head is turned sideways beyond the threshold angle
        Returns (is_sideways, sustained_frames)
        """
        if threshold is None:
            threshold = Config.HEAD_POSE_YAW_THRESHOLD
            
        # Use dynamic baseline if available
        dynamic_threshold = threshold
        if face_id and len(self.normal_yaw_baseline[face_id]) >= 10:
            baseline = np.mean(self.normal_yaw_baseline[face_id])
            dynamic_threshold = threshold + abs(baseline)
        
        is_sideways = abs(yaw) > dynamic_threshold
        
        # Track sustained sideways looking
        if face_id:
            if is_sideways:
                self.sideways_count[face_id] += 1
            else:
                self.sideways_count[face_id] = max(0, self.sideways_count[face_id] - 1)
                
            return is_sideways, self.sideways_count[face_id]
        
        return is_sideways, 0
    
    def is_looking_down(self, pitch: float, threshold: float = None) -> Tuple[bool, int]:
        """
        Check if the head is tilted down (looking at test paper on desk)
        Returns (is_looking_down, sustained_frames)
        """
        if threshold is None:
            threshold = Config.HEAD_POSE_PITCH_DOWN_THRESHOLD
            
        is_down = pitch > threshold
        return is_down, 0

    def draw_face_orientation(self, frame, orientations: List[Tuple[float, float, float, tuple, float, str]]):
        """Draw head pose indicators and bounding boxes with improved visualization"""
        for pitch, yaw, roll, face_box, confidence, face_id in orientations:
            is_sideways, sideways_frames = self.is_looking_sideways(yaw, face_id)
            is_down, _ = self.is_looking_down(pitch)
            
            # Color coding: Green = normal, Yellow = looking down, Red = sideways
            color = (0, 255, 0)  # Green
            status = "Normal"
            
            if is_down:
                color = (0, 255, 255)  # Yellow
                status = "Down"
            
            if is_sideways:
                color = (0, 0, 255)  # Red
                status = f"Sideways ({sideways_frames}f)"
            
            # Draw bounding box
            cv2.rectangle(frame, 
                        (face_box[0], face_box[1]), 
                        (face_box[2], face_box[3]), 
                        color, 2)
            
            # Draw status and angles
            cv2.putText(frame, status, 
                       (face_box[0], face_box[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.putText(frame, f"Y:{int(yaw)} P:{int(pitch)}", 
                       (face_box[0], face_box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return frame
    
    def reset_tracking(self):
        """Reset all tracking dictionaries"""
        self.sideways_count.clear()
        self.downward_count.clear()
        self.normal_yaw_baseline.clear() 