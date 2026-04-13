from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict
import os
from config import Config

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = None):
        """
        Initialize object detector with YOLOv8
        Args:
            model_name: Name or path of the YOLOv8 model
            conf_threshold: Confidence threshold for detections
        """
        try:
            # Try to load the model directly
            self.model = YOLO('yolov8n.pt')  # This will download if not present
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.conf_threshold = conf_threshold or Config.OBJECT_CONFIDENCE_THRESHOLD
        # COCO classes we're interested in
        self.target_classes = {
            0: 'person',
            67: 'cell phone',
            73: 'book',
            84: 'book',  # Sometimes books are classified differently
            64: 'mouse',  # Sometimes phones are misclassified as mouse
        }
        
        # Specific thresholds for different objects
        self.class_thresholds = {
            67: Config.PHONE_CONFIDENCE_THRESHOLD,  # Higher threshold for phones
            73: 0.5,  # Books
            84: 0.5,  # Books
        }

    def detect_objects(self, frame) -> List[Tuple[str, tuple, float]]:
        """
        Detect objects in frame with class-specific confidence thresholds
        Returns list of (class_name, bbox, confidence)
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = r
            class_id = int(class_id)
            
            # Get class-specific threshold
            threshold = self.class_thresholds.get(class_id, self.conf_threshold)
            
            if class_id in self.target_classes and conf > threshold:
                # Normalize class names
                class_name = self.target_classes[class_id]
                if class_id == 64 and conf < 0.7:  # Mouse might be phone
                    class_name = 'cell phone'
                
                detections.append((
                    class_name,
                    (int(x1), int(y1), int(x2), int(y2)),
                    conf
                ))

        return detections

    def draw_detections(self, frame, detections: List[Tuple[str, tuple, float]]):
        """Draw bounding boxes and labels for detected objects with improved styling"""
        for class_name, bbox, conf in detections:
            # Color coding by object type
            if class_name == 'cell phone':
                color = (0, 0, 255)  # Red
            elif class_name == 'book':
                color = (255, 140, 0)  # Orange
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box with thicker line for high confidence
            thickness = 3 if conf > 0.8 else 2
            cv2.rectangle(frame, 
                        (bbox[0], bbox[1]), 
                        (bbox[2], bbox[3]), 
                        color, thickness)
            
            # Draw filled background for label
            label = f"{class_name}: {conf:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(frame,
                         (bbox[0], bbox[1] - label_height - 10),
                         (bbox[0] + label_width, bbox[1]),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame, label, 
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 2)

        return frame

    def get_person_boxes(self, detections: List[Tuple[str, tuple, float]]) -> List[Tuple[tuple, float]]:
        """Extract bounding boxes and confidence scores for persons only"""
        return [(bbox, conf) for class_name, bbox, conf in detections if class_name == 'person']

    def get_phone_boxes(self, detections: List[Tuple[str, tuple, float]]) -> List[Tuple[tuple, float]]:
        """Extract bounding boxes and confidence for phones only"""
        return [(bbox, conf) for class_name, bbox, conf in detections if class_name == 'cell phone']
    
    def get_book_boxes(self, detections: List[Tuple[str, tuple, float]]) -> List[Tuple[tuple, float]]:
        """Extract bounding boxes and confidence for books only"""
        return [(bbox, conf) for class_name, bbox, conf in detections if class_name == 'book']
    
    def detect_object_passing(self, detections: List[Tuple[str, tuple, float]], 
                             person_boxes: List[Tuple[tuple, float]]) -> List[Dict]:
        """
        Detect potential object passing between students
        Returns list of interactions with positions
        """
        interactions = []
        phones = self.get_phone_boxes(detections)
        books = self.get_book_boxes(detections)
        
        all_objects = phones + books
        
        for obj_bbox, obj_conf in all_objects:
            obj_center = self._get_bbox_center(obj_bbox)
            
            # Check which students are near this object
            nearby_students = []
            for person_bbox, person_conf in person_boxes:
                person_center = self._get_bbox_center(person_bbox)
                distance = self._calculate_distance(obj_center, person_center)
                
                if distance < Config.INTERACTION_DISTANCE:
                    nearby_students.append({
                        'bbox': person_bbox,
                        'distance': distance
                    })
            
            # If object is between 2+ students, flag as potential passing
            if len(nearby_students) >= 2:
                interactions.append({
                    'object_bbox': obj_bbox,
                    'object_confidence': obj_conf,
                    'students': nearby_students,
                    'type': 'potential_passing'
                })
        
        return interactions
    
    def _get_bbox_center(self, bbox: tuple) -> Tuple[int, int]:
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) 