"""
Behavior Analyzer - Aggregates all detection signals into cheating probability scores
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
from config import Config

class BehaviorAnalyzer:
    def __init__(self):
        """Initialize behavior analyzer with tracking for individual students"""
        self.student_scores: Dict[int, List[float]] = defaultdict(list)
        self.student_events: Dict[int, List[Dict]] = defaultdict(list)
        self.frame_count = 0
        
    def update_frame(self):
        """Increment frame counter"""
        self.frame_count += 1
    
    def analyze_behavior(self, 
                        track_id: int,
                        is_sideways: bool,
                        sideways_frames: int,
                        has_phone: bool,
                        is_moving_suspiciously: bool,
                        in_interaction: bool,
                        gaze_away: bool = False) -> float:
        """
        Analyze behavior for a single student and return cheating probability score
        
        Args:
            track_id: Student tracking ID
            is_sideways: Currently looking sideways
            sideways_frames: Number of consecutive frames looking sideways
            has_phone: Phone detected near student
            is_moving_suspiciously: Excessive or erratic movement
            in_interaction: Too close to another student
            gaze_away: Eyes directed away from own paper
            
        Returns:
            Cheating probability score (0.0 - 1.0)
        """
        score = 0.0
        weights = Config.SCORE_WEIGHTS
        
        # Sideways looking (sustained is worse)
        if is_sideways:
            sustained_factor = min(1.0, sideways_frames / Config.SUSTAINED_LOOK_FRAMES)
            score += weights['sideways_looking'] * sustained_factor
        
        # Phone detection (very suspicious)
        if has_phone:
            score += weights['phone_detected']
        
        # Suspicious movement
        if is_moving_suspiciously:
            score += weights['suspicious_movement']
        
        # Gaze away from own paper
        if gaze_away:
            score += weights['gaze_away']
        
        # Interaction with other students
        if in_interaction:
            score += weights['interaction']
        
        # Clamp score to [0, 1]
        score = min(1.0, max(0.0, score))
        
        # Store score
        self.student_scores[track_id].append(score)
        if len(self.student_scores[track_id]) > 100:
            self.student_scores[track_id].pop(0)
        
        # Log event if score is significant
        if score > Config.SEVERITY_LOW:
            self.student_events[track_id].append({
                'frame': self.frame_count,
                'score': score,
                'sideways': is_sideways,
                'phone': has_phone,
                'movement': is_moving_suspiciously,
                'interaction': in_interaction,
                'gaze_away': gaze_away,
                'timestamp': datetime.now().isoformat()
            })
        
        return score
    
    def get_student_summary(self, track_id: int) -> Dict:
        """Get comprehensive summary for a student"""
        if track_id not in self.student_scores or not self.student_scores[track_id]:
            return {
                'track_id': track_id,
                'avg_score': 0.0,
                'max_score': 0.0,
                'severity': 'normal',
                'event_count': 0,
                'events': []
            }
        
        scores = self.student_scores[track_id]
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        
        # Determine severity
        severity = 'normal'
        if avg_score >= Config.SEVERITY_HIGH:
            severity = 'high'
        elif avg_score >= Config.SEVERITY_MEDIUM:
            severity = 'medium'
        elif avg_score >= Config.SEVERITY_LOW:
            severity = 'low'
        
        return {
            'track_id': track_id,
            'avg_score': float(avg_score),
            'max_score': float(max_score),
            'severity': severity,
            'event_count': len(self.student_events[track_id]),
            'events': self.student_events[track_id][-10:]  # Last 10 events
        }
    
    def get_all_summaries(self) -> List[Dict]:
        """Get summaries for all tracked students"""
        summaries = []
        all_track_ids = set(self.student_scores.keys()) | set(self.student_events.keys())
        
        for track_id in all_track_ids:
            summaries.append(self.get_student_summary(track_id))
        
        # Sort by average score (highest first)
        summaries.sort(key=lambda x: x['avg_score'], reverse=True)
        return summaries
    
    def get_severity_color(self, score: float) -> Tuple[int, int, int]:
        """Get BGR color for score visualization"""
        if score >= Config.SEVERITY_HIGH:
            return (0, 0, 255)  # Red
        elif score >= Config.SEVERITY_MEDIUM:
            return (0, 165, 255)  # Orange
        elif score >= Config.SEVERITY_LOW:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 255, 0)  # Green
    
    def reset(self):
        """Reset all tracking data"""
        self.student_scores.clear()
        self.student_events.clear()
        self.frame_count = 0
