"""
Person Tracker - Phase 1 Step 3 Implementation
Handles ID assignment, frame-to-frame association, and trajectory smoothing
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

@dataclass
class TrackedPerson:
    """Represents a tracked person with history and state."""
    id: int
    current_detection: Optional['PersonDetection'] = None
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=10))
    zone_history: deque = field(default_factory=lambda: deque(maxlen=5))
    
    # Tracking state
    frames_since_seen: int = 0
    total_frames_seen: int = 0
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    
    # Kalman filter state for position prediction
    kalman_filter: Optional[cv2.KalmanFilter] = None
    predicted_position: Optional[Tuple[float, float, float]] = None
    
    # Trajectory smoothing
    smoothed_position: Optional[Tuple[float, float, float]] = None
    velocity: Optional[Tuple[float, float, float]] = None
    
    def __post_init__(self):
        """Initialize Kalman filter after object creation."""
        self._init_kalman_filter()
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter for position prediction."""
        # 6D state: [x, y, z, vx, vy, vz]
        self.kalman_filter = cv2.KalmanFilter(6, 3)
        
        # Transition matrix (constant velocity model)
        dt = 1.0  # Assume 1 frame time step
        self.kalman_filter.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position)
        self.kalman_filter.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        self.kalman_filter.processNoiseCov = np.eye(6, dtype=np.float32) * 0.1
        
        # Measurement noise
        self.kalman_filter.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.5
        
        # Error covariance
        self.kalman_filter.errorCovPost = np.eye(6, dtype=np.float32)
    
    def update(self, detection: 'PersonDetection'):
        """Update tracked person with new detection."""
        self.current_detection = detection
        self.frames_since_seen = 0
        self.total_frames_seen += 1
        self.last_seen_time = detection.timestamp
        
        if self.first_seen_time == 0.0:
            self.first_seen_time = detection.timestamp
        
        # Update position history
        self.position_history.append(detection.position_3d)
        self.confidence_history.append(detection.confidence)
        self.zone_history.append(detection.zone)
        
        # Update Kalman filter
        if len(self.position_history) == 1:
            # Initialize Kalman filter state
            self.kalman_filter.statePre = np.array([
                detection.position_3d[0], detection.position_3d[1], detection.position_3d[2],
                0.0, 0.0, 0.0  # Initial velocity = 0
            ], dtype=np.float32)
            self.kalman_filter.statePost = self.kalman_filter.statePre.copy()
        else:
            # Predict and update
            self.kalman_filter.predict()
            measurement = np.array(detection.position_3d, dtype=np.float32)
            self.kalman_filter.correct(measurement)
        
        # Update smoothed position and velocity
        self._update_trajectory()
    
    def predict_next_position(self) -> Tuple[float, float, float]:
        """Predict next position using Kalman filter."""
        if self.kalman_filter is None:
            return self.get_latest_position()
        
        prediction = self.kalman_filter.predict()
        self.predicted_position = (float(prediction[0]), float(prediction[1]), float(prediction[2]))
        return self.predicted_position
    
    def mark_missing(self):
        """Mark person as missing for current frame."""
        self.frames_since_seen += 1
        self.current_detection = None
        
        # Update predicted position
        if self.kalman_filter is not None:
            self.predict_next_position()
    
    def _update_trajectory(self):
        """Update smoothed trajectory and velocity estimation."""
        if len(self.position_history) < 2:
            self.smoothed_position = self.position_history[-1] if self.position_history else None
            self.velocity = (0.0, 0.0, 0.0)
            return
        
        # Simple trajectory smoothing using moving average
        recent_positions = list(self.position_history)[-5:]  # Last 5 positions
        if len(recent_positions) >= 2:
            # Smoothed position (weighted average favoring recent positions)
            weights = np.linspace(0.5, 1.0, len(recent_positions))
            weights /= np.sum(weights)
            
            smoothed_x = np.average([p[0] for p in recent_positions], weights=weights)
            smoothed_y = np.average([p[1] for p in recent_positions], weights=weights)
            smoothed_z = np.average([p[2] for p in recent_positions], weights=weights)
            
            self.smoothed_position = (smoothed_x, smoothed_y, smoothed_z)
            
            # Velocity estimation (difference between first and last in recent history)
            if len(recent_positions) >= 3:
                dt = max(1.0, len(recent_positions) - 1)  # Time difference in frames
                dx = recent_positions[-1][0] - recent_positions[0][0]
                dy = recent_positions[-1][1] - recent_positions[0][1]
                dz = recent_positions[-1][2] - recent_positions[0][2]
                
                self.velocity = (dx/dt, dy/dt, dz/dt)
    
    def get_latest_position(self) -> Tuple[float, float, float]:
        """Get the most recent position (smoothed if available)."""
        if self.smoothed_position:
            return self.smoothed_position
        elif self.current_detection:
            return self.current_detection.position_3d
        elif self.predicted_position:
            return self.predicted_position
        elif self.position_history:
            return self.position_history[-1]
        else:
            return (0.0, 0.0, 0.0)
    
    def get_average_confidence(self) -> float:
        """Get average confidence over recent detections."""
        if not self.confidence_history:
            return 0.0
        return float(np.mean(self.confidence_history))
    
    def get_dominant_zone(self) -> str:
        """Get the most common zone in recent history."""
        if not self.zone_history:
            return 'low'
        
        # Count zone occurrences
        zone_counts = {}
        for zone in self.zone_history:
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        return max(zone_counts, key=zone_counts.get)
    
    def is_stable(self) -> bool:
        """Check if tracking is stable (enough history and low variance)."""
        if len(self.position_history) < 5:
            return False
        
        # Check position variance
        recent_positions = list(self.position_history)[-5:]
        positions_array = np.array(recent_positions)
        variance = np.mean(np.var(positions_array, axis=0))
        
        return variance < 0.5  # Low variance threshold

class PersonTracker:
    """Manages tracking of multiple people across frames."""
    
    def __init__(self, config: dict):
        """Initialize the person tracker."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Tracking parameters
        self.max_distance_jump = config.get('max_distance_jump', 1.0)  # meters
        self.max_disappeared_frames = config.get('max_disappeared_frames', 30)
        self.min_confidence_for_new_track = config.get('min_confidence_for_new_track', 0.6)
        
        # State
        self.tracked_people: Dict[int, TrackedPerson] = {}
        self.next_id = 1
        self.frame_count = 0
        
        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        self.id_switches = 0
        
    def update(self, detections: List['PersonDetection'], timestamp: float) -> List[TrackedPerson]:
        """Update tracker with new detections and return tracked people."""
        self.frame_count += 1
        
        # Step 1: Predict positions for existing tracks
        for person in self.tracked_people.values():
            person.predict_next_position()
        
        # Step 2: Associate detections with existing tracks
        associations, unmatched_detections, unmatched_tracks = self._associate_detections(detections)
        
        # Step 3: Update matched tracks
        for track_id, detection_idx in associations.items():
            self.tracked_people[track_id].update(detections[detection_idx])
        
        # Step 4: Mark unmatched tracks as missing
        for track_id in unmatched_tracks:
            self.tracked_people[track_id].mark_missing()
        
        # Step 5: Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            if detection.confidence >= self.min_confidence_for_new_track:
                self._create_new_track(detection)
        
        # Step 6: Remove lost tracks
        self._remove_lost_tracks()
        
        # Step 7: Log tracking statistics
        if self.frame_count % 100 == 0:
            self._log_statistics()
        
        return list(self.tracked_people.values())
    
    def _associate_detections(self, detections: List['PersonDetection']) -> Tuple[Dict[int, int], List[int], List[int]]:
        """Associate detections with existing tracks using Hungarian algorithm."""
        if not self.tracked_people or not detections:
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(self.tracked_people.keys())
            return {}, unmatched_detections, unmatched_tracks
        
        # Get current positions for all tracks
        track_ids = list(self.tracked_people.keys())
        track_positions = [self.tracked_people[tid].get_latest_position() for tid in track_ids]
        detection_positions = [d.position_3d for d in detections]
        
        # Calculate distance matrix
        if track_positions and detection_positions:
            distance_matrix = cdist(track_positions, detection_positions, metric='euclidean')
            
            # Apply distance threshold
            distance_matrix[distance_matrix > self.max_distance_jump] = 1e6
            
            # Hungarian algorithm for optimal assignment
            track_indices, detection_indices = linear_sum_assignment(distance_matrix)
            
            # Filter out assignments that are too far
            associations = {}
            for t_idx, d_idx in zip(track_indices, detection_indices):
                if distance_matrix[t_idx, d_idx] < self.max_distance_jump:
                    track_id = track_ids[t_idx]
                    associations[track_id] = d_idx
            
            # Find unmatched detections and tracks
            matched_detection_indices = set(associations.values())
            matched_track_ids = set(associations.keys())
            
            unmatched_detections = [i for i in range(len(detections)) if i not in matched_detection_indices]
            unmatched_tracks = [tid for tid in track_ids if tid not in matched_track_ids]
            
            return associations, unmatched_detections, unmatched_tracks
        
        # Fallback if no positions available
        return {}, list(range(len(detections))), list(self.tracked_people.keys())
    
    def _create_new_track(self, detection: 'PersonDetection'):
        """Create a new track for an unmatched detection."""
        track_id = self.next_id
        self.next_id += 1
        
        # Assign the ID to the detection
        detection.id = track_id
        
        # Create new tracked person
        tracked_person = TrackedPerson(id=track_id)
        tracked_person.update(detection)
        
        self.tracked_people[track_id] = tracked_person
        self.total_tracks_created += 1
        
        self.logger.info(f"Created new track ID {track_id} at position {detection.position_3d}")
    
    def _remove_lost_tracks(self):
        """Remove tracks that have been missing for too long."""
        tracks_to_remove = []
        
        for track_id, person in self.tracked_people.items():
            if person.frames_since_seen > self.max_disappeared_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            removed_person = self.tracked_people.pop(track_id)
            self.total_tracks_lost += 1
            self.logger.info(f"Removed track ID {track_id} after {removed_person.frames_since_seen} missing frames")
    
    def _log_statistics(self):
        """Log tracking statistics."""
        active_tracks = len(self.tracked_people)
        stable_tracks = sum(1 for p in self.tracked_people.values() if p.is_stable())
        
        self.logger.info(f"Tracking stats - Active: {active_tracks}, Stable: {stable_tracks}, "
                        f"Created: {self.total_tracks_created}, Lost: {self.total_tracks_lost}")
    
    def get_track_by_id(self, track_id: int) -> Optional[TrackedPerson]:
        """Get tracked person by ID."""
        return self.tracked_people.get(track_id)
    
    def get_all_tracks(self) -> List[TrackedPerson]:
        """Get all currently tracked people."""
        return list(self.tracked_people.values())
    
    def get_stable_tracks(self) -> List[TrackedPerson]:
        """Get only stable tracks."""
        return [person for person in self.tracked_people.values() if person.is_stable()]
    
    def reset(self):
        """Reset tracker state."""
        self.tracked_people.clear()
        self.next_id = 1
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_lost = 0
        self.id_switches = 0
        self.logger.info("Tracker reset")