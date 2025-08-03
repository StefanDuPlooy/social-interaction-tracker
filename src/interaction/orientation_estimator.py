"""
Orientation Estimator - Phase 2 Step 2
Multi-method body and face orientation detection for social interaction analysis
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import deque
import time
import math

try:
    from ultralytics import YOLO
    POSE_AVAILABLE = True
except ImportError:
    POSE_AVAILABLE = False

@dataclass
class OrientationEstimate:
    """Represents an orientation estimate for a person."""
    person_id: int
    timestamp: float
    orientation_angle: float  # degrees (0 = facing forward/north)
    confidence: float  # 0.0-1.0
    method: str  # 'skeleton', 'movement', 'depth_gradient', 'combined'
    
    # Detailed orientation info
    facing_vector: Tuple[float, float]  # (x, y) unit vector
    head_angle: Optional[float] = None  # degrees, if available
    body_angle: Optional[float] = None  # degrees, if available
    
    # Quality metrics
    joint_visibility_count: int = 0
    movement_magnitude: float = 0.0
    temporal_consistency: float = 0.0

@dataclass 
class MutualOrientation:
    """Represents mutual orientation analysis between two people."""
    person1_id: int
    person2_id: int
    timestamp: float
    
    # Orientation metrics
    person1_to_person2_angle: float  # How much person1 would turn to face person2
    person2_to_person1_angle: float  # How much person2 would turn to face person1
    mutual_facing_score: float  # 0.0-1.0, higher = more likely facing each other
    
    # F-formation analysis
    in_f_formation: bool
    o_space_center: Optional[Tuple[float, float]] = None
    group_coherence: float = 0.0

class OrientationEstimator:
    """Multi-method orientation estimation for tracked people."""
    
    def __init__(self, config: dict):
        """Initialize orientation estimator with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Method configurations
        self.methods = config.get('orientation_methods', {})
        self.skeleton_config = config.get('skeleton_orientation', {})
        self.movement_config = config.get('movement_orientation', {})
        self.gradient_config = config.get('depth_gradient_orientation', {})
        
        # State tracking
        self.orientation_history: Dict[int, deque] = {}
        self.velocity_history: Dict[int, deque] = {}
        self.last_orientations: Dict[int, OrientationEstimate] = {}
        self.pose_model = None
        
        # Statistics
        self.method_success_counts = {'skeleton': 0, 'movement': 0, 'depth_gradient': 0}
        self.total_estimates = 0
        
        # Initialize pose model if available
        self._init_pose_model()
    
    def _init_pose_model(self):
        """Initialize YOLO pose model for skeleton-based orientation."""
        if not POSE_AVAILABLE:
            self.logger.warning("YOLO not available for pose estimation")
            return
        
        try:
            self.pose_model = YOLO('yolov8n-pose.pt')
            self.logger.info("YOLO pose model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load pose model: {e}")
            self.pose_model = None
    
    def estimate_orientations(self, tracked_people: List, depth_frame: np.ndarray, 
                            color_frame: np.ndarray, timestamp: float) -> Tuple[List[OrientationEstimate], Dict]:
        """Estimate orientations for all tracked people and return debug data."""
        orientations = []
        debug_data = {}
        
        for person in tracked_people:
            if person.current_detection is None:
                continue
            
            orientation, person_debug = self.estimate_person_orientation_with_debug(
                person, depth_frame, color_frame, timestamp
            )
            
            if orientation:
                orientations.append(orientation)
                self._update_history(person.id, orientation)
            
            # Store debug data
            debug_data[person.id] = person_debug
        
        self.total_estimates += len(orientations)
        return orientations, debug_data
    
    def estimate_person_orientation_with_debug(self, person, depth_frame: np.ndarray, 
                                             color_frame: np.ndarray, timestamp: float) -> Tuple[Optional[OrientationEstimate], Dict]:
        """Estimate orientation with detailed debug information."""
        person_id = person.id
        bbox = person.current_detection.bounding_box
        
        # Initialize debug data
        debug_data = {
            'skeleton_data': {},
            'movement_data': {},
            'depth_data': {},
            'method_attempts': {}
        }
        
        # Try methods in priority order
        methods_to_try = sorted(
            [(name, config) for name, config in self.methods.items() if config.get('enabled', True)],
            key=lambda x: x[1].get('priority', 999)
        )
        
        estimates = []
        
        for method_name, method_config in methods_to_try:
            try:
                if method_name == 'skeleton_based':
                    estimate, skeleton_debug = self._skeleton_based_orientation_with_debug(person, color_frame, timestamp)
                    debug_data['skeleton_data'] = skeleton_debug
                elif method_name == 'movement_based':
                    estimate, movement_debug = self._movement_based_orientation_with_debug(person, timestamp)
                    debug_data['movement_data'] = movement_debug
                elif method_name == 'depth_gradient':
                    estimate, depth_debug = self._depth_gradient_orientation_with_debug(person, depth_frame, timestamp)
                    debug_data['depth_data'] = depth_debug
                else:
                    continue
                
                # Record method attempt
                if estimate and estimate.confidence >= method_config.get('confidence_threshold', 0.5):
                    estimates.append(estimate)
                    self.method_success_counts[method_name] += 1
                    debug_data['method_attempts'][method_name] = {
                        'success': True,
                        'angle': estimate.orientation_angle,
                        'confidence': estimate.confidence
                    }
                else:
                    debug_data['method_attempts'][method_name] = {
                        'success': False,
                        'reason': 'Low confidence' if estimate else 'Detection failed'
                    }
                    
            except Exception as e:
                self.logger.debug(f"Method {method_name} failed for person {person_id}: {e}")
                debug_data['method_attempts'][method_name] = {
                    'success': False,
                    'reason': f'Exception: {str(e)}'
                }
        
        # Combine estimates if multiple methods succeeded
        if estimates:
            final_orientation = self._combine_orientation_estimates(estimates, timestamp)
            debug_data['final_result'] = {
                'angle': final_orientation.orientation_angle,
                'confidence': final_orientation.confidence,
                'method': final_orientation.method,
                'num_methods_combined': len(estimates)
            }
            return final_orientation, debug_data
        else:
            fallback = self._fallback_orientation(person, timestamp)
            debug_data['final_result'] = {
                'angle': fallback.orientation_angle if fallback else 0,
                'confidence': fallback.confidence if fallback else 0,
                'method': 'fallback',
                'num_methods_combined': 0
            }
        return fallback, debug_data
    
    def _skeleton_based_orientation(self, person, color_frame: np.ndarray, 
                                  timestamp: float) -> Optional[OrientationEstimate]:
        """Estimate orientation from skeleton keypoints."""
        if not self.pose_model:
            return None
        
        person_id = person.id
        bbox = person.current_detection.bounding_box
        x, y, w, h = bbox
        
        # Extract person region
        person_roi = color_frame[y:y+h, x:x+w]
        if person_roi.size == 0:
            return None
        
        try:
            # Run pose estimation
            results = self.pose_model(person_roi, verbose=False)
            
            if not results or len(results) == 0:
                return None
            
            # Get keypoints
            keypoints = results[0].keypoints
            if keypoints is None or len(keypoints.data) == 0:
                return None
            
            # Extract keypoints (COCO format)
            kpts = keypoints.data[0].cpu().numpy()  # [17, 3] - (x, y, confidence)
            
            # Key indices in COCO format
            nose = 0; left_eye = 1; right_eye = 2; left_ear = 3; right_ear = 4
            left_shoulder = 5; right_shoulder = 6; left_elbow = 7; right_elbow = 8
            
            # Calculate body orientation from shoulders
            body_angle = None
            if (kpts[left_shoulder][2] > 0.3 and kpts[right_shoulder][2] > 0.3):
                # Shoulder vector in ROI coordinates
                shoulder_vec = kpts[right_shoulder][:2] - kpts[left_shoulder][:2]
                
                # Convert to global coordinates
                shoulder_vec_global = np.array([shoulder_vec[0], shoulder_vec[1]])
                
                # Body orientation is perpendicular to shoulder line
                # Assuming person faces forward from their right side
                body_angle = math.degrees(math.atan2(-shoulder_vec_global[0], shoulder_vec_global[1]))
            
            # Calculate head orientation from face keypoints
            head_angle = None
            face_points = [nose, left_eye, right_eye, left_ear, right_ear]
            valid_face_points = [(i, kpts[i]) for i in face_points if kpts[i][2] > 0.3]
            
            if len(valid_face_points) >= 2:
                # Use eye/nose configuration to determine head direction
                if (kpts[left_eye][2] > 0.3 and kpts[right_eye][2] > 0.3 and kpts[nose][2] > 0.3):
                    eye_center = (kpts[left_eye][:2] + kpts[right_eye][:2]) / 2
                    nose_pos = kpts[nose][:2]
                    
                    # Face direction vector
                    face_vec = nose_pos - eye_center
                    head_angle = math.degrees(math.atan2(face_vec[1], face_vec[0]))
            
            # Combine body and head angles
            final_angle = None
            confidence = 0.0
            joint_count = 0
            
            if body_angle is not None and head_angle is not None:
                # Weight body more than head
                final_angle = (self.skeleton_config['shoulder_vector_weight'] * body_angle + 
                              self.skeleton_config['head_vector_weight'] * head_angle)
                confidence = 0.9
                joint_count = len([i for i in range(len(kpts)) if kpts[i][2] > 0.3])
                
            elif body_angle is not None:
                final_angle = body_angle
                confidence = 0.8
                joint_count = 2  # shoulders
                
            elif head_angle is not None:
                final_angle = head_angle
                confidence = 0.6
                joint_count = 3  # face points
            
            if final_angle is None:
                return None
            
            # Normalize angle to 0-360
            final_angle = (final_angle + 360) % 360
            
            # Create facing vector
            facing_vector = (
                math.cos(math.radians(final_angle)),
                math.sin(math.radians(final_angle))
            )
            
            # Apply temporal smoothing if enabled
            if self.skeleton_config.get('temporal_smoothing', True):
                final_angle, confidence = self._apply_temporal_smoothing(
                    person_id, final_angle, confidence, timestamp
                )
            
            return OrientationEstimate(
                person_id=person_id,
                timestamp=timestamp,
                orientation_angle=final_angle,
                confidence=confidence,
                method='skeleton',
                facing_vector=facing_vector,
                head_angle=head_angle,
                body_angle=body_angle,
                joint_visibility_count=joint_count
            )
            
        except Exception as e:
            self.logger.debug(f"Skeleton orientation failed for person {person_id}: {e}")
            return None
    
    def _movement_based_orientation(self, person, timestamp: float) -> Optional[OrientationEstimate]:
        """Estimate orientation from movement direction."""
        person_id = person.id
        
        # Get recent positions
        if len(person.position_history) < 2:
            return None
        
        # Calculate velocity over recent history
        positions = list(person.position_history)
        if len(positions) < self.movement_config['velocity_history_length']:
            positions = positions[-len(positions):]
        else:
            positions = positions[-self.movement_config['velocity_history_length']:]
        
        # Compute movement vector
        movement_vectors = []
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]
            
            movement = np.array([curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]])
            movement_vectors.append(movement)
        
        if not movement_vectors:
            return None
        
        # Average movement direction
        avg_movement = np.mean(movement_vectors, axis=0)
        movement_magnitude = np.linalg.norm(avg_movement)
        
        # Check if movement is significant enough
        if movement_magnitude < self.movement_config['min_velocity_threshold']:
            return None
        
        # Calculate orientation angle from movement
        movement_angle = math.degrees(math.atan2(avg_movement[1], avg_movement[0]))
        movement_angle = (movement_angle + 360) % 360
        
        # Confidence based on movement magnitude and consistency
        confidence = min(1.0, movement_magnitude / self.movement_config['confidence_boost_speed'])
        confidence *= self.movement_config.get('movement_base_confidence', 0.7)
        
        # Check consistency across recent movements
        if len(movement_vectors) > 3:
            angles = [math.degrees(math.atan2(v[1], v[0])) for v in movement_vectors]
            angle_variance = np.var(angles)
            consistency = max(0, 1 - angle_variance / 180)  # Normalize variance
            confidence *= (0.5 + 0.5 * consistency)
        
        facing_vector = (
            math.cos(math.radians(movement_angle)),
            math.sin(math.radians(movement_angle))
        )
        
        return OrientationEstimate(
            person_id=person_id,
            timestamp=timestamp,
            orientation_angle=movement_angle,
            confidence=confidence,
            method='movement',
            facing_vector=facing_vector,
            movement_magnitude=movement_magnitude
        )
    
    def _depth_gradient_orientation(self, person, depth_frame: np.ndarray, 
                                  timestamp: float) -> Optional[OrientationEstimate]:
        """Estimate orientation from depth gradient analysis."""
        person_id = person.id
        bbox = person.current_detection.bounding_box
        x, y, w, h = bbox
        
        # Extract person region from depth frame
        roi = depth_frame[y:y+h, x:x+w].copy()
        if roi.size == 0:
            return None
        
        # Clean depth data
        roi[roi == 0] = np.nan
        if np.all(np.isnan(roi)):
            return None
        
        # Calculate depth gradients
        roi_clean = np.nan_to_num(roi, nan=np.nanmean(roi))
        
        # Sobel gradients
        grad_x = cv2.Sobel(roi_clean, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(roi_clean, cv2.CV_64F, 0, 1, ksize=5)
        
        # Analyze asymmetry to determine front/back
        # Front of person typically closer (smaller depth values)
        roi_h, roi_w = roi_clean.shape
        
        # Compare left vs right halves
        left_half = roi_clean[:, :roi_w//2]
        right_half = roi_clean[:, roi_w//2:]
        
        left_mean = np.nanmean(left_half)
        right_mean = np.nanmean(right_half)
        
        # Compare top vs bottom halves  
        top_half = roi_clean[:roi_h//2, :]
        bottom_half = roi_clean[roi_h//2:, :]
        
        top_mean = np.nanmean(top_half)
        bottom_mean = np.nanmean(bottom_half)
        
        # Determine orientation based on asymmetry
        lr_diff = left_mean - right_mean
        tb_diff = top_mean - bottom_mean
        
        # Assume person faces toward closer (lower depth) side
        if abs(lr_diff) > abs(tb_diff):
            # Left-right asymmetry dominates
            if lr_diff > 0:
                orientation_angle = 270  # Facing right
            else:
                orientation_angle = 90   # Facing left
        else:
            # Top-bottom asymmetry dominates  
            if tb_diff > 0:
                orientation_angle = 180  # Facing down/back
            else:
                orientation_angle = 0    # Facing up/front
        
        # Confidence based on asymmetry strength
        max_asymmetry = max(abs(lr_diff), abs(tb_diff))
        confidence = min(1.0, max_asymmetry / 0.5)  # 0.5m difference = full confidence
        confidence *= self.gradient_config.get('confidence_threshold', 0.3)
        
        facing_vector = (
            math.cos(math.radians(orientation_angle)),
            math.sin(math.radians(orientation_angle))
        )
        
        return OrientationEstimate(
            person_id=person_id,
            timestamp=timestamp,
            orientation_angle=orientation_angle,
            confidence=confidence,
            method='depth_gradient',
            facing_vector=facing_vector
        )
    
    def _combine_orientation_estimates(self, estimates: List[OrientationEstimate], 
                                     timestamp: float) -> OrientationEstimate:
        """Combine multiple orientation estimates using weighted averaging."""
        if len(estimates) == 1:
            return estimates[0]
        
        person_id = estimates[0].person_id
        
        # Weight by confidence and method priority
        method_weights = {'skeleton': 1.0, 'movement': 0.7, 'depth_gradient': 0.4}
        
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        weighted_confidence = 0
        
        for estimate in estimates:
            method_weight = method_weights.get(estimate.method, 0.5)
            weight = estimate.confidence * method_weight
            
            # Convert angle to unit vector for averaging
            x = math.cos(math.radians(estimate.orientation_angle))
            y = math.sin(math.radians(estimate.orientation_angle))
            
            weighted_x += weight * x
            weighted_y += weight * y
            weighted_confidence += weight
            total_weight += weight
        
        if total_weight == 0:
            return estimates[0]  # Fallback
        
        # Average the vectors
        avg_x = weighted_x / total_weight
        avg_y = weighted_y / total_weight
        
        # Convert back to angle
        combined_angle = math.degrees(math.atan2(avg_y, avg_x))
        combined_angle = (combined_angle + 360) % 360
        
        # Average confidence with bonus for method agreement
        combined_confidence = weighted_confidence / total_weight
        if len(estimates) > 1:
            combined_confidence += self.config.get('confidence_scoring', {}).get('multi_method_agreement_bonus', 0.15)
            combined_confidence = min(1.0, combined_confidence)
        
        return OrientationEstimate(
            person_id=person_id,
            timestamp=timestamp,
            orientation_angle=combined_angle,
            confidence=combined_confidence,
            method='combined',
            facing_vector=(avg_x, avg_y),
            joint_visibility_count=max(e.joint_visibility_count for e in estimates),
            movement_magnitude=max(e.movement_magnitude for e in estimates)
        )
    
    def _apply_temporal_smoothing(self, person_id: int, angle: float, 
                                confidence: float, timestamp: float) -> Tuple[float, float]:
        """Apply temporal smoothing to orientation estimates."""
        if person_id not in self.last_orientations:
            return angle, confidence
        
        last_estimate = self.last_orientations[person_id]
        time_diff = timestamp - last_estimate.timestamp
        
        if time_diff > 2.0:  # Too much time passed, don't smooth
            return angle, confidence
        
        # Calculate angular difference (handling wrap-around)
        angle_diff = abs(angle - last_estimate.orientation_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # If change is too rapid, reduce confidence
        max_change = self.skeleton_config.get('max_angle_change', 45)
        if angle_diff > max_change:
            confidence *= 0.7
        
        # Exponential smoothing
        alpha = self.skeleton_config.get('smoothing_alpha', 0.7)
        
        # Handle angle wrap-around for smoothing
        if abs(angle - last_estimate.orientation_angle) > 180:
            if angle > last_estimate.orientation_angle:
                angle -= 360
            else:
                angle += 360
        
        smoothed_angle = alpha * angle + (1 - alpha) * last_estimate.orientation_angle
        smoothed_angle = (smoothed_angle + 360) % 360
        
        # Boost confidence for temporal consistency
        consistency_bonus = self.config.get('confidence_scoring', {}).get('temporal_consistency_bonus', 0.2)
        if angle_diff < 15:  # Small change
            confidence = min(1.0, confidence + consistency_bonus)
        
        return smoothed_angle, confidence
    
    def _fallback_orientation(self, person, timestamp: float) -> Optional[OrientationEstimate]:
        """Provide fallback orientation when all methods fail."""
        person_id = person.id
        
        # Use last known orientation if available
        if person_id in self.last_orientations:
            last = self.last_orientations[person_id]
            if timestamp - last.timestamp < self.config.get('error_handling', {}).get('last_known_timeout', 60):
                # Return last known with reduced confidence
                return OrientationEstimate(
                    person_id=person_id,
                    timestamp=timestamp,
                    orientation_angle=last.orientation_angle,
                    confidence=max(0.1, last.confidence * 0.5),
                    method='fallback_last_known',
                    facing_vector=last.facing_vector
                )
        
        # Default to forward-facing with low confidence
        default_angle = self.config.get('classroom_orientation', {}).get('default_forward_direction', 0)
        
        return OrientationEstimate(
            person_id=person_id,
            timestamp=timestamp,
            orientation_angle=default_angle,
            confidence=0.1,
            method='fallback_default',
            facing_vector=(math.cos(math.radians(default_angle)), 
                          math.sin(math.radians(default_angle)))
        )
    
    def _update_history(self, person_id: int, orientation: OrientationEstimate):
        """Update orientation history for a person."""
        if person_id not in self.orientation_history:
            self.orientation_history[person_id] = deque(maxlen=100)
        
        self.orientation_history[person_id].append(orientation)
        self.last_orientations[person_id] = orientation
    
    def analyze_mutual_orientations(self, orientations: List[OrientationEstimate], 
                                  tracked_people: List) -> List[MutualOrientation]:
        """Analyze mutual orientations between all pairs of people."""
        mutual_orientations = []
        
        # Create position lookup
        position_lookup = {}
        for person in tracked_people:
            if person.current_detection:
                position_lookup[person.id] = person.get_latest_position()
        
        # Analyze all pairs
        for i, orient1 in enumerate(orientations):
            for j, orient2 in enumerate(orientations[i+1:], i+1):
                if orient1.person_id == orient2.person_id:
                    continue
                
                # Get positions
                pos1 = position_lookup.get(orient1.person_id)
                pos2 = position_lookup.get(orient2.person_id)
                
                if pos1 is None or pos2 is None:
                    continue
                
                mutual = self._calculate_mutual_orientation(orient1, orient2, pos1, pos2)
                if mutual:
                    mutual_orientations.append(mutual)
        
        return mutual_orientations
    
    def _calculate_mutual_orientation(self, orient1: OrientationEstimate, orient2: OrientationEstimate,
                                    pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> Optional[MutualOrientation]:
        """Calculate mutual orientation metrics between two people."""
        # Calculate angle from person1 to person2
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.1:  # Too close for meaningful orientation
            return None
        
        # Angle from person1 to person2
        target_angle_1_to_2 = math.degrees(math.atan2(dy, dx))
        target_angle_1_to_2 = (target_angle_1_to_2 + 360) % 360
        
        # Angle from person2 to person1  
        target_angle_2_to_1 = (target_angle_1_to_2 + 180) % 360
        
        # Calculate how much each person would need to turn
        turn_angle_1 = self._calculate_turn_angle(orient1.orientation_angle, target_angle_1_to_2)
        turn_angle_2 = self._calculate_turn_angle(orient2.orientation_angle, target_angle_2_to_1)
        
        # Calculate mutual facing score
        max_turn = max(turn_angle_1, turn_angle_2)
        facing_threshold = self.config.get('mutual_orientation', {}).get('facing_angle_threshold', 60)
        
        if max_turn <= facing_threshold:
            mutual_facing_score = 1.0 - (max_turn / facing_threshold)
        else:
            mutual_facing_score = 0.0
        
        # Weight by orientation confidence
        combined_confidence = (orient1.confidence + orient2.confidence) / 2
        mutual_facing_score *= combined_confidence
        
        # F-formation detection
        f_formation_config = self.config.get('mutual_orientation', {})
        in_f_formation = False
        o_space_center = None
        group_coherence = 0.0
        
        if f_formation_config.get('f_formation_enabled', True):
            # Check if both people are oriented toward a common center point
            o_space_radius = f_formation_config.get('o_space_radius', 1.5)
            
            # Calculate potential O-space center
            center_x = (pos1[0] + pos2[0]) / 2
            center_y = (pos1[1] + pos2[1]) / 2
            o_space_center = (center_x, center_y)
            
            # Check if both people are facing toward the center
            angle_to_center_1 = math.degrees(math.atan2(center_y - pos1[1], center_x - pos1[0]))
            angle_to_center_2 = math.degrees(math.atan2(center_y - pos2[1], center_x - pos2[0]))
            
            turn_to_center_1 = self._calculate_turn_angle(orient1.orientation_angle, angle_to_center_1)
            turn_to_center_2 = self._calculate_turn_angle(orient2.orientation_angle, angle_to_center_2)
            
            # F-formation if both face toward center and are within reasonable distance
            center_facing_threshold = 45  # degrees
            if (turn_to_center_1 <= center_facing_threshold and 
                turn_to_center_2 <= center_facing_threshold and
                distance <= o_space_radius * 2):
                
                in_f_formation = True
                group_coherence = 1.0 - (turn_to_center_1 + turn_to_center_2) / (2 * center_facing_threshold)
                group_coherence *= combined_confidence
        
        return MutualOrientation(
            person1_id=orient1.person_id,
            person2_id=orient2.person_id,
            timestamp=orient1.timestamp,
            person1_to_person2_angle=turn_angle_1,
            person2_to_person1_angle=turn_angle_2,
            mutual_facing_score=mutual_facing_score,
            in_f_formation=in_f_formation,
            o_space_center=o_space_center,
            group_coherence=group_coherence
        )
    
    def _calculate_turn_angle(self, current_angle: float, target_angle: float) -> float:
        """Calculate minimum turn angle between two orientations."""
        diff = abs(current_angle - target_angle)
        return min(diff, 360 - diff)
    
    def get_orientation_statistics(self) -> Dict:
        """Get statistics about orientation estimation performance."""
        total_success = sum(self.method_success_counts.values())
        
        stats = {
            'total_estimates': self.total_estimates,
            'method_success_counts': self.method_success_counts.copy(),
            'method_success_rates': {},
            'active_people_tracked': len(self.orientation_history),
            'avg_confidence_by_method': {},
        }
        
        # Calculate success rates
        for method, count in self.method_success_counts.items():
            if self.total_estimates > 0:
                stats['method_success_rates'][method] = count / self.total_estimates
            else:
                stats['method_success_rates'][method] = 0.0
        
        # Calculate average confidence by method
        for person_history in self.orientation_history.values():
            for orientation in person_history:
                method = orientation.method
                if method not in stats['avg_confidence_by_method']:
                    stats['avg_confidence_by_method'][method] = []
                stats['avg_confidence_by_method'][method].append(orientation.confidence)
        
        # Average the confidence scores
        for method in stats['avg_confidence_by_method']:
            confidences = stats['avg_confidence_by_method'][method]
            stats['avg_confidence_by_method'][method] = np.mean(confidences) if confidences else 0.0
        
        return stats
    
    def reset(self):
        """Reset all orientation tracking state."""
        self.orientation_history.clear()
        self.velocity_history.clear()
        self.last_orientations.clear()
        self.method_success_counts = {'skeleton': 0, 'movement': 0, 'depth_gradient': 0}
        self.total_estimates = 0
        self.logger.info("Orientation estimator reset")
    
    def visualize_orientations(self, frame: np.ndarray, orientations: List[OrientationEstimate],
                             mutual_orientations: List[MutualOrientation] = None) -> np.ndarray:
        """Create visualization of orientation estimates."""
        vis_frame = frame.copy()
        viz_config = self.config.get('orientation_visualization', {})
        
        if not viz_config.get('draw_orientation_vectors', True):
            return vis_frame
        
        # Draw individual orientations
        for orientation in orientations:
            self._draw_orientation_vector(vis_frame, orientation, viz_config)
        
        # Draw mutual orientations
        if mutual_orientations and viz_config.get('draw_facing_connections', True):
            for mutual in mutual_orientations:
                self._draw_mutual_orientation(vis_frame, mutual, orientations, viz_config)
        
        return vis_frame
    
    def _draw_orientation_vector(self, frame: np.ndarray, orientation: OrientationEstimate, viz_config: Dict):
        """Draw orientation vector for a single person."""
        # Find person position (simplified - would need person lookup in real implementation)
        # For now, we'll use a placeholder center point
        person_center = (320, 240)  # This should come from the person's detection
        
        vector_length = viz_config.get('vector_length', 100)
        colors = viz_config.get('vector_colors', {})
        color = colors.get(orientation.method, (255, 255, 255))
        
        # Calculate end point of orientation vector
        angle_rad = math.radians(orientation.orientation_angle)
        end_x = int(person_center[0] + vector_length * math.cos(angle_rad))
        end_y = int(person_center[1] + vector_length * math.sin(angle_rad))
        
        # Draw orientation vector
        cv2.arrowedLine(frame, person_center, (end_x, end_y), color, 2, tipLength=0.3)
        
        # Draw confidence circle
        if viz_config.get('draw_confidence_circle', True):
            radius = int(viz_config.get('confidence_circle_radius', 30) * orientation.confidence)
            cv2.circle(frame, person_center, radius, color, 1)
        
        # Show orientation angle
        if viz_config.get('show_orientation_angle', True):
            angle_text = f"{orientation.orientation_angle:.0f}°"
            text_offset = viz_config.get('angle_text_offset', (10, -10))
            text_pos = (person_center[0] + text_offset[0], person_center[1] + text_offset[1])
            cv2.putText(frame, angle_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _draw_mutual_orientation(self, frame: np.ndarray, mutual: MutualOrientation,
                               orientations: List[OrientationEstimate], viz_config: Dict):
        """Draw mutual orientation connection between two people."""
        # This would need actual person positions
        # Placeholder implementation
        if mutual.mutual_facing_score > 0.5:
            color = viz_config.get('mutual_attention_color', (255, 0, 255))
            thickness = viz_config.get('facing_line_thickness', 3)
            
            # Draw connection line (would need real positions)
            # cv2.line(frame, person1_pos, person2_pos, color, thickness)
            
            # Draw F-formation indicator
            if mutual.in_f_formation and mutual.o_space_center:
                center = (int(mutual.o_space_center[0]), int(mutual.o_space_center[1]))
                cv2.circle(frame, center, 20, color, 2)
                cv2.putText(frame, "F-Form", (center[0]-25, center[1]-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)