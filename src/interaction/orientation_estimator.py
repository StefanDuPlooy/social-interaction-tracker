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
    
    def _skeleton_based_orientation_with_debug(self, person, color_frame: np.ndarray, 
                                             timestamp: float) -> Tuple[Optional[OrientationEstimate], Dict]:
        """Estimate orientation from skeleton keypoints with enhanced debugging."""
        debug_data = {'keypoints': [], 'method': 'skeleton', 'issues': []}
        
        if not self.pose_model:
            debug_data['issues'].append("No pose model available")
            self.logger.warning("No pose model available for skeleton detection")
            return None, debug_data
        
        # Extract person region from color frame
        detection = person.current_detection
        if not detection:
            debug_data['issues'].append("No current detection")
            self.logger.debug(f"No current detection for person {person.id}")
            return None, debug_data
        
        bbox = detection.bounding_box
        x, y, w, h = bbox
        
        # Add bounds checking
        if x < 0 or y < 0 or x + w > color_frame.shape[1] or y + h > color_frame.shape[0]:
            debug_data['issues'].append(f"Bbox out of bounds: {bbox}")
            self.logger.warning(f"Person {person.id} bbox out of frame bounds: {bbox}")
            return None, debug_data
        
        try:
            person_roi = color_frame[y:y+h, x:x+w]
            
            # ENHANCED DEBUG: Log ROI details
            debug_data['roi_size'] = person_roi.shape
            self.logger.info(f"Processing person {person.id} ROI size: {person_roi.shape}")
            
            # Check for valid ROI
            if person_roi.size == 0:
                debug_data['issues'].append("Empty ROI")
                return None, debug_data
            
            # Run pose detection on person ROI
            results = self.pose_model(person_roi, verbose=False)
            
            if not results or len(results) == 0:
                debug_data['issues'].append("No pose results")
                self.logger.warning(f"No pose results for person {person.id}")
                return None, debug_data
            
            # Extract keypoints from first detection
            result = results[0]
            if not hasattr(result, 'keypoints') or result.keypoints is None:
                debug_data['issues'].append("No keypoints in result")
                self.logger.warning(f"No keypoints in pose result for person {person.id}")
                return None, debug_data
            
            keypoints = result.keypoints.data[0].cpu().numpy()  # Shape: (17, 3)
            debug_data['keypoints'] = keypoints.tolist()
            
            # ENHANCED DEBUG: Count and log visible keypoints
            visible_keypoints = keypoints[keypoints[:, 2] > 0.3]
            debug_data['visible_keypoints_count'] = len(visible_keypoints)
            debug_data['total_keypoints'] = len(keypoints)
            
            self.logger.info(f"Person {person.id}: {len(visible_keypoints)} visible keypoints out of {len(keypoints)}")
            
            if len(visible_keypoints) < 3:
                debug_data['issues'].append(f"Insufficient keypoints ({len(visible_keypoints)} < 3)")
                self.logger.warning(f"Person {person.id}: Insufficient keypoints ({len(visible_keypoints)} < 3)")
                return None, debug_data
            
            # Check for key skeletal landmarks
            nose = keypoints[0] if keypoints[0, 2] > 0.3 else None
            left_shoulder = keypoints[5] if keypoints[5, 2] > 0.3 else None
            right_shoulder = keypoints[6] if keypoints[6, 2] > 0.3 else None
            
            # ENHANCED DEBUG: Log key point availability
            debug_data['key_points'] = {
                'nose': nose is not None,
                'left_shoulder': left_shoulder is not None,
                'right_shoulder': right_shoulder is not None
            }
            
            self.logger.info(f"Person {person.id} key points - Nose: {nose is not None}, "
                            f"L_Shoulder: {left_shoulder is not None}, R_Shoulder: {right_shoulder is not None}")
            
            # Calculate orientation angles
            person_id = person.id
            body_angle = None
            head_angle = None
            confidence = 0.3  # Base confidence
            joint_count = len(visible_keypoints)
            
            # Try body orientation from shoulders
            if left_shoulder is not None and right_shoulder is not None:
                # Calculate shoulder vector (in ROI coordinates)
                shoulder_vec = right_shoulder[:2] - left_shoulder[:2]
                
                # Body faces perpendicular to shoulder line
                # Perpendicular vector (rotate 90 degrees)
                body_vec = np.array([-shoulder_vec[1], shoulder_vec[0]])
                
                # Convert to angle (0 = facing up in image)
                body_angle = math.degrees(math.atan2(body_vec[1], body_vec[0]))
                confidence = 0.7
                joint_count = 2  # shoulders
                
                debug_data['shoulder_vector'] = shoulder_vec.tolist()
                debug_data['body_angle'] = body_angle
                
                self.logger.info(f"Person {person.id}: Body angle from shoulders: {body_angle:.1f}°")
                
            # Try head orientation from face keypoints  
            if nose is not None:
                left_eye = keypoints[1] if keypoints[1, 2] > 0.3 else None
                right_eye = keypoints[2] if keypoints[2, 2] > 0.3 else None
                
                if left_eye is not None and right_eye is not None:
                    # Eye vector
                    eye_vec = right_eye[:2] - left_eye[:2]
                    # Face direction perpendicular to eye line
                    face_vec = np.array([-eye_vec[1], eye_vec[0]])
                    head_angle = math.degrees(math.atan2(face_vec[1], face_vec[0]))
                    
                    debug_data['eye_vector'] = eye_vec.tolist()
                    debug_data['head_angle'] = head_angle
                    
                    self.logger.info(f"Person {person.id}: Head angle from eyes: {head_angle:.1f}°")
            
            # Choose best orientation estimate
            final_angle = None
            if body_angle is not None and head_angle is not None:
                # Combine body and head estimates
                final_angle = (body_angle * 0.7 + head_angle * 0.3)
                confidence = 0.8
                joint_count = len(visible_keypoints)
                debug_data['final_method'] = 'body_and_head_combined'
                
            elif body_angle is not None:
                final_angle = body_angle
                confidence = 0.6
                joint_count = 2  # shoulders
                debug_data['final_method'] = 'body_only'
                
            elif head_angle is not None:
                final_angle = head_angle
                confidence = 0.5
                joint_count = 3  # face points
                debug_data['final_method'] = 'head_only'
            
            if final_angle is None:
                debug_data['issues'].append("No valid orientation calculated")
                return None, debug_data
            
            # Normalize angle to 0-360
            final_angle = (final_angle + 360) % 360
            debug_data['final_angle'] = final_angle
            debug_data['final_confidence'] = confidence
            
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
                debug_data['temporal_smoothing_applied'] = True
                debug_data['smoothed_angle'] = final_angle
                debug_data['smoothed_confidence'] = confidence
            
            self.logger.info(f"Person {person.id}: Final skeleton orientation: {final_angle:.1f}° (conf: {confidence:.2f})")
            
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
            ), debug_data
            
        except Exception as e:
            debug_data['issues'].append(f"Exception: {str(e)}")
            self.logger.error(f"Skeleton orientation failed for person {person.id}: {e}")
            return None, debug_data
    
    def _skeleton_based_orientation(self, person, color_frame: np.ndarray, 
                                  timestamp: float) -> Optional[OrientationEstimate]:
        """Estimate orientation from skeleton keypoints (legacy method)."""
        estimate, _ = self._skeleton_based_orientation_with_debug(person, color_frame, timestamp)
        return estimate
    
    def _movement_based_orientation_with_debug(self, person, timestamp: float) -> Tuple[Optional[OrientationEstimate], Dict]:
        """Estimate orientation from movement direction with debug info."""
        debug_data = {'method': 'movement', 'issues': []}
        
        person_id = person.id
        
        # Get position history
        if not hasattr(person, 'position_history') or len(person.position_history) < 2:
            debug_data['issues'].append("Insufficient position history")
            return None, debug_data
        
        # Calculate movement vector from recent positions
        recent_positions = person.position_history[-5:]  # Last 5 positions
        if len(recent_positions) < 2:
            debug_data['issues'].append("Insufficient recent positions")
            return None, debug_data
        
        # Calculate velocity vector
        pos_current = np.array(recent_positions[-1][:2])  # x, y only
        pos_previous = np.array(recent_positions[-2][:2])
        
        movement_vec = pos_current - pos_previous
        movement_magnitude = np.linalg.norm(movement_vec)
        
        debug_data['movement_vector'] = movement_vec.tolist()
        debug_data['movement_magnitude'] = movement_magnitude
        debug_data['recent_positions'] = [list(pos[:2]) for pos in recent_positions]
        
        # Check if there's sufficient movement
        min_movement = self.movement_config.get('min_movement_threshold', 0.1)
        if movement_magnitude < min_movement:
            debug_data['issues'].append(f"Insufficient movement ({movement_magnitude:.3f} < {min_movement})")
            return None, debug_data
        
        # Calculate orientation angle from movement direction
        orientation_angle = math.degrees(math.atan2(movement_vec[1], movement_vec[0]))
        orientation_angle = (orientation_angle + 360) % 360
        
        # Confidence based on movement magnitude and consistency
        confidence = min(0.8, movement_magnitude * 2.0)  # Scale movement to confidence
        
        debug_data['orientation_angle'] = orientation_angle
        debug_data['confidence'] = confidence
        
        # Create facing vector
        facing_vector = (
            math.cos(math.radians(orientation_angle)),
            math.sin(math.radians(orientation_angle))
        )
        
        return OrientationEstimate(
            person_id=person_id,
            timestamp=timestamp,
            orientation_angle=orientation_angle,
            confidence=confidence,
            method='movement',
            facing_vector=facing_vector,
            movement_magnitude=movement_magnitude
        ), debug_data
    
    def _movement_based_orientation(self, person, timestamp: float) -> Optional[OrientationEstimate]:
        """Estimate orientation from movement direction (legacy method)."""
        estimate, _ = self._movement_based_orientation_with_debug(person, timestamp)
        return estimate
    
    def _depth_gradient_orientation_with_debug(self, person, depth_frame: np.ndarray, 
                                             timestamp: float) -> Tuple[Optional[OrientationEstimate], Dict]:
        """Estimate orientation from depth gradients with debug info."""
        debug_data = {'method': 'depth_gradient', 'issues': []}
        
        person_id = person.id
        bbox = person.current_detection.bounding_box
        x, y, w, h = bbox
        
        # Extract person region from depth frame
        person_depth = depth_frame[y:y+h, x:x+w]
        
        if person_depth.size == 0:
            debug_data['issues'].append("Empty depth ROI")
            return None, debug_data
        
        # Remove invalid depth values (0 or very far)
        valid_depth = person_depth[(person_depth > 0) & (person_depth < 5000)]  # mm
        
        if len(valid_depth) < 10:
            debug_data['issues'].append("Insufficient valid depth points")
            return None, debug_data
        
        debug_data['roi_size'] = person_depth.shape
        debug_data['valid_depth_points'] = len(valid_depth)
        debug_data['depth_range'] = [float(valid_depth.min()), float(valid_depth.max())]
        
        # Calculate depth gradients (simplified approach)
        # Split person region into left/right halves
        mid_x = w // 2
        left_half = person_depth[:, :mid_x]
        right_half = person_depth[:, mid_x:]
        
        # Get average depths for each half (excluding zeros)
        left_valid = left_half[left_half > 0]
        right_valid = right_half[right_half > 0]
        
        if len(left_valid) < 5 or len(right_valid) < 5:
            debug_data['issues'].append("Insufficient depth data in halves")
            return None, debug_data
        
        left_avg_depth = np.mean(left_valid)
        right_avg_depth = np.mean(right_valid)
        
        debug_data['left_avg_depth'] = float(left_avg_depth)
        debug_data['right_avg_depth'] = float(right_avg_depth)
        
        # Depth difference indicates facing direction
        depth_diff = left_avg_depth - right_avg_depth
        debug_data['depth_difference'] = float(depth_diff)
        
        # Threshold for significant depth difference
        min_gradient = self.gradient_config.get('min_gradient_strength', 50)  # mm
        if abs(depth_diff) < min_gradient:
            debug_data['issues'].append(f"Insufficient depth gradient ({abs(depth_diff):.1f} < {min_gradient})")
            return None, debug_data
        
        # If left side is closer (smaller depth), person is facing right
        # If right side is closer, person is facing left
        if depth_diff > 0:
            # Left side farther, right side closer -> facing right
            orientation_angle = 90.0
        else:
            # Right side farther, left side closer -> facing left  
            orientation_angle = 270.0
        
        # Confidence based on depth difference magnitude
        confidence = min(0.6, abs(depth_diff) / 100.0)  # Scale to confidence
        
        debug_data['orientation_angle'] = orientation_angle
        debug_data['confidence'] = confidence
        
        # Create facing vector
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
        ), debug_data
    
    def _depth_gradient_orientation(self, person, depth_frame: np.ndarray, 
                                  timestamp: float) -> Optional[OrientationEstimate]:
        """Estimate orientation from depth gradients (legacy method)."""
        estimate, _ = self._depth_gradient_orientation_with_debug(person, depth_frame, timestamp)
        return estimate
    
    def _combine_orientation_estimates(self, estimates: List[OrientationEstimate], 
                                     timestamp: float) -> OrientationEstimate:
        """Combine multiple orientation estimates into a single best estimate."""
        if len(estimates) == 1:
            return estimates[0]
        
        # Weight estimates by confidence and method priority
        method_weights = {'skeleton': 3.0, 'movement': 2.0, 'depth_gradient': 1.0}
        
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        best_estimate = estimates[0]
        
        for estimate in estimates:
            method_weight = method_weights.get(estimate.method, 1.0)
            weight = estimate.confidence * method_weight
            
            # Convert angle to unit vector for averaging
            x = math.cos(math.radians(estimate.orientation_angle))
            y = math.sin(math.radians(estimate.orientation_angle))
            
            weighted_x += x * weight
            weighted_y += y * weight
            total_weight += weight
            
            # Keep the highest confidence estimate as base
            if estimate.confidence > best_estimate.confidence:
                best_estimate = estimate
        
        # Calculate combined angle
        if total_weight > 0:
            avg_x = weighted_x / total_weight
            avg_y = weighted_y / total_weight
            combined_angle = math.degrees(math.atan2(avg_y, avg_x))
            combined_angle = (combined_angle + 360) % 360
        else:
            combined_angle = best_estimate.orientation_angle
        
        # Boost confidence for multi-method agreement
        combined_confidence = min(1.0, best_estimate.confidence + 0.15)
        
        return OrientationEstimate(
            person_id=best_estimate.person_id,
            timestamp=timestamp,
            orientation_angle=combined_angle,
            confidence=combined_confidence,
            method='combined',
            facing_vector=(math.cos(math.radians(combined_angle)), 
                          math.sin(math.radians(combined_angle))),
            head_angle=best_estimate.head_angle,
            body_angle=best_estimate.body_angle,
            joint_visibility_count=best_estimate.joint_visibility_count,
            movement_magnitude=best_estimate.movement_magnitude
        )
    
    def _apply_temporal_smoothing(self, person_id: int, angle: float, 
                                confidence: float, timestamp: float) -> Tuple[float, float]:
        """Apply temporal smoothing to orientation estimates."""
        if person_id not in self.last_orientations:
            return angle, confidence
        
        last_estimate = self.last_orientations[person_id]
        time_diff = timestamp - last_estimate.timestamp
        
        # Skip smoothing if too much time has passed
        if time_diff > 2.0:  # 2 seconds
            return angle, confidence
        
        # Calculate angular difference (handling wrap-around)
        angle_diff = abs(angle - last_estimate.orientation_angle)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        
        # Apply smoothing if change is reasonable
        if angle_diff < 90:  # Reasonable change
            alpha = self.skeleton_config.get('smoothing_alpha', 0.7)
            
            # Smooth angle (handling wrap-around)
            if abs(angle - last_estimate.orientation_angle) <= 180:
                smoothed_angle = alpha * angle + (1 - alpha) * last_estimate.orientation_angle
            else:
                # Handle wrap-around case
                if angle > last_estimate.orientation_angle:
                    adjusted_last = last_estimate.orientation_angle + 360
                else:
                    adjusted_last = last_estimate.orientation_angle - 360
                smoothed_angle = alpha * angle + (1 - alpha) * adjusted_last
                
            smoothed_angle = (smoothed_angle + 360) % 360
            
            # Boost confidence for temporal consistency
            consistency_bonus = max(0, 0.2 - angle_diff / 180.0)
            smoothed_confidence = min(1.0, confidence + consistency_bonus)
            
            return smoothed_angle, smoothed_confidence
        
        return angle, confidence
    
    def _fallback_orientation(self, person, timestamp: float) -> Optional[OrientationEstimate]:
        """Provide fallback orientation when all methods fail."""
        person_id = person.id
        
        # Use last known orientation if available and recent
        if person_id in self.last_orientations:
            last_estimate = self.last_orientations[person_id]
            time_diff = timestamp - last_estimate.timestamp
            
            if time_diff < 60.0:  # Use last known if within 60 seconds
                # Decay confidence over time
                decayed_confidence = max(0.1, last_estimate.confidence * 0.5)
                
                return OrientationEstimate(
                    person_id=person_id,
                    timestamp=timestamp,
                    orientation_angle=last_estimate.orientation_angle,
                    confidence=decayed_confidence,
                    method='fallback_last_known',
                    facing_vector=last_estimate.facing_vector
                )
        
        # Default fallback - assume facing forward
        return OrientationEstimate(
            person_id=person_id,
            timestamp=timestamp,
            orientation_angle=0.0,
            confidence=0.1,
            method='fallback_default',
            facing_vector=(1.0, 0.0)
        )
    
    def _update_history(self, person_id: int, estimate: OrientationEstimate):
        """Update orientation history for a person."""
        if person_id not in self.orientation_history:
            self.orientation_history[person_id] = deque(maxlen=50)
        
        self.orientation_history[person_id].append(estimate)
        self.last_orientations[person_id] = estimate
    
    def analyze_mutual_orientations(self, orientations: List[OrientationEstimate],
                                  tracked_people: List) -> List[MutualOrientation]:
        """Analyze mutual orientations between pairs of people."""
        mutual_orientations = []
        
        # Create lookup for positions
        position_lookup = {}
        for person in tracked_people:
            if person.current_detection:
                position_lookup[person.id] = person.get_latest_position()
        
        # Analyze all pairs of orientations
        for i, orient1 in enumerate(orientations):
            for j, orient2 in enumerate(orientations[i+1:], i+1):
                if orient1.person_id == orient2.person_id:
                    continue
                
                pos1 = position_lookup.get(orient1.person_id)
                pos2 = position_lookup.get(orient2.person_id)
                
                if pos1 and pos2:
                    mutual = self._calculate_mutual_orientation(orient1, orient2, pos1, pos2)
                    if mutual:
                        mutual_orientations.append(mutual)
        
        return mutual_orientations
    
    def _calculate_mutual_orientation(self, orient1: OrientationEstimate, 
                                    orient2: OrientationEstimate,
                                    pos1: Tuple[float, float, float],
                                    pos2: Tuple[float, float, float]) -> Optional[MutualOrientation]:
        """Calculate mutual orientation metrics between two people."""
        
        # Calculate relative position vector
        rel_pos = np.array([pos2[0] - pos1[0], pos2[1] - pos1[1]])
        distance = np.linalg.norm(rel_pos)
        
        # Skip if too far apart
        if distance > 3.0:  # 3 meters max for interaction
            return None
        
        # Calculate angles each person would need to turn to face the other
        # Angle from person1's position to person2
        angle_to_person2 = math.degrees(math.atan2(rel_pos[1], rel_pos[0]))
        angle_to_person2 = (angle_to_person2 + 360) % 360
        
        # Angle from person2's position to person1
        angle_to_person1 = math.degrees(math.atan2(-rel_pos[1], -rel_pos[0]))
        angle_to_person1 = (angle_to_person1 + 360) % 360
        
        # Calculate turn angles
        turn_angle_1_to_2 = self._calculate_turn_angle(orient1.orientation_angle, angle_to_person2)
        turn_angle_2_to_1 = self._calculate_turn_angle(orient2.orientation_angle, angle_to_person1)
        
        # Calculate mutual facing score
        # Lower turn angles = higher score
        score1 = max(0, 1.0 - turn_angle_1_to_2 / 180.0)
        score2 = max(0, 1.0 - turn_angle_2_to_1 / 180.0)
        mutual_facing_score = (score1 + score2) / 2.0
        
        # Weight by orientation confidence
        confidence_weight = (orient1.confidence + orient2.confidence) / 2.0
        mutual_facing_score *= confidence_weight
        
        # Check for F-formation
        in_f_formation = self._detect_f_formation([orient1, orient2], [pos1, pos2])
        
        return MutualOrientation(
            person1_id=orient1.person_id,
            person2_id=orient2.person_id,
            timestamp=orient1.timestamp,
            person1_to_person2_angle=turn_angle_1_to_2,
            person2_to_person1_angle=turn_angle_2_to_1,
            mutual_facing_score=mutual_facing_score,
            in_f_formation=in_f_formation,
            group_coherence=mutual_facing_score if in_f_formation else 0.0
        )
    
    def _calculate_turn_angle(self, current_angle: float, target_angle: float) -> float:
        """Calculate the minimum turn angle from current to target orientation."""
        diff = target_angle - current_angle
        
        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        
        return abs(diff)
    
    def _detect_f_formation(self, orientations: List[OrientationEstimate], 
                          positions: List[Tuple[float, float, float]]) -> bool:
        """Detect if people are in F-formation (facing a common interaction space)."""
        if len(orientations) < 2:
            return False
        
        # For 2 people: check if they're facing each other or a common area
        if len(orientations) == 2:
            # Calculate if facing vectors intersect in reasonable space
            pos1 = np.array(positions[0][:2])
            pos2 = np.array(positions[1][:2])
            
            # Get facing vectors
            face1 = np.array(orientations[0].facing_vector)
            face2 = np.array(orientations[1].facing_vector)
            
            # Project facing vectors and see if they intersect
            # This is a simplified F-formation detection
            distance = np.linalg.norm(pos2 - pos1)
            
            # Check if people are close enough and roughly facing each other
            if distance < 2.0:  # Within 2 meters
                # Calculate dot product of facing vectors (should be negative for facing each other)
                facing_dot = np.dot(face1, face2)
                if facing_dot < -0.3:  # Somewhat facing each other
                    return True
        
        return False
    
    def get_orientation_statistics(self) -> Dict:
        """Get statistics about orientation estimation performance."""
        total_attempts = sum(self.method_success_counts.values())
        
        stats = {
            'total_estimates': self.total_estimates,
            'method_success_counts': self.method_success_counts.copy(),
            'method_success_rates': {},
            'active_people_tracked': len(self.orientation_history),
            'avg_confidence_by_method': {}
        }
        
        # Calculate success rates
        for method, count in self.method_success_counts.items():
            if total_attempts > 0:
                stats['method_success_rates'][method] = count / total_attempts
            else:
                stats['method_success_rates'][method] = 0.0
        
        # Calculate average confidence by method (simplified)
        for method in self.method_success_counts.keys():
            # This would ideally track confidence per method
            stats['avg_confidence_by_method'][method] = 0.7  # Placeholder
        
        return stats
    
    def reset(self):
        """Reset the orientation estimator state."""
        self.orientation_history.clear()
        self.velocity_history.clear()
        self.last_orientations.clear()
        self.method_success_counts = {'skeleton': 0, 'movement': 0, 'depth_gradient': 0}
        self.total_estimates = 0