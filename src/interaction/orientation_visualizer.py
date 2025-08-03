"""
Enhanced Orientation Visualizer with Debug Features
Provides comprehensive visual debugging for orientation detection methods
"""

import cv2
import numpy as np
import math
from typing import List, Dict, Tuple, Optional

class EnhancedOrientationVisualizer:
    """Enhanced visualizer with detailed debugging information for orientation detection."""
    
    def __init__(self, config: dict):
        """Initialize enhanced visualizer."""
        self.config = config
        self.viz_config = config.get('orientation_visualization', {})
        self.debug_config = config.get('debug_visualization', {
            'show_skeleton_keypoints': True,
            'show_movement_vectors': True,
            'show_depth_analysis': True,
            'show_method_breakdown': True,
            'show_confidence_breakdown': True,
            'keypoint_size': 3,
            'vector_thickness': 2,
            'text_size': 0.4
        })
        
        # Colors for different debug elements
        self.debug_colors = {
            'skeleton_keypoints': (0, 255, 0),
            'skeleton_connections': (0, 200, 0),
            'movement_trail': (255, 255, 0),
            'movement_vector': (255, 200, 0),
            'depth_roi': (255, 0, 0),
            'depth_gradient': (200, 0, 0),
            'confidence_high': (0, 255, 0),
            'confidence_medium': (255, 255, 0),
            'confidence_low': (255, 0, 0),
            'method_skeleton': (0, 255, 0),
            'method_movement': (255, 255, 0),
            'method_depth': (255, 0, 0),
            'method_combined': (0, 255, 255)
        }
    
    def visualize_with_debug(self, frame: np.ndarray, tracked_people: List, 
                           orientations: List, orientation_debug_data: Dict = None) -> np.ndarray:
        """Create comprehensive visualization with debug information."""
        debug_frame = frame.copy()
        
        # Create debug data lookup
        debug_lookup = orientation_debug_data or {}
        orientation_lookup = {o.person_id: o for o in orientations}
        
        for person in tracked_people:
            if person.current_detection is None:
                continue
            
            person_id = person.id
            orientation = orientation_lookup.get(person_id)
            person_debug = debug_lookup.get(person_id, {})
            
            # Draw person with all debug information
            self._draw_person_debug_complete(debug_frame, person, orientation, person_debug)
        
        # Add debug legend
        self._draw_debug_legend(debug_frame)
        
        return debug_frame
    
    def _draw_person_debug_complete(self, frame: np.ndarray, person, orientation, debug_data: Dict):
        """Draw complete debug information for a person."""
        bbox = person.current_detection.bounding_box
        x, y, w, h = bbox
        center_x, center_y = x + w // 2, y + h // 2
        
        # 1. Draw basic bounding box with person ID
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(frame, f"Person {person.id}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 2. Draw skeleton debugging if available
        if self.debug_config.get('show_skeleton_keypoints', True):
            self._draw_skeleton_debug(frame, debug_data.get('skeleton_data', {}), bbox)
        
        # 3. Draw movement debugging
        if self.debug_config.get('show_movement_vectors', True):
            self._draw_movement_debug(frame, person, debug_data.get('movement_data', {}), (center_x, center_y))
        
        # 4. Draw depth analysis debugging
        if self.debug_config.get('show_depth_analysis', True):
            self._draw_depth_debug(frame, debug_data.get('depth_data', {}), bbox)
        
        # 5. Draw final orientation result
        if orientation:
            self._draw_final_orientation(frame, orientation, (center_x, center_y))
        
        # 6. Draw method breakdown panel
        if self.debug_config.get('show_method_breakdown', True):
            self._draw_method_breakdown(frame, debug_data.get('method_attempts', {}), bbox)
    
    def _draw_skeleton_debug(self, frame: np.ndarray, skeleton_data: Dict, bbox: Tuple[int, int, int, int]):
        """Draw skeleton keypoint debugging information."""
        if not skeleton_data:
            return
        
        x, y, w, h = bbox
        keypoints = skeleton_data.get('keypoints', [])
        
        if not keypoints:
            # Show "No skeleton detected"
            cv2.putText(frame, "No Skeleton", (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            return
        
        # Draw detected keypoints
        keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        visible_keypoints = 0
        for i, (kx, ky, conf) in enumerate(keypoints):
            if conf > 0.3:  # Visible keypoint
                # Convert ROI coordinates to global coordinates
                global_x = int(x + kx)
                global_y = int(y + ky)
                
                # Color based on confidence
                if conf > 0.7:
                    color = self.debug_colors['confidence_high']
                elif conf > 0.5:
                    color = self.debug_colors['confidence_medium']
                else:
                    color = self.debug_colors['confidence_low']
                
                # Draw keypoint
                cv2.circle(frame, (global_x, global_y), 
                          self.debug_config.get('keypoint_size', 3), color, -1)
                
                # Label important keypoints
                if i in [0, 5, 6]:  # nose, left_shoulder, right_shoulder
                    cv2.putText(frame, keypoint_names[i][:3], (global_x + 5, global_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                
                visible_keypoints += 1
        
        # Draw shoulder line if both shoulders visible
        if (len(keypoints) > 6 and keypoints[5][2] > 0.3 and keypoints[6][2] > 0.3):
            left_shoulder = (int(x + keypoints[5][0]), int(y + keypoints[5][1]))
            right_shoulder = (int(x + keypoints[6][0]), int(y + keypoints[6][1]))
            
            cv2.line(frame, left_shoulder, right_shoulder, 
                    self.debug_colors['skeleton_connections'], 2)
            
            # Show shoulder angle
            shoulder_vec = np.array(right_shoulder) - np.array(left_shoulder)
            shoulder_angle = math.degrees(math.atan2(shoulder_vec[1], shoulder_vec[0]))
            
            mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                           (left_shoulder[1] + right_shoulder[1]) // 2)
            cv2.putText(frame, f"Shoulder: {shoulder_angle:.0f}°", 
                       (mid_shoulder[0] - 30, mid_shoulder[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.debug_colors['skeleton_connections'], 1)
        
        # Show detection status
        status_text = f"Skeleton: {visible_keypoints} pts"
        cv2.putText(frame, status_text, (x, y + h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.debug_colors['skeleton_keypoints'], 1)
        
        # Show body vs head orientation if calculated
        body_angle = skeleton_data.get('body_angle')
        head_angle = skeleton_data.get('head_angle')
        
        if body_angle is not None:
            cv2.putText(frame, f"Body: {body_angle:.0f}°", (x, y + h + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 100), 1)
        
        if head_angle is not None:
            cv2.putText(frame, f"Head: {head_angle:.0f}°", (x, y + h + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    
    def _draw_movement_debug(self, frame: np.ndarray, person, movement_data: Dict, center: Tuple[int, int]):
        """Draw movement analysis debugging."""
        # Draw recent position trail
        if hasattr(person, 'position_history') and len(person.position_history) > 1:
            positions = list(person.position_history)[-10:]  # Last 10 positions
            
            # Convert 3D positions to 2D screen coordinates (simplified)
            screen_positions = []
            for pos in positions:
                screen_x = int(320 + pos[0] * 100)  # Rough conversion
                screen_y = int(240 - pos[1] * 100)
                screen_x = max(0, min(frame.shape[1] - 1, screen_x))
                screen_y = max(0, min(frame.shape[0] - 1, screen_y))
                screen_positions.append((screen_x, screen_y))
            
            # Draw trail
            for i in range(1, len(screen_positions)):
                alpha = i / len(screen_positions)  # Fade older positions
                color = tuple(int(c * alpha) for c in self.debug_colors['movement_trail'])
                cv2.line(frame, screen_positions[i-1], screen_positions[i], color, 1)
        
        # Draw movement vector and statistics
        movement_magnitude = movement_data.get('magnitude', 0)
        movement_angle = movement_data.get('angle')
        movement_confidence = movement_data.get('confidence', 0)
        
        if movement_angle is not None and movement_magnitude > 0.05:
            # Draw movement direction arrow
            arrow_length = min(60, movement_magnitude * 300)  # Scale magnitude
            angle_rad = math.radians(movement_angle)
            
            end_x = int(center[0] + arrow_length * math.cos(angle_rad))
            end_y = int(center[1] + arrow_length * math.sin(angle_rad))
            
            cv2.arrowedLine(frame, center, (end_x, end_y), 
                           self.debug_colors['movement_vector'], 2, tipLength=0.3)
            
            # Show movement stats
            cv2.putText(frame, f"Move: {movement_angle:.0f}° ({movement_magnitude:.2f}m/s)", 
                       (center[0] - 50, center[1] + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.debug_colors['movement_vector'], 1)
        else:
            # Show stationary status
            cv2.putText(frame, "Stationary", (center[0] - 30, center[1] + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    def _draw_depth_debug(self, frame: np.ndarray, depth_data: Dict, bbox: Tuple[int, int, int, int]):
        """Draw depth analysis debugging."""
        if not depth_data:
            return
        
        x, y, w, h = bbox
        
        # Draw ROI boundary used for depth analysis
        roi_margin_x = w // 4
        roi_margin_y = h // 4
        roi_x = x + roi_margin_x
        roi_y = y + roi_margin_y
        roi_w = w - 2 * roi_margin_x
        roi_h = h - 2 * roi_margin_y
        
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), 
                     self.debug_colors['depth_roi'], 1)
        cv2.putText(frame, "Depth ROI", (roi_x, roi_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.debug_colors['depth_roi'], 1)
        
        # Show depth analysis results
        left_depth = depth_data.get('left_mean_depth', 0)
        right_depth = depth_data.get('right_mean_depth', 0)
        top_depth = depth_data.get('top_mean_depth', 0)
        bottom_depth = depth_data.get('bottom_mean_depth', 0)
        
        # Draw depth comparison bars
        if left_depth > 0 and right_depth > 0:
            # Normalize depths for visualization
            max_depth = max(left_depth, right_depth)
            if max_depth > 0:
                left_bar_height = int(30 * (left_depth / max_depth))
                right_bar_height = int(30 * (right_depth / max_depth))
                
                # Left depth bar
                cv2.rectangle(frame, (x - 15, y + h - left_bar_height), 
                             (x - 5, y + h), (255, 0, 0), -1)
                cv2.putText(frame, f"{left_depth:.1f}", (x - 25, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                
                # Right depth bar  
                cv2.rectangle(frame, (x + w + 5, y + h - right_bar_height), 
                             (x + w + 15, y + h), (0, 0, 255), -1)
                cv2.putText(frame, f"{right_depth:.1f}", (x + w + 5, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
                
                # Show which side is closer
                if abs(left_depth - right_depth) > 0.1:
                    closer_side = "L" if left_depth < right_depth else "R"
                    cv2.putText(frame, f"Closer: {closer_side}", (x, y + h + 65),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.debug_colors['depth_gradient'], 1)
        
        # Show depth gradient confidence
        depth_confidence = depth_data.get('confidence', 0)
        cv2.putText(frame, f"Depth conf: {depth_confidence:.2f}", (x, y + h + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.debug_colors['depth_gradient'], 1)
    
    def _draw_final_orientation(self, frame: np.ndarray, orientation, center: Tuple[int, int]):
        """Draw the final combined orientation result."""
        # Main orientation vector
        vector_length = 80
        angle_rad = math.radians(orientation.orientation_angle)
        end_x = int(center[0] + vector_length * math.cos(angle_rad))
        end_y = int(center[1] + vector_length * math.sin(angle_rad))
        
        # Color based on method
        method_color = self.debug_colors.get(f'method_{orientation.method}', (255, 255, 255))
        
        # Draw thick orientation arrow
        cv2.arrowedLine(frame, center, (end_x, end_y), method_color, 4, tipLength=0.3)
        
        # Draw confidence circle
        radius = int(40 * orientation.confidence)
        cv2.circle(frame, center, radius, method_color, 2)
        
        # Show final angle and confidence
        cv2.putText(frame, f"Final: {orientation.orientation_angle:.0f}°", 
                   (center[0] - 30, center[1] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, method_color, 2)
        cv2.putText(frame, f"Conf: {orientation.confidence:.2f} ({orientation.method})", 
                   (center[0] - 40, center[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, method_color, 1)
    
    def _draw_method_breakdown(self, frame: np.ndarray, method_attempts: Dict, bbox: Tuple[int, int, int, int]):
        """Draw breakdown of which methods were attempted and their results."""
        if not method_attempts:
            return
        
        x, y, w, h = bbox
        panel_x = x + w + 10
        panel_y = y
        
        # Method attempt status
        cv2.putText(frame, "Methods:", (panel_x, panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset = 15
        for method, result in method_attempts.items():
            success = result.get('success', False)
            confidence = result.get('confidence', 0)
            angle = result.get('angle')
            
            # Status indicator
            status_color = (0, 255, 0) if success else (128, 128, 128)
            status_text = "✓" if success else "✗"
            
            cv2.putText(frame, f"{status_text} {method}", (panel_x, panel_y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, status_color, 1)
            
            if success and angle is not None:
                cv2.putText(frame, f"  {angle:.0f}° ({confidence:.2f})", 
                           (panel_x + 10, panel_y + y_offset + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, status_color, 1)
                y_offset += 25
            else:
                y_offset += 15
    
    def _draw_debug_legend(self, frame: np.ndarray):
        """Draw legend explaining debug visualizations."""
        legend_x = frame.shape[1] - 250
        legend_y = 30
        
        # Background for legend
        cv2.rectangle(frame, (legend_x - 5, legend_y - 20), 
                     (frame.shape[1] - 5, legend_y + 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x - 5, legend_y - 20), 
                     (frame.shape[1] - 5, legend_y + 200), (255, 255, 255), 1)
        
        cv2.putText(frame, "DEBUG LEGEND", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        legend_items = [
            ("Green dots: Skeleton keypoints", self.debug_colors['skeleton_keypoints']),
            ("Green line: Shoulder connection", self.debug_colors['skeleton_connections']),
            ("Yellow trail: Movement history", self.debug_colors['movement_trail']),
            ("Yellow arrow: Movement direction", self.debug_colors['movement_vector']),
            ("Red box: Depth analysis ROI", self.debug_colors['depth_roi']),
            ("Bars: Left/Right depth values", self.debug_colors['depth_gradient']),
            ("Thick arrow: Final orientation", (255, 255, 255)),
            ("Circle: Confidence level", (255, 255, 255))
        ]
        
        y_pos = legend_y + 20
        for text, color in legend_items:
            cv2.putText(frame, text, (legend_x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            y_pos += 15
    
    def create_method_comparison_view(self, frame: np.ndarray, person_id: int, 
                                    skeleton_result: Dict, movement_result: Dict, 
                                    depth_result: Dict, final_result) -> np.ndarray:
        """Create side-by-side comparison of all orientation methods."""
        comparison_frame = np.zeros((300, 800, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(comparison_frame, f"Person {person_id} - Method Comparison", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create four columns: Skeleton, Movement, Depth, Final
        col_width = 180
        methods = [
            ("Skeleton", skeleton_result, self.debug_colors['method_skeleton']),
            ("Movement", movement_result, self.debug_colors['method_movement']),
            ("Depth", depth_result, self.debug_colors['method_depth']),
            ("Final", final_result, self.debug_colors['method_combined'])
        ]
        
        for i, (method_name, result, color) in enumerate(methods):
            x_offset = 20 + i * col_width
            
            # Method name
            cv2.putText(comparison_frame, method_name, (x_offset, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            if result and result.get('success', True):
                # Draw orientation circle and arrow
                center = (x_offset + 80, 150)
                cv2.circle(comparison_frame, center, 60, color, 2)
                
                # Orientation arrow
                if isinstance(result, dict):
                    angle = result.get('angle', 0)
                    confidence = result.get('confidence', 0)
                else:
                    angle = result.orientation_angle
                    confidence = result.confidence
                
                angle_rad = math.radians(angle)
                end_x = int(center[0] + 50 * math.cos(angle_rad))
                end_y = int(center[1] + 50 * math.sin(angle_rad))
                
                cv2.arrowedLine(comparison_frame, center, (end_x, end_y), color, 3, tipLength=0.3)
                
                # Angle text
                cv2.putText(comparison_frame, f"{angle:.0f}°", 
                           (x_offset + 60, 230),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(comparison_frame, f"Conf: {confidence:.2f}", 
                           (x_offset + 40, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            else:
                # Method failed
                cv2.putText(comparison_frame, "FAILED", (x_offset + 40, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        return comparison_frame