"""
Phase 2 Step 2: Orientation Detection Test Runner
Tests body orientation estimation and mutual orientation analysis
"""

import sys
import os
import cv2
import numpy as np
import time
import logging
import json
import math
from pathlib import Path
from typing import List, Dict

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from camera.realsense_capture import RealSenseCapture
from camera.frame_processor import FrameProcessor
from detection.person_detector import PersonDetector
from detection.tracker import PersonTracker
from interaction.proximity_analyzer import ProximityAnalyzer
from interaction.orientation_estimator import OrientationEstimator
from interaction.orientation_visualizer import EnhancedOrientationVisualizer
from config.detection_config import PERSON_DETECTION, MOUNT_CONFIG, DETECTION_ZONES
from config.tracking_config import TRACKING_PARAMETERS
from config.proximity_config import CLASSROOM_PROXIMITY
from config.orientation_config import ORIENTATION_METHODS, SKELETON_ORIENTATION, MUTUAL_ORIENTATION

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

    
class SinglePanelLayout:
    """Single window layout with main video and side panels."""
    
    def __init__(self, screen_width=1920, screen_height=1080):
        """Initialize single panel layout."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Layout: Main video (70%) + Right panel (30%)
        self.main_video_width = int(screen_width * 0.70)
        self.panel_width = screen_width - self.main_video_width
        self.total_height = screen_height
        
        # Split right panel: Top stats + Bottom debug
        self.stats_height = int(self.total_height * 0.6)
        self.debug_height = self.total_height - self.stats_height
    
    def create_combined_view(self, vis_frame, orientations, mutual_orientations, stats, debug_data):
        """Create single combined view with main video + side panels."""
        
        # Create the full combined frame
        combined = np.zeros((self.total_height, self.screen_width, 3), dtype=np.uint8)
        combined[:] = (20, 20, 20)  # Dark background
        
        # 1. Main video (left side, 70% width)
        main_resized = cv2.resize(vis_frame, (self.main_video_width, self.total_height))
        combined[:, :self.main_video_width] = main_resized
        
        # 2. Statistics panel (top right)
        stats_panel = self.create_stats_panel(orientations, mutual_orientations, stats)
        combined[:self.stats_height, self.main_video_width:] = stats_panel
        
        # 3. Debug panel (bottom right)
        debug_panel = self.create_debug_panel(debug_data)
        combined[self.stats_height:, self.main_video_width:] = debug_panel
        
        # Add separating lines
        line_color = (80, 80, 80)
        # Vertical line between main video and panels
        cv2.line(combined, (self.main_video_width, 0), (self.main_video_width, self.total_height), line_color, 3)
        # Horizontal line between stats and debug panels
        cv2.line(combined, (self.main_video_width, self.stats_height), (self.screen_width, self.stats_height), line_color, 3)
        
        return combined
    
    def create_stats_panel(self, orientations, mutual_orientations, stats):
        """Create statistics panel."""
        panel = np.zeros((self.stats_height, self.panel_width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # Dark gray background
        
        # Add border
        cv2.rectangle(panel, (5, 5), (self.panel_width-5, self.stats_height-5), (60, 60, 60), 2)
        
        y = 30
        line_height = 22
        
        # Title
        cv2.putText(panel, "ORIENTATION ANALYSIS", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 45
        
        # Current status with icons
        cv2.putText(panel, "STATUS", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
        y += 30
        
        # Active people
        cv2.circle(panel, (25, y-5), 4, (0, 255, 0), -1)
        cv2.putText(panel, f"People Detected: {len(orientations)}", (40, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += line_height
        
        # Mutual orientations
        cv2.circle(panel, (25, y-5), 4, (255, 255, 0), -1)
        cv2.putText(panel, f"Mutual Orientations: {len(mutual_orientations)}", (40, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += line_height
        
        # F-formations
        f_formations = sum(1 for m in mutual_orientations if m.in_f_formation)
        cv2.circle(panel, (25, y-5), 4, (255, 0, 255), -1)
        cv2.putText(panel, f"F-Formations: {f_formations}", (40, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 35
        
        # Individual orientations
        cv2.putText(panel, "ORIENTATIONS", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
        y += 30
        
        method_colors = {
            'skeleton': (0, 255, 0),
            'movement': (255, 255, 0),
            'depth_gradient': (255, 100, 0),
            'combined': (0, 255, 255),
            'fallback_last_known': (150, 150, 150)
        }
        
        for i, orientation in enumerate(orientations[:6]):  # Show max 6
            color = method_colors.get(orientation.method, (255, 255, 255))
            
            # Method indicator
            cv2.circle(panel, (25, y-5), 4, color, -1)
            
            # Person info
            text = f"P{orientation.person_id}: {orientation.orientation_angle:.0f}°"
            cv2.putText(panel, text, (40, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y += 16
            
            # Confidence and method
            conf_text = f"  {orientation.method} (conf: {orientation.confidence:.2f})"
            cv2.putText(panel, conf_text, (40, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            y += 20
        
        # Method performance section
        y += 20
        cv2.putText(panel, "METHOD PERFORMANCE", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
        y += 30
        
        orientation_stats = stats.get('orientation_stats', {})
        method_counts = orientation_stats.get('method_success_counts', {})
        
        for method, count in method_counts.items():
            if count > 0:  # Only show methods with successes
                color = method_colors.get(method, (255, 255, 255))
                # Method indicator
                cv2.rectangle(panel, (20, y-8), (30, y-2), color, -1)
                cv2.putText(panel, f"{method.replace('_', ' ').title()}: {count}", (40, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y += line_height
        
        # Performance metrics
        y += 25
        cv2.putText(panel, "PERFORMANCE", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
        y += 30
        
        total_frames = stats.get('total_frames', 0)
        frames_with_orientations = stats.get('frames_with_orientations', 0)
        
        if total_frames > 0:
            success_rate = (frames_with_orientations / total_frames) * 100
            cv2.putText(panel, f"Success Rate: {success_rate:.1f}%", (25, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += line_height
        
        if 'processing_times' in stats and stats['processing_times']:
            avg_time = np.mean(stats['processing_times'][-10:])
            fps = 1.0 / max(avg_time, 0.001)
            cv2.putText(panel, f"FPS: {fps:.1f}", (25, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += line_height
        
        return panel
    
    def create_debug_panel(self, debug_data):
        """Create debug information panel."""
        panel = np.zeros((self.debug_height, self.panel_width, 3), dtype=np.uint8)
        panel[:] = (20, 20, 40)  # Dark blue background
        
        # Add border
        cv2.rectangle(panel, (5, 5), (self.panel_width-5, self.debug_height-5), (60, 60, 60), 2)
        
        y = 30
        line_height = 18
        
        # Title
        cv2.putText(panel, "DEBUG INFO", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 35
        
        # Legend first
        cv2.putText(panel, "LEGEND", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 25
        
        legend_items = [
            ("Green circles", "Skeleton keypoints", (0, 255, 0)),
            ("Green arrows", "Skeleton orientation", (0, 255, 0)),
            ("Yellow arrows", "Movement orientation", (255, 255, 0)),
            ("Red arrows", "Depth gradient", (255, 100, 0)),
            ("Cyan arrows", "Combined estimate", (0, 255, 255))
        ]
        
        for item, desc, color in legend_items:
            cv2.circle(panel, (25, y-5), 3, color, -1)
            cv2.putText(panel, desc, (35, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            y += 16
        
        y += 15
        
        # Show debug data for each person
        for person_id, person_debug in debug_data.items():
            if y > self.debug_height - 100:  # Prevent overflow
                break
                
            cv2.putText(panel, f"Person {person_id}:", (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 100), 1)
            y += 22
            
            # Skeleton debug info
            skeleton_data = person_debug.get('skeleton_data', {})
            if skeleton_data:
                visible_kpts = skeleton_data.get('visible_keypoints_count', 0)
                total_kpts = skeleton_data.get('total_keypoints', 17)
                cv2.putText(panel, f"  Keypoints: {visible_kpts}/{total_kpts}", (25, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                y += line_height
                
                # Key points status
                key_points = skeleton_data.get('key_points', {})
                if key_points:
                    available_points = [k for k, v in key_points.items() if v]
                    cv2.putText(panel, f"  Available: {', '.join(available_points)}", (25, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                    y += line_height
                
                issues = skeleton_data.get('issues', [])
                if issues:
                    cv2.putText(panel, f"  Issues: {', '.join(issues[:2])}", (25, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1)
                    y += line_height
            
            # Method attempts summary
            method_attempts = person_debug.get('method_attempts', {})
            successful_methods = [m for m, r in method_attempts.items() if r.get('success', False)]
            failed_methods = [m for m, r in method_attempts.items() if not r.get('success', False)]
            
            if successful_methods:
                cv2.putText(panel, f"  Working: {', '.join(successful_methods)}", (25, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                y += line_height
            
            if failed_methods:
                cv2.putText(panel, f"  Failed: {', '.join(failed_methods)}", (25, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1)
                y += line_height
            
            y += 10  # Space between people
        
        # Controls at bottom
        y = self.debug_height - 60
        cv2.putText(panel, "CONTROLS: D=Debug, S=Save, Q=Quit", (15, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        return panel

class OrientationVisualizer:
    """Enhanced visualizer for orientation detection and mutual analysis."""
    
    def __init__(self, config: dict):
        """Initialize visualizer with orientation configuration."""
        self.config = config
        self.viz_config = config.get('orientation_visualization', {})
        
    def visualize_orientations(self, frame: np.ndarray, tracked_people: List, 
                             orientations: List, mutual_orientations: List,
                             proximity_events: List, stats: Dict) -> np.ndarray:
        """Create comprehensive orientation and interaction visualization."""
        vis_frame = frame.copy()
        
        # Create lookup for orientations by person ID
        orientation_lookup = {o.person_id: o for o in orientations}
        
        # Draw tracked people with orientation vectors
        for person in tracked_people:
            if person.current_detection is None:
                continue
                
            self._draw_person_with_orientation(vis_frame, person, orientation_lookup.get(person.id))
        
        # Draw mutual orientations and F-formations
        for mutual in mutual_orientations:
            self._draw_mutual_orientation(vis_frame, mutual, tracked_people, orientation_lookup)
        
        # Draw proximity connections (dimmed)
        for proximity in proximity_events:
            self._draw_proximity_connection(vis_frame, proximity, tracked_people, alpha=0.3)
        
        # Draw statistics overlay
        self._draw_orientation_stats(vis_frame, stats, orientations, mutual_orientations)
        
        return vis_frame
    
    def _draw_person_with_orientation(self, frame: np.ndarray, person, orientation):
        """Draw person with orientation vector and confidence indicators."""
        bbox = person.current_detection.bounding_box
        x, y, w, h = bbox
        
        # Person center
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Draw bounding box with orientation-based coloring
        if orientation:
            # Color based on confidence and method
            method_colors = self.viz_config.get('vector_colors', {
                'skeleton': (0, 255, 0),
                'movement': (255, 255, 0), 
                'depth_gradient': (255, 0, 0),
                'combined': (0, 255, 255)
            })
            color = method_colors.get(orientation.method, (255, 255, 255))
            
            # Adjust color intensity based on confidence
            color = tuple(int(c * orientation.confidence) for c in color)
        else:
            color = (128, 128, 128)  # Gray for no orientation
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw person ID
        cv2.putText(frame, f"ID:{person.id}", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw orientation vector if available
        if orientation:
            self._draw_orientation_vector(frame, (center_x, center_y), orientation)
            
            # Draw confidence circle
            if self.viz_config.get('draw_confidence_circle', True):
                radius = int(self.viz_config.get('confidence_circle_radius', 30) * orientation.confidence)
                cv2.circle(frame, (center_x, center_y), radius, color, 1)
            
            # Show orientation details
            if self.viz_config.get('show_orientation_angle', True):
                angle_text = f"{orientation.orientation_angle:.0f}°"
                method_text = f"({orientation.method})"
                confidence_text = f"conf:{orientation.confidence:.2f}"
                
                cv2.putText(frame, angle_text, (x, y + h + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                cv2.putText(frame, method_text, (x, y + h + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                cv2.putText(frame, confidence_text, (x, y + h + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def _draw_orientation_vector(self, frame: np.ndarray, center: tuple, orientation):
        """Draw orientation vector from person center."""
        vector_length = self.viz_config.get('vector_length', 80)
        
        # Calculate end point
        angle_rad = math.radians(orientation.orientation_angle)
        end_x = int(center[0] + vector_length * math.cos(angle_rad))
        end_y = int(center[1] + vector_length * math.sin(angle_rad))
        
        # Get color based on method
        method_colors = self.viz_config.get('vector_colors', {})
        color = method_colors.get(orientation.method, (255, 255, 255))
        
        # Adjust thickness based on confidence
        thickness = max(1, int(3 * orientation.confidence))
        
        # Draw arrow
        cv2.arrowedLine(frame, center, (end_x, end_y), color, thickness, tipLength=0.3)
        
        # Draw method indicator at tip
        cv2.circle(frame, (end_x, end_y), 4, color, -1)
    
    def _draw_mutual_orientation(self, frame: np.ndarray, mutual, tracked_people: List, orientation_lookup: Dict):
        """Draw mutual orientation analysis between two people."""
        # Find the two people
        person1 = None
        person2 = None
        for person in tracked_people:
            if person.id == mutual.person1_id:
                person1 = person
            elif person.id == mutual.person2_id:
                person2 = person
        
        if not person1 or not person2 or not person1.current_detection or not person2.current_detection:
            return
        
        # Get centers
        bbox1 = person1.current_detection.bounding_box
        bbox2 = person2.current_detection.bounding_box
        center1 = (bbox1[0] + bbox1[2]//2, bbox1[1] + bbox1[3]//2)
        center2 = (bbox2[0] + bbox2[2]//2, bbox2[1] + bbox2[3]//2)
        
        # Draw mutual facing connection if significant
        if mutual.mutual_facing_score > 0.3:
            # Color intensity based on mutual facing score
            color = self.viz_config.get('mutual_attention_color', (255, 0, 255))
            alpha = mutual.mutual_facing_score
            color = tuple(int(c * alpha) for c in color)
            
            thickness = int(self.viz_config.get('facing_line_thickness', 3) * mutual.mutual_facing_score)
            cv2.line(frame, center1, center2, color, max(1, thickness))
            
            # Show mutual facing score at midpoint
            mid_x = (center1[0] + center2[0]) // 2
            mid_y = (center1[1] + center2[1]) // 2
            
            score_text = f"Mutual: {mutual.mutual_facing_score:.2f}"
            cv2.putText(frame, score_text, (mid_x - 40, mid_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw F-formation if detected
        if mutual.in_f_formation and mutual.o_space_center:
            center = (int(mutual.o_space_center[0] * 100 + 320), int(mutual.o_space_center[1] * 100 + 240))  # Rough conversion
            f_color = (0, 255, 255)  # Cyan for F-formation
            
            cv2.circle(frame, center, 25, f_color, 2)
            cv2.putText(frame, "F-Formation", (center[0] - 35, center[1] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, f_color, 1)
            cv2.putText(frame, f"Coherence: {mutual.group_coherence:.2f}", (center[0] - 40, center[1] + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, f_color, 1)
    
    def _draw_proximity_connection(self, frame: np.ndarray, proximity_event, tracked_people: List, alpha: float = 1.0):
        """Draw proximity connection (dimmed when showing orientations)."""
        # Find the two people
        person1 = None
        person2 = None
        for person in tracked_people:
            if person.id == proximity_event.person1_id:
                person1 = person
            elif person.id == proximity_event.person2_id:
                person2 = person
        
        if not person1 or not person2 or not person1.current_detection or not person2.current_detection:
            return
        
        # Get centers
        bbox1 = person1.current_detection.bounding_box
        bbox2 = person2.current_detection.bounding_box
        center1 = (bbox1[0] + bbox1[2]//2, bbox1[1] + bbox1[3]//2)
        center2 = (bbox2[0] + bbox2[2]//2, bbox2[1] + bbox2[3]//2)
        
        # Zone colors (dimmed)
        zone_colors = {
            'intimate': (255, 0, 0),
            'personal': (255, 165, 0),
            'social': (255, 255, 0),
            'public': (0, 255, 0)
        }
        
        color = zone_colors.get(proximity_event.zone_type, (128, 128, 128))
        color = tuple(int(c * alpha) for c in color)
        
        # Draw thin proximity line
        cv2.line(frame, center1, center2, color, 1)
    
    def _draw_orientation_stats(self, frame: np.ndarray, stats: Dict, orientations: List, mutual_orientations: List):
        """Draw orientation statistics overlay."""
        y_offset = 30
        
        # Title
        cv2.putText(frame, "ORIENTATION ANALYSIS", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # Current orientations
        cv2.putText(frame, f"People with Orientation: {len(orientations)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
        
        # Mutual orientations
        mutual_facing = sum(1 for m in mutual_orientations if m.mutual_facing_score > 0.5)
        cv2.putText(frame, f"Mutual Facing Pairs: {mutual_facing}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
        
        # F-formations
        f_formations = sum(1 for m in mutual_orientations if m.in_f_formation)
        cv2.putText(frame, f"F-Formations: {f_formations}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
        
        # Method breakdown
        method_counts = {}
        for orientation in orientations:
            method = orientation.method
            method_counts[method] = method_counts.get(method, 0) + 1
        
        cv2.putText(frame, "Methods Used:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 15
        
        for method, count in method_counts.items():
            method_color = {
                'skeleton': (0, 255, 0),
                'movement': (255, 255, 0),
                'depth_gradient': (255, 0, 0),
                'combined': (0, 255, 255)
            }.get(method, (255, 255, 255))
            
            cv2.putText(frame, f"  {method}: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, method_color, 1)
            y_offset += 15
        
        # Average confidence
        if orientations:
            avg_confidence = np.mean([o.confidence for o in orientations])
            cv2.putText(frame, f"Avg Confidence: {avg_confidence:.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def create_orientation_dashboard(orientations: List, mutual_orientations: List, stats: Dict) -> np.ndarray:
    """Create detailed dashboard for orientation analysis."""
    dashboard = np.zeros((400, 800, 3), dtype=np.uint8)
    dashboard[:] = (40, 40, 40)  # Dark background
    
    y = 30
    
    # Title
    cv2.putText(dashboard, "ORIENTATION DETECTION DASHBOARD", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 40
    
    # Current orientations summary
    cv2.putText(dashboard, f"Active Orientations: {len(orientations)}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 25
    
    # List individual orientations
    cv2.putText(dashboard, "Individual Orientations:", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 20
    
    for i, orientation in enumerate(orientations[:8]):  # Show max 8
        method_colors = {
            'skeleton': (0, 255, 0),
            'movement': (255, 255, 0),
            'depth_gradient': (255, 0, 0),
            'combined': (0, 255, 255)
        }
        color = method_colors.get(orientation.method, (255, 255, 255))
        
        orientation_text = (f"P{orientation.person_id}: {orientation.orientation_angle:.0f}° "
                          f"({orientation.method}, conf:{orientation.confidence:.2f})")
        
        cv2.putText(dashboard, orientation_text, (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 18
    
    # Mutual orientations section
    y += 20
    cv2.putText(dashboard, "Mutual Orientations:", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 20
    
    for i, mutual in enumerate(mutual_orientations[:6]):  # Show max 6
        if mutual.mutual_facing_score > 0.3:
            color = (255, 0, 255) if mutual.in_f_formation else (255, 255, 0)
            
            mutual_text = (f"P{mutual.person1_id}<->P{mutual.person2_id}: "
                         f"score={mutual.mutual_facing_score:.2f}")
            
            if mutual.in_f_formation:
                mutual_text += f" F-FORM(coh:{mutual.group_coherence:.2f})"
            
            cv2.putText(dashboard, mutual_text, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += 18
    
    # Statistics section
    y += 30
    cv2.putText(dashboard, "STATISTICS:", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 25
    
    # Method success rates
    orientation_stats = stats.get('orientation_stats', {})
    method_counts = orientation_stats.get('method_success_counts', {})
    
    for method, count in method_counts.items():
        cv2.putText(dashboard, f"{method.title()} successes: {count}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 18
    
    return dashboard

def run_phase2_step2_test():
    """Run Phase 2 Step 2: Orientation Detection test."""
    print("Phase 2 Step 2: Orientation Detection Implementation Test")
    print("=" * 60)
    
    # Configuration
    config = {
        'depth_width': 640,
        'depth_height': 480,
        'depth_fps': 30,
        'color_width': 640,
        'color_height': 480,
        'color_fps': 30,
        'align_streams': True,
        'median_filter_size': 5,
        'morph_kernel_size': (7, 7),
        'depth_range': (0.3, 20.0),
        'background_update_rate': 0.01,
        **PERSON_DETECTION,
        **MOUNT_CONFIG,
        'detection_zones': DETECTION_ZONES,
        **TRACKING_PARAMETERS,
        **CLASSROOM_PROXIMITY,
        'orientation_methods': ORIENTATION_METHODS,
        'skeleton_orientation': SKELETON_ORIENTATION,
        'mutual_orientation': MUTUAL_ORIENTATION,
        'confidence_scoring': {
            'temporal_consistency_bonus': 0.2,
            'multi_method_agreement_bonus': 0.15
        },
        'orientation_visualization': {
            'draw_orientation_vectors': True,
            'vector_length': 80,
            'vector_colors': {
                'skeleton': (0, 255, 0),
                'movement': (255, 255, 0),
                'depth_gradient': (255, 0, 0),
                'combined': (0, 255, 255)
            },
            'draw_confidence_circle': True,
            'confidence_circle_radius': 30,
            'draw_facing_connections': True,
            'facing_line_thickness': 3,
            'mutual_attention_color': (255, 0, 255),
            'show_orientation_angle': True
        }
    }
    
    # Initialize components
    logger.info("=== Testing Phase 2 Step 2: Enhanced Orientation Detection with Debug ===")
    
    camera = RealSenseCapture(config)
    processor = FrameProcessor(config)
    detector = PersonDetector(config)
    tracker = PersonTracker(config)
    proximity_analyzer = ProximityAnalyzer(config)
    orientation_estimator = OrientationEstimator(config)
    enhanced_visualizer = EnhancedOrientationVisualizer(config)
    fullscreen_layout = SimpleFullScreenLayout()
    
    if not (camera.configure_camera() and camera.start_streaming()):
        logger.error("Failed to initialize camera")
        return False
    
    logger.info("All components initialized successfully")
    logger.info("Enhanced debugging features enabled:")
    logger.info("  🔍 Skeleton keypoint visualization")
    logger.info("  📍 Movement trail and direction vectors") 
    logger.info("  📊 Depth gradient analysis visualization")
    logger.info("  🎯 Method-by-method breakdown")
    logger.info("  📈 Confidence scoring breakdown")
    logger.info("  🔄 Real-time method comparison")
    logger.info("")
    logger.info("Testing instructions:")
    logger.info("  - Face different directions to test skeleton detection")
    logger.info("  - Walk around to test movement-based orientation")
    logger.info("  - Stand at different distances for depth analysis")
    logger.info("  - Press 'd' to toggle debug modes")
    logger.info("  - Press 'c' to show method comparison view")
    
    # Create output directory
    output_dir = Path("data/test_sessions/phase2_step2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test parameters with debug modes
    test_duration = 180  # 3 minutes for thorough testing
    target_fps = 12  # Slightly lower for debug processing
    frame_interval = 1.0 / target_fps
    
    # Debug mode settings
    debug_modes = {
        'show_skeleton': True,
        'show_movement': True, 
        'show_depth': True,
        'show_method_comparison': False,
        'show_confidence_breakdown': True
    }
    
    # Statistics tracking
    stats = {
        'total_frames': 0,
        'frames_with_orientations': 0,
        'max_simultaneous_orientations': 0,
        'mutual_orientations_detected': 0,
        'f_formations_detected': 0,
        'processing_times': [],
        'orientation_stats': {},
    }
    
    start_time = time.time()
    last_frame_time = start_time
    
    try:
        while time.time() - start_time < test_duration:
            depth, color = camera.get_frames()
            if depth is None or color is None:
                continue
            
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_frame_time < frame_interval:
                continue
            
            # Process frames
            timestamp = current_time
            processed = processor.process_frame_pair(depth, color, timestamp)
            
            # Detection and tracking
            detection_start = time.time()
            detections = detector.detect_people(depth, timestamp, color)
            tracked_people = tracker.update(detections, timestamp)
            
            # Proximity analysis
            proximity_events = proximity_analyzer.analyze_frame(tracked_people, timestamp)
            
            # Orientation estimation with debug data
            orientations, debug_data = orientation_estimator.estimate_orientations(
                tracked_people, depth, color, timestamp
            )
            
            # Mutual orientation analysis
            mutual_orientations = orientation_estimator.analyze_mutual_orientations(
                orientations, tracked_people
            )
            
            processing_time = time.time() - detection_start
            
            # Update statistics
            stats['total_frames'] += 1
            stats['processing_times'].append(processing_time)
            
            if orientations:
                stats['frames_with_orientations'] += 1
                stats['max_simultaneous_orientations'] = max(
                    stats['max_simultaneous_orientations'], len(orientations)
                )
            
            if mutual_orientations:
                stats['mutual_orientations_detected'] += len(mutual_orientations)
                f_formations = sum(1 for m in mutual_orientations if m.in_f_formation)
                stats['f_formations_detected'] += f_formations
            
            # Get orientation statistics
            stats['orientation_stats'] = orientation_estimator.get_orientation_statistics()
            
            # Create enhanced debug visualizations
            if debug_modes['show_method_comparison'] and orientations:
                # Show method comparison for first person
                primary_person = orientations[0]
                person_debug = debug_data.get(primary_person.person_id, {})
                
                # Create comparison view
                comparison_view = enhanced_visualizer.create_method_comparison_view(
                    color, primary_person.person_id,
                    person_debug.get('skeleton_data', {}),
                    person_debug.get('movement_data', {}), 
                    person_debug.get('depth_data', {}),
                    person_debug.get('final_result', {})
                )
                
                cv2.imshow('Method Comparison', comparison_view)
            
            # Main debug visualization
            vis_frame = enhanced_visualizer.visualize_with_debug(
                color, tracked_people, orientations, debug_data
            )
            
            dashboard = create_orientation_dashboard(orientations, mutual_orientations, stats)
            
           # Create single panel layout
            if 'single_panel' not in locals():
                single_panel = SinglePanelLayout()

            combined_view = single_panel.create_combined_view(
            vis_frame, orientations, mutual_orientations, stats, debug_data
    )

            # Display single combined view
            cv2.imshow('Phase 2 Step 2: Orientation Detection', combined_view)      
            
            # Handle key press with debug controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                # Toggle debug modes
                debug_modes['show_skeleton'] = not debug_modes['show_skeleton']
                debug_modes['show_movement'] = not debug_modes['show_movement']
                debug_modes['show_depth'] = not debug_modes['show_depth']
                logger.info(f"Debug modes toggled: skeleton={debug_modes['show_skeleton']}, "
                           f"movement={debug_modes['show_movement']}, depth={debug_modes['show_depth']}")
            elif key == ord('c'):
                # Toggle method comparison view
                debug_modes['show_method_comparison'] = not debug_modes['show_method_comparison']
                logger.info(f"Method comparison view: {debug_modes['show_method_comparison']}")
                if not debug_modes['show_method_comparison']:
                    cv2.destroyWindow('Method Comparison')
            elif key == ord('s'):
                # Save current state with debug data
                timestamp_str = f"{timestamp:.3f}".replace('.', '_')
                cv2.imwrite(str(output_dir / f"orientation_debug_{timestamp_str}.png"), combined_vis)
                
                # Save detailed debug data
                debug_export = {
                    'timestamp': timestamp,
                    'orientations': [
                        {
                            'person_id': o.person_id,
                            'orientation_angle': o.orientation_angle,
                            'confidence': o.confidence,
                            'method': o.method,
                            'facing_vector': o.facing_vector,
                            'joint_visibility_count': o.joint_visibility_count,
                            'movement_magnitude': o.movement_magnitude,
                        }
                        for o in orientations
                    ],
                    'debug_data': debug_data,
                    'mutual_orientations': [
                        {
                            'person1_id': m.person1_id,
                            'person2_id': m.person2_id,
                            'mutual_facing_score': m.mutual_facing_score,
                            'in_f_formation': m.in_f_formation,
                            'group_coherence': m.group_coherence,
                        }
                        for m in mutual_orientations
                    ],
                    'statistics': stats
                }
                
                with open(output_dir / f"debug_data_{timestamp_str}.json", 'w') as f:
                    json.dump(debug_export, f, indent=2)
                
                logger.info(f"Saved detailed debug data at frame {stats['total_frames']}")
                
            elif key == ord('r'):
                # Reset analyzers
                orientation_estimator.reset()
                proximity_analyzer.reset()
                tracker.reset()
                logger.info("All analyzers reset")
            
            last_frame_time = current_time
            
            # Periodic logging with debug info
            if stats['total_frames'] % 50 == 0 and stats['total_frames'] > 0:
                active_orientations = len(orientations)
                mutual_facing = sum(1 for m in mutual_orientations if m.mutual_facing_score > 0.5)
                avg_processing_time = np.mean(stats['processing_times'][-50:])
                fps = 1.0 / max(avg_processing_time, 0.001)
                
                # Debug breakdown
                if orientations:
                    method_breakdown = {}
                    for o in orientations:
                        method_breakdown[o.method] = method_breakdown.get(o.method, 0) + 1
                    
                    method_str = ", ".join([f"{k}:{v}" for k, v in method_breakdown.items()])
                    logger.info(f"Frame {stats['total_frames']}: "
                               f"{active_orientations} orientations ({method_str}), "
                               f"{mutual_facing} mutual facing, FPS: {fps:.1f}")
                else:
                    logger.info(f"Frame {stats['total_frames']}: No orientations detected, FPS: {fps:.1f}")
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()
    
    # Enhanced final analysis
    final_orientation_stats = orientation_estimator.get_orientation_statistics()
    
    logger.info("=== Phase 2 Step 2 Enhanced Test Results ===")
    logger.info(f"Total frames processed: {stats['total_frames']}")
    logger.info(f"Frames with orientations: {stats['frames_with_orientations']}")
    logger.info(f"Max simultaneous orientations: {stats['max_simultaneous_orientations']}")
    logger.info(f"Total mutual orientations: {stats['mutual_orientations_detected']}")
    logger.info(f"Total F-formations detected: {stats['f_formations_detected']}")
    
    if stats['processing_times']:
        avg_processing_time = np.mean(stats['processing_times'])
        logger.info(f"Average processing time: {avg_processing_time:.3f}s")
        logger.info(f"Effective FPS: {1/avg_processing_time:.1f}")
    
    # Enhanced method analysis
    logger.info(f"\n=== Detailed Method Analysis ===")
    method_counts = final_orientation_stats.get('method_success_counts', {})
    total_attempts = sum(method_counts.values())
    
    for method, count in method_counts.items():
        percentage = (count / total_attempts * 100) if total_attempts > 0 else 0
        logger.info(f"{method.title()} method: {count} successes ({percentage:.1f}%)")
    
    avg_confidences = final_orientation_stats.get('avg_confidence_by_method', {})
    for method, confidence in avg_confidences.items():
        logger.info(f"{method.title()} average confidence: {confidence:.2f}")
    
    # Debug insights
    logger.info(f"\n=== Debug Insights ===")
    logger.info("Key findings from debug analysis:")
    
    if method_counts.get('skeleton', 0) > 0:
        logger.info("✓ Skeleton detection working - keypoints visible")
    else:
        logger.info("⚠️ Skeleton detection issues - check YOLO pose model")
    
    if method_counts.get('movement', 0) > 0:
        logger.info("✓ Movement-based orientation working")
    else:
        logger.info("ℹ️ Limited movement detected - people may be stationary")
    
    if method_counts.get('depth_gradient', 0) > 0:
        logger.info("✓ Depth gradient analysis working")
    else:
        logger.info("ℹ️ Depth gradients insufficient - may need closer positioning")
    
    # Enhanced validation criteria
    logger.info("\n=== Enhanced Phase 2 Step 2 Validation ===")
    
    success_criteria = {
        'orientation_detection_works': stats['frames_with_orientations'] > 0,
        'multiple_methods_work': len([c for c in method_counts.values() if c > 0]) >= 2,
        'skeleton_method_works': method_counts.get('skeleton', 0) > 0,
        'movement_method_works': method_counts.get('movement', 0) > 0,
        'mutual_orientation_works': stats['mutual_orientations_detected'] > 0,
        'performance_good': np.mean(stats['processing_times']) < 0.15 if stats['processing_times'] else False,
        'real_time_capable': (1/np.mean(stats['processing_times']) if stats['processing_times'] else 0) > 8,
        'confidence_reasonable': any(c > 0.5 for c in avg_confidences.values()) if avg_confidences else False,
        'method_diversity': len([c for c in method_counts.values() if c > 0]) >= 1,
    }
    
    logger.info(f"✓ Basic orientation detection: {'PASS' if success_criteria['orientation_detection_works'] else 'FAIL'}")
    logger.info(f"✓ Multiple methods working: {'PASS' if success_criteria['multiple_methods_work'] else 'FAIL'}")
    logger.info(f"✓ Skeleton method functional: {'PASS' if success_criteria['skeleton_method_works'] else 'FAIL'}")
    logger.info(f"✓ Movement method functional: {'PASS' if success_criteria['movement_method_works'] else 'FAIL'}")
    logger.info(f"✓ Mutual orientation analysis: {'PASS' if success_criteria['mutual_orientation_works'] else 'FAIL'}")
    logger.info(f"✓ Performance (>8 FPS): {'PASS' if success_criteria['real_time_capable'] else 'FAIL'}")
    logger.info(f"✓ Confidence levels reasonable: {'PASS' if success_criteria['confidence_reasonable'] else 'FAIL'}")
    
    overall_success = all(success_criteria.values())
    logger.info(f"\n🎯 PHASE 2 STEP 2 ENHANCED OVERALL: {'SUCCESS' if overall_success else 'NEEDS WORK'}")
    
    if overall_success:
        logger.info("✅ Enhanced orientation detection is working excellently!")
        logger.info("🔍 Debug features revealed detailed method performance")
        logger.info("Ready for Phase 2 Step 3: Combined Proximity + Orientation!")
        logger.info("\nNext steps:")
        logger.info("  - Combine proximity and orientation for robust interaction detection")
        logger.info("  - Implement temporal validation and interaction inference")  
        logger.info("  - Create comprehensive social network graphs")
    else:
        logger.info("⚠️ Debug-informed recommendations:")
        if not success_criteria['skeleton_method_works']:
            logger.info("  - Install/check YOLO pose model: pip install ultralytics")
            logger.info("  - Ensure good lighting for skeleton detection")
        if not success_criteria['movement_method_works']:
            logger.info("  - Test with more movement - walk around during testing")
        if not success_criteria['performance_good']:
            logger.info("  - Consider reducing debug visualization for better performance")
        if not success_criteria['confidence_reasonable']:
            logger.info("  - Adjust confidence thresholds in orientation_config.py")
    
    # Save enhanced report with debug data
    report = {
        'timestamp': time.time(),
        'test_duration': test_duration,
        'statistics': stats,
        'orientation_statistics': final_orientation_stats,
        'success_criteria': success_criteria,
        'overall_success': overall_success,
        'debug_modes_used': debug_modes,
        'method_breakdown': method_counts,
        'confidence_breakdown': avg_confidences,
    }
    
    with open(output_dir / "phase2_step2_enhanced_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nEnhanced report with debug data saved to: {output_dir / 'phase2_step2_enhanced_report.json'}")
    logger.info("🔍 Check saved debug JSON files for detailed method analysis")
    
    return overall_success

if __name__ == "__main__":
    success = run_phase2_step2_test()
    sys.exit(0 if success else 1)