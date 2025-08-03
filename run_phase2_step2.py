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
    logger.info("=== Testing Phase 2 Step 2: Orientation Detection ===")
    
    camera = RealSenseCapture(config)
    processor = FrameProcessor(config)
    detector = PersonDetector(config)
    tracker = PersonTracker(config)
    proximity_analyzer = ProximityAnalyzer(config)
    orientation_estimator = OrientationEstimator(config)
    visualizer = OrientationVisualizer(config)
    
    if not (camera.configure_camera() and camera.start_streaming()):
        logger.error("Failed to initialize camera")
        return False
    
    logger.info("All components initialized successfully")
    logger.info("Testing orientation detection...")
    logger.info("Instructions for testing:")
    logger.info("  - Face different directions to test orientation detection")
    logger.info("  - Walk around to test movement-based orientation")
    logger.info("  - Stand facing each other to test mutual orientations")
    logger.info("  - Form small groups (2-3 people) to test F-formations")
    
    # Create output directory
    output_dir = Path("data/test_sessions/phase2_step2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test parameters
    test_duration = 150  # 2.5 minutes
    target_fps = 15
    frame_interval = 1.0 / target_fps
    
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
            
            # Orientation estimation
            orientations = orientation_estimator.estimate_orientations(
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
            
            # Create visualizations
            vis_frame = visualizer.visualize_orientations(
                color, tracked_people, orientations, mutual_orientations, proximity_events, stats
            )
            
            dashboard = create_orientation_dashboard(orientations, mutual_orientations, stats)
            
            # Combine visualizations
            combined_height = vis_frame.shape[0] + dashboard.shape[0]
            combined_width = max(vis_frame.shape[1], dashboard.shape[1])
            combined_vis = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            # Place main visualization
            combined_vis[:vis_frame.shape[0], :vis_frame.shape[1]] = vis_frame
            
            # Place dashboard
            dashboard_y = vis_frame.shape[0]
            combined_vis[dashboard_y:dashboard_y + dashboard.shape[0], :dashboard.shape[1]] = dashboard
            
            # Display
            cv2.imshow('Phase 2 Step 2: Orientation Detection', combined_vis)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current state
                timestamp_str = f"{timestamp:.3f}".replace('.', '_')
                cv2.imwrite(str(output_dir / f"orientation_{timestamp_str}.png"), combined_vis)
                
                # Save orientation data
                orientation_data = {
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
                    'mutual_orientations': [
                        {
                            'person1_id': m.person1_id,
                            'person2_id': m.person2_id,
                            'mutual_facing_score': m.mutual_facing_score,
                            'in_f_formation': m.in_f_formation,
                            'group_coherence': m.group_coherence,
                            'person1_to_person2_angle': m.person1_to_person2_angle,
                            'person2_to_person1_angle': m.person2_to_person1_angle,
                        }
                        for m in mutual_orientations
                    ],
                    'statistics': stats
                }
                
                with open(output_dir / f"orientation_data_{timestamp_str}.json", 'w') as f:
                    json.dump(orientation_data, f, indent=2)
                
                logger.info(f"Saved orientation analysis at frame {stats['total_frames']}")
                
            elif key == ord('r'):
                # Reset analyzers
                orientation_estimator.reset()
                proximity_analyzer.reset()
                tracker.reset()
                logger.info("All analyzers reset")
            
            last_frame_time = current_time
            
            # Periodic logging
            if stats['total_frames'] % 50 == 0 and stats['total_frames'] > 0:
                active_orientations = len(orientations)
                mutual_facing = sum(1 for m in mutual_orientations if m.mutual_facing_score > 0.5)
                avg_processing_time = np.mean(stats['processing_times'][-50:])
                fps = 1.0 / max(avg_processing_time, 0.001)
                
                logger.info(f"Frame {stats['total_frames']}: "
                           f"{active_orientations} orientations, "
                           f"{mutual_facing} mutual facing pairs, "
                           f"FPS: {fps:.1f}")
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()
    
    # Final analysis and validation
    final_orientation_stats = orientation_estimator.get_orientation_statistics()
    
    logger.info("=== Phase 2 Step 2 Test Results ===")
    logger.info(f"Total frames processed: {stats['total_frames']}")
    logger.info(f"Frames with orientations: {stats['frames_with_orientations']}")
    logger.info(f"Max simultaneous orientations: {stats['max_simultaneous_orientations']}")
    logger.info(f"Total mutual orientations: {stats['mutual_orientations_detected']}")
    logger.info(f"Total F-formations detected: {stats['f_formations_detected']}")
    
    if stats['processing_times']:
        avg_processing_time = np.mean(stats['processing_times'])
        logger.info(f"Average processing time: {avg_processing_time:.3f}s")
        logger.info(f"Effective FPS: {1/avg_processing_time:.1f}")
    
    # Orientation-specific results
    logger.info(f"\n=== Orientation Detection Results ===")
    method_counts = final_orientation_stats.get('method_success_counts', {})
    for method, count in method_counts.items():
        logger.info(f"{method.title()} method successes: {count}")
    
    avg_confidences = final_orientation_stats.get('avg_confidence_by_method', {})
    for method, confidence in avg_confidences.items():
        logger.info(f"{method.title()} average confidence: {confidence:.2f}")
    
    # Validation criteria
    logger.info("\n=== Phase 2 Step 2 Validation ===")
    
    success_criteria = {
        'orientation_detection_works': stats['frames_with_orientations'] > 0,
        'multiple_methods_work': len([c for c in method_counts.values() if c > 0]) >= 2,
        'mutual_orientation_works': stats['mutual_orientations_detected'] > 0,
        'performance_good': np.mean(stats['processing_times']) < 0.15 if stats['processing_times'] else False,
        'real_time_capable': (1/np.mean(stats['processing_times']) if stats['processing_times'] else 0) > 8,
        'confidence_reasonable': any(c > 0.5 for c in avg_confidences.values()) if avg_confidences else False,
    }
    
    logger.info(f"✓ Orientation detection functionality: {'PASS' if success_criteria['orientation_detection_works'] else 'FAIL'}")
    logger.info(f"✓ Multiple methods working: {'PASS' if success_criteria['multiple_methods_work'] else 'FAIL'}")
    logger.info(f"✓ Mutual orientation analysis: {'PASS' if success_criteria['mutual_orientation_works'] else 'FAIL'}")
    logger.info(f"✓ Performance (>8 FPS): {'PASS' if success_criteria['real_time_capable'] else 'FAIL'}")
    logger.info(f"✓ Confidence levels reasonable: {'PASS' if success_criteria['confidence_reasonable'] else 'FAIL'}")
    
    overall_success = all(success_criteria.values())
    logger.info(f"\n🎯 PHASE 2 STEP 2 OVERALL: {'SUCCESS' if overall_success else 'NEEDS WORK'}")
    
    if overall_success:
        logger.info("✅ Ready for Phase 2 Step 3: Combined Proximity + Orientation Interaction Detection!")
        logger.info("Next steps:")
        logger.info("  - Combine proximity and orientation for robust interaction detection")
        logger.info("  - Implement temporal validation and interaction inference")
        logger.info("  - Create comprehensive social network graphs")
    else:
        logger.info("⚠️ Recommendations:")
        if not success_criteria['orientation_detection_works']:
            logger.info("  - Check YOLO pose model installation")
            logger.info("  - Verify depth frame processing")
        if not success_criteria['multiple_methods_work']:
            logger.info("  - Tune method parameters and thresholds")
        if not success_criteria['performance_good']:
            logger.info("  - Optimize orientation calculation algorithms")
        if not success_criteria['confidence_reasonable']:
            logger.info("  - Adjust confidence scoring parameters")
    
    # Save final report
    report = {
        'timestamp': time.time(),
        'test_duration': test_duration,
        'statistics': stats,
        'orientation_statistics': final_orientation_stats,
        'success_criteria': success_criteria,
        'overall_success': overall_success,
    }
    
    with open(output_dir / "phase2_step2_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nDetailed report saved to: {output_dir / 'phase2_step2_report.json'}")
    
    return overall_success

if __name__ == "__main__":
    success = run_phase2_step2_test()
    sys.exit(0 if success else 1)