#!/usr/bin/env python3
"""
Enhanced run_phase2_step2.py with skeleton point distance debugging
Shows all major skeleton points with their distances and debug info
"""

import sys
import os
import time
import json
import logging
import numpy as np
import cv2
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from camera.realsense_capture import RealSenseCapture
from camera.frame_processor import FrameProcessor
from detection.person_detector import PersonDetector
from detection.tracker import PersonTracker
from interaction.proximity_analyzer import ProximityAnalyzer
from interaction.orientation_estimator import OrientationEstimator

# Import configurations
sys.path.append(str(project_root / "config"))
from detection_config import PERSON_DETECTION, MOUNT_CONFIG, DETECTION_ZONES
from tracking_config import TRACKING_PARAMETERS
from proximity_config import CLASSROOM_PROXIMITY
from orientation_config import ORIENTATION_METHODS, SKELETON_ORIENTATION, MUTUAL_ORIENTATION

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DebugOrientationVisualizer:
    """Enhanced visualizer with skeleton point distance debugging."""
    
    def __init__(self, config):
        self.config = config
        self.debug_enabled = True
        
        # Skeleton keypoint names (YOLO pose format - 17 keypoints)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Define major keypoints for debugging
        self.major_keypoints = {
            'nose': 0, 'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8, 'left_hip': 11, 'right_hip': 12
        }
    
    def draw_skeleton_distances(self, frame, tracked_people, debug_data, depth_frame=None):
        """Draw skeleton points with distances on the side panel."""
        if not tracked_people:
            return frame
        
        # Create side panel for debugging info
        frame_height, frame_width = frame.shape[:2]
        panel_width = 400
        debug_frame = np.zeros((frame_height, frame_width + panel_width, 3), dtype=np.uint8)
        debug_frame[:, :frame_width] = frame
        
        # Panel background
        debug_frame[:, frame_width:] = (40, 40, 40)
        
        panel_x = frame_width + 10
        y_offset = 30
        
        # Title
        cv2.putText(debug_frame, "SKELETON DEBUG INFO", (panel_x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 40
        
        for person in tracked_people:
            person_debug = debug_data.get(person.id, {})
            skeleton_data = person_debug.get('skeleton_data', {})
            
            # Person header
            cv2.putText(debug_frame, f"=== PERSON {person.id} ===", (panel_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            
            # Show skeleton status
            keypoints = skeleton_data.get('keypoints', [])
            joint_count = skeleton_data.get('joint_count', 0)
            final_confidence = skeleton_data.get('final_confidence', 0.0)
            
            cv2.putText(debug_frame, f"Joints detected: {joint_count}/17", (panel_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(debug_frame, f"Final confidence: {final_confidence:.3f}", (panel_x, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            
            # Show issues if any
            issues = skeleton_data.get('issues', [])
            if issues:
                cv2.putText(debug_frame, "Issues:", (panel_x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                y_offset += 15
                for issue in issues[:3]:  # Show first 3 issues
                    cv2.putText(debug_frame, f"• {issue[:25]}", (panel_x + 10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1)
                    y_offset += 15
            
            # Show major keypoints with distances
            if keypoints:
                cv2.putText(debug_frame, "Major keypoints:", (panel_x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 20
                
                bbox = person.current_detection.bounding_box if person.current_detection else (0, 0, 100, 100)
                person_center = self._get_person_3d_position(person, depth_frame)
                
                for name, idx in self.major_keypoints.items():
                    if idx < len(keypoints):
                        kx, ky, conf = keypoints[idx]
                        
                        # Calculate distance if we have depth info
                        distance_str = "N/A"
                        if depth_frame is not None and conf > 0.1:
                            try:
                                # Convert to global coordinates
                                global_x = int(bbox[0] + kx)
                                global_y = int(bbox[1] + ky)
                                
                                if (0 <= global_x < depth_frame.shape[1] and 
                                    0 <= global_y < depth_frame.shape[0]):
                                    depth_value = depth_frame[global_y, global_x] / 1000.0  # Convert to meters
                                    if depth_value > 0:
                                        distance_str = f"{depth_value:.2f}m"
                            except:
                                distance_str = "Error"
                        
                        # Color based on confidence
                        if conf > 0.7:
                            color = (0, 255, 0)  # Green - high confidence
                        elif conf > 0.3:
                            color = (0, 255, 255)  # Yellow - medium confidence
                        elif conf > 0.1:
                            color = (100, 100, 255)  # Light red - low confidence
                        else:
                            color = (100, 100, 100)  # Gray - not detected
                        
                        text = f"{name}: {conf:.2f} ({distance_str})"
                        cv2.putText(debug_frame, text, (panel_x + 5, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                        y_offset += 15
                    else:
                        cv2.putText(debug_frame, f"{name}: Not detected", (panel_x + 5, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)
                        y_offset += 15
            
            # Show method attempts and why they failed
            method_attempts = person_debug.get('method_attempts', {})
            if method_attempts:
                y_offset += 10
                cv2.putText(debug_frame, "Method Results:", (panel_x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 20
                
                for method, result in method_attempts.items():
                    if isinstance(result, dict):
                        success = result.get('success', False)
                        confidence = result.get('confidence', 0.0)
                        reason = result.get('failure_reason', 'Unknown')
                        
                        status_color = (0, 255, 0) if success else (0, 0, 255)
                        status_text = "PASS" if success else "FAIL"
                        
                        cv2.putText(debug_frame, f"{method}: {status_text}", (panel_x + 5, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1)
                        y_offset += 15
                        
                        if not success and reason != 'Unknown':
                            cv2.putText(debug_frame, f"  {reason[:30]}", (panel_x + 10, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 100, 100), 1)
                            y_offset += 12
            
            y_offset += 30  # Space between people
        
        return debug_frame
    
    def _get_person_3d_position(self, person, depth_frame):
        """Get person's 3D position from depth frame."""
        if depth_frame is None or not person.current_detection:
            return None
        
        bbox = person.current_detection.bounding_box
        center_x = bbox[0] + bbox[2] // 2
        center_y = bbox[1] + bbox[3] // 2
        
        if (0 <= center_x < depth_frame.shape[1] and 
            0 <= center_y < depth_frame.shape[0]):
            depth_value = depth_frame[center_y, center_x] / 1000.0
            return (center_x, center_y, depth_value) if depth_value > 0 else None
        return None

def run_enhanced_phase2_step2_debug():
    """Run Phase 2 Step 2 with enhanced skeleton debugging."""
    
    print("Phase 2 Step 2: Enhanced Orientation Detection with Skeleton Distance Debug")
    print("=" * 80)
    
    # Configuration with debug settings
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
    }
    
    # Initialize components
    logger.info("Initializing components with debug mode enabled...")
    
    camera = RealSenseCapture(config)
    processor = FrameProcessor(config)
    detector = PersonDetector(config)
    tracker = PersonTracker(config)
    proximity_analyzer = ProximityAnalyzer(config)
    orientation_estimator = OrientationEstimator(config)
    debug_visualizer = DebugOrientationVisualizer(config)
    
    if not (camera.configure_camera() and camera.start_streaming()):
        logger.error("Failed to initialize camera")
        return False
    
    logger.info("All components initialized successfully")
    logger.info("Enhanced debugging enabled - skeleton points with distances will be shown")
    logger.info("Press 'q' to quit, 's' to save debug frame")
    
    # Create output directory
    output_dir = Path("data/debug_sessions/phase2_step2_skeleton")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistics tracking
    stats = {
        'total_frames': 0,
        'frames_with_orientations': 0,
        'skeleton_method_successes': 0,
        'movement_method_successes': 0,
        'depth_method_successes': 0,
        'processing_times': [],
    }
    
    try:
        while True:
            start_time = time.time()
            
            # Get frames
            depth, color = camera.get_frames()
            if depth is None or color is None:
                continue
            
            # Process frame
            timestamp = time.time()
            processed = processor.process_frame_pair(depth, color, timestamp)
            
            if processed is None:
                continue
                
            # Detect and track people
            detections = detector.detect_people(processed.depth_filtered, timestamp, processed.color_frame)
            tracked_people = tracker.update(detections, timestamp)
            
            # Estimate orientations with debug data
            orientations, debug_data = orientation_estimator.estimate_orientations(
                tracked_people, processed.depth_filtered, processed.color_frame, timestamp
            )
            
            # Count method successes
            for person_id, person_debug in debug_data.items():
                method_attempts = person_debug.get('method_attempts', {})
                if method_attempts.get('skeleton', {}).get('success', False):
                    stats['skeleton_method_successes'] += 1
                if method_attempts.get('movement', {}).get('success', False):
                    stats['movement_method_successes'] += 1
                if method_attempts.get('depth_gradient', {}).get('success', False):
                    stats['depth_method_successes'] += 1
            
            # Create debug visualization
            display_frame = debug_visualizer.draw_skeleton_distances(
                processed.color_frame, tracked_people, debug_data, processed.depth_filtered
            )
            
            # Add overall statistics
            cv2.putText(display_frame, f"Frame: {stats['total_frames']}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Orientations: {len(orientations)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Skeleton success: {stats['skeleton_method_successes']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Movement success: {stats['movement_method_successes']}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Phase 2 Step 2 - Skeleton Distance Debug', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save debug frame
                save_path = output_dir / f"debug_frame_{stats['total_frames']:04d}.jpg"
                cv2.imwrite(str(save_path), display_frame)
                logger.info(f"Saved debug frame: {save_path}")
            
            # Update statistics
            stats['total_frames'] += 1
            if orientations:
                stats['frames_with_orientations'] += 1
            
            processing_time = time.time() - start_time
            stats['processing_times'].append(processing_time)
            
            # Print periodic debug info
            if stats['total_frames'] % 30 == 0:  # Every 30 frames
                logger.info(f"Frame {stats['total_frames']}: "
                           f"Skeleton: {stats['skeleton_method_successes']}, "
                           f"Movement: {stats['movement_method_successes']}, "
                           f"Depth: {stats['depth_method_successes']}")
    
    except KeyboardInterrupt:
        logger.info("Stopping debug session...")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()
    
    # Final statistics
    logger.info("\n" + "=" * 60)
    logger.info("ENHANCED DEBUG SESSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total frames processed: {stats['total_frames']}")
    logger.info(f"Frames with orientations: {stats['frames_with_orientations']}")
    logger.info(f"Skeleton method successes: {stats['skeleton_method_successes']}")
    logger.info(f"Movement method successes: {stats['movement_method_successes']}")
    logger.info(f"Depth method successes: {stats['depth_method_successes']}")
    
    if stats['processing_times']:
        avg_time = np.mean(stats['processing_times'])
        logger.info(f"Average processing time: {avg_time:.3f}s ({1/avg_time:.1f} FPS)")
    
    # Save detailed statistics
    debug_report = {
        'timestamp': time.time(),
        'statistics': stats,
        'debug_session': True,
        'skeleton_distance_debugging': True
    }
    
    with open(output_dir / "enhanced_debug_report.json", 'w') as f:
        json.dump(debug_report, f, indent=2)
    
    logger.info(f"\nDetailed debug report saved to: {output_dir / 'enhanced_debug_report.json'}")
    
    return True

if __name__ == "__main__":
    success = run_enhanced_phase2_step2_debug()
    sys.exit(0 if success else 1)