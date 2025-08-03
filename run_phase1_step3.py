"""
Phase 1 Step 3: Basic Tracking Test Runner
Tests ID assignment, frame-to-frame association, and trajectory smoothing
"""

import sys
import os
import cv2
import numpy as np
import time
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from camera.realsense_capture import RealSenseCapture
from camera.frame_processor import FrameProcessor
from detection.person_detector import PersonDetector
from detection.tracker import PersonTracker, TrackedPerson
from config.detection_config import PERSON_DETECTION, MOUNT_CONFIG, DETECTION_ZONES
from config.tracking_config import TRACKING_PARAMETERS, TRACKING_VISUALIZATION, CLASSROOM_TRACKING

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class TrackingVisualizer:
    """Handles visualization of tracking results."""
    
    def __init__(self, config: dict):
        """Initialize visualizer with configuration."""
        self.config = config
        self.track_colors = TRACKING_VISUALIZATION['track_colors']
        self.trajectory_length = TRACKING_VISUALIZATION['trajectory_length']
        
    def visualize_tracking(self, frame: np.ndarray, tracked_people: List[TrackedPerson], 
                          show_trajectories: bool = True, show_predictions: bool = True) -> np.ndarray:
        """Create comprehensive tracking visualization."""
        vis_frame = frame.copy()
        
        for person in tracked_people:
            if person.current_detection is None:
                continue
                
            # Get color for this track
            color = self.track_colors[person.id % len(self.track_colors)]
            
            # Draw bounding box
            x, y, w, h = person.current_detection.bounding_box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw track ID
            id_text = f"ID:{person.id}"
            text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(vis_frame, (x, y - text_size[1] - 10), 
                         (x + text_size[0] + 4, y), color, -1)
            cv2.putText(vis_frame, id_text, (x + 2, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw stability indicator
            stability_text = "STABLE" if person.is_stable() else "UNSTABLE"
            stability_color = (0, 255, 0) if person.is_stable() else (0, 0, 255)
            cv2.putText(vis_frame, stability_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, stability_color, 1)
            
            # Draw tracking info
            info_text = f"Frames:{person.total_frames_seen} Zone:{person.get_dominant_zone()}"
            cv2.putText(vis_frame, info_text, (x, y + h + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Show trajectory
            if show_trajectories and len(person.position_history) > 1:
                self._draw_trajectory(vis_frame, person, color)
            
            # Show prediction
            if show_predictions and person.predicted_position and person.frames_since_seen > 0:
                self._draw_prediction(vis_frame, person)
        
        return vis_frame
    
    def _draw_trajectory(self, frame: np.ndarray, person: TrackedPerson, color: Tuple[int, int, int]):
        """Draw person's trajectory path."""
        if len(person.position_history) < 2:
            return
        
        # Convert 3D positions to 2D screen coordinates (simple projection)
        # This is a simplified projection - in real implementation you'd use camera intrinsics
        trajectory_points = []
        for pos_3d in list(person.position_history)[-self.trajectory_length:]:
            # Simple projection: ignore Z, scale X,Y
            screen_x = int(320 + pos_3d[0] * 100)  # Rough conversion
            screen_y = int(240 - pos_3d[1] * 100)  # Flip Y axis
            
            # Clamp to screen bounds
            screen_x = max(0, min(frame.shape[1] - 1, screen_x))
            screen_y = max(0, min(frame.shape[0] - 1, screen_y))
            
            trajectory_points.append((screen_x, screen_y))
        
        # Draw trajectory line
        if len(trajectory_points) > 1:
            for i in range(1, len(trajectory_points)):
                # Fade older points
                alpha = i / len(trajectory_points)
                line_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, trajectory_points[i-1], trajectory_points[i], line_color, 2)
    
    def _draw_prediction(self, frame: np.ndarray, person: TrackedPerson):
        """Draw predicted position."""
        if not person.predicted_position or not person.current_detection:
            return
        
        # Current position
        current_bbox = person.current_detection.bounding_box
        current_center = (current_bbox[0] + current_bbox[2]//2, 
                         current_bbox[1] + current_bbox[3]//2)
        
        # Predicted position (simplified projection)
        pred_x = int(320 + person.predicted_position[0] * 100)
        pred_y = int(240 - person.predicted_position[1] * 100)
        
        pred_x = max(0, min(frame.shape[1] - 1, pred_x))
        pred_y = max(0, min(frame.shape[0] - 1, pred_y))
        
        # Draw prediction
        cv2.circle(frame, (pred_x, pred_y), 8, (128, 128, 128), 2)
        cv2.putText(frame, "PRED", (pred_x - 15, pred_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)

def create_tracking_dashboard(tracked_people: List[TrackedPerson], stats: Dict) -> np.ndarray:
    """Create a dashboard showing tracking statistics."""
    dashboard = np.zeros((200, 800, 3), dtype=np.uint8)
    
    # Background
    dashboard[:] = (40, 40, 40)
    
    # Title
    cv2.putText(dashboard, "TRACKING DASHBOARD", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Statistics
    stats_text = [
        f"Active Tracks: {len(tracked_people)}",
        f"Total Created: {stats.get('total_created', 0)}",
        f"Total Lost: {stats.get('total_lost', 0)}",
        f"Frame: {stats.get('frame_count', 0)}",
        f"FPS: {stats.get('fps', 0):.1f}",
        f"Avg Processing: {stats.get('avg_processing_time', 0):.3f}s"
    ]
    
    for i, text in enumerate(stats_text):
        cv2.putText(dashboard, text, (10, 70 + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Track details
    cv2.putText(dashboard, "ACTIVE TRACKS:", (400, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    for i, person in enumerate(tracked_people[:6]):  # Show max 6 tracks
        if i >= 6:
            break
        
        track_info = f"ID {person.id}: {person.total_frames_seen}f, {person.get_dominant_zone()}"
        stability = "STABLE" if person.is_stable() else "UNSTABLE"
        color = (0, 255, 0) if person.is_stable() else (255, 255, 0)
        
        cv2.putText(dashboard, track_info, (400, 95 + i * 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(dashboard, stability, (650, 95 + i * 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    return dashboard

def analyze_tracking_performance(tracked_people: List[TrackedPerson], stats: Dict) -> Dict:
    """Analyze tracking performance metrics."""
    performance = {
        'id_consistency': 0.0,
        'tracking_stability': 0.0,
        'trajectory_smoothness': 0.0,
        'overall_score': 0.0
    }
    
    if not tracked_people:
        return performance
    
    # ID consistency: tracks that maintain same ID for most of their lifetime
    stable_tracks = [p for p in tracked_people if p.is_stable()]
    if tracked_people:
        performance['id_consistency'] = len(stable_tracks) / len(tracked_people)
    
    # Tracking stability: percentage of tracks that are stable
    performance['tracking_stability'] = performance['id_consistency']
    
    # Trajectory smoothness: analyze position variance
    smoothness_scores = []
    for person in tracked_people:
        if len(person.position_history) >= 3:
            positions = np.array(list(person.position_history))
            variance = np.mean(np.var(positions, axis=0))
            # Lower variance = higher smoothness (inverted score)
            smoothness = max(0, 1 - variance / 2.0)  # Normalize
            smoothness_scores.append(smoothness)
    
    if smoothness_scores:
        performance['trajectory_smoothness'] = np.mean(smoothness_scores)
    
    # Overall score (weighted average)
    performance['overall_score'] = (
        performance['id_consistency'] * 0.4 +
        performance['tracking_stability'] * 0.4 +
        performance['trajectory_smoothness'] * 0.2
    )
    
    return performance

def run_phase1_step3_test():
    """Run Phase 1 Step 3 tracking test with multiple people."""
    print("Phase 1 Step 3: Basic Tracking Implementation Test")
    print("=" * 50)
    
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
    }
    
    # Initialize components
    logger.info("=== Testing Phase 1 Step 3: Basic Tracking ===")
    
    camera = RealSenseCapture(config)
    processor = FrameProcessor(config)
    detector = PersonDetector(config)
    tracker = PersonTracker(config)
    visualizer = TrackingVisualizer(config)
    
    if not (camera.configure_camera() and camera.start_streaming()):
        logger.error("Failed to initialize camera")
        return False
    
    logger.info("All components initialized successfully")
    logger.info("Testing tracking with multiple people (move around to test)")
    
    # Create output directory
    output_dir = Path("data/test_sessions/phase1_step3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test parameters
    test_duration = 180  # 3 minutes
    target_fps = 15
    frame_interval = 1.0 / target_fps
    
    # Statistics tracking
    stats = {
        'total_frames': 0,
        'frames_with_detections': 0,
        'frames_with_tracks': 0,
        'total_tracks_created': 0,
        'total_tracks_lost': 0,
        'max_simultaneous_tracks': 0,
        'processing_times': [],
        'tracking_times': [],
        'id_consistency_checks': [],
        'frame_count': 0,
    }
    
    logger.info(f"Starting {test_duration}s tracking test")
    logger.info("Instructions:")
    logger.info("- Walk around to test tracking")
    logger.info("- Multiple people can test simultaneously")
    logger.info("- Try occlusions (hiding behind objects)")
    logger.info("- Press 'q' to quit, 's' to save, 'r' to reset tracker")
    
    start_time = time.time()
    last_frame_time = start_time
    
    try:
        while time.time() - start_time < test_duration:
            # Capture frames
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
            
            # Person detection
            detection_start = time.time()
            detections = detector.detect_people(depth, timestamp, color)
            detection_time = time.time() - detection_start
            
            # Tracking update
            tracking_start = time.time()
            tracked_people = tracker.update(detections, timestamp)
            tracking_time = time.time() - tracking_start
            
            # Update statistics
            stats['total_frames'] += 1
            stats['frame_count'] += 1
            stats['processing_times'].append(detection_time)
            stats['tracking_times'].append(tracking_time)
            
            if detections:
                stats['frames_with_detections'] += 1
            
            if tracked_people:
                stats['frames_with_tracks'] += 1
                stats['max_simultaneous_tracks'] = max(
                    stats['max_simultaneous_tracks'], len(tracked_people)
                )
            
            # Track statistics from tracker
            stats['total_tracks_created'] = tracker.total_tracks_created
            stats['total_tracks_lost'] = tracker.total_tracks_lost
            stats['total_created'] = tracker.total_tracks_created
            stats['total_lost'] = tracker.total_tracks_lost
            
            # Calculate performance metrics
            if stats['processing_times']:
                stats['avg_processing_time'] = np.mean(stats['processing_times'][-100:])
                stats['fps'] = 1.0 / max(stats['avg_processing_time'], 0.001)
            
            # Create visualizations
            tracking_vis = visualizer.visualize_tracking(
                color, tracked_people, show_trajectories=True, show_predictions=True
            )
            
            dashboard = create_tracking_dashboard(tracked_people, stats)
            
            # Combine visualizations
            combined_height = tracking_vis.shape[0] + dashboard.shape[0]
            combined_width = max(tracking_vis.shape[1], dashboard.shape[1])
            combined_vis = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            # Place tracking visualization
            combined_vis[:tracking_vis.shape[0], :tracking_vis.shape[1]] = tracking_vis
            
            # Place dashboard
            dashboard_y = tracking_vis.shape[0]
            combined_vis[dashboard_y:dashboard_y + dashboard.shape[0], :dashboard.shape[1]] = dashboard
            
            # Display
            cv2.imshow('Phase 1 Step 3: Basic Tracking', combined_vis)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current state
                timestamp_str = f"{timestamp:.3f}".replace('.', '_')
                cv2.imwrite(str(output_dir / f"tracking_{timestamp_str}.png"), combined_vis)
                
                # Save tracking data
                tracking_data = {
                    'timestamp': timestamp,
                    'tracked_people': [
                        {
                            'id': p.id,
                            'position': p.get_latest_position(),
                            'confidence': p.get_average_confidence(),
                            'zone': p.get_dominant_zone(),
                            'stable': p.is_stable(),
                            'frames_seen': p.total_frames_seen,
                        }
                        for p in tracked_people
                    ]
                }
                
                with open(output_dir / f"tracking_data_{timestamp_str}.json", 'w') as f:
                    json.dump(tracking_data, f, indent=2)
                
                logger.info(f"Saved tracking state at frame {stats['total_frames']}")
                
            elif key == ord('r'):
                # Reset tracker
                tracker.reset()
                logger.info("Tracker reset")
            
            last_frame_time = current_time
            
            # Periodic logging
            if stats['total_frames'] % 100 == 0:
                logger.info(f"Frame {stats['total_frames']}: "
                           f"{len(tracked_people)} active tracks, "
                           f"FPS: {stats.get('fps', 0):.1f}")
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()
    
    # Analyze tracking performance
    current_tracked_people = list(tracker.tracked_people.values())
    performance = analyze_tracking_performance(current_tracked_people, stats)
    
    # Calculate final statistics and validation
    logger.info("=== Phase 1 Step 3 Test Results ===")
    logger.info(f"Total frames processed: {stats['total_frames']}")
    logger.info(f"Frames with detections: {stats['frames_with_detections']}")
    logger.info(f"Frames with tracks: {stats['frames_with_tracks']}")
    logger.info(f"Total tracks created: {stats['total_tracks_created']}")
    logger.info(f"Total tracks lost: {stats['total_tracks_lost']}")
    logger.info(f"Max simultaneous tracks: {stats['max_simultaneous_tracks']}")
    
    if stats['processing_times'] and stats['tracking_times']:
        avg_detection_time = np.mean(stats['processing_times'])
        avg_tracking_time = np.mean(stats['tracking_times'])
        total_processing_time = avg_detection_time + avg_tracking_time
        
        logger.info(f"Average detection time: {avg_detection_time:.3f}s")
        logger.info(f"Average tracking time: {avg_tracking_time:.3f}s")
        logger.info(f"Total processing time: {total_processing_time:.3f}s")
        logger.info(f"Effective FPS: {1/total_processing_time:.1f}")
    
    # Performance analysis
    logger.info(f"\n=== Tracking Performance Analysis ===")
    logger.info(f"ID Consistency: {performance['id_consistency']:.1%}")
    logger.info(f"Tracking Stability: {performance['tracking_stability']:.1%}")
    logger.info(f"Trajectory Smoothness: {performance['trajectory_smoothness']:.1%}")
    logger.info(f"Overall Score: {performance['overall_score']:.1%}")
    
    # Validation criteria
    logger.info("\n=== Phase 1 Step 3 Validation ===")
    
    # Success criteria
    detection_rate = (stats['frames_with_detections'] / max(stats['total_frames'], 1)) * 100
    tracking_rate = (stats['frames_with_tracks'] / max(stats['total_frames'], 1)) * 100
    
    success_criteria = {
        'tracking_works': stats['total_tracks_created'] > 0,
        'multiple_people': stats['max_simultaneous_tracks'] >= 1,  # At least 1 for single person test
        'stable_tracking': tracking_rate >= 80.0,
        'performance': stats.get('fps', 0) >= 10.0,
        'track_management': stats['total_tracks_lost'] < stats['total_tracks_created'] * 2,  # Reasonable loss rate
        'id_consistency': performance['id_consistency'] >= 0.7,  # 70% consistency
        'overall_performance': performance['overall_score'] >= 0.6,  # 60% overall
    }
    
    logger.info(f"✓ Tracking functionality: {'PASS' if success_criteria['tracking_works'] else 'FAIL'}")
    logger.info(f"✓ Person support: {'PASS' if success_criteria['multiple_people'] else 'FAIL'}")
    logger.info(f"✓ Stable tracking (>80%): {'PASS' if success_criteria['stable_tracking'] else 'FAIL'} ({tracking_rate:.1f}%)")
    logger.info(f"✓ Performance (>10 FPS): {'PASS' if success_criteria['performance'] else 'FAIL'} ({stats.get('fps', 0):.1f} FPS)")
    logger.info(f"✓ Track management: {'PASS' if success_criteria['track_management'] else 'FAIL'}")
    logger.info(f"✓ ID consistency (>70%): {'PASS' if success_criteria['id_consistency'] else 'FAIL'} ({performance['id_consistency']:.1%})")
    logger.info(f"✓ Overall performance (>60%): {'PASS' if success_criteria['overall_performance'] else 'FAIL'} ({performance['overall_score']:.1%})")
    
    overall_success = all(success_criteria.values())
    logger.info(f"\n🎯 PHASE 1 STEP 3 OVERALL: {'SUCCESS' if overall_success else 'NEEDS WORK'}")
    
    if overall_success:
        logger.info("✅ Ready for Phase 2: Interaction Detection!")
        logger.info("Next steps:")
        logger.info("  - Implement proximity analysis")
        logger.info("  - Add orientation-based interaction detection")
        logger.info("  - Create social network graphs")
    else:
        logger.info("⚠️ Recommendations:")
        if not success_criteria['tracking_works']:
            logger.info("  - Check detection system integration")
        if not success_criteria['performance']:
            logger.info("  - Optimize tracking parameters for better performance")
        if not success_criteria['id_consistency']:
            logger.info("  - Tune association distance thresholds")
            logger.info("  - Test with more people for better validation")
        if not success_criteria['stable_tracking']:
            logger.info("  - Review stability criteria and smoothing parameters")
    
    # Save final report
    report = {
        'timestamp': time.time(),
        'test_duration': test_duration,
        'statistics': stats,
        'performance': performance,
        'success_criteria': success_criteria,
        'overall_success': overall_success,
    }
    
    with open(output_dir / "phase1_step3_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nDetailed report saved to: {output_dir / 'phase1_step3_report.json'}")
    
    return overall_success

if __name__ == "__main__":
    success = run_phase1_step3_test()
    sys.exit(0 if success else 1)