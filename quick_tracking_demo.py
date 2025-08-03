"""
Quick Tracking Demo for Phase 1 Step 3
Demonstrates tracking with single person (good for initial testing)
"""

import sys
import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from camera.realsense_capture import RealSenseCapture
from camera.frame_processor import FrameProcessor
from detection.person_detector import PersonDetector
from detection.tracker import PersonTracker
from config.detection_config import PERSON_DETECTION, MOUNT_CONFIG, DETECTION_ZONES
from config.tracking_config import TRACKING_PARAMETERS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_tracking_visualization(frame: np.ndarray, tracked_people, stats: dict) -> np.ndarray:
    """Create simple tracking visualization for demo."""
    vis_frame = frame.copy()
    
    # Track colors (cycle through for multiple people)
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for person in tracked_people:
        if person.current_detection is None:
            continue
        
        color = colors[person.id % len(colors)]
        
        # Bounding box
        x, y, w, h = person.current_detection.bounding_box
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 3)
        
        # Track ID (large and clear)
        cv2.putText(vis_frame, f"ID: {person.id}", (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        # Tracking status
        status = "STABLE" if person.is_stable() else f"TRACKING ({person.total_frames_seen}f)"
        cv2.putText(vis_frame, status, (x, y + h + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show position
        pos = person.get_latest_position()
        pos_text = f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
        cv2.putText(vis_frame, pos_text, (x, y + h + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Show trajectory (simple version)
        if len(person.position_history) > 1:
            # Draw last few positions
            positions = list(person.position_history)[-5:]
            for i in range(1, len(positions)):
                # Convert 3D to 2D (simple projection)
                p1_x = int(320 + positions[i-1][0] * 50)
                p1_y = int(240 - positions[i-1][1] * 50)
                p2_x = int(320 + positions[i][0] * 50)
                p2_y = int(240 - positions[i][1] * 50)
                
                # Clamp to screen
                p1_x = max(0, min(vis_frame.shape[1]-1, p1_x))
                p1_y = max(0, min(vis_frame.shape[0]-1, p1_y))
                p2_x = max(0, min(vis_frame.shape[1]-1, p2_x))
                p2_y = max(0, min(vis_frame.shape[0]-1, p2_y))
                
                cv2.line(vis_frame, (p1_x, p1_y), (p2_x, p2_y), color, 2)
    
    # Add stats overlay
    stats_y = 30
    cv2.putText(vis_frame, f"Active Tracks: {len(tracked_people)}", 
               (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    stats_y += 30
    cv2.putText(vis_frame, f"Total Created: {stats.get('total_created', 0)}", 
               (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    stats_y += 30
    cv2.putText(vis_frame, f"FPS: {stats.get('fps', 0):.1f}", 
               (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Instructions
    instructions = [
        "Phase 1 Step 3: Basic Tracking Demo",
        "Move around to test tracking!",
        "Press 'q' to quit, 'r' to reset tracker"
    ]
    
    for i, instruction in enumerate(instructions):
        cv2.putText(vis_frame, instruction, (10, vis_frame.shape[0] - 60 + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return vis_frame

def run_quick_tracking_demo():
    """Run quick tracking demo for Phase 1 Step 3 validation."""
    print("Quick Tracking Demo - Phase 1 Step 3")
    print("=" * 40)
    print("This demo tests basic tracking functionality:")
    print("• ID assignment and consistency")
    print("• Frame-to-frame association") 
    print("• Basic trajectory smoothing")
    print("• Track stability detection")
    print()
    
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
    logger.info("Initializing tracking demo components...")
    
    camera = RealSenseCapture(config)
    processor = FrameProcessor(config)
    detector = PersonDetector(config)
    tracker = PersonTracker(config)
    
    if not (camera.configure_camera() and camera.start_streaming()):
        logger.error("Failed to initialize camera")
        return False
    
    logger.info("✓ Camera initialized")
    logger.info("✓ Detector loaded")
    logger.info("✓ Tracker ready")
    print()
    print("Demo Instructions:")
    print("• Stand 1-3 meters from camera")
    print("• Move around slowly to test tracking")
    print("• Try stepping out of view and returning")
    print("• Watch for stable track ID assignment")
    print()
    
    # Demo parameters
    demo_duration = 120  # 2 minutes
    target_fps = 15
    frame_interval = 1.0 / target_fps
    
    # Statistics
    stats = {
        'total_frames': 0,
        'total_detections': 0,
        'active_tracks': 0,
        'max_tracks': 0,
        'processing_times': [],
        'frame_count': 0,
    }
    
    start_time = time.time()
    last_frame_time = start_time
    
    try:
        while time.time() - start_time < demo_duration:
            depth, color = camera.get_frames()
            if depth is None or color is None:
                continue
            
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_frame_time < frame_interval:
                continue
            
            # Process frame
            timestamp = current_time
            processed = processor.process_frame_pair(depth, color, timestamp)
            
            # Detection
            start_detect = time.time()
            detections = detector.detect_people(depth, timestamp, color)
            
            # Tracking
            tracked_people = tracker.update(detections, timestamp)
            processing_time = time.time() - start_detect
            
            # Update statistics
            stats['total_frames'] += 1
            stats['frame_count'] += 1
            stats['processing_times'].append(processing_time)
            stats['total_detections'] += len(detections)
            stats['active_tracks'] = len(tracked_people)
            stats['max_tracks'] = max(stats['max_tracks'], len(tracked_people))
            stats['total_created'] = tracker.total_tracks_created
            stats['total_lost'] = tracker.total_tracks_lost
            
            # Calculate FPS
            if stats['processing_times']:
                avg_time = np.mean(stats['processing_times'][-30:])  # Last 30 frames
                stats['fps'] = 1.0 / max(avg_time, 0.001)
            
            # Create visualization
            vis_frame = create_simple_tracking_visualization(color, tracked_people, stats)
            
            # Display
            cv2.imshow('Quick Tracking Demo', vis_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nDemo ended by user")
                break
            elif key == ord('r'):
                tracker.reset()
                logger.info("Tracker reset")
                print("Tracker reset - IDs will restart from 1")
            
            # Periodic status
            if stats['total_frames'] % 50 == 0 and stats['total_frames'] > 0:
                print(f"Frame {stats['total_frames']}: {len(tracked_people)} active tracks, "
                      f"FPS: {stats.get('fps', 0):.1f}")
            
            last_frame_time = current_time
    
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()
    
    # Results summary
    print("\n" + "=" * 50)
    print("QUICK TRACKING DEMO RESULTS")
    print("=" * 50)
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Tracks created: {stats['total_created']}")
    print(f"Tracks lost: {stats['total_lost']}")
    print(f"Max simultaneous tracks: {stats['max_tracks']}")
    
    if stats['processing_times']:
        avg_time = np.mean(stats['processing_times'])
        print(f"Average processing time: {avg_time:.3f}s")
        print(f"Average FPS: {1/avg_time:.1f}")
    
    # Simple validation
    tracking_worked = stats['total_created'] > 0
    performance_good = stats.get('fps', 0) > 10
    
    print(f"\n✓ Tracking functionality: {'WORKING' if tracking_worked else 'FAILED'}")
    print(f"✓ Performance: {'GOOD' if performance_good else 'NEEDS IMPROVEMENT'} ({stats.get('fps', 0):.1f} FPS)")
    
    if tracking_worked and performance_good:
        print(f"\n🎉 Phase 1 Step 3 Basic Requirements: SATISFIED")
        print("Ready for multi-person testing with run_phase1_step3.py")
    else:
        print(f"\n⚠️ Phase 1 Step 3: NEEDS WORK")
        print("Check tracking parameters or hardware performance")
    
    return tracking_worked and performance_good

if __name__ == "__main__":
    success = run_quick_tracking_demo()
    sys.exit(0 if success else 1)