#!/usr/bin/env python3
"""
Test script specifically for movement-based orientation detection
Provides detailed debugging information to help identify movement issues
"""

import sys
import os
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

import numpy as np
import cv2
import logging
import time
import json

# Test imports
try:
    from camera.realsense_capture import RealSenseCapture
    from detection.person_detector import PersonDetector
    from detection.tracker import PersonTracker
    from interaction.orientation_estimator import OrientationEstimator
    from config.detection_config import PERSON_DETECTION, MOUNT_CONFIG, DETECTION_ZONES
    from config.tracking_config import TRACKING_PARAMETERS
    from config.orientation_config import ORIENTATION_METHODS, SKELETON_ORIENTATION, MOVEMENT_ORIENTATION
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Setup detailed logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_movement_detection():
    """Test movement detection with detailed debugging."""
    print("\n=== Movement Detection Debug Test ===")
    print("Walk around continuously to generate movement data")
    print("The test will show detailed movement detection information")
    
    config = {
        'depth_width': 640, 'depth_height': 480, 'depth_fps': 30,
        'color_width': 640, 'color_height': 480, 'color_fps': 30,
        'align_streams': True,
        **PERSON_DETECTION, **MOUNT_CONFIG,
        'detection_zones': DETECTION_ZONES,
        **TRACKING_PARAMETERS,
        'orientation_methods': ORIENTATION_METHODS,
        'skeleton_orientation': SKELETON_ORIENTATION,
        'movement_orientation': MOVEMENT_ORIENTATION,
    }
    
    try:
        # Initialize components
        camera = RealSenseCapture(config)
        if not (camera.configure_camera() and camera.start_streaming()):
            logger.error("Failed to initialize camera")
            return False
        
        detector = PersonDetector(config)
        tracker = PersonTracker(config)
        estimator = OrientationEstimator(config)
        
        print("✓ All components initialized")
        print("\n--- Starting movement detection test ---")
        print("WALK AROUND CONTINUOUSLY to test movement detection")
        print("Press 'q' to quit")
        
        start_time = time.time()
        frame_count = 0
        movement_detections = 0
        
        while time.time() - start_time < 30:  # 30 second test
            depth, color = camera.get_frames()
            if depth is None or color is None:
                continue
                
            frame_count += 1
            timestamp = time.time()
            
            # Detection and tracking
            detections = detector.detect_people(depth, timestamp, color)
            tracked_people = tracker.update(detections, timestamp)
            
            if tracked_people:
                print(f"\n--- Frame {frame_count} ---")
                for person in tracked_people:
                    print(f"Person {person.id}:")
                    print(f"  Total frames seen: {person.total_frames_seen}")
                    print(f"  Position history length: {len(person.position_history) if hasattr(person, 'position_history') else 'N/A'}")
                    
                    if hasattr(person, 'position_history') and len(person.position_history) >= 2:
                        recent_pos = list(person.position_history)[-3:]
                        print(f"  Recent positions: {[f'({p[0]:.2f}, {p[1]:.2f})' for p in recent_pos]}")
                        
                        # Calculate movement manually
                        if len(recent_pos) >= 2:
                            move_vec = np.array(recent_pos[-1][:2]) - np.array(recent_pos[-2][:2])
                            move_mag = np.linalg.norm(move_vec)
                            print(f"  Manual movement calc: {move_mag:.4f}m")
                            
                            if hasattr(person, 'velocity') and person.velocity:
                                tracker_vel = np.linalg.norm(person.velocity[:2])
                                print(f"  Tracker velocity: {tracker_vel:.4f}m/frame")
                    
                    # Test movement orientation
                    orientation, debug_data = estimator.estimate_orientations(
                        [person], depth, color, timestamp
                    )
                    
                    if orientation:
                        # Check if movement method contributed to the final orientation
                        person_debug = debug_data.get(person.id, {})
                        movement_attempts = person_debug.get('method_attempts', {})
                        
                        if 'movement_based' in movement_attempts and movement_attempts['movement_based']['success']:
                            movement_detections += 1
                            movement_result = movement_attempts['movement_based']
                            print(f"  ✓ MOVEMENT DETECTED: {movement_result['angle']:.1f}° (conf: {movement_result['confidence']:.2f})")
                            
                            # Show the final combined result
                            final_orient = orientation[0] if orientation else None
                            if final_orient:
                                print(f"    Final result: {final_orient.orientation_angle:.1f}° (method: {final_orient.method}, conf: {final_orient.confidence:.2f})")
                        else:
                            print(f"  ✗ No movement orientation")
                            # Show what methods were attempted
                            if movement_attempts:
                                for method, result in movement_attempts.items():
                                    status = "✓" if result['success'] else "✗"
                                    print(f"    {status} {method}: conf {result.get('confidence', 0):.2f}")
                            else:
                                print(f"    No method attempts recorded")
                    
                    # Show debug data
                    person_debug = debug_data.get(person.id, {})
                    movement_data = person_debug.get('movement_data', {})
                    
                    if movement_data.get('issues'):
                        print(f"  Issues: {', '.join(movement_data['issues'])}")
                    
                    if 'movement_magnitude' in movement_data:
                        print(f"  Movement magnitude: {movement_data['movement_magnitude']:.4f}")
                        print(f"  Velocity source: {movement_data.get('velocity_source', 'unknown')}")
            
            # Quick visualization
            if tracked_people and frame_count % 5 == 0:  # Every 5 frames
                vis_frame = color.copy()
                for person in tracked_people:
                    if person.current_detection:
                        bbox = person.current_detection.bounding_box
                        x, y, w, h = bbox
                        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(vis_frame, f"ID:{person.id}", (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow('Movement Test', cv2.resize(vis_frame, (640, 480)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()
    
    print(f"\n=== Movement Test Results ===")
    print(f"Frames processed: {frame_count}")
    print(f"Movement orientations detected: {movement_detections}")
    
    if movement_detections > 0:
        print(f"✓ SUCCESS: Movement detection is working!")
        print(f"Detection rate: {(movement_detections/frame_count*100):.1f}%")
        return True
    else:
        print(f"✗ FAILED: No movement orientations detected")
        print("Possible issues:")
        print("  - Movement threshold too high (try lower values)")
        print("  - Not enough movement (walk around more)")
        print("  - Position history not being built up")
        print("  - Tracker not calculating velocity correctly")
        return False

if __name__ == "__main__":
    print("Movement Detection Debug Test")
    print("=" * 35)
    
    success = test_movement_detection()
    
    if success:
        print("\n🎉 Movement detection is working!")
    else:
        print("\n⚠️ Movement detection needs investigation")
        print("Check the debug output above for specific issues")
    
    sys.exit(0 if success else 1)