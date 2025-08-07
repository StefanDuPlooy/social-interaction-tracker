#!/usr/bin/env python3
"""
Test script to verify orientation detection fixes
This script runs a quick test to see if skeleton and movement detection are working
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s: %(message)s')

def test_orientation_estimator():
    """Test orientation estimator initialization and configuration."""
    print("\n=== Testing Orientation Estimator ===")
    
    config = {
        'orientation_methods': ORIENTATION_METHODS,
        'skeleton_orientation': SKELETON_ORIENTATION,  
        'movement_orientation': MOVEMENT_ORIENTATION,
        'depth_gradient_orientation': {'min_gradient_strength': 30}
    }
    
    try:
        estimator = OrientationEstimator(config)
        print(f"✓ OrientationEstimator initialized")
        print(f"✓ Pose model available: {estimator.pose_model is not None}")
        print(f"✓ Methods configured: {list(estimator.methods.keys())}")
        
        # Check method configuration
        for method_name, method_config in estimator.methods.items():
            enabled = method_config.get('enabled', False)
            threshold = method_config.get('confidence_threshold', 'N/A')
            print(f"  - {method_name}: enabled={enabled}, threshold={threshold}")
        
        return estimator
    except Exception as e:
        print(f"✗ OrientationEstimator failed: {e}")
        return None

def test_camera_and_detection():
    """Test camera and detection components."""
    print("\n=== Testing Camera and Detection ===")
    
    config = {
        'depth_width': 640, 'depth_height': 480, 'depth_fps': 30,
        'color_width': 640, 'color_height': 480, 'color_fps': 30,
        'align_streams': True,
        **PERSON_DETECTION, **MOUNT_CONFIG,
        'detection_zones': DETECTION_ZONES,
        **TRACKING_PARAMETERS
    }
    
    try:
        # Test camera
        camera = RealSenseCapture(config)
        if not camera.configure_camera():
            print("✗ Camera configuration failed")
            return None, None, None
        
        if not camera.start_streaming():
            print("✗ Camera streaming failed")  
            return None, None, None
            
        print("✓ Camera initialized and streaming")
        
        # Test detector
        detector = PersonDetector(config)
        print("✓ PersonDetector initialized")
        
        # Test tracker
        tracker = PersonTracker(config)
        print("✓ PersonTracker initialized")
        
        return camera, detector, tracker
        
    except Exception as e:
        print(f"✗ Camera/Detection setup failed: {e}")
        return None, None, None

def run_quick_test():
    """Run a quick 10-second test to see if orientation detection works."""
    print("\n=== Running Quick Orientation Test (10 seconds) ===")
    print("Stand in front of camera and move around / turn to test orientation detection")
    
    # Test estimator first
    estimator = test_orientation_estimator()
    if not estimator:
        return False
    
    # Test camera and detection
    camera, detector, tracker = test_camera_and_detection()
    if not all([camera, detector, tracker]):
        return False
    
    import time
    start_time = time.time()
    frame_count = 0
    orientation_count = 0
    
    try:
        while time.time() - start_time < 10:  # 10 second test
            depth, color = camera.get_frames()
            if depth is None or color is None:
                continue
                
            frame_count += 1
            timestamp = time.time()
            
            # Detection and tracking
            detections = detector.detect_people(depth, timestamp, color)
            tracked_people = tracker.update(detections, timestamp)
            
            if tracked_people:
                # Test orientation estimation
                orientations, debug_data = estimator.estimate_orientations(
                    tracked_people, depth, color, timestamp
                )
                
                if orientations:
                    orientation_count += 1
                    for orient in orientations:
                        print(f"Frame {frame_count}: Person {orient.person_id} - "
                              f"Angle: {orient.orientation_angle:.1f}°, "
                              f"Method: {orient.method}, "
                              f"Confidence: {orient.confidence:.2f}")
                
                # Print debug info for first person
                if debug_data:
                    person_id = list(debug_data.keys())[0]
                    person_debug = debug_data[person_id]
                    method_attempts = person_debug.get('method_attempts', {})
                    
                    if frame_count % 30 == 0:  # Every 30 frames
                        print(f"  Debug P{person_id}:")
                        for method, result in method_attempts.items():
                            success = result.get('success', False)
                            reason = result.get('reason', 'No reason')
                            print(f"    {method}: {'✓' if success else '✗'} {reason}")
            
            # Exit early if we got some good results
            if orientation_count > 5:
                print(f"✓ Got {orientation_count} orientation detections - test successful!")
                break
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False
    finally:
        camera.stop_streaming()
    
    print(f"\n=== Test Results ===")
    print(f"Frames processed: {frame_count}")
    print(f"Orientations detected: {orientation_count}")
    print(f"Success rate: {(orientation_count/frame_count*100) if frame_count > 0 else 0:.1f}%")
    
    # Get final stats
    stats = estimator.get_orientation_statistics()
    print(f"Method success counts: {stats['method_success_counts']}")
    
    success = orientation_count > 0
    print(f"Overall test result: {'✓ SUCCESS' if success else '✗ FAILED'}")
    
    return success

if __name__ == "__main__":
    print("Orientation Detection Fix Test")
    print("=" * 40)
    
    success = run_quick_test()
    
    if success:
        print("\n🎉 Orientation detection is working!")
        print("You can now run: python run_phase2_step2.py")
    else:
        print("\n⚠️  Orientation detection needs more work")
        print("Check the debug output above for specific issues")
    
    sys.exit(0 if success else 1)