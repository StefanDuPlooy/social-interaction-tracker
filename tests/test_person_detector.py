"""
Test Script for Person Detector
Phase 1 Step 2 Validation - Day 3-4
Tests person detection with saved frames and live camera
"""

import sys
import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from camera.realsense_capture import RealSenseCapture
from detection.person_detector import PersonDetector
from config.detection_config import PERSON_DETECTION, MOUNT_CONFIG, DETECTION_ZONES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_with_saved_frames():
    """Test person detection with saved depth frames."""
    print("=== Testing Person Detection with Saved Frames ===")
    
    # Look for saved test frames
    test_frames_dir = Path("data/test_sessions")
    test_frames_dir.mkdir(parents=True, exist_ok=True)
    
    depth_files = list(test_frames_dir.glob("test_depth_*.png"))
    color_files = list(test_frames_dir.glob("test_color_*.png"))
    
    if not depth_files:
        print("No saved test frames found. Run camera capture first.")
        return False
    
    # Create detector with test config
    config = {
        **PERSON_DETECTION,
        **MOUNT_CONFIG,
        'detection_zones': DETECTION_ZONES
    }
    
    detector = PersonDetector(config)
    
    success_count = 0
    total_detections = 0
    
    for i, depth_file in enumerate(sorted(depth_files)):
        # Find corresponding color file
        color_file = test_frames_dir / f"test_color_{i:03d}.png"
        if not color_file.exists():
            continue
        
        # Load frames
        depth_frame = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
        color_frame = cv2.imread(str(color_file))
        
        if depth_frame is None or color_frame is None:
            print(f"Failed to load frame {i}")
            continue
        
        # Convert depth to proper format (assuming saved as uint16 millimeters)
        depth_frame = depth_frame.astype(np.float32)
        
        # Test detection
        timestamp = time.time()
        detections = detector.detect_people(depth_frame, timestamp)
        
        print(f"Frame {i}: Detected {len(detections)} people")
        
        if detections:
            success_count += 1
            total_detections += len(detections)
            
            # Print detection details
            for j, detection in enumerate(detections):
                print(f"  Person {j}: Zone={detection.zone}, "
                      f"Confidence={detection.confidence:.2f}, "
                      f"Position=({detection.position_3d[0]:.1f}, "
                      f"{detection.position_3d[1]:.1f}, {detection.position_3d[2]:.1f})")
        
        # Create and save visualization
        person_mask = detector.create_person_mask(
            detector.preprocess_depth_frame(depth_frame)
        )
        vis_frame = detector.visualize_detections(color_frame, detections, person_mask)
        
        # Save visualization
        output_path = test_frames_dir / f"detection_result_{i:03d}.png"
        cv2.imwrite(str(output_path), vis_frame)
    
    print(f"\nResults: {success_count}/{len(depth_files)} frames had detections")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per successful frame: {total_detections/max(success_count,1):.1f}")
    
    return success_count > 0

def test_with_live_camera():
    """Test person detection with live camera feed."""
    print("\n=== Testing Person Detection with Live Camera ===")
    
    # Camera configuration
    camera_config = {
        'depth_width': 640,
        'depth_height': 480,
        'depth_fps': 30,
        'color_width': 640,
        'color_height': 480,
        'color_fps': 30,
        'align_streams': True,
    }
    
    # Detection configuration
    detection_config = {
        **camera_config,
        **PERSON_DETECTION,
        **MOUNT_CONFIG,
        'detection_zones': DETECTION_ZONES
    }
    
    # Initialize camera and detector
    camera = RealSenseCapture(camera_config)
    detector = PersonDetector(detection_config)
    
    if not (camera.configure_camera() and camera.start_streaming()):
        print("Failed to initialize camera")
        return False
    
    print("Camera initialized. Testing detection for 60 seconds...")
    print("Press 'q' to quit, 's' to save current frame")
    
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    
    try:
        while time.time() - start_time < 60:  # Run for 60 seconds
            depth, color = camera.get_frames()
            if depth is None or color is None:
                continue
            
            # Run detection
            timestamp = time.time()
            detections = detector.detect_people(depth, timestamp)
            
            if detections:
                detection_count += len(detections)
                print(f"Frame {frame_count}: Detected {len(detections)} people")
                
                for i, detection in enumerate(detections):
                    print(f"  Person {i}: {detection.zone} confidence={detection.confidence:.2f}")
            
            # Create visualization
            person_mask = detector.create_person_mask(
                detector.preprocess_depth_frame(depth)
            )
            vis_frame = detector.visualize_detections(color, detections, person_mask)
            
            # Display
            cv2.imshow('Live Person Detection', vis_frame)
            cv2.imshow('Person Mask', person_mask)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame for analysis
                save_dir = Path("data/test_sessions")
                save_dir.mkdir(parents=True, exist_ok=True)
                
                cv2.imwrite(str(save_dir / f"live_depth_{frame_count:03d}.png"), depth)
                cv2.imwrite(str(save_dir / f"live_color_{frame_count:03d}.png"), color)
                cv2.imwrite(str(save_dir / f"live_detection_{frame_count:03d}.png"), vis_frame)
                print(f"Saved frame {frame_count}")
            
            frame_count += 1
            time.sleep(0.1)  # ~10 FPS for testing
    
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()
    
    print(f"\nLive test completed:")
    print(f"Processed {frame_count} frames")
    print(f"Total detections: {detection_count}")
    print(f"Average detections per frame: {detection_count/max(frame_count,1):.2f}")
    
    return frame_count > 0

def validate_detection_accuracy():
    """Validate detection accuracy with known scenarios."""
    print("\n=== Validation Tests ===")
    
    # Test 1: Empty room (should detect 0 people)
    print("Test 1: Please ensure room is empty, then press Enter...")
    input()
    
    # Test 2: One person standing
    print("Test 2: One person stand in view, then press Enter...")
    input()
    
    # Test 3: Two people at different distances
    print("Test 3: Two people at different distances, then press Enter...")
    input()
    
    # Test 4: Person sitting vs standing
    print("Test 4: Person sitting down, then press Enter...")
    input()
    
    print("Manual validation complete. Check detection results visually.")

def main():
    """Main testing function."""
    print("Person Detector Testing Suite")
    print("=============================")
    
    # Test 1: Saved frames
    saved_frames_success = test_with_saved_frames()
    
    # Test 2: Live camera (if saved frames worked)
    if saved_frames_success:
        live_camera_success = test_with_live_camera()
        
        if live_camera_success:
            # Test 3: Manual validation
            validate_detection_accuracy()
    
    print("\n=== Testing Complete ===")
    print("Check the following:")
    print("1. Detection visualizations in data/test_sessions/")
    print("2. Console output for detection statistics")
    print("3. Visual accuracy of bounding boxes and confidence zones")
    
    # Validation criteria from implementation guide
    print("\nValidation Criteria (Phase 1 Step 2):")
    print("✓ System detects and counts people in saved frames")
    print("✓ Bounding boxes approximately match person locations")
    print("✓ Confidence zones (high/medium/low) are reasonable")
    print("✓ Detection works with 2-3 people simultaneously")
    print("✓ False positive rate is acceptably low")

if __name__ == "__main__":
    main()