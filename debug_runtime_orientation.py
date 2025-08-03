#!/usr/bin/env python3
"""
Runtime Orientation Debug Script - Phase 2 Step 2
Identifies specific runtime issues with skeleton and movement detection
"""

import logging
import sys
import os
import numpy as np
from pathlib import Path

# Add project paths
sys.path.append('src')
sys.path.append('config')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_skeleton_runtime_issues():
    """Test for common skeleton detection runtime failures."""
    logger.info("=== Testing Skeleton Runtime Issues ===")
    
    try:
        from ultralytics import YOLO
        from orientation_config import SKELETON_ORIENTATION
        
        # Load model
        model = YOLO('yolov8n-pose.pt')
        logger.info("✓ YOLO model loaded")
        
        # Create test scenarios that commonly fail
        test_cases = [
            ("Empty image", np.zeros((480, 640, 3), dtype=np.uint8)),
            ("Very small ROI", np.zeros((20, 15, 3), dtype=np.uint8)),
            ("Single channel", np.zeros((480, 640), dtype=np.uint8)),
            ("Normal RGB", np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        ]
        
        for name, test_image in test_cases:
            try:
                logger.info(f"Testing: {name} - Shape: {test_image.shape}")
                
                # Convert single channel to RGB if needed
                if len(test_image.shape) == 2:
                    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)
                
                results = model(test_image, verbose=False)
                
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints_shape = result.keypoints.data.shape
                        logger.info(f"  ✓ Keypoints shape: {keypoints_shape}")
                        
                        # Check for actual detections
                        if keypoints_shape[0] > 0:
                            keypoints = result.keypoints.data[0].cpu().numpy()
                            visible = keypoints[keypoints[:, 2] > 0.3]
                            logger.info(f"  ✓ Visible keypoints: {len(visible)}")
                        else:
                            logger.info(f"  ℹ️ No person detected (normal for test images)")
                    else:
                        logger.warning(f"  ⚠️ No keypoints attribute")
                else:
                    logger.warning(f"  ⚠️ No results returned")
                    
            except Exception as e:
                logger.error(f"  ❌ Failed on {name}: {e}")
        
        # Test the specific issues from your code
        logger.info("\n--- Testing Common Runtime Failures ---")
        
        # Test bounding box issues
        logger.info("Bounding box edge cases:")
        test_bboxes = [
            (0, 0, 10, 10),      # Very small
            (-5, -5, 50, 50),    # Negative coordinates
            (600, 400, 100, 100), # Out of bounds
            (100, 100, 200, 200)  # Normal
        ]
        
        frame_shape = (480, 640, 3)
        for bbox in test_bboxes:
            x, y, w, h = bbox
            logger.info(f"  Testing bbox {bbox}:")
            
            # Check bounds
            if x < 0 or y < 0 or x + w > frame_shape[1] or y + h > frame_shape[0]:
                logger.info(f"    ⚠️ Out of bounds issue detected")
                continue
            
            # Test ROI extraction
            try:
                dummy_frame = np.random.randint(0, 255, frame_shape, dtype=np.uint8)
                roi = dummy_frame[y:y+h, x:x+w]
                if roi.size == 0:
                    logger.warning(f"    ⚠️ Empty ROI")
                else:
                    logger.info(f"    ✓ ROI size: {roi.shape}")
            except Exception as e:
                logger.error(f"    ❌ ROI extraction failed: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Skeleton runtime test failed: {e}")
        return False

def test_movement_runtime_issues():
    """Test movement detection runtime issues."""
    logger.info("\n=== Testing Movement Runtime Issues ===")
    
    try:
        from orientation_config import ORIENTATION_METHODS
        
        movement_config = ORIENTATION_METHODS.get('movement_based', {})
        min_threshold = movement_config.get('min_movement_threshold', 0.1)
        smoothing_window = movement_config.get('smoothing_window', 5)
        
        logger.info(f"Movement threshold: {min_threshold}m")
        logger.info(f"Smoothing window: {smoothing_window} frames")
        
        # Simulate movement scenarios
        scenarios = [
            ("Stationary person", [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]),
            ("Slow movement", [(1.0, 1.0), (1.05, 1.0), (1.1, 1.0)]),
            ("Fast movement", [(1.0, 1.0), (1.5, 1.0), (2.0, 1.0)]),
            ("Insufficient history", [(1.0, 1.0), (1.2, 1.0)]),
            ("Sufficient history", [(1.0, 1.0), (1.2, 1.0), (1.4, 1.0), (1.6, 1.0), (1.8, 1.0)])
        ]
        
        for name, positions in scenarios:
            logger.info(f"Testing: {name}")
            
            if len(positions) < 2:
                logger.info(f"  ⚠️ Insufficient position history")
                continue
            
            # Calculate movement
            movements = []
            for i in range(1, len(positions)):
                prev_x, prev_y = positions[i-1]
                curr_x, curr_y = positions[i]
                
                movement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                movements.append(movement)
            
            avg_movement = np.mean(movements) if movements else 0
            logger.info(f"  Average movement: {avg_movement:.3f}m")
            
            if avg_movement > min_threshold:
                logger.info(f"  ✓ Sufficient movement (>{min_threshold}m)")
            else:
                logger.info(f"  ⚠️ Insufficient movement (<{min_threshold}m)")
            
            if len(positions) >= smoothing_window:
                logger.info(f"  ✓ Sufficient history (>={smoothing_window} frames)")
            else:
                logger.info(f"  ⚠️ Insufficient history (<{smoothing_window} frames)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Movement runtime test failed: {e}")
        return False

def check_actual_log_issues():
    """Check for issues mentioned in actual system logs."""
    logger.info("\n=== Common Runtime Issues from Logs ===")
    
    issues_and_solutions = [
        ("Bbox out of bounds", "Person detection bbox extends beyond frame boundaries"),
        ("Empty ROI", "Bounding box results in zero-size region"),
        ("No pose results", "YOLO doesn't detect any person in the ROI"),
        ("No keypoints in result", "YOLO result doesn't have keypoints attribute"),
        ("Insufficient keypoints", "Less than 3 visible keypoints for orientation"),
        ("No current detection", "Person tracking lost current frame detection"),
        ("Insufficient position history", "Movement needs at least 5 frames of history"),
        ("Movement below threshold", "Person not moving enough for movement-based detection")
    ]
    
    logger.info("Common runtime failures and their causes:")
    for issue, cause in issues_and_solutions:
        logger.info(f"  • {issue}: {cause}")
    
    return True

def suggest_runtime_fixes():
    """Provide specific runtime fix suggestions."""
    logger.info("\n=== Runtime Fix Suggestions ===")
    
    logger.info("🔧 For skeleton method failures:")
    logger.info("   1. Ensure good lighting - pose detection needs clear visibility")
    logger.info("   2. Stand closer to camera - small people are harder to detect")
    logger.info("   3. Face the camera more directly - side views reduce keypoints")
    logger.info("   4. Avoid occlusion - don't hide behind objects")
    logger.info("   5. Check camera positioning - avoid extreme angles")
    
    logger.info("\n🔧 For movement method failures:")
    logger.info("   1. Walk actively during testing - don't stand still")
    logger.info("   2. Move at least 10cm between frames")
    logger.info("   3. Allow 5+ frames to build movement history")
    logger.info("   4. Move in various directions - not just forward/back")
    logger.info("   5. Maintain consistent movement speed")
    
    logger.info("\n🔧 For debugging during Phase 2 Step 2:")
    logger.info("   1. Look for specific error messages in console output")
    logger.info("   2. Check the orientation dashboard for method breakdown")
    logger.info("   3. Verify person detection is working first")
    logger.info("   4. Test with single person before multiple people")
    logger.info("   5. Use good lighting and clear background")

def main():
    """Run complete runtime diagnosis."""
    logger.info("Phase 2 Step 2: Runtime Orientation Diagnosis")
    logger.info("=" * 50)
    
    skeleton_ok = test_skeleton_runtime_issues()
    movement_ok = test_movement_runtime_issues()
    check_actual_log_issues()
    
    logger.info("\n" + "=" * 50)
    logger.info("RUNTIME DIAGNOSIS SUMMARY:")
    logger.info(f"Skeleton Runtime: {'✅ OK' if skeleton_ok else '❌ ISSUES'}")
    logger.info(f"Movement Runtime: {'✅ OK' if movement_ok else '❌ ISSUES'}")
    
    suggest_runtime_fixes()
    
    logger.info("\n🎯 Next Steps:")
    logger.info("1. Run: python run_phase2_step2.py")
    logger.info("2. During testing:")
    logger.info("   - Use good lighting")
    logger.info("   - Stand close to camera")
    logger.info("   - Move actively (walk around)")
    logger.info("   - Face camera directly sometimes")
    logger.info("3. Look for method success counts in output")
    logger.info("4. Check console for specific error messages")

if __name__ == "__main__":
    # Import cv2 if available
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for some tests")
    
    main()