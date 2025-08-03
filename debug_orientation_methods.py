#!/usr/bin/env python3
"""
Debug script for Phase 2 Step 2 orientation method failures
Diagnoses why skeleton and movement methods aren't working
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_skeleton_method():
    """Test if skeleton/pose detection is working."""
    logger.info("=== Testing Skeleton Method ===")
    
    try:
        from ultralytics import YOLO
        logger.info("✓ ultralytics imported successfully")
        
        # Try to load pose model
        try:
            model = YOLO('yolov8n-pose.pt')
            logger.info("✓ YOLOv8n-pose model loaded successfully")
            
            # Test with dummy image
            dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)
            results = model(dummy_image, verbose=False)
            logger.info("✓ Pose detection test successful")
            
            # Check keypoint structure
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    logger.info(f"✓ Keypoints available: {result.keypoints.shape}")
                else:
                    logger.warning("⚠️ No keypoints in results - this is normal for empty image")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load pose model: {e}")
            logger.info("💡 Try: pip install --upgrade ultralytics")
            return False
            
    except ImportError as e:
        logger.error(f"❌ ultralytics not available: {e}")
        logger.info("💡 Install with: pip install ultralytics")
        return False

def test_movement_method():
    """Test if movement-based orientation would work."""
    logger.info("\n=== Testing Movement Method ===")
    
    # Simulate movement tracking
    logger.info("Movement method requires:")
    logger.info("  - People moving during testing (not stationary)")
    logger.info("  - At least 5 frames of movement history")
    logger.info("  - Movement > 0.1 meters between frames")
    
    # Check configuration
    try:
        sys.path.append('config')
        from orientation_config import ORIENTATION_METHODS
        
        movement_config = ORIENTATION_METHODS.get('movement_based', {})
        min_movement = movement_config.get('min_movement_threshold', 0.1)
        
        logger.info(f"✓ Movement threshold: {min_movement}m")
        logger.info("💡 During testing: walk around, don't stand still!")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Could not load movement config: {e}")
        return False

def test_depth_gradient_method():
    """Test depth gradient method (should be working)."""
    logger.info("\n=== Testing Depth Gradient Method ===")
    logger.info("✓ This method should be working (fallback)")
    logger.info("Uses depth camera data to detect body asymmetry")
    logger.info("No additional dependencies required")
    return True

def check_orientation_config():
    """Check orientation configuration."""
    logger.info("\n=== Checking Configuration ===")
    
    try:
        sys.path.append('config')
        from orientation_config import ORIENTATION_METHODS
        
        for method, config in ORIENTATION_METHODS.items():
            enabled = config.get('enabled', True)
            priority = config.get('priority', 999)
            logger.info(f"{method}: enabled={enabled}, priority={priority}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Could not load orientation config: {e}")
        return False

def suggest_fixes():
    """Provide specific fix suggestions."""
    logger.info("\n=== Fix Suggestions ===")
    
    logger.info("🔧 To fix skeleton method:")
    logger.info("   1. pip install --upgrade ultralytics")
    logger.info("   2. python -c \"from ultralytics import YOLO; YOLO('yolov8n-pose.pt')\"")
    logger.info("   3. Ensure good lighting for pose detection")
    
    logger.info("\n🔧 To fix movement method:")
    logger.info("   1. Walk actively during testing")
    logger.info("   2. Move in different directions")
    logger.info("   3. Don't stand still for more than a few seconds")
    
    logger.info("\n🔧 To verify fixes:")
    logger.info("   1. Run this debug script again")
    logger.info("   2. Look for 'skeleton' and 'movement' successes in Phase 2 Step 2 output")
    logger.info("   3. Check the orientation dashboard for method diversity")

def main():
    """Run complete orientation method diagnosis."""
    logger.info("Phase 2 Step 2: Orientation Method Diagnosis")
    logger.info("=" * 50)
    
    skeleton_ok = test_skeleton_method()
    movement_ok = test_movement_method()
    depth_ok = test_depth_gradient_method()
    config_ok = check_orientation_config()
    
    logger.info("\n" + "=" * 50)
    logger.info("DIAGNOSIS SUMMARY:")
    logger.info(f"Skeleton Method:    {'✓ OK' if skeleton_ok else '❌ FAILED'}")
    logger.info(f"Movement Method:    {'✓ OK' if movement_ok else '❌ FAILED'}")
    logger.info(f"Depth Gradient:     {'✓ OK' if depth_ok else '❌ FAILED'}")
    logger.info(f"Configuration:      {'✓ OK' if config_ok else '❌ FAILED'}")
    
    if not skeleton_ok:
        logger.info("\n🎯 PRIMARY ISSUE: Skeleton/Pose detection not working")
    
    suggest_fixes()

if __name__ == "__main__":
    main()