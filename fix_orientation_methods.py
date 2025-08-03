#!/usr/bin/env python3
"""
Windows-Compatible Quick Fix Script for Phase 2 Step 2 orientation method issues
Automatically resolves common skeleton and movement method problems
"""

import subprocess
import sys
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def fix_skeleton_method():
    """Fix skeleton/pose detection issues - Windows compatible."""
    logger.info("Fixing skeleton method...")
    
    try:
        # Update ultralytics - Windows compatible subprocess call
        logger.info("Updating ultralytics...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✓ ultralytics updated successfully")
        else:
            logger.warning(f"ultralytics update warning: {result.stderr}")
        
        # Test and download pose model
        logger.info("Testing pose model...")
        from ultralytics import YOLO
        
        model = YOLO('yolov8n-pose.pt')
        logger.info("✓ YOLOv8n-pose model ready")
        
        # Verify model works
        import numpy as np
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        results = model(dummy, verbose=False)
        logger.info("✓ Skeleton method fixed!")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to fix skeleton method: {e}")
        logger.info("Try manual install: pip install --upgrade ultralytics")
        return False

def fix_movement_method():
    """Verify movement method configuration."""
    logger.info("Checking movement method...")
    
    try:
        # Check if config exists and is reasonable
        config_path = Path('config/orientation_config.py')
        if config_path.exists():
            logger.info("✓ Movement configuration file exists")
            
            # Read and check movement threshold
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "min_movement_threshold" in content:
                logger.info("✓ Movement threshold configured")
            else:
                logger.warning("Movement threshold not found in config")
        
        logger.info("Movement method needs active movement during testing:")
        logger.info("   - Walk around during Phase 2 Step 2 test")
        logger.info("   - Don't stand still")
        logger.info("   - Move in different directions")
        
        return True
        
    except Exception as e:
        logger.warning(f"Could not verify movement config: {e}")
        return False

def create_test_movement_script():
    """Create a script to test movement detection - Windows compatible."""
    logger.info("Creating movement test helper...")
    
    test_script = '''#!/usr/bin/env python3
"""
Movement Test Helper - Use during Phase 2 Step 2 testing
Reminds you to move around for movement-based orientation detection
"""

import time
import sys

def movement_reminder():
    """Provide movement guidance during testing."""
    print("MOVEMENT TEST HELPER")
    print("=" * 30)
    print("For movement-based orientation to work:")
    print()
    
    movements = [
        "Walk forward slowly",
        "Walk backward",
        "Step left and right", 
        "Turn in a circle",
        "Walk diagonally",
        "Face different directions while moving"
    ]
    
    for i, movement in enumerate(movements, 1):
        print(f"{i}. {movement}")
        print("   (Move for 5-10 seconds)")
        input("   Press Enter when done...")
        print()
    
    print("Movement test complete!")
    print("Now run: python run_phase2_step2.py")

if __name__ == "__main__":
    movement_reminder()
'''
    
    try:
        with open('test_movement.py', 'w', encoding='utf-8') as f:
            f.write(test_script)
        logger.info("✓ Created test_movement.py helper script")
        return True
    except Exception as e:
        logger.error(f"Failed to create movement script: {e}")
        return False

def check_ultralytics_manually():
    """Check if ultralytics is working manually."""
    logger.info("Manual ultralytics check...")
    
    try:
        from ultralytics import YOLO
        logger.info("✓ ultralytics can be imported")
        
        # Try to create model
        model = YOLO('yolov8n-pose.pt')
        logger.info("✓ YOLO pose model accessible")
        
        return True
    except Exception as e:
        logger.error(f"ultralytics issue: {e}")
        return False

def main():
    """Run complete orientation method fixes - Windows compatible."""
    logger.info("Phase 2 Step 2: Orientation Method Auto-Fix (Windows)")
    logger.info("=" * 50)
    
    # Try manual check first
    ultralytics_ok = check_ultralytics_manually()
    
    if not ultralytics_ok:
        logger.info("Attempting to fix ultralytics installation...")
        skeleton_fixed = fix_skeleton_method()
    else:
        logger.info("✓ ultralytics already working")
        skeleton_fixed = True
    
    movement_ok = fix_movement_method()
    script_created = create_test_movement_script()
    
    logger.info("\n" + "=" * 50)
    logger.info("FIX SUMMARY:")
    logger.info(f"Skeleton Method: {'FIXED' if skeleton_fixed else 'NEEDS MANUAL FIX'}")
    logger.info(f"Movement Method: {'READY' if movement_ok else 'NEEDS TESTING'}")
    logger.info(f"Test Script: {'CREATED' if script_created else 'FAILED'}")
    
    if skeleton_fixed and movement_ok:
        logger.info("\nOrientation methods should now work!")
        logger.info("\nNext steps:")
        logger.info("1. Optional: python test_movement.py  (practice movements)")
        logger.info("2. python run_phase2_step2.py  (run the actual test)")
        logger.info("3. IMPORTANT: Move around actively during testing!")
        logger.info("4. Look for 'skeleton' and 'movement' successes in output")
    else:
        logger.info("\nManual fixes needed:")
        if not skeleton_fixed:
            logger.info("   - Try: pip install --upgrade ultralytics")
            logger.info("   - Check internet connection for model download")
        logger.info("   - Ensure active movement during testing")
        
    logger.info("\nTesting Tips:")
    logger.info("- Use good lighting")
    logger.info("- Stand 1-3 meters from camera")
    logger.info("- Walk around continuously during test")
    logger.info("- Don't stand still for more than 2 seconds")

if __name__ == "__main__":
    main()