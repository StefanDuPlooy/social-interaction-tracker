#!/usr/bin/env python3
"""
Quick diagnostic to check actual configuration values being used
"""

import sys
from pathlib import Path

# Add config to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "config"))

try:
    from orientation_config import ORIENTATION_METHODS, SKELETON_ORIENTATION
    
    print("🔍 CURRENT CONFIGURATION VALUES")
    print("=" * 50)
    
    # Check skeleton method threshold
    skeleton_method = ORIENTATION_METHODS.get('skeleton_based', {})
    skeleton_threshold = skeleton_method.get('confidence_threshold', 'NOT_SET')
    print(f"Skeleton method confidence threshold: {skeleton_threshold}")
    
    # Check joint threshold
    joint_threshold = SKELETON_ORIENTATION.get('joint_confidence_threshold', 'NOT_SET')
    print(f"Joint confidence threshold: {joint_threshold}")
    
    # Check movement threshold
    movement_method = ORIENTATION_METHODS.get('movement_based', {})
    movement_threshold = movement_method.get('min_movement_threshold', 'NOT_SET')
    print(f"Movement threshold: {movement_threshold}")
    
    print("\n✅ DIAGNOSIS:")
    
    # Check if thresholds are reasonable
    if skeleton_threshold != 'NOT_SET' and skeleton_threshold <= 0.3:
        print(f"✓ Skeleton threshold ({skeleton_threshold}) looks good")
    else:
        print(f"❌ Skeleton threshold ({skeleton_threshold}) too high or not set")
        print("   SOLUTION: Change 'confidence_threshold': 0.2 in skeleton_based method")
    
    if joint_threshold != 'NOT_SET' and joint_threshold <= 0.2:
        print(f"✓ Joint threshold ({joint_threshold}) looks good")
    else:
        print(f"❌ Joint threshold ({joint_threshold}) too high or not set")
        print("   SOLUTION: Change 'joint_confidence_threshold': 0.1 in SKELETON_ORIENTATION")
        
    if movement_threshold != 'NOT_SET' and movement_threshold <= 0.05:
        print(f"✓ Movement threshold ({movement_threshold}) looks good")
    else:
        print(f"❌ Movement threshold ({movement_threshold}) too high or not set")
        print("   SOLUTION: Change 'min_movement_threshold': 0.03 in movement_based method")
    
    print(f"\nFull skeleton method config:")
    print(f"  {skeleton_method}")
    
    print(f"\nFull skeleton orientation config:")
    print(f"  {SKELETON_ORIENTATION}")
    
except ImportError as e:
    print(f"❌ Could not import orientation_config: {e}")
    print("Make sure you're running from the project root directory")
except Exception as e:
    print(f"❌ Error checking configuration: {e}")

print("\n🎯 NEXT STEPS:")
print("1. If any thresholds are too high, edit config/orientation_config.py")
print("2. Look for the exact variable names shown above")
print("3. Change the values to the recommended ones")
print("4. Save the file and run your test again")
print("5. You should see skeleton/movement method successes > 0")