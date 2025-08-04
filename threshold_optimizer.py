#!/usr/bin/env python3
"""
Orientation Threshold Optimizer
Interactive script to find optimal confidence thresholds for skeleton tracking
"""

import sys
import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "config"))

def test_threshold_combination(joint_threshold, skeleton_threshold, movement_threshold):
    """Test a specific combination of thresholds."""
    try:
        from ultralytics import YOLO
        
        # Load pose model
        model = YOLO('yolov8n-pose.pt')
        
        # Create test scenarios
        test_results = {
            'skeleton_detections': 0,
            'confident_skeletons': 0,
            'movement_detections': 0,
            'total_tests': 5
        }
        
        # Test with different synthetic scenarios
        for i in range(test_results['total_tests']):
            # Create test image with some variation
            brightness = 120 + (i * 20)  # Vary brightness
            test_img = np.ones((300, 250, 3), dtype=np.uint8) * brightness
            
            # Add noise
            noise = np.random.randint(0, 30, test_img.shape, dtype=np.uint8)
            test_img = cv2.add(test_img, noise)
            
            # Run detection
            results = model(test_img, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.data
                    
                    if keypoints_data is not None and len(keypoints_data) > 0:
                        test_results['skeleton_detections'] += 1
                        
                        # Check if skeleton would pass thresholds
                        person_keypoints = keypoints_data[0]
                        
                        # Count confident joints
                        confident_joints = 0
                        for x, y, conf in person_keypoints:
                            if conf > joint_threshold:
                                confident_joints += 1
                        
                        # Simulate orientation calculation
                        if confident_joints >= 2:  # Minimum joints
                            # Check shoulder keypoints (indices 5, 6)
                            left_shoulder = person_keypoints[5]
                            right_shoulder = person_keypoints[6]
                            
                            if (left_shoulder[2] > joint_threshold and 
                                right_shoulder[2] > joint_threshold):
                                
                                # Calculate confidence
                                base_confidence = 0.8
                                joint_bonus = confident_joints * 0.02
                                final_confidence = min(1.0, base_confidence + joint_bonus)
                                
                                if final_confidence >= skeleton_threshold:
                                    test_results['confident_skeletons'] += 1
            
            # Simulate movement test (simple)
            simulated_movement = 0.03 + (i * 0.02)  # Vary movement
            if simulated_movement >= movement_threshold:
                test_results['movement_detections'] += 1
        
        return test_results
        
    except Exception as e:
        logger.error(f"Threshold test failed: {e}")
        return None

def find_optimal_thresholds():
    """Find optimal threshold combinations through testing."""
    logger.info("🔍 ORIENTATION THRESHOLD OPTIMIZER")
    logger.info("=" * 50)
    
    # Define threshold ranges to test
    joint_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    skeleton_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    movement_thresholds = [0.02, 0.03, 0.05, 0.08, 0.1]
    
    best_combinations = []
    
    logger.info("Testing threshold combinations...")
    logger.info("Format: (joint_thresh, skeleton_thresh, movement_thresh) -> success_rate")
    
    total_tests = len(joint_thresholds) * len(skeleton_thresholds) * len(movement_thresholds)
    current_test = 0
    
    for joint_thresh in joint_thresholds:
        for skeleton_thresh in skeleton_thresholds:
            for movement_thresh in movement_thresholds:
                current_test += 1
                
                # Show progress
                if current_test % 10 == 0:
                    progress = (current_test / total_tests) * 100
                    logger.info(f"Progress: {progress:.1f}% ({current_test}/{total_tests})")
                
                # Test this combination
                results = test_threshold_combination(joint_thresh, skeleton_thresh, movement_thresh)
                
                if results:
                    # Calculate success rates
                    skeleton_success_rate = results['confident_skeletons'] / results['total_tests']
                    movement_success_rate = results['movement_detections'] / results['total_tests']
                    
                    # Combined score (weighted)
                    combined_score = (skeleton_success_rate * 0.7) + (movement_success_rate * 0.3)
                    
                    # Store if promising
                    if combined_score >= 0.4:  # At least 40% success
                        best_combinations.append({
                            'joint_threshold': joint_thresh,
                            'skeleton_threshold': skeleton_thresh,
                            'movement_threshold': movement_thresh,
                            'skeleton_success': skeleton_success_rate,
                            'movement_success': movement_success_rate,
                            'combined_score': combined_score
                        })
    
    # Sort by combined score
    best_combinations.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Report results
    logger.info("\n" + "=" * 50)
    logger.info("OPTIMAL THRESHOLD COMBINATIONS")
    logger.info("=" * 50)
    
    if best_combinations:
        logger.info("Top 5 combinations:")
        for i, combo in enumerate(best_combinations[:5], 1):
            logger.info(f"\n{i}. Joint: {combo['joint_threshold']:.2f}, "
                       f"Skeleton: {combo['skeleton_threshold']:.2f}, "
                       f"Movement: {combo['movement_threshold']:.2f}")
            logger.info(f"   Skeleton success: {combo['skeleton_success']:.1%}")
            logger.info(f"   Movement success: {combo['movement_success']:.1%}")
            logger.info(f"   Combined score: {combo['combined_score']:.3f}")
        
        # Generate config for best combination
        best = best_combinations[0]
        generate_optimized_config(best)
        
    else:
        logger.warning("No successful combinations found!")
        logger.info("This suggests a more fundamental issue:")
        logger.info("1. YOLO pose model may not be working")
        logger.info("2. Environment may be too challenging")
        logger.info("3. Need to test with real camera data")
        
        # Generate very permissive config as fallback
        fallback_config = {
            'joint_threshold': 0.1,
            'skeleton_threshold': 0.1,
            'movement_threshold': 0.02
        }
        generate_optimized_config(fallback_config)

def generate_optimized_config(best_combo):
    """Generate optimized configuration file."""
    logger.info(f"\n📄 Generating optimized configuration...")
    
    config_content = f'''#!/usr/bin/env python3
"""
OPTIMIZED Configuration for Phase 2 Step 2
Generated by threshold optimizer - these values should work better
"""

# Optimized thresholds based on testing
OPTIMIZED_ORIENTATION_METHODS = {{
    'skeleton_based': {{
        'enabled': True,
        'priority': 1,
        'confidence_threshold': {best_combo.get('skeleton_threshold', 0.2):.2f},
        'required_joints': ['left_shoulder', 'right_shoulder', 'neck', 'nose'],
        'fallback_joints': ['left_shoulder', 'right_shoulder'],
    }},
    'movement_based': {{
        'enabled': True,
        'priority': 2,
        'min_movement_threshold': {best_combo.get('movement_threshold', 0.05):.3f},
        'smoothing_window': 5,
        'confidence_decay': 0.9,
    }},
    'depth_gradient': {{
        'enabled': True,
        'priority': 3,
        'roi_size': (60, 120),
        'gradient_threshold': 0.03,
        'confidence_threshold': 0.2,
    }}
}}

OPTIMIZED_SKELETON_ORIENTATION = {{
    'joint_confidence_threshold': {best_combo.get('joint_threshold', 0.2):.2f},
    'required_joint_count': 2,
    'shoulder_vector_weight': 0.7,
    'head_vector_weight': 0.3,
    'temporal_smoothing': True,
    'smoothing_alpha': 0.7,
    'max_angle_change': 45,
    'use_face_keypoints': True,
    'face_keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
    'face_confidence_threshold': {best_combo.get('joint_threshold', 0.2):.2f},
}}

# Instructions for use:
# 1. Backup your current config/orientation_config.py
# 2. Replace ORIENTATION_METHODS with OPTIMIZED_ORIENTATION_METHODS
# 3. Replace SKELETON_ORIENTATION with OPTIMIZED_SKELETON_ORIENTATION
# 4. Test with: python run_phase2_step2.py
# 5. If still not working, try the debug version with even lower thresholds

# Current threshold values:
# - Joint confidence threshold: {best_combo.get('joint_threshold', 0.2):.2f}
# - Skeleton confidence threshold: {best_combo.get('skeleton_threshold', 0.2):.2f}  
# - Movement threshold: {best_combo.get('movement_threshold', 0.05):.3f}m

print("Optimized configuration loaded!")
print(f"Joint threshold: {best_combo.get('joint_threshold', 0.2):.2f}")
print(f"Skeleton threshold: {best_combo.get('skeleton_threshold', 0.2):.2f}")
print(f"Movement threshold: {best_combo.get('movement_threshold', 0.05):.3f}m")
'''
    
    config_path = project_root / "optimized_orientation_config.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"✓ Optimized config saved to: {config_path}")
    logger.info("  Use these values in your orientation_config.py")

def interactive_threshold_tester():
    """Interactive threshold testing mode."""
    logger.info("\n🎮 INTERACTIVE THRESHOLD TESTER")
    logger.info("Test specific threshold combinations manually")
    logger.info("Press 'q' to quit, Enter to test")
    
    while True:
        try:
            print("\n" + "-" * 40)
            joint_thresh = float(input("Joint confidence threshold (0.1-0.5): ") or "0.2")
            skeleton_thresh = float(input("Skeleton confidence threshold (0.1-0.5): ") or "0.3")
            movement_thresh = float(input("Movement threshold in meters (0.01-0.2): ") or "0.05")
            
            print("\nTesting combination...")
            results = test_threshold_combination(joint_thresh, skeleton_thresh, movement_thresh)
            
            if results:
                skeleton_rate = results['confident_skeletons'] / results['total_tests']
                movement_rate = results['movement_detections'] / results['total_tests']
                
                print(f"Results:")
                print(f"  Skeleton success rate: {skeleton_rate:.1%}")
                print(f"  Movement success rate: {movement_rate:.1%}")
                print(f"  Combined score: {(skeleton_rate + movement_rate) / 2:.1%}")
                
                if skeleton_rate > 0.6:
                    print("  ✓ Skeleton detection looks good!")
                elif skeleton_rate > 0.3:
                    print("  ⚠️ Skeleton detection marginal - might work with real data")
                else:
                    print("  ❌ Skeleton detection poor - try lower thresholds")
            else:
                print("  ❌ Test failed - check YOLO installation")
            
            continue_test = input("\nTest another combination? (y/n): ").lower()
            if continue_test in ['n', 'no', 'q', 'quit']:
                break
                
        except KeyboardInterrupt:
            break
        except ValueError:
            print("Invalid input - please enter numbers")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function - choose between automatic optimization or interactive testing."""
    logger.info("ORIENTATION THRESHOLD OPTIMIZER")
    logger.info("=" * 40)
    
    print("Choose mode:")
    print("1. Automatic optimization (recommended)")
    print("2. Interactive threshold testing")
    print("3. Generate safe/conservative config")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            find_optimal_thresholds()
        elif choice == "2":
            interactive_threshold_tester()
        elif choice == "3":
            # Generate safe config
            safe_config = {
                'joint_threshold': 0.15,
                'skeleton_threshold': 0.2,
                'movement_threshold': 0.03
            }
            generate_optimized_config(safe_config)
            logger.info("Generated conservative/safe configuration")
        else:
            logger.info("Invalid choice, running automatic optimization...")
            find_optimal_thresholds()
            
    except KeyboardInterrupt:
        logger.info("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {e}")
    
    print("\n🎯 Next Steps:")
    print("1. Use the generated optimized_orientation_config.py")
    print("2. Update your config/orientation_config.py with these values")
    print("3. Run: python run_phase2_step2.py")
    print("4. If still failing, run the diagnostic script")

if __name__ == "__main__":
    main()