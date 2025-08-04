#!/usr/bin/env python3
"""
Comprehensive diagnostic script for Phase 2 Step 2 skeleton tracking failures
Analyzes exactly why skeleton and movement detection are not working despite detecting points
"""

import sys
import os
import logging
import numpy as np
import json
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root / "config"))

class SkeletonFailureDiagnostic:
    """Comprehensive diagnostic for skeleton tracking failures."""
    
    def __init__(self):
        self.issues_found = []
        self.recommendations = []
        
    def run_complete_diagnosis(self):
        """Run all diagnostic tests."""
        logger.info("🔍 SKELETON TRACKING FAILURE DIAGNOSTIC")
        logger.info("=" * 60)
        
        # Test 1: Check YOLO availability and functionality
        self.test_yolo_installation()
        
        # Test 2: Check configuration values
        self.test_configuration_thresholds()
        
        # Test 3: Simulate actual skeleton detection scenario
        self.test_skeleton_detection_pipeline()
        
        # Test 4: Check movement detection logic
        self.test_movement_detection_logic()
        
        # Test 5: Analyze likely failure points
        self.analyze_common_failure_points()
        
        # Generate final diagnosis
        self.generate_diagnosis_report()
    
    def test_yolo_installation(self):
        """Test YOLO pose model installation and functionality."""
        logger.info("\n=== TEST 1: YOLO Installation & Functionality ===")
        
        try:
            from ultralytics import YOLO
            logger.info("✓ ultralytics library available")
            
            # Try to load the pose model
            try:
                model = YOLO('yolov8n-pose.pt')
                logger.info("✓ yolov8n-pose.pt model loaded successfully")
                
                # Test pose detection on dummy image
                test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
                results = model(test_image, verbose=False)
                
                if results and len(results) > 0:
                    result = results[0]
                    if hasattr(result, 'keypoints'):
                        logger.info("✓ Pose detection pipeline working")
                        
                        # Check keypoint structure
                        if result.keypoints is not None and result.keypoints.data is not None:
                            keypoints_shape = result.keypoints.data.shape
                            logger.info(f"✓ Keypoints structure: {keypoints_shape}")
                            logger.info("  Expected: (num_persons, 17, 3) where 3 = (x, y, confidence)")
                        else:
                            logger.warning("⚠️ Keypoints data is None (normal for empty image)")
                    else:
                        logger.error("❌ No keypoints attribute in results")
                        self.issues_found.append("yolo_no_keypoints_attr")
                else:
                    logger.error("❌ No results from YOLO detection")
                    self.issues_found.append("yolo_no_results")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load YOLO pose model: {e}")
                self.issues_found.append("yolo_model_load_failed")
                self.recommendations.append("Run: pip install --upgrade ultralytics")
                
        except ImportError as e:
            logger.error(f"❌ ultralytics not available: {e}")
            self.issues_found.append("ultralytics_missing")
            self.recommendations.append("Install ultralytics: pip install ultralytics")
    
    def test_configuration_thresholds(self):
        """Test if configuration thresholds are too strict."""
        logger.info("\n=== TEST 2: Configuration Threshold Analysis ===")
        
        try:
            from orientation_config import ORIENTATION_METHODS, SKELETON_ORIENTATION
            
            # Analyze skeleton method thresholds
            skeleton_method = ORIENTATION_METHODS.get('skeleton_based', {})
            skeleton_config = SKELETON_ORIENTATION
            
            logger.info("Current thresholds:")
            
            # Check skeleton method confidence threshold
            skel_conf_thresh = skeleton_method.get('confidence_threshold', 0.6)
            logger.info(f"  Skeleton method confidence threshold: {skel_conf_thresh}")
            if skel_conf_thresh > 0.4:
                logger.warning(f"  ⚠️ Skeleton confidence threshold ({skel_conf_thresh}) may be too high")
                self.issues_found.append(f"skeleton_confidence_too_high_{skel_conf_thresh}")
                self.recommendations.append(f"Lower skeleton confidence threshold to 0.2-0.3")
            
            # Check joint confidence threshold
            joint_conf_thresh = skeleton_config.get('joint_confidence_threshold', 0.3)
            logger.info(f"  Joint confidence threshold: {joint_conf_thresh}")
            if joint_conf_thresh > 0.3:
                logger.warning(f"  ⚠️ Joint confidence threshold ({joint_conf_thresh}) may be too high")
                self.issues_found.append(f"joint_confidence_too_high_{joint_conf_thresh}")
                self.recommendations.append(f"Lower joint confidence threshold to 0.1-0.2")
            
            # Check required joint count
            required_joints = skeleton_config.get('required_joint_count', 2)
            logger.info(f"  Required joint count: {required_joints}")
            if required_joints > 3:
                logger.warning(f"  ⚠️ Required joint count ({required_joints}) may be too high")
                self.issues_found.append(f"required_joints_too_high_{required_joints}")
                self.recommendations.append(f"Lower required joint count to 2")
            
            # Check movement thresholds
            movement_method = ORIENTATION_METHODS.get('movement_based', {})
            movement_thresh = movement_method.get('min_movement_threshold', 0.1)
            logger.info(f"  Movement threshold: {movement_thresh}m")
            if movement_thresh > 0.1:
                logger.warning(f"  ⚠️ Movement threshold ({movement_thresh}m) may be too high")
                self.issues_found.append(f"movement_threshold_too_high_{movement_thresh}")
                self.recommendations.append(f"Lower movement threshold to 0.02-0.05m")
                
        except Exception as e:
            logger.error(f"❌ Could not load configuration: {e}")
            self.issues_found.append("config_load_failed")
    
    def test_skeleton_detection_pipeline(self):
        """Simulate the actual skeleton detection pipeline."""
        logger.info("\n=== TEST 3: Skeleton Detection Pipeline Simulation ===")
        
        try:
            from ultralytics import YOLO
            
            # Load model
            model = YOLO('yolov8n-pose.pt')
            
            # Simulate different detection scenarios
            scenarios = [
                ("Full person - good lighting", self._create_test_person_image("good")),
                ("Partial person - medium lighting", self._create_test_person_image("medium")),
                ("Small person ROI", self._create_test_person_image("small")),
                ("Poor lighting", self._create_test_person_image("poor")),
            ]
            
            for scenario_name, test_image in scenarios:
                logger.info(f"\n--- Testing: {scenario_name} ---")
                
                # Run detection
                results = model(test_image, verbose=False)
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                        keypoints_data = result.keypoints.data
                        
                        if keypoints_data is not None and len(keypoints_data) > 0:
                            # Analyze first person's keypoints
                            person_keypoints = keypoints_data[0]  # Shape: (17, 3)
                            
                            # Count confident keypoints
                            confident_keypoints = []
                            for i, (x, y, conf) in enumerate(person_keypoints):
                                if conf > 0.1:  # Very low threshold for testing
                                    confident_keypoints.append((i, conf.item()))
                            
                            logger.info(f"  Detected {len(confident_keypoints)} keypoints with conf > 0.1")
                            logger.info(f"  Top keypoints: {confident_keypoints[:5]}")
                            
                            # Simulate orientation estimation logic
                            self._simulate_orientation_calculation(person_keypoints, scenario_name)
                        else:
                            logger.info(f"  No keypoints detected for {scenario_name}")
                    else:
                        logger.info(f"  No keypoints attribute for {scenario_name}")
                else:
                    logger.info(f"  No detection results for {scenario_name}")
                    
        except Exception as e:
            logger.error(f"❌ Pipeline simulation failed: {e}")
            self.issues_found.append("pipeline_simulation_failed")
    
    def _create_test_person_image(self, quality):
        """Create test images of different qualities."""
        if quality == "good":
            # Large, well-lit image
            img = np.ones((400, 300, 3), dtype=np.uint8) * 180
        elif quality == "medium":
            # Medium size, medium lighting
            img = np.ones((300, 200, 3), dtype=np.uint8) * 140
        elif quality == "small":
            # Small ROI
            img = np.ones((150, 100, 3), dtype=np.uint8) * 160
        else:  # poor
            # Poor lighting
            img = np.ones((300, 200, 3), dtype=np.uint8) * 80
        
        # Add some noise to make it more realistic
        noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
        try:
            import cv2
            img = cv2.add(img, noise)
        except ImportError:
            img = np.clip(img.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)
        return img
    
    def _simulate_orientation_calculation(self, keypoints, scenario_name):
        """Simulate the orientation calculation logic."""
        try:
            from orientation_config import SKELETON_ORIENTATION
            
            # Extract key joints for orientation
            joint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            joint_conf_thresh = SKELETON_ORIENTATION.get('joint_confidence_threshold', 0.3)
            required_joint_count = SKELETON_ORIENTATION.get('required_joint_count', 2)
            
            # Count joints above threshold
            confident_joints = []
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > joint_conf_thresh:
                    confident_joints.append((joint_names[i], conf.item()))
            
            logger.info(f"    Joints above threshold ({joint_conf_thresh}): {len(confident_joints)}")
            logger.info(f"    Required joints: {required_joint_count}")
            
            if len(confident_joints) >= required_joint_count:
                logger.info(f"    ✓ {scenario_name}: Would PASS joint count test")
                
                # Try to calculate orientation from shoulders
                left_shoulder = keypoints[5]  # Index 5 = left_shoulder
                right_shoulder = keypoints[6]  # Index 6 = right_shoulder
                
                if left_shoulder[2] > joint_conf_thresh and right_shoulder[2] > joint_conf_thresh:
                    logger.info(f"    ✓ {scenario_name}: Would PASS shoulder orientation test")
                    
                    # Calculate orientation angle
                    shoulder_vector = (right_shoulder[0] - left_shoulder[0], 
                                     right_shoulder[1] - left_shoulder[1])
                    angle = np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
                    
                    # Simulate confidence calculation
                    base_confidence = 0.9
                    joint_bonus = len(confident_joints) * 0.05
                    final_confidence = min(1.0, base_confidence + joint_bonus)
                    
                    logger.info(f"    Calculated angle: {angle:.1f}°, confidence: {final_confidence:.3f}")
                    
                    # Check against method threshold
                    from orientation_config import ORIENTATION_METHODS
                    method_thresh = ORIENTATION_METHODS.get('skeleton_based', {}).get('confidence_threshold', 0.6)
                    
                    if final_confidence >= method_thresh:
                        logger.info(f"    ✓ {scenario_name}: Would PASS final confidence test")
                    else:
                        logger.warning(f"    ❌ {scenario_name}: Would FAIL confidence test ({final_confidence:.3f} < {method_thresh})")
                        self.issues_found.append(f"confidence_too_low_{scenario_name}")
                else:
                    logger.warning(f"    ❌ {scenario_name}: Would FAIL shoulder detection test")
                    self.issues_found.append(f"shoulder_detection_failed_{scenario_name}")
            else:
                logger.warning(f"    ❌ {scenario_name}: Would FAIL joint count test")
                self.issues_found.append(f"joint_count_failed_{scenario_name}")
                
        except Exception as e:
            logger.error(f"    ❌ Simulation failed for {scenario_name}: {e}")
    
    def test_movement_detection_logic(self):
        """Test movement detection logic."""
        logger.info("\n=== TEST 4: Movement Detection Logic ===")
        
        try:
            from orientation_config import ORIENTATION_METHODS
            
            movement_config = ORIENTATION_METHODS.get('movement_based', {})
            min_movement = movement_config.get('min_movement_threshold', 0.1)
            
            logger.info(f"Movement threshold: {min_movement}m")
            
            # Simulate different movement scenarios
            movement_scenarios = [
                ("Walking normally", 0.15),      # Should pass
                ("Slow walking", 0.08),          # Might fail
                ("Standing still", 0.01),        # Should fail
                ("Gesturing while stationary", 0.03),  # Might fail
                ("Quick movement", 0.25),        # Should pass
            ]
            
            for scenario, movement_magnitude in movement_scenarios:
                if movement_magnitude >= min_movement:
                    logger.info(f"  ✓ {scenario} ({movement_magnitude}m): Would PASS")
                else:
                    logger.warning(f"  ❌ {scenario} ({movement_magnitude}m): Would FAIL")
                    if "stationary" not in scenario.lower():
                        self.issues_found.append(f"movement_threshold_too_high_for_{scenario}")
            
            # Check if movement threshold is realistic
            if min_movement > 0.1:
                logger.warning(f"⚠️ Movement threshold ({min_movement}m) may be too high for normal testing")
                self.recommendations.append("Consider lowering movement threshold to 0.02-0.05m")
                
        except Exception as e:
            logger.error(f"❌ Movement logic test failed: {e}")
    
    def analyze_common_failure_points(self):
        """Analyze common points where skeleton tracking fails."""
        logger.info("\n=== TEST 5: Common Failure Point Analysis ===")
        
        common_issues = {
            "Confidence Thresholds Too High": [
                "joint_confidence_threshold > 0.3",
                "skeleton confidence_threshold > 0.4",
                "movement min_movement_threshold > 0.1"
            ],
            "Detection Environment": [
                "Poor lighting conditions",
                "Person too far from camera (>3m)",
                "Person partially occluded",
                "Person wearing complex patterns"
            ],
            "Technical Issues": [
                "YOLO model not downloaded properly",
                "OpenCV version compatibility",
                "Numpy array shape mismatches",
                "Coordinate system transformations"
            ],
            "Testing Methodology": [
                "Not moving enough during testing",
                "Standing at wrong angle to camera",
                "Testing duration too short",
                "Camera calibration issues"
            ]
        }
        
        for category, potential_issues in common_issues.items():
            logger.info(f"\n--- {category} ---")
            for issue in potential_issues:
                logger.info(f"  • {issue}")
        
        # Check if we detected any of these issues
        logger.info(f"\n--- Issues Detected in This Session ---")
        if self.issues_found:
            for issue in self.issues_found:
                logger.info(f"  ❌ {issue}")
        else:
            logger.info("  ✓ No major technical issues detected")
    
    def generate_diagnosis_report(self):
        """Generate final diagnosis and recommendations."""
        logger.info("\n" + "=" * 60)
        logger.info("FINAL DIAGNOSIS REPORT")
        logger.info("=" * 60)
        
        # Categorize issues
        critical_issues = []
        configuration_issues = []
        environmental_issues = []
        
        for issue in self.issues_found:
            if "missing" in issue or "failed" in issue or "load" in issue:
                critical_issues.append(issue)
            elif "threshold" in issue or "confidence" in issue:
                configuration_issues.append(issue)
            else:
                environmental_issues.append(issue)
        
        # Report critical issues
        if critical_issues:
            logger.info("\n🚨 CRITICAL ISSUES (Fix These First):")
            for issue in critical_issues:
                logger.info(f"  ❌ {issue}")
        
        # Report configuration issues
        if configuration_issues:
            logger.info("\n⚙️ CONFIGURATION ISSUES (Likely Main Problem):")
            for issue in configuration_issues:
                logger.info(f"  ⚠️ {issue}")
        
        # Report environmental issues
        if environmental_issues:
            logger.info("\n🌍 ENVIRONMENTAL/TESTING ISSUES:")
            for issue in environmental_issues:
                logger.info(f"  ℹ️ {issue}")
        
        # Provide specific recommendations
        logger.info("\n🔧 RECOMMENDED FIXES:")
        if not self.recommendations:
            self.recommendations = [
                "Lower confidence thresholds in orientation_config.py",
                "Ensure good lighting during testing",
                "Move around actively during testing",
                "Stand 1.5-2m from camera"
            ]
        
        for i, rec in enumerate(self.recommendations, 1):
            logger.info(f"  {i}. {rec}")
        
        # Generate quick fix config
        self.generate_quick_fix_config()
        
        # Final verdict
        logger.info("\n" + "=" * 60)
        if critical_issues:
            logger.info("🔴 VERDICT: Critical technical issues need fixing first")
        elif configuration_issues:
            logger.info("🟡 VERDICT: Configuration thresholds likely too strict")
            logger.info("   Use the generated debug config to test with lower thresholds")
        else:
            logger.info("🟢 VERDICT: Technical setup appears correct")
            logger.info("   Issues likely environmental - improve lighting and movement")
    
    def generate_quick_fix_config(self):
        """Generate a quick fix configuration file."""
        logger.info("\n📄 Generating quick fix configuration...")
        
        quick_fix_config = '''#!/usr/bin/env python3
"""
QUICK FIX Configuration for Phase 2 Step 2
Use this configuration to test with very low thresholds
"""

# Replace these values in your orientation_config.py for debugging

DEBUG_ORIENTATION_METHODS = {
    'skeleton_based': {
        'enabled': True,
        'priority': 1,
        'confidence_threshold': 0.1,  # Very low for debugging
        'required_joints': ['left_shoulder', 'right_shoulder'],
        'fallback_joints': ['left_shoulder', 'right_shoulder'],
    },
    'movement_based': {
        'enabled': True,
        'priority': 2,
        'min_movement_threshold': 0.02,  # Very low for debugging
        'smoothing_window': 3,
        'confidence_decay': 0.9,
    },
    'depth_gradient': {
        'enabled': True,
        'priority': 3,
        'roi_size': (60, 120),
        'gradient_threshold': 0.02,  # Very low for debugging
        'confidence_threshold': 0.1,  # Very low for debugging
    }
}

DEBUG_SKELETON_ORIENTATION = {
    'joint_confidence_threshold': 0.1,  # Very low for debugging
    'required_joint_count': 2,
    'shoulder_vector_weight': 0.7,
    'head_vector_weight': 0.3,
    'temporal_smoothing': True,
    'smoothing_alpha': 0.7,
    'max_angle_change': 45,
    'use_face_keypoints': True,
    'face_keypoints': ['nose', 'left_eye', 'right_eye'],
    'face_confidence_threshold': 0.1,  # Very low for debugging
}

# Usage:
# 1. Backup your current orientation_config.py
# 2. Replace ORIENTATION_METHODS with DEBUG_ORIENTATION_METHODS
# 3. Replace SKELETON_ORIENTATION with DEBUG_SKELETON_ORIENTATION  
# 4. Run your test again
# 5. If it works, gradually increase thresholds until you find the right balance
'''
        
        debug_config_path = project_root / "debug_orientation_config.py"
        with open(debug_config_path, 'w') as f:
            f.write(quick_fix_config)
        
        logger.info(f"✓ Quick fix config saved to: {debug_config_path}")
        logger.info("  Instructions:")
        logger.info("  1. Backup your current config/orientation_config.py")
        logger.info("  2. Replace the values as shown in debug_orientation_config.py")
        logger.info("  3. Run python run_phase2_step2.py again")
        logger.info("  4. Look for skeleton/movement method successes")

def main():
    """Run the complete diagnostic."""
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available for some tests")
    
    diagnostic = SkeletonFailureDiagnostic()
    diagnostic.run_complete_diagnosis()
    
    logger.info("\n🎯 Next Steps:")
    logger.info("1. Apply the recommended fixes above")
    logger.info("2. Use the generated debug_orientation_config.py")
    logger.info("3. Run the enhanced debug version: python enhanced_orientation_debug.py")
    logger.info("4. Look for skeleton points and their confidence values")
    logger.info("5. Gradually increase thresholds once basic detection works")

if __name__ == "__main__":
    main()