#!/usr/bin/env python3
"""
Fixed Orientation Debugging Script - Phase 2 Step 2
Works with your actual project structure and focuses on the real issues
"""

import sys
import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from collections import deque
import json

# Fix path issues - add all necessary paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'config'))

# Enhanced logging with UTF-8 encoding to fix Unicode issues
log_dir = Path("logs/debug")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"orientation_debug_{time.strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OrientationDebugger:
    """Focused debugger for orientation detection issues."""
    
    def __init__(self):
        self.issues_found = []
        self.debug_data = {}
        
    def run_diagnosis(self):
        """Run focused diagnostic tests."""
        logger.info("="*80)
        logger.info("ORIENTATION DEBUGGING - FOCUSING ON YOUR ACTUAL ISSUE")
        logger.info("="*80)
        
        # Test 1: Quick environment check
        logger.info("\n[TEST 1] Quick Environment Check...")
        self.quick_environment_check()
        
        # Test 2: Direct skeleton method test
        logger.info("\n[TEST 2] Direct Skeleton Method Test...")
        self.test_skeleton_directly()
        
        # Test 3: Simulate your exact scenario
        logger.info("\n[TEST 3] Simulating Your Exact Scenario...")
        self.simulate_phase2_step2()
        
        # Generate focused report
        self.generate_focused_report()
        
    def quick_environment_check(self):
        """Quick check of critical components."""
        try:
            # Check YOLO
            from ultralytics import YOLO
            model = YOLO('yolov8n-pose.pt')
            logger.info("✓ YOLO pose model loaded")
            
            # Check config
            from orientation_config import ORIENTATION_METHODS, SKELETON_ORIENTATION
            logger.info("✓ Configuration loaded")
            logger.info(f"  Skeleton confidence: {ORIENTATION_METHODS['skeleton_based']['confidence_threshold']}")
            logger.info(f"  Joint confidence: {SKELETON_ORIENTATION['joint_confidence_threshold']}")
            
        except Exception as e:
            logger.error(f"Environment check failed: {e}")
            self.issues_found.append("environment_check_failed")
            
    def test_skeleton_directly(self):
        """Direct test of skeleton detection on sample images."""
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n-pose.pt')
            
            # Create test scenarios
            logger.info("\nTesting skeleton detection on different scenarios...")
            
            # Test 1: Full frame detection
            logger.info("\n--- Test 1: Full Frame Detection ---")
            test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray image
            results = model(test_image, verbose=False)
            self.analyze_pose_results(results, "full_frame")
            
            # Test 2: Small ROI (typical person bbox)
            logger.info("\n--- Test 2: Typical Person ROI ---")
            roi_sizes = [(150, 200), (100, 150), (80, 120), (50, 80)]
            
            for w, h in roi_sizes:
                roi = np.ones((h, w, 3), dtype=np.uint8) * 128
                logger.info(f"\nTesting ROI size: {w}x{h}")
                results = model(roi, verbose=False)
                self.analyze_pose_results(results, f"roi_{w}x{h}")
                
            # Test 3: Load and test actual captured frame if exists
            if Path("debug_color_frame.jpg").exists():
                logger.info("\n--- Test 3: Actual Captured Frame ---")
                actual_frame = cv2.imread("debug_color_frame.jpg")
                results = model(actual_frame, verbose=False)
                self.analyze_pose_results(results, "actual_frame")
                
        except Exception as e:
            logger.error(f"Skeleton test failed: {e}")
            self.issues_found.append("skeleton_test_failed")
            
    def analyze_pose_results(self, results, test_name):
        """Analyze pose detection results in detail."""
        if not results or len(results) == 0:
            logger.warning(f"  {test_name}: No results returned")
            self.issues_found.append(f"no_results_{test_name}")
            return
            
        result = results[0]
        
        # Check if keypoints exist
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            logger.warning(f"  {test_name}: No keypoints attribute")
            self.issues_found.append(f"no_keypoints_attr_{test_name}")
            return
            
        # Check keypoints data
        if result.keypoints.data is None or len(result.keypoints.data) == 0:
            logger.warning(f"  {test_name}: No keypoints data")
            self.issues_found.append(f"no_keypoints_data_{test_name}")
            return
            
        # Analyze keypoints
        keypoints_data = result.keypoints.data
        logger.info(f"  {test_name}: Found {len(keypoints_data)} person(s)")
        
        for person_idx, person_keypoints in enumerate(keypoints_data):
            kpts = person_keypoints.cpu().numpy()
            visible_count = np.sum(kpts[:, 2] > 0.3)
            high_conf_count = np.sum(kpts[:, 2] > 0.5)
            
            logger.info(f"    Person {person_idx}: {visible_count} visible keypoints (>{high_conf_count} high conf)")
            
            # Check critical keypoints
            critical_kpts = {
                'nose': 0,
                'left_shoulder': 5,
                'right_shoulder': 6
            }
            
            for name, idx in critical_kpts.items():
                conf = kpts[idx, 2]
                status = "✓" if conf > 0.3 else "✗"
                logger.info(f"      {status} {name}: {conf:.3f}")
                
    def simulate_phase2_step2(self):
        """Simulate the exact scenario from run_phase2_step2.py"""
        try:
            logger.info("\nSimulating orientation estimation process...")
            
            # Import what we can
            from ultralytics import YOLO
            from orientation_config import ORIENTATION_METHODS, SKELETON_ORIENTATION
            
            model = YOLO('yolov8n-pose.pt')
            
            # Simulate person detection with typical bounding boxes
            test_bboxes = [
                (100, 100, 150, 200),  # Normal sized person
                (50, 50, 80, 120),     # Small person
                (10, 10, 50, 80),      # Very small person
                (550, 350, 80, 120),   # Near edge
                (600, 400, 80, 120),   # Partially out of bounds
            ]
            
            frame_shape = (480, 640, 3)
            test_frame = np.ones(frame_shape, dtype=np.uint8) * 128
            
            for bbox_idx, bbox in enumerate(test_bboxes):
                x, y, w, h = bbox
                logger.info(f"\n--- Testing bbox {bbox_idx}: x={x}, y={y}, w={w}, h={h} ---")
                
                # Check bounds
                if x < 0 or y < 0 or x + w > frame_shape[1] or y + h > frame_shape[0]:
                    logger.warning("  ISSUE: Bbox out of bounds!")
                    self.issues_found.append(f"bbox_out_of_bounds_{bbox_idx}")
                    
                    # Fix bounds for ROI extraction
                    x_start = max(0, x)
                    y_start = max(0, y)
                    x_end = min(frame_shape[1], x + w)
                    y_end = min(frame_shape[0], y + h)
                    
                    fixed_w = x_end - x_start
                    fixed_h = y_end - y_start
                    logger.info(f"  Fixed to: x={x_start}, y={y_start}, w={fixed_w}, h={fixed_h}")
                else:
                    x_start, y_start = x, y
                    x_end, y_end = x + w, y + h
                
                # Extract ROI
                try:
                    roi = test_frame[y_start:y_end, x_start:x_end]
                    logger.info(f"  ROI shape: {roi.shape}")
                    
                    if roi.size == 0:
                        logger.warning("  ISSUE: Empty ROI!")
                        self.issues_found.append(f"empty_roi_{bbox_idx}")
                        continue
                        
                    # Run pose detection
                    results = model(roi, verbose=False)
                    
                    if results and len(results) > 0 and hasattr(results[0], 'keypoints'):
                        if results[0].keypoints.data is not None and len(results[0].keypoints.data) > 0:
                            kpts = results[0].keypoints.data[0].cpu().numpy()
                            visible = np.sum(kpts[:, 2] > SKELETON_ORIENTATION['joint_confidence_threshold'])
                            logger.info(f"  Keypoints detected: {visible} visible")
                            
                            # Check if enough for orientation
                            if visible >= SKELETON_ORIENTATION['required_joint_count']:
                                logger.info("  ✓ Sufficient keypoints for orientation")
                            else:
                                logger.warning("  ISSUE: Insufficient keypoints for orientation")
                                self.issues_found.append(f"insufficient_keypoints_{bbox_idx}")
                        else:
                            logger.warning("  ISSUE: No person detected in ROI")
                            self.issues_found.append(f"no_person_in_roi_{bbox_idx}")
                    else:
                        logger.warning("  ISSUE: No pose results")
                        self.issues_found.append(f"no_pose_results_{bbox_idx}")
                        
                except Exception as e:
                    logger.error(f"  ROI processing error: {e}")
                    self.issues_found.append(f"roi_error_{bbox_idx}")
                    
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            self.issues_found.append("simulation_failed")
            
    def generate_focused_report(self):
        """Generate a focused report on the actual issues."""
        logger.info("\n" + "="*80)
        logger.info("DIAGNOSIS COMPLETE - HERE'S WHAT'S HAPPENING")
        logger.info("="*80)
        
        # Analyze patterns
        bbox_issues = [i for i in self.issues_found if 'bbox_out_of_bounds' in i]
        roi_issues = [i for i in self.issues_found if 'empty_roi' in i]
        keypoint_issues = [i for i in self.issues_found if 'insufficient_keypoints' in i or 'no_person_in_roi' in i]
        
        logger.info(f"\nIssues found: {len(self.issues_found)}")
        logger.info(f"  Bbox out of bounds: {len(bbox_issues)}")
        logger.info(f"  Empty ROI: {len(roi_issues)}")
        logger.info(f"  Keypoint detection: {len(keypoint_issues)}")
        
        logger.info("\n" + "="*80)
        logger.info("THE MAIN PROBLEM:")
        logger.info("="*80)
        
        if bbox_issues or roi_issues:
            logger.info("\n1. BOUNDING BOX ISSUES (Most Likely)")
            logger.info("   The person detection is creating bounding boxes that:")
            logger.info("   - Extend beyond frame boundaries")
            logger.info("   - Create empty or invalid ROIs")
            logger.info("   - Are too small for reliable pose detection")
            logger.info("\n   FIX: Add bounds checking in orientation_estimator.py:")
            logger.info("   ```python")
            logger.info("   # Before ROI extraction:")
            logger.info("   x, y, w, h = bbox")
            logger.info("   x = max(0, x)")
            logger.info("   y = max(0, y)")
            logger.info("   w = min(w, color_frame.shape[1] - x)")
            logger.info("   h = min(h, color_frame.shape[0] - y)")
            logger.info("   ```")
            
        if keypoint_issues:
            logger.info("\n2. SMALL ROI DETECTION ISSUES")
            logger.info("   Pose detection fails on small ROIs because:")
            logger.info("   - ROI is too small (< 80x120 pixels)")
            logger.info("   - Person is too far from camera")
            logger.info("   - Partial person in frame")
            logger.info("\n   FIX: Add minimum ROI size check:")
            logger.info("   ```python")
            logger.info("   if w < 80 or h < 120:")
            logger.info("       # Skip pose detection for very small ROIs")
            logger.info("       return None, debug_data")
            logger.info("   ```")
            
        logger.info("\n" + "="*80)
        logger.info("IMMEDIATE ACTIONS:")
        logger.info("="*80)
        logger.info("1. Add boundary checking to ROI extraction")
        logger.info("2. Add minimum ROI size validation")
        logger.info("3. Handle edge cases where person is partially visible")
        logger.info("4. For movement method: ensure tracking maintains IDs across frames")
        
        # Save detailed report
        report_file = f"orientation_fix_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'issues': self.issues_found,
            'bbox_issues': bbox_issues,
            'roi_issues': roi_issues,
            'keypoint_issues': keypoint_issues,
            'recommendations': [
                "Add bounds checking before ROI extraction",
                "Validate ROI size before pose detection",
                "Handle partially visible persons",
                "Check tracking consistency for movement method"
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    debugger = OrientationDebugger()
    debugger.run_diagnosis()