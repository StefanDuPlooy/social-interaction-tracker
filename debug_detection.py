"""
Debug Detection Script
Diagnose why no people are being detected
"""

import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from camera.realsense_capture import RealSenseCapture
from camera.frame_processor import FrameProcessor
from detection.person_detector import PersonDetector
from config.detection_config import PERSON_DETECTION, MOUNT_CONFIG, DETECTION_ZONES

def debug_detection_pipeline():
    """Debug the detection pipeline step by step."""
    print("=== Debug Detection Pipeline ===")
    
    # Configuration
    config = {
        'depth_width': 640,
        'depth_height': 480,
        'depth_fps': 30,
        'color_width': 640,
        'color_height': 480,
        'color_fps': 30,
        'align_streams': True,
        'median_filter_size': 5,
        'morph_kernel_size': (7, 7),
        'depth_range': (1.0, 10.0),
        'background_update_rate': 0.01,
        **PERSON_DETECTION,
        **MOUNT_CONFIG,
        'detection_zones': DETECTION_ZONES,
    }
    
    # Initialize components
    camera = RealSenseCapture(config)
    processor = FrameProcessor(config)
    detector = PersonDetector(config)
    
    if not (camera.configure_camera() and camera.start_streaming()):
        print("Failed to initialize camera")
        return
    
    print("Camera initialized. Stand in front of camera for debugging...")
    print("Press 'q' to quit, 's' to save debug frame, 'r' to reset background")
    
    debug_dir = Path("data/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    
    try:
        while True:
            depth, color = camera.get_frames()
            if depth is None or color is None:
                continue
            
            timestamp = time.time()
            
            # Step 1: Process frames
            processed = processor.process_frame_pair(depth, color, timestamp)
            
            # Step 2: Analyze depth data
            valid_depth_pixels = np.sum(~np.isnan(processed.depth_filtered))
            total_pixels = processed.depth_filtered.size
            valid_ratio = valid_depth_pixels / total_pixels
            
            depth_valid = processed.depth_filtered[~np.isnan(processed.depth_filtered)]
            if len(depth_valid) > 0:
                depth_min = np.min(depth_valid)
                depth_max = np.max(depth_valid)
                depth_mean = np.mean(depth_valid)
            else:
                depth_min = depth_max = depth_mean = 0
            
            # Step 3: Analyze person mask
            mask_pixels = np.sum(processed.person_mask > 0)
            mask_ratio = mask_pixels / total_pixels
            
            # Step 4: Try detection
            detections = detector.detect_people(processed.depth_filtered, timestamp)
            
            # Step 5: Find contours for debugging
            contours, _ = cv2.findContours(processed.person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            large_contours = [c for c in contours if cv2.contourArea(c) > 1000]
            
            # Display debug info
            print(f"\nFrame {frame_count}:")
            print(f"  Valid depth ratio: {valid_ratio:.1%}")
            print(f"  Depth range: {depth_min:.2f}m to {depth_max:.2f}m (mean: {depth_mean:.2f}m)")
            print(f"  Person mask pixels: {mask_pixels} ({mask_ratio:.1%})")
            print(f"  Large contours found: {len(large_contours)}")
            print(f"  People detected: {len(detections)}")
            
            if large_contours:
                print("  Contour areas:", [int(cv2.contourArea(c)) for c in large_contours])
            
            # Create debug visualization
            debug_vis = create_debug_visualization(
                color, processed.depth_filtered, processed.person_mask, 
                detections, large_contours
            )
            
            # Display
            cv2.imshow('Debug Visualization', debug_vis)
            cv2.imshow('Person Mask', processed.person_mask)
            cv2.imshow('Depth Filtered', create_depth_visualization(processed.depth_filtered))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save debug frame
                save_debug_frame(debug_dir, frame_count, depth, color, processed, detections)
                print(f"Saved debug frame {frame_count}")
            elif key == ord('r'):
                # Reset background model
                processor.reset_background_model()
                print("Background model reset")
            
            frame_count += 1
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Debug interrupted")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()

def create_debug_visualization(color, depth, mask, detections, contours):
    """Create comprehensive debug visualization."""
    # Create a 2x2 grid
    h, w = color.shape[:2]
    debug_vis = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    
    # Top-left: Original color
    debug_vis[0:h, 0:w] = color
    
    # Top-right: Depth visualization
    depth_vis = create_depth_visualization(depth)
    if len(depth_vis.shape) == 2:
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    debug_vis[0:h, w:w*2] = depth_vis
    
    # Bottom-left: Person mask
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    debug_vis[h:h*2, 0:w] = mask_vis
    
    # Bottom-right: Detections and contours
    detection_vis = color.copy()
    
    # Draw large contours in blue
    for contour in contours:
        cv2.drawContours(detection_vis, [contour], -1, (255, 0, 0), 2)
        area = cv2.contourArea(contour)
        x, y, w_c, h_c = cv2.boundingRect(contour)
        cv2.putText(detection_vis, f"Area: {int(area)}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw detections in green
    for detection in detections:
        x, y, w_d, h_d = detection.bounding_box
        cv2.rectangle(detection_vis, (x, y), (x + w_d, y + h_d), (0, 255, 0), 2)
        cv2.putText(detection_vis, f"Person: {detection.confidence:.2f}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    debug_vis[h:h*2, w:w*2] = detection_vis
    
    # Add labels
    cv2.putText(debug_vis, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_vis, "Depth", (w+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_vis, "Mask", (10, h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_vis, "Detections", (w+10, h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return debug_vis

def create_depth_visualization(depth_frame):
    """Create visualization of depth frame."""
    # Handle NaN values
    depth_clean = depth_frame.copy()
    depth_clean[np.isnan(depth_clean)] = 0
    
    # Normalize for visualization
    if np.max(depth_clean) > 0:
        depth_norm = (depth_clean / np.max(depth_clean) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth_clean, dtype=np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    return depth_colored

def save_debug_frame(debug_dir, frame_count, depth, color, processed, detections):
    """Save debug information to files."""
    frame_dir = debug_dir / f"frame_{frame_count:04d}"
    frame_dir.mkdir(exist_ok=True)
    
    # Save raw frames
    cv2.imwrite(str(frame_dir / "raw_color.png"), color)
    cv2.imwrite(str(frame_dir / "raw_depth.png"), depth)
    
    # Save processed frames
    cv2.imwrite(str(frame_dir / "processed_color.png"), processed.color_frame)
    cv2.imwrite(str(frame_dir / "person_mask.png"), processed.person_mask)
    cv2.imwrite(str(frame_dir / "depth_filtered.png"), 
                create_depth_visualization(processed.depth_filtered))
    
    # Save depth statistics
    with open(frame_dir / "depth_stats.txt", "w") as f:
        depth_valid = processed.depth_filtered[~np.isnan(processed.depth_filtered)]
        if len(depth_valid) > 0:
            f.write(f"Valid pixels: {len(depth_valid)}\n")
            f.write(f"Min depth: {np.min(depth_valid):.3f}m\n")
            f.write(f"Max depth: {np.max(depth_valid):.3f}m\n")
            f.write(f"Mean depth: {np.mean(depth_valid):.3f}m\n")
            f.write(f"Std depth: {np.std(depth_valid):.3f}m\n")
        
        f.write(f"Mask pixels: {np.sum(processed.person_mask > 0)}\n")
        f.write(f"Detections: {len(detections)}\n")
        
        for i, detection in enumerate(detections):
            f.write(f"Detection {i}: {detection}\n")

def suggest_parameter_adjustments():
    """Suggest parameter adjustments based on common issues."""
    print("\n=== Parameter Adjustment Suggestions ===")
    print("If no people are detected, try adjusting these parameters in config/detection_config.py:")
    print()
    print("1. Reduce minimum blob area:")
    print("   'min_blob_area': 2000  # was 5000")
    print()
    print("2. Increase depth threshold:")
    print("   'depth_threshold': 0.2  # was 0.1")
    print()
    print("3. Adjust height range:")
    print("   'min_height': 0.3  # was 0.5")
    print("   'max_height': 2.5  # was 2.2")
    print()
    print("4. Relax aspect ratio:")
    print("   'min_aspect_ratio': 0.5  # was 0.8")
    print("   'max_aspect_ratio': 5.0  # was 4.0")
    print()
    print("5. Extend detection zones:")
    print("   'high_confidence': {'distance_range': (1.0, 8.0), 'weight': 1.0}")

if __name__ == "__main__":
    debug_detection_pipeline()
    suggest_parameter_adjustments()