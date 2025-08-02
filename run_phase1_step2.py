"""
Phase 1 Step 2: YOLO Person Detection Implementation
Updated to work with fixed depth processing and provide better diagnostics
"""

import sys
import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from camera.realsense_capture import RealSenseCapture
from camera.frame_processor import FrameProcessor
from detection.person_detector import PersonDetector
from config.detection_config import PERSON_DETECTION, MOUNT_CONFIG, DETECTION_ZONES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def check_yolo_availability():
    """Check if YOLO is available and properly installed."""
    try:
        from ultralytics import YOLO
        print("✓ YOLOv8 is available")
        return True
    except ImportError:
        print("✗ YOLOv8 not found. Run: pip install ultralytics")
        return False

def analyze_depth_frame(depth_frame: np.ndarray) -> Dict:
    """Analyze depth frame to diagnose issues."""
    stats = {}
    
    # Handle different depth formats
    if depth_frame.dtype == np.uint16:
        depth_meters = depth_frame.astype(np.float32) / 1000.0
    else:
        depth_meters = depth_frame.astype(np.float32)
        # Check if values are in millimeters
        if np.nanmean(depth_meters[depth_meters > 0]) > 100:
            depth_meters = depth_meters / 1000.0
    
    valid_mask = (depth_meters > 0.1) & (depth_meters < 20.0) & ~np.isnan(depth_meters)
    valid_depths = depth_meters[valid_mask]
    
    stats['total_pixels'] = depth_frame.size
    stats['valid_pixels'] = len(valid_depths)
    stats['valid_ratio'] = len(valid_depths) / depth_frame.size if depth_frame.size > 0 else 0
    
    if len(valid_depths) > 0:
        stats['min_depth'] = float(np.min(valid_depths))
        stats['max_depth'] = float(np.max(valid_depths))
        stats['mean_depth'] = float(np.mean(valid_depths))
        stats['std_depth'] = float(np.std(valid_depths))
    else:
        stats['min_depth'] = 0
        stats['max_depth'] = 0
        stats['mean_depth'] = 0
        stats['std_depth'] = 0
    
    return stats

def visualize_depth_coverage(depth_frame: np.ndarray, detections: List) -> np.ndarray:
    """Create a visualization showing depth coverage and detection zones."""
    # Convert depth to meters
    if depth_frame.dtype == np.uint16:
        depth_meters = depth_frame.astype(np.float32) / 1000.0
    else:
        depth_meters = depth_frame.astype(np.float32)
        if np.nanmean(depth_meters[depth_meters > 0]) > 100:
            depth_meters = depth_meters / 1000.0
    
    # Create colored visualization based on zones
    vis = np.zeros((depth_frame.shape[0], depth_frame.shape[1], 3), dtype=np.uint8)
    
    # High confidence zone (0.3-5.0m) - Green
    mask_high = (depth_meters >= 0.3) & (depth_meters < 5.0)
    vis[mask_high] = [0, 255, 0]
    
    # Medium confidence zone (5.0-10.0m) - Yellow
    mask_medium = (depth_meters >= 5.0) & (depth_meters < 10.0)
    vis[mask_medium] = [0, 255, 255]
    
    # Low confidence zone (10.0-25.0m) - Red
    mask_low = (depth_meters >= 10.0) & (depth_meters < 25.0)
    vis[mask_low] = [0, 0, 255]
    
    # No depth - Black
    mask_no_depth = (depth_meters <= 0.1) | np.isnan(depth_meters)
    vis[mask_no_depth] = [0, 0, 0]
    
    # Add detection overlays
    for detection in detections:
        x, y, w, h = detection.bounding_box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        # Add depth value
        if hasattr(detection, 'depth_meters'):
            depth_text = f"{detection.depth_meters:.1f}m"
            cv2.putText(vis, depth_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis

def run_phase1_step2_test():
    """Run comprehensive YOLO person detection test with depth integration."""
    print("Phase 1 Step 2: YOLO Person Detection Implementation")
    print("=" * 55)
    
    # Check prerequisites
    if not check_yolo_availability():
        return False
    
    # Configuration with extended zones
    config = {
        'depth_width': 640,
        'depth_height': 480,
        'depth_fps': 30,
        'color_width': 640,
        'color_height': 480,
        'color_fps': 30,
        'align_streams': True,  # Critical for depth alignment
        'median_filter_size': 5,
        'morph_kernel_size': (7, 7),
        'depth_range': (0.3, 20.0),  # Extended range
        'background_update_rate': 0.01,
        **PERSON_DETECTION,
        **MOUNT_CONFIG,
        'detection_zones': {
            'high_confidence': {
                'distance_range': (0.3, 5.0),
                'weight': 1.0,
                'interaction_threshold': 0.8,
            },
            'medium_confidence': {
                'distance_range': (5.0, 10.0),
                'weight': 0.7,
                'interaction_threshold': 0.6,
            },
            'low_confidence': {
                'distance_range': (10.0, 25.0),
                'weight': 0.4,
                'interaction_threshold': 0.5,
            }
        }
    }
    
    # Initialize components
    logger.info("=== Testing YOLO Person Detection Pipeline ===")
    
    camera = RealSenseCapture(config)
    processor = FrameProcessor(config)
    detector = PersonDetector(config)
    
    if not (camera.configure_camera() and camera.start_streaming()):
        logger.error("Failed to initialize camera")
        return False
    
    logger.info("Camera initialized successfully")
    logger.info("YOLO model loaded successfully")
    
    # Create output directory
    output_dir = Path("data/test_sessions/phase1_step2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test parameters
    test_duration = 60  # seconds
    target_fps = 10
    frame_interval = 1.0 / target_fps
    
    # Statistics tracking
    stats = {
        'total_frames': 0,
        'frames_with_detections': 0,
        'total_detections': 0,
        'detections_by_zone': {'high': 0, 'medium': 0, 'low': 0},
        'depth_success_count': 0,
        'processing_times': [],
        'confidence_values': [],
        'depth_values': [],
        'depth_frame_stats': []
    }
    
    logger.info(f"Starting {test_duration}s test at {target_fps} FPS")
    logger.info("Stand in front of the camera to test YOLO detection...")
    logger.info("Optimal distance: 1-3 meters from camera")
    
    # Visual windows
    cv2.namedWindow('YOLO Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Analysis', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Zones', cv2.WINDOW_NORMAL)
    
    start_time = time.time()
    last_frame_time = start_time
    sample_saved = False
    
    try:
        while time.time() - start_time < test_duration:
            # Capture frames
            depth, color = camera.get_frames()
            if depth is None or color is None:
                continue
            
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_frame_time < frame_interval:
                continue
            
            # Process frames
            timestamp = current_time
            processed = processor.process_frame_pair(depth, color, timestamp)
            
            # Analyze depth frame
            depth_stats = analyze_depth_frame(depth)
            stats['depth_frame_stats'].append(depth_stats)
            
            # Detect people
            detection_start = time.time()
            detections = detector.detect_people(depth, timestamp, color)
            detection_time = time.time() - detection_start
            
            # Update statistics
            stats['total_frames'] += 1
            stats['processing_times'].append(detection_time)
            
            if detections:
                stats['frames_with_detections'] += 1
                stats['total_detections'] += len(detections)
                
                for detection in detections:
                    stats['detections_by_zone'][detection.zone] += 1
                    stats['confidence_values'].append(detection.confidence)
                    
                    # Track depth values
                    if hasattr(detection, 'depth_meters') and detection.depth_meters > 0:
                        stats['depth_values'].append(detection.depth_meters)
                        stats['depth_success_count'] += 1
                
                # Save sample frames for first detection
                if not sample_saved and len(detections) > 0:
                    for detection in detections:
                        if detection.confidence > 0.3 and hasattr(detection, 'depth_meters') and detection.depth_meters > 0:
                            sample_saved = True
                            cv2.imwrite(str(output_dir / f"sample_color_{stats['total_frames']:04d}.png"), color)
                            cv2.imwrite(str(output_dir / f"sample_depth_{stats['total_frames']:04d}.png"), depth)
                            logger.info(f"Saved detection sample: {len(detections)} people with valid depth")
                            break
            
            # Create visualizations
            vis_frame = detector.visualize_detections(color, detections)
            depth_vis = processor._depth_to_uint8(processed.depth_filtered)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            zone_vis = visualize_depth_coverage(depth, detections)
            
            # Add statistics overlay
            info_text = [
                f"Frame: {stats['total_frames']}",
                f"Detections: {len(detections)}",
                f"Valid Depth: {depth_stats['valid_ratio']:.1%}",
                f"Mean Depth: {depth_stats['mean_depth']:.2f}m",
                f"FPS: {1/detection_time:.1f}"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(vis_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            # Display frames
            cv2.imshow('YOLO Detection', vis_frame)
            cv2.imshow('Depth Analysis', depth_colored)
            cv2.imshow('Depth Zones', zone_vis)
            
            # Log periodic updates
            if stats['total_frames'] % 50 == 0 and stats['total_frames'] > 0:
                avg_time = np.mean(stats['processing_times'])
                if stats['confidence_values']:
                    avg_conf = np.mean(stats['confidence_values'])
                else:
                    avg_conf = 0
                
                if stats['depth_values']:
                    avg_depth = np.mean(stats['depth_values'])
                    depth_success_rate = stats['depth_success_count'] / stats['total_detections'] * 100
                else:
                    avg_depth = 0
                    depth_success_rate = 0
                
                logger.info(f"Frame {stats['total_frames']}: "
                           f"{len(detections)} people detected, "
                           f"Processing time: {avg_time:.3f}s")
                
                if detections:
                    for i, d in enumerate(detections):
                        logger.info(f"  Person {i}: Confidence={d.confidence:.2f}, "
                                  f"Zone={d.zone}, "
                                  f"Depth={d.depth_meters if hasattr(d, 'depth_meters') else 0:.1f}m")
                
                logger.info(f"  Depth success rate: {depth_success_rate:.1f}%")
                logger.info(f"  Average depth: {avg_depth:.2f}m")
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                cv2.imwrite(str(output_dir / f"manual_color_{stats['total_frames']:04d}.png"), color)
                cv2.imwrite(str(output_dir / f"manual_depth_{stats['total_frames']:04d}.png"), depth)
                cv2.imwrite(str(output_dir / f"manual_vis_{stats['total_frames']:04d}.png"), vis_frame)
                cv2.imwrite(str(output_dir / f"manual_zones_{stats['total_frames']:04d}.png"), zone_vis)
                logger.info(f"Saved manual capture at frame {stats['total_frames']}")
            
            last_frame_time = current_time
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()
    
    # Calculate final statistics
    logger.info("=== Phase 1 Step 2 Test Results ===")
    logger.info(f"Total frames processed: {stats['total_frames']}")
    logger.info(f"Frames with detections: {stats['frames_with_detections']}")
    
    if stats['total_frames'] > 0:
        detection_rate = stats['frames_with_detections'] / stats['total_frames'] * 100
        logger.info(f"Detection rate: {detection_rate:.1f}%")
    
    logger.info(f"Total people detected: {stats['total_detections']}")
    
    if stats['total_frames'] > 0:
        avg_people = stats['total_detections'] / stats['total_frames']
        logger.info(f"Average people per frame: {avg_people:.2f}")
    
    if stats['confidence_values']:
        logger.info(f"Average confidence: {np.mean(stats['confidence_values']):.2f}")
    
    logger.info(f"High confidence detections: {stats['detections_by_zone']['high']}")
    logger.info(f"Medium confidence detections: {stats['detections_by_zone']['medium']}")
    logger.info(f"Low confidence detections: {stats['detections_by_zone']['low']}")
    
    if stats['depth_values']:
        logger.info(f"Depth extraction success rate: {stats['depth_success_count']/stats['total_detections']*100:.1f}%")
        logger.info(f"Average depth: {np.mean(stats['depth_values']):.2f}m")
        logger.info(f"Depth range: {np.min(stats['depth_values']):.2f}m - {np.max(stats['depth_values']):.2f}m")
    else:
        logger.warning("No valid depth measurements obtained!")
    
    if stats['processing_times']:
        avg_time = np.mean(stats['processing_times'])
        logger.info(f"Average processing time: {avg_time:.3f}s")
        logger.info(f"Effective FPS: {1/avg_time:.1f}")
    
    # Depth frame analysis
    if stats['depth_frame_stats']:
        avg_valid_ratio = np.mean([s['valid_ratio'] for s in stats['depth_frame_stats']])
        logger.info(f"Average valid depth pixel ratio: {avg_valid_ratio:.1%}")
    
    # Validation criteria
    validation_results = {
        'minimum_fps': 1/avg_time >= 8 if stats['processing_times'] else False,
        'detection_accuracy': detection_rate >= 70 if stats['total_frames'] > 0 else False,
        'processing_speed': avg_time < 0.15 if stats['processing_times'] else False,
        'depth_extraction': stats['depth_success_count'] > 0,
        'confidence_variety': len(set(stats['confidence_values'])) > 1 if stats['confidence_values'] else False
    }
    
    for criterion, passed in validation_results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"{criterion}: {status}")
    
    # Overall assessment
    all_passed = all(validation_results.values())
    if all_passed:
        logger.info("✓ Phase 1 Step 2 - Ready to proceed!")
    else:
        logger.warning("⚠ Phase 1 Step 2 - Needs refinement before proceeding")
        
        # Provide specific guidance
        if not validation_results['depth_extraction']:
            logger.warning("Depth extraction failed. Check:")
            logger.warning("  1. Camera alignment is enabled")
            logger.warning("  2. You're standing 1-3 meters from camera")
            logger.warning("  3. Adequate lighting in the room")
        
        if not validation_results['confidence_variety']:
            logger.warning("All detections have same confidence. Check:")
            logger.warning("  1. Depth data is being properly processed")
            logger.warning("  2. Detection zones are configured correctly")
    
    return all_passed

if __name__ == "__main__":
    success = run_phase1_step2_test()
    sys.exit(0 if success else 1)