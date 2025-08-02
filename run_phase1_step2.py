"""
Phase 1 Step 2: Enhanced YOLO Person Detection Implementation
Updated with full-screen display layout and larger connected windows
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

class FullScreenDisplay:
    """Manages full-screen display with connected windows."""
    
    def __init__(self, screen_width=1920, screen_height=1080):
        """Initialize the full-screen display manager."""
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Calculate window dimensions (3 windows side by side)
        self.window_width = screen_width // 3
        self.window_height = int(screen_height * 0.8)  # 80% of screen height
        self.panel_height = screen_height - self.window_height
        
        # Window positions
        self.window_positions = {
            'detection': (0, self.panel_height),
            'depth': (self.window_width, self.panel_height),
            'zones': (2 * self.window_width, self.panel_height)
        }
        
        # Create the main panel image
        self.panel = np.zeros((self.panel_height, self.screen_width, 3), dtype=np.uint8)
        self.panel[:] = (40, 40, 40)  # Dark gray background
        
        self.setup_windows()
    
    def setup_windows(self):
        """Setup OpenCV windows with proper positioning."""
        window_names = ['YOLO Detection', 'Depth Analysis', 'Depth Zones']
        
        for i, name in enumerate(window_names):
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, self.window_width, self.window_height)
            
            # Position windows side by side
            x_pos = i * self.window_width
            y_pos = self.panel_height
            cv2.moveWindow(name, x_pos, y_pos)
        
        # Create info panel window
        cv2.namedWindow('Info Panel', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Info Panel', self.screen_width, self.panel_height)
        cv2.moveWindow('Info Panel', 0, 0)
    
    def create_info_panel(self, stats: Dict, detections: List, quality_metrics: Dict) -> np.ndarray:
        """Create an information panel with system statistics."""
        panel = self.panel.copy()
        
        # Title
        title = "Social Interaction Tracking System - Phase 1 Step 2: YOLO Detection"
        cv2.putText(panel, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # System info section
        sys_info = [
            f"Frame: {stats.get('total_frames', 0)}",
            f"Processing Time: {stats.get('avg_processing_time', 0):.3f}s",
            f"Effective FPS: {1/stats.get('avg_processing_time', 1):.1f}",
            f"Detection Rate: {stats.get('detection_rate', 0):.1f}%"
        ]
        
        for i, info in enumerate(sys_info):
            cv2.putText(panel, info, (20, 80 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detection info section
        det_info = [
            f"Current Detections: {len(detections)}",
            f"Total Detections: {stats.get('total_detections', 0)}",
            f"Depth Success Rate: {stats.get('depth_success_rate', 0):.1f}%",
            f"Avg Confidence: {stats.get('avg_confidence', 0):.2f}"
        ]
        
        for i, info in enumerate(det_info):
            cv2.putText(panel, info, (400, 80 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Zone statistics
        zone_stats = stats.get('detections_by_zone', {'high': 0, 'medium': 0, 'low': 0})
        zone_info = [
            f"High Confidence Zone: {zone_stats['high']}",
            f"Medium Confidence Zone: {zone_stats['medium']}",
            f"Low Confidence Zone: {zone_stats['low']}"
        ]
        
        for i, info in enumerate(zone_info):
            cv2.putText(panel, info, (800, 80 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Quality metrics
        quality_info = [
            f"Valid Depth Ratio: {quality_metrics.get('valid_ratio', 0):.1%}",
            f"Mean Depth: {quality_metrics.get('mean_depth', 0):.2f}m",
            f"Depth Variance: {quality_metrics.get('depth_variance', 0):.3f}",
            f"Color Sharpness: {quality_metrics.get('color_sharpness', 0):.0f}"
        ]
        
        for i, info in enumerate(quality_info):
            cv2.putText(panel, info, (1200, 80 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Current detections details
        if detections:
            details_y = 180
            cv2.putText(panel, "Current Detections:", (20, details_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            for i, detection in enumerate(detections[:5]):  # Show max 5 detections
                det_text = f"Person {i+1}: Conf={detection.confidence:.2f}, Zone={detection.zone}"
                if hasattr(detection, 'depth_meters') and detection.depth_meters > 0:
                    det_text += f", Depth={detection.depth_meters:.1f}m"
                
                cv2.putText(panel, det_text, (30, details_y + 30 + i * 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instructions
        instructions = [
            "Controls: 'q' = quit, 's' = save frame, 'r' = reset background",
            "Stand 1-3 meters from camera for optimal detection"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(panel, instruction, (20, self.panel_height - 40 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
        
        return panel
    
    def resize_frame_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to fit in the display window."""
        target_height = self.window_height - 50  # Leave some margin
        target_width = self.window_width - 20
        
        h, w = frame.shape[:2]
        
        # Calculate scaling factor maintaining aspect ratio
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a canvas with target size and center the resized frame
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate position to center the frame
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        if len(resized.shape) == 2:  # Grayscale
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas

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
        stats['depth_variance'] = float(np.var(valid_depths))
    else:
        stats['min_depth'] = 0
        stats['max_depth'] = 0
        stats['mean_depth'] = 0
        stats['std_depth'] = 0
        stats['depth_variance'] = 0
    
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
    
    # Add zone legend
    cv2.putText(vis, "Green: High Conf (0.3-5m)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis, "Yellow: Med Conf (5-10m)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(vis, "Red: Low Conf (10-25m)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return vis

def run_phase1_step2_test():
    """Run comprehensive YOLO person detection test with enhanced full-screen display."""
    print("Phase 1 Step 2: Enhanced YOLO Person Detection Implementation")
    print("=" * 55)
    
    # Check prerequisites
    if not check_yolo_availability():
        return False
    
    # Get screen resolution (you can adjust these values for your screen)
    screen_width = 1920
    screen_height = 1080
    
    # Try to get actual screen resolution
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        print(f"Detected screen resolution: {screen_width}x{screen_height}")
    except:
        print(f"Using default screen resolution: {screen_width}x{screen_height}")
    
    # Initialize display manager
    display = FullScreenDisplay(screen_width, screen_height)
    
    # Configuration with extended zones
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
        'depth_range': (0.3, 20.0),
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
    logger.info("=== Testing Enhanced YOLO Person Detection Pipeline ===")
    
    camera = RealSenseCapture(config)
    processor = FrameProcessor(config)
    detector = PersonDetector(config)
    
    if not (camera.configure_camera() and camera.start_streaming()):
        logger.error("Failed to initialize camera")
        return False
    
    logger.info("Camera initialized successfully")
    logger.info("YOLO model loaded successfully")
    logger.info(f"Display setup: {screen_width}x{screen_height}")
    
    # Create output directory
    output_dir = Path("data/test_sessions/phase1_step2_enhanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test parameters
    test_duration = 120  # 2 minutes
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
    
    logger.info(f"Starting {test_duration}s test with enhanced display")
    logger.info("Stand 1-3 meters from camera for optimal detection")
    
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
            
            # Calculate quality metrics
            quality_metrics = processor.calculate_frame_quality(
                processed.depth_filtered, processed.color_frame
            )
            
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
                    
                    if hasattr(detection, 'depth_meters') and detection.depth_meters > 0:
                        stats['depth_values'].append(detection.depth_meters)
                        stats['depth_success_count'] += 1
            
            # Calculate running averages for display
            if stats['processing_times']:
                stats['avg_processing_time'] = np.mean(stats['processing_times'])
            else:
                stats['avg_processing_time'] = 0
            
            if stats['total_frames'] > 0:
                stats['detection_rate'] = (stats['frames_with_detections'] / stats['total_frames']) * 100
            else:
                stats['detection_rate'] = 0
            
            if stats['confidence_values']:
                stats['avg_confidence'] = np.mean(stats['confidence_values'])
            else:
                stats['avg_confidence'] = 0
            
            if stats['total_detections'] > 0 and stats['depth_success_count'] > 0:
                stats['depth_success_rate'] = (stats['depth_success_count'] / stats['total_detections']) * 100
            else:
                stats['depth_success_rate'] = 0
            
            # Create visualizations
            vis_frame = detector.visualize_detections(color, detections)
            depth_vis = processor._depth_to_uint8(processed.depth_filtered)
            depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            zone_vis = visualize_depth_coverage(depth, detections)
            
            # Resize frames for display
            vis_resized = display.resize_frame_for_display(vis_frame)
            depth_resized = display.resize_frame_for_display(depth_colored)
            zones_resized = display.resize_frame_for_display(zone_vis)
            
            # Create info panel
            info_panel = display.create_info_panel(stats, detections, quality_metrics)
            
            # Display all frames
            cv2.imshow('YOLO Detection', vis_resized)
            cv2.imshow('Depth Analysis', depth_resized)
            cv2.imshow('Depth Zones', zones_resized)
            cv2.imshow('Info Panel', info_panel)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                cv2.imwrite(str(output_dir / f"enhanced_color_{stats['total_frames']:04d}.png"), color)
                cv2.imwrite(str(output_dir / f"enhanced_depth_{stats['total_frames']:04d}.png"), depth)
                cv2.imwrite(str(output_dir / f"enhanced_vis_{stats['total_frames']:04d}.png"), vis_frame)
                cv2.imwrite(str(output_dir / f"enhanced_zones_{stats['total_frames']:04d}.png"), zone_vis)
                cv2.imwrite(str(output_dir / f"enhanced_panel_{stats['total_frames']:04d}.png"), info_panel)
                logger.info(f"Saved enhanced capture at frame {stats['total_frames']}")
            elif key == ord('r'):
                # Reset background model
                processor.reset_background_model()
                logger.info("Background model reset")
            
            last_frame_time = current_time
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()
    
    # Calculate final statistics and display results
    logger.info("=== Enhanced Phase 1 Step 2 Test Results ===")
    logger.info(f"Total frames processed: {stats['total_frames']}")
    logger.info(f"Detection rate: {stats['detection_rate']:.1f}%")
    logger.info(f"Total people detected: {stats['total_detections']}")
    logger.info(f"Average confidence: {stats['avg_confidence']:.2f}")
    logger.info(f"Depth success rate: {stats['depth_success_rate']:.1f}%")
    logger.info(f"Average processing time: {stats['avg_processing_time']:.3f}s")
    logger.info(f"Effective FPS: {1/stats['avg_processing_time']:.1f}")
    
    return True

if __name__ == "__main__":
    success = run_phase1_step2_test()
    sys.exit(0 if success else 1)