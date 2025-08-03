"""
Phase 2 Step 1: Proximity Analysis Test Runner
Tests proximity calculation and proxemic zone detection
"""

import sys
import os
import cv2
import numpy as np
import time
import logging
import json
from pathlib import Path
from typing import List, Dict

# Setup paths
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))

from camera.realsense_capture import RealSenseCapture
from camera.frame_processor import FrameProcessor
from detection.person_detector import PersonDetector
from detection.tracker import PersonTracker
from interaction.proximity_analyzer import ProximityAnalyzer
from config.detection_config import PERSON_DETECTION, MOUNT_CONFIG, DETECTION_ZONES
from config.tracking_config import TRACKING_PARAMETERS
from config.proximity_config import CLASSROOM_PROXIMITY, PROXIMITY_VISUALIZATION

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class ProximityVisualizer:
    """Visualizes proximity events and proxemic zones."""
    
    def __init__(self, config: dict):
        """Initialize visualizer with proximity configuration."""
        self.config = config
        self.zone_colors = PROXIMITY_VISUALIZATION['zone_colors']
        self.line_thickness = PROXIMITY_VISUALIZATION['line_thickness']
        
    def visualize_proximities(self, frame: np.ndarray, tracked_people: List, 
                             proximity_events: List, stats: Dict) -> np.ndarray:
        """Create comprehensive proximity visualization."""
        vis_frame = frame.copy()
        
        # Draw tracked people first
        for person in tracked_people:
            if person.current_detection is None:
                continue
                
            x, y, w, h = person.current_detection.bounding_box
            
            # Draw person bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw person ID
            cv2.putText(vis_frame, f"ID:{person.id}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw position coordinates
            pos = person.get_latest_position()
            pos_text = f"({pos[0]:.1f}, {pos[1]:.1f})"
            cv2.putText(vis_frame, pos_text, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw proximity connections
        for event in proximity_events:
            self._draw_proximity_connection(vis_frame, tracked_people, event)
        
        # Draw proximity statistics
        self._draw_proximity_stats(vis_frame, stats)
        
        return vis_frame
    
    def _draw_proximity_connection(self, frame: np.ndarray, tracked_people: List, event):
        """Draw a line between two people showing their proximity."""
        # Find the people involved
        person1 = None
        person2 = None
        
        for person in tracked_people:
            if person.id == event.person1_id:
                person1 = person
            elif person.id == event.person2_id:
                person2 = person
        
        if person1 is None or person2 is None:
            return
        
        if person1.current_detection is None or person2.current_detection is None:
            return
        
        # Get center points of bounding boxes
        bbox1 = person1.current_detection.bounding_box
        bbox2 = person2.current_detection.bounding_box
        
        center1 = (bbox1[0] + bbox1[2]//2, bbox1[1] + bbox1[3]//2)
        center2 = (bbox2[0] + bbox2[2]//2, bbox2[1] + bbox2[3]//2)
        
        # Get visualization settings for this zone
        color = self.zone_colors.get(event.zone_type, (128, 128, 128))
        thickness = self.line_thickness.get(event.zone_type, 2)
        
        # Draw connection line
        cv2.line(frame, center1, center2, color, thickness)
        
        # Draw distance and zone info at midpoint
        mid_x = (center1[0] + center2[0]) // 2
        mid_y = (center1[1] + center2[1]) // 2
        
        # Distance text
        distance_text = f"{event.distance:.2f}m"
        cv2.putText(frame, distance_text, (mid_x - 30, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Zone type text
        zone_text = event.zone_type.upper()
        cv2.putText(frame, zone_text, (mid_x - 30, mid_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Duration text
        if event.duration > 0:
            duration_text = f"{event.duration:.1f}s"
            cv2.putText(frame, duration_text, (mid_x - 20, mid_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def _draw_proximity_stats(self, frame: np.ndarray, stats: Dict):
        """Draw proximity statistics on the frame."""
        y_offset = 30
        
        # Title
        cv2.putText(frame, "PROXIMITY ANALYSIS", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # Active proximities
        cv2.putText(frame, f"Active Proximities: {stats.get('active_proximities', 0)}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
        
        # Total events
        cv2.putText(frame, f"Total Events: {stats.get('total_events_created', 0)}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
        
        # Average distance
        avg_dist = stats.get('average_distance', 0)
        if avg_dist > 0:
            cv2.putText(frame, f"Avg Distance: {avg_dist:.2f}m",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
        
        # Zone distribution
        zone_dist = stats.get('zone_distribution', {})
        if zone_dist:
            cv2.putText(frame, "Zone Distribution:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 15
            
            for zone, count in zone_dist.items():
                color = self.zone_colors.get(zone, (255, 255, 255))
                cv2.putText(frame, f"  {zone}: {count}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 15

def create_proximity_dashboard(proximity_events: List, stats: Dict) -> np.ndarray:
    """Create a dashboard showing proximity analysis details."""
    dashboard = np.zeros((300, 600, 3), dtype=np.uint8)
    dashboard[:] = (40, 40, 40)  # Dark background
    
    y = 30
    
    # Title
    cv2.putText(dashboard, "PROXIMITY ANALYSIS DASHBOARD", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 40
    
    # Current proximities
    cv2.putText(dashboard, f"Active Proximities: {len(proximity_events)}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y += 25
    
    # List active proximities
    for i, event in enumerate(proximity_events[:8]):  # Show max 8
        proximity_text = (f"P{event.person1_id}<->P{event.person2_id}: "
                         f"{event.distance:.2f}m ({event.zone_type}) {event.duration:.1f}s")
        
        # Color based on zone
        zone_colors = {
            'intimate': (0, 0, 255),
            'personal': (0, 165, 255),
            'social': (0, 255, 255),
            'public': (0, 255, 0)
        }
        color = zone_colors.get(event.zone_type, (255, 255, 255))
        
        cv2.putText(dashboard, proximity_text, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        y += 20
    
    # Statistics
    y += 20
    cv2.putText(dashboard, "STATISTICS:", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 25
    
    stats_text = [
        f"Total Events Created: {stats.get('total_events_created', 0)}",
        f"Average Distance: {stats.get('average_distance', 0):.2f}m",
        f"Average Duration: {stats.get('average_duration', 0):.1f}s",
    ]
    
    for text in stats_text:
        cv2.putText(dashboard, text, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 20
    
    return dashboard

def run_phase2_step1_test():
    """Run Phase 2 Step 1: Proximity Analysis test."""
    print("Phase 2 Step 1: Proximity Analysis Implementation Test")
    print("=" * 55)
    
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
        'depth_range': (0.3, 20.0),
        'background_update_rate': 0.01,
        **PERSON_DETECTION,
        **MOUNT_CONFIG,
        'detection_zones': DETECTION_ZONES,
        **TRACKING_PARAMETERS,
        **CLASSROOM_PROXIMITY,
    }
    
    # Initialize components
    logger.info("=== Testing Phase 2 Step 1: Proximity Analysis ===")
    
    camera = RealSenseCapture(config)
    processor = FrameProcessor(config)
    detector = PersonDetector(config)
    tracker = PersonTracker(config)
    proximity_analyzer = ProximityAnalyzer(config)
    visualizer = ProximityVisualizer(config)
    
    if not (camera.configure_camera() and camera.start_streaming()):
        logger.error("Failed to initialize camera")
        return False
    
    logger.info("All components initialized successfully")
    logger.info("Testing proximity analysis...")
    logger.info("Move around to test different proximity zones:")
    logger.info("  - Get very close (< 30cm) for INTIMATE zone")
    logger.info("  - Stay close (30cm-1m) for PERSONAL zone")
    logger.info("  - Normal distance (1-2.5m) for SOCIAL zone")
    logger.info("  - Far apart (2.5-5m) for PUBLIC zone")
    
    # Create output directory
    output_dir = Path("data/test_sessions/phase2_step1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test parameters
    test_duration = 120  # 2 minutes
    target_fps = 15
    frame_interval = 1.0 / target_fps
    
    # Statistics tracking
    stats = {
        'total_frames': 0,
        'frames_with_proximities': 0,
        'max_simultaneous_proximities': 0,
        'processing_times': [],
        'proximity_events_created': 0,
    }
    
    start_time = time.time()
    last_frame_time = start_time
    
    try:
        while time.time() - start_time < test_duration:
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
            
            # Person detection and tracking
            detection_start = time.time()
            detections = detector.detect_people(depth, timestamp, color)
            tracked_people = tracker.update(detections, timestamp)
            
            # Proximity analysis
            proximity_events = proximity_analyzer.analyze_frame(tracked_people, timestamp)
            processing_time = time.time() - detection_start
            
            # Update statistics
            stats['total_frames'] += 1
            stats['processing_times'].append(processing_time)
            
            if proximity_events:
                stats['frames_with_proximities'] += 1
                stats['max_simultaneous_proximities'] = max(
                    stats['max_simultaneous_proximities'], len(proximity_events)
                )
            
            # Get proximity statistics
            proximity_stats = proximity_analyzer.get_proximity_statistics()
            stats['proximity_events_created'] = proximity_stats.get('total_events_created', 0)
            
            # Create visualizations
            vis_frame = visualizer.visualize_proximities(
                color, tracked_people, proximity_events, proximity_stats
            )
            
            dashboard = create_proximity_dashboard(proximity_events, proximity_stats)
            
            # Combine visualizations
            combined_height = vis_frame.shape[0] + dashboard.shape[0]
            combined_width = max(vis_frame.shape[1], dashboard.shape[1])
            combined_vis = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
            
            # Place main visualization
            combined_vis[:vis_frame.shape[0], :vis_frame.shape[1]] = vis_frame
            
            # Place dashboard
            dashboard_y = vis_frame.shape[0]
            combined_vis[dashboard_y:dashboard_y + dashboard.shape[0], :dashboard.shape[1]] = dashboard
            
            # Display
            cv2.imshow('Phase 2 Step 1: Proximity Analysis', combined_vis)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current state
                timestamp_str = f"{timestamp:.3f}".replace('.', '_')
                cv2.imwrite(str(output_dir / f"proximity_{timestamp_str}.png"), combined_vis)
                
                # Save proximity data
                proximity_data = {
                    'timestamp': timestamp,
                    'proximity_events': [
                        {
                            'person1_id': e.person1_id,
                            'person2_id': e.person2_id,
                            'distance': e.distance,
                            'zone_type': e.zone_type,
                            'duration': e.duration,
                            'confidence': e.confidence,
                        }
                        for e in proximity_events
                    ],
                    'statistics': proximity_stats
                }
                
                with open(output_dir / f"proximity_data_{timestamp_str}.json", 'w') as f:
                    json.dump(proximity_data, f, indent=2)
                
                logger.info(f"Saved proximity analysis at frame {stats['total_frames']}")
                
            elif key == ord('r'):
                # Reset proximity analyzer
                proximity_analyzer.reset()
                tracker.reset()
                logger.info("Proximity analyzer and tracker reset")
            
            last_frame_time = current_time
            
            # Periodic logging
            if stats['total_frames'] % 50 == 0 and stats['total_frames'] > 0:
                active_proximities = len(proximity_events)
                avg_processing_time = np.mean(stats['processing_times'][-50:])
                fps = 1.0 / max(avg_processing_time, 0.001)
                
                logger.info(f"Frame {stats['total_frames']}: "
                           f"{active_proximities} active proximities, "
                           f"FPS: {fps:.1f}")
    
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()
    
    # Final analysis and validation
    final_proximity_stats = proximity_analyzer.get_proximity_statistics()
    
    logger.info("=== Phase 2 Step 1 Test Results ===")
    logger.info(f"Total frames processed: {stats['total_frames']}")
    logger.info(f"Frames with proximities: {stats['frames_with_proximities']}")
    logger.info(f"Max simultaneous proximities: {stats['max_simultaneous_proximities']}")
    logger.info(f"Total proximity events created: {stats['proximity_events_created']}")
    
    if stats['processing_times']:
        avg_processing_time = np.mean(stats['processing_times'])
        logger.info(f"Average processing time: {avg_processing_time:.3f}s")
        logger.info(f"Effective FPS: {1/avg_processing_time:.1f}")
    
    # Proximity-specific results
    logger.info(f"\n=== Proximity Analysis Results ===")
    logger.info(f"Zone distribution: {final_proximity_stats.get('zone_distribution', {})}")
    
    if final_proximity_stats.get('average_distance', 0) > 0:
        logger.info(f"Average proximity distance: {final_proximity_stats['average_distance']:.2f}m")
        logger.info(f"Average proximity duration: {final_proximity_stats['average_duration']:.1f}s")
    
    # Validation criteria
    logger.info("\n=== Phase 2 Step 1 Validation ===")
    
    success_criteria = {
        'proximity_detection_works': stats['proximity_events_created'] > 0,
        'performance_good': np.mean(stats['processing_times']) < 0.1 if stats['processing_times'] else False,
        'multiple_zones_detected': len(final_proximity_stats.get('zone_distribution', {})) > 1,
        'temporal_tracking': final_proximity_stats.get('average_duration', 0) > 0,
        'real_time_capable': (1/np.mean(stats['processing_times']) if stats['processing_times'] else 0) > 10,
    }
    
    logger.info(f"✓ Proximity detection functionality: {'PASS' if success_criteria['proximity_detection_works'] else 'FAIL'}")
    logger.info(f"✓ Performance (>10 FPS): {'PASS' if success_criteria['real_time_capable'] else 'FAIL'}")
    logger.info(f"✓ Multiple proxemic zones: {'PASS' if success_criteria['multiple_zones_detected'] else 'FAIL'}")
    logger.info(f"✓ Temporal tracking: {'PASS' if success_criteria['temporal_tracking'] else 'FAIL'}")
    
    overall_success = all(success_criteria.values())
    logger.info(f"\n🎯 PHASE 2 STEP 1 OVERALL: {'SUCCESS' if overall_success else 'NEEDS WORK'}")
    
    if overall_success:
        logger.info("✅ Ready for Phase 2 Step 2: Orientation Detection!")
        logger.info("Next steps:")
        logger.info("  - Implement body orientation estimation")
        logger.info("  - Add mutual orientation analysis")
        logger.info("  - Combine proximity + orientation for interaction detection")
    else:
        logger.info("⚠️ Recommendations:")
        if not success_criteria['proximity_detection_works']:
            logger.info("  - Check proximity thresholds and zone definitions")
        if not success_criteria['performance_good']:
            logger.info("  - Optimize proximity calculation algorithms")
        if not success_criteria['multiple_zones_detected']:
            logger.info("  - Test with people at different distances")
    
    # Save final report
    report = {
        'timestamp': time.time(),
        'test_duration': test_duration,
        'statistics': stats,
        'proximity_statistics': final_proximity_stats,
        'success_criteria': success_criteria,
        'overall_success': overall_success,
    }
    
    with open(output_dir / "phase2_step1_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\nDetailed report saved to: {output_dir / 'phase2_step1_report.json'}")
    
    return overall_success

if __name__ == "__main__":
    success = run_phase2_step1_test()
    sys.exit(0 if success else 1)