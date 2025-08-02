"""
Person Detector using YOLO + Depth
Reliable person detection using YOLOv8 on RGB + depth for 3D positioning
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLOv8 not available. Install with: pip install ultralytics")

@dataclass
class PersonDetection:
    """Data class for person detection results."""
    id: int
    timestamp: float
    position_3d: Tuple[float, float, float]  # (x, y, z) in meters
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h) in pixels
    confidence: float  # 0.0-1.0 (zone-based confidence)
    detection_method: str  # 'yolo', 'yolo+depth'
    zone: str  # 'high', 'medium', 'low' confidence
    blob_area: int  # For compatibility with existing code

class PersonDetector:
    """YOLO + Depth person detector."""
    
    def __init__(self, config: dict):
        """Initialize YOLO person detector."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters
        self.min_confidence = config.get('yolo_confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.4)
        
        # Camera configuration for 3D calculations
        self.camera_height = config.get('camera_height', 2.5)
        self.camera_tilt = config.get('camera_tilt', 30)
        
        # Zone configuration
        self.detection_zones = config.get('detection_zones', {
            'high_confidence': {'distance_range': (0.5, 8.0), 'weight': 1.0},
            'medium_confidence': {'distance_range': (8.0, 15.0), 'weight': 0.7},
            'low_confidence': {'distance_range': (15.0, 25.0), 'weight': 0.4}
        })
        
        # Initialize YOLO model
        self.model = None
        self.model_loaded = False
        self._load_yolo_model()
    
    def _load_yolo_model(self):
        """Load YOLOv8 model."""
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available. Please install ultralytics: pip install ultralytics")
            return False
        
        try:
            # Load YOLOv8n (nano) model - fastest for real-time
            self.model = YOLO('yolov8n.pt')
            self.model_loaded = True
            self.logger.info("YOLOv8 model loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def detect_people_rgb(self, color_frame: np.ndarray) -> List[Dict]:
        """Detect people in RGB image using YOLO."""
        if not self.model_loaded or self.model is None:
            return []
        
        try:
            # Run YOLO inference
            results = self.model(color_frame, verbose=False, conf=self.min_confidence)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # Only keep person detections (class 0 in COCO dataset)
                    if int(box.cls) == 0:  # Person class
                        confidence = float(box.conf)
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                        
                        # Ensure bounding box is within image bounds
                        h_img, w_img = color_frame.shape[:2]
                        x = max(0, min(x, w_img-1))
                        y = max(0, min(y, h_img-1))
                        w = max(1, min(w, w_img-x))
                        h = max(1, min(h, h_img-y))
                        
                        detection = {
                            'bounding_box': (x, y, w, h),
                            'confidence': confidence,
                            'centroid': (x + w//2, y + h//2),
                            'method': 'yolo',
                            'area': w * h
                        }
                        detections.append(detection)
            
            self.logger.debug(f"YOLO detected {len(detections)} people")
            return detections
            
        except Exception as e:
            self.logger.error(f"YOLO detection failed: {e}")
            return []
    
    def add_depth_to_detections(self, rgb_detections: List[Dict], depth_frame: np.ndarray) -> List[Dict]:
        """Add 3D position information to RGB detections using depth data."""
        enhanced_detections = []
        
        # Convert depth from millimeters to meters
        depth_meters = depth_frame.astype(np.float32) / 1000.0
        
        for detection in rgb_detections:
            x, y, w, h = detection['bounding_box']
            centroid_x, centroid_y = detection['centroid']
            
            # Get robust depth from detection area
            centroid_depth = self._get_robust_depth(depth_meters, x, y, w, h)
            
            if centroid_depth > 0 and not np.isnan(centroid_depth):
                # Calculate 3D position
                position_3d = self._pixel_to_3d_position(centroid_x, centroid_y, centroid_depth)
                
                # Determine detection zone based on depth
                zone, zone_confidence = self._calculate_detection_confidence(centroid_depth)
                
                # Enhanced detection with depth
                enhanced_detection = {
                    **detection,
                    'position_3d': position_3d,
                    'depth': centroid_depth,
                    'zone': zone,
                    'zone_confidence': zone_confidence,
                    'method': 'yolo+depth'
                }
                enhanced_detections.append(enhanced_detection)
            else:
                # Keep RGB-only detection but mark as low confidence
                enhanced_detection = {
                    **detection,
                    'position_3d': (0, 0, 0),
                    'depth': 0,
                    'zone': 'low',
                    'zone_confidence': 0.3,
                    'method': 'yolo'
                }
                enhanced_detections.append(enhanced_detection)
        
        return enhanced_detections
    
    def _get_robust_depth(self, depth_frame: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """Get robust depth estimate from detection bounding box."""
        try:
            # Sample the center 50% of the bounding box
            margin_x = w // 4
            margin_y = h // 4
            
            roi_x = x + margin_x
            roi_y = y + margin_y
            roi_w = w - 2 * margin_x
            roi_h = h - 2 * margin_y
            
            # Ensure ROI is within bounds
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_w = max(1, min(roi_w, depth_frame.shape[1] - roi_x))
            roi_h = max(1, min(roi_h, depth_frame.shape[0] - roi_y))
            
            # Extract region of interest
            roi = depth_frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            valid_depths = roi[~np.isnan(roi) & (roi > 0) & (roi < 15.0)]
            
            if len(valid_depths) > 10:
                # Use 25th percentile to avoid background pixels
                return np.percentile(valid_depths, 25)
            else:
                # Fallback: try center pixel
                center_y = y + h // 2
                center_x = x + w // 2
                if (0 <= center_y < depth_frame.shape[0] and 
                    0 <= center_x < depth_frame.shape[1]):
                    center_depth = depth_frame[center_y, center_x]
                    if not np.isnan(center_depth) and center_depth > 0:
                        return center_depth
                
                return 0.0
        except Exception as e:
            self.logger.warning(f"Depth calculation failed: {e}")
            return 0.0
    
    def _pixel_to_3d_position(self, pixel_x: int, pixel_y: int, depth_meters: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates and depth to 3D position."""
        image_width = 640
        image_height = 480
        
        h_fov = np.radians(87)  # Horizontal FOV
        v_fov = np.radians(58)  # Vertical FOV
        
        h_angle = (pixel_x - image_width/2) * (h_fov / image_width)
        v_angle = (pixel_y - image_height/2) * (v_fov / image_height)
        
        x = depth_meters * np.tan(h_angle)
        y = depth_meters * np.tan(v_angle)
        z = depth_meters
        
        return (x, y, z)
    
    def _calculate_detection_confidence(self, depth: float) -> Tuple[str, float]:
        """Calculate detection confidence based on distance zones."""
        for zone_name, zone_config in self.detection_zones.items():
            min_dist, max_dist = zone_config['distance_range']
            if min_dist <= depth <= max_dist:
                return zone_name.split('_')[0], zone_config['weight']
        
        return 'low', 0.3
    
    def detect_people(self, depth_frame: np.ndarray, timestamp: float, color_frame: Optional[np.ndarray] = None) -> List[PersonDetection]:
        """Main detection function - detect people using YOLO + depth."""
        if color_frame is None:
            self.logger.error("YOLO detector requires color frame")
            return []
        
        try:
            # Step 1: Detect people in RGB image using YOLO
            rgb_detections = self.detect_people_rgb(color_frame)
            
            if not rgb_detections:
                self.logger.debug("No people detected by YOLO")
                return []
            
            # Step 2: Add depth information to detections
            enhanced_detections = self.add_depth_to_detections(rgb_detections, depth_frame)
            
            # Step 3: Convert to PersonDetection objects
            person_detections = []
            for i, detection in enumerate(enhanced_detections):
                person_detection = PersonDetection(
                    id=-1,  # Will be assigned by tracker
                    timestamp=timestamp,
                    position_3d=detection['position_3d'],
                    bounding_box=detection['bounding_box'],
                    confidence=detection['zone_confidence'],
                    detection_method=detection['method'],
                    zone=detection['zone'],
                    blob_area=detection['area']  # For compatibility
                )
                person_detections.append(person_detection)
            
            self.logger.debug(f"Final detection: {len(person_detections)} people")
            return person_detections
            
        except Exception as e:
            self.logger.error(f"Person detection failed: {e}")
            return []
    
    def create_person_mask(self, depth_frame: np.ndarray) -> np.ndarray:
        """Create a dummy person mask for compatibility."""
        return np.zeros((depth_frame.shape[0], depth_frame.shape[1]), dtype=np.uint8)
    
    def preprocess_depth_frame(self, depth_frame: np.ndarray) -> np.ndarray:
        """Preprocess depth frame."""
        depth_meters = depth_frame.astype(np.float32) / 1000.0
        depth_meters[depth_meters == 0] = np.nan
        return depth_meters
    
    def visualize_detections(self, color_frame: np.ndarray, detections: List[PersonDetection], 
                           person_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Create visualization of YOLO detections."""
        vis_frame = color_frame.copy()
        
        for detection in detections:
            x, y, w, h = detection.bounding_box
            
            # Color based on confidence zone
            if detection.zone == 'high':
                color = (0, 255, 0)  # Green
            elif detection.zone == 'medium':
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(vis_frame, (center_x, center_y), 5, color, -1)
            
            # Add confidence text
            conf_text = f"Person: {detection.confidence:.2f}"
            cv2.putText(vis_frame, conf_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add depth info if available
            if detection.position_3d[2] > 0:
                depth_text = f"Depth: {detection.position_3d[2]:.1f}m"
                cv2.putText(vis_frame, depth_text, (x, y + h + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return vis_frame


# Test the detector
if __name__ == "__main__":
    import time
    import sys
    import os
    
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from camera.realsense_capture import RealSenseCapture
    
    test_config = {
        'depth_width': 640, 'depth_height': 480, 'depth_fps': 30,
        'color_width': 640, 'color_height': 480, 'color_fps': 30,
        'align_streams': True, 'yolo_confidence_threshold': 0.5,
        'camera_height': 2.5, 'camera_tilt': 30,
        'detection_zones': {
            'high_confidence': {'distance_range': (0.5, 8.0), 'weight': 1.0},
            'medium_confidence': {'distance_range': (8.0, 15.0), 'weight': 0.7},
            'low_confidence': {'distance_range': (15.0, 25.0), 'weight': 0.4}
        }
    }
    
    print("Testing YOLO Person Detection...")
    camera = RealSenseCapture(test_config)
    detector = PersonDetector(test_config)
    
    if camera.configure_camera() and camera.start_streaming():
        print("Testing for 30 seconds...")
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < 30:
                depth, color = camera.get_frames()
                if depth is not None and color is not None:
                    detections = detector.detect_people(depth, time.time(), color)
                    vis_frame = detector.visualize_detections(color, detections)
                    
                    print(f"Frame {frame_count}: {len(detections)} people")
                    cv2.imshow('YOLO Detection', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    frame_count += 1
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Test interrupted")
        finally:
            camera.stop_streaming()
            cv2.destroyAllWindows()
    else:
        print("Failed to initialize camera")