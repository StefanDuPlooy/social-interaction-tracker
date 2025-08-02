"""
Fixed Person Detector with Proper Depth Processing
Resolves the 0.0m depth and 0.30 confidence issue
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
    depth_meters: float  # Add explicit depth field

class PersonDetector:
    """Fixed YOLO + Depth person detector."""
    
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
        
        # Zone configuration - FIXED: Extended ranges for better coverage
        self.detection_zones = config.get('detection_zones', {
            'high_confidence': {'distance_range': (0.3, 5.0), 'weight': 1.0},
            'medium_confidence': {'distance_range': (5.0, 10.0), 'weight': 0.7},
            'low_confidence': {'distance_range': (10.0, 25.0), 'weight': 0.4}
        })
        
        # Initialize YOLO model
        self.model = None
        self.model_loaded = False
        self._load_yolo_model()
        
        # Debug counters
        self.frame_count = 0
        self.depth_success_count = 0
    
    def _load_yolo_model(self):
        """Load YOLOv8 model."""
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available. Please install ultralytics")
            return False
        
        try:
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
    
    def _get_robust_depth(self, depth_frame: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """FIXED: Get robust depth estimate from detection bounding box."""
        try:
            # Use the center region of the bounding box (50% of area)
            margin_x = max(1, w // 4)
            margin_y = max(1, h // 4)
            
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
            
            # FIXED: Filter valid depths more carefully
            # Remove zeros, NaN, and unrealistic values
            valid_mask = (roi > 0.1) & (roi < 20.0) & ~np.isnan(roi) & ~np.isinf(roi)
            valid_depths = roi[valid_mask]
            
            if len(valid_depths) > 10:  # Need enough valid pixels
                # Use median instead of percentile for robustness
                depth_value = np.median(valid_depths)
                
                # Sanity check the depth value
                if 0.1 < depth_value < 20.0:
                    self.depth_success_count += 1
                    self.logger.debug(f"Valid depth found: {depth_value:.2f}m ({len(valid_depths)} pixels)")
                    return float(depth_value)
            
            # Fallback: Try center pixel
            center_y = y + h // 2
            center_x = x + w // 2
            if (0 <= center_y < depth_frame.shape[0] and 
                0 <= center_x < depth_frame.shape[1]):
                center_depth = depth_frame[center_y, center_x]
                if not np.isnan(center_depth) and 0.1 < center_depth < 20.0:
                    self.logger.debug(f"Using center depth: {center_depth:.2f}m")
                    return float(center_depth)
            
            # Log why depth extraction failed
            self.logger.debug(f"Depth extraction failed: valid_pixels={len(valid_depths)}, "
                            f"roi_size={roi_w}x{roi_h}, frame_size={depth_frame.shape}")
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Depth calculation error: {e}")
            return 0.0
    
    def add_depth_to_detections(self, rgb_detections: List[Dict], depth_frame: np.ndarray) -> List[Dict]:
        """FIXED: Add 3D position information to RGB detections using depth data."""
        enhanced_detections = []
        
        # FIXED: Ensure depth is in meters (convert if needed)
        if depth_frame.dtype == np.uint16:
            # Assume depth is in millimeters, convert to meters
            depth_meters = depth_frame.astype(np.float32) / 1000.0
        else:
            depth_meters = depth_frame.astype(np.float32)
            # If values are too large, they're probably in millimeters
            if np.nanmean(depth_meters[depth_meters > 0]) > 100:
                depth_meters = depth_meters / 1000.0
        
        self.logger.debug(f"Depth stats: min={np.nanmin(depth_meters):.2f}, "
                         f"max={np.nanmax(depth_meters):.2f}, "
                         f"mean={np.nanmean(depth_meters[depth_meters>0]):.2f}")
        
        for detection in rgb_detections:
            x, y, w, h = detection['bounding_box']
            centroid_x, centroid_y = detection['centroid']
            
            # Get robust depth from detection area
            centroid_depth = self._get_robust_depth(depth_meters, x, y, w, h)
            
            if centroid_depth > 0.1:  # Valid depth found
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
        
        # Log success rate
        if len(enhanced_detections) > 0:
            depth_success_rate = sum(1 for d in enhanced_detections if d['depth'] > 0) / len(enhanced_detections)
            self.logger.info(f"Depth extraction success rate: {depth_success_rate:.1%}")
        
        return enhanced_detections
    
    def _pixel_to_3d_position(self, pixel_x: int, pixel_y: int, depth_meters: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates and depth to 3D position."""
        # RealSense D455 parameters
        image_width = 640
        image_height = 480
        h_fov = np.radians(87)  # Horizontal FOV
        v_fov = np.radians(58)  # Vertical FOV
        
        # Calculate angles from center
        h_angle = (pixel_x - image_width/2) * (h_fov / image_width)
        v_angle = (pixel_y - image_height/2) * (v_fov / image_height)
        
        # Convert to 3D coordinates
        x = depth_meters * np.tan(h_angle)
        y = depth_meters * np.tan(v_angle)
        z = depth_meters
        
        return (float(x), float(y), float(z))
    
    def _calculate_detection_confidence(self, depth: float) -> Tuple[str, float]:
        """Calculate detection confidence based on distance zones."""
        for zone_name, zone_config in self.detection_zones.items():
            min_dist, max_dist = zone_config['distance_range']
            if min_dist <= depth <= max_dist:
                zone_type = zone_name.split('_')[0]  # Extract 'high', 'medium', or 'low'
                return zone_type, zone_config['weight']
        
        # Default to low confidence if outside all zones
        return 'low', 0.3
    
    def detect_people(self, depth_frame: np.ndarray, timestamp: float, 
                     color_frame: Optional[np.ndarray] = None) -> List[PersonDetection]:
        """Main detection function - detect people using YOLO + depth."""
        self.frame_count += 1
        
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
                    blob_area=detection['area'],
                    depth_meters=detection['depth']  # Add explicit depth
                )
                person_detections.append(person_detection)
            
            # Log overall statistics every 100 frames
            if self.frame_count % 100 == 0:
                success_rate = (self.depth_success_count / max(1, self.frame_count * len(person_detections))) * 100
                self.logger.info(f"Overall depth success rate: {success_rate:.1f}%")
            
            return person_detections
            
        except Exception as e:
            self.logger.error(f"Person detection failed: {e}", exc_info=True)
            return []
    
    def visualize_detections(self, color_frame: np.ndarray, detections: List[PersonDetection], 
                           person_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Create visualization of YOLO detections with depth info."""
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
            
            # Add text with actual depth value
            conf_text = f"Conf: {detection.confidence:.2f}"
            cv2.putText(vis_frame, conf_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show actual depth value
            if hasattr(detection, 'depth_meters') and detection.depth_meters > 0:
                depth_text = f"Depth: {detection.depth_meters:.2f}m"
            elif detection.position_3d[2] > 0:
                depth_text = f"Depth: {detection.position_3d[2]:.2f}m"
            else:
                depth_text = "Depth: N/A"
            
            cv2.putText(vis_frame, depth_text, (x, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Show zone
            zone_text = f"Zone: {detection.zone}"
            cv2.putText(vis_frame, zone_text, (x, y + h + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add frame statistics
        stats_text = f"Detections: {len(detections)}"
        cv2.putText(vis_frame, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    # Compatibility methods
    def create_person_mask(self, depth_frame: np.ndarray) -> np.ndarray:
        """Create a dummy person mask for compatibility."""
        return np.zeros((depth_frame.shape[0], depth_frame.shape[1]), dtype=np.uint8)
    
    def preprocess_depth_frame(self, depth_frame: np.ndarray) -> np.ndarray:
        """Preprocess depth frame."""
        if depth_frame.dtype == np.uint16:
            depth_meters = depth_frame.astype(np.float32) / 1000.0
        else:
            depth_meters = depth_frame.astype(np.float32)
            if np.nanmean(depth_meters[depth_meters > 0]) > 100:
                depth_meters = depth_meters / 1000.0
        
        depth_meters[depth_meters == 0] = np.nan
        return depth_meters