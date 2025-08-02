"""
Person Detection Module - Phase 1 Step 2
Implements depth-based blob detection for person identification
Following the implementation guide's Phase 1: Foundation
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy import ndimage
from sklearn.cluster import DBSCAN

@dataclass
class PersonDetection:
    """Data class for person detection results."""
    id: int
    timestamp: float
    position_3d: Tuple[float, float, float]  # (x, y, z) in meters
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h) in pixels
    confidence: float  # 0.0-1.0
    detection_method: str  # 'blob', 'skeleton', 'movement'
    zone: str  # 'high', 'medium', 'low' confidence
    blob_area: int  # pixel area of detected blob

class PersonDetector:
    """Person detection using depth-based blob analysis."""
    
    def __init__(self, config: dict):
        """Initialize person detector with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters from config
        self.min_blob_area = config.get('min_blob_area', 5000)
        self.max_blob_area = config.get('max_blob_area', 50000)
        self.depth_threshold = config.get('depth_threshold', 0.1)  # meters
        self.min_height = config.get('min_height', 0.5)  # meters
        self.max_height = config.get('max_height', 2.2)  # meters
        
        # Camera configuration for 3D calculations
        self.camera_height = config.get('camera_height', 2.5)  # meters
        self.camera_tilt = config.get('camera_tilt', 30)  # degrees
        
        # Zone configuration for confidence weighting
        self.detection_zones = config.get('detection_zones', {
            'high_confidence': {'distance_range': (2.0, 6.0), 'weight': 1.0},
            'medium_confidence': {'distance_range': (6.0, 8.0), 'weight': 0.7},
            'low_confidence': {'distance_range': (8.0, 10.0), 'weight': 0.4}
        })
        
    def preprocess_depth_frame(self, depth_frame: np.ndarray) -> np.ndarray:
        """Preprocess depth frame for person detection."""
        # Convert depth from millimeters to meters
        depth_meters = depth_frame.astype(np.float32) / 1000.0
        
        # Remove invalid depths (0 values)
        depth_meters[depth_meters == 0] = np.nan
        
        # Apply median filter to reduce noise
        depth_filtered = cv2.medianBlur(depth_meters.astype(np.float32), 5)
        
        # Fill small holes using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        depth_filled = cv2.morphologyEx(depth_filtered, cv2.MORPH_CLOSE, kernel)
        
        return depth_filled
    
    def create_person_mask(self, depth_frame: np.ndarray) -> np.ndarray:
        """Create binary mask highlighting potential person regions."""
        # Define depth range for people (1-10 meters from camera)
        min_depth = 1.0  # meters
        max_depth = 10.0  # meters
        
        # Create initial mask based on depth range
        depth_mask = (depth_frame >= min_depth) & (depth_frame <= max_depth)
        depth_mask = depth_mask.astype(np.uint8) * 255
        
        # Remove background using statistical analysis
        # Assume background is the most common depth value
        depth_valid = depth_frame[~np.isnan(depth_frame)]
        if len(depth_valid) > 0:
            # Use mode of depth values as background estimate
            hist, bin_edges = np.histogram(depth_valid, bins=50)
            background_depth = bin_edges[np.argmax(hist)]
            
            # Remove areas close to background depth
            background_mask = np.abs(depth_frame - background_depth) > self.depth_threshold
            depth_mask = depth_mask & background_mask.astype(np.uint8) * 255
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Remove noise
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to ensure we capture full person silhouettes
        depth_mask = cv2.dilate(depth_mask, kernel, iterations=2)
        
        return depth_mask
    
    def find_person_blobs(self, mask: np.ndarray, depth_frame: np.ndarray) -> List[Dict]:
        """Find and analyze person-shaped blobs in the mask."""
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        person_blobs = []
        
        for i, contour in enumerate(contours):
            # Calculate contour properties
            area = cv2.contourArea(contour)
            
            # Filter by area (person-sized objects)
            if area < self.min_blob_area or area > self.max_blob_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate blob centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            
            # Get depth at centroid
            if 0 <= centroid_y < depth_frame.shape[0] and 0 <= centroid_x < depth_frame.shape[1]:
                centroid_depth = depth_frame[centroid_y, centroid_x]
                
                # Skip if invalid depth
                if np.isnan(centroid_depth) or centroid_depth <= 0:
                    continue
            else:
                continue
            
            # Calculate aspect ratio for person-like shape validation
            aspect_ratio = h / w if w > 0 else 0
            
            # People typically have aspect ratio between 1.5 and 3.0 when standing
            # Lower values when sitting (1.0 - 2.0)
            if aspect_ratio < 0.8 or aspect_ratio > 4.0:
                continue
            
            # Estimate person height in 3D space
            person_height_pixels = h
            # Simple approximation: height in meters based on distance and pixel height
            estimated_height = self._pixel_height_to_meters(person_height_pixels, centroid_depth)
            
            # Filter by realistic human height
            if estimated_height < self.min_height or estimated_height > self.max_height:
                continue
            
            # Calculate 3D position
            position_3d = self._pixel_to_3d_position(centroid_x, centroid_y, centroid_depth)
            
            # Determine detection zone and confidence
            zone, confidence = self._calculate_detection_confidence(centroid_depth)
            
            # Create blob detection result
            blob = {
                'centroid': (centroid_x, centroid_y),
                'position_3d': position_3d,
                'bounding_box': (x, y, w, h),
                'area': area,
                'aspect_ratio': aspect_ratio,
                'estimated_height': estimated_height,
                'depth': centroid_depth,
                'confidence': confidence,
                'zone': zone,
                'contour': contour
            }
            
            person_blobs.append(blob)
        
        return person_blobs
    
    def _pixel_height_to_meters(self, pixel_height: int, depth_meters: float) -> float:
        """Convert pixel height to estimated real-world height in meters."""
        # Simplified calculation - in reality, would need camera intrinsics
        # Assume approximate field of view and calculate based on distance
        
        # Intel RealSense D455 vertical FOV is approximately 58 degrees
        vertical_fov_rad = np.radians(58)
        
        # Height of field of view at given depth
        fov_height_at_depth = 2 * depth_meters * np.tan(vertical_fov_rad / 2)
        
        # Assume 480 pixels in vertical direction (from camera config)
        pixels_per_meter = 480 / fov_height_at_depth
        
        height_meters = pixel_height / pixels_per_meter
        
        return height_meters
    
    def _pixel_to_3d_position(self, pixel_x: int, pixel_y: int, depth_meters: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates and depth to 3D position."""
        # Simplified 3D conversion - would need camera intrinsics for accuracy
        # Intel RealSense D455 FOV: ~87° horizontal, ~58° vertical
        
        # Assume image center and calculate angles
        image_width = 640  # from camera config
        image_height = 480
        
        # Horizontal and vertical field of view in radians
        h_fov = np.radians(87)
        v_fov = np.radians(58)
        
        # Calculate angle from center for this pixel
        h_angle = (pixel_x - image_width/2) * (h_fov / image_width)
        v_angle = (pixel_y - image_height/2) * (v_fov / image_height)
        
        # Convert to 3D coordinates (camera-centric)
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
        
        # Default to low confidence if outside all zones
        return 'low', 0.3
    
    def detect_people(self, depth_frame: np.ndarray, timestamp: float) -> List[PersonDetection]:
        """Main detection function - find all people in depth frame."""
        try:
            # Preprocess depth frame
            depth_processed = self.preprocess_depth_frame(depth_frame)
            
            # Create person detection mask
            person_mask = self.create_person_mask(depth_processed)
            
            # Find person blobs
            person_blobs = self.find_person_blobs(person_mask, depth_processed)
            
            # Convert to PersonDetection objects
            detections = []
            for i, blob in enumerate(person_blobs):
                detection = PersonDetection(
                    id=-1,  # Will be assigned by tracker
                    timestamp=timestamp,
                    position_3d=blob['position_3d'],
                    bounding_box=blob['bounding_box'],
                    confidence=blob['confidence'],
                    detection_method='blob',
                    zone=blob['zone'],
                    blob_area=blob['area']
                )
                detections.append(detection)
            
            self.logger.debug(f"Detected {len(detections)} people at timestamp {timestamp}")
            return detections
            
        except Exception as e:
            self.logger.error(f"Person detection failed: {e}")
            return []
    
    def visualize_detections(self, color_frame: np.ndarray, detections: List[PersonDetection], 
                           person_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Create visualization of detections for debugging."""
        vis_frame = color_frame.copy()
        
        # Draw detection mask if provided
        if person_mask is not None:
            # Overlay mask in blue
            mask_colored = cv2.applyColorMap(person_mask, cv2.COLORMAP_WINTER)
            vis_frame = cv2.addWeighted(vis_frame, 0.7, mask_colored, 0.3, 0)
        
        # Draw detections
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
            
            # Draw centroid
            centroid_x = x + w // 2
            centroid_y = y + h // 2
            cv2.circle(vis_frame, (centroid_x, centroid_y), 5, color, -1)
            
            # Add text with detection info
            info_text = f"ID:{detection.id} {detection.zone} {detection.confidence:.2f}"
            cv2.putText(vis_frame, info_text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add 3D position info
            pos_text = f"({detection.position_3d[0]:.1f}, {detection.position_3d[1]:.1f}, {detection.position_3d[2]:.1f})"
            cv2.putText(vis_frame, pos_text, (x, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return vis_frame


# Test the person detector
if __name__ == "__main__":
    import time
    from camera.realsense_capture import RealSenseCapture
    
    # Test configuration
    test_config = {
        # Camera config
        'depth_width': 640,
        'depth_height': 480,
        'depth_fps': 30,
        'color_width': 640,
        'color_height': 480,
        'color_fps': 30,
        'align_streams': True,
        
        # Detection config
        'min_blob_area': 5000,
        'max_blob_area': 50000,
        'depth_threshold': 0.1,
        'min_height': 0.5,
        'max_height': 2.2,
        'camera_height': 2.5,
        'camera_tilt': 30,
        
        # Zone config
        'detection_zones': {
            'high_confidence': {'distance_range': (2.0, 6.0), 'weight': 1.0},
            'medium_confidence': {'distance_range': (6.0, 8.0), 'weight': 0.7},
            'low_confidence': {'distance_range': (8.0, 10.0), 'weight': 0.4}
        }
    }
    
    print("Testing Person Detection...")
    
    # Initialize camera and detector
    camera = RealSenseCapture(test_config)
    detector = PersonDetector(test_config)
    
    if camera.configure_camera() and camera.start_streaming():
        print("Camera started. Testing person detection for 30 seconds...")
        print("Press 'q' to quit early")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < 30:  # Run for 30 seconds
                depth, color = camera.get_frames()
                if depth is not None and color is not None:
                    # Detect people
                    timestamp = time.time()
                    detections = detector.detect_people(depth, timestamp)
                    
                    # Create visualization
                    person_mask = detector.create_person_mask(
                        detector.preprocess_depth_frame(depth)
                    )
                    vis_frame = detector.visualize_detections(color, detections, person_mask)
                    
                    # Display results
                    print(f"Frame {frame_count}: Detected {len(detections)} people")
                    for i, detection in enumerate(detections):
                        print(f"  Person {i}: Zone={detection.zone}, "
                              f"Confidence={detection.confidence:.2f}, "
                              f"Position={detection.position_3d}")
                    
                    # Show visualization (if display available)
                    cv2.imshow('Person Detection', vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    frame_count += 1
                
                time.sleep(0.1)  # Limit to ~10 FPS for testing
                
        except KeyboardInterrupt:
            print("Test interrupted by user")
        
        finally:
            camera.stop_streaming()
            cv2.destroyAllWindows()
            print(f"Test completed. Processed {frame_count} frames.")
    
    else:
        print("Failed to initialize camera")