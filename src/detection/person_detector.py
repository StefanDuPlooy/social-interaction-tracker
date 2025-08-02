"""
Improved Person Detection Module
Handles fragmented masks by merging nearby blobs
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
    detection_method: str  # 'blob', 'merged', 'skeleton'
    zone: str  # 'high', 'medium', 'low' confidence
    blob_area: int  # pixel area of detected blob

class ImprovedPersonDetector:
    """Person detection with blob merging for fragmented masks."""
    
    def __init__(self, config: dict):
        """Initialize person detector with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Detection parameters from config
        self.min_blob_area = config.get('min_blob_area', 500)
        self.max_blob_area = config.get('max_blob_area', 100000)
        self.depth_threshold = config.get('depth_threshold', 0.05)
        self.min_height = config.get('min_height', 0.2)
        self.max_height = config.get('max_height', 3.0)
        
        # Camera configuration for 3D calculations
        self.camera_height = config.get('camera_height', 2.5)
        self.camera_tilt = config.get('camera_tilt', 30)
        
        # Zone configuration
        self.detection_zones = config.get('detection_zones', {
            'high_confidence': {'distance_range': (0.5, 15.0), 'weight': 1.0},
            'medium_confidence': {'distance_range': (15.0, 20.0), 'weight': 0.7},
            'low_confidence': {'distance_range': (20.0, 25.0), 'weight': 0.4}
        })
        
    def preprocess_depth_frame(self, depth_frame: np.ndarray) -> np.ndarray:
        """Preprocess depth frame for person detection."""
        # Convert depth from millimeters to meters
        depth_meters = depth_frame.astype(np.float32) / 1000.0
        
        # Remove invalid depths (0 values)
        depth_meters[depth_meters == 0] = np.nan
        
        # Apply smaller median filter to preserve detail
        depth_filtered = cv2.medianBlur(depth_meters.astype(np.float32), 3)
        
        # Less aggressive morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        depth_filled = cv2.morphologyEx(depth_filtered, cv2.MORPH_CLOSE, kernel)
        
        return depth_filled
    
    def create_person_mask(self, depth_frame: np.ndarray) -> np.ndarray:
        """Create binary mask highlighting potential person regions."""
        # Define wider depth range
        min_depth = 0.3  # meters
        max_depth = 20.0  # meters
        
        # Create initial mask based on depth range
        depth_mask = (depth_frame >= min_depth) & (depth_frame <= max_depth)
        depth_mask = depth_mask.astype(np.uint8) * 255
        
        # More sensitive background removal
        depth_valid = depth_frame[~np.isnan(depth_frame)]
        if len(depth_valid) > 0:
            # Use median instead of mode for more robust background estimation
            background_depth = np.median(depth_valid)
            
            # More sensitive threshold
            background_mask = np.abs(depth_frame - background_depth) > self.depth_threshold
            depth_mask = depth_mask & background_mask.astype(np.uint8) * 255
        
        # Less aggressive morphological operations to preserve fragments
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        larger_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Remove very small noise
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, small_kernel)
        
        # Connect nearby fragments with more dilation
        depth_mask = cv2.dilate(depth_mask, larger_kernel, iterations=3)
        
        # Fill holes
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_CLOSE, larger_kernel)
        
        return depth_mask
    
    def merge_nearby_blobs(self, contours: List, depth_frame: np.ndarray, max_distance: float = 100.0) -> List:
        """Merge nearby blob contours that likely belong to the same person."""
        if len(contours) <= 1:
            return contours
        
        # Calculate centroids of all contours
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
            else:
                centroids.append(None)
        
        # Group nearby contours using DBSCAN clustering
        valid_centroids = [(i, c) for i, c in enumerate(centroids) if c is not None]
        if len(valid_centroids) < 2:
            return contours
        
        # Extract coordinates for clustering
        coords = np.array([c[1] for c in valid_centroids])
        indices = [c[0] for c in valid_centroids]
        
        # Cluster nearby centroids
        clustering = DBSCAN(eps=max_distance, min_samples=1).fit(coords)
        
        # Merge contours in the same cluster
        merged_contours = []
        for cluster_id in set(clustering.labels_):
            cluster_indices = [indices[i] for i, label in enumerate(clustering.labels_) if label == cluster_id]
            
            if len(cluster_indices) == 1:
                # Single contour - keep as is
                merged_contours.append(contours[cluster_indices[0]])
            else:
                # Multiple contours - merge them
                merged_contour = self._merge_contour_group([contours[i] for i in cluster_indices])
                if merged_contour is not None:
                    merged_contours.append(merged_contour)
        
        return merged_contours
    
    def _merge_contour_group(self, contour_group: List) -> Optional[np.ndarray]:
        """Merge a group of contours into one."""
        if not contour_group:
            return None
        
        if len(contour_group) == 1:
            return contour_group[0]
        
        # Create combined mask from all contours
        # Get the bounding box of all contours combined
        all_points = np.vstack(contour_group)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Create mask for this region
        mask = np.zeros((h + 10, w + 10), dtype=np.uint8)
        
        # Draw all contours on the mask
        for contour in contour_group:
            # Offset contour to mask coordinates
            offset_contour = contour - [x - 5, y - 5]
            cv2.fillPoly(mask, [offset_contour], 255)
        
        # Find the outer contour of the merged mask
        merged_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if merged_contours:
            # Convert back to original coordinates
            largest_contour = max(merged_contours, key=cv2.contourArea)
            return largest_contour + [x - 5, y - 5]
        
        return None
    
    def find_person_blobs(self, mask: np.ndarray, depth_frame: np.ndarray) -> List[Dict]:
        """Find and analyze person-shaped blobs in the mask."""
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by minimum area first
        filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_blob_area]
        
        # Merge nearby blobs that might be fragments of the same person
        merged_contours = self.merge_nearby_blobs(filtered_contours, depth_frame)
        
        person_blobs = []
        
        for i, contour in enumerate(merged_contours):
            # Calculate contour properties
            area = cv2.contourArea(contour)
            
            # Filter by area again after merging
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
            
            # More flexible aspect ratio validation
            aspect_ratio = h / w if w > 0 else 0
            
            # Much more flexible aspect ratio range
            if aspect_ratio < 0.2 or aspect_ratio > 10.0:
                continue
            
            # Estimate person height in 3D space
            person_height_pixels = h
            estimated_height = self._pixel_height_to_meters(person_height_pixels, centroid_depth)
            
            # More flexible height validation
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
                'contour': contour,
                'detection_method': 'merged' if len(filtered_contours) > len(merged_contours) else 'blob'
            }
            
            person_blobs.append(blob)
        
        return person_blobs
    
    def _pixel_height_to_meters(self, pixel_height: int, depth_meters: float) -> float:
        """Convert pixel height to estimated real-world height in meters."""
        # Intel RealSense D455 vertical FOV is approximately 58 degrees
        vertical_fov_rad = np.radians(58)
        
        # Height of field of view at given depth
        fov_height_at_depth = 2 * depth_meters * np.tan(vertical_fov_rad / 2)
        
        # Assume 480 pixels in vertical direction
        pixels_per_meter = 480 / fov_height_at_depth
        
        height_meters = pixel_height / pixels_per_meter
        
        return height_meters
    
    def _pixel_to_3d_position(self, pixel_x: int, pixel_y: int, depth_meters: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates and depth to 3D position."""
        # Intel RealSense D455 FOV: ~87° horizontal, ~58° vertical
        image_width = 640
        image_height = 480
        
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
            
            # Find person blobs with merging
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
                    detection_method=blob['detection_method'],
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
            
            # Add method info
            method_text = f"Method: {detection.detection_method}"
            cv2.putText(vis_frame, method_text, (x, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return vis_frame