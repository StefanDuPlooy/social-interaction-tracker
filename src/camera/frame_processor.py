"""
Frame Processor Module
Handles depth/RGB frame preprocessing for person detection
Part of Phase 1 Step 2 implementation
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class ProcessedFrames:
    """Container for processed frame data."""
    depth_raw: np.ndarray
    depth_filtered: np.ndarray
    color_frame: np.ndarray
    person_mask: np.ndarray
    background_mask: Optional[np.ndarray] = None
    timestamp: float = 0.0

class FrameProcessor:
    """Handles preprocessing of depth and color frames."""
    
    def __init__(self, config: dict):
        """Initialize frame processor with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Processing parameters
        self.median_filter_size = config.get('median_filter_size', 5)
        self.morph_kernel_size = config.get('morph_kernel_size', (7, 7))
        self.depth_range = config.get('depth_range', (1.0, 10.0))
        self.background_update_rate = config.get('background_update_rate', 0.01)
        
        # Background subtraction for improved detection
        self.background_model = None
        self.frame_count = 0
        
        # Morphological kernels
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self.morph_kernel_size
        )
        self.small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
    def process_depth_frame(self, depth_frame: np.ndarray) -> np.ndarray:
        """Process raw depth frame for person detection."""
        # Convert from millimeters to meters
        depth_meters = depth_frame.astype(np.float32) / 1000.0
        
        # Remove invalid depths
        depth_meters[depth_meters == 0] = np.nan
        
        # Apply depth range filtering
        min_depth, max_depth = self.depth_range
        depth_meters[(depth_meters < min_depth) | (depth_meters > max_depth)] = np.nan
        
        # Noise reduction with median filter
        depth_filtered = cv2.medianBlur(
            depth_meters.astype(np.float32), self.median_filter_size
        )
        
        # Fill small holes using morphological closing
        # Convert to uint8 for morphological operations
        depth_uint8 = self._depth_to_uint8(depth_filtered)
        depth_uint8 = cv2.morphologyEx(depth_uint8, cv2.MORPH_CLOSE, self.small_kernel)
        
        # Convert back to float
        depth_filled = self._uint8_to_depth(depth_uint8, depth_filtered)
        
        return depth_filled
    
    def process_color_frame(self, color_frame: np.ndarray) -> np.ndarray:
        """Process color frame for better visualization."""
        # Apply basic image enhancement
        # Convert to HSV for better processing
        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        
        # Enhance contrast slightly
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Apply slight Gaussian blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def create_background_model(self, depth_frame: np.ndarray):
        """Create or update background model for subtraction."""
        if self.background_model is None:
            # Initialize background model
            self.background_model = depth_frame.copy()
            self.logger.info("Background model initialized")
        else:
            # Update background model using running average
            # Only update pixels that are stable (not moving)
            
            # Calculate difference from current background
            diff = np.abs(depth_frame - self.background_model)
            stable_mask = diff < 0.1  # Pixels that haven't changed much
            
            # Update stable pixels slowly
            alpha = self.background_update_rate
            self.background_model[stable_mask] = (
                (1 - alpha) * self.background_model[stable_mask] + 
                alpha * depth_frame[stable_mask]
            )
    
    def create_foreground_mask(self, depth_frame: np.ndarray) -> np.ndarray:
        """Create foreground mask using background subtraction."""
        if self.background_model is None:
            # No background model yet, create basic mask
            return self._create_basic_depth_mask(depth_frame)
        
        # Calculate difference from background
        diff = np.abs(depth_frame - self.background_model)
        
        # Threshold to create foreground mask
        # Pixels significantly different from background are foreground
        threshold = 0.15  # meters
        foreground_mask = diff > threshold
        
        # Remove invalid pixels
        valid_mask = ~np.isnan(depth_frame) & ~np.isnan(self.background_model)
        foreground_mask = foreground_mask & valid_mask
        
        # Convert to uint8
        mask = (foreground_mask * 255).astype(np.uint8)
        
        # Clean up mask with morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.small_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        return mask
    
    def _create_basic_depth_mask(self, depth_frame: np.ndarray) -> np.ndarray:
        """Create basic mask when no background model is available."""
        # Create mask based on depth range
        min_depth, max_depth = self.depth_range
        valid_mask = (depth_frame >= min_depth) & (depth_frame <= max_depth)
        valid_mask = valid_mask & ~np.isnan(depth_frame)
        
        # Convert to uint8
        mask = (valid_mask * 255).astype(np.uint8)
        
        # Clean up
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.small_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        return mask
    
    def _depth_to_uint8(self, depth_frame: np.ndarray) -> np.ndarray:
        """Convert depth frame to uint8 for morphological operations."""
        # Handle NaN values
        depth_clean = depth_frame.copy()
        depth_clean[np.isnan(depth_clean)] = 0
        
        # Normalize to 0-255 range
        min_depth, max_depth = self.depth_range
        depth_normalized = np.clip((depth_clean - min_depth) / (max_depth - min_depth), 0, 1)
        
        return (depth_normalized * 255).astype(np.uint8)
    
    def _uint8_to_depth(self, uint8_frame: np.ndarray, original_depth: np.ndarray) -> np.ndarray:
        """Convert uint8 frame back to depth values."""
        # Denormalize
        min_depth, max_depth = self.depth_range
        depth_restored = (uint8_frame.astype(np.float32) / 255.0) * (max_depth - min_depth) + min_depth
        
        # Restore NaN values where original had them
        depth_restored[np.isnan(original_depth)] = np.nan
        
        return depth_restored
    
    def calculate_frame_quality(self, depth_frame: np.ndarray, color_frame: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for the frame."""
        metrics = {}
        
        # Depth frame quality
        valid_depth_ratio = np.sum(~np.isnan(depth_frame)) / depth_frame.size
        metrics['valid_depth_ratio'] = valid_depth_ratio
        
        # Depth variance (higher = more interesting/detailed scene)
        valid_depths = depth_frame[~np.isnan(depth_frame)]
        if len(valid_depths) > 0:
            metrics['depth_variance'] = np.var(valid_depths)
            metrics['depth_mean'] = np.mean(valid_depths)
        else:
            metrics['depth_variance'] = 0.0
            metrics['depth_mean'] = 0.0
        
        # Color frame quality (sharpness)
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['color_sharpness'] = laplacian_var
        
        # Overall quality score
        quality_score = (
            valid_depth_ratio * 0.4 +
            min(metrics['depth_variance'] / 2.0, 1.0) * 0.3 +
            min(laplacian_var / 1000.0, 1.0) * 0.3
        )
        metrics['overall_quality'] = quality_score
        
        return metrics
    
    def process_frame_pair(self, depth_frame: np.ndarray, color_frame: np.ndarray, 
                          timestamp: float) -> ProcessedFrames:
        """Process both depth and color frames together."""
        self.frame_count += 1
        
        # Process individual frames
        depth_filtered = self.process_depth_frame(depth_frame)
        color_processed = self.process_color_frame(color_frame)
        
        # Update background model
        self.create_background_model(depth_filtered)
        
        # Create person detection mask
        person_mask = self.create_foreground_mask(depth_filtered)
        
        # Calculate frame quality
        quality_metrics = self.calculate_frame_quality(depth_filtered, color_processed)
        
        # Log quality if it's particularly bad
        if quality_metrics['overall_quality'] < 0.3:
            self.logger.warning(f"Low quality frame at {timestamp}: "
                              f"score={quality_metrics['overall_quality']:.2f}")
        
        return ProcessedFrames(
            depth_raw=depth_frame,
            depth_filtered=depth_filtered,
            color_frame=color_processed,
            person_mask=person_mask,
            background_mask=self.background_model,
            timestamp=timestamp
        )
    
    def reset_background_model(self):
        """Reset the background model (useful when scene changes significantly)."""
        self.background_model = None
        self.frame_count = 0
        self.logger.info("Background model reset")
    
    def save_debug_frames(self, processed_frames: ProcessedFrames, output_dir: str):
        """Save processed frames for debugging."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp_str = f"{processed_frames.timestamp:.3f}".replace('.', '_')
        
        # Save depth frames
        depth_vis = self._depth_to_uint8(processed_frames.depth_filtered)
        cv2.imwrite(f"{output_dir}/depth_filtered_{timestamp_str}.png", depth_vis)
        
        # Save masks
        cv2.imwrite(f"{output_dir}/person_mask_{timestamp_str}.png", processed_frames.person_mask)
        
        if processed_frames.background_mask is not None:
            bg_vis = self._depth_to_uint8(processed_frames.background_mask)
            cv2.imwrite(f"{output_dir}/background_{timestamp_str}.png", bg_vis)
        
        # Save color frame
        cv2.imwrite(f"{output_dir}/color_{timestamp_str}.png", processed_frames.color_frame)


# Test the frame processor
if __name__ == "__main__":
    import time
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from camera.realsense_capture import RealSenseCapture
    
    # Test configuration
    config = {
        'median_filter_size': 5,
        'morph_kernel_size': (7, 7),
        'depth_range': (1.0, 10.0),
        'background_update_rate': 0.01,
        
        # Camera config
        'depth_width': 640,
        'depth_height': 480,
        'depth_fps': 30,
        'color_width': 640,
        'color_height': 480,
        'color_fps': 30,
        'align_streams': True,
    }
    
    print("Testing Frame Processor...")
    
    # Initialize camera and processor
    camera = RealSenseCapture(config)
    processor = FrameProcessor(config)
    
    if camera.configure_camera() and camera.start_streaming():
        print("Testing frame processing for 30 seconds...")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < 30:
                depth, color = camera.get_frames()
                if depth is not None and color is not None:
                    # Process frames
                    timestamp = time.time()
                    processed = processor.process_frame_pair(depth, color, timestamp)
                    
                    # Calculate quality metrics
                    quality = processor.calculate_frame_quality(
                        processed.depth_filtered, processed.color_frame
                    )
                    
                    print(f"Frame {frame_count}: Quality={quality['overall_quality']:.2f}, "
                          f"Valid depth={quality['valid_depth_ratio']:.1%}")
                    
                    # Show processed frames
                    cv2.imshow('Original Color', color)
                    cv2.imshow('Processed Color', processed.color_frame)
                    cv2.imshow('Person Mask', processed.person_mask)
                    
                    # Show depth visualization
                    depth_vis = processor._depth_to_uint8(processed.depth_filtered)
                    cv2.imshow('Depth Filtered', depth_vis)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    frame_count += 1
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("Test interrupted")
        
        finally:
            camera.stop_streaming()
            cv2.destroyAllWindows()
            print(f"Processed {frame_count} frames")
    
    else:
        print("Failed to initialize camera")