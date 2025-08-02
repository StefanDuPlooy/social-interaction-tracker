"""
Intel RealSense D455 Camera Interface
First implementation following the guide's Phase 1: Foundation
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import logging
from typing import Tuple, Optional
import time

class RealSenseCapture:
    """Handle Intel RealSense D455 camera operations."""
    
    def __init__(self, config: dict):
        """Initialize camera with configuration parameters."""
        self.config = config
        self.pipeline = rs.pipeline()
        self.pipeline_config = rs.config()
        self.align = None
        self.is_streaming = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def configure_camera(self) -> bool:
        """Configure camera streams according to config parameters."""
        try:
            # Configure depth and color streams
            self.pipeline_config.enable_stream(
                rs.stream.depth,
                self.config.get('depth_width', 640),
                self.config.get('depth_height', 480),
                rs.format.z16,
                self.config.get('depth_fps', 30)
            )
            
            self.pipeline_config.enable_stream(
                rs.stream.color,
                self.config.get('color_width', 640),
                self.config.get('color_height', 480),
                rs.format.bgr8,
                self.config.get('color_fps', 30)
            )
            
            # Create alignment object (align depth to color)
            if self.config.get('align_streams', True):
                self.align = rs.align(rs.stream.color)
            
            self.logger.info("Camera configuration completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Camera configuration failed: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """Start camera streaming."""
        try:
            # Start streaming
            profile = self.pipeline.start(self.pipeline_config)
            self.is_streaming = True
            
            # Get device info
            device = profile.get_device()
            self.logger.info(f"Connected to: {device.get_info(rs.camera_info.name)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return False
    
    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Capture and return depth and color frames."""
        if not self.is_streaming:
            return None, None
            
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Align frames if configured
            if self.align:
                frames = self.align.process(frames)
            
            # Get depth and color frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return depth_image, color_image
            
        except Exception as e:
            self.logger.error(f"Frame capture failed: {e}")
            return None, None
    
    def stop_streaming(self):
        """Stop camera streaming and cleanup."""
        if self.is_streaming:
            self.pipeline.stop()
            self.is_streaming = False
            self.logger.info("Camera streaming stopped")


# Test the camera interface
if __name__ == "__main__":
    # Test configuration from implementation guide
    test_config = {
        'depth_width': 640,
        'depth_height': 480,
        'depth_fps': 30,
        'color_width': 640,
        'color_height': 480,
        'color_fps': 30,
        'align_streams': True
    }
    
    camera = RealSenseCapture(test_config)
    
    print("Testing RealSense D455 camera interface...")
    
    if camera.configure_camera() and camera.start_streaming():
        print("Camera started successfully. Capturing 10 test frames...")
        
        for i in range(10):
            depth, color = camera.get_frames()
            if depth is not None and color is not None:
                print(f"Frame {i+1}: Depth shape {depth.shape}, Color shape {color.shape}")
                
                # Save test frames (as per implementation guide)
                cv2.imwrite(f"test_depth_{i:03d}.png", depth)
                cv2.imwrite(f"test_color_{i:03d}.png", color)
            else:
                print(f"Frame {i+1}: Failed to capture")
            
            time.sleep(0.1)
        
        camera.stop_streaming()
        print("Test completed successfully!")
    else:
        print("Failed to initialize camera")