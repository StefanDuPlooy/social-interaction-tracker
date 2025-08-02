# src/__init__.py
"""
Social Interaction Tracking System
Main package initialization
"""

__version__ = "0.1.0"
__author__ = "JS du Plooy"
__email__ = "40954129@mynwu.ac.za"

# src/camera/__init__.py
"""
Camera module for depth camera interfaces
"""

from .realsense_capture import RealSenseCapture
from .frame_processor import FrameProcessor, ProcessedFrames

__all__ = ['RealSenseCapture', 'FrameProcessor', 'ProcessedFrames']