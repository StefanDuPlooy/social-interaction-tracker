"""
Detection Configuration - YOLO + Depth
Configuration for YOLO-based person detection
"""

# Person Detection Parameters - YOLO-based
PERSON_DETECTION = {
    'detector_type': 'yolo',
    'yolo_confidence_threshold': 0.5,  # YOLO detection confidence
    'nms_threshold': 0.4,              # Non-maximum suppression
    'yolo_model': 'yolov8n.pt',        # Model: n(ano), s(mall), m(edium), l(arge), x(large)
}

# Camera Mount Configuration
MOUNT_CONFIG = {
    'height_meters': 2.5,
    'tilt_angle_degrees': 30,
    'corner_position': (1.0, 1.0),
    'room_dimensions': (8.0, 6.0),
    'camera_height': 2.5,
    'camera_tilt': 30,
}

# Detection Zones - Optimized for YOLO
DETECTION_ZONES = {
    'high_confidence': {
        'distance_range': (0.5, 8.0),
        'weight': 1.0,
        'interaction_threshold': 0.8,
    },
    'medium_confidence': {
        'distance_range': (8.0, 15.0),
        'weight': 0.7,
        'interaction_threshold': 0.6,
    },
    'low_confidence': {
        'distance_range': (15.0, 25.0),
        'weight': 0.4,
        'interaction_threshold': 0.5,
    }
}

# Preprocessing Parameters
PREPROCESSING = {
    'depth_range': (0.3, 25.0),
    'depth_filter_percentile': 25,
}

# Visualization Settings
VISUALIZATION = {
    'colors': {
        'high_confidence': (0, 255, 0),    # Green
        'medium_confidence': (0, 255, 255), # Yellow
        'low_confidence': (0, 0, 255),     # Red
    },
    'thickness': 2,
    'font_scale': 0.5,
}