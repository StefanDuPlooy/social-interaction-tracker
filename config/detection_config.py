"""
Detection Configuration
Centralized configuration for person detection parameters
"""

# Person Detection Parameters
PERSON_DETECTION = {
    'min_blob_area': 5000,      # minimum pixels for person blob
    'max_blob_area': 50000,     # maximum pixels for person blob
    'depth_threshold': 0.1,     # meters depth difference for background removal
    'min_height': 0.5,          # meters minimum person height
    'max_height': 2.2,          # meters maximum person height
    'min_aspect_ratio': 0.8,    # minimum height/width ratio
    'max_aspect_ratio': 4.0,    # maximum height/width ratio
}

# Camera Mount Configuration
MOUNT_CONFIG = {
    'height_meters': 2.5,
    'tilt_angle_degrees': 30,
    'corner_position': (1.0, 1.0),  # meters from walls
    'room_dimensions': (8.0, 6.0),  # width, depth in meters
}

# Detection Zones - Based on distance from camera
DETECTION_ZONES = {
    'high_confidence': {
        'distance_range': (2.0, 6.0),  # meters from camera
        'accuracy_weight': 1.0,
        'interaction_threshold': 0.8,
    },
    'medium_confidence': {
        'distance_range': (6.0, 8.0),
        'accuracy_weight': 0.7,
        'interaction_threshold': 0.6,
    },
    'low_confidence': {
        'distance_range': (8.0, 10.0),
        'accuracy_weight': 0.4,
        'interaction_threshold': 0.5,
    }
}

# Preprocessing Parameters
PREPROCESSING = {
    'median_filter_size': 5,     # for noise reduction
    'morph_kernel_size': (7, 7), # for morphological operations
    'dilation_iterations': 2,     # for mask cleanup
    'depth_range': (1.0, 10.0),  # valid depth range in meters
}

# Visualization Settings
VISUALIZATION = {
    'colors': {
        'high_confidence': (0, 255, 0),    # Green
        'medium_confidence': (0, 255, 255), # Yellow
        'low_confidence': (0, 0, 255),     # Red
    },
    'thickness': 2,
    'font': 'cv2.FONT_HERSHEY_SIMPLEX',
    'font_scale': 0.5,
}