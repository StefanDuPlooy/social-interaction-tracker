"""
Detection Configuration - Aggressive settings for fragmented masks
Centralized configuration for person detection parameters
"""

# Person Detection Parameters - Very aggressive settings
PERSON_DETECTION = {
    'min_blob_area': 500,       # much smaller - catch fragments
    'max_blob_area': 100000,    # much larger - catch combined blobs
    'depth_threshold': 0.05,    # smaller - more sensitive background removal
    'min_height': 0.2,          # much smaller - sitting/partial people
    'max_height': 3.0,          # larger - account for perspective
    'min_aspect_ratio': 0.2,    # much more flexible
    'max_aspect_ratio': 10.0,   # much more flexible
}

# Camera Mount Configuration
MOUNT_CONFIG = {
    'height_meters': 2.5,
    'tilt_angle_degrees': 30,
    'corner_position': (1.0, 1.0),  # meters from walls
    'room_dimensions': (8.0, 6.0),  # width, depth in meters
    'camera_height': 2.5,
    'camera_tilt': 30,
}

# Detection Zones - Very permissive
DETECTION_ZONES = {
    'high_confidence': {
        'distance_range': (0.5, 15.0),  # very wide range
        'accuracy_weight': 1.0,
        'interaction_threshold': 0.8,
        'weight': 1.0
    },
    'medium_confidence': {
        'distance_range': (15.0, 20.0),
        'accuracy_weight': 0.7,
        'interaction_threshold': 0.6,
        'weight': 0.7
    },
    'low_confidence': {
        'distance_range': (20.0, 25.0),
        'accuracy_weight': 0.4,
        'interaction_threshold': 0.5,
        'weight': 0.4
    }
}

# Preprocessing Parameters - Less aggressive cleaning to preserve fragments
PREPROCESSING = {
    'median_filter_size': 3,     # smaller - preserve detail
    'morph_kernel_size': (3, 3), # much smaller - don't over-clean
    'dilation_iterations': 3,    # more dilation to connect fragments
    'depth_range': (0.3, 20.0),  # very wide range
    'background_update_rate': 0.05,  # faster adaptation
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