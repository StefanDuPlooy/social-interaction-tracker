"""
Tracking Configuration Parameters
Phase 1 Step 3 - Optimized for classroom environment
"""

# Tracking Parameters
TRACKING_PARAMETERS = {
    # Association thresholds
    'max_distance_jump': 1.5,  # Maximum distance a person can move between frames (meters)
    'max_disappeared_frames': 30,  # Frames before removing a lost track (1 second at 30 FPS)
    'min_confidence_for_new_track': 0.6,  # Minimum confidence to create new track
    
    # Kalman filter parameters
    'kalman_process_noise': 0.1,  # Process noise covariance
    'kalman_measurement_noise': 0.5,  # Measurement noise covariance
    
    # Trajectory smoothing
    'trajectory_history_length': 30,  # Number of positions to keep in history
    'smoothing_window': 5,  # Number of recent positions for smoothing
    'velocity_estimation_window': 3,  # Minimum positions needed for velocity
    
    # Stability criteria
    'min_frames_for_stability': 5,  # Minimum frames to consider track stable
    'position_variance_threshold': 0.5,  # Maximum variance for stable track
    
    # Performance tuning
    'max_tracks': 20,  # Maximum number of simultaneous tracks
    'cleanup_interval': 100,  # Frames between cleanup operations
}

# Zone-specific tracking adjustments
ZONE_TRACKING_ADJUSTMENTS = {
    'high_confidence': {
        'distance_multiplier': 1.0,  # Full distance threshold
        'confidence_boost': 0.1,  # Add to detection confidence
        'stability_bonus': True,  # Easier to achieve stability
    },
    'medium_confidence': {
        'distance_multiplier': 0.8,  # Reduce distance threshold
        'confidence_boost': 0.05,
        'stability_bonus': False,
    },
    'low_confidence': {
        'distance_multiplier': 0.6,  # Further reduce distance
        'confidence_boost': 0.0,
        'stability_bonus': False,
    }
}

# Visualization settings for tracking
TRACKING_VISUALIZATION = {
    'track_colors': [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (255, 192, 203), # Pink
        (0, 128, 0),    # Dark Green
    ],
    'trajectory_length': 10,  # Number of past positions to show
    'trajectory_thickness': 2,
    'id_text_size': 0.8,
    'id_text_thickness': 2,
    'prediction_color': (128, 128, 128),  # Gray for predictions
    'prediction_style': 'dashed',
}

# Classroom-specific parameters
CLASSROOM_TRACKING = {
    # Typical classroom movement patterns
    'typical_walking_speed': 1.0,  # meters per second
    'max_running_speed': 3.0,  # meters per second
    'sitting_movement_threshold': 0.2,  # Small movements when sitting
    
    # Seating area considerations
    'desk_spacing': 1.2,  # Typical distance between desk centers
    'row_spacing': 1.5,   # Distance between rows
    'aisle_width': 0.8,   # Width of aisles
    
    # Common occlusion scenarios
    'teacher_desk_area': (1.0, 1.0, 3.0, 2.0),  # x1, y1, x2, y2 in room coordinates
    'high_occlusion_zones': [
        # Areas where people frequently occlude each other
        (2.0, 2.0, 4.0, 4.0),  # Group work area
        (0.5, 3.0, 1.5, 5.0),  # Near door
    ],
}

# Advanced tracking features
ADVANCED_TRACKING = {
    # Multi-hypothesis tracking
    'enable_multi_hypothesis': False,  # Enable for complex scenarios
    'max_hypotheses_per_track': 3,
    
    # Re-identification features
    'enable_reidentification': False,  # Enable if person leaves and returns
    'reid_similarity_threshold': 0.7,
    
    # Interaction-aware tracking
    'interaction_influenced_tracking': True,  # Adjust based on interactions
    'group_movement_detection': True,  # Detect people moving together
    
    # Adaptive parameters
    'adaptive_distance_threshold': True,  # Adjust based on movement patterns
    'adaptive_confidence_threshold': True,  # Adjust based on zone performance
}

# Debug and logging settings
TRACKING_DEBUG = {
    'log_associations': False,  # Log every detection-track association
    'log_new_tracks': True,     # Log when new tracks are created
    'log_lost_tracks': True,    # Log when tracks are lost
    'log_statistics_interval': 100,  # Frames between statistics logging
    'save_tracking_debug': False,    # Save debug images
    'debug_output_dir': 'data/tracking_debug',
}

# Performance monitoring
TRACKING_PERFORMANCE = {
    'target_fps': 20,  # Target frames per second
    'max_processing_time': 0.05,  # Maximum seconds per frame
    'memory_limit_mb': 500,  # Maximum memory usage
    'enable_profiling': False,  # Enable performance profiling
    'profile_output': 'data/performance/tracking_profile.txt',
}