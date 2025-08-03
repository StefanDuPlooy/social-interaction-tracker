"""
Proximity Analysis Configuration
Phase 2 Step 1 - Proximity-based interaction detection parameters
"""

# Proxemic Zones based on Edward T. Hall's research
# Distances are in meters
PROXEMIC_ZONES = {
    'intimate': (0.0, 0.45),     # 0-45cm: intimate distance (close family, romantic partners)
    'personal': (0.45, 1.2),     # 45cm-1.2m: personal distance (close friends, casual conversation)
    'social': (1.2, 3.6),        # 1.2-3.6m: social distance (acquaintances, formal interactions)
    'public': (3.6, 7.5),        # 3.6-7.5m: public distance (presentations, formal settings)
}

# Proximity Analysis Parameters
PROXIMITY_ANALYSIS = {
    'proxemic_zones': PROXEMIC_ZONES,
    'min_proximity_duration': 2.0,           # Minimum seconds for valid proximity
    'proximity_update_interval': 0.1,        # Update frequency in seconds
    'max_tracking_distance': 10.0,           # Maximum distance to track (meters)
    'temporal_smoothing_window': 5,           # Frames to smooth over
    'confidence_threshold': 0.5,              # Minimum confidence for proximity events
}

# Zone-specific parameters for different contexts
CLASSROOM_PROXIMITY = {
    **PROXIMITY_ANALYSIS,
    'proxemic_zones': {
        'intimate': (0.0, 0.3),     # Very close (whispering, sharing materials)
        'personal': (0.3, 1.0),     # Close collaboration
        'social': (1.0, 2.5),       # Normal classroom interaction
        'public': (2.5, 5.0),       # Across desks, teacher-student
    },
    'min_proximity_duration': 3.0,           # Longer for classroom (less movement)
    'max_tracking_distance': 8.0,            # Classroom size limit
}

# Visualization settings for proximity events
PROXIMITY_VISUALIZATION = {
    'zone_colors': {
        'intimate': (255, 0, 0),      # Red - very close
        'personal': (255, 165, 0),    # Orange - close
        'social': (255, 255, 0),      # Yellow - social
        'public': (0, 255, 0),        # Green - public
        'distant': (128, 128, 128),   # Gray - too far
    },
    'line_thickness': {
        'intimate': 4,
        'personal': 3,
        'social': 2,
        'public': 1,
        'distant': 1,
    },
    'show_distance_labels': True,
    'show_zone_labels': True,
    'show_duration': True,
    'label_font_size': 0.4,
}

# Performance tuning
PROXIMITY_PERFORMANCE = {
    'max_people_analyzed': 20,                # Limit for performance
    'distance_calculation_optimization': True, # Use spatial indexing if available
    'history_retention_seconds': 300,        # 5 minutes of history
    'statistics_update_interval': 10.0,      # Update stats every 10 seconds
}

# Quality control parameters
PROXIMITY_QUALITY = {
    'min_detection_confidence': 0.3,         # Person must be detected with this confidence
    'min_tracking_stability': 0.5,           # Track stability requirement
    'distance_noise_threshold': 0.05,        # 5cm noise tolerance
    'temporal_consistency_check': True,       # Validate proximity over time
    'outlier_rejection_enabled': True,       # Remove distance outliers
}