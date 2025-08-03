"""
Orientation Detection Configuration
Phase 2 Step 2 - Body and face orientation estimation parameters
"""

import numpy as np

# Orientation Detection Methods
ORIENTATION_METHODS = {
    'skeleton_based': {
        'enabled': True,
        'priority': 1,
        'confidence_threshold': 0.6,
        'required_joints': ['left_shoulder', 'right_shoulder', 'neck', 'nose'],
        'fallback_joints': ['left_shoulder', 'right_shoulder'],
    },
    'movement_based': {
        'enabled': True,
        'priority': 2,
        'min_movement_threshold': 0.1,  # meters
        'smoothing_window': 5,  # frames
        'confidence_decay': 0.9,  # per frame when stationary
    },
    'depth_gradient': {
        'enabled': True,
        'priority': 3,
        'roi_size': (60, 120),  # width, height in pixels
        'gradient_threshold': 0.05,  # meters depth difference
        'confidence_threshold': 0.3,
    }
}

# Skeleton-Based Orientation Parameters
SKELETON_ORIENTATION = {
    # Joint confidence thresholds (YOLO pose estimation scores)
    'joint_confidence_threshold': 0.3,
    'required_joint_count': 2,  # Minimum joints needed
    
    # Orientation calculation methods
    'shoulder_vector_weight': 0.7,  # Primary orientation from shoulders
    'head_vector_weight': 0.3,     # Secondary from head direction
    
    # Smoothing parameters
    'temporal_smoothing': True,
    'smoothing_alpha': 0.7,  # Exponential smoothing factor
    'max_angle_change': 45,  # Maximum degrees change per frame
    
    # Face direction estimation
    'use_face_keypoints': True,
    'face_keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
    'face_confidence_threshold': 0.4,
}

# Movement-Based Orientation Parameters
MOVEMENT_ORIENTATION = {
    'velocity_history_length': 10,  # Frames to analyze
    'min_velocity_threshold': 0.05,  # m/s minimum for reliable orientation
    'direction_smoothing_window': 7,  # Frames for direction averaging
    'stationary_timeout': 30,  # Frames before considering person stationary
    'confidence_boost_speed': 0.5,  # m/s speed that gives full confidence
}

# Depth Gradient Analysis Parameters
DEPTH_GRADIENT_ORIENTATION = {
    'analysis_roi_ratio': 0.6,  # Fraction of bounding box to analyze
    'gradient_kernel_size': 5,  # Sobel kernel size
    'edge_detection_threshold': 0.03,  # Depth gradient threshold
    'body_front_assumption': True,  # Assume closer parts are front
    'asymmetry_threshold': 0.1,  # Required asymmetry for orientation
}

# Orientation Confidence Scoring
CONFIDENCE_SCORING = {
    'skeleton_base_confidence': 0.9,
    'movement_base_confidence': 0.7,
    'depth_gradient_base_confidence': 0.4,
    
    # Confidence modifiers
    'joint_visibility_bonus': 0.1,  # Per visible joint above minimum
    'temporal_consistency_bonus': 0.2,  # For consistent orientation
    'multi_method_agreement_bonus': 0.15,  # When methods agree
    
    # Confidence penalties
    'occlusion_penalty': 0.3,
    'edge_detection_penalty': 0.2,
    'rapid_change_penalty': 0.25,
}

# Mutual Orientation Analysis
MUTUAL_ORIENTATION = {
    'facing_angle_threshold': 60,  # degrees - people considered "facing" each other
    'attention_angle_threshold': 30,  # degrees - people paying attention to each other
    'orientation_agreement_threshold': 0.5,  # Confidence needed for mutual analysis
    
    # F-formation detection
    'f_formation_enabled': True,
    'o_space_radius': 1.5,  # meters - interaction space radius
    'min_group_size': 2,
    'max_group_size': 6,
    'group_coherence_threshold': 0.6,
}

# Debug Visualization Settings
DEBUG_VISUALIZATION = {
    'show_skeleton_keypoints': True,
    'show_movement_vectors': True,
    'show_depth_analysis': True,
    'show_method_breakdown': True,
    'show_confidence_breakdown': True,
    'keypoint_size': 4,
    'vector_thickness': 2,
    'text_size': 0.4,
    'debug_text_color': (255, 255, 255),
    'error_text_color': (0, 0, 255),
    'success_text_color': (0, 255, 0),
    'warning_text_color': (0, 255, 255),
}

# Enhanced Real-time Processing Parameters
PROCESSING_PARAMS = {
    'orientation_update_interval': 0.1,  # seconds between orientation updates
    'batch_processing_size': 10,  # Process N people at once
    'parallel_processing': False,  # Enable if CPU allows
    'debug_visualization': True,
    'save_orientation_history': True,
    'history_retention_seconds': 300,  # 5 minutes
    'detailed_debug_logging': True,  # Enable detailed debug data collection
    'method_performance_tracking': True,  # Track per-method performance
}

# Visualization Settings
ORIENTATION_VISUALIZATION = {
    'draw_orientation_vectors': True,
    'vector_length': 100,  # pixels
    'vector_colors': {
        'skeleton': (0, 255, 0),      # Green - most reliable
        'movement': (255, 255, 0),    # Yellow - medium reliability  
        'depth_gradient': (255, 0, 0), # Red - least reliable
        'combined': (0, 255, 255),    # Cyan - combined estimate
    },
    'draw_confidence_circle': True,
    'confidence_circle_radius': 30,  # pixels
    'draw_facing_connections': True,
    'facing_line_thickness': 3,
    'mutual_attention_color': (255, 0, 255),  # Magenta
    'show_orientation_angle': True,
    'angle_text_offset': (10, -10),
}

# Classroom-Specific Adjustments
CLASSROOM_ORIENTATION = {
    # Default orientations based on seating
    'default_forward_direction': 0,  # degrees (0 = toward front of class)
    'seat_orientation_prior': True,  # Use seating arrangement as prior
    'desk_facing_penalty': 0.1,  # Reduce confidence when facing away from people
    
    # Teacher detection and handling
    'teacher_area_bounds': None,  # Set based on classroom layout
    'teacher_orientation_weight': 1.5,  # Teachers get higher orientation confidence
    
    # Standing vs sitting adjustments
    'sitting_height_threshold': 1.3,  # meters - assume sitting below this
    'sitting_orientation_focus': 'upper_body',  # Focus on torso/head when sitting
    'standing_orientation_focus': 'full_body',  # Use full body when standing
}

# Error Handling and Fallbacks
ERROR_HANDLING = {
    'skeleton_failure_fallback': 'movement',
    'movement_failure_fallback': 'depth_gradient',
    'all_methods_failure_action': 'use_last_known',
    'last_known_timeout': 60,  # frames
    'default_orientation_uncertainty': 90,  # degrees - full uncertainty
    'error_logging_level': 'WARNING',
}

# Performance Optimization
OPTIMIZATION = {
    'skip_frames_when_stationary': True,
    'stationary_detection_threshold': 0.05,  # meters movement
    'reduce_computation_distance': 8.0,  # meters - simplified processing beyond this
    'orientation_caching': True,
    'cache_validity_frames': 5,
    'parallel_person_processing': False,  # Set True if multi-core available
}