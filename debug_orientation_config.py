#!/usr/bin/env python3
"""
QUICK FIX Configuration for Phase 2 Step 2
Use this configuration to test with very low thresholds
"""

# Replace these values in your orientation_config.py for debugging

DEBUG_ORIENTATION_METHODS = {
    'skeleton_based': {
        'enabled': True,
        'priority': 1,
        'confidence_threshold': 0.1,  # Very low for debugging
        'required_joints': ['left_shoulder', 'right_shoulder'],
        'fallback_joints': ['left_shoulder', 'right_shoulder'],
    },
    'movement_based': {
        'enabled': True,
        'priority': 2,
        'min_movement_threshold': 0.02,  # Very low for debugging
        'smoothing_window': 3,
        'confidence_decay': 0.9,
    },
    'depth_gradient': {
        'enabled': True,
        'priority': 3,
        'roi_size': (60, 120),
        'gradient_threshold': 0.02,  # Very low for debugging
        'confidence_threshold': 0.1,  # Very low for debugging
    }
}

DEBUG_SKELETON_ORIENTATION = {
    'joint_confidence_threshold': 0.1,  # Very low for debugging
    'required_joint_count': 2,
    'shoulder_vector_weight': 0.7,
    'head_vector_weight': 0.3,
    'temporal_smoothing': True,
    'smoothing_alpha': 0.7,
    'max_angle_change': 45,
    'use_face_keypoints': True,
    'face_keypoints': ['nose', 'left_eye', 'right_eye'],
    'face_confidence_threshold': 0.1,  # Very low for debugging
}

# Usage:
# 1. Backup your current orientation_config.py
# 2. Replace ORIENTATION_METHODS with DEBUG_ORIENTATION_METHODS
# 3. Replace SKELETON_ORIENTATION with DEBUG_SKELETON_ORIENTATION  
# 4. Run your test again
# 5. If it works, gradually increase thresholds until you find the right balance
