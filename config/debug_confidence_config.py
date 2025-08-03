
# DEBUG: Add this to your orientation_config.py to see actual confidence values

# Temporary debug settings - VERY LOW thresholds to ensure methods pass
DEBUG_ORIENTATION_METHODS = {
    'skeleton_based': {
        'enabled': True,
        'priority': 1,
        'confidence_threshold': 0.1,  # Very low for debugging
        'required_joints': ['left_shoulder', 'right_shoulder', 'neck', 'nose'],
        'fallback_joints': ['left_shoulder', 'right_shoulder'],
    },
    'movement_based': {
        'enabled': True,
        'priority': 2,
        'min_movement_threshold': 0.01,  # Very low for debugging
        'smoothing_window': 3,  # Reduced window
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

# DEBUG: Very low joint confidence threshold
DEBUG_SKELETON_ORIENTATION = {
    'joint_confidence_threshold': 0.1,  # Very low for debugging
    'required_joint_count': 2,
    'shoulder_vector_weight': 0.7,
    'head_vector_weight': 0.3,
    'temporal_smoothing': True,
    'smoothing_alpha': 0.7,
    'max_angle_change': 45,
    'use_face_keypoints': True,
    'face_keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
    'face_confidence_threshold': 0.1,  # Very low for debugging
}
