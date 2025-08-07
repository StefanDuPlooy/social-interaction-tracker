# Social Interaction Tracking System - Implementation Guide

## Core System Requirements

### 1. Hardware Setup

- **Primary Camera**: Intel RealSense D455 depth camera
- **Mounting Position**: Front corner of classroom at 2.5m height with 30° downward tilt
  - Corner mount maximizes single-camera coverage (87° × 58° FOV)
  - Diagonal viewing angle reduces occlusion between students
  - Optimal 3-8m detection range keeps most participants in high-accuracy zone
- **Coverage**: Captures ~6m × 8m classroom area from corner vantage point
- **Calibration**: Set up coordinate system and distance markers for ground truth validation

### 2. Software Components to Develop

#### A. Person Detection & Tracking Module

```python
# Multi-Modal Detection Strategy:

# Primary Method: Depth-Based Blob Detection
- Extract person-shaped "blobs" from depth frames
- More reliable than skeleton tracking with single camera
- Identifies location and rough size of each person
- Works consistently even when skeleton tracking fails

# Secondary Method: Skeleton Tracking (when available)
- Use OpenPose on depth-registered RGB frames
- Extract 3D joint coordinates (shoulders, spine, head)
- Provides more precise orientation data
- Falls back to blob detection when skeleton fails

# Trajectory Tracking:
- Assign unique IDs to each detected person
- Use Kalman filtering for smooth trajectory tracking
- Handle occlusions with predictive tracking
- Maintain ID consistency as people move around
```

#### B. Spatial Analysis & Orientation Detection Module

```python
# Orientation Detection Methods:

# Method 1: Skeleton-Based Orientation (Preferred)
- Calculate shoulder line vector from left/right shoulder joints
- Determine spine direction from spine base to neck
- Combine shoulder and spine vectors for body orientation
- Face direction = perpendicular to shoulder line, forward from spine

# Method 2: Movement-Based Orientation (Fallback)
- Track person's movement direction over time
- Assume facing direction aligns with movement direction
- Use trajectory smoothing to reduce noise
- Less accurate but provides basic orientation estimate

# Method 3: Depth Gradient Analysis (Backup)
- Analyze depth patterns within person blob
- Front of body typically closer to camera than back
- Use asymmetry in depth profile to estimate facing direction
- Least accurate but better than no orientation data

# Classroom-Specific Adaptations:
- Sitting vs Standing detection based on person height
- Sitting: Focus on upper body orientation (shoulders/head)
- Standing: Use full body posture for orientation
- Confidence scoring based on detection method quality

# Distance Calculations:
- 3D distance between person centroids
- Account for camera angle and perspective correction
- Zone-based confidence weighting (high/medium/low zones)
```

#### C. Interaction Detection Engine

```python
# Multi-Layer Interaction Detection:

# Layer 1: Mutual Facing Detection
- Calculate orientation vectors for each person pair
- Compute angle between their facing directions
- Interaction candidate if angle difference < 60°
- Account for orientation confidence levels

# Layer 2: Proximity Analysis
- Distance threshold: < 1.5m (adjustable for sitting/standing)
- Apply zone-based confidence weighting
- Different thresholds for high/medium/low confidence zones

# Layer 3: Temporal Validation
- Minimum duration: 3 seconds sustained
- Temporal smoothing with sliding window
- Handle brief interruptions (person looks away momentarily)

# Layer 4: Effort Angle Calculation (REFORM-style)
- Calculate how much each person would need to rotate to face other directly
- Range: 0° (facing directly) to 180° (facing away)
- Use as feature for machine learning classification

# Adaptive Learning Enhancement:
- Collect labeled data from initial deployments
- Train classifier on Distance + Effort Angle + Duration features
- Ensemble approach: combine rule-based + ML predictions
- Dynamic threshold adaptation based on classroom context

# Quality Control:
- Confidence scoring for each detected interaction
- Flag uncertain interactions for manual review
- Conservative detection to minimize false positives
```

#### D. Social Network Generator

```python
# Create dynamic graphs:
- Nodes = individual participants
- Edges = detected interactions
- Weights = interaction frequency/duration
- Real-time graph updates
```

## Complete System Architecture & Data Flow

### Core System Pipeline:

```
Intel RealSense D455 → Frame Capture → Person Detection →
Skeleton/Blob Tracking → Orientation Calculation → Interaction Detection →
Social Graph Generation → Real-time Visualization → Data Storage
```

### Modular Code Structure:

```
src/
├── camera/
│   ├── realsense_capture.py      # Camera interface & calibration
│   └── frame_processor.py        # Depth/RGB frame preprocessing
├── detection/
│   ├── person_detector.py        # Blob detection + skeleton tracking
│   ├── tracker.py                # ID management & trajectory tracking
│   └── orientation_estimator.py  # Multi-method orientation detection
├── interaction/
│   ├── proximity_analyzer.py     # Distance calculations & zones
│   ├── interaction_detector.py   # Rule-based + ML interaction detection
│   └── temporal_filter.py        # Sliding window & duration validation
├── network/
│   ├── graph_builder.py          # NetworkX graph construction
│   ├── metrics_calculator.py     # Centrality & network analysis
│   └── visualizer.py             # Real-time graph visualization
├── utils/
│   ├── calibration.py            # Camera calibration & coordinate systems
│   ├── config.py                 # System parameters & thresholds
│   └── logger.py                 # Data logging & privacy compliance
└── main.py                       # Main system orchestrator
```

#### Required Libraries & Tools:

- **Intel RealSense SDK** (librealsense2) - depth data capture
- **OpenCV** - computer vision processing
- **OpenPose** (optional enhancement) - skeleton tracking when feasible
- **scikit-learn** - for learning-based interaction classification
- **NetworkX** - social network graph creation and analysis
- **NumPy/SciPy** - numerical computations and filtering
- **Matplotlib/Plotly** - real-time visualization
- **Python** - primary programming language

#### Handling Single Camera Limitations:

```python
# Strategies to maximize D455 effectiveness:
- Optimal positioning: Corner mount at 2.5m height, 30° downward angle
- Use both depth + RGB streams for robust person detection
- Implement occlusion handling with predictive tracking
- Zone-based analysis to handle edge distortions
- Confidence scoring for detections based on distance from camera

# Occlusion Management:
- Predictive tracking using Kalman filters
- Maintain last known orientation during brief occlusions
- Resume tracking when person becomes visible again
- Interpolate missing data for short gaps (<2 seconds)

# Edge Effect Mitigation:
- Define reliable detection zones (high/medium/low confidence)
- Apply different interaction thresholds based on zone
- Flag detections near field-of-view boundaries
- Weight interactions by detection confidence
```

#### System Architecture:

```
Depth Camera → Frame Capture → Person Detection →
Skeleton Tracking → Spatial Analysis → Interaction Detection →
Social Graph Generation → Visualization & Analysis
```

## Detailed Configuration Parameters

### Camera Configuration:

```python
# realsense_config.py
CAMERA_CONFIG = {
    'depth_width': 640,
    'depth_height': 480,
    'depth_fps': 30,
    'color_width': 640,
    'color_height': 480,
    'color_fps': 30,
    'depth_format': 'z16',  # 16-bit depth
    'color_format': 'bgr8',
    'align_streams': True,  # Align depth to color
}

# Coordinate system calibration
MOUNT_CONFIG = {
    'height_meters': 2.5,
    'tilt_angle_degrees': 30,
    'corner_position': (1.0, 1.0),  # meters from walls
    'room_dimensions': (8.0, 6.0),  # width, depth in meters
}
```

### Detection Parameters:

```python
# detection_config.py
PERSON_DETECTION = {
    'min_blob_area': 5000,      # minimum pixels for person
    'max_blob_area': 50000,     # maximum pixels for person
    'depth_threshold': 0.1,     # meters depth difference
    'min_height': 0.5,          # meters minimum person height
    'max_height': 2.2,          # meters maximum person height
}

INTERACTION_THRESHOLDS = {
    'max_distance_meters': 1.5,
    'max_orientation_angle': 60,    # degrees
    'min_duration_seconds': 3.0,
    'temporal_window_seconds': 5.0,
    'confidence_threshold': 0.7,
}

TRACKING_PARAMETERS = {
    'max_disappeared_frames': 30,   # frames before ID removal
    'max_distance_jump': 1.0,       # meters for ID association
    'kalman_process_noise': 0.1,
    'kalman_measurement_noise': 0.5,
}
```

### Zone-Based Configuration:

```python
# zone_config.py
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
```

### 4. Data Collection & Processing

#### Input Data Specifications:

- **Depth frames**: 640×480 @ 30fps, 16-bit depth values
- **RGB frames**: 640×480 @ 30fps, aligned to depth
- **Coordinate system**: 3D positions in meters from camera origin
- **Temporal data**: Synchronized timestamps for all detections

#### Output Data Format:

```python
# Person detection output
PersonDetection = {
    'id': int,                    # unique person identifier
    'timestamp': float,           # seconds since start
    'position_3d': (x, y, z),    # meters in camera coordinates
    'bounding_box': (x, y, w, h), # pixel coordinates
    'orientation_angle': float,   # degrees (0=facing camera)
    'confidence': float,          # 0.0-1.0 detection confidence
    'detection_method': str,      # 'skeleton', 'blob', 'movement'
    'zone': str,                 # 'high', 'medium', 'low' confidence
}

# Interaction detection output
InteractionEvent = {
    'id1': int, 'id2': int,      # person IDs involved
    'start_time': float,         # interaction start timestamp
    'end_time': float,           # interaction end timestamp
    'duration': float,           # seconds
    'avg_distance': float,       # average distance during interaction
    'avg_orientation_diff': float, # average angle difference
    'confidence': float,         # interaction confidence score
    'detection_method': str,     # 'rule_based', 'ml', 'ensemble'
}

# Social network output
NetworkSnapshot = {
    'timestamp': float,
    'nodes': [PersonDetection],  # all detected people
    'edges': [InteractionEvent], # all active interactions
    'metrics': {
        'density': float,
        'avg_clustering': float,
        'num_components': int,
    }
}
```

### 5. Privacy & Ethical Implementation

#### Must Include:

- **No RGB/video recording** - depth data only
- **Anonymous skeleton IDs** - no facial recognition
- **Data encryption** for stored interaction logs
- **Automatic data deletion** of raw depth frames after processing
- **Consent verification** system integration

### 6. System Robustness & Quality Assurance

#### Orientation Confidence Management:

- **High confidence**: Clear skeleton with visible shoulders/spine
- **Medium confidence**: Partial skeleton or clear movement direction
- **Low confidence**: Depth-gradient based estimation only
- **Very low confidence**: Stationary person with unclear orientation

#### Interaction Validation Strategies:

- Require higher precision for high-confidence detections
- Use more permissive thresholds for low-confidence orientations
- Cross-validate interactions using multiple detection methods
- Temporal consistency checks across multiple frames

#### Known Limitations & Mitigation:

- **People looking down at desks**: Use body orientation as proxy for attention
- **Multiple people close together**: Enhanced blob separation algorithms
- **Head vs body orientation mismatch**: Prioritize sustained body orientation
- **Temporary occlusions**: Predictive tracking with confidence decay

#### Classroom-Specific Challenges:

- **Students seated facing forward**: Detect side-turning as interaction signal
- **Teacher movement patterns**: Separate teacher-student from student-student interactions
- **Group work vs individual work**: Context-aware threshold adjustment
- **Varying lighting conditions**: Depth-primary detection with RGB enhancement

### 7. Evaluation & Validation System

- **Precision**: True interactions / (True + False positives)
- **Recall**: True interactions / (True + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Temporal accuracy**: Timestamp deviation from ground truth
- **ID tracking consistency**: Maintaining person identities across frames

#### Validation Methods:

- **Synchronized ground truth collection**: Use timestamped manual observations
- **Inter-observer reliability**: Multiple observers for validation sessions
- **Controlled interaction scenarios**: Scripted interactions for precision testing
- **Multiple session testing**: Same participants across different days
- **Cross-validation with instructor feedback**: Validate network insights with teacher observations
- **Statistical significance testing**: Use appropriate tests for performance metrics

### 7. Risk Mitigation Strategies

#### Technical Risks & Mitigation:

- **Skeleton tracking failure**: Implement depth-blob fallback method
- **Occlusion issues**: Use predictive tracking and trajectory interpolation
- **Camera positioning constraints**: Develop zone-based confidence weighting
- **Real-time performance**: Start with offline processing, optimize incrementally
- **Classroom lighting variations**: Use depth-primary detection with RGB as enhancement

#### Data Quality Risks & Mitigation:

- **False interaction detection**: Implement conservative temporal thresholds
- **Missing interactions**: Use ensemble approach with multiple detection methods
- **ID switching**: Add trajectory smoothing and appearance-based verification
- **Edge effects**: Define reliable detection zones and flag uncertain detections

### 7. Real-Time System Features

#### Must Implement:

- **Live visualization** of detected interactions
- **Real-time social graph updates**
- **Performance monitoring** (FPS, detection accuracy)
- **System status indicators**
- **Error handling** for camera disconnection/occlusion

### 8. Analysis & Reporting Components

#### Social Network Analysis:

- **Degree centrality** - number of interactions per person
- **Betweenness centrality** - identification of social bridges
- **Clustering coefficient** - detection of subgroups
- **Temporal interaction patterns** - peak interaction times
- **Network density** - overall classroom connectivity

#### Output Reports:

- Interaction frequency matrices
- Network visualization graphs
- Temporal interaction timelines
- Statistical summaries of social patterns

## Implementation Priority & Step-by-Step Instructions

### Phase 1: Foundation (Week 1) - IMPLEMENT FIRST

```python
# Step 1: Camera Interface (Day 1-2)
# File: src/camera/realsense_capture.py
# - Initialize RealSense camera with config parameters
# - Implement frame capture loop with error handling
# - Add basic depth/color alignment
# - Test: Verify camera streams and save test frames

# Step 2: Person Detection (Day 3-4)
# File: src/detection/person_detector.py
# - Implement depth-based blob detection
# - Add basic person filtering (size, height)
# - Create person bounding boxes
# - Test: Detect and count people in saved frames

# Step 3: Basic Tracking (Day 5-7)
# File: src/detection/tracker.py
# - Implement simple ID assignment
# - Add frame-to-frame person association
# - Basic trajectory smoothing
# - Test: Track 2-3 people for 30 seconds
```

### Phase 2: Core Functionality (Week 2) - IMPLEMENT SECOND

```python
# Step 4: Orientation Detection (Day 8-10)
# File: src/detection/orientation_estimator.py
# - Implement movement-based orientation (fallback method)
# - Add depth gradient analysis for stationary people
# - Integrate skeleton tracking if OpenPose available
# - Test: Compare orientation estimates with manual observation

# Step 5: Distance & Proximity (Day 11-12)
# File: src/interaction/proximity_analyzer.py
# - Calculate 3D distances between all person pairs
# - Implement zone-based confidence weighting
# - Account for camera angle and perspective
# - Test: Verify distance accuracy with known measurements

# Step 6: Basic Interaction Detection (Day 13-14)
# File: src/interaction/interaction_detector.py
# - Implement rule-based interaction detection
# - Add temporal filtering and duration requirements
# - Create interaction confidence scoring
# - Test: Detect interactions in controlled scenarios
```

### Phase 3: Advanced Features (Week 3) - IMPLEMENT THIRD

```python
# Step 7: Social Network Generation (Day 15-17)
# File: src/network/graph_builder.py
# - Create NetworkX graphs from interaction events
# - Implement real-time graph updates
# - Add network metrics calculation
# - Test: Generate graphs from recorded interaction data

# Step 8: Visualization & Monitoring (Day 18-19)
# File: src/network/visualizer.py
# - Create real-time person position display
# - Add interaction visualization (lines between people)
# - Implement social network graph display
# - Test: Real-time visualization during live capture

# Step 9: System Integration (Day 20-21)
# File: src/main.py
# - Integrate all modules into single pipeline
# - Add system monitoring and error handling
# - Implement graceful shutdown and cleanup
# - Test: End-to-end system operation
```

### Critical Implementation Notes:

1. **Start with offline processing**: Save frames first, process later for initial development
2. **Use test data**: Create controlled scenarios with 2-3 people for initial testing
3. **Implement fallbacks**: Every component needs error handling and fallback methods
4. **Validate incrementally**: Test each component before moving to next phase
5. **Keep it simple initially**: Use basic algorithms first, optimize later

## Common Implementation Challenges & Solutions

### Challenge 1: RealSense Camera Setup

**Problem**: Camera initialization fails or produces poor quality data
**Solution**:

```python
# Always check camera connection and permissions
# Use specific camera serial if multiple devices
# Implement retry logic for unstable connections
# Add frame validation (check for empty or corrupt frames)
```

### Challenge 2: Person Detection Accuracy

**Problem**: Missed detections or false positives
**Solution**:

```python
# Tune blob detection parameters for your environment
# Use morphological operations to clean depth masks
# Implement multi-frame validation (person must appear in N consecutive frames)
# Add manual threshold adjustment interface for different lighting
```

### Challenge 3: ID Tracking Consistency

**Problem**: Person IDs switch or get lost during movement
**Solution**:

```python
# Use conservative distance thresholds for ID association
# Implement appearance-based verification (depth profile matching)
# Add trajectory prediction during brief occlusions
# Use Kalman filters for smooth position estimation
```

### Challenge 4: Orientation Detection Reliability

**Problem**: Inconsistent or inaccurate orientation estimates  
**Solution**:

```python
# Always implement multiple orientation methods
# Use temporal smoothing (average over multiple frames)
# Add confidence thresholds - ignore low-confidence orientations
# Validate orientation estimates against known classroom layout
```

### Challenge 5: Real-time Performance

**Problem**: System runs too slowly for real-time operation
**Solution**:

```python
# Profile code to identify bottlenecks
# Use frame skipping if necessary (process every 2nd or 3rd frame)
# Implement multi-threading (separate capture and processing threads)
# Reduce frame resolution if needed (320x240 instead of 640x480)
```

## Testing & Validation Procedures

### Unit Testing Requirements:

```python
# test_person_detector.py
- Test blob detection with known depth images
- Verify person filtering (size, height constraints)
- Test edge cases (person at image boundary)

# test_tracker.py
- Test ID assignment with simple scenarios
- Verify trajectory smoothing algorithms
- Test occlusion handling (person disappears/reappears)

# test_interaction_detector.py
- Test distance calculations with known positions
- Verify temporal filtering with synthetic data
- Test orientation-based interaction detection
```

### Integration Testing:

```python
# Create controlled test scenarios:
1. Two people walking toward each other
2. Two people having conversation (facing each other)
3. Multiple people moving randomly (no interactions)
4. One person temporarily occluded by another
5. People sitting at desks vs standing

# Validation criteria for each scenario:
- Person detection accuracy > 90%
- ID consistency > 80% over 60 seconds
- Interaction detection matches manual observation
```

### System Performance Benchmarks:

```python
# Minimum acceptable performance:
- Frame processing rate: > 10 FPS
- Person detection latency: < 100ms per frame
- Memory usage: < 2GB RAM
- CPU usage: < 70% on development machine

# Target performance goals:
- Frame processing rate: > 20 FPS
- Person detection latency: < 50ms per frame
- Memory usage: < 1GB RAM
- CPU usage: < 50% on development machine
```

## Quick Start Implementation Guide

### Prerequisites Checklist:

- [ ] Intel RealSense D455 camera connected via USB 3.0
- [ ] Python 3.8+ installed
- [ ] All required packages installed (see dependency list above)
- [ ] Camera mounted at specified position (corner, 2.5m height, 30° tilt)
- [ ] Test room with known dimensions for calibration

### First Steps (Day 1):

1. **Clone/create project structure** using the modular layout above
2. **Test camera connection** with basic RealSense viewer
3. **Implement basic frame capture** (save 100 test frames)
4. **Manual verification**: Count people in saved frames vs system detection

### Development Workflow:

1. **Write component**: Implement one module at a time
2. **Unit test**: Test with synthetic or saved data
3. **Integration test**: Test with live camera feed
4. **Validation**: Compare against manual observation
5. **Iterate**: Refine based on test results

### Debug Output Requirements:

Every module should output debug information:

```python
# Example debug output format:
[TIMESTAMP] [MODULE] [LEVEL] Message
[12:34:56] [PersonDetector] [INFO] Detected 3 people in frame
[12:34:56] [Tracker] [DEBUG] Person ID 1 moved 0.3m since last frame
[12:34:56] [InteractionDetector] [INFO] Interaction detected: ID 1 <-> ID 2
[12:34:56] [System] [WARNING] Person ID 3 lost for 5 consecutive frames
```

### Success Validation at Each Phase:

- **Phase 1**: System detects and tracks 2-3 people for 60 seconds
- **Phase 2**: System detects obvious interactions (people talking face-to-face)
- **Phase 3**: System generates meaningful social network graphs
- **Phase 4**: System accuracy validated against human observers

This implementation guide now provides everything needed for an AI to start building the system: detailed architecture, specific parameters, step-by-step instructions, common problems and solutions, testing procedures, and validation criteria.

## Success Criteria (Tiered Approach)

### Minimum Viable System:

- **Basic operation** at 10+ FPS with 5-10 people
- **Accuracy ≥ 70%** for interaction detection in controlled scenarios
- **Privacy compliance** with complete data anonymization
- **Stable tracking** for 80%+ of session duration
- **Basic network metrics** generation

### Target Performance:

- **Real-time operation** at 15+ FPS
- **Accuracy ≥ 80%** for interaction detection
- **Robust tracking** handling moderate occlusions
- **Comprehensive social insights** validated by instructor feedback

### Stretch Goals:

- **High performance** at 30 FPS
- **Accuracy ≥ 90%** approaching human observer level
- **Advanced analytics** with temporal pattern recognition
- **Automated parameter tuning** based on classroom context

## Potential Extensions & Improvements

### If Time Permits:

1. **Multi-zone analysis**: Different interaction thresholds for different classroom areas
2. **Activity context integration**: Adapt detection based on class activity (lecture vs group work)
3. **Longitudinal analysis**: Track social network evolution over multiple sessions
4. **Teacher-student interaction detection**: Identify instructor-student interactions
5. **Group formation detection**: Identify formation and dissolution of student groups

### Future Work Recommendations:

- Multi-camera integration for larger spaces
- Integration with learning management systems
- Real-time feedback for instructors
- Mobile app for network visualization

## Deliverables

1. **Working prototype system** with all components integrated
2. **Performance evaluation** comparing to ground truth
3. **Social network analysis** of classroom interactions
4. **Technical documentation** of implementation
5. **Research report** following your proposed chapter structure
