"""
Unit Tests for Orientation Estimator
Phase 2 Step 2 - Tests orientation detection methods and mutual analysis
"""

import sys
import os
import numpy as np
import time
import unittest
import math
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from interaction.orientation_estimator import OrientationEstimator, OrientationEstimate, MutualOrientation
from detection.person_detector import PersonDetection
from detection.tracker import TrackedPerson
from config.orientation_config import ORIENTATION_METHODS, SKELETON_ORIENTATION, MUTUAL_ORIENTATION

class MockTrackedPerson:
    """Mock tracked person for testing orientation estimation."""
    
    def __init__(self, person_id: int, position: tuple, bbox: tuple = (100, 100, 50, 100)):
        self.id = person_id
        self.position_3d = position
        self.position_history = [position]
        self.current_detection = MockDetection(position, bbox)
        
    def get_latest_position(self):
        return self.position_3d
    
    def add_position(self, position):
        self.position_history.append(position)
        self.position_3d = position

class MockDetection:
    """Mock detection for testing."""
    
    def __init__(self, position: tuple, bbox: tuple):
        self.position_3d = position
        self.bounding_box = bbox  # (x, y, w, h)
        self.confidence = 0.8

class TestOrientationEstimator(unittest.TestCase):
    """Test OrientationEstimator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'orientation_methods': ORIENTATION_METHODS,
            'skeleton_orientation': SKELETON_ORIENTATION,
            'mutual_orientation': MUTUAL_ORIENTATION,
            'confidence_scoring': {
                'temporal_consistency_bonus': 0.2,
                'multi_method_agreement_bonus': 0.15
            },
            'error_handling': {
                'last_known_timeout': 60
            }
        }
        self.estimator = OrientationEstimator(self.config)
    
    def test_initialization(self):
        """Test OrientationEstimator initialization."""
        self.assertIsNotNone(self.estimator)
        self.assertEqual(len(self.estimator.orientation_history), 0)
        self.assertEqual(self.estimator.total_estimates, 0)
        
        # Check method configurations are loaded
        self.assertIn('skeleton_based', self.estimator.methods)
        self.assertIn('movement_based', self.estimator.methods)
        self.assertIn('depth_gradient', self.estimator.methods)
    
    def test_movement_based_orientation(self):
        """Test movement-based orientation detection."""
        person = MockTrackedPerson(1, (0.0, 0.0, 1.5))
        
        # Add movement history (moving east)
        positions = [
            (0.0, 0.0, 1.5),
            (0.1, 0.0, 1.5),
            (0.2, 0.0, 1.5),
            (0.3, 0.0, 1.5),
            (0.4, 0.0, 1.5),
        ]
        
        for pos in positions:
            person.add_position(pos)
        
        timestamp = time.time()
        orientation = self.estimator._movement_based_orientation(person, timestamp)
        
        self.assertIsNotNone(orientation)
        self.assertEqual(orientation.person_id, 1)
        self.assertEqual(orientation.method, 'movement')
        
        # Should be facing east (0 degrees or close)
        self.assertLess(abs(orientation.orientation_angle - 0), 15)  # Within 15 degrees
        self.assertGreater(orientation.confidence, 0.3)
        self.assertGreater(orientation.movement_magnitude, 0)
    
    def test_movement_based_orientation_insufficient_movement(self):
        """Test movement-based orientation with insufficient movement."""
        person = MockTrackedPerson(1, (0.0, 0.0, 1.5))
        
        # Add minimal movement history
        positions = [
            (0.0, 0.0, 1.5),
            (0.01, 0.0, 1.5),  # Very small movement
            (0.01, 0.01, 1.5),
        ]
        
        for pos in positions:
            person.add_position(pos)
        
        timestamp = time.time()
        orientation = self.estimator._movement_based_orientation(person, timestamp)
        
        # Should return None for insufficient movement
        self.assertIsNone(orientation)
    
    def test_depth_gradient_orientation(self):
        """Test depth gradient-based orientation detection."""
        person = MockTrackedPerson(1, (2.0, 2.0, 3.0), bbox=(50, 50, 100, 150))
        
        # Create a synthetic depth frame with gradient
        depth_frame = np.ones((480, 640), dtype=np.float32) * 3.0
        
        # Create depth gradient in person's bounding box
        x, y, w, h = person.current_detection.bounding_box
        
        # Make left side closer (lower depth) than right side
        for i in range(h):
            for j in range(w):
                if j < w // 2:
                    depth_frame[y + i, x + j] = 2.5  # Closer on left
                else:
                    depth_frame[y + i, x + j] = 3.5  # Farther on right
        
        timestamp = time.time()
        orientation = self.estimator._depth_gradient_orientation(person, depth_frame, timestamp)
        
        self.assertIsNotNone(orientation)
        self.assertEqual(orientation.person_id, 1)
        self.assertEqual(orientation.method, 'depth_gradient')
        self.assertGreater(orientation.confidence, 0.1)
        
        # Should detect orientation based on gradient
        self.assertIn(orientation.orientation_angle, [90, 270])  # Facing left or right
    
    def test_combine_orientation_estimates(self):
        """Test combining multiple orientation estimates."""
        person_id = 1
        timestamp = time.time()
        
        # Create multiple estimates pointing in similar directions
        estimates = [
            OrientationEstimate(
                person_id=person_id, timestamp=timestamp, orientation_angle=45.0,
                confidence=0.8, method='skeleton', facing_vector=(0.707, 0.707)
            ),
            OrientationEstimate(
                person_id=person_id, timestamp=timestamp, orientation_angle=50.0,
                confidence=0.6, method='movement', facing_vector=(0.643, 0.766)
            ),
            OrientationEstimate(
                person_id=person_id, timestamp=timestamp, orientation_angle=40.0,
                confidence=0.4, method='depth_gradient', facing_vector=(0.766, 0.643)
            )
        ]
        
        combined = self.estimator._combine_orientation_estimates(estimates, timestamp)
        
        self.assertEqual(combined.person_id, person_id)
        self.assertEqual(combined.method, 'combined')
        
        # Combined angle should be roughly in the middle
        self.assertGreater(combined.orientation_angle, 40)
        self.assertLess(combined.orientation_angle, 55)
        
        # Combined confidence should be higher due to agreement bonus
        self.assertGreater(combined.confidence, 0.6)
    
    def test_temporal_smoothing(self):
        """Test temporal smoothing of orientation estimates."""
        person_id = 1
        
        # First estimate
        first_estimate = OrientationEstimate(
            person_id=person_id, timestamp=1.0, orientation_angle=90.0,
            confidence=0.8, method='skeleton', facing_vector=(0, 1)
        )
        self.estimator.last_orientations[person_id] = first_estimate
        
        # Second estimate close in time and angle
        smoothed_angle, smoothed_confidence = self.estimator._apply_temporal_smoothing(
            person_id, 95.0, 0.7, 1.1
        )
        
        # Should be smoothed toward previous estimate
        self.assertLess(smoothed_angle, 95.0)
        self.assertGreater(smoothed_angle, 90.0)
        
        # Confidence should be boosted for consistency
        self.assertGreater(smoothed_confidence, 0.7)
    
    def test_calculate_mutual_orientation(self):
        """Test mutual orientation calculation between two people."""
        timestamp = time.time()
        
        # Two people facing each other
        orient1 = OrientationEstimate(
            person_id=1, timestamp=timestamp, orientation_angle=0.0,  # Facing east
            confidence=0.8, method='skeleton', facing_vector=(1, 0)
        )
        
        orient2 = OrientationEstimate(
            person_id=2, timestamp=timestamp, orientation_angle=180.0,  # Facing west
            confidence=0.7, method='skeleton', facing_vector=(-1, 0)
        )
        
        # Positions: person1 at origin, person2 to the east
        pos1 = (0.0, 0.0, 1.5)
        pos2 = (2.0, 0.0, 1.5)
        
        mutual = self.estimator._calculate_mutual_orientation(orient1, orient2, pos1, pos2)
        
        self.assertIsNotNone(mutual)
        self.assertEqual(mutual.person1_id, 1)
        self.assertEqual(mutual.person2_id, 2)
        
        # Should have high mutual facing score (facing each other)
        self.assertGreater(mutual.mutual_facing_score, 0.7)
        
        # Turn angles should be small (already facing each other)
        self.assertLess(mutual.person1_to_person2_angle, 15)
        self.assertLess(mutual.person2_to_person1_angle, 15)
    
    def test_calculate_turn_angle(self):
        """Test turn angle calculation."""
        # Test basic angles
        self.assertEqual(self.estimator._calculate_turn_angle(0, 90), 90)
        self.assertEqual(self.estimator._calculate_turn_angle(90, 0), 90)
        self.assertEqual(self.estimator._calculate_turn_angle(0, 180), 180)
        
        # Test wrap-around
        self.assertEqual(self.estimator._calculate_turn_angle(10, 350), 20)
        self.assertEqual(self.estimator._calculate_turn_angle(350, 10), 20)
        
        # Test same angle
        self.assertEqual(self.estimator._calculate_turn_angle(45, 45), 0)
    
    def test_f_formation_detection(self):
        """Test F-formation detection."""
        timestamp = time.time()
        
        # Two people facing toward a common center
        orient1 = OrientationEstimate(
            person_id=1, timestamp=timestamp, orientation_angle=45.0,  # Northeast
            confidence=0.8, method='skeleton', facing_vector=(0.707, 0.707)
        )
        
        orient2 = OrientationEstimate(
            person_id=2, timestamp=timestamp, orientation_angle=315.0,  # Northwest
            confidence=0.8, method='skeleton', facing_vector=(0.707, -0.707)
        )
        
        # Positions forming a potential F-formation
        pos1 = (0.0, 0.0, 1.5)
        pos2 = (1.0, 1.0, 1.5)
        
        mutual = self.estimator._calculate_mutual_orientation(orient1, orient2, pos1, pos2)
        
        self.assertIsNotNone(mutual)
        
        # Should detect F-formation
        self.assertTrue(mutual.in_f_formation)
        self.assertIsNotNone(mutual.o_space_center)
        self.assertGreater(mutual.group_coherence, 0.5)
    
    def test_fallback_orientation(self):
        """Test fallback orientation when all methods fail."""
        person = MockTrackedPerson(1, (0.0, 0.0, 1.5))
        timestamp = time.time()
        
        # Test with no previous orientation
        fallback = self.estimator._fallback_orientation(person, timestamp)
        
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback.person_id, 1)
        self.assertEqual(fallback.method, 'fallback_default')
        self.assertLessEqual(fallback.confidence, 0.2)
        
        # Test with previous orientation
        previous = OrientationEstimate(
            person_id=1, timestamp=timestamp - 5, orientation_angle=90.0,
            confidence=0.8, method='skeleton', facing_vector=(0, 1)
        )
        self.estimator.last_orientations[1] = previous
        
        fallback2 = self.estimator._fallback_orientation(person, timestamp)
        
        self.assertEqual(fallback2.method, 'fallback_last_known')
        self.assertEqual(fallback2.orientation_angle, 90.0)
        self.assertLess(fallback2.confidence, previous.confidence)
    
    def test_orientation_statistics(self):
        """Test orientation statistics collection."""
        # Add some mock estimates
        self.estimator.method_success_counts = {'skeleton': 10, 'movement': 5, 'depth_gradient': 2}
        self.estimator.total_estimates = 20
        
        # Add some orientation history
        person_id = 1
        self.estimator.orientation_history[person_id] = []
        for i in range(5):
            estimate = OrientationEstimate(
                person_id=person_id, timestamp=i, orientation_angle=i*10,
                confidence=0.8, method='skeleton', facing_vector=(1, 0)
            )
            self.estimator.orientation_history[person_id].append(estimate)
        
        stats = self.estimator.get_orientation_statistics()
        
        self.assertEqual(stats['total_estimates'], 20)
        self.assertEqual(stats['method_success_counts']['skeleton'], 10)
        self.assertEqual(stats['active_people_tracked'], 1)
        
        # Check success rates
        self.assertEqual(stats['method_success_rates']['skeleton'], 0.5)
        self.assertEqual(stats['method_success_rates']['movement'], 0.25)
    
    def test_reset_functionality(self):
        """Test resetting the orientation estimator."""
        # Add some data
        self.estimator.total_estimates = 10
        self.estimator.method_success_counts['skeleton'] = 5
        self.estimator.orientation_history[1] = [Mock()]
        self.estimator.last_orientations[1] = Mock()
        
        # Reset
        self.estimator.reset()
        
        # Should be clean
        self.assertEqual(self.estimator.total_estimates, 0)
        self.assertEqual(self.estimator.method_success_counts['skeleton'], 0)
        self.assertEqual(len(self.estimator.orientation_history), 0)
        self.assertEqual(len(self.estimator.last_orientations), 0)

class Mock:
    """Simple mock object."""
    pass

def run_orientation_tests():
    """Run all orientation estimator tests."""
    print("Running Phase 2 Step 2 Orientation Estimator Tests")
    print("=" * 55)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test class
    suite.addTests(loader.loadTestsFromTestCase(TestOrientationEstimator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, failure in result.failures:
            print(f"- {test}: {failure}")
    
    if result.errors:
        print("\nErrors:")
        for test, error in result.errors:
            print(f"- {test}: {error}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ ALL TESTS PASSED' if success else '❌ SOME TESTS FAILED'}")
    
    return success

if __name__ == "__main__":
    run_orientation_tests()