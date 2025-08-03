"""
Unit Tests for Tracking System
Phase 1 Step 3 - Test ID assignment, trajectory smoothing, and association
"""

import sys
import os
import numpy as np
import time
import unittest
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from detection.person_detector import PersonDetection
from detection.tracker import PersonTracker, TrackedPerson
from config.tracking_config import TRACKING_PARAMETERS

class TestTrackedPerson(unittest.TestCase):
    """Test TrackedPerson functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracked_person = TrackedPerson(id=1)
        
        # Create sample detection
        self.sample_detection = PersonDetection(
            id=1,
            timestamp=time.time(),
            position_3d=(1.0, 2.0, 3.0),
            bounding_box=(100, 100, 50, 100),
            confidence=0.8,
            detection_method='yolo+depth',
            zone='high',
            blob_area=5000,
            depth_meters=3.0
        )
    
    def test_initialization(self):
        """Test TrackedPerson initialization."""
        self.assertEqual(self.tracked_person.id, 1)
        self.assertEqual(self.tracked_person.frames_since_seen, 0)
        self.assertEqual(self.tracked_person.total_frames_seen, 0)
        self.assertIsNotNone(self.tracked_person.kalman_filter)
    
    def test_update_with_detection(self):
        """Test updating tracked person with detection."""
        self.tracked_person.update(self.sample_detection)
        
        self.assertEqual(self.tracked_person.current_detection, self.sample_detection)
        self.assertEqual(self.tracked_person.total_frames_seen, 1)
        self.assertEqual(self.tracked_person.frames_since_seen, 0)
        self.assertEqual(len(self.tracked_person.position_history), 1)
        self.assertEqual(self.tracked_person.position_history[0], (1.0, 2.0, 3.0))
    
    def test_trajectory_smoothing(self):
        """Test trajectory smoothing with multiple positions."""
        positions = [
            (1.0, 1.0, 3.0),
            (1.1, 1.1, 3.0),
            (1.2, 1.2, 3.0),
            (1.3, 1.3, 3.0),
            (1.4, 1.4, 3.0),
        ]
        
        for i, pos in enumerate(positions):
            detection = PersonDetection(
                id=1, timestamp=time.time() + i * 0.1, position_3d=pos,
                bounding_box=(100, 100, 50, 100), confidence=0.8,
                detection_method='yolo+depth', zone='high',
                blob_area=5000, depth_meters=pos[2]
            )
            self.tracked_person.update(detection)
        
        # Check smoothed position is reasonable
        smoothed = self.tracked_person.smoothed_position
        self.assertIsNotNone(smoothed)
        
        # Should be somewhere in the middle of the trajectory
        self.assertGreater(smoothed[0], 1.0)
        self.assertLess(smoothed[0], 1.4)
    
    def test_stability_detection(self):
        """Test stability detection."""
        # Initially unstable (not enough history)
        self.assertFalse(self.tracked_person.is_stable())
        
        # Add enough detections with low variance
        for i in range(10):
            pos = (1.0 + i * 0.01, 1.0 + i * 0.01, 3.0)  # Small movements
            detection = PersonDetection(
                id=1, timestamp=time.time() + i * 0.1, position_3d=pos,
                bounding_box=(100, 100, 50, 100), confidence=0.8,
                detection_method='yolo+depth', zone='high',
                blob_area=5000, depth_meters=pos[2]
            )
            self.tracked_person.update(detection)
        
        # Should now be stable
        self.assertTrue(self.tracked_person.is_stable())
    
    def test_prediction(self):
        """Test position prediction."""
        # Add some detections to establish movement
        for i in range(3):
            pos = (1.0 + i * 0.1, 1.0, 3.0)
            detection = PersonDetection(
                id=1, timestamp=time.time() + i * 0.1, position_3d=pos,
                bounding_box=(100, 100, 50, 100), confidence=0.8,
                detection_method='yolo+depth', zone='high',
                blob_area=5000, depth_meters=pos[2]
            )
            self.tracked_person.update(detection)
        
        # Test prediction
        predicted = self.tracked_person.predict_next_position()
        self.assertIsNotNone(predicted)
        self.assertEqual(len(predicted), 3)  # x, y, z
        
        # Prediction should be reasonable (forward movement)
        self.assertGreater(predicted[0], 1.2)  # Should move forward

class TestPersonTracker(unittest.TestCase):
    """Test PersonTracker functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TRACKING_PARAMETERS.copy()
        self.tracker = PersonTracker(self.config)
    
    def create_detection(self, pos: tuple, confidence: float = 0.8, 
                        detection_id: int = -1) -> PersonDetection:
        """Helper to create test detections."""
        return PersonDetection(
            id=detection_id,
            timestamp=time.time(),
            position_3d=pos,
            bounding_box=(100, 100, 50, 100),
            confidence=confidence,
            detection_method='yolo+depth',
            zone='high',
            blob_area=5000,
            depth_meters=pos[2]
        )
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(len(self.tracker.tracked_people), 0)
        self.assertEqual(self.tracker.next_id, 1)
        self.assertEqual(self.tracker.total_tracks_created, 0)
    
    def test_single_person_tracking(self):
        """Test tracking a single person."""
        # First detection - should create new track
        detection1 = self.create_detection((1.0, 1.0, 3.0))
        tracked = self.tracker.update([detection1], time.time())
        
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].id, 1)
        self.assertEqual(self.tracker.total_tracks_created, 1)
        
        # Second detection nearby - should associate with existing track
        detection2 = self.create_detection((1.1, 1.1, 3.0))
        tracked = self.tracker.update([detection2], time.time() + 0.1)
        
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].id, 1)  # Same ID
        self.assertEqual(self.tracker.total_tracks_created, 1)  # No new track
    
    def test_multiple_people_tracking(self):
        """Test tracking multiple people simultaneously."""
        # Two detections far apart - should create two tracks
        detection1 = self.create_detection((1.0, 1.0, 3.0))
        detection2 = self.create_detection((5.0, 5.0, 4.0))
        
        tracked = self.tracker.update([detection1, detection2], time.time())
        
        self.assertEqual(len(tracked), 2)
        self.assertEqual(self.tracker.total_tracks_created, 2)
        
        # Check IDs are different
        ids = [p.id for p in tracked]
        self.assertEqual(len(set(ids)), 2)  # Two unique IDs
    
    def test_track_association(self):
        """Test correct association of detections to tracks."""
        # Create initial tracks
        detection1 = self.create_detection((1.0, 1.0, 3.0))
        detection2 = self.create_detection((5.0, 5.0, 4.0))
        tracked = self.tracker.update([detection1, detection2], time.time())
        
        # Get the IDs
        id1 = tracked[0].id if tracked[0].get_latest_position()[0] < 3.0 else tracked[1].id
        id2 = tracked[1].id if tracked[1].get_latest_position()[0] > 3.0 else tracked[0].id
        
        # Next frame - move both people
        detection1_moved = self.create_detection((1.2, 1.2, 3.0))
        detection2_moved = self.create_detection((5.2, 5.2, 4.0))
        
        tracked = self.tracker.update([detection1_moved, detection2_moved], time.time() + 0.1)
        
        # Should still have 2 tracks with same IDs
        self.assertEqual(len(tracked), 2)
        current_ids = [p.id for p in tracked]
        self.assertIn(id1, current_ids)
        self.assertIn(id2, current_ids)
    
    def test_track_loss_and_recovery(self):
        """Test track loss when person disappears."""
        # Create track
        detection = self.create_detection((1.0, 1.0, 3.0))
        tracked = self.tracker.update([detection], time.time())
        track_id = tracked[0].id
        
        # Person disappears for several frames
        for i in range(self.config['max_disappeared_frames'] + 5):
            tracked = self.tracker.update([], time.time() + i * 0.1)
        
        # Track should be removed
        self.assertEqual(len(tracked), 0)
        self.assertEqual(self.tracker.total_tracks_lost, 1)
        
        # Person reappears - should get new ID
        detection_return = self.create_detection((1.0, 1.0, 3.0))
        tracked = self.tracker.update([detection_return], time.time() + 10)
        
        self.assertEqual(len(tracked), 1)
        self.assertNotEqual(tracked[0].id, track_id)  # New ID
    
    def test_distance_threshold(self):
        """Test distance threshold for association."""
        # Create track
        detection = self.create_detection((1.0, 1.0, 3.0))
        tracked = self.tracker.update([detection], time.time())
        original_id = tracked[0].id
        
        # Detection very far away - should create new track
        far_detection = self.create_detection((10.0, 10.0, 5.0))
        tracked = self.tracker.update([far_detection], time.time() + 0.1)
        
        # Should have 2 tracks now (original missing, new one created)
        self.assertEqual(self.tracker.total_tracks_created, 2)
    
    def test_confidence_filtering(self):
        """Test that low confidence detections don't create tracks."""
        # Low confidence detection
        low_conf_detection = self.create_detection((1.0, 1.0, 3.0), confidence=0.3)
        tracked = self.tracker.update([low_conf_detection], time.time())
        
        # Should not create track due to low confidence
        self.assertEqual(len(tracked), 0)
        self.assertEqual(self.tracker.total_tracks_created, 0)
        
        # High confidence detection
        high_conf_detection = self.create_detection((1.0, 1.0, 3.0), confidence=0.8)
        tracked = self.tracker.update([high_conf_detection], time.time())
        
        # Should create track
        self.assertEqual(len(tracked), 1)
        self.assertEqual(self.tracker.total_tracks_created, 1)

class TestTrackingIntegration(unittest.TestCase):
    """Integration tests for the complete tracking system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = TRACKING_PARAMETERS.copy()
        self.tracker = PersonTracker(self.config)
    
    def test_realistic_scenario(self):
        """Test a realistic classroom scenario."""
        # Simulate a person walking across the room
        path = [
            (1.0, 1.0, 3.0),
            (1.5, 1.2, 3.1),
            (2.0, 1.4, 3.2),
            (2.5, 1.6, 3.3),
            (3.0, 1.8, 3.4),
        ]
        
        track_ids = []
        
        for i, pos in enumerate(path):
            detection = PersonDetection(
                id=-1, timestamp=time.time() + i * 0.1, position_3d=pos,
                bounding_box=(100 + i * 10, 100, 50, 100), confidence=0.8,
                detection_method='yolo+depth', zone='high',
                blob_area=5000, depth_meters=pos[2]
            )
            
            tracked = self.tracker.update([detection], time.time() + i * 0.1)
            
            if tracked:
                track_ids.append(tracked[0].id)
        
        # Should maintain same ID throughout
        self.assertTrue(len(set(track_ids)) == 1, f"ID consistency failed: {track_ids}")
        
        # Track should be stable by the end
        final_track = self.tracker.get_all_tracks()[0]
        self.assertTrue(final_track.is_stable())
    
    def test_occlusion_handling(self):
        """Test handling of temporary occlusions."""
        # Person visible
        detection1 = PersonDetection(
            id=-1, timestamp=time.time(), position_3d=(2.0, 2.0, 3.0),
            bounding_box=(200, 200, 50, 100), confidence=0.8,
            detection_method='yolo+depth', zone='high',
            blob_area=5000, depth_meters=3.0
        )
        
        tracked = self.tracker.update([detection1], time.time())
        original_id = tracked[0].id
        
        # Person occluded for a few frames
        for i in range(5):
            tracked = self.tracker.update([], time.time() + (i + 1) * 0.1)
        
        # Person reappears nearby
        detection2 = PersonDetection(
            id=-1, timestamp=time.time() + 0.6, position_3d=(2.2, 2.1, 3.0),
            bounding_box=(210, 200, 50, 100), confidence=0.8,
            detection_method='yolo+depth', zone='high',
            blob_area=5000, depth_meters=3.0
        )
        
        tracked = self.tracker.update([detection2], time.time() + 0.6)
        
        # Should recover the same track
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0].id, original_id)

def run_tracking_tests():
    """Run all tracking tests."""
    print("Running Phase 1 Step 3 Tracking Tests")
    print("=" * 40)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTrackedPerson))
    suite.addTests(loader.loadTestsFromTestCase(TestPersonTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestTrackingIntegration))
    
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
    run_tracking_tests()