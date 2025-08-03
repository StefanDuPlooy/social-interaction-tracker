"""
Unit Tests for Proximity Analyzer
Phase 2 Step 1 - Tests proximity calculation and zone detection
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

from interaction.proximity_analyzer import ProximityAnalyzer, ProximityEvent
from detection.person_detector import PersonDetection
from detection.tracker import TrackedPerson
from config.proximity_config import CLASSROOM_PROXIMITY

class MockTrackedPerson:
    """Mock tracked person for testing."""
    
    def __init__(self, person_id: int, position: tuple, confidence: float = 0.8):
        self.id = person_id
        self.position_3d = position
        self.confidence = confidence
        self.current_detection = MockDetection(position, confidence)
        
    def get_latest_position(self):
        return self.position_3d
    
    def get_average_confidence(self):
        return self.confidence
    
    def is_stable(self):
        return True

class MockDetection:
    """Mock detection for testing."""
    
    def __init__(self, position: tuple, confidence: float):
        self.position_3d = position
        self.confidence = confidence
        self.bounding_box = (100, 100, 50, 100)  # x, y, w, h

class TestProximityAnalyzer(unittest.TestCase):
    """Test ProximityAnalyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CLASSROOM_PROXIMITY.copy()
        self.analyzer = ProximityAnalyzer(self.config)
    
    def test_initialization(self):
        """Test ProximityAnalyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(len(self.analyzer.active_proximities), 0)
        self.assertEqual(self.analyzer.total_proximity_events, 0)
        
        # Check proxemic zones are loaded
        self.assertIn('intimate', self.analyzer.proxemic_zones)
        self.assertIn('personal', self.analyzer.proxemic_zones)
        self.assertIn('social', self.analyzer.proxemic_zones)
    
    def test_proxemic_zone_classification(self):
        """Test classification of distances into proxemic zones."""
        # Test intimate zone (0-0.3m)
        self.assertEqual(self.analyzer._classify_proxemic_zone(0.2), 'intimate')
        
        # Test personal zone (0.3-1.0m)
        self.assertEqual(self.analyzer._classify_proxemic_zone(0.5), 'personal')
        
        # Test social zone (1.0-2.5m)
        self.assertEqual(self.analyzer._classify_proxemic_zone(1.5), 'social')
        
        # Test public zone (2.5-5.0m)
        self.assertEqual(self.analyzer._classify_proxemic_zone(3.0), 'public')
        
        # Test distant (beyond zones)
        self.assertEqual(self.analyzer._classify_proxemic_zone(6.0), 'distant')
    
    def test_distance_calculation(self):
        """Test pairwise distance calculation."""
        # Create two people at known positions
        person1 = MockTrackedPerson(1, (0.0, 0.0, 1.5))  # Origin
        person2 = MockTrackedPerson(2, (1.0, 0.0, 1.5))  # 1 meter away in X
        
        tracked_people = [person1, person2]
        timestamp = time.time()
        
        distances = self.analyzer._calculate_pairwise_distances(tracked_people, timestamp)
        
        # Should have one distance measurement
        self.assertEqual(len(distances), 1)
        
        # Check the distance calculation
        pair_key = (1, 2)  # Sorted IDs
        self.assertIn(pair_key, distances)
        
        distance_info = distances[pair_key]
        self.assertAlmostEqual(distance_info['distance_2d'], 1.0, places=2)
        self.assertEqual(distance_info['zone_type'], 'personal')
    
    def test_proximity_event_creation(self):
        """Test creation of proximity events."""
        # Create two people close together
        person1 = MockTrackedPerson(1, (0.0, 0.0, 1.5))
        person2 = MockTrackedPerson(2, (0.5, 0.0, 1.5))  # 0.5m away (personal zone)
        
        tracked_people = [person1, person2]
        timestamp = time.time()
        
        # Analyze frame
        events = self.analyzer.analyze_frame(tracked_people, timestamp)
        
        # Should create one proximity event
        self.assertEqual(len(events), 1)
        self.assertEqual(len(self.analyzer.active_proximities), 1)
        
        event = events[0]
        self.assertIn(event.person1_id, [1, 2])
        self.assertIn(event.person2_id, [1, 2])
        self.assertEqual(event.zone_type, 'personal')
        self.assertAlmostEqual(event.distance, 0.5, places=2)
    
    def test_proximity_duration_tracking(self):
        """Test tracking of proximity duration over time."""
        person1 = MockTrackedPerson(1, (0.0, 0.0, 1.5))
        person2 = MockTrackedPerson(2, (0.8, 0.0, 1.5))  # Personal zone
        
        tracked_people = [person1, person2]
        
        # First frame
        timestamp1 = time.time()
        events1 = self.analyzer.analyze_frame(tracked_people, timestamp1)
        self.assertEqual(len(events1), 1)
        self.assertEqual(events1[0].duration, 0.0)  # New event
        
        # Second frame (1 second later)
        timestamp2 = timestamp1 + 1.0
        self.analyzer.last_update_time = timestamp1  # Force update
        events2 = self.analyzer.analyze_frame(tracked_people, timestamp2)
        
        self.assertEqual(len(events2), 1)
        self.assertGreater(events2[0].duration, 0.9)  # Should be ~1 second
    
    def test_multiple_people_proximities(self):
        """Test proximity analysis with multiple people."""
        # Create three people in different configurations
        person1 = MockTrackedPerson(1, (0.0, 0.0, 1.5))
        person2 = MockTrackedPerson(2, (0.6, 0.0, 1.5))  # Close to person1
        person3 = MockTrackedPerson(3, (5.0, 0.0, 1.5))  # Far from others
        
        tracked_people = [person1, person2, person3]
        timestamp = time.time()
        
        events = self.analyzer.analyze_frame(tracked_people, timestamp)
        
        # Should have one proximity (person1-person2), person3 is too far
        self.assertEqual(len(events), 1)
        
        # Verify the correct pair is detected
        event = events[0]
        pair_ids = {event.person1_id, event.person2_id}
        self.assertEqual(pair_ids, {1, 2})
    
    def test_proximity_zone_filtering(self):
        """Test that only relevant proximity zones create events."""
        # Test distant people (should not create events)
        person1 = MockTrackedPerson(1, (0.0, 0.0, 1.5))
        person2 = MockTrackedPerson(2, (8.0, 0.0, 1.5))  # Very far (distant zone)
        
        tracked_people = [person1, person2]
        timestamp = time.time()
        
        events = self.analyzer.analyze_frame(tracked_people, timestamp)
        
        # Should not create events for distant people
        self.assertEqual(len(events), 0)
    
    def test_proximity_expiration(self):
        """Test that proximity events expire when people move apart."""
        person1 = MockTrackedPerson(1, (0.0, 0.0, 1.5))
        person2 = MockTrackedPerson(2, (0.5, 0.0, 1.5))  # Close initially
        
        # Create proximity
        timestamp1 = time.time()
        events1 = self.analyzer.analyze_frame([person1, person2], timestamp1)
        self.assertEqual(len(events1), 1)
        
        # Move person2 far away
        person2.position_3d = (10.0, 0.0, 1.5)
        person2.current_detection.position_3d = (10.0, 0.0, 1.5)
        
        # Update analyzer
        timestamp2 = timestamp1 + 1.0
        self.analyzer.last_update_time = timestamp1
        events2 = self.analyzer.analyze_frame([person1, person2], timestamp2)
        
        # Proximity should be removed
        self.assertEqual(len(events2), 0)
        self.assertEqual(len(self.analyzer.active_proximities), 0)
    
    def test_statistics_calculation(self):
        """Test proximity statistics calculation."""
        # Create some proximity events
        person1 = MockTrackedPerson(1, (0.0, 0.0, 1.5))
        person2 = MockTrackedPerson(2, (0.5, 0.0, 1.5))  # Personal zone
        person3 = MockTrackedPerson(3, (1.8, 0.0, 1.5))  # Social zone with person1
        
        tracked_people = [person1, person2, person3]
        timestamp = time.time()
        
        events = self.analyzer.analyze_frame(tracked_people, timestamp)
        stats = self.analyzer.get_proximity_statistics()
        
        # Should have statistics
        self.assertGreater(stats['active_proximities'], 0)
        self.assertGreater(stats['total_events_created'], 0)
        self.assertIn('zone_distribution', stats)
        
        # Should have distance statistics if events exist
        if events:
            self.assertGreater(stats['average_distance'], 0)
    
    def test_validated_proximities(self):
        """Test filtering of proximities by duration."""
        person1 = MockTrackedPerson(1, (0.0, 0.0, 1.5))
        person2 = MockTrackedPerson(2, (0.5, 0.0, 1.5))
        
        timestamp = time.time()
        
        # Create proximity but don't let it mature
        events = self.analyzer.analyze_frame([person1, person2], timestamp)
        validated = self.analyzer.get_validated_proximities(min_duration=2.0)
        
        # Should not be validated yet (duration = 0)
        self.assertEqual(len(validated), 0)
        
        # Manually set duration for testing
        if events:
            events[0].duration = 3.0
            validated = self.analyzer.get_validated_proximities(min_duration=2.0)
            self.assertEqual(len(validated), 1)
    
    def test_confidence_calculation(self):
        """Test proximity confidence calculation."""
        # High confidence people
        person1 = MockTrackedPerson(1, (0.0, 0.0, 1.5), confidence=0.9)
        person2 = MockTrackedPerson(2, (0.5, 0.0, 1.5), confidence=0.8)
        
        confidence = self.analyzer._calculate_proximity_confidence(person1, person2, 0.5)
        
        # Should be reasonably high confidence
        self.assertGreater(confidence, 0.5)
        self.assertLessEqual(confidence, 1.0)
        
        # Low confidence people
        person3 = MockTrackedPerson(3, (0.0, 0.0, 1.5), confidence=0.3)
        person4 = MockTrackedPerson(4, (0.5, 0.0, 1.5), confidence=0.2)
        
        low_confidence = self.analyzer._calculate_proximity_confidence(person3, person4, 0.5)
        
        # Should be lower confidence
        self.assertLess(low_confidence, confidence)
    
    def test_reset_functionality(self):
        """Test resetting the proximity analyzer."""
        # Create some proximities
        person1 = MockTrackedPerson(1, (0.0, 0.0, 1.5))
        person2 = MockTrackedPerson(2, (0.5, 0.0, 1.5))
        
        events = self.analyzer.analyze_frame([person1, person2], time.time())
        self.assertGreater(len(events), 0)
        
        # Reset
        self.analyzer.reset()
        
        # Should be clean
        self.assertEqual(len(self.analyzer.active_proximities), 0)
        self.assertEqual(self.analyzer.total_proximity_events, 0)
        self.assertEqual(len(self.analyzer.proximity_history), 0)

def run_proximity_tests():
    """Run all proximity analyzer tests."""
    print("Running Phase 2 Step 1 Proximity Analyzer Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test class
    suite.addTests(loader.loadTestsFromTestCase(TestProximityAnalyzer))
    
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
    run_proximity_tests()