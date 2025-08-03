"""
Proximity Analyzer - Phase 2 Step 1
Calculates distances between tracked people and identifies proximity events
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from collections import defaultdict, deque

@dataclass
class ProximityEvent:
    """Represents a proximity event between two people."""
    person1_id: int
    person2_id: int
    distance: float  # meters
    timestamp: float
    duration: float  # seconds (0 for new events)
    zone_type: str  # 'intimate', 'personal', 'social', 'public'
    confidence: float  # based on zone quality of both people

class ProximityAnalyzer:
    """Analyzes proximity between tracked people based on Hall's proxemic zones."""
    
    def __init__(self, config: dict):
        """Initialize proximity analyzer with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Proxemic zones based on Hall's theory (meters)
        self.proxemic_zones = config.get('proxemic_zones', {
            'intimate': (0.0, 0.45),    # 0-45cm: very close personal space
            'personal': (0.45, 1.2),    # 45cm-1.2m: close friends, family
            'social': (1.2, 3.6),       # 1.2-3.6m: acquaintances, colleagues
            'public': (3.6, 7.5),       # 3.6-7.5m: formal interactions
        })
        
        # Analysis parameters
        self.min_duration = config.get('min_proximity_duration', 2.0)  # seconds
        self.update_interval = config.get('proximity_update_interval', 0.1)  # seconds
        self.max_tracking_distance = config.get('max_tracking_distance', 10.0)  # meters
        
        # State tracking
        self.active_proximities: Dict[Tuple[int, int], ProximityEvent] = {}
        self.proximity_history: Dict[Tuple[int, int], deque] = defaultdict(lambda: deque(maxlen=100))
        self.last_update_time = 0.0
        
        # Statistics
        self.total_proximity_events = 0
        self.zone_counts = defaultdict(int)
        
    def analyze_frame(self, tracked_people: List, timestamp: float) -> List[ProximityEvent]:
        """Analyze proximity for current frame and return active proximity events."""
        # Skip if not enough time has passed
        if timestamp - self.last_update_time < self.update_interval:
            return list(self.active_proximities.values())
        
        self.last_update_time = timestamp
        
        # Calculate all pairwise distances
        current_distances = self._calculate_pairwise_distances(tracked_people, timestamp)
        
        # Update active proximities
        self._update_proximity_events(current_distances, timestamp)
        
        # Clean up expired events
        self._cleanup_expired_events(timestamp)
        
        # Return current active proximities
        active_events = list(self.active_proximities.values())
        
        # Log statistics periodically
        if len(active_events) > 0:
            self.logger.debug(f"Active proximities: {len(active_events)}")
        
        return active_events
    
    def _calculate_pairwise_distances(self, tracked_people: List, timestamp: float) -> Dict[Tuple[int, int], Dict]:
        """Calculate distances between all pairs of tracked people."""
        distances = {}
        
        # Only analyze people with valid positions
        valid_people = [p for p in tracked_people if p.current_detection is not None]
        
        if len(valid_people) < 2:
            return distances
        
        for i, person1 in enumerate(valid_people):
            for j, person2 in enumerate(valid_people[i+1:], i+1):
                # Get 3D positions
                pos1 = person1.get_latest_position()
                pos2 = person2.get_latest_position()
                
                # Skip if either position is invalid
                if pos1 == (0, 0, 0) or pos2 == (0, 0, 0):
                    continue
                
                # Calculate 3D Euclidean distance
                distance_3d = np.sqrt(
                    (pos1[0] - pos2[0])**2 + 
                    (pos1[1] - pos2[1])**2 + 
                    (pos1[2] - pos2[2])**2
                )
                
                # Calculate 2D distance (floor plane) for proxemics
                distance_2d = np.sqrt(
                    (pos1[0] - pos2[0])**2 + 
                    (pos1[1] - pos2[1])**2
                )
                
                # Skip if too far away
                if distance_2d > self.max_tracking_distance:
                    continue
                
                # Determine proxemic zone
                zone_type = self._classify_proxemic_zone(distance_2d)
                
                # Calculate confidence based on detection zones
                confidence = self._calculate_proximity_confidence(person1, person2, distance_2d)
                
                # Create distance info
                pair_key = tuple(sorted([person1.id, person2.id]))
                distances[pair_key] = {
                    'person1_id': person1.id,
                    'person2_id': person2.id,
                    'distance_3d': distance_3d,
                    'distance_2d': distance_2d,
                    'zone_type': zone_type,
                    'confidence': confidence,
                    'position1': pos1,
                    'position2': pos2,
                    'timestamp': timestamp
                }
        
        return distances
    
    def _classify_proxemic_zone(self, distance: float) -> str:
        """Classify distance into proxemic zone."""
        for zone_name, (min_dist, max_dist) in self.proxemic_zones.items():
            if min_dist <= distance < max_dist:
                return zone_name
        return 'distant'  # Beyond public zone
    
    def _calculate_proximity_confidence(self, person1, person2, distance: float) -> float:
        """Calculate confidence score for proximity measurement."""
        # Base confidence from person detection zones
        conf1 = person1.get_average_confidence() if hasattr(person1, 'get_average_confidence') else 0.8
        conf2 = person2.get_average_confidence() if hasattr(person2, 'get_average_confidence') else 0.8
        
        # Distance-based confidence (closer = higher confidence in measurement)
        distance_conf = max(0.3, 1.0 - (distance / 10.0))  # Confidence decreases with distance
        
        # Stability confidence (more stable tracks = higher confidence)
        stability1 = 1.0 if person1.is_stable() else 0.7
        stability2 = 1.0 if person2.is_stable() else 0.7
        
        # Combined confidence
        overall_confidence = (conf1 + conf2) / 2.0 * distance_conf * (stability1 + stability2) / 2.0
        
        return min(1.0, max(0.1, overall_confidence))
    
    def _update_proximity_events(self, current_distances: Dict, timestamp: float):
        """Update active proximity events based on current distances."""
        current_pairs = set(current_distances.keys())
        active_pairs = set(self.active_proximities.keys())
        
        # Update existing proximities
        for pair_key in active_pairs.intersection(current_pairs):
            distance_info = current_distances[pair_key]
            existing_event = self.active_proximities[pair_key]
            
            # Update existing event
            existing_event.distance = distance_info['distance_2d']
            existing_event.timestamp = timestamp
            existing_event.duration = timestamp - existing_event.timestamp + existing_event.duration
            existing_event.zone_type = distance_info['zone_type']
            existing_event.confidence = distance_info['confidence']
            
            # Add to history
            self.proximity_history[pair_key].append({
                'timestamp': timestamp,
                'distance': distance_info['distance_2d'],
                'zone': distance_info['zone_type'],
                'confidence': distance_info['confidence']
            })
        
        # Create new proximity events
        for pair_key in current_pairs - active_pairs:
            distance_info = current_distances[pair_key]
            
            # Only create events for close proximities (not public/distant)
            if distance_info['zone_type'] in ['intimate', 'personal', 'social']:
                new_event = ProximityEvent(
                    person1_id=distance_info['person1_id'],
                    person2_id=distance_info['person2_id'],
                    distance=distance_info['distance_2d'],
                    timestamp=timestamp,
                    duration=0.0,
                    zone_type=distance_info['zone_type'],
                    confidence=distance_info['confidence']
                )
                
                self.active_proximities[pair_key] = new_event
                self.total_proximity_events += 1
                self.zone_counts[distance_info['zone_type']] += 1
                
                self.logger.info(f"New proximity: Person {distance_info['person1_id']} <-> "
                               f"Person {distance_info['person2_id']} at {distance_info['distance_2d']:.2f}m "
                               f"({distance_info['zone_type']} zone)")
        
        # Remove proximities that are no longer active
        for pair_key in active_pairs - current_pairs:
            removed_event = self.active_proximities.pop(pair_key)
            self.logger.info(f"Proximity ended: Person {removed_event.person1_id} <-> "
                           f"Person {removed_event.person2_id} after {removed_event.duration:.1f}s")
    
    def _cleanup_expired_events(self, timestamp: float):
        """Remove events that haven't been updated recently."""
        expired_keys = []
        max_gap = 5.0  # seconds
        
        for pair_key, event in self.active_proximities.items():
            if timestamp - event.timestamp > max_gap:
                expired_keys.append(pair_key)
        
        for pair_key in expired_keys:
            removed_event = self.active_proximities.pop(pair_key)
            self.logger.info(f"Proximity expired: Person {removed_event.person1_id} <-> "
                           f"Person {removed_event.person2_id}")
    
    def get_proximity_statistics(self) -> Dict:
        """Get statistics about proximity events."""
        active_events = list(self.active_proximities.values())
        
        return {
            'active_proximities': len(active_events),
            'total_events_created': self.total_proximity_events,
            'zone_distribution': dict(self.zone_counts),
            'average_distance': np.mean([e.distance for e in active_events]) if active_events else 0.0,
            'average_duration': np.mean([e.duration for e in active_events]) if active_events else 0.0,
            'confidence_scores': [e.confidence for e in active_events]
        }
    
    def get_validated_proximities(self, min_duration: Optional[float] = None) -> List[ProximityEvent]:
        """Get proximity events that meet duration requirements."""
        if min_duration is None:
            min_duration = self.min_duration
        
        return [event for event in self.active_proximities.values() 
                if event.duration >= min_duration]
    
    def get_proximity_history(self, person1_id: int, person2_id: int) -> List[Dict]:
        """Get proximity history for a specific pair."""
        pair_key = tuple(sorted([person1_id, person2_id]))
        return list(self.proximity_history.get(pair_key, []))
    
    def reset(self):
        """Reset all proximity tracking state."""
        self.active_proximities.clear()
        self.proximity_history.clear()
        self.total_proximity_events = 0
        self.zone_counts.clear()
        self.last_update_time = 0.0
        self.logger.info("Proximity analyzer reset")