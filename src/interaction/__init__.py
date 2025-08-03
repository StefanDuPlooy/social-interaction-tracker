# src/interaction/__init__.py
"""
Interaction module for proximity analysis and interaction detection
Phase 2 implementation: proximity, orientation, and interaction inference
"""

from .proximity_analyzer import ProximityAnalyzer, ProximityEvent

__all__ = ['ProximityAnalyzer', 'ProximityEvent']