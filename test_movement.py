#!/usr/bin/env python3
"""
Movement Test Helper - Use during Phase 2 Step 2 testing
Reminds you to move around for movement-based orientation detection
"""

import time
import sys

def movement_reminder():
    """Provide movement guidance during testing."""
    print("MOVEMENT TEST HELPER")
    print("=" * 30)
    print("For movement-based orientation to work:")
    print()
    
    movements = [
        "Walk forward slowly",
        "Walk backward",
        "Step left and right", 
        "Turn in a circle",
        "Walk diagonally",
        "Face different directions while moving"
    ]
    
    for i, movement in enumerate(movements, 1):
        print(f"{i}. {movement}")
        print("   (Move for 5-10 seconds)")
        input("   Press Enter when done...")
        print()
    
    print("Movement test complete!")
    print("Now run: python run_phase2_step2.py")

if __name__ == "__main__":
    movement_reminder()
