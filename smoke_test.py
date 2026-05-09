"""Headless smoke test for simplified ShotIQ tracker and predictor.

Tests the Kalman tracker and gravity-based trajectory predictor
without requiring a camera or display.
"""
import numpy as np
from tracker import BallTracker
from trajectory import TrajectoryPredictor

tracker = BallTracker()
predictor = TrajectoryPredictor()

# Simulate 15 detections along a parabolic arc (pixel space).
dt = 1.0 / 30.0
for i in range(15):
    t = i / 30.0
    px = 580 + i * 8
    py = 650 - 200 * t + 100 * t * t
    det = {
        'cx': int(px), 'cy': int(py),
        'x': int(px) - 10, 'y': int(py) - 10, 'w': 20, 'h': 20,
        'radius': 10,
    }
    tracker.update(det, confidence=0.85, dt=dt)

# --- Test 1: tracker state ---
state = tracker.get_state()
print('Tracker state:', {k: round(v, 2) if isinstance(v, float) else v for k, v in state.items()})
assert state["tracking"], "Should be tracking after 15 detections"

# --- Test 2: positions recorded ---
positions = tracker.get_positions()
assert len(positions) >= 10, f"Expected 10+ positions, got {len(positions)}"
print(f"Tracked {len(positions)} positions")

# --- Test 3: trajectory prediction ---
hoop_rect = (680, 250, 120, 80)
result = predictor.predict(positions, hoop_rect, dt=dt)
print(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.3f}")
print(f"Predicted points: {len(result['predicted_points'])}")
assert len(result['predicted_points']) > 0, "Expected predicted points"

# --- Test 4: velocity is sensible ---
speed = tracker.get_speed()
print(f"Speed: {speed:.1f} px/s")
assert speed > 0, "Speed should be positive after moving detections"

# --- Test 5: reset works ---
tracker.reset()
assert not tracker.is_tracking
assert len(tracker.get_positions()) == 0
print("Reset OK")

print("\nAll tests PASSED")
