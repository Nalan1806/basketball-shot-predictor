"""Headless smoke test for EKF tracker and Monte Carlo module.

Uses pixel coordinates within the default calibrated camera region
(640x700 = free-throw line, 640x240 = basket) so the homography
produces sensible metre values.
"""
import numpy as np
from ekf_tracker import EKFBallTracker
from shot_probability import MonteCarloProbability

ekf = EKFBallTracker()

# Simulate 15 detections along a parabolic arc (pixel space).
# Start near the calibrated free-throw region and arc upward toward basket.
# Basket is ~(640, 240) in default camera, free-throw line is ~(640, 700).
for i in range(15):
    t = i / 30.0
    # Move from x=580 to x=700, arc from y=650 up toward y=350
    px = 580 + i * 8
    py = 650 - 200 * t + 100 * t * t   # parabola opening down (upward arc)
    det = {
        'cx': int(px), 'cy': int(py),
        'x': int(px) - 10, 'y': int(py) - 10, 'w': 20, 'h': 20,
        'radius': 10, 'area': 314, 'circularity': 0.9,
    }
    ekf.update(det, confidence=0.85)

state = ekf.get_ekf_state()
print('EKF state:', {k: round(v, 3) for k, v in state.items() if k != 'P_pos'})

# --- Test 1: forward trajectory ---
future = ekf.predict_trajectory(n_steps=50)
assert len(future) > 0, "Expected some future points"
print(f'Future points: {len(future)} first={future[0]} last={future[-1]}')

# --- Test 2: uncertainty ellipse ---
ell = ekf.get_uncertainty_ellipse(min(10, len(future) - 1))
print('Ellipse step 10:', ell)

# --- Test 3: Monte Carlo ---
mc = MonteCarloProbability(n_simulations=100)
res = mc.estimate(ekf)
assert 0.0 <= res["Pr"] <= 1.0, f"Pr={res['Pr']} out of bounds"
print('MC result:', {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})

# --- Test 4: State values are finite ---
s = ekf.get_ekf_state()
for key in ('px_m', 'py_m', 'vx_ms', 'vy_ms', 'beta', 'speed_ms'):
    assert np.isfinite(s[key]), f"State value {key}={s[key]} is not finite"

# --- Test 5: reset works ---
ekf.reset()
assert not ekf.initialized
assert len(ekf.get_positions()) == 0
print('Reset OK')

print('\nAll tests PASSED')
