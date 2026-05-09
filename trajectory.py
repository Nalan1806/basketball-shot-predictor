"""
Trajectory Prediction — ShotIQ

Simple gravity-based trajectory prediction in pixel space.

Strategy:
  1. Take the last few tracked positions to estimate current velocity.
  2. Project forward using simple kinematics: x += vx*dt, y += vy*dt + 0.5*g*dt²
  3. Check if any projected point intersects the hoop rectangle.
  4. Return SCORE or MISS with a confidence score.

No Monte Carlo. No EKF forward projection. No world-space transforms.
Just clean, stable, gravity-based prediction.
"""

import numpy as np
import config


class TrajectoryPredictor:
    """
    Simple gravity-based trajectory predictor.

    Projects the ball's future path using pixel-space kinematics
    and checks for intersection with the hoop region.
    """

    def __init__(self):
        self.predicted_points = []
        self.prediction = None      # "ON TARGET" or "OFF TARGET" or None
        self.confidence = 0.0

    def predict(self, positions, hoop_rect, dt=1.0/30.0):
        """
        Predict shot outcome using gravity-based projection.

        Args:
            positions: list of (px, py) tracked positions (Kalman-filtered)
            hoop_rect: (x, y, w, h) hoop region in pixels, or None
            dt: current frame delta time in seconds

        Returns:
            dict:
              prediction       — "ON TARGET" | "OFF TARGET" | None
              confidence       — float 0–1
              predicted_points — list[(px, py)] future arc
        """
        self.predicted_points = []
        self.prediction = None
        self.confidence = 0.0

        if len(positions) < config.MIN_POINTS_FOR_PREDICTION:
            return self._result()

        # ── Estimate velocity from recent positions ──
        # Use last 5 positions for a smoothed velocity estimate
        n_vel = min(5, len(positions))
        recent = positions[-n_vel:]

        # Velocity as average displacement per frame (in pixels/frame)
        # Then convert to pixels/second using dt
        total_dx = recent[-1][0] - recent[0][0]
        total_dy = recent[-1][1] - recent[0][1]
        n_frames = max(n_vel - 1, 1)

        # Velocity in pixels per second
        vx = total_dx / (n_frames * dt) if dt > 0 else 0.0
        vy = total_dy / (n_frames * dt) if dt > 0 else 0.0

        # Current position (last filtered point)
        x0, y0 = positions[-1]

        # ── Forward projection with gravity ──
        gravity = config.GRAVITY_PX_PER_S2  # px/s² downward (positive Y = down)
        step_dt = dt  # project in frame-sized steps

        x, y = float(x0), float(y0)
        cur_vx, cur_vy = float(vx), float(vy)

        for _ in range(config.PREDICTION_STEPS):
            x += cur_vx * step_dt
            y += cur_vy * step_dt
            cur_vy += gravity * step_dt  # gravity accelerates downward

            # Stop if projected far out of frame
            if (x < -200 or x > config.FRAME_WIDTH + 200 or
                    y < -200 or y > config.FRAME_HEIGHT + 200):
                break

            self.predicted_points.append((int(x), int(y)))

        # ── Check hoop intersection ──
        if hoop_rect is not None and len(self.predicted_points) >= 2:
            self._check_intersection(hoop_rect, positions)

        return self._result()

    def _check_intersection(self, hoop_rect, observed_positions):
        """
        Check if predicted arc crosses through the hoop region.

        Uses two complementary checks:
          1. Minimum distance from any predicted point to hoop center
          2. Y-crossing check: does the trajectory cross the hoop Y level
             with X within the hoop bounds?
        """
        hx, hy, hw, hh = hoop_rect
        hoop_cx = hx + hw // 2
        hoop_cy = hy + hh // 2
        tolerance = config.INTERSECTION_TOLERANCE

        # Check 1: minimum distance to hoop center
        min_distance = float("inf")
        for pt in self.predicted_points:
            d = np.hypot(pt[0] - hoop_cx, pt[1] - hoop_cy)
            if d < min_distance:
                min_distance = d

        # Check 2: Y-crossing logic
        all_points = list(observed_positions[-5:]) + self.predicted_points
        crosses_hoop_y = False
        for i in range(1, len(all_points)):
            y_prev = all_points[i - 1][1]
            y_curr = all_points[i][1]
            # Check if trajectory crosses hoop Y level
            if (y_prev <= hy <= y_curr) or (y_prev >= hy >= y_curr):
                x_at_crossing = all_points[i][0]
                if hx - tolerance <= x_at_crossing <= hx + hw + tolerance:
                    crosses_hoop_y = True
                    break

        if min_distance < tolerance or crosses_hoop_y:
            self.prediction = "ON TARGET"
        else:
            self.prediction = "OFF TARGET"

        # ── Confidence calculation ──
        # Based on distance to hoop and data sufficiency
        if min_distance < tolerance:
            dist_conf = 1.0 - (min_distance / tolerance) * 0.5
        else:
            dist_conf = max(0.1, 1.0 - (min_distance / (tolerance * 3)))

        data_conf = min(1.0, len(observed_positions) /
                        (config.MIN_POINTS_FOR_PREDICTION * 2))

        self.confidence = float(np.clip(
            0.55 * dist_conf + 0.45 * data_conf, 0.0, 1.0
        ))

    def _result(self):
        """Package prediction results."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "predicted_points": self.predicted_points,
        }

    # Accessors for backwards compatibility
    def get_predicted_points(self):
        return self.predicted_points

    def get_prediction(self):
        return self.prediction

    def get_confidence(self):
        return self.confidence
