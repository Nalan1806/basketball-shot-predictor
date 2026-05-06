"""
Trajectory Prediction Module for the Basketball Shot Prediction System.

Upgraded from polynomial curve fitting to EKF-based forward projection.

Strategy:
  1. Call tracker.predict_trajectory() to get the physics-based arc.
  2. Collect uncertainty ellipses at each future step.
  3. Determine SCORE / MISS by checking whether any future point falls
     inside the hoop region (pixel-space check, same tolerance as before).
  4. Compute a confidence score from:
       - EKF covariance trace at the crossing step (smaller = more confident)
       - Number of observations used
       - Monte Carlo Pr (if available)

The Monte Carlo probability estimator (MonteCarloProbability) is created
once and reused for efficiency.  It runs asynchronously — if it hasn't
finished it returns the last cached Pr so it never blocks the main loop.
"""

import numpy as np
import config
import utils
from ekf_tracker import FORWARD_STEPS
from shot_probability import MonteCarloProbability


class TrajectoryPredictor:
    """
    EKF-based trajectory predictor with uncertainty ellipses and Monte Carlo
    shot probability.

    drop-in replacement for the polynomial TrajectoryPredictor.
    predict() returns the same dict shape as before, plus extra EKF fields.
    """

    def __init__(self):
        self._mc = MonteCarloProbability()

        self.predicted_points   = []    # (px, py) list for drawing
        self.uncertainty_ellipses = []  # list of ellipse dicts (one per future step)
        self.prediction         = None  # "SCORE" or "MISS" or None
        self.confidence         = 0.0
        self.Pr                 = 0.0   # Monte Carlo probability

        # MC is expensive — run it only every N frames to avoid lag.
        self._mc_interval       = 10    # frames between MC runs
        self._mc_frame_counter  = 0
        self._last_mc_result    = None

    # ------------------------------------------------------------------
    def predict(self, positions, hoop_rect, tracker=None):
        """
        Predict the shot trajectory and scoring outcome.

        Args:
            positions: list of (px, py) tracked positions (unused for EKF
                       path; kept for API compatibility with original code).
            hoop_rect: (x, y, w, h) hoop region in pixels, or None.
            tracker:   BallTracker instance (provides EKF state & projection).
                       When None, falls back to a no-op result.

        Returns:
            dict:
              prediction       — "SCORE" | "MISS" | None
              confidence       — float 0–1
              predicted_points — list[(px, py)] future arc
              ellipses         — list[dict] uncertainty ellipses
              Pr               — Monte Carlo scoring probability (float 0–1)
        """
        self.predicted_points     = []
        self.uncertainty_ellipses = []
        self.prediction           = None
        self.confidence           = 0.0

        # ── Need the tracker for EKF projection ──
        if tracker is None or not hasattr(tracker, "predict_trajectory"):
            return self._result()

        if len(positions) < config.MIN_POINTS_FOR_PREDICTION:
            return self._result()

        # ── Step 1: EKF forward projection ──
        future_px = tracker.predict_trajectory(n_steps=FORWARD_STEPS)
        if len(future_px) < 2:
            return self._result()

        self.predicted_points = future_px

        # ── Step 2: Collect uncertainty ellipses ──
        self.uncertainty_ellipses = []
        # Draw ellipses every few steps to avoid clutter (every 5th step)
        for step in range(0, len(future_px), 5):
            ell = tracker.get_uncertainty_ellipse(step)
            if ell is not None:
                self.uncertainty_ellipses.append(ell)

        # ── Step 3: Score / Miss check ──
        if hoop_rect is not None:
            self._check_intersection(hoop_rect, positions)

        # ── Step 4: Monte Carlo Pr (throttled) ──
        self._mc_frame_counter += 1
        if self._mc_frame_counter >= self._mc_interval:
            self._mc_frame_counter = 0
            self._last_mc_result = self._mc.estimate(tracker._ekf)

        if self._last_mc_result and self._last_mc_result["valid"]:
            self.Pr = self._last_mc_result["Pr"]
        else:
            self.Pr = 0.0

        return self._result()

    # ------------------------------------------------------------------
    def _check_intersection(self, hoop_rect, observed_positions):
        """
        Check if any predicted point falls inside (or near) the hoop.

        Same pixel-space logic as the original polyfit version for
        backwards compatibility with the calibrated hoop rectangle.
        """
        hx, hy, hw, hh = hoop_rect
        hoop_center = (hx + hw // 2, hy + hh // 2)
        tolerance = config.INTERSECTION_TOLERANCE

        min_distance = float("inf")

        for pt in self.predicted_points:
            d = utils.distance(pt, hoop_center)
            if d < min_distance:
                min_distance = d

        # Also check Y-crossing logic
        all_points = list(observed_positions) + self.predicted_points
        crosses_hoop_y = False
        for i in range(1, len(all_points)):
            y_prev = all_points[i - 1][1]
            y_curr = all_points[i][1]
            if (y_prev <= hy <= y_curr) or (y_prev >= hy >= y_curr):
                x_at_crossing = all_points[i][0]
                if hx - tolerance <= x_at_crossing <= hx + hw + tolerance:
                    crosses_hoop_y = True
                    break

        if min_distance < tolerance or crosses_hoop_y:
            self.prediction = "SCORE"
        else:
            self.prediction = "MISS"

        self._calculate_confidence(min_distance, tolerance, len(observed_positions))

    def _calculate_confidence(self, min_distance, tolerance, num_points):
        """
        Confidence score from EKF projection quality.

        Combines:
          - How close the predicted arc comes to the hoop
          - Number of observed data points
          - Monte Carlo Pr (if available)
        """
        # Distance factor
        if min_distance < tolerance:
            dist_conf = 1.0 - (min_distance / tolerance) * 0.5
        else:
            dist_conf = max(0.1, 1.0 - (min_distance / (tolerance * 3)))

        # Data sufficiency factor
        data_conf = min(1.0, num_points / (config.MIN_POINTS_FOR_PREDICTION * 2))

        # MC factor (if available)
        mc_conf = self.Pr if self._last_mc_result and self._last_mc_result["valid"] else 0.5

        self.confidence = (
            0.40 * dist_conf +
            0.25 * data_conf +
            0.35 * mc_conf
        )
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _result(self):
        return {
            "prediction":        self.prediction,
            "confidence":        self.confidence,
            "predicted_points":  self.predicted_points,
            "ellipses":          self.uncertainty_ellipses,
            "Pr":                self.Pr,
        }

    # ------------------------------------------------------------------
    # Accessors (kept for backwards compatibility)
    # ------------------------------------------------------------------
    def get_predicted_points(self):
        return self.predicted_points

    def get_prediction(self):
        return self.prediction

    def get_confidence(self):
        return self.confidence

    def get_Pr(self):
        return self.Pr
