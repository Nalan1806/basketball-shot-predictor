"""
Trajectory Prediction Module for the Basketball Shot Prediction System.

Uses polynomial curve fitting on tracked ball positions to estimate
the future flight path and determine if it intersects the hoop region.
"""

import numpy as np
import config
import utils


class TrajectoryPredictor:
    """
    Predicts the basketball's future trajectory using polynomial regression.

    The ball follows a roughly parabolic path (projectile motion),
    so a degree-2 polynomial y = ax² + bx + c is the natural fit.

    Also computes intersection with the hoop region and a
    confidence score based on fit quality and data sufficiency.
    """

    def __init__(self):
        self.poly_coeffs = None          # Polynomial coefficients
        self.predicted_points = []       # List of (x, y) future points
        self.prediction = None           # "SCORE" or "MISS"
        self.confidence = 0.0            # 0.0 to 1.0
        self.fit_residual = float("inf") # How well the polynomial fits

    def predict(self, positions, hoop_rect):
        """
        Fit a trajectory and predict whether the ball will score.

        Args:
            positions: list of (x, y) tuples — tracked ball positions.
            hoop_rect: (x, y, w, h) of the hoop region, or None.

        Returns:
            dict with keys:
              - prediction: "SCORE" or "MISS" or None
              - confidence: float 0.0 - 1.0
              - predicted_points: list of (x,y) for the future trajectory
        """
        self.predicted_points = []
        self.prediction = None
        self.confidence = 0.0

        if len(positions) < config.MIN_POINTS_FOR_PREDICTION:
            return self._result()

        # Extract x and y arrays
        xs = np.array([p[0] for p in positions], dtype=np.float64)
        ys = np.array([p[1] for p in positions], dtype=np.float64)

        # Determine extrapolation direction from recent movement
        dx = xs[-1] - xs[0]
        direction = 1 if dx >= 0 else -1

        try:
            # Fit a polynomial (degree 2 = parabola by default)
            # Using polyfit with the x-values as the independent variable
            self.poly_coeffs = np.polyfit(xs, ys, config.POLY_DEGREE)
            poly_fn = np.poly1d(self.poly_coeffs)

            # Calculate fit quality (R² score)
            y_pred = poly_fn(xs)
            ss_res = np.sum((ys - y_pred) ** 2)
            ss_tot = np.sum((ys - np.mean(ys)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.fit_residual = max(0, r_squared)

        except (np.linalg.LinAlgError, ValueError):
            # Polyfit can fail with collinear or insufficient points
            return self._result()

        # Extrapolate future positions
        last_x = xs[-1]
        step = direction * config.PREDICTION_STEP_SIZE

        for i in range(1, config.PREDICTION_STEPS + 1):
            future_x = last_x + step * i
            future_y = poly_fn(future_x)

            # Sanity check: don't predict off-screen
            if future_y < -200 or future_y > 2000 or future_x < -200 or future_x > 2500:
                break

            self.predicted_points.append((int(future_x), int(future_y)))

        # Check intersection with hoop region
        if hoop_rect is not None and len(self.predicted_points) > 0:
            self._check_intersection(hoop_rect, positions)

        return self._result()

    def _check_intersection(self, hoop_rect, positions):
        """
        Determine if the predicted trajectory passes through the hoop.

        Uses a combination of:
          1. Distance from predicted points to hoop center
          2. Whether trajectory crosses the hoop's Y-level
          3. Direction of ball movement (must be heading toward hoop)
        """
        hx, hy, hw, hh = hoop_rect
        hoop_center = (hx + hw // 2, hy + hh // 2)

        # Find the closest predicted point to the hoop center
        min_distance = float("inf")
        closest_point = None

        for pt in self.predicted_points:
            d = utils.distance(pt, hoop_center)
            if d < min_distance:
                min_distance = d
                closest_point = pt

        # Also check all trajectory points (past + predicted) for Y-crossing
        all_points = list(positions) + self.predicted_points
        crosses_hoop_y = False
        for i in range(1, len(all_points)):
            y_prev = all_points[i - 1][1]
            y_curr = all_points[i][1]
            # Check if trajectory crosses the hoop's vertical band
            if (y_prev <= hy and y_curr >= hy) or (y_prev >= hy and y_curr <= hy):
                # Check if the X position is within the hoop's horizontal range
                x_at_crossing = all_points[i][0]
                if hx - config.INTERSECTION_TOLERANCE <= x_at_crossing <= hx + hw + config.INTERSECTION_TOLERANCE:
                    crosses_hoop_y = True
                    break

        # Scoring decision
        tolerance = config.INTERSECTION_TOLERANCE

        if min_distance < tolerance or crosses_hoop_y:
            self.prediction = "SCORE"
        else:
            self.prediction = "MISS"

        # Calculate confidence score
        self._calculate_confidence(min_distance, tolerance, len(positions))

    def _calculate_confidence(self, min_distance, tolerance, num_points):
        """
        Compute a confidence score from 0.0 to 1.0 based on:
          - How close the trajectory comes to the hoop
          - Quality of the polynomial fit (R²)
          - Number of data points available
        """
        # Distance factor: closer to hoop → higher confidence
        if min_distance < tolerance:
            dist_confidence = 1.0 - (min_distance / tolerance) * 0.5
        else:
            dist_confidence = max(0.1, 1.0 - (min_distance / (tolerance * 3)))

        # Fit quality factor
        fit_confidence = max(0.0, min(1.0, self.fit_residual))

        # Data sufficiency factor: more points → more confident
        data_confidence = min(1.0, num_points / (config.MIN_POINTS_FOR_PREDICTION * 2))

        # Weighted combination
        self.confidence = (
            0.4 * dist_confidence +
            0.35 * fit_confidence +
            0.25 * data_confidence
        )
        self.confidence = max(0.0, min(1.0, self.confidence))

    def _result(self):
        """Package the prediction results."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "predicted_points": self.predicted_points,
        }

    def get_predicted_points(self):
        """Return the list of predicted future positions."""
        return self.predicted_points

    def get_prediction(self):
        """Return the current prediction ('SCORE', 'MISS', or None)."""
        return self.prediction

    def get_confidence(self):
        """Return the current confidence score."""
        return self.confidence
