"""
Ball Tracking Module for the Basketball Shot Prediction System.

This module now wraps the Extended Kalman Filter tracker (EKFBallTracker)
while preserving the exact public interface used by main.py and trajectory.py.

All EKF physics / covariance / uncertainty logic lives in ekf_tracker.py.
This module is the single import point for main.py.

Public interface (unchanged from previous BallTracker):
    update(detection, confidence)
    get_positions()
    get_smoothed_positions()
    get_raw_positions()
    get_frames_missing()
    get_velocity()
    get_speed()
    has_enough_data()
    get_predicted_position()
    get_last_detection_position()
    get_confidence()
    is_using_prediction()
    reset()

New EKF-specific accessors (used by trajectory.py and main.py):
    predict_trajectory(n_steps)        — EKF forward projection (pixels)
    get_uncertainty_ellipse(step)      — 66% confidence ellipse at step
    get_ekf_state()                    — dict for telemetry panel
"""

from ekf_tracker import EKFBallTracker
from pixel_to_world import PixelToWorld
import config


class BallTracker:
    """
    Drop-in replacement for the original Kalman-based BallTracker.

    Delegates all work to EKFBallTracker, which implements a 5-state
    Extended Kalman Filter with aerodynamic drag physics.
    """

    def __init__(self):
        ptw = PixelToWorld()
        self._ekf = EKFBallTracker(pixel_to_world=ptw)

    # ------------------------------------------------------------------
    # Core update — called once per frame by main.py
    # ------------------------------------------------------------------
    def update(self, detection, confidence=0.0):
        """
        Update the EKF tracker with the latest ball detection.

        Args:
            detection: dict from BallDetector, or None.
            confidence: float 0–1 detection confidence.

        Returns:
            True if ball is actively tracked, False otherwise.
        """
        return self._ekf.update(detection, confidence=confidence)

    # ------------------------------------------------------------------
    # Position accessors
    # ------------------------------------------------------------------
    def get_positions(self):
        """Tracked positions in pixel space."""
        return self._ekf.get_positions()

    def get_smoothed_positions(self):
        """Moving-average smoothed positions in pixel space."""
        return self._ekf.get_smoothed_positions()

    def get_raw_positions(self):
        """Unsmoothed raw detection positions in pixel space."""
        return self._ekf.get_raw_positions()

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------
    def get_frames_missing(self):
        """Number of consecutive frames with no detection."""
        return self._ekf.get_frames_missing()

    def get_velocity(self):
        """Estimated velocity (dx, dy) in pixels/frame."""
        return self._ekf.get_velocity()

    def get_speed(self):
        """Scalar speed in pixels/frame."""
        return self._ekf.get_speed()

    def has_enough_data(self):
        """True if enough data for trajectory prediction."""
        return self._ekf.has_enough_data()

    def get_predicted_position(self):
        """EKF predicted pixel position (for debug overlay)."""
        return self._ekf.get_predicted_position()

    def get_last_detection_position(self):
        """Last raw detection pixel position."""
        return self._ekf.get_last_detection_position()

    def get_confidence(self):
        """Current detection confidence."""
        return self._ekf.get_confidence()

    def is_using_prediction(self):
        """True if in predict-only mode (no recent detection)."""
        return self._ekf.is_using_prediction()

    # ------------------------------------------------------------------
    # EKF-specific accessors (new, used by trajectory.py / main.py)
    # ------------------------------------------------------------------
    def predict_trajectory(self, n_steps=None):
        """
        Run EKF forward projection for n_steps steps.

        Returns:
            list of (px, py) pixel tuples.
        """
        return self._ekf.predict_trajectory(n_steps=n_steps)

    def get_uncertainty_ellipse(self, step):
        """
        66% confidence ellipse parameters at a future step.

        Returns:
            dict {center, axes, angle} or None.
        """
        return self._ekf.get_uncertainty_ellipse(step)

    def get_ekf_state(self):
        """
        Current EKF state for the telemetry overlay.

        Returns:
            dict {px_m, py_m, vx_ms, vy_ms, beta, speed_ms, P_pos}
        """
        return self._ekf.get_ekf_state()

    def get_lost_frames(self):
        """
        Current lost_frames counter.

        Returns:
            int: 0 = real detection, 1-10 = coasting, >10 = reset
        """
        return self._ekf.get_lost_frames()

    def covariance_healthy(self):
        """
        Check if EKF covariance is well-conditioned and actively tracking.

        Returns:
            bool: True if trace(P) <= 200 AND lost_frames == 0
        """
        return self._ekf.covariance_healthy()

    def reset(self):
        """Reset all tracking state."""
        self._ekf.reset()
