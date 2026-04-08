"""
Ball Tracking Module for the Basketball Shot Prediction System.

Maintains a history of ball positions across frames, handles
detection gaps, and provides smoothed position data for
trajectory estimation.
"""

from collections import deque
import config
import utils


class BallTracker:
    """
    Tracks a single basketball across consecutive frames.

    Responsibilities:
      - Store position history in a fixed-size buffer
      - Validate new detections (reject impossible jumps)
      - Handle frames where the ball is not detected
      - Provide smoothed trajectory data
      - Track ball direction and velocity
    """

    def __init__(self):
        # Position history: deque of (x, y) tuples
        self.positions = deque(maxlen=config.MAX_TRACK_POINTS)

        # Raw (unsmoothed) positions for trajectory fitting
        self.raw_positions = deque(maxlen=config.MAX_TRACK_POINTS)

        # Consecutive frames without a detection
        self.frames_missing = 0

        # Current state
        self.is_tracking = False
        self.last_position = None
        self.velocity = (0, 0)  # (dx, dy) per frame
        self.direction = None   # "up", "down", "left", "right"

    def update(self, detection):
        """
        Update tracker with a new detection (or None if ball not found).

        Args:
            detection: dict from BallDetector.detect(), or None.

        Returns:
            True if the ball is being actively tracked, False otherwise.
        """
        if detection is not None:
            new_pos = (detection["cx"], detection["cy"])

            # Validate: reject if the ball "jumped" too far in one frame
            if self.last_position is not None:
                dist = utils.distance(self.last_position, new_pos)
                if dist > config.MAX_DISTANCE_JUMP:
                    # Suspicious jump — could be noise. Skip this frame.
                    self.frames_missing += 1
                    return self._check_tracking_status()

            # Valid detection — update state
            self.raw_positions.append(new_pos)
            self.positions.append(new_pos)

            # Calculate velocity
            if self.last_position is not None:
                dx = new_pos[0] - self.last_position[0]
                dy = new_pos[1] - self.last_position[1]
                self.velocity = (dx, dy)
                self._update_direction(dx, dy)

            self.last_position = new_pos
            self.frames_missing = 0
            self.is_tracking = True

        else:
            # No detection this frame
            self.frames_missing += 1

        return self._check_tracking_status()

    def _check_tracking_status(self):
        """Check if we should still consider ourselves as tracking."""
        if self.frames_missing > config.MAX_FRAMES_MISSING:
            # Ball has been missing too long — reset tracking
            self.is_tracking = False
        return self.is_tracking

    def _update_direction(self, dx, dy):
        """Determine the primary direction of ball movement."""
        if abs(dx) > abs(dy):
            self.direction = "right" if dx > 0 else "left"
        else:
            self.direction = "down" if dy > 0 else "up"

    def get_positions(self):
        """Return the list of tracked positions."""
        return list(self.positions)

    def get_smoothed_positions(self):
        """Return smoothed positions for cleaner trajectory visualization."""
        return utils.smooth_positions(list(self.positions))

    def get_raw_positions(self):
        """Return unsmoothed positions for polynomial fitting."""
        return list(self.raw_positions)

    def get_frames_missing(self):
        """Return how many consecutive frames the ball has been missing."""
        return self.frames_missing

    def get_velocity(self):
        """Return the current velocity (dx, dy) in pixels/frame."""
        return self.velocity

    def get_speed(self):
        """Return the scalar speed in pixels/frame."""
        return utils.distance((0, 0), self.velocity)

    def has_enough_data(self):
        """Check if we have enough data points for trajectory prediction."""
        return len(self.positions) >= config.MIN_POINTS_FOR_PREDICTION

    def reset(self):
        """Clear all tracking data."""
        self.positions.clear()
        self.raw_positions.clear()
        self.frames_missing = 0
        self.is_tracking = False
        self.last_position = None
        self.velocity = (0, 0)
        self.direction = None
