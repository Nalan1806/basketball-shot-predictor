"""
Lightweight Kalman Tracker — ShotIQ

Simple 4-state Kalman filter operating entirely in pixel space:
  State: [x, y, vx, vy]

No drag estimation, no beta, no world-space transforms.
Uses real dynamic dt (passed from main loop) for timing accuracy.

Features:
  - OpenCV KalmanFilter with 4 states, 2 measurements
  - Dynamic dt updates the transition matrix each frame
  - Jump rejection: ignores detections too far from last position
  - Coast mode: predicts for up to MAX_FRAMES_MISSING frames
  - Auto-reset after too many missed frames
  - Smooth velocity estimation from filtered state
"""

import cv2
import numpy as np
from collections import deque
import config


class BallTracker:
    """
    Lightweight pixel-space Kalman tracker for basketball tracking.

    State vector: [x, y, vx, vy]
    Measurement: [x, y]

    Public interface:
        update(detection, confidence, dt)
        get_positions() / get_raw_positions()
        get_velocity() / get_speed()
        has_enough_data()
        is_using_prediction()
        get_frames_missing()
        reset()
    """

    def __init__(self):
        """Initialize the Kalman filter and tracking state."""
        self._init_kalman()

        # Position history
        self.positions = deque(maxlen=config.MAX_TRACK_POINTS)
        self.raw_positions = deque(maxlen=config.MAX_TRACK_POINTS)

        # Tracking state
        self.is_tracking = False
        self.frames_missing = 0
        self.last_position = None       # Kalman-filtered position (px, py)
        self.velocity = (0.0, 0.0)      # estimated velocity (px/s)
        self.current_confidence = 0.0
        self._initialized = False

    def _init_kalman(self):
        """Create and configure the OpenCV Kalman filter."""
        # 4 state variables (x, y, vx, vy), 2 measurements (x, y)
        self._kf = cv2.KalmanFilter(4, 2, 0)

        # Transition matrix (updated each frame with real dt)
        # [1, 0, dt, 0 ]
        # [0, 1, 0,  dt]
        # [0, 0, 1,  0 ]
        # [0, 0, 0,  1 ]
        self._kf.transitionMatrix = np.eye(4, dtype=np.float32)

        # Measurement matrix: we observe x and y
        self._kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise covariance
        pn = config.KALMAN_PROCESS_NOISE
        self._kf.processNoiseCov = np.array([
            [pn,  0,   0,   0  ],
            [0,   pn,  0,   0  ],
            [0,   0,   pn*2, 0  ],
            [0,   0,   0,   pn*2],
        ], dtype=np.float32)

        # Measurement noise covariance
        mn = config.KALMAN_MEASUREMENT_NOISE
        self._kf.measurementNoiseCov = np.array([
            [mn, 0],
            [0,  mn],
        ], dtype=np.float32)

        # Error covariance (initial)
        self._kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0

    def update(self, detection, confidence=0.0, dt=1.0/30.0):
        """
        Update tracker with latest ball detection.

        Args:
            detection: dict with "cx", "cy" keys, or None
            confidence: float 0–1 detection confidence
            dt: real delta time in seconds since last frame

        Returns:
            bool: True if actively tracking
        """
        self.current_confidence = confidence

        # Update transition matrix with real dt
        self._kf.transitionMatrix[0, 2] = float(dt)
        self._kf.transitionMatrix[1, 3] = float(dt)

        if detection is not None:
            px, py = float(detection["cx"]), float(detection["cy"])

            # Jump rejection
            if self.last_position is not None:
                dist = np.hypot(px - self.last_position[0],
                                py - self.last_position[1])
                if dist > config.MAX_JUMP_PX * 3 and confidence < 0.6:
                    # Extreme jump with low confidence — skip
                    self.frames_missing += 1
                    if self._initialized:
                        self._kf.predict()
                    return self._check_tracking()

            if not self._initialized:
                # First detection — initialize state
                self._kf.statePost = np.array(
                    [[px], [py], [0.0], [0.0]], dtype=np.float32
                )
                self._initialized = True
                self.last_position = (px, py)
                self.positions.append((px, py))
                self.raw_positions.append((px, py))
                self.frames_missing = 0
                self.is_tracking = True
                return True

            # Predict + correct (standard Kalman cycle)
            self._kf.predict()
            measurement = np.array([[px], [py]], dtype=np.float32)
            self._kf.correct(measurement)

            # Extract filtered state
            state = self._kf.statePost.flatten()
            fx, fy = float(state[0]), float(state[1])
            vx, vy = float(state[2]), float(state[3])

            self.last_position = (fx, fy)
            self.velocity = (vx, vy)
            self.positions.append((fx, fy))
            self.raw_positions.append((px, py))
            self.frames_missing = 0
            self.is_tracking = True

        else:
            # No detection — coast (predict only)
            self.frames_missing += 1

            if self.frames_missing > config.MAX_FRAMES_MISSING:
                self.reset()
                return False

            if self._initialized:
                predicted = self._kf.predict().flatten()
                fx, fy = float(predicted[0]), float(predicted[1])
                vx, vy = float(predicted[2]), float(predicted[3])

                # Bounds check — reset if predicted position leaves frame
                if (fx < -100 or fx > config.FRAME_WIDTH + 100 or
                        fy < -100 or fy > config.FRAME_HEIGHT + 100):
                    self.reset()
                    return False

                self.last_position = (fx, fy)
                self.velocity = (vx, vy)
                self.positions.append((fx, fy))

        return self._check_tracking()

    def _check_tracking(self):
        """Update is_tracking flag."""
        if self.frames_missing > config.MAX_FRAMES_MISSING:
            self.is_tracking = False
        return self.is_tracking

    # ──────────────────────────────────────────────────────────────────
    # Public accessors
    # ──────────────────────────────────────────────────────────────────

    def get_positions(self):
        """Return Kalman-filtered positions in pixel space."""
        return list(self.positions)

    def get_raw_positions(self):
        """Return unfiltered raw detection positions."""
        return list(self.raw_positions)

    def get_velocity(self):
        """Return estimated velocity (vx, vy) in pixels/second."""
        return self.velocity

    def get_speed(self):
        """Return scalar speed in pixels/second."""
        return float(np.hypot(*self.velocity))

    def get_speed_px_per_frame(self, dt=1.0/30.0):
        """Return speed in pixels/frame (for heuristic compatibility)."""
        speed_per_sec = self.get_speed()
        return speed_per_sec * dt

    def has_enough_data(self):
        """Return True if enough positions for trajectory prediction."""
        return len(self.positions) >= config.MIN_POINTS_FOR_PREDICTION

    def get_frames_missing(self):
        """Return consecutive frames without detection."""
        return self.frames_missing

    def is_using_prediction(self):
        """Return True if in predict-only (coast) mode."""
        return self.frames_missing > 0 and self.is_tracking

    def get_confidence(self):
        """Return current detection confidence."""
        return self.current_confidence

    def get_state(self):
        """
        Return current tracker state for telemetry display.

        Returns:
            dict with position, velocity, speed info
        """
        if not self._initialized:
            return {
                "x": 0.0, "y": 0.0,
                "vx": 0.0, "vy": 0.0,
                "speed": 0.0,
                "tracking": False,
                "frames_missing": self.frames_missing,
            }

        state = self._kf.statePost.flatten()
        return {
            "x": float(state[0]),
            "y": float(state[1]),
            "vx": float(state[2]),
            "vy": float(state[3]),
            "speed": float(np.hypot(state[2], state[3])),
            "tracking": self.is_tracking,
            "frames_missing": self.frames_missing,
        }

    def reset(self):
        """Full reset of all tracking state."""
        self._init_kalman()
        self.positions.clear()
        self.raw_positions.clear()
        self.is_tracking = False
        self.frames_missing = 0
        self.last_position = None
        self.velocity = (0.0, 0.0)
        self.current_confidence = 0.0
        self._initialized = False
